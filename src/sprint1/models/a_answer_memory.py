"""A1 - Answer memory (AM).  Idea card: sprint1/ideas/A_answer_memory.md

A slot table that is WRITTEN, not trained: during every training forward pass each token writes the code
of the token that actually came next (the "answer"; a fixed random code per vocabulary item) into the
slots its context addressed, by a normalised LMS / delta rule.  A slot therefore holds a sketch of the
next-token distribution of the contexts routed to it.  The table is a buffer (no SGD, no optimizer
state); slots are chosen per head by a hash of the current token (bucket) and a learned query from the
body's final hidden state (slot inside the bucket); the read is decoded with the same codes and added to
the logits through a zero-init gain (a scalar plus a linear function of the hidden state, so the model can
switch the memory off where its slots are noise).  Reads in a step use the table as it was before that step's writes, so a
sequence never reads its own answers.  Nothing is written unless the module is in train() mode AND autograd
is enabled, so at eval the table is a frozen, training-data-only memory, like any other weight.

Why: SGD stores a fact seen once only weakly; the window holds no information about context-new tokens.
A one-shot written memory of (context -> answer) over the whole training stream is the cheapest way to
put long-tail corpus knowledge in front of the LM head (kNN-LM-like, but compressive, fixed-size, O(1)
reads, trained end-to-end on the read side).

Knobs (cfg.extra):
  null: bool=False         control: the written answers come from a fixed random token block drawn once at
                           build time (same params, FLOPs and write traffic; no context->answer information;
                           never reads another batch row)
  am_addr: "tok"|"pk"|"ngram"
                           "tok" (default): per head, a hash of the current token x_t picks a bucket of
                           am_inner slots and a learned query picks the top-k slots inside it, so contexts
                           compete for slots only with contexts ending in the same token (protects the
                           long tail from frequent tokens; a learned back-off below the unigram).
                           "pk": product-key search over all slots (PKM); diluted by frequent contexts in
                           the CPU unit test, kept as an option.
                           "ngram" = known-lever control: slot = hash(x_{t-1}, x_t) per head (count-table
                           analogue with delta writes); the learned query/keys are then unused
  am_heads: int=4          memory heads (independent hashes / keys)
  am_nsub: int=256         table size am_nsub**2 slots (65,536 at default)
  am_inner: int=16         "tok": slots per bucket (buckets = slots / am_inner)
  am_dkey: int=64          query width per head
  am_topk: int=2 (tok) / 8 (pk)   slots read/written per head per token
  am_eta: float=0.5        write step (fraction of the slot's weighted mean error)
  am_c: float=None         write damping constant; default 1/(heads*topk)
  am_write: bool=True      disable writes (ablation: table stays zero -> read path is dead)
  am_code: "sketch"|"out"|"in"
                           answer code written into the table.  "sketch" (default): a fixed random unit
                           code per vocabulary item (am_dcode wide); the read is decoded to logits with the
                           same codes, logits += gain * (am_dcode/4) * read @ codes^T, so a slot is a sketch
                           of its next-token distribution and nothing has to be aligned by learning (costs one
                           V x am_dcode matmul per token).  "out": the answer's LM-head row, read added to the
                           hidden state (identity-init W_o); "in": RMS-normed input embedding, same path.
                           (CPU unit test: "out"/"in" barely used the memory because the head/alignment must
                           be learned first.)
  am_dcode: int=64         sketch code width (32 at tiny)
  am_learn: "write"|"sgd"  "sgd" = known memory-layer control: the same table is an nn.Parameter trained by
                           the optimizer (adam group) and never written (+N*d trained params; PKM / memory
                           layers).  Tests whether one-shot writes beat gradient learning of the same slots.
  body: "auto"|"baseline"|"mini";  a_equal_params: bool=True (shrink body MLP by the added params)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.a_common import answer_code, build_body, lm_loss_aux, make_null_tokens, null_like, rms, split_groups


def _defaults(cfg: ModelConfig) -> dict:
    tiny = cfg.n_embd < 128
    e = cfg.extra
    addr = str(e.get("am_addr", "tok"))
    heads = int(e.get("am_heads", 2 if tiny else 4))
    topk = int(e.get("am_topk", (2 if addr == "tok" else 4) if tiny else (2 if addr == "tok" else 8)))
    return dict(
        null=bool(e.get("null", False)),
        addr=addr,
        heads=heads,
        nsub=int(e.get("am_nsub", 32 if tiny else 256)),
        inner=int(e.get("am_inner", 8 if tiny else 16)),
        dkey=int(e.get("am_dkey", 16 if tiny else 64)),
        topk=topk,
        eta=float(e.get("am_eta", 0.5)),
        c=float(e.get("am_c") or 1.0 / (heads * topk)),
        write=bool(e.get("am_write", True)),
        learn=str(e.get("am_learn", "write")),
        code=str(e.get("am_code", "sketch")),
        dcode=int(e.get("am_dcode", 32 if tiny else 64)),
    )


class AnswerMemoryLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        o = _defaults(cfg)
        self.o = o
        d = cfg.n_embd
        H, dk, n = o["heads"], o["dkey"], o["nsub"]
        self.H, self.dh, self.n, self.k = H, dk // 2, n, o["topk"]
        self.d = d
        self.N = n * n
        if o["addr"] == "tok":
            # token-bucketed: the current token's hash picks a bucket of `inner` slots (per head), a learned
            # query picks top-k slots inside it -> contexts compete only with contexts of the same token
            self.inner = o["inner"]
            self.n_bucket = self.N // self.inner
            key_shape = (H, 1, self.inner, dk)
        else:
            key_shape = (H, 2, n, dk // 2)
        self.sketch = o["code"] == "sketch"
        dv = o["dcode"] if self.sketch else d          # width of a table row
        self.dv = dv
        extra = d * H * dk + math.prod(key_shape) + (0 if self.sketch else d * d) + 1 + d
        self.body = build_body(cfg, extra_params=extra)
        self.wq = nn.Linear(d, H * dk, bias=False)
        # keys stored FLAT (1-D) so the harness's Adam split uses the scalar LR (0.052), not the embedding LR (0.78)
        self.key_shape = key_shape
        self.subkeys_flat = nn.Parameter((torch.randn(*key_shape) / math.sqrt(key_shape[-1])).flatten())
        self.gain = nn.Parameter(torch.zeros(()))       # the read enters through a zero-init gain ...
        self.gate = nn.Linear(d, 1, bias=False)         # ... plus a zero-init context-dependent part
        nn.init.zeros_(self.gate.weight)
        if self.sketch:
            # fixed random unit codes, one per vocabulary item: a row of the table is an LMS running mean of
            # the codes of the answers written to it, i.e. a sketch of that slot's next-token distribution;
            # it is decoded to logits by the same codes (no alignment to learn): logit_mem(y) ~ p_slot(y)
            g = torch.Generator().manual_seed(int(cfg.extra.get("am_code_seed", 1234)))
            codes = torch.randn(cfg.vocab_size, dv, generator=g)
            self.register_buffer("codes", codes / codes.norm(dim=-1, keepdim=True))
            self.kappa = float(dv) / 4.0
        else:
            self.wo = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.wo.weight)                # read is already in LM-head space
        if o["learn"] == "sgd":
            self.table = nn.Parameter(torch.randn(self.N, dv) * dv ** -0.5)
            o["write"] = False
        else:
            self.register_buffer("table", torch.zeros(self.N, dv))
        # writes of a step are accumulated here and applied at the start of the next forward, after that
        # step's backward has used `table` (embedding_bag saves it for the per-sample-weight gradient)
        self.register_buffer("pend_num", torch.zeros(self.N, dv))
        self.register_buffer("pend_den", torch.zeros(self.N))
        self.register_buffer("n_writes", torch.zeros((), dtype=torch.float64))
        if o["null"]:
            self.register_buffer("null_ids", make_null_tokens(cfg.vocab_size), persistent=False)
        self.cfg = cfg

    @property
    def subkeys(self):
        return self.subkeys_flat.view(self.key_shape)

    # ---- addressing -------------------------------------------------------------------------
    def _address_pk(self, h):
        B, T, _ = h.shape
        H, dh, k, n = self.H, self.dh, self.k, self.n
        q = rms(self.wq(h).view(B, T, H, 2, dh))
        s = torch.einsum("bthcd,hcnd->bthcn", q, self.subkeys.to(q.dtype)).float()
        s1, i1 = s[..., 0, :].topk(k, dim=-1)
        s2, i2 = s[..., 1, :].topk(k, dim=-1)
        comb = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)          # [B,T,H,k*k]
        sc, pos = comb.topk(k, dim=-1)                                      # [B,T,H,k]
        slot = torch.gather(i1, -1, pos // k) * n + torch.gather(i2, -1, pos % k)
        w = torch.softmax(sc, dim=-1) / H
        return slot.reshape(B * T, H * k), w.reshape(B * T, H * k)

    def _address_tok(self, h, idx):
        B, T, _ = h.shape
        H, k, m = self.H, self.k, self.inner
        q = rms(self.wq(h).view(B, T, H, -1))
        s = torch.einsum("bthd,hnd->bthn", q, self.subkeys[:, 0].to(q.dtype)).float()   # [B,T,H,inner]
        sc, j = s.topk(k, dim=-1)
        salts = torch.arange(H, device=idx.device, dtype=torch.long) * 0x9E3779B1 + 777
        bucket = ((idx.long()[..., None] * 2654435761 + salts) % 2147483647) % self.n_bucket   # [B,T,H]
        slot = bucket[..., None] * m + j
        w = torch.softmax(sc, dim=-1) / H
        return slot.reshape(B * T, H * k), w.reshape(B * T, H * k)

    def _address_ngram(self, idx):
        B, T = idx.shape
        prev = F.pad(idx, (1, 0))[:, :-1].long()
        cur = idx.long()
        salts = torch.arange(self.H, device=idx.device, dtype=torch.long) * 0x9E3779B1 + 12345
        hsh = (prev[..., None] * 1000003) ^ (cur[..., None] * 998244353 + salts)
        slot = (hsh % self.N).reshape(B * T, self.H)
        w = torch.full((B * T, self.H), 1.0 / self.H, device=idx.device)
        return slot, w

    # ---- write (training only, no autograd) ---------------------------------------------------
    @torch.no_grad()
    def _write(self, slot, w, read, targets):
        BT = slot.size(0)
        if self.o["null"]:      # fixed random answers, same valid mask; never another row of the batch
            y = torch.where(targets >= 0, null_like(self.null_ids, *targets.shape), targets)
        else:
            y = targets
        valid = (y >= 0).reshape(BT, 1).float()
        if self.sketch:
            ans = F.embedding(y.clamp_min(0), self.codes).reshape(BT, self.dv)
        else:
            ans = answer_code(self.body, y.clamp_min(0), self.o["code"]).reshape(BT, self.d)
        err = (ans - read.float()) * valid                                  # LMS error of the full read
        ww = w.float() * valid                                              # [BT, Hk]
        self.pend_den.index_add_(0, slot.reshape(-1), ww.reshape(-1))
        for j in range(slot.size(1)):                                       # static loop, Hk <= 32
            self.pend_num.index_add_(0, slot[:, j], err * ww[:, j:j + 1])
        self.n_writes.add_(valid.sum().double())

    @torch.no_grad()
    def _apply_pending(self):
        if not self.o["write"]:
            return
        self.table.add_(self.o["eta"] * self.pend_num / (self.pend_den + self.o["c"]).unsqueeze(-1))
        self.pend_num.zero_()
        self.pend_den.zero_()

    def _logits(self, idx):
        h = self.body.hidden(idx)
        B, T, d = h.shape
        if self.o["addr"] == "ngram":
            slot, w = self._address_ngram(idx)
        elif self.o["addr"] == "tok":
            slot, w = self._address_tok(h, idx)
        else:
            slot, w = self._address_pk(h)
        read = F.embedding_bag(slot, self.table, per_sample_weights=w.float(), mode="sum")  # [BT, dv]
        if self.sketch:
            z_mem = (read @ self.codes.t()).view(B, T, -1)                            # [B,T,V]
            g = (self.gain + self.gate(h).float()) * self.kappa                      # [B,T,1]
            return self.body.logits(h) + g * z_mem, slot, w, read
        g = self.gain + self.gate(h)
        h2 = h + g * self.wo(read.view(B, T, d).to(h.dtype))
        return self.body.logits(h2), slot, w, read

    def forward_logits(self, idx):
        """Eval-semantics logits [B, T, V] (for per-token / novelty-split scoring). Never writes."""
        self._apply_pending()
        return self._logits(idx)[0]

    def forward(self, idx, targets):
        self._apply_pending()            # no-op when nothing is pending, e.g. during eval
        logits, slot, w, read = self._logits(idx)
        loss, aux = lm_loss_aux(logits, targets, self.training)
        # write only in a real training step: train() mode AND autograd on (a val loop that forgets eval()
        # but runs under no_grad still cannot write val answers into the table)
        if self.training and self.o["write"] and torch.is_grad_enabled():
            self._write(slot.detach(), w.detach(), read.detach(), targets)
        aux["am_read_rms"] = read.detach().pow(2).mean().sqrt()
        return loss, aux

    # ---- contract hooks ---------------------------------------------------------------------------
    def param_groups(self):
        extra = [self.subkeys_flat] + ([self.table] if isinstance(self.table, nn.Parameter) else [])
        return split_groups(self, self.body, adam_extra=extra)          # wq, gate (2-D) -> muon

    def flops_per_token(self):
        o = self.o
        d = self.d
        dv = self.dv
        mech = 6.0 * (self.wq.weight.numel() + self.gate.weight.numel())          # query projection, gate
        if self.sketch:
            mech += 2.0 * 2 * self.codes.numel()           # decode fwd + grad-to-read bwd (codes fixed)
        else:
            mech += 6.0 * self.wo.weight.numel()
        mech += 6.0 * self.subkeys.numel()                                      # key scores
        mech += 2.0 * 3 * o["heads"] * o["topk"] * dv                           # weighted read fwd+bwd(w)
        mech += 2.0 * 2 * o["heads"] * o["topk"] * dv                           # write (num/den scatter)
        if o["addr"] == "ngram":
            mech -= 6.0 * (self.wq.weight.numel() + self.subkeys.numel())       # no learned query
        return float(self.body.body_flops_per_token()) + mech

    def memory_report(self) -> dict:
        used = (self.table.abs().sum(-1) > 0).sum().item()
        return {"slots": self.N, "slots_written": used, "table_numbers": self.table.numel(),
                "gain": float(self.gain.detach()),
                "writes": float(self.n_writes.item())}


def build(cfg: ModelConfig) -> nn.Module:
    return AnswerMemoryLM(cfg)
