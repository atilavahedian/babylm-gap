"""H1 MaskSlot: one network that is an exact left-to-right LM AND its own parallel drafter.

Idea card: sprint1/ideas/H_maskslot.md. Contract: sprint1/harness/model_api.py.

Training. The main stream is a standard causal decoder; `loss` is its plain next-token CE (the scored quantity).
A second, sparse query stream of "lookahead slots" shares every weight. For A random anchors a per sequence,
K-1 slots sit at positions a+1..a+K-1 with input = (main residual at anchor a, taken at layer `slot_from`)
+ a learned per-offset mask vector. Slot j attends to the main stream's keys at positions <= a and to earlier
slots of its own anchor, and predicts x[a+j+1] (the token j steps beyond the main stream's next token) without
seeing x[a+1..a+j]. Its CE (times `slot_weight`) is returned as aux["aux_loss"]; the main stream never attends
to slots, so the scored loss is an ordinary causal-LM likelihood.

Decoding (lossless self-speculation, `spec_generate`). One forward pass processes a verify block
[b, d_1..d_{K-1}] (last committed token plus K-1 drafts) together with K-1 slots attached to EACH verify
position (a K x (K-1) slot tree, K^2 query positions). Greedy acceptance keeps the longest draft prefix that
matches the main stream's argmax, plus one bonus token; the slots attached to the last accepted position supply
the next K-1 drafts. Output is identical to plain greedy decoding of the main stream (tested); the gain is fewer
sequential full-model passes per token (latency at small batch), paid for with more FLOPs per pass.

extra knobs (cfg.extra):
    slot_k        K, verify width; K-1 slots per anchor (default 4; 1 disables slots)
    slot_anchors  A, anchors per training sequence (default seq_len // 16)
    slot_weight   lambda on the slot CE (default 0.5; 0 = no slot compute: plain causal LM, the backbone arm)
    slot_from     layer at which slots branch off the main stream (default 0 = full depth;
                  n_layer-1 = a Medusa-like cheap comparator that only uses the last block)
    slot_null     "none" (default) or "shuffle": slot targets permuted within the batch (scrambled-target null,
                  same FLOPs; drafts must then be useless, acceptance ~ chance)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sprint1.harness.model_api import ModelConfig
from sprint1.models.h_common import (Block, KVCache, Rotary, block_matmul_params, ce, rms, softcap,
                                     split_param_groups)


class MaskSlotLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ex = cfg.extra or {}
        self.cfg = cfg
        d, L = cfg.n_embd, cfg.n_layer
        self.d, self.L, self.H, self.hd = d, L, cfg.n_head, d // cfg.n_head
        self.V = cfg.vocab_size
        self.K = int(ex.get("slot_k", 4))
        self.J = max(self.K - 1, 0)
        self.n_anchor = int(ex.get("slot_anchors", max(1, cfg.seq_len // 16)))
        self.slot_weight = float(ex.get("slot_weight", 0.5))
        self.slot_from = int(ex.get("slot_from", 0))
        self.slot_null = str(ex.get("slot_null", "none"))
        assert 0 <= self.slot_from < L, "slot_from must be a layer index"
        assert self.slot_null in ("none", "shuffle")
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.blocks = nn.ModuleList([Block(d, cfg.n_head, cfg.mlp_ratio) for _ in range(L)])
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        # per-offset mask vectors, same scale as token embeddings (N(0,1), the nn.Embedding default)
        self.mask_emb = nn.Parameter(torch.randn(max(self.J, 1), d))
        self.rotary = Rotary(self.hd)

    # ------------------------------------------------------------------ helpers
    def head(self, x: torch.Tensor) -> torch.Tensor:
        return softcap(self.lm_head(rms(x)))

    def slots_active(self, T: int) -> bool:
        return self.slot_weight > 0 and self.J > 0 and T >= self.K

    def param_groups(self):
        return split_param_groups(self)

    # ------------------------------------------------------------------ training / scoring
    def forward(self, idx: torch.Tensor, targets: torch.Tensor, return_slots: bool = False):
        B, T = idx.shape
        dev = idx.device
        x = self.embed(idx)
        pos = torch.arange(T, device=dev)
        cos, sin = self.rotary(pos)
        use_slots = self.slots_active(T) and (self.training or return_slots)
        if use_slots:
            A, J, d = min(self.n_anchor, T - self.K + 1), self.J, self.d
            anchors = torch.rand(B, T - self.K + 1, device=dev).topk(A, dim=1).indices.sort(dim=1).values
            jj = torch.arange(1, J + 1, device=dev)
            spos = (anchors[:, :, None] + jj).reshape(B, A * J)  # slot j of anchor a sits at a+j
            scos, ssin = self.rotary(spos)
            anc_s = anchors[:, :, None].expand(B, A, J).reshape(B, A * J)
            main_mask = pos[None, None, :] <= anc_s[:, :, None]  # [B, S, T]
            grp = torch.arange(A * J, device=dev) // J
            jx = torch.arange(A * J, device=dev) % J
            own = (grp[:, None] == grp[None, :]) & (jx[None, :] <= jx[:, None])  # [S, S]
            mask = torch.cat([main_mask, own[None].expand(B, -1, -1)], 2)[:, None]  # [B, 1, S, T+S]
            slot_tgt = targets.gather(1, spos)  # slot at position p predicts x[p+1] = targets[p]
            if self.slot_null == "shuffle":
                flat = slot_tgt.reshape(-1)
                slot_tgt = flat[torch.randperm(flat.numel(), device=dev)].view_as(slot_tgt)
        xs = None
        for li, blk in enumerate(self.blocks):
            if use_slots and li == self.slot_from:
                xa = x.gather(1, anchors[:, :, None].expand(B, A, self.d))
                xs = (xa[:, :, None, :] + self.mask_emb[None, None].to(x.dtype)).reshape(B, A * J, self.d)
            x_new, k, v = blk.causal(x, cos, sin)
            if xs is not None:
                xs, _, _ = blk.masked(xs, scos, ssin, k, v, mask)
            x = x_new
        logits = self.head(x)
        loss = ce(logits, targets)
        aux: dict = {}
        if use_slots:
            sl = self.head(xs)
            per = ce(sl, slot_tgt, reduction="none").view(B, A, J)
            valid = (slot_tgt != -1).view(B, A, J).float()
            slot_loss = (per * valid).sum() / valid.sum().clamp_min(1.0)
            aux["aux_loss"] = self.slot_weight * slot_loss
            aux["slot_ce"] = slot_loss.detach()
            pj = (per * valid).sum((0, 1)) / valid.sum((0, 1)).clamp_min(1.0)
            for j in range(J):
                aux[f"slot_ce_{j + 1}"] = pj[j].detach()
            if return_slots:
                aux["_slot_logits"] = sl.view(B, A, J, -1)
                aux["_slot_pos"] = spos.view(B, A, J)
                aux["_main_logits"] = logits
        return loss, aux

    @torch.no_grad()
    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = self.embed(idx)
        cos, sin = self.rotary(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x, _, _ = blk.causal(x, cos, sin)
        return self.head(x)

    # ------------------------------------------------------------------ cost accounting
    def _per_pos_flops(self, ctx: float, layers: int, with_head: bool = True) -> float:
        body = 2 * layers * block_matmul_params(self.d, self.cfg.mlp_ratio)
        attn = layers * 4 * ctx * self.d
        return body + attn + (2 * self.d * self.V if with_head else 0)

    def flops_per_token(self) -> float:
        """Training forward+backward FLOPs per main-stream token, slot overhead included (dense-mask slot
        attention over all T+S keys, as executed)."""
        T = self.cfg.seq_len
        f = 3 * self._per_pos_flops(T / 2, self.L)
        if self.slots_active(T):
            A = min(self.n_anchor, T - self.K + 1)
            S = A * self.J
            f += 3 * (S / T) * self._per_pos_flops(T + S, self.L - self.slot_from)
        return float(f)

    def inference_flops_per_token(self, ctx_len: float) -> float:
        """Plain cached AR decoding (decode_step), forward only."""
        return float(self._per_pos_flops(ctx_len, self.L))

    def spec_pass_flops(self, ctx_len: float, n_verify: int | None = None) -> float:
        """Forward FLOPs of one speculative tree pass: n verify positions + n*(K-1) slots."""
        n = self.K if n_verify is None else n_verify
        Q = n + n * self.J
        f = n * self._per_pos_flops(ctx_len + Q, self.L)
        f += n * self.J * self._per_pos_flops(ctx_len + Q, self.L - self.slot_from)
        return float(f)

    # ------------------------------------------------------------------ plain cached decoding (infer_bench hooks)
    def _new_cache(self, B, cap, ref):
        return KVCache(self.L, B, self.H, self.hd, cap, ref.device, self.embed.weight.dtype)

    @torch.no_grad()
    def prefill(self, idx: torch.Tensor):
        B, P = idx.shape
        cache = self._new_cache(B, P + 64, idx)
        x = self.embed(idx)
        cos, sin = self.rotary(torch.arange(P, device=idx.device))
        for li, blk in enumerate(self.blocks):
            x, k, v = blk.causal(x, cos, sin)
            cache.write(li, k, v, 0)
        cache.n = P
        return self.head(x[:, -1]), cache

    @torch.no_grad()
    def decode_step(self, tok: torch.Tensor, cache: KVCache):
        n = cache.n
        cache.ensure(n + 1)
        x = self.embed(tok)
        cos, sin = self.rotary(torch.tensor([n], device=tok.device))
        for li, blk in enumerate(self.blocks):
            kc, vc = cache.get(li)
            x, k, v = blk.masked(x, cos, sin, kc, vc, None)
            cache.write(li, k, v, n)
        cache.n = n + 1
        return self.head(x[:, -1]), cache

    def cache_bytes(self, cache: KVCache) -> dict:
        return cache.bytes()

    @torch.no_grad()
    def generate_plain(self, prompt: torch.Tensor, n_new: int):
        lg, cache = self.prefill(prompt)
        out = []
        for _ in range(n_new):
            t = lg.argmax(-1, keepdim=True)
            out.append(t)
            lg, cache = self.decode_step(t, cache)
        return torch.cat(out, 1)

    # ------------------------------------------------------------------ speculative tree pass
    def _tree_mask(self, base: int, n: int, with_slots: bool, dev) -> torch.Tensor:
        J = self.J if with_slots else 0
        Q = n + n * J
        vi = torch.arange(n, device=dev)
        # query descriptors: anchor index i and slot offset j (0 = the verify position itself)
        qi = torch.cat([vi, vi.repeat_interleave(J)]) if J else vi
        qj = torch.cat([torch.zeros(n, dtype=torch.long, device=dev), torch.arange(1, J + 1, device=dev).repeat(n)]) \
            if J else torch.zeros(n, dtype=torch.long, device=dev)
        is_slot_q = qj > 0
        cache_part = torch.ones(Q, base, dtype=torch.bool, device=dev)
        verify_part = vi[None, :] <= qi[:, None]  # every query sees verify positions up to its anchor
        if J:
            si = vi.repeat_interleave(J)
            sj = torch.arange(1, J + 1, device=dev).repeat(n)
            slot_part = is_slot_q[:, None] & (si[None, :] == qi[:, None]) & (sj[None, :] <= qj[:, None])
            m = torch.cat([cache_part, verify_part, slot_part], 1)
        else:
            m = torch.cat([cache_part, verify_part], 1)
        return m[None, None]

    @torch.no_grad()
    def _tree_pass(self, cache: KVCache, vtok: torch.Tensor):
        """vtok [1, n] tokens at positions cache.n .. cache.n+n-1. Returns main logits [n, V],
        slot logits [n, J, V] and the verify positions' per-layer (k, v) for committing."""
        dev = vtok.device
        base, n, J = cache.n, vtok.shape[1], self.J
        vpos = base + torch.arange(n, device=dev)
        spos = (vpos[:, None] + torch.arange(1, J + 1, device=dev)).reshape(-1)
        mask_v = self._tree_mask(base, n, False, dev)
        mask_f = self._tree_mask(base, n, True, dev)
        cos_v, sin_v = self.rotary(vpos)
        cos_f, sin_f = self.rotary(torch.cat([vpos, spos]))
        x = self.embed(vtok)
        stream = x
        pend = []
        for li, blk in enumerate(self.blocks):
            if li == self.slot_from:
                xs = (stream[:, :, None, :] + self.mask_emb[None, None].to(stream.dtype)).reshape(1, n * J, self.d)
                stream = torch.cat([stream, xs], 1)
            full = stream.shape[1] > n
            kc, vc = cache.get(li)
            stream, k, v = blk.masked(stream, cos_f if full else cos_v, sin_f if full else sin_v, kc, vc,
                                      mask_f if full else mask_v)
            pend.append((k[:, :, :n], v[:, :, :n]))
        logits = self.head(stream)[0]
        return logits[:n], logits[n:].view(n, J, -1), pend

    def _commit(self, cache: KVCache, pend, m: int):
        base = cache.n
        cache.ensure(base + m)
        for li, (k, v) in enumerate(pend):
            cache.write(li, k[:, :, :m], v[:, :, :m], base)
        cache.n = base + m

    @torch.no_grad()
    def spec_generate(self, prompt: torch.Tensor, n_new: int, max_verify: int | None = None):
        """Greedy lossless self-speculative decoding, batch 1. Returns (tokens [1, n_new], stats)."""
        assert prompt.shape[0] == 1 and self.J > 0
        P = prompt.shape[1]
        kmax = self.K if max_verify is None else max(1, min(max_verify, self.K))
        if P > 1:
            _, cache = self.prefill(prompt[:, :-1])
        else:
            cache = self._new_cache(1, 64, prompt)
        cache.ensure(P + n_new + 2 * self.K)
        vtok = prompt[:, -1:]
        out: list[int] = []
        passes, accepted_hist = 0, []
        while len(out) < n_new:
            ml, sl, pend = self._tree_pass(cache, vtok)
            passes += 1
            n = vtok.shape[1]
            pred = ml.argmax(-1)
            m = 0
            while m < n - 1 and int(pred[m]) == int(vtok[0, m + 1]):
                m += 1
            out.extend(int(t) for t in vtok[0, 1:m + 1])
            out.append(int(pred[m]))
            accepted_hist.append(m)
            self._commit(cache, pend, m + 1)
            drafts = sl[m].argmax(-1)[:kmax - 1]
            vtok = torch.cat([pred[m:m + 1], drafts])[None]
        toks = torch.tensor(out[:n_new], device=prompt.device)[None]
        later = accepted_hist[1:]
        stats = dict(passes=passes, tokens=len(out),
                     tokens_per_pass=len(out) / passes,
                     tokens_per_pass_after_first=(sum(later) + len(later)) / max(len(later), 1),
                     accepted_hist=[later.count(i) for i in range(self.K)])
        return toks, stats


def build(cfg: ModelConfig) -> nn.Module:
    return MaskSlotLM(cfg)
