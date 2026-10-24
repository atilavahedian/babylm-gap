"""E1: Candidate-Refined Softmax (CRS).  Idea card: sprint1/ideas/E_refine.md

The dense LM head (rank <= d) proposes; a tiny NONLINEAR context-token scorer re-scores only its top-k
candidates; ONE softmax over the full vocabulary normalises the result exactly:

    z_v  = softcap(<h, W_v>)                                  base logits, all V tokens (as baseline)
    K(h) = top-k of z (no gradient through the choice)
    r(h, v) = cap * tanh( w . relu(A h + c + b_v) / cap )    for v in K(h), else 0
    b_v  = T_v + P W_v                                        free per-token code + projected output emb
    p(v | h) = softmax(z + r)_v                               exact, sums to 1 over all V tokens

Why: log p from a linear head is a rank-(d+1) matrix over (contexts x tokens) -- the softmax bottleneck
(Yang et al. 2018), which Godey et al. 2024 tie to small-model saturation at d < ~1000. r(h, v) is a joint
additive-ReLU interaction, so the refined log-prob matrix is no longer low rank, but only where the
probability mass is (top-k), at ~k*m extra FLOPs/token instead of MoS's (K-1)*V*d. High-entropy positions
with several plausible modes (typically context-new content tokens) are where a rank-d head is most
constrained, so that is where the gain is predicted to land.

Loss: exact NLL of the refined distribution, computed without a second [N, V] tensor:
    LSE' = logaddexp( LSE(z) + log(1 - sum_K softmax(z)_K),  LSE(z_K + r_K) ),  NLL = LSE' - z_t - r_t[t in K]
(r = 0 reproduces plain CE exactly; tested.)

Null control (cfg.extra["null"]=True): identical parameters and FLOPs (+0.0003% for the bank refresh), but the
scorer's context input is A h' with h' from a fixed TRAIN text block (e_common.NullBank: the block's hidden
states under the current body, refreshed only in on_step_end every null_refresh=50 steps), never from any row of
the current batch (protocol 7.4). So r can still learn per-token and candidate-set effects, but no
context-token interaction. Ablation rf_linear=True drops the ReLU (r becomes
additive in context and token: no rank gain). rf_k=0 is the plain-head reference on the same body.

Dense variant (rf_mode="block", the GPU-friendly follow-up, brainstorm #3): no top-k. The vocabulary is cut
into rf_blocks contiguous id blocks; each block c gets ReLU features g_c(h) = relu(A_c h + c_c) in R^r
(r = rf_block_dim) and each token a free code U_v in R^r, so z'_v = z_v + <g_{c(v)}(h), U_v>. Rank of the
log-prob matrix rises from d+1 to ~d + rf_blocks*r at +7% FLOPs (small), one extra [N, V] tensor; U is
zero-initialised (step 0 == plain). Same null (features of the fixed TRAIN block) and rf_linear ablation.

Knobs (cfg.extra): null=False, null_refresh=50, null_ids_path="auto" (see e_common.load_null_ids), rf_mode="topk"|"block", rf_blocks=64, rf_block_dim=32, rf_k=64 (candidates), rf_dim=128 (scorer width m), rf_cap=8.0,
rf_wfeat=True (b_v includes P W_v, shares strength with the output embedding), rf_linear=False,
softcap=15.0, log_novel=True (aux ce_new / ce_seen split), tokens_per_step=65536 (for the amortised
per-step P W projection in flops_per_token).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.e_common import (EBody, NullBank, body_flops_per_token, head_flops_per_token, make_head,
                                     softcap, split_ce_aux)


def refined_nll(z: torch.Tensor, cand: torch.Tensor, r: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Exact per-row NLL of softmax(z + scatter(cand, r)) at tgt.  z [N, V] fp32, cand [N, k] (distinct
    per row), r [N, k] fp32, tgt [N] (>= 0). No [N, V] intermediate beyond z itself."""
    lse = torch.logsumexp(z, dim=-1)                                   # [N]
    zc = z.gather(-1, cand)                                            # [N, k]
    p_k = torch.exp(zc - lse.unsqueeze(-1)).sum(-1)                    # top-k mass under base head
    log_rest = lse + torch.log((1.0 - p_k).clamp_min(1e-30))           # log sum_{v not in K} e^{z_v}
    lse_new = torch.logaddexp(log_rest, torch.logsumexp(zc + r, dim=-1))
    zt = z.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    rt = (r * (cand == tgt.unsqueeze(-1)).to(r.dtype)).sum(-1)
    return lse_new - zt - rt


class RefineLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ex = cfg.extra
        self.cfg = cfg
        d, V = cfg.n_embd, cfg.vocab_size
        self.body = EBody(cfg)
        self.lm_head = make_head(cfg, self.body)
        self.cap = float(ex.get("softcap", 15.0))
        self.mode = str(ex.get("rf_mode", "topk"))
        assert self.mode in ("topk", "block"), self.mode
        self.k = int(ex.get("rf_k", 64))
        self.m = int(ex.get("rf_dim", 128))
        self.rcap = float(ex.get("rf_cap", 8.0))
        self.null = bool(ex.get("null", False))
        self.linear = bool(ex.get("rf_linear", False))
        self.wfeat = bool(ex.get("rf_wfeat", True))
        self.log_novel = bool(ex.get("log_novel", True))
        self.tokens_per_step = float(ex.get("tokens_per_step", 65536))
        if self.k > 0 and self.mode == "block":
            C, r = int(ex.get("rf_blocks", 64)), int(ex.get("rf_block_dim", 32))
            self.C, self.r, self.S = C, r, -(-V // C)
            self.bctx = nn.Linear(d, C * r, bias=False)                  # A_c for all blocks, stacked
            self.bbias = nn.Parameter(torch.zeros(C * r))
            self.head_bcode = nn.Parameter(torch.zeros(C, self.S, r))    # U_v, zero-init -> step 0 == plain
        elif self.k > 0:
            m = self.m
            self.ctx = nn.Linear(d, m, bias=False)          # A: context features
            self.head_code = nn.Embedding(V, m)             # T_v: free per-token scorer code (head LR bucket)
            nn.init.normal_(self.head_code.weight, std=0.5)
            self.wproj = nn.Linear(d, m, bias=False) if self.wfeat else None  # P: from output embedding
            if self.wproj is not None:
                nn.init.normal_(self.wproj.weight, std=1.0)  # W_v ~ 0.002 at init: keep P W_v visible
            self.cbias = nn.Parameter(torch.zeros(m))
            self.out = nn.Parameter(torch.zeros(m))         # w: zero-init -> r = 0, model == baseline at step 0

        self.nullbank = NullBank(cfg, self.body, every=int(ex.get("null_refresh", 50))) if (self.null and self.k > 0) else None

    # ----- contract hooks -----
    def on_step_end(self, step: int, total_steps: int) -> None:
        if self.nullbank is not None:
            self.nullbank.maybe_refresh(self.body, step)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = self.body.muon_params()
        if self.k > 0 and self.mode == "block":
            muon = muon + [self.bctx.weight]
        elif self.k > 0:
            muon = muon + [self.ctx.weight] + ([self.wproj.weight] if self.wproj is not None else [])
        ids = {id(p) for p in muon}
        return {"muon": muon, "adam": [p for p in self.parameters() if id(p) not in ids]}

    def flops_per_token(self) -> float:
        cfg = self.cfg
        f = body_flops_per_token(cfg) + head_flops_per_token(cfg)
        if self.nullbank is not None:  # amortised refresh of the null's fixed TRAIN block
            f += self.nullbank.flops_per_token(body_flops_per_token(cfg) / 3.0, self.tokens_per_step)
        if self.k > 0 and self.mode == "block":
            d, V = cfg.n_embd, cfg.vocab_size
            f += 6.0 * d * self.C * self.r                # block features A h
            f += 6.0 * self.C * self.S * self.r            # <g_c, U_v> for every token (dense bmm)
            f += 3.0 * 2.0 * V                             # extra [N, V] add + its backward
        elif self.k > 0:
            d, V, k, m = cfg.n_embd, cfg.vocab_size, self.k, self.m
            f += 6.0 * d * m                       # A h
            f += 3.0 * (4.0 * k * m)               # add, relu, mul-add per candidate unit; fwd + 2x bwd
            f += 2.0 * V                           # top-k selection over V (fwd only, memory bound)
            f += 3.0 * 6.0 * k                     # exp / gathers / logaddexp terms per candidate
            if self.wproj is not None:             # P W for all V once per step, amortised
                f += 6.0 * V * d * m / self.tokens_per_step
        return f

    # ----- model -----
    def base_logits(self, h: torch.Tensor) -> torch.Tensor:
        return softcap(self.lm_head(h).float(), self.cap)

    def token_codes(self) -> torch.Tensor:
        b = self.head_code.weight
        if self.wproj is not None:
            b = b + self.wproj(self.lm_head.weight)
        return b                                                       # [V, m]

    def scores(self, h_ctx: torch.Tensor, cand: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """r [N, k] for contexts h_ctx [N, d] and candidate ids cand [N, k]."""
        a = self.ctx(h_ctx) + self.cbias                               # [N, m]
        u = a.unsqueeze(1) + F.embedding(cand, codes).to(a.dtype)      # [N, k, m]
        if not self.linear:
            u = F.relu(u)
        r = torch.einsum("nkm,m->nk", u, self.out.to(u.dtype)).float()
        return self.rcap * torch.tanh(r / self.rcap)

    def block_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Dense rank-raising term [N, V] (rf_mode="block"); h [N, d] is the (possibly null) context."""
        g = self.bctx(h) + self.bbias
        if not self.linear:
            g = F.relu(g)
        e = torch.einsum("ncr,csr->ncs", g.view(-1, self.C, self.r), self.head_bcode.to(g.dtype))
        return e.reshape(h.size(0), -1)[:, : self.cfg.vocab_size].float()

    def _refine_inputs(self, h: torch.Tensor):
        B, T, d = h.shape
        z = self.base_logits(h).view(B * T, -1)
        cand = torch.topk(z.detach(), self.k, dim=-1).indices
        h_ctx = self.nullbank.context(B, T, h.dtype) if self.null else h
        r = self.scores(h_ctx.reshape(B * T, d), cand, self.token_codes())
        return z, cand, r

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        B, T = idx.shape
        h = self.body.hidden(idx)
        valid = targets >= 0
        if self.k == 0 or self.mode == "block":
            z = self.base_logits(h).view(B * T, -1)
            aux = {}
            if self.k > 0:
                hc = self.nullbank.context(B, T, h.dtype) if self.null else h
                e = self.block_logits(hc.reshape(B * T, -1))
                z = z + e
                with torch.no_grad():
                    aux = {"rf_abs_r": e.abs().mean()}
            ce = F.cross_entropy(z, targets.reshape(-1).clamp_min(0), reduction="none")
        else:
            z, cand, r = self._refine_inputs(h)
            ce = refined_nll(z, cand, r, targets.reshape(-1).clamp_min(0))
            with torch.no_grad():
                aux = {"rf_abs_r": r.abs().mean()}
        ce = ce.view(B, T) * valid
        loss = ce.sum() / valid.sum().clamp_min(1)
        if self.log_novel:
            aux.update(split_ce_aux(ce, idx, targets))
        return loss, aux

    @torch.no_grad()
    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Refined logits [B, T, V] (eval / sampling / infer_bench recompute path)."""
        B, T = idx.shape
        h = self.body.hidden(idx)
        if self.k == 0:
            return self.base_logits(h)
        if self.mode == "block":
            hc = self.nullbank.context(B, T, h.dtype) if self.null else h
            return self.base_logits(h) + self.block_logits(hc.reshape(B * T, -1)).view(B, T, -1)
        z, cand, r = self._refine_inputs(h)
        return z.scatter_add(-1, cand, r).view(B, T, -1)


def build(cfg: ModelConfig) -> nn.Module:
    return RefineLM(cfg)
