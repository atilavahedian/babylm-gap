"""I2 KWTA: sparse-expansion FFN with k-winners-take-all inhibition and homeostasis
(mushroom-body / dentate-gyrus / cerebellar-granule analogue; idea card sprint1/ideas/I_kwta.md).

Per block, the dense ReLU^2 FFN (d -> h -> d, 2*d*h MACs) is replaced by
  1. EXPANSION with sparse connectivity: H = e*d units (default e=32) arranged in R receptive fields; each
     field reads a fixed random subset of s input dims (in-degree s << d, default s=d/8, as in Kenyon /
     granule cells) and holds H/R units with learned weights.  MACs: H*s  (= half the dense FFN budget).
  2. GLOBAL INHIBITION (k-WTA, the APL / Golgi interneuron): per token only the k most excited units fire,
     with a subtractive threshold equal to the (k+1)-th excitation, so units enter and leave the active set
     continuously:  a_j = relu(z_j + b_j - theta)^2 for the winners, 0 otherwise.
  3. HOMEOSTATIC excitability b_j = beta * log(target / duty_j) (duty_j = running firing rate, a buffer with
     no gradient; training forwards accumulate counts, the harness hook on_step_end folds them in once per
     optimizer step): silent units become excitable, over-used ones less.
  4. SPARSE READOUT y = sum_{j in winners} a_j R_j via embedding_bag: MACs k*d (default k = 4d, the other
     half of the dense budget).  At equal FLOPs the readout table is e/4 = 8x the dense FFN's output matrix.
Optimizer: w_in (batched [R, H/R, s]) and readout ([d, H]) go to the harness Muon like the dense FFN's fc /
proj (Muon keeps never-active readout columns exactly at zero, so updates stay sparse).
Built on the sprint baseline (i_common.IGPT subclasses baseline.GPT and swaps only the FFNs).
Equal FLOPs with the dense FFN; ~4.5x its parameters (that capacity is the point; the null holds it fixed).

cfg.extra knobs (all optional):
  ffn           "kwta" (default) | "dense" (= sprint1/models/baseline.py itself, the equal-code reference)
  null          True -> kwta_mode="hash": winners chosen by a fixed hash of the input token id (hash-layer
                style), same expansion, same k, same readout, same params and FLOPs; no competition.
  kwta_mode     "kwta" | "hash"
  kwta_expand   32     H = expand * n_embd
  kwta_indeg    None   in-degree s (default: chosen so H*s = d*mlp_hidden, half the dense MACs)
  kwta_fields   None   receptive fields R (default 2*expand; must divide H)
  kwta_k        None   winners per token (default: (2*d*mlp_hidden - H*s) / d, FLOP-matched)
  kwta_boost    0.25   homeostasis strength beta (0 disables); boost clipped to +-1
  kwta_rho      0.01   duty-cycle EMA rate per optimizer step
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.i_common import IGPT, build_baseline, init_bound


def hash_indices(tok: torch.Tensor, H: int, k: int) -> torch.Tensor:
    """k indices in [0, H) per token id: arithmetic progression with token-dependent start and stride.
    Distinct whenever gcd(stride, H) = 1, guaranteed for H = 2^a 3^b (all default sizes)."""
    h = (tok * 2654435761) % H
    stride = 1 + 6 * ((tok * 40503) % max(1, H // 6))
    j = torch.arange(k, device=tok.device, dtype=tok.dtype)
    return (h[:, None] + j[None, :] * stride[:, None]) % H


class KWTAMLP(nn.Module):
    def __init__(self, d: int, H: int, s: int, R: int, k: int, seed: int, mode: str = "kwta",
                 boost: float = 0.25, rho: float = 0.01):
        super().__init__()
        assert H % R == 0 and 0 < k < H and 0 < s <= d
        self.d, self.H, self.s, self.R, self.k, self.u = d, H, s, R, k, H // R
        self.mode, self.boost, self.rho = mode, float(boost), float(rho)
        g = torch.Generator().manual_seed(1009 + seed)
        sub = torch.stack([torch.randperm(d, generator=g)[:s] for _ in range(R)])
        self.register_buffer("sub_idx", sub.reshape(-1), persistent=True)
        bound = init_bound(s)  # baseline CastedLinear init with fan_in = in-degree
        # stored [out, in] per field (batched 2-D) so the harness Muon treats each field like a Linear weight
        self.w_in = nn.Parameter(torch.empty(R, self.u, s).uniform_(-bound, bound))
        # readout stored [d, H] (= Linear(H, d).weight, like the dense FFN's proj; Muon keeps unused rows at 0)
        self.readout = nn.Parameter(torch.zeros(d, H))
        self.register_buffer("duty", torch.full((H,), k / H), persistent=True)
        # firing counts accumulated by training forwards; folded into `duty` only in on_step_end, so a forward's
        # output never depends on other rows of the same or a repeated call (batch isolation / causality audit)
        self.register_buffer("cnt_acc", torch.zeros(H + 1), persistent=True)  # [H counts, n tokens]
        self.log_target = math.log(k / H)

    def excitation(self, x2: torch.Tensor) -> torch.Tensor:
        N = x2.size(0)
        xs = x2[:, self.sub_idx].view(N, self.R, self.s)
        return torch.einsum("nrs,rus->nru", xs, self.w_in).reshape(N, self.H)

    def forward(self, x: torch.Tensor, ctx=None) -> torch.Tensor:
        B, T, d = x.shape
        N = B * T
        z = self.excitation(x.reshape(N, d))
        if self.mode == "kwta":
            b = (self.boost * (self.log_target - torch.log(self.duty + 1e-6))).clamp(-1.0, 1.0)
            sel = z + b.to(z.dtype)
            topv, topi = torch.topk(sel, self.k + 1, dim=-1)
            vals = F.relu(topv[:, : self.k] - topv[:, self.k:]).square()
            idx = topi[:, : self.k]
            if self.training:
                with torch.no_grad():
                    self.cnt_acc[: self.H].scatter_add_(
                        0, idx.reshape(-1), torch.ones(idx.numel(), device=x.device, dtype=self.cnt_acc.dtype))
                    self.cnt_acc[self.H] += N
        else:  # hash null: same units, same k, same readout, selection by token identity only
            idx = hash_indices(ctx["idx"].reshape(N), self.H, self.k)
            vals = F.relu(torch.gather(z, 1, idx)).square()
        y = F.embedding_bag(idx, self.readout.t(), per_sample_weights=vals.to(self.readout.dtype), mode="sum")
        return y.view(B, T, d).to(x.dtype)

    @torch.no_grad()
    def on_step_end(self, step: int, total_steps: int) -> None:
        n = self.cnt_acc[self.H]
        if n > 0:
            self.duty.mul_(1.0 - self.rho).add_(self.cnt_acc[: self.H] / n, alpha=self.rho)
        self.cnt_acc.zero_()

    def fwd_macs_per_token(self) -> float:
        return float(self.H * self.s + self.k * self.d)

    def stats(self):
        tgt = self.k / self.H
        return {"dead_frac": (self.duty < 0.1 * tgt).float().mean(),
                "duty_max_over_target": self.duty.max() / tgt}


def kwta_dims(cfg: ModelConfig) -> dict:
    ex = cfg.extra or {}
    d = cfg.n_embd
    hidden = int(round(cfg.mlp_ratio * d))
    budget = 2 * d * hidden                      # dense FFN forward MACs per token
    e = int(ex.get("kwta_expand", 32))
    H = e * d
    s = int(ex.get("kwta_indeg") or max(1, round(budget / 2 / H)))
    R = int(ex.get("kwta_fields") or 2 * e)
    k = int(ex.get("kwta_k") or max(1, round((budget - H * s) / d)))
    return dict(H=H, s=min(s, d), R=R, k=k)


def build(cfg: ModelConfig) -> nn.Module:
    ex = cfg.extra or {}
    d = cfg.n_embd
    hidden = int(round(cfg.mlp_ratio * d))
    if ex.get("ffn", "kwta") == "dense":
        return build_baseline(cfg)  # the equal-code reference arm is the baseline itself
    mode = "hash" if ex.get("null", False) else str(ex.get("kwta_mode", "kwta"))
    dims = kwta_dims(cfg)
    boost = float(ex.get("kwta_boost", 0.25))
    rho = float(ex.get("kwta_rho", 0.01))
    return IGPT(cfg, lambda i: KWTAMLP(d, dims["H"], dims["s"], dims["R"], dims["k"], seed=i, mode=mode,
                                       boost=boost, rho=rho), needs_idx=(mode == "hash"))
