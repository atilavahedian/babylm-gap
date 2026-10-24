"""I1 Dendrite: two-compartment pyramidal-neuron FFN (idea card sprint1/ideas/I_dendrite.md).

Every FFN hidden unit is a neuron with
  * a BASAL tree of `nb` branches; branch b sees only its own slice x_b (d/nb inputs) of the block input
    (sparse dendritic connectivity) and has a local supralinear "dendritic spike" (NMDA-like Hill function):
        u_j = sum_b [ z_jb + a_jb * hill(z_jb - th_jb) ],   z_jb = w_jb . x_b,   hill(r) = relu(r)^2/(1+relu(r)^2)
    The somatic output is h_j = relu(u_j)^2 (the baseline nonlinearity).  Because the branch inputs are
    disjoint slices and every branch weight is learned, sum_b z_jb is exactly a dense projection: the ONLY
    change from the point-neuron FFN is that the extra nonlinearity is applied per branch (compartmentalised).
  * an APICAL tuft driven by slow context (multi-timescale causal leaky integration of a learned rank-r
    projection of the block input, time constants 2..1024 tokens) that multiplies the somatic output:
        h_j <- h_j * 2 sigmoid(U_j . c_t)   (gain = 1 at init; context modulates, never drives).

Built on the sprint baseline (i_common.IGPT subclasses baseline.GPT and swaps only the FFNs). Equal parameters
with the baseline ReLU^2 FFN of width mlp_ratio*d (neuron count n is solved for); equal FLOPs up to O(nb*n)
elementwise ops.

cfg.extra knobs (all optional):
  ffn               "dendrite" (default) | "dense"  (dense = sprint1/models/baseline.py itself)
  null              True -> compartments off AND apical context taken from a fixed training row ("foreign")
  dend_branches     4      basal branches per neuron (must divide n_embd)
  dend_compartments True   False -> same extra nonlinearity applied to the whole-neuron drive (point neuron)
  dend_apical       "ctx"  "ctx" | "foreign" (fixed training row from the previous step; null) | "off"
  dend_apical_rank  32     apical context channels r
  dend_tau          (2, 1024) min / max leaky-integration time constants (tokens), log-spaced over r
  dend_spike_init   1.0    initial spike amplitude a
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.i_common import IGPT, CastedLinear, build_baseline, init_bound


def hill(r: torch.Tensor) -> torch.Tensor:
    """Bounded supralinear dendritic spike: 0 below threshold, ~r^2 just above, saturates at 1."""
    r2 = F.relu(r).square()
    return r2 / (1.0 + r2)


def causal_ema(u: torch.Tensor, log_lam: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """c_t = lam * c_{t-1} + (1 - lam) * u_t per channel, causal, chunk-parallel, in fp32 (autocast off).
    u [B,T,r], log_lam [r] < 0."""
    with torch.autocast(device_type=u.device.type, enabled=False):
        return _causal_ema_fp32(u.float(), log_lam.float(), chunk)


def _causal_ema_fp32(u: torch.Tensor, log_lam: torch.Tensor, chunk: int) -> torch.Tensor:
    B, T, r = u.shape
    C = min(chunk, T)
    pad = (-T) % C
    if pad:
        u = F.pad(u, (0, 0, 0, pad))
    n = u.size(1) // C
    u = u.view(B, n, C, r)
    t = torch.arange(C, device=u.device, dtype=u.dtype)
    diff = t[:, None] - t[None, :]
    lam = log_lam.exp()
    w = torch.exp(diff.clamp(min=0)[None] * log_lam[:, None, None]) * (diff >= 0)[None].to(u.dtype)
    w = w * (1.0 - lam)[:, None, None]                         # [r, C, C]
    intra = torch.einsum("rts,bnsr->bntr", w, u)               # within-chunk contribution
    carry = torch.exp((t + 1)[:, None] * log_lam[None, :])     # [C, r]  lam^(t+1)
    state = u.new_zeros(B, r)
    outs = []
    for i in range(n):  # static trip count (T is fixed): compile-friendly
        o = intra[:, i] + carry[None] * state[:, None, :]
        outs.append(o)
        state = o[:, -1]
    return torch.stack(outs, dim=1).reshape(B, n * C, r)[:, :T]


class DendriticMLP(nn.Module):
    def __init__(self, d: int, n: int, branches: int = 4, compartments: bool = True, apical: str = "ctx",
                 apical_rank: int = 32, tau=(2.0, 1024.0), spike_init: float = 1.0, max_len: int = 1024):
        super().__init__()
        assert d % branches == 0, "dend_branches must divide n_embd"
        self.d, self.n, self.nb, self.s = d, n, branches, d // branches
        self.compartments = compartments
        self.apical = apical if apical_rank > 0 else "off"
        bound = init_bound(d)  # the baseline c_fc init: with spike 0 the module equals the baseline MLP
        # basal weights stored exactly like the dense fc weight [n, d] (same Muon shape and scale);
        # branch b of neuron j owns columns b*s:(b+1)*s
        self.w_basal = nn.Parameter(torch.empty(n, d).uniform_(-bound, bound))
        # per-branch spike amplitude / threshold, flat 1-D so the harness gives them the scalar Adam LR
        self.spike = nn.Parameter(torch.full((branches * n,), float(spike_init)))
        self.thresh = nn.Parameter(torch.zeros(branches * n))
        self.proj = CastedLinear(n, d, zero_init=True)
        self.r = apical_rank if self.apical != "off" else 0
        if self.r > 0:
            self.ap_in = CastedLinear(d, self.r)
            self.ap_out = CastedLinear(self.r, n, zero_init=True)
            taus = torch.logspace(math.log10(tau[0]), math.log10(tau[1]), self.r)
            self.register_buffer("log_lam", torch.log1p(-1.0 / taus), persistent=False)
        if self.apical == "foreign":
            # null: apical input comes from a fixed TRAIN row (the last row of the previous optimizer step's
            # last micro-batch), never from another row of the current batch (protocol 7.4)
            self.register_buffer("foreign_x", torch.zeros(max_len, d), persistent=True)
            self.register_buffer("pending_x", torch.zeros(max_len, d), persistent=False)

    def forward(self, x: torch.Tensor, ctx=None) -> torch.Tensor:
        B, T, d = x.shape
        xs = x.view(B, T, self.nb, self.s)
        W = self.w_basal.view(self.n, self.nb, self.s)
        z = torch.einsum("btgs,ngs->btgn", xs, W)               # branch potentials [B, T, nb, n]
        spike = self.spike.view(self.nb, self.n)
        thresh = self.thresh.view(self.nb, self.n)
        if self.compartments:
            u = z.sum(2) + (spike * hill(z - thresh)).sum(2)
        else:  # point neuron: same extra nonlinearity, same parameters, applied to the summed drive
            zt = z.sum(2)
            u = zt + self.nb * spike.mean(0) * hill(zt / math.sqrt(self.nb) - thresh.mean(0))
        h = F.relu(u).square()
        if self.r > 0:
            if self.apical == "foreign":  # null: same FLOPs, context of a fixed training row from an earlier step
                if self.training:
                    with torch.no_grad():
                        self.pending_x[:T].copy_(x[-1].detach().float())
                xa = self.foreign_x[:T].to(x.dtype).unsqueeze(0).expand(B, T, d)
            else:
                xa = x
            c = causal_ema(self.ap_in(xa).float(), self.log_lam).to(x.dtype)
            h = h * (2.0 * torch.sigmoid(self.ap_out(c)))
        return self.proj(h)

    @torch.no_grad()
    def on_step_end(self, step: int, total_steps: int) -> None:
        if self.apical == "foreign":
            self.foreign_x.copy_(self.pending_x)

    def fwd_macs_per_token(self) -> float:
        macs = 2.0 * self.n * self.d
        if self.r > 0:
            macs += self.d * self.r + self.r * self.n + 64 * self.r  # + in-chunk leaky integration
        return macs

    def stats(self):
        out = {"spike_mean": self.spike.detach().mean()}
        if self.r > 0:
            out["apical_w_norm"] = self.ap_out.weight.detach().norm()
        return out


def solve_neurons(d: int, budget: int, branches: int, r: int) -> int:
    """Largest n (multiple of 4) with params 2*d*n + 2*branches*n + d*r + r*n <= budget."""
    n = (budget - d * r) // (2 * d + 2 * branches + r)
    return int(n - n % 4)


def build(cfg: ModelConfig) -> nn.Module:
    ex = cfg.extra or {}
    d = cfg.n_embd
    hidden = int(round(cfg.mlp_ratio * d))
    if ex.get("ffn", "dendrite") == "dense":
        return build_baseline(cfg)  # the equal-code reference arm is the baseline itself
    null = bool(ex.get("null", False))
    nb = int(ex.get("dend_branches", 4))
    compartments = bool(ex.get("dend_compartments", True)) and not null
    apical = str(ex.get("dend_apical", "ctx"))
    if null and apical != "off":
        apical = "foreign"
    r = int(ex.get("dend_apical_rank", 32)) if apical != "off" else 0
    tau = tuple(ex.get("dend_tau", (2.0, 1024.0)))
    n = solve_neurons(d, 2 * d * hidden, nb, r)
    spike_init = float(ex.get("dend_spike_init", 1.0))
    return IGPT(cfg, lambda i: DendriticMLP(d, n, nb, compartments, apical, r, tau, spike_init, cfg.seq_len))
