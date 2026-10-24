"""F1 SyncPhase: Kuramoto phase-synchronisation attention for a causal LM. Owned by agent F.

Card: sprint1/ideas/F_kuramoto.md.

Every token carries, per head, m oscillator phases theta[b, h, t, 0:m] (float32). They live OUTSIDE the
residual stream and evolve across depth by Kuramoto dynamics whose coupling graph is the layer's own causal
attention:

    score_ij  uses q_i, k_j rotated by RoPE on pairs 0..P-m-1 and by the DYNAMIC phases theta_i, theta_j on the
              last m (lowest-frequency) pairs  ->  phase-coherent tokens get a cos(theta_i - theta_j) score term
    z_i       = sum_j a_ij exp(i theta_j)                     (order parameter; 2m extra value channels, ~free)
    theta_i  <- theta_i + K_lh * Im(exp(-i theta_i) z_i)      (Kuramoto pull toward the attended mean phase;
                                                                K learned per layer/head/pair, may go negative)
                         + nu_l(rms(x_i))                      (content-driven natural frequency, zero-init)
    x_i      += W_fb [cos theta_i, sin theta_i]                 (phase read-out into the residual, zero-init)

Optional interference variant (extra["kp_interf"]=True): values are mixed with SIGNED weights
a_ij * cos(phi_i - phi_j), phi = theta[..., 0], i.e. phase-coherent tokens add, anti-phase tokens cancel.

Arms (cfg.extra):
    kp_mode   "kuramoto" (candidate, default) | "static" (= null) | "hash" | "off"
    null      True -> kp_mode="static": identical modules, kernels and FLOPs; coupling K multiplied by 0, so the
              phase is a content-driven rotation (theta_0 + sum_l nu_l) WITHOUT synchronisation dynamics.
    "hash"    phases = 2*pi*hash(token id, head, pair), fixed, no learning.
    "off"     theta = 0: the dynamic pairs are plain zero-frequency (partial-RoPE) pairs. Matched reference.
    kp_pairs  m, default 4 (8 of hd dims)       kp_interf  False      kp_feedback True
    kp_K0     initial coupling, default 1.0     kp_theta_std initial phase spread (rad), default 1.0
    rope_base 10000.0
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.f_common import FGPT, attn_core_flops, hash_uniform, rms, rope_freqs, rotate

MODES = ("kuramoto", "static", "hash", "off")


def _settings(cfg: ModelConfig) -> dict:
    ex = dict(cfg.extra or {})
    mode = "static" if ex.get("null", False) else ex.get("kp_mode", "kuramoto")
    assert mode in MODES, mode
    hd = cfg.n_embd // cfg.n_head
    m = int(ex.get("kp_pairs", 4))
    assert 1 <= m <= hd // 4, "kp_pairs must leave at least half of the pairs for positional RoPE"
    return dict(mode=mode, m=m, interf=bool(ex.get("kp_interf", False)), feedback=bool(ex.get("kp_feedback", True)),
                K0=float(ex.get("kp_K0", 1.0)), theta_std=float(ex.get("kp_theta_std", 1.0)),
                base=float(ex.get("rope_base", 10000.0)))


class SyncAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, s: dict):
        super().__init__()
        d, H = cfg.n_embd, cfg.n_head
        self.d, self.H, self.hd = d, H, d // H
        self.P = self.hd // 2
        self.m, self.mode, self.interf, self.feedback = s["m"], s["mode"], s["interf"], s["feedback"]
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        freqs = rope_freqs(self.hd, s["base"])
        self.register_buffer("pos_freqs", freqs[: self.P - self.m].clone(), persistent=False)
        # Kuramoto coupling per head and pair, and content-driven natural frequency (phase velocity)
        # 1-D so the harness files it under AdamW "scalar" (2-D adam params would get the embedding LR)
        self.K = nn.Parameter(torch.full((H * self.m,), s["K0"]))
        self.nu = nn.Linear(d, H * self.m, bias=False)
        nn.init.zeros_(self.nu.weight)
        if self.feedback:
            self.fb = nn.Linear(2 * H * self.m, d, bias=False)
            nn.init.zeros_(self.fb.weight)

    def d_v(self) -> int:
        return (2 * self.hd if self.interf else self.hd) + 2 * self.m

    def forward(self, h: torch.Tensor, x: torch.Tensor, ctx: dict):
        B, T, _ = h.shape
        H, hd, P, m = self.H, self.hd, self.P, self.m
        q, k, v = self.qkv(h).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k = rms(q), rms(k)
        theta = ctx["theta"]                                                     # [B, H, T, m] float32
        pos = torch.arange(T, device=h.device, dtype=torch.float32)
        ang_pos = (pos[:, None] * self.pos_freqs[None, :]).expand(B, H, T, P - m)
        ang = torch.cat([ang_pos, theta], dim=-1)                                # [B, H, T, P]
        cos, sin = ang.cos(), ang.sin()
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        ct, st = theta.cos(), theta.sin()
        if self.interf:
            cp, sp = ct[..., :1].to(v.dtype), st[..., :1].to(v.dtype)
            vext = torch.cat([v * cp, v * sp, ct.to(v.dtype), st.to(v.dtype)], dim=-1)
        else:
            vext = torch.cat([v, ct.to(v.dtype), st.to(v.dtype)], dim=-1)
        y = F.scaled_dot_product_attention(q, k, vext, is_causal=True)          # [B, H, T, d_v]
        if self.interf:
            yv = y[..., :hd] * cp + y[..., hd:2 * hd] * sp
        else:
            yv = y[..., :hd]
        C, S = y[..., -2 * m:-m].float(), y[..., -m:].float()                   # attended mean of e^{i theta_j}
        out = self.proj(yv.transpose(1, 2).reshape(B, T, self.d))
        if self.mode in ("kuramoto", "static"):
            drive = S * ct - C * st                                              # r sin(psi - theta), |.| <= 1
            K = (self.K if self.mode == "kuramoto" else self.K * 0.0).view(H, m)
            vel = self.nu(rms(x + out)).view(B, T, H, m).transpose(1, 2).float()
            theta = theta + K[None, :, None, :] * drive + vel
        if self.feedback:
            ph = torch.cat([theta.cos(), theta.sin()], dim=-1)                   # [B, H, T, 2m]
            out = out + self.fb(ph.transpose(1, 2).reshape(B, T, 2 * H * m).to(out.dtype))
        ctx = dict(ctx)
        ctx["theta"] = theta
        ctx["r_last"] = torch.sqrt(C.square() + S.square() + 1e-12)
        return out, ctx


class SyncPhaseGPT(FGPT):
    def __init__(self, cfg: ModelConfig):
        s = _settings(cfg)
        self.s = s
        super().__init__(cfg, lambda i: SyncAttention(cfg, s))
        d, H, m = cfg.n_embd, cfg.n_head, s["m"]
        self.H, self.m = H, m
        self.phase_init = nn.Linear(d, H * m, bias=False)                       # content-dependent initial phase
        nn.init.normal_(self.phase_init.weight, std=s["theta_std"] / math.sqrt(d))

    def init_ctx(self, idx: torch.Tensor, x0: torch.Tensor) -> dict:
        B, T = idx.shape
        H, m, mode = self.H, self.m, self.s["mode"]
        if mode == "off":
            theta = x0.new_zeros(B, H, T, m, dtype=torch.float32)
        elif mode == "hash":
            u = torch.stack([hash_uniform(idx, 1000 + j) for j in range(H * m)], dim=-1)  # [B, T, H*m]
            theta = (2 * math.pi * u).view(B, T, H, m).transpose(1, 2).contiguous()
        else:
            theta = self.phase_init(x0).view(B, T, H, m).transpose(1, 2).float()
        return {"idx": idx, "theta": theta}

    def aux_from_ctx(self, ctx: dict) -> dict:
        aux = {"kp_r_last": ctx["r_last"].mean().detach()}
        Ks = torch.stack([a.K for a in self.attns])
        aux["kp_K_mean"] = Ks.mean().detach()
        aux["kp_K_absmax"] = Ks.abs().max().detach()
        return aux

    def flops_per_token(self) -> float:
        """Forward+backward FLOPs per token as executed at cfg.seq_len: 6 x matmul params (includes phase_init,
        nu, fb) + causal attention core with d_qk = hd and d_v = hd + 2m (or 2hd + 2m with interference).
        Elementwise phase work (sin/cos, rotations, drive) is O(H*m) per token per layer and not counted."""
        c = self.cfg
        a = self.attns[0]
        core = attn_core_flops(c.seq_len, a.H, a.hd, a.d_v(), c.seq_len / 2)
        return 6.0 * self.matmul_params() + c.n_layer * core

    def reference_flops_per_token(self) -> float:
        """Same body with a plain d_v = hd attention and no phase modules (what baseline-style attention costs)."""
        c = self.cfg
        a = self.attns[0]
        extra = self.phase_init.weight.numel() + sum(
            at.nu.weight.numel() + (at.fb.weight.numel() if at.feedback else 0) for at in self.attns)
        core = attn_core_flops(c.seq_len, a.H, a.hd, a.hd, c.seq_len / 2)
        return 6.0 * (self.matmul_params() - extra) + c.n_layer * core


def build(cfg: ModelConfig) -> nn.Module:
    return SyncPhaseGPT(cfg)
