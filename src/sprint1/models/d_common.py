"""Shared building blocks for Agent D's sprint-1 models (sprint1/models/d_*.py).

Self-contained (does not import baseline.py, which was written in parallel): a modded-nanogpt-style
pre-norm block (parameter-free RMSNorm, RoPE, QK-norm, ReLU^2 MLP, zero-initialised output
projections, untied head, logit soft-cap) plus the pieces Agent D's models need for honest
inference measurement: incremental KV-cache decoding with ragged (per-row variable length) caches.

Every d_* model built from these blocks exposes, in addition to the sprint-1 contract:
    forward_logits(idx) -> logits [B, T, V]            (eval semantics, for tests)
    prefill(idx) -> (logits_last [B, V], cache)
    decode_step(tok [B, 1], cache) -> (logits [B, V], cache)
    cache_bytes(cache) -> dict(allocated=int, used=int)
    inference_flops_per_token(ctx_len) -> float          (forward-only, analytic)
These are the optional decode hooks sprint1/scripts/infer_bench.py looks for.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SOFTCAP = 30.0


def rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, pos: torch.Tensor):
        f = pos.to(torch.float32)[:, None] * self.inv_freq[None, :]
        return f.cos(), f.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [B, H, T, hd]; cos/sin [T, hd/2]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    c, s = cos[None, None], sin[None, None]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, d: int, n_head: int):
        super().__init__()
        assert d % n_head == 0
        self.d, self.h, self.hd = d, n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.rotary = Rotary(self.hd)

    def heads(self, x: torch.Tensor, pos: torch.Tensor):
        """x [B, T, d] (already normalised) -> q, k, v [B, H, T, hd] with QK-norm and RoPE."""
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rms(q), rms(k)
        cos, sin = self.rotary(pos)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin), v

    def out(self, y: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = y.shape
        return self.proj(y.transpose(1, 2).reshape(B, T, H * hd))

    def forward(self, x, pos, mask=None):
        q, k, v = self.heads(x, pos)
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if mask.dtype != torch.bool:
                mask = mask.to(q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y), (k, v)


class MLP(nn.Module):
    def __init__(self, d: int, ratio: float):
        super().__init__()
        hidden = int(ratio * d)
        self.fc = nn.Linear(d, hidden, bias=False)
        self.proj = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    """Pre-norm block. Optional per-call modulation vectors (1 + g) on the normalised inputs."""

    def __init__(self, d: int, n_head: int, mlp_ratio: float):
        super().__init__()
        self.attn = Attention(d, n_head)
        self.mlp = MLP(d, mlp_ratio)

    def forward(self, x, pos, mask=None, g_attn=None, g_mlp=None):
        a_in = rms(x) if g_attn is None else rms(x) * (1.0 + g_attn).to(x.dtype)
        a, kv = self.attn(a_in, pos, mask)
        x = x + a
        m_in = rms(x) if g_mlp is None else rms(x) * (1.0 + g_mlp).to(x.dtype)
        return x + self.mlp(m_in), kv

    def step(self, x, pos, kv_fn, g_attn=None, g_mlp=None):
        """Incremental single-position forward. x [b, 1, d]. kv_fn(k, v, q) appends the new k/v to the
        caller's cache and returns attention output y [b, H, 1, hd]."""
        a_in = rms(x) if g_attn is None else rms(x) * (1.0 + g_attn).to(x.dtype)
        q, k, v = self.attn.heads(a_in, pos)
        y = kv_fn(k, v, q)
        x = x + self.attn.out(y)
        m_in = rms(x) if g_mlp is None else rms(x) * (1.0 + g_mlp).to(x.dtype)
        return x + self.mlp(m_in)


class RaggedKV:
    """Per-row variable-length KV cache, stored compactly as [B, H, cap, hd] with per-row lengths.
    Only rows named in `rows` are written; capacity grows geometrically. Memory reported honestly as
    allocated (cap * B) and used (sum of row lengths)."""

    def __init__(self, B, H, hd, cap, device, dtype):
        cap = max(int(cap), 1)
        self.k = torch.zeros(B, H, cap, hd, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.n = torch.zeros(B, dtype=torch.long, device=device)

    def _ensure(self, need: int):
        cap = self.k.shape[2]
        if need <= cap:
            return
        new = max(need, int(cap * 1.5) + 1)
        pad = new - cap
        z = torch.zeros(*self.k.shape[:2], pad, self.k.shape[3], device=self.k.device, dtype=self.k.dtype)
        self.k = torch.cat([self.k, z], 2)
        self.v = torch.cat([self.v, z.clone()], 2)

    def append(self, k, v, rows=None):
        """k, v [b, H, t, hd] for the given rows (LongTensor [b]) or all rows. Writes t positions/row."""
        if rows is None:
            rows = torch.arange(self.k.shape[0], device=self.k.device)
        t = k.shape[2]
        if rows.numel() == 0:
            return
        start = self.n[rows]
        self._ensure(int(start.max().item()) + t)
        idx = start[:, None] + torch.arange(t, device=k.device)[None, :]  # [b, t]
        self.k[rows[:, None], :, idx] = k.permute(0, 2, 1, 3).to(self.k.dtype)
        self.v[rows[:, None], :, idx] = v.permute(0, 2, 1, 3).to(self.v.dtype)
        self.n[rows] += t

    def append_masked(self, k, v, keep):
        """Prefill helper: k, v [B, H, T, hd]; keep [B, T] bool. Appends kept positions in order."""
        B, H, T, hd = k.shape
        counts = keep.sum(1)
        self._ensure(int((self.n + counts).max().item()) if B else 1)
        order = torch.cumsum(keep.long(), 1) - 1 + self.n[:, None]  # destination slot per kept pos
        b_idx, t_idx = keep.nonzero(as_tuple=True)
        dst = order[b_idx, t_idx]
        self.k[b_idx, :, dst] = k[b_idx, :, t_idx].to(self.k.dtype)
        self.v[b_idx, :, dst] = v[b_idx, :, t_idx].to(self.v.dtype)
        self.n += counts

    def view(self, rows=None):
        L = int(self.n.max().item()) if self.n.numel() else 0
        L = max(L, 1)
        if rows is not None and rows.numel() == self.k.shape[0]:
            rows = None  # all rows active (rows are always sorted aranges subsets): avoid a cache copy
        if rows is None:
            k, v, n = self.k[:, :, :L], self.v[:, :, :L], self.n
        else:
            k, v, n = self.k[rows, :, :L], self.v[rows, :, :L], self.n[rows]
        valid = torch.arange(L, device=n.device)[None, :] < n[:, None]
        return k, v, valid

    def bytes(self):
        el = self.k.element_size()
        per_pos = self.k.shape[1] * self.k.shape[3] * 2 * el
        return dict(allocated=self.k.numel() * 2 * el, used=int(self.n.sum().item()) * per_pos)


def attend(q, k, v, valid):
    """q [b, H, 1, hd], k/v [b, H, S, hd], valid [b, S] bool -> y [b, H, 1, hd]."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])


def attend_parts(q, parts):
    """Attention of one query over several KV segments without concatenating K/V (no cache copy):
    softmax over concatenated scores. parts: list of (K [b,H,S,hd], V, valid [b,S])."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores, vs = [], []
    for K, V, valid in parts:
        sc = torch.matmul(q, K.transpose(-1, -2)).float() * scale
        scores.append(sc.masked_fill(~valid[:, None, None, :], float("-inf")))
        vs.append(V)
    p = torch.softmax(torch.cat(scores, -1), -1).to(q.dtype)
    out, off = 0, 0
    for V in vs:
        S = V.shape[2]
        out = out + torch.matmul(p[..., off:off + S], V)
        off += S
    return out


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)


def lm_loss_eval(logits: torch.Tensor, targets: torch.Tensor):
    """Eval-mode loss plus per-token CE [B, T] (same aux key and semantics as baseline.py)."""
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.float().view(B * T, V), targets.reshape(B * T), ignore_index=-1,
                         reduction="none").view(B, T)
    n = (targets >= 0).sum().clamp_min(1)
    return ce.sum() / n, ce.detach()


def softcap(logits: torch.Tensor) -> torch.Tensor:
    return SOFTCAP * torch.tanh(logits.float() / SOFTCAP)


def split_param_groups(model: nn.Module, adam_names=("embed", "lm_head")) -> dict:
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(name.startswith(a) for a in adam_names) and "router" not in name \
                and "gate" not in name and "mod_" not in name:
            muon.append(p)
        else:
            adam.append(p)
    return {"muon": muon, "adam": adam}


def block_matmul_params(d: int, mlp_ratio: float) -> int:
    return 4 * d * d + 2 * d * int(mlp_ratio * d)


def sum_bytes(caches) -> dict:
    tot = dict(allocated=0, used=0)
    for c in caches:
        b = c.bytes()
        tot["allocated"] += b["allocated"]
        tot["used"] += b["used"]
    return tot


def hash_uniform(idx: torch.Tensor, pos: torch.Tensor, salt: int) -> torch.Tensor:
    """Deterministic content+position hash in [0, 1): a causal, parameter-free random score used by
    null controls (depends only on the token's own id, absolute position and a salt). Nonlinear
    32-bit integer mixing so different salts give independent-looking draws."""
    M = 0xFFFFFFFF
    x = (idx.long() * 0x9E3779B1 + pos.long()[None, :] * 0x85EBCA77 + (salt * 0xC2B2AE3D + 0x27D4EB2F)) & M
    for _ in range(3):
        x = x ^ (x >> 16)
        x = (x * 0x45D9F3B) & M
    x = x ^ (x >> 16)
    return x.to(torch.float64).div(4294967296.0).to(torch.float32)


__all__ = [
    "rms", "Rotary", "apply_rotary", "Attention", "MLP", "Block", "RaggedKV", "attend", "attend_parts", "lm_loss",
    "lm_loss_eval",
    "softcap", "split_param_groups", "block_matmul_params", "sum_bytes", "hash_uniform", "math",
]
