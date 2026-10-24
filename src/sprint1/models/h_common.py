"""Shared building blocks for Agent H's sprint-1 models (sprint1/models/h_*.py).

Self-contained (no import of baseline.py or other agents' files, which are written in parallel): a
modded-nanogpt-style pre-norm decoder (parameter-free RMSNorm, RoPE, QK-norm, ReLU^2 MLP, zero-initialised
output projections, untied head, logit soft-cap). The attention primitive takes explicit per-query
positions and an optional boolean mask so the same block serves
  * the causal main stream (flash path, is_causal=True),
  * extra query streams that read the main stream's K/V under a custom mask (lookahead slots), and
  * cached decoding (verify blocks plus slot trees attending to a KV cache).
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
        """pos: LongTensor of any shape -> cos, sin of shape pos.shape + (hd/2,), float32."""
        f = pos.to(torch.float32)[..., None] * self.inv_freq
        return f.cos(), f.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, hd]; cos/sin [T, hd/2] or [B, T, hd/2]."""
    if cos.dim() == 2:
        c, s = cos[None, None], sin[None, None]
    else:
        c, s = cos[:, None], sin[:, None]
    wd = torch.float64 if x.dtype == torch.float64 else torch.float32
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].to(wd), x[..., h:].to(wd)
    c, s = c.to(wd), s.to(wd)
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, d: int, n_head: int):
        super().__init__()
        assert d % n_head == 0
        self.d, self.h, self.hd = d, n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)

    def heads(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """x [B, T, d] (already normalised) -> q, k, v [B, H, T, hd] with QK-norm and RoPE."""
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rms(q), rms(k)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin), v

    def out(self, y: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = y.shape
        return self.proj(y.transpose(1, 2).reshape(B, T, H * hd))


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
    def __init__(self, d: int, n_head: int, mlp_ratio: float):
        super().__init__()
        self.attn = Attention(d, n_head)
        self.mlp = MLP(d, mlp_ratio)

    def causal(self, x, cos, sin, extra_in: torch.Tensor | None = None):
        """Standard causal block on the main stream. Returns (x_out, k, v) so other query streams can read
        this layer's keys/values. extra_in (optional, [B,T,d]) is added to the residual before the block."""
        if extra_in is not None:
            x = x + extra_in
        q, k, v = self.attn.heads(rms(x), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn.out(y)
        return x + self.mlp(rms(x)), k, v

    def masked(self, x, cos, sin, k_ctx, v_ctx, mask):
        """Block for a query stream x [B,Q,d] attending to [k_ctx; own keys] under boolean mask
        [B or 1, 1, Q, S_ctx + Q] (True = may attend). Returns (x_out, k_own, v_own)."""
        q, k, v = self.attn.heads(rms(x), cos, sin)
        if k_ctx is not None:
            K = torch.cat([k_ctx.to(k.dtype), k], 2)
            V = torch.cat([v_ctx.to(v.dtype), v], 2)
        else:
            K, V = k, v
        y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        x = x + self.attn.out(y)
        return x + self.mlp(rms(x)), k, v


def softcap(logits: torch.Tensor) -> torch.Tensor:
    return SOFTCAP * torch.tanh(logits.float() / SOFTCAP)


def ce(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1,
                           reduction=reduction)


def split_param_groups(model: nn.Module, adam_prefixes=("embed", "lm_head")) -> dict:
    """muon: 2-D hidden matrices inside the blocks; adam: embeddings, head, small heads, vectors."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and name.startswith("blocks."):
            muon.append(p)
        else:
            adam.append(p)
    return {"muon": muon, "adam": adam}


def block_matmul_params(d: int, mlp_ratio: float) -> int:
    return 4 * d * d + 2 * d * int(mlp_ratio * d)


class KVCache:
    """Per-layer KV cache for batch decoding with a common length (rows advance together)."""

    def __init__(self, n_layer, B, H, hd, cap, device, dtype):
        self.k = [torch.zeros(B, H, cap, hd, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(B, H, cap, hd, device=device, dtype=dtype) for _ in range(n_layer)]
        self.n = 0

    def ensure(self, need: int):
        cap = self.k[0].shape[2]
        if need <= cap:
            return
        new = max(need, int(cap * 1.5) + 1)
        for lst in (self.k, self.v):
            for i, t in enumerate(lst):
                z = torch.zeros(*t.shape[:2], new - cap, t.shape[3], device=t.device, dtype=t.dtype)
                lst[i] = torch.cat([t, z], 2)

    def get(self, layer: int):
        return self.k[layer][:, :, :self.n], self.v[layer][:, :, :self.n]

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int):
        t = k.shape[2]
        self.k[layer][:, :, start:start + t] = k.to(self.k[layer].dtype)
        self.v[layer][:, :, start:start + t] = v.to(self.v[layer].dtype)

    def bytes(self) -> dict:
        el = self.k[0].element_size()
        per_pos = sum(t.shape[0] * t.shape[1] * t.shape[3] for t in self.k) * 2 * el
        alloc = sum(t.numel() for t in self.k) * 2 * el
        return dict(allocated=alloc, used=self.n * per_pos)


def hash_uniform_int(x: torch.Tensor, salt: int, modulo: int) -> torch.Tensor:
    """Deterministic parameter-free integer hash of an int tensor, reduced mod `modulo`."""
    M = 0xFFFFFFFF
    h = (x.long() * 0x9E3779B1 + (salt * 0xC2B2AE3D + 0x27D4EB2F)) & M
    for _ in range(3):
        h = h ^ (h >> 16)
        h = (h * 0x45D9F3B) & M
    h = h ^ (h >> 16)
    return h % modulo


__all__ = ["rms", "Rotary", "apply_rotary", "Attention", "MLP", "Block", "softcap", "ce", "split_param_groups",
           "block_matmul_params", "KVCache", "hash_uniform_int", "math", "SOFTCAP"]
