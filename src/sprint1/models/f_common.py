"""Shared building blocks for invention agent F's sprint-1 models (sprint1/models/f_*.py). Owned by agent F.

Self-contained (does not import baseline.py, which is written in parallel): a modded-nanogpt-style
pre-norm GPT (parameter-free RMSNorm, RoPE in half-split pair layout, QK-norm, ReLU^2 MLP, zero-initialised
writer projections, untied embeddings/head, logit soft-cap 30). F models replace only the attention
sub-layer; everything else is shared here so candidate, null and reference arms differ only in the mechanism.

Conventions
- Pair layout: pair p of a head vector x[..., hd] is (x[p], x[p + hd/2]), p = 0..hd/2-1. RoPE frequencies
  decrease with p, so the LAST pairs are the lowest-frequency ("semantic") pairs.
- Angles are float32 and rotations are applied in float32, then cast back (bf16-safe).
- No data-dependent Python control flow in forward: every loop runs a shape-determined number of times.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig

SOFTCAP = 30.0
# parameters that are tables of per-head scalars, not matrices: AdamW, never Muon (F models keep them 1-D)
ADAM_NAMES = ("K",)


def rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def rope_freqs(hd: int, base: float = 10000.0) -> torch.Tensor:
    """[hd/2] standard RoPE inverse frequencies (decreasing with pair index)."""
    return 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[p], x[p+hd/2]) by angle with given cos/sin [..., hd/2] (broadcastable to x[..., :hd/2])."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1).type_as(x)


class MLP(nn.Module):
    def __init__(self, d: int, ratio: float):
        super().__init__()
        hidden = int(ratio * d)
        self.fc = nn.Linear(d, hidden, bias=False)
        self.proj = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.fc(x)).square())


class FGPT(nn.Module):
    """Generic F body. `make_attn(layer_index)` returns the attention sub-module for that layer. The attention
    module is called as attn(h, x, ctx) -> (delta_x, ctx) where h = rms(x), x the raw residual, ctx a dict of
    per-forward state threaded through the layers (e.g. oscillator phases). The MLP sub-layer is standard."""

    def __init__(self, cfg: ModelConfig, make_attn):
        super().__init__()
        self.cfg = cfg
        d = cfg.n_embd
        assert d % cfg.n_head == 0
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.attns = nn.ModuleList([make_attn(i) for i in range(cfg.n_layer)])
        self.mlps = nn.ModuleList([MLP(d, cfg.mlp_ratio) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=1.0)
        nn.init.zeros_(self.lm_head.weight)
        for m in self.mlps:
            nn.init.normal_(m.fc.weight, std=0.5 / math.sqrt(d))

    # hooks for subclasses -------------------------------------------------------------------------------
    def init_ctx(self, idx: torch.Tensor, x0: torch.Tensor) -> dict:
        return {"idx": idx}

    def aux_from_ctx(self, ctx: dict) -> dict:
        return {}

    # forward ------------------------------------------------------------------------------------------
    def forward_logits(self, idx: torch.Tensor):
        x = rms(self.embed(idx))
        ctx = self.init_ctx(idx, x)
        for attn, mlp in zip(self.attns, self.mlps):
            dx, ctx = attn(rms(x), x, ctx)
            x = x + dx
            x = x + mlp(rms(x))
        logits = self.lm_head(rms(x)).float()
        return SOFTCAP * torch.tanh(logits / SOFTCAP), ctx

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits, ctx = self.forward_logits(idx)
        aux = self.aux_from_ctx(ctx)
        if targets is None:
            return logits, aux
        if self.training:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            return loss, aux
        pt = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1,
                             reduction="none").view(targets.shape)
        valid = targets != -1
        aux["per_token_loss"] = pt.detach()
        return pt.sum() / valid.sum().clamp_min(1), aux

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon, adam = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and not name.startswith(("embed", "lm_head")) and name.split(".")[-1] not in ADAM_NAMES:
                muon.append(p)
            else:
                adam.append(p)
        return {"muon": muon, "adam": adam}

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def matmul_params(self) -> int:
        """Parameters that take part in a per-token matmul (all 2-D weights except the embedding lookup)."""
        return sum(p.numel() for n, p in self.named_parameters() if p.ndim == 2 and not n.startswith("embed"))


def attn_core_flops(T: int, H: int, d_qk: int, d_v: int, keys_per_query: float) -> float:
    """Forward+backward FLOPs per token per layer of the score and value matmuls as EXECUTED:
    fwd = 2*keys*H*(d_qk + d_v); bwd = 2x fwd. `keys_per_query` is the mean number of keys the kernel
    actually multiplies (T/2 for a causal kernel; full key length for a dense masked kernel)."""
    return 3.0 * 2.0 * keys_per_query * H * (d_qk + d_v)


def hash_uniform(idx: torch.Tensor, salt: int) -> torch.Tensor:
    """Deterministic parameter-free hash of token ids -> [0, 1). Used by hash/null arms."""
    M = 0xFFFFFFFF
    x = (idx.long() * 0x9E3779B1 + (salt * 0xC2B2AE3D + 0x27D4EB2F)) & M
    for _ in range(3):
        x = x ^ (x >> 16)
        x = (x * 0x45D9F3B) & M
    x = x ^ (x >> 16)
    return x.to(torch.float64).div(4294967296.0).to(torch.float32)


__all__ = ["rms", "rope_freqs", "rotate", "MLP", "FGPT", "attn_core_flops", "hash_uniform", "SOFTCAP"]
