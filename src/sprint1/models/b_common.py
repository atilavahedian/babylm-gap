"""Shared pieces for invention-agent-B candidates (training-objective territory).

B's candidates change the TRAINING OBJECTIVE, not the scored model: at eval time every B model is a
plain causal Transformer that returns plain next-token CE. So all B models share one body:

  * if cfg.extra["body"] == "baseline" (or "auto" and sprint1/models/baseline.py exposes the split
    API below), the baseline's own module is reused by import, never edited;
  * otherwise a minimal modded-nanogpt-style GPT defined here is used (RMSNorm, RoPE, QK-norm,
    ReLU^2 MLP, untied embeddings, zero-init output projections, optional logit softcap).

Split API used by B heads (any body must provide it):
    body.hidden(idx) -> h [B, T, d]   final normalised hidden state (input to the LM head)
    body.logits(h)   -> z [B, T, V]   float32 logits (softcap applied if any)
    body.out_weight() -> W [V, d]     output (unembedding) matrix, as used by logits()
    body.in_embed(ids) -> e [..., d]  input embedding lookup
    body.base_flops_per_token(T) -> float   fwd+bwd FLOPs/token of the plain body + head
"""
from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, H, T, hd]
        T = x.size(-2)
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, mlp_ratio: float):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        hidden = int(mlp_ratio * d)
        self.fc = nn.Linear(d, hidden, bias=False)
        self.fc_out = nn.Linear(hidden, d, bias=False)
        self.rotary = Rotary(d // n_head)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.fc_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(rms_norm(x)).split(C, dim=-1)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        q, k = self.rotary(rms_norm(q)), self.rotary(rms_norm(k))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + self.fc_out(F.relu(self.fc(rms_norm(x))).square())
        return x


class MiniGPT(nn.Module):
    """Minimal stand-in body; used only when the shared baseline is unavailable."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, V = cfg.n_embd, cfg.vocab_size
        self.wte = nn.Embedding(V, d)
        nn.init.normal_(self.wte.weight, std=0.5)
        self.blocks = nn.ModuleList(Block(d, cfg.n_head, cfg.mlp_ratio) for _ in range(cfg.n_layer))
        self.lm_head = nn.Linear(d, V, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.002)  # near-zero (modded-nanogpt zero-inits)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.softcap = float(cfg.extra.get("softcap", 15.0))

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        x = rms_norm(self.wte(idx))
        for blk in self.blocks:
            x = blk(x)
        return rms_norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        z = self.lm_head(h).float()
        if self.softcap > 0:
            z = self.softcap * torch.tanh(z / self.softcap)
        return z

    def out_weight(self) -> torch.Tensor:
        return self.lm_head.weight

    def in_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.wte(ids)

    def base_flops_per_token(self, T: int | None = None) -> float:
        cfg = self.cfg
        T = cfg.seq_len if T is None else T
        d, L, V = cfg.n_embd, cfg.n_layer, cfg.vocab_size
        hidden = int(cfg.mlp_ratio * d)
        matmul_params = L * (4 * d * d + 2 * d * hidden) + V * d  # blocks + lm head
        attn = L * T * d  # causal QK^T + AV MACs per token (triangle-averaged)
        return 6.0 * (matmul_params + attn)

    def body_param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for blk in self.blocks for p in blk.parameters() if p.ndim == 2]
        ids = {id(p) for p in muon}
        adam = [p for p in self.parameters() if id(p) not in ids]
        return {"muon": muon, "adam": adam}


class BaselineAdapter(nn.Module):
    """Wraps sprint1.models.baseline if it exposes the split API (hidden/logits/out_weight/in_embed)."""

    def __init__(self, inner: nn.Module, cfg: ModelConfig):
        super().__init__()
        self.inner = inner
        self.cfg = cfg

    def hidden(self, idx):
        return self.inner.hidden(idx)

    def logits(self, h):
        return self.inner.logits(h).float()

    def out_weight(self):
        return self.inner.out_weight()

    def in_embed(self, ids):
        return self.inner.in_embed(ids)

    def base_flops_per_token(self, T=None):
        if hasattr(self.inner, "flops_per_token"):
            return float(self.inner.flops_per_token())
        return MiniGPT.base_flops_per_token(self, T)  # type: ignore[arg-type]

    def body_param_groups(self):
        if hasattr(self.inner, "param_groups"):
            return self.inner.param_groups()
        muon = [p for n, p in self.inner.named_parameters()
                if p.ndim == 2 and "emb" not in n and "wte" not in n and "lm_head" not in n]
        ids = {id(p) for p in muon}
        return {"muon": muon, "adam": [p for p in self.inner.parameters() if id(p) not in ids]}


_SPLIT_API = ("hidden", "logits", "out_weight", "in_embed")


def build_body(cfg: ModelConfig) -> nn.Module:
    choice = cfg.extra.get("body", "auto")
    if choice in ("auto", "baseline"):
        try:
            mod = importlib.import_module("sprint1.models.baseline")
            inner = mod.build(cfg)
            if all(hasattr(inner, a) for a in _SPLIT_API):
                return BaselineAdapter(inner, cfg)
            if choice == "baseline":
                raise RuntimeError("baseline lacks split API " + str(_SPLIT_API))
        except (ImportError, AttributeError):
            if choice == "baseline":
                raise
    return MiniGPT(cfg)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def next_token_ce(z: torch.Tensor, targets: torch.Tensor):
    """Plain scored CE. Returns (mean loss over non-ignored targets, per-token CE [B,T], valid mask)."""
    B, T, V = z.shape
    ce = F.cross_entropy(z.reshape(B * T, V), targets.reshape(B * T), ignore_index=-1,
                         reduction="none").view(B, T)
    valid = targets >= 0
    return masked_mean(ce, valid), ce, valid


def shift_future(x: torch.Tensor, j: int) -> torch.Tensor:
    """out[:, t] = x[:, t + j] (zero padded at the end). x: [B, T, ...]."""
    if j == 0:
        return x
    pad = torch.zeros_like(x[:, :j])
    return torch.cat([x[:, j:], pad], dim=1)


def roll_batch(x: torch.Tensor) -> torch.Tensor:
    """Null-control source: the same tensor taken from a different sequence of the batch
    (or, with B == 1, from the other half of the same sequence)."""
    if x.size(0) > 1:
        return torch.roll(x, shifts=1, dims=0)
    return torch.roll(x, shifts=x.size(1) // 2, dims=1)
