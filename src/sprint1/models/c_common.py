"""Shared helpers for invention agent C model files (learning-rule territory). Owned by agent C.

C candidates do not change the scored network: each C model file returns the sprint baseline network
(sprint1/models/baseline.py, reused by import, never edited) and adds a `make_optimizer` hook that
builds sprint1/harness/optim_c.COptimizer. If the baseline cannot be imported (it is being written in
parallel) or cfg.extra["body"] == "fallback", a minimal modded-nanogpt-style GPT defined here is used
so tests and CPU smokes still run; the report states which body was used.
"""
from __future__ import annotations

import importlib
import math
import warnings
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig


# --------------------------------------------------------------------------------------------------
# Fallback body (only when the shared baseline is unavailable)
# --------------------------------------------------------------------------------------------------
def _rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def _rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:  # x [B, H, T, hd]
    T, hd = x.size(-2), x.size(-1)
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    f = torch.outer(torch.arange(T, device=x.device, dtype=torch.float32), inv)
    cos, sin = f.cos().to(x.dtype), f.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class _Block(nn.Module):
    def __init__(self, d: int, n_head: int, mlp_ratio: float):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)       # reader
        self.proj = nn.Linear(d, d, bias=False)          # writer
        h = int(mlp_ratio * d)
        self.fc = nn.Linear(d, h, bias=False)            # reader
        self.fc_out = nn.Linear(h, d, bias=False)        # writer
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.fc_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(_rms(x)).view(B, T, 3, self.n_head, hd).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = _rope(_rms(q)), _rope(_rms(k))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.fc_out(F.relu(self.fc(_rms(x))).square())


class CFallbackGPT(nn.Module):
    """RMSNorm, RoPE, QK-norm, ReLU^2 MLP, untied embeddings, zero-init writers, logit softcap 30."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.n_embd
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.blocks = nn.ModuleList([_Block(d, cfg.n_head, cfg.mlp_ratio) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=1.0)
        nn.init.zeros_(self.lm_head.weight)
        for b in self.blocks:
            for lin in (b.qkv, b.fc):
                nn.init.normal_(lin.weight, std=0.5 / math.sqrt(d))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _rms(self.embed(idx))
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(_rms(x)).float()
        logits = 30.0 * torch.tanh(logits / 30.0)
        if targets is None:
            return logits, {}
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return loss, {}

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for b in self.blocks for p in b.parameters()]
        return {"muon": muon, "adam": [self.embed.weight, self.lm_head.weight]}

    def c_stream_roles(self) -> dict[nn.Parameter, str]:
        roles = {}
        for b in self.blocks:
            roles[b.qkv.weight] = "reader"
            roles[b.fc.weight] = "reader"
            roles[b.proj.weight] = "writer"
            roles[b.fc_out.weight] = "writer"
        return roles

    def flops_per_token(self) -> float:
        c = self.cfg
        n_mat = sum(p.numel() for p in self.param_groups()["muon"]) + self.lm_head.weight.numel()
        attn = 12 * c.n_layer * c.seq_len * c.n_embd // 2  # causal: half the T x T scores
        return 6.0 * n_mat + attn


# --------------------------------------------------------------------------------------------------
# Body selection and hook attachment
# --------------------------------------------------------------------------------------------------
def build_body(cfg: ModelConfig) -> nn.Module:
    body = (cfg.extra or {}).get("body", "auto")
    if body != "fallback":
        try:
            baseline = importlib.import_module("sprint1.models.baseline")
            m = baseline.build(cfg)
            m._c_body = "baseline"
            return m
        except Exception as e:  # noqa: BLE001 - baseline is written in parallel; fall back loudly
            if body == "baseline":
                raise
            warnings.warn(f"agent C: baseline unavailable ({type(e).__name__}: {e}); using CFallbackGPT")
    m = CFallbackGPT(cfg)
    m._c_body = "fallback"
    return m


def attach_hooks(model: nn.Module, make_optimizer: Callable[..., torch.optim.Optimizer]) -> nn.Module:
    """Attach `make_optimizer` (callable as model.make_optimizer(model, cfg, base_lrs) or
    model.make_optimizer(cfg, base_lrs)) and chain `on_step_end` so the optimizer learns total_steps."""

    def _mk(*args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        args_l = list(args)
        m = args_l.pop(0) if args_l and isinstance(args_l[0], nn.Module) else kwargs.pop("model", model)
        cfg = args_l.pop(0) if args_l else kwargs.pop("cfg")
        base_lrs = args_l.pop(0) if args_l else kwargs.pop("base_lrs", None)
        opt = make_optimizer(m, cfg, base_lrs, **kwargs)
        model.__dict__["_c_optimizer"] = opt  # plain attribute, not registered as a submodule
        return opt

    model.__dict__["make_optimizer"] = _mk
    orig = getattr(model, "on_step_end", None)

    def _on_step_end(step: int, total_steps: int) -> None:
        opt = model.__dict__.get("_c_optimizer")
        if opt is not None:
            opt.total_steps = int(total_steps)
        if orig is not None:
            orig(step, total_steps)

    model.__dict__["on_step_end"] = _on_step_end
    return model
