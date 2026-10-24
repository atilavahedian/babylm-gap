"""Shared backbone for Agent I (neuroscience territory) models: sprint1/models/i_*.py.

IGPT subclasses the sprint baseline (sprint1/models/baseline.py, GPT) and changes exactly one thing: the
feed-forward module of every block is replaced by `ffn_factory(layer_idx)`. Embeddings, value embeddings,
attention, U-net skips, x0 injection, softcap, init and parameter names are the baseline's, so the baseline
runs ARE the equal-code dense reference arm (every i_* model also accepts --extra ffn=dense, which returns the
baseline itself).

FFN module protocol (duck-typed):
    forward(x, ctx) -> y          x: [B, T, d] normed block input; ctx: dict, ctx["idx"] = token ids when the
                                  model was built with needs_idx=True (only nulls that hash token ids need it)
    fwd_macs_per_token() -> float forward multiply-accumulates per token (FLOPs = 6 * MACs, fwd+bwd)
    stats() -> dict[str, Tensor]  0-d detached tensors for logging (optional; added to aux in training)
Parameter groups: every block parameter with ndim >= 2 goes to Muon (the harness Muon batches leading dims;
weights are stored [.., out, in] like nn.Linear), everything else to the baseline's Adam groups. Avoid 2-D
Adam params: the harness puts them in the EMBEDDING bucket (lr 0.6 at the reference batch).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline as _base

INIT_GAIN = math.sqrt(3) * 0.5  # baseline CastedLinear: U(-INIT_GAIN/sqrt(fan_in), +INIT_GAIN/sqrt(fan_in))
CastedLinear = _base.CastedLinear
BaselineMLP = _base.MLP


def init_bound(fan_in: int) -> float:
    return INIT_GAIN / math.sqrt(fan_in)


class FFNAdapter(nn.Module):
    """Adapts an i_* FFN (forward(x, ctx)) to the baseline Block call self.mlp(norm(x))."""

    def __init__(self, inner: nn.Module, ctx: dict):
        super().__init__()
        self.inner = inner
        self._ctx = ctx  # plain dict shared with the model (not a submodule)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x, self._ctx)


class IGPT(_base.GPT):
    def __init__(self, cfg: ModelConfig, ffn_factory, needs_idx: bool = False):
        super().__init__(cfg)
        self._ctx: dict = {}
        self._needs_idx = needs_idx
        for i, blk in enumerate(self.blocks):
            blk.mlp = FFNAdapter(ffn_factory(i), self._ctx)

    def ffns(self):
        return [blk.mlp.inner for blk in self.blocks]

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        if self._needs_idx:
            self._ctx["idx"] = idx
        return super().hidden(idx)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        loss, aux = super().forward(idx, targets)
        if self.training:
            aux = dict(aux)
            for i, f in enumerate(self.ffns()):
                if hasattr(f, "stats"):
                    for k, v in f.stats().items():
                        aux[f"l{i}_{k}"] = v
        return loss, aux

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for p in self.blocks.parameters() if p.ndim >= 2]
        ids = {id(p) for p in muon}
        return {"muon": muon, "adam": [p for p in self.parameters() if id(p) not in ids]}

    def flops_per_token(self) -> float:
        c = self.cfg
        macs = self.lm_head.weight.numel()
        for blk in self.blocks:
            macs += sum(p.numel() for p in blk.attn.parameters() if p.ndim == 2)
            macs += blk.mlp.inner.fwd_macs_per_token()
        return 6.0 * macs + 6.0 * c.n_layer * c.seq_len * c.n_embd

    def on_step_end(self, step: int, total_steps: int) -> None:
        for f in self.ffns():
            if hasattr(f, "on_step_end"):
                f.on_step_end(step, total_steps)


def build_baseline(cfg: ModelConfig) -> nn.Module:
    return _base.build(cfg)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
