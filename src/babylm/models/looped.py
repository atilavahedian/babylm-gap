"""Sprint-1 baseline with looped depth, for data-wall rung 0 (critic ref-0368; design agent 3 memo, section 5).

cfg.extra["n_loops"] = k (default 1): the baseline's L blocks run k times with shared weights, so the effective depth
is Le = L * k and effective layer j runs blocks[j % L]. Everything the baseline lays out over its L layers is laid out
over Le instead: x0 injection (per block, shared), value embeddings (first and last n_ve effective layers) and U-net
skips (encoder = the first Le // 2 effective layers, one learned skip weight each). With k = 1 the module has exactly
the baseline's parameters, initialisation, state_dict and outputs (tested in babylm/tests/test_looped.py).

FLOPs count the shared block matrices once per pass (critic ref-0368), so 12L looped 2x and plain 24L are FLOPs
twins. A plain weight-shared unroll with no per-loop embeddings; a known idea (Chen+ 2026, Parcae, Ouro).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sprint1.harness.model_api import ModelConfig
from sprint1.models.baseline import GPT, norm


def n_loops_of(extra: dict) -> int:
    k = int(extra.get("n_loops", 1))
    if k < 1:
        raise ValueError(f"n_loops must be >= 1, got {k}")
    return k


class LoopedGPT(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.n_loops = n_loops_of(cfg.extra)
        if self.n_loops > 1:  # torch.ones draws no RNG: init of every other parameter is the baseline's
            self.n_enc = cfg.n_layer * self.n_loops // 2
            self.skip_weights = nn.Parameter(torch.ones(max(self.n_enc, 1)))

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        L = self.cfg.n_layer
        Le = L * self.n_loops
        e = self.embed(idx)
        x = x0 = norm(e) if self.embnorm else e
        ves = [emb(idx) for emb in self.value_embeds]
        ve_at: list[torch.Tensor | None] = [None] * Le
        for i, ve in enumerate(ves):
            ve_at[i] = ve
            ve_at[Le - self.n_ve + i] = ve
        skips: list[torch.Tensor] = []
        for j in range(Le):
            if self.unet and j >= Le - self.n_enc and skips:
                x = x + self.skip_weights[j - (Le - self.n_enc)] * skips.pop()
            x = self.blocks[j % L](x, ve_at[j], x0)
            if self.unet and j < self.n_enc:
                skips.append(x)
        return norm(x)

    def flops_per_token(self) -> float:
        c = self.cfg
        k = self.n_loops
        blocks = sum(p.numel() for p in self.blocks.parameters() if p.ndim == 2)
        attn = 6 * c.n_layer * k * c.seq_len * c.n_embd
        return 6.0 * (k * blocks + self.lm_head.weight.numel()) + attn


def build(cfg: ModelConfig) -> nn.Module:
    return LoopedGPT(cfg)
