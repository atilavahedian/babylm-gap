"""sprint1.models.baseline with dropout, for the BabyLM regularisation arms.

cfg.extra["dropout"] = p (default 0): dropout on the RMS-normalised token embedding (before it becomes both the
residual stream x and the x0 injected into every block), and residual dropout on every attention output and every
MLP output before it is added to the stream. Value embeddings, U-net skips and the head are untouched.
With p = 0 the module has exactly the frozen model's parameters, initialisation (same RNG consumption) and outputs
(tested in babylm/tests/test_model_reg.py). Everything else (optimizer groups, flops, per-token loss) is inherited.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.baseline import GPT, Block, norm


class RegBlock(Block):
    def __init__(self, cfg: ModelConfig, hidden: int, p: float):
        super().__init__(cfg, hidden)
        self.p = p

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, x0: torch.Tensor) -> torch.Tensor:
        if self.x0_inject:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + F.dropout(self.attn(norm(x), ve), self.p, self.training)
        x = x + F.dropout(self.mlp(norm(x)), self.p, self.training)
        return x


class RegGPT(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.p = float(cfg.extra.get("dropout", 0.0))
        if not 0.0 <= self.p < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.p}")
        if self.p > 0:  # swap the blocks' class in place: same parameters, same init, same state_dict keys
            for b in self.blocks:
                b.__class__ = RegBlock
                b.p = self.p

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        if self.p == 0:
            return super().hidden(idx)
        L = self.cfg.n_layer
        e = self.embed(idx)
        x = x0 = F.dropout(norm(e) if self.embnorm else e, self.p, self.training)
        ves = [emb(idx) for emb in self.value_embeds]
        ve_at: list[torch.Tensor | None] = [None] * L
        for i, ve in enumerate(ves):
            ve_at[i] = ve
            ve_at[L - self.n_ve + i] = ve
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            if self.unet and i >= L - self.n_enc and skips:
                x = x + self.skip_weights[i - (L - self.n_enc)] * skips.pop()
            x = block(x, ve_at[i], x0)
            if self.unet and i < self.n_enc:
                skips.append(x)
        return norm(x)


def build(cfg: ModelConfig) -> nn.Module:
    return RegGPT(cfg)
