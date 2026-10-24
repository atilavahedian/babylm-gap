"""Sprint-1 baseline with a switchable attention mask, for the MNTP hybrid (babylm.harness.levers (a)).

The network, parameters, init and state_dict keys are the vendored baseline's, unchanged: build() makes the baseline
and swaps each attention module's class for SwitchableAttention, whose forward is the baseline forward with
`is_causal=self.is_causal` (True unless levers.BidirectionalAttention flips it; levers.make_switchable swaps the class in any model). Causal outputs are bit-identical to
the baseline (tested), so HF export and the official causal evaluation are untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sprint1.models import baseline as B
from sprint1.harness.model_api import ModelConfig


class SwitchableAttention(B.CausalSelfAttention):
    is_causal = True

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None) -> torch.Tensor:
        Bsz, T, d = x.shape
        H, D = self.n_head, self.head_dim
        q = self.c_q(x).view(Bsz, T, H, D)
        k = self.c_k(x).view(Bsz, T, H, D)
        v = self.c_v(x).view(Bsz, T, H, D)
        dt = q.dtype
        q, k = self.rotary(B.norm(q)).to(dt), self.rotary(B.norm(k)).to(dt)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.to(dt).transpose(1, 2),
                                           is_causal=self.is_causal, scale=self.scale)
        return self.c_proj(y.transpose(1, 2).reshape(Bsz, T, d))


def build(cfg: ModelConfig):
    m = B.build(cfg)
    for blk in m.blocks:
        blk.attn.__class__ = SwitchableAttention
    return m
