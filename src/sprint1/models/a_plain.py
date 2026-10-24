"""Agent A's plain reference: the a_common MiniGPT body alone (full MLP, no mechanism).

Use it as the same-body baseline for a_answer_memory / a_answer_delta when sprint1/models/baseline.py does
not expose the split API (hidden(idx), logits(h), in_embed(ids)) and the A models therefore fall back to
the MiniGPT body.  When the baseline does expose it, the A models wrap the real baseline and
sprint1/models/baseline.py is the reference instead.  No knobs.
"""
from __future__ import annotations

import torch.nn as nn

from sprint1.harness.model_api import ModelConfig
from sprint1.models.a_common import MiniGPT


def build(cfg: ModelConfig) -> nn.Module:
    return MiniGPT(cfg, int(round(cfg.mlp_ratio * cfg.n_embd)))
