"""C_ltmuon: log-time memory Muon (agent C, learning-rule territory). Idea card: sprint1/ideas/C_ltmuon.md

The scored network is the sprint baseline, unchanged. Only the learning rule changes: every parameter
gets a second, slow momentum buffer whose memory horizon grows linearly with training time
(beta3(t) = 1 - delta/(delta + t)), mixed into the step direction with weight alpha(t) = (1+t)^(1-kappa).
Hidden matrices: update = Orth(nesterov_fast + alpha(t) * slow)  (Muon geometry);
embeddings / head / scalars: AdamW with the same slow buffer (AdEMAMix-style mix, magnitude-normalised).

cfg.extra knobs (defaults in brackets):
  lt_mode ["logtime"]   "logtime" (the bet) | "frozen" (NULL: beta3 frozen at the fast momentum, same
                        buffers / FLOPs / alpha schedule) | "fixed" (AdEMAMix-style constant beta3) | "off"
  lt_delta [8.0]  lt_kappa [0.85]  lt_alpha0 [0.25, toy-amended from ADana's 1.0]  lt_alpha_max [None]  lt_fixed_beta3 [0.999]
  lt_beta3_max [0.9999]  lt_adam [True] (apply the slow buffer to the adam groups too)
  stream_mode ["off"]   composes with C_streammuon when set
  body ["auto"]         "baseline" (require sprint1/models/baseline.py) | "fallback" | "auto"

Harness call: opt = model.make_optimizer(model, cfg, base_lrs)  (or the module-level make_optimizer),
returning ONE torch.optim.Optimizer whose groups carry "initial_lr" and "kind" in {"muon","adam"}.
"""
from __future__ import annotations

import copy

import torch

from sprint1.harness.model_api import ModelConfig
from sprint1.harness.optim_c import build_c_optimizer
from sprint1.models.c_common import attach_hooks, build_body


# Toy-informed amendment (21:25 PDT 5 Oct, before any GPU result): ADana's alpha0 = 1 lost to plain Muon on
# both CPU-toy seeds with a gap that grew from 250 to 1000 steps (toy kill); alpha0 = 0.25 was neutral.
# The screened default is therefore 0.25 for the bet AND its frozen null; pass lt_alpha0=1 to reproduce.
DEFAULT_LT_ALPHA0 = 0.25


def make_optimizer(model: torch.nn.Module, cfg: ModelConfig, base_lrs=None) -> torch.optim.Optimizer:
    ex = dict(cfg.extra or {})
    ex.setdefault("lt_alpha0", DEFAULT_LT_ALPHA0)
    cfg2 = copy.copy(cfg)
    cfg2.extra = ex
    return build_c_optimizer(model, cfg2, base_lrs, lt_mode=ex.get("lt_mode", "logtime"),
                             stream_mode=ex.get("stream_mode", "off"))


def build(cfg: ModelConfig) -> torch.nn.Module:
    return attach_hooks(build_body(cfg), make_optimizer)
