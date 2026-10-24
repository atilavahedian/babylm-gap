"""C_streammuon: residual-stream-coupled orthogonalisation (agent C). Idea card: sprint1/ideas/C_streammuon.md

The scored network is the sprint baseline, unchanged. Muon orthogonalises each hidden matrix on its own.
Here all matrices that WRITE into the residual stream (attention and MLP output projections of every
layer) are orthogonalised jointly as one [d, sum h] matrix -- the first-order map from all layers' hidden
states to the change of the stream -- and all matrices that READ from the stream are orthogonalised
jointly as one [sum h, d] matrix. Total update energy is matched to per-matrix Muon, so the LR is the
baseline's; what changes is allocation across depth and decorrelation of layers that push the same
stream directions. Same Newton-Schulz FLOPs as per-matrix Muon.

cfg.extra knobs (defaults in brackets):
  stream_mode ["stream"]  "stream" (bet) | "writers" (writers only) | "transpose" (NULL, equal overhead:
                          same-shape matrices coupled along their unshared hidden side) | "alloc" (NULL:
                          per-matrix Orth, energy re-allocated by gradient-norm share) | "off"
  lt_mode ["off"]         composes with C_ltmuon when set
  body ["auto"]           as in c_ltmuon
Residual roles come from model.c_stream_roles() if defined, else optim_c.infer_stream_roles (names,
then shapes); the role map is printed by sprint1/tests/test_c_streammuon.py for audit.
"""
from __future__ import annotations

import torch

from sprint1.harness.model_api import ModelConfig
from sprint1.harness.optim_c import build_c_optimizer
from sprint1.models.c_common import attach_hooks, build_body


def make_optimizer(model: torch.nn.Module, cfg: ModelConfig, base_lrs=None) -> torch.optim.Optimizer:
    ex = cfg.extra or {}
    return build_c_optimizer(model, cfg, base_lrs, lt_mode=ex.get("lt_mode", "off"),
                             stream_mode=ex.get("stream_mode", "stream"))


def build(cfg: ModelConfig) -> torch.nn.Module:
    return attach_hooks(build_body(cfg), make_optimizer)
