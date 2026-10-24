"""M2 candidate: the frozen baseline network trained with a linear (row-factored) second moment on its sparse
tables (LM head, token embedding, value embeddings). Learning rule only; the scored network is baseline.py unchanged.

Theory and predictions: sprint1/math/m2/THEORY.md (card: sprint1/ideas/M2_rowadam.md). Optimizer:
sprint1/harness/optim_m2.py. The harness calls make_optimizer(model, cfg, base_lrs) with the frozen lrs; the
harness forwards muon_momentum, muon_wd, adam_betas, adam_eps, adam_wd into cfg.extra.

cfg.extra knobs (defaults in brackets):
  m2_mode   ["factored"]  factored | row | adam   ("adam" = per-coordinate reference through the same code)
  m2_groups ["head,embed"] which Adam groups get the m2 second moment (embed = token + value-embedding tables)
  m2_null   ["none"]      none | global  (global: one second moment per table, the per-row information destroyed)
  m2_beta2  [None]        beta2 for the m2 groups (None = the harness's --adam_beta2)
All other knobs are baseline.py's.
"""
from __future__ import annotations

from sprint1.harness.model_api import ModelConfig
from sprint1.harness.optim_m2 import build_m2_optimizer
from sprint1.models import baseline


def build(cfg: ModelConfig):
    return baseline.build(cfg)


def make_optimizer(model, cfg: ModelConfig, base_lrs: dict):
    ex = cfg.extra
    groups = ex.get("m2_groups", "head,embed")
    if isinstance(groups, str):
        groups = tuple(x.strip() for x in groups.split(",") if x.strip())
    null = ex.get("m2_null", "none")
    if null is True:
        null = "global"
    elif null in (False, None):
        null = "none"
    b2 = ex.get("m2_beta2", None)
    return build_m2_optimizer(
        model, base_lrs, m2_mode=str(ex.get("m2_mode", "factored")), m2_groups=groups, m2_null=str(null),
        m2_beta2=None if b2 is None else float(b2),
        muon_momentum=float(ex.get("muon_momentum", 0.95)), muon_wd=float(ex.get("muon_wd", 0.0)),
        adam_betas=tuple(ex.get("adam_betas", (0.8, 0.95))), adam_eps=float(ex.get("adam_eps", 1e-10)),
        adam_wd=float(ex.get("adam_wd", 0.0)))
