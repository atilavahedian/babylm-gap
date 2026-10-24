"""Logit distillation from a fixed teacher checkpoint (babylm.harness.train --kd_teacher). Off by default; off leaves the
port path unchanged.

Used by the moonshot's "ceiling map" diagnostics (docs/babylm/moonshot/ceiling_map.md), never as an attack by itself:
  C1  oracle targets: the teacher is a model trained on 100M words (illegal under R0; a diagnostic of how much of the
      10M->100M gap the 10M *contexts* can carry when the *targets* are those of a 100M learner);
  C1b self targets: the teacher is a 10M-trained model of the student's own recipe (separates the regularising effect
      of soft targets from the information they carry).
Known idea: knowledge distillation (Hinton et al. 2015), full-vocabulary token-level KD for LMs (Kim & Rush 2016).

Loss per micro-batch over the scored targets (y >= 0) of the causal rows:
    (1 - alpha) * CE(y, q) + alpha * tau^2 * KL(p_tau || q_tau)
with p = softmax(teacher logits / tau) and q = softmax(student logits / tau) over the full vocabulary (the teacher's
vocabulary is a prefix of the student's when the student pads it for MNTP: teacher mass on pad ids is 0). No top-k
cache (Anshumann et al. 2025: cached top-k targets are biased); the teacher runs on the same input, in eval mode,
no grad, under the run's autocast. From --kd_start_epoch on; before it the loss is plain CE.
Logged per step: kd_ce, kd_kl, kd_t_label (teacher mass on the true next token), kd_t_ent (teacher entropy).

Several teachers (--kd_teacher A,B,...; card F1 stage 1, a "T1 diagnostic, not BabyLM-legal"): the teacher
distribution is their probability average, p_bar = (1/K) sum_k softmax(z_k), passed on as log p_bar (tau then applies
to the mixture). --kd_perm (F1's null N-perm): each micro-batch's teacher distributions are permuted across its scored
and unscored positions by a seeded generator, so the student gets the same K teacher forwards with the per-position
information removed.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("logit distillation from a fixed teacher (babylm.moonshot.kd; diagnostics)")
    g.add_argument("--kd_teacher", default=None, help="teacher checkpoint (chck_*.pt with model/cfg/model_name), or "
                   "several comma-separated (their probability average is the teacher)")
    g.add_argument("--kd_perm", action="store_true", help="null: permute the teacher distributions across positions")
    g.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the KL term; CE gets 1 - alpha")
    g.add_argument("--kd_temp", type=float, default=1.0)
    g.add_argument("--kd_start_epoch", type=float, default=0.0)


class TeacherEnsemble(nn.Module):
    """Probability average of several teachers; forward(x) = log p_bar [B, T, V] (fp32)."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = None
        for t in self.members:
            lp = F.log_softmax(t.logits(t.hidden(x)).float(), dim=-1)
            acc = lp if acc is None else torch.logaddexp(acc, lp)
        return acc - math.log(len(self.members))


def teacher_logits(t: nn.Module, x: torch.Tensor, perm_gen: torch.Generator | None = None) -> torch.Tensor:
    z = t(x) if isinstance(t, TeacherEnsemble) else t.logits(t.hidden(x))
    if perm_gen is not None:
        B, T, V = z.shape
        idx = torch.randperm(B * T, generator=perm_gen).to(z.device)
        z = z.reshape(B * T, V)[idx].view(B, T, V)
    return z


def load_teacher(path: str | Path, device: torch.device, load_model_module) -> tuple[nn.Module, dict[str, Any]]:
    paths = [q for q in str(path).split(",") if q]
    if len(paths) > 1:
        ts = [load_teacher(q, device, load_model_module) for q in paths]
        shas = {m.get("tokenizer_sha256") for _, m in ts} - {None}
        vocab = {m["vocab_size"] for _, m in ts}
        if len(shas) > 1 or len(vocab) > 1:
            raise SystemExit("--kd_teacher: the teachers differ in tokenizer or vocabulary")
        return TeacherEnsemble([t for t, _ in ts]).eval(), {
            "path": str(path), "k": len(ts), "members": [m for _, m in ts], "vocab_size": vocab.pop(),
            "words_seen": sum(m.get("words_seen") or 0 for _, m in ts), "tokenizer_sha256": shas.pop() if shas else None}
    from sprint1.harness.model_api import ModelConfig
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    mod = load_model_module(ck.get("model_name", "baseline"))
    t = mod.build(cfg)
    missing, unexpected = t.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"--kd_teacher {path}: state dict mismatch (missing {missing[:5]}, unexpected {unexpected[:5]})")
    t.to(device).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    if not (hasattr(t, "hidden") and hasattr(t, "logits")):
        raise SystemExit("--kd_teacher: the teacher model has no hidden()/logits() split API")
    meta = {"path": str(path), "words_seen": ck.get("words_seen"), "label": ck.get("label"),
            "model_name": ck.get("model_name"), "vocab_size": cfg.vocab_size, "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd, "git_commit": ck.get("git_commit"),
            "tokenizer_sha256": ck.get("tokenizer_sha256")}
    return t, meta


class StudentLogits(nn.Module):
    """logits(hidden(x)) of a split-API model, as one module so it can be torch.compile'd."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.logits(self.m.hidden(x))


def kd_terms(zs: torch.Tensor, zt: torch.Tensor, y: torch.Tensor, alpha: float, tau: float
             ) -> tuple[torch.Tensor, dict[str, float]]:
    """zs [B, T, Vs] student logits, zt [B, T, Vt] teacher logits (Vt <= Vs), y [B, T] (-1 = unscored).
    Returns the mean loss over scored targets and detached stats."""
    zs = zs.float()
    zt = zt.float()
    B, T, Vs = zs.shape
    Vt = zt.shape[-1]
    if Vt > Vs:
        raise ValueError(f"teacher vocab {Vt} > student vocab {Vs}")
    m = y >= 0
    n = m.sum().clamp_min(1)
    ce = F.cross_entropy(zs.view(B * T, Vs), y.reshape(B * T), ignore_index=-1, reduction="sum") / n
    if alpha <= 0:
        return ce, {"kd_ce": float(ce.detach()), "kd_kl": 0.0}
    zs_m, zt_m = zs[m], zt[m]                       # [n, V]
    lq = F.log_softmax(zs_m / tau, dim=-1)[:, :Vt]  # student log-probs on the teacher's ids (full normaliser)
    lp = F.log_softmax(zt_m / tau, dim=-1)
    p = lp.exp()
    kl = (p * (lp - lq)).sum(-1).mean()
    loss = (1.0 - alpha) * ce + alpha * (tau ** 2) * kl
    with torch.no_grad():
        lpt1 = F.log_softmax(zt_m, dim=-1)
        t_label = lpt1.gather(1, y[m].clamp_max(Vt - 1).unsqueeze(1)).exp().mean()
        t_ent = -(lpt1.exp() * lpt1).sum(-1).mean()
    return loss, {"kd_ce": float(ce.detach()), "kd_kl": float(kl.detach()), "kd_t_label": float(t_label),
                  "kd_t_ent": float(t_ent)}
