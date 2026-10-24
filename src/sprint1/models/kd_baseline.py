"""Frozen baseline + distillation from a teacher distribution (ruler experiment 1 and the teacher ladder).

Loss backpropagated: (1 - alpha) * CE + alpha * T^2 * KL(t || s), via aux_loss = alpha*(T^2 KL - CE) added to the
scored CE. The scored loss (train and eval) is the student's plain CE; eval never touches the teacher.

cfg.extra:
  kd_teacher: 'gpt2' | 'gpt2-medium' (HF, frozen)  |  'ckpt:<path or ${BUCKET}' (a baseline.py checkpoint)
              | 'ngram:<npz>' (top-k table per previous token)  |  'uniform'  |  'unigram:<npy>'  |  'tiny_random' (tests)
  kd_topk:   0 = full teacher distribution; k > 0 = teacher top-k renormalized, KL on those ids (model teachers)
  kd_alpha (0.5), kd_temp (1.0)
Teachers live outside the module tree (no optimizer state, not checkpointed), loaded lazily on the training device.
"""
from __future__ import annotations
GCS_SCHEME = "gs:" + "//"  # Google Cloud Storage URI scheme

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sprint1.models.baseline as bl
from sprint1.harness.model_api import ModelConfig

# forward FLOPs per token, 2 x matmul params (+ attention at T=1024); table and smoothing teachers ~0
TEACHER_FWD_FLOPS = {"gpt2": 2 * 124.4e6 + 2 * 2 * 12 * 1024 * 768,
                     "gpt2-medium": 2 * 354.8e6 + 2 * 2 * 24 * 1024 * 1024}


def _local(path: str) -> str:
    if path.startswith(GCS_SCHEME):
        loc = Path("_teachers") / Path(path).name
        loc.parent.mkdir(exist_ok=True)
        if not loc.exists():
            subprocess.run(["gsutil", "-q", "cp", path, str(loc)], check=True)
        return str(loc)
    return path


def _check_build_meta(path: str):
    """Ladder leakage rule (b): count tables must come from train shards 000001-000002 only, with sha256 recorded."""
    import json, os, re
    meta = os.path.join(os.path.dirname(_local(path)), "ngram_build.json")
    if not os.path.exists(meta):
        meta = _local(os.path.join(os.path.dirname(path), "ngram_build.json"))
    rec = json.load(open(meta))
    names = rec["train_shards_used"]
    if not all(n.startswith("synthetic") for n in names):
        assert sorted(names) == ["fineweb_train_000001.bin", "fineweb_train_000002.bin"], names
        assert all(re.fullmatch(r"fineweb_train_00000[12]\.bin", n) for n in names), names
        assert set(rec["sha256"]) == set(names), rec.get("sha256")
    print(f"[kd] teacher table provenance ok: {names} {rec.get('sha256')}", flush=True)


class KDGPT(bl.GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        ex = cfg.extra
        self.kd_teacher = str(ex.get("kd_teacher", "gpt2"))
        self.kd_alpha = float(ex.get("kd_alpha", 0.5))
        self.kd_temp = float(ex.get("kd_temp", 1.0))
        self.kd_topk = int(ex.get("kd_topk", 0))
        self._t: dict = {}

    # ---- teachers -------------------------------------------------------------------------------
    def _load(self, device):
        if "kind" in self._t:
            return
        t = self.kd_teacher
        if t in ("gpt2", "gpt2-medium", "tiny_random"):
            from transformers import GPT2Config, GPT2LMHeadModel
            if t == "tiny_random":
                torch.manual_seed(0)
                m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=50257, n_positions=1024))
            else:
                try:
                    m = GPT2LMHeadModel.from_pretrained(t, attn_implementation="sdpa")
                except TypeError:
                    m = GPT2LMHeadModel.from_pretrained(t)
            m.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
            self._t.update(kind="hf", m=m)
        elif t.startswith("ckpt:"):
            ck = torch.load(_local(t[5:]), map_location="cpu", weights_only=False)
            m = bl.build(ModelConfig.from_dict(ck["cfg"]))
            m.load_state_dict(ck["model"])
            m.eval().requires_grad_(False).to(device)
            self._t.update(kind="ckpt", m=m)
        elif t.startswith("ngram:"):
            _check_build_meta(t[6:])
            z = np.load(_local(t[6:]))
            self._t.update(kind="ngram", ids=torch.as_tensor(z["top_ids"].astype(np.int64), device=device),
                           p=torch.as_tensor(z["top_p"].astype(np.float32), device=device))
        elif t == "uniform":
            self._t.update(kind="uniform")
        elif t.startswith("unigram:"):
            _check_build_meta(t[8:])
            u = torch.as_tensor(np.load(_local(t[8:])).astype(np.float64))
            u = (u / u.sum()).float()
            self._t.update(kind="unigram", u=u.to(device), negH=float((u[u > 0] * u[u > 0].log()).sum()))
        else:
            raise ValueError(f"unknown kd_teacher {t!r}")

    @torch.compiler.disable
    def _teacher_targets(self, idx: torch.Tensor):
        """Returns (ids or None, logp): ids [B,T,k] and log-probs over them, or None and full log-probs."""
        self._load(idx.device)
        k = self._t["kind"]
        with torch.no_grad():
            if k == "hf":
                zt = self._t["m"](input_ids=idx).logits
            elif k == "ckpt":
                m = self._t["m"]
                zt = m.logits(m.hidden(idx))
            elif k == "ngram":
                ids = self._t["ids"][idx]
                return ids, self._t["p"][idx].clamp_min(1e-12).log()
            else:
                return None, None
            zt = zt.float() / self.kd_temp
            if self.kd_topk > 0:
                v, ids = zt.topk(self.kd_topk, dim=-1)
                return ids, F.log_softmax(v, -1)
            return None, F.log_softmax(zt, -1)

    def teacher_flops_per_token(self) -> float:
        if self.kd_teacher.startswith("ckpt:"):
            return float(super().flops_per_token()) / 3.0
        return float(TEACHER_FWD_FLOPS.get(self.kd_teacher, 0.0))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        if not self.training:
            return super().forward(idx, targets)
        z = self.logits(self.hidden(idx))
        B, T, V = z.shape
        ce = F.cross_entropy(z.view(B * T, V), targets.reshape(B * T), ignore_index=-1)
        lps = F.log_softmax(z.float() / self.kd_temp, -1)
        ids, lpt = self._teacher_targets(idx)
        kind = self._t["kind"]
        if kind == "uniform":
            kl_tok = -math_log(V) - lps.mean(-1)
        elif kind == "unigram":
            kl_tok = self._t["negH"] - lps @ self._t["u"][:V]
        elif ids is not None:
            kl_tok = (lpt.exp() * (lpt - lps.gather(-1, ids))).sum(-1)
        else:
            lps = lps[..., : lpt.shape[-1]]
            kl_tok = (lpt.exp() * (lpt - lps)).sum(-1)
        mask = (targets >= 0).float()
        kl = (kl_tok * mask).sum() / mask.sum().clamp_min(1) * self.kd_temp ** 2
        return ce, {"aux_loss": self.kd_alpha * (kl - ce), "kd_kl": kl.detach()}


def math_log(x: int) -> float:
    import math
    return math.log(x)


def build(cfg: ModelConfig) -> KDGPT:
    return KDGPT(cfg)
