"""BabyLM training harness: the frozen sprint-1 recipe on the BabyLM corpus, multi-epoch, word milestones.

    python -m babylm.harness.train --model baseline --size small --split train_10M --epochs 10 --seed 0 \
        --compile --save --out babylm/runs/port_small_10M_e10_s0

Reused UNCHANGED from sprint-1 (imported): the model (sprint1.models.baseline, or babylm.models.baseline_reg for
the dropout arms), the optimizer (sprint1.harness.optim.build_optimizer: Muon + AdamW groups), the LR schedule
(sprint1.harness.train.lr_factor) and momentum warmup, the eval-state snapshot/restore and the interleaved eval
layout. Every optimizer/schedule default equals babylm/configs/frozen_port.json (= sprint1 frozen_baseline.json).

Data: babylm.data.corpus (run python -m babylm.data.prepare first). Budget: total tokens = epochs x split tokens,
steps = ceil(total / batch_tokens); windows from babylm.harness.stream.EpochSampler.
words_seen = word-initial target tokens consumed (BabyLM counts whitespace words of input).
Milestones (1M..10M by 1M, 20M..100M by 10M, 200M..1B by 100M words seen; the first milestone not reached is
also fired at the end if words_seen >= 99% of it, e.g. 10 epochs of train_10M = 99.5M words -> chck_100M):
  evaluate DEV-sub and TRAIN-sub (babylm/data/subsets) -> curves.csv row + pt/pt_{dev,train}_<n>M.npy (float16
  per-token nats in stream order, NaN for the unscored first token); with --save also checkpoints/chck_<n>M.pt
  (fp32 state_dict + config + words_seen) and resume.pt (model, optimizer, sampler, RNG, counters). A run that ends
  short of that 99% (e.g. 3.33 epochs) evaluates "final" and saves its end weights as checkpoints/chck_final.pt.
End: full DEV split scored the same way (pt/pt_devfull_final.npy). FINAL (test) only with --final.
Run dir: config.json, train_log.csv, log.jsonl, curves.csv, result.json, pt/, checkpoints/, resume.pt.
Attack-base levers (babylm.harness.levers; each off by default, and off leaves this file's port path unchanged):
  --lr_scale S (every base LR x S), --window_plan (per-epoch line/stream windows and lengths), --mntp_frac (GPT-BERT
  MNTP hybrid; pads the model vocab to --vocab_pad, MASK = 16384), --ema_decay (EMA weights for every eval and for
  checkpoints; raw weights kept as model_raw).
Attack A3 (babylm.harness.bpe_dropout; off by default, off leaves the port path unchanged): --bpe_dropout 0.1x8,0x2
  trains epoch e on a seeded BPE-dropout re-tokenisation of the split (same whitespace words, more tokens; steps =
  ceil(sum of epoch stream tokens / batch)); every evaluation stays canonical.
Second LR cycle (attack A4; babylm.harness.cycle; off by default, and off builds nothing): --init_from CKPT (its weights,
  fresh optimizer, a new schedule over this run's steps; --warmup_frac), --sources KIND:WHERE:EPOCHS ... (real and
  synthetic streams interleaved in proportion, --cycle_window), words_seen continuing from the checkpoint's lineage
  exposure, words per source logged, and an R0 ledger (--exposure_extra, --exposure_cap) refused when over the cap.
  A4's T2 arm (babylm.a4.t2): --exposure_rule t2 caps corpus words only (synthetic uncapped, labelled outside R0.2);
  --kd_teacher G --kd_rows synthetic puts the KD loss on the synthetic rows only, plain CE on the real rows.
Logit distillation (babylm.moonshot.kd; off by default): --kd_teacher CKPT, --kd_alpha, --kd_temp, --kd_start_epoch.
Devices (--device auto = cuda > mps > cpu): cuda = bf16 autocast (+ --compile); mps = fp32 by default
(--amp_dtype bf16|fp16 untested), never compiled, PYTORCH_ENABLE_MPS_FALLBACK must be unset; cpu = fp32.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from babylm.data import corpus as C
from babylm.harness import cycle as CY
from babylm.harness import levers as LV
from babylm.moonshot import kd as KD
from babylm.a4 import t2 as T2
from babylm.harness.stream import (MILESTONES, EpochSampler, gather_windows, milestone_label, stream_batches,
                                   words_in_targets)
from sprint1.harness.model_api import ModelConfig
from sprint1.harness.optim import build_optimizer
from sprint1.harness.train import (count_params, estimate_flops, harness_flops_per_token, lr_factor, parse_extra,
                                   restore_state, scalarize, snapshot_state, write_json)

REPO_ROOT = C.REPO_ROOT
SIZES: dict[str, dict[str, Any]] = {
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, seq_len=128),     # CPU tests only
    "small": dict(n_layer=6, n_head=6, n_embd=384, seq_len=1024),  # the frozen sprint-1 size
    "S": dict(n_layer=12, n_head=6, n_embd=384, seq_len=1024),
    "M": dict(n_layer=12, n_head=12, n_embd=768, seq_len=1024),
    "L": dict(n_layer=16, n_head=16, n_embd=1024, seq_len=1024),
}
BATCH_TOKENS = {"tiny": 8192, "small": 65536, "S": 65536, "M": 65536, "L": 65536}
# sequences per forward (gradient accumulation fills the batch); see README for the memory estimates
MICRO = {"cuda": {"tiny": 64, "small": 16, "S": 16, "M": 8, "L": 8},
         "mps": {"tiny": 64, "small": 8, "S": 4, "M": 2, "L": 1},
         "cpu": {"tiny": 64, "small": 4, "S": 2, "M": 1, "L": 1}}
EVAL_BATCH = {"cuda": 32, "mps": 8, "cpu": 8}
PT_REL_TOL = 1e-3
END_FIRE_FRAC = 0.99
TRAIN_LOG_FIELDS = ["step", "tokens_seen", "words_seen", "epoch", "loss", "lr_factor", "momentum", "step_time",
                    "wallclock"]
CURVE_FIELDS = ["step", "tokens_seen", "words_seen", "epoch", "label", "train_loss",
                "dev_nats_per_token", "dev_nats_per_word", "dev_bpb",
                "train_nats_per_token", "train_nats_per_word", "train_bpb", "gap_nats_per_word",
                "wallclock", "train_time"]


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="baseline", help="babylm.models.<name> or sprint1.models.<name> (build(cfg))")
    ap.add_argument("--size", default="small", choices=sorted(SIZES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=None, help="window order seed (default: --seed)")
    b = ap.add_argument_group("data / budget")
    b.add_argument("--split", default="train_10M", help="training split (train_10M, train_20M, ..., train_100M)")
    b.add_argument("--epochs", type=float, default=10.0, help="total tokens = epochs x split tokens")
    b.add_argument("--steps", type=int, default=None, help="override the step count (tests)")
    b.add_argument("--batch_tokens", type=int, default=None)
    b.add_argument("--micro_batch_size", type=int, default=None, help="sequences per forward (grad accumulation)")
    b.add_argument("--data_dir", default=None, help=f"default {C.DEFAULT_DATA_DIR}")
    o = ap.add_argument_group("optimizer (defaults = sprint1 frozen_baseline.json)")
    o.add_argument("--muon_lr", type=float, default=0.065)
    o.add_argument("--muon_momentum", type=float, default=0.95)
    o.add_argument("--muon_momentum_start", type=float, default=0.85)
    o.add_argument("--muon_momentum_warmup_steps", type=int, default=150)
    o.add_argument("--muon_wd", type=float, default=0.0)
    o.add_argument("--lr_embed", type=float, default=0.78)
    o.add_argument("--lr_head", type=float, default=0.0208)
    o.add_argument("--lr_scalar", type=float, default=0.052)
    o.add_argument("--lr_batch_scaling", default="none", choices=["sqrt", "none"])
    o.add_argument("--ref_batch_tokens", type=int, default=524288)
    o.add_argument("--adam_beta1", type=float, default=0.8)
    o.add_argument("--adam_beta2", type=float, default=0.95)
    o.add_argument("--adam_eps", type=float, default=1e-10)
    o.add_argument("--adam_wd", type=float, default=0.0)
    o.add_argument("--warmup_steps", type=int, default=0)
    o.add_argument("--cooldown_frac", type=float, default=0.5)
    o.add_argument("--lr_floor", type=float, default=0.1)
    o.add_argument("--grad_clip", type=float, default=0.0)
    m = ap.add_argument_group("model overrides")
    for k, t in (("n_layer", int), ("n_head", int), ("n_embd", int), ("seq_len", int), ("mlp_ratio", float)):
        m.add_argument(f"--{k}", type=t, default=None)
    m.add_argument("--extra", action="append", nargs="+", default=None, metavar="KEY=VAL",
                   help="cfg.extra entries, e.g. softcap=15 unet=true dropout=0.1")
    e = ap.add_argument_group("evaluation")
    e.add_argument("--eval_batch_size", type=int, default=None,
                   help=f"rows per eval forward, the same for every arm (default {EVAL_BATCH})")
    e.add_argument("--milestone_scale", type=float, default=1.0,
                   help="TEST ONLY: multiply every word milestone (e.g. 0.001 -> 1k, 2k, ...)")
    e.add_argument("--no_full_dev", action="store_true", help="skip the end-of-run full DEV pass (tests)")
    e.add_argument("--no_pt", action="store_true", help="do not write per-token npy dumps")
    e.add_argument("--final", action="store_true",
                   help="score the protected FINAL split (test) at the end instead of full DEV (warns, marks)")
    r = ap.add_argument_group("run")
    r.add_argument("--out", default=None, help="run dir (default babylm/runs/<run_id>)")
    r.add_argument("--run_id", default=None)
    r.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="auto: cuda > mps > cpu")
    r.add_argument("--amp_dtype", default="auto", choices=["auto", "bf16", "fp16", "none"],
                   help="autocast dtype; auto = bf16 on cuda, none (fp32) on mps and cpu. fp16 uses dynamic loss "
                        "scaling. bf16/fp16 autocast on MPS is untested here")
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA; ignored on MPS)")
    r.add_argument("--save", action="store_true", help="milestone checkpoints + resume.pt")
    r.add_argument("--resume", default=None, help="'auto' (= <out>/resume.pt if present) or a resume.pt path")
    r.add_argument("--max_session_steps", type=int, default=None,
                   help="stop after this many steps in this invocation, write resume.pt, exit (status paused)")
    r.add_argument("--log_every", type=int, default=20)
    r.add_argument("--tag", default="")
    v = ap.add_argument_group("attack-base levers (babylm.harness.levers; all off by default)")
    v.add_argument("--lr_scale", type=float, default=1.0, help="multiplies every base LR (Muon and the Adam groups)")
    v.add_argument("--window_plan", default=None,
                   help="per-epoch windows fmt:T[xN],... with fmt line|stream, e.g. line:512 or "
                        "line:128x6,line:256x2,line:512x2 (needs integer --epochs; model seq_len = max T)")
    v.add_argument("--mntp_frac", type=float, default=0.0, help="MNTP share of the sequences of every micro-batch")
    v.add_argument("--mntp_p_start", type=float, default=0.3)
    v.add_argument("--mntp_p_end", type=float, default=0.15)
    v.add_argument("--mntp_random", type=float, default=0.1)
    v.add_argument("--mntp_keep", type=float, default=0.1)
    v.add_argument("--vocab_pad", type=int, default=None,
                   help="model vocab (default 16384; 16512 when --mntp_frac > 0; MASK = 16384, never emitted)")
    v.add_argument("--ema_decay", type=float, default=0.0)
    v.add_argument("--tie_embeddings", action="store_true",
                   help="tie input embedding and head (Press & Wolf 2017): one weight, initialised and optimised as "
                        "the head (N(0, head_init_std), head AdamW group); embnorm makes the input side scale-free")
    v.add_argument("--head_wd", type=float, default=None,
                   help="AdamW weight decay of the untied head group only (default: --adam_wd); decay per step = lr x wd")
    v.add_argument("--embed_wd", type=float, default=None,
                   help="AdamW weight decay of the embedding group only (default: --adam_wd); lr_embed is 0.78, keep tiny")
    a3 = ap.add_argument_group("attack A3: BPE-dropout (babylm.harness.bpe_dropout; off by default)")
    a3.add_argument("--bpe_dropout", default=None,
                    help="per-epoch dropout p, e.g. 0.1x8,0x2 (needs integer --epochs; eval stays canonical)")
    a3.add_argument("--bpe_dropout_scope", default="all", choices=["all", "rare", "frequent"],
                    help="which whitespace words are re-segmented, by their training-split count")
    a3.add_argument("--bpe_dropout_rare_max", type=int, default=63, help="rare = count <= this (bands 1-63)")
    a3.add_argument("--bpe_dropout_workers", type=int, default=None, help="processes building the streams")
    c = ap.add_argument_group("second LR cycle (attack A4; babylm.harness.cycle; off by default)")
    c.add_argument("--init_from", default=None,
                   help="start from this checkpoint's weights (chck_*.pt or resume.pt; same model cfg required); "
                        "fresh optimizer state and a new LR schedule over this run's steps")
    c.add_argument("--init_weights", default="eval", choices=["eval", "raw"],
                   help="eval = the weights the run evaluated (EMA shadow if it had EMA); raw = optimizer weights")
    c.add_argument("--sources", nargs="+", default=None, metavar="KIND:WHERE:EPOCHS",
                   help="cycle data instead of --split/--epochs: KIND real|synthetic, WHERE a split name or a "
                        "babylm.a4 synthetic data dir, EPOCHS whole passes; sources are interleaved in proportion")
    c.add_argument("--cycle_window", default=None, help="window spec of the cycle, fmt:T (default stream:<seq_len>)")
    c.add_argument("--warmup_frac", type=float, default=0.0,
                   help="linear LR warmup over this share of the run's steps (overrides --warmup_steps when > 0)")
    c.add_argument("--exposure_prior", type=int, default=None,
                   help="lineage words before this run (default: the --init_from checkpoint's words_seen)")
    c.add_argument("--exposure_extra", nargs="+", default=None, metavar="NAME=WORDS",
                   help="other training in this arm's R0 budget, e.g. generator trainings: gen_A=9953773, or "
                        "gen_A=@RUN/result.json (that run's words_seen)")
    c.add_argument("--exposure_cap", type=int, default=100_000_000,
                   help="R0 cap on prior + extra + planned cycle words (refused above it)")
    c.add_argument("--exposure_rule", default="r0", choices=list(CY.RULES),
                   help="r0: the cap covers every word (R0.2); t2: closed-system augmentation (official FAQ, critic "
                        "ref-0014), the cap covers corpus words (prior + extra + real) and synthetic words are uncapped")
    c.add_argument("--kd_rows", default="all", choices=["all", "synthetic"],
                   help="with --kd_teacher and --sources: synthetic = KD on the rows of synthetic sources only, plain "
                        "CE on real rows (attack A4's T2 arm)")
    KD.add_args(ap)
    return ap


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def pick_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_for(device: torch.device, amp_dtype: str = "auto"):
    """(autocast context, precision label). auto = bf16 on CUDA, fp32 elsewhere."""
    if amp_dtype == "auto":
        amp_dtype = "bf16" if device.type == "cuda" else "none"
    if amp_dtype == "none":
        return contextlib.nullcontext(), "fp32"
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]
    return torch.autocast(device.type, dtype=dt), f"{amp_dtype}-autocast({device.type})"


def sync(device: torch.device) -> None:
    """Device-generic synchronize (for timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class LossScaler:
    """Dynamic loss scaling for fp16 autocast (no-op otherwise): grads are unscaled in place before the optimizer;
    a step with a non-finite gradient is skipped and the scale halved; the scale doubles after `interval` good steps."""

    def __init__(self, enabled: bool, init: float = 2.0 ** 15, interval: int = 1000):
        self.enabled, self.scale, self.interval, self.good, self.skipped = enabled, init, interval, 0, 0

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale if self.enabled else x

    def unscale_and_check(self, params) -> bool:
        if not self.enabled:
            return True
        grads = [p.grad for p in params if p.grad is not None]
        for g in grads:
            g.div_(self.scale)
        finite = bool(torch.stack([g.float().abs().max() for g in grads]).isfinite().all()) if grads else True
        if not finite:
            self.scale /= 2
            self.good = 0
            self.skipped += 1
            return False
        self.good += 1
        if self.good % self.interval == 0:
            self.scale *= 2
        return True

    def state_dict(self) -> dict:
        return {"scale": self.scale, "good": self.good, "skipped": self.skipped}

    def load_state_dict(self, sd: dict) -> None:
        self.scale, self.good, self.skipped = sd["scale"], sd["good"], sd["skipped"]


def git_commit() -> str:
    env = os.environ.get("HINTON_GIT_COMMIT")
    if env:
        return env
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "babylm",
                                "sprint1/models", "sprint1/harness"], capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return sha + ("+dirty" if dirty else "") if sha else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def make_cfg(args) -> ModelConfig:
    d = dict(SIZES[args.size])
    for k in ("n_layer", "n_head", "n_embd", "seq_len", "mlp_ratio"):
        v = getattr(args, k)
        if v is not None:
            d[k] = v
    extra = parse_extra([x for group in (args.extra or []) for x in group])
    if getattr(args, "window_plan", None):
        d["seq_len"] = max(sp.T for sp in LV.parse_plan(args.window_plan, int(args.epochs), d["seq_len"]))
    if getattr(args, "sources", None) and getattr(args, "cycle_window", None):
        d["seq_len"] = CY.parse_window(args.cycle_window, d["seq_len"]).T
    vocab = getattr(args, "vocab_pad", None) or (16512 if getattr(args, "mntp_frac", 0) > 0 else C.VOCAB_SIZE)
    return ModelConfig(vocab_size=vocab, extra=extra, tie_embeddings=bool(getattr(args, "tie_embeddings", False)), **d)


def load_model_module(name: str):
    for pkg in ("babylm.models", "sprint1.models"):
        try:
            return importlib.import_module(f"{pkg}.{name}")
        except ModuleNotFoundError as ex:
            if ex.name != f"{pkg}.{name}":
                raise
    raise SystemExit(f"no model module {name!r} in babylm.models or sprint1.models")


def mem_stats(device: torch.device) -> dict[str, int]:
    s = {"host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
         * (1 if sys.platform == "darwin" else 1024)}
    if device.type == "cuda":
        s["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated(device))
    elif device.type == "mps":
        s["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    return s


def is_oom(ex: BaseException) -> bool:
    return isinstance(ex, torch.cuda.OutOfMemoryError) or (isinstance(ex, RuntimeError) and "out of memory" in str(ex))


def label_words(m: int) -> str:
    return f"{m // 1_000_000}M"


def rng_state(device: torch.device) -> dict[str, Any]:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if device.type == "cuda" and st.get("cuda") is not None:
        torch.cuda.set_rng_state_all(st["cuda"])
    if device.type == "mps" and st.get("mps") is not None:
        torch.mps.set_rng_state(st["mps"])


def atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def truncate_logs(out: Path, step: int) -> None:
    """On resume: drop log rows written after the resumed step (a crash after the last resume.pt)."""
    for name in ("train_log.csv", "curves.csv"):
        p = out / name
        if p.exists():
            rows = list(csv.DictReader(p.open()))
            fields = rows[0].keys() if rows else None
            if fields:
                keep = [r for r in rows if int(r["step"]) <= step]
                with p.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(fields))
                    w.writeheader()
                    w.writerows(keep)
    p = out / "log.jsonl"
    if p.exists():
        keep = [ln for ln in p.read_text().splitlines() if ln.strip() and json.loads(ln).get("step", 0) <= step]
        p.write_text("".join(ln + "\n" for ln in keep))


def dev_sub_minimum(curves: Path) -> dict[str, Any] | None:
    """Lowest DEV-sub nats/word over every milestone in curves.csv (all sessions), with where it happened. Logged
    for diagnosis only: selection stays on the final model."""
    if not curves.exists():
        return None
    rows = [r for r in csv.DictReader(curves.open()) if r.get("dev_nats_per_word")]
    if not rows:
        return None
    r = min(rows, key=lambda r: float(r["dev_nats_per_word"]))
    return {"dev_nats_per_word": float(r["dev_nats_per_word"]), "label": r["label"], "epoch": float(r["epoch"]),
            "words_seen": int(float(r["words_seen"])), "step": int(r["step"]),
            "final_minus_min": float(rows[-1]["dev_nats_per_word"]) - float(r["dev_nats_per_word"])}


class CsvLog:
    def __init__(self, path: Path, fields: list[str]):
        new = not path.exists() or path.stat().st_size == 0
        self.f = path.open("a", newline="", buffering=1)
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def row(self, d: dict[str, Any]) -> None:
        self.w.writerow(d)

    def close(self) -> None:
        self.f.close()


# ------------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_stream(model: nn.Module, raw_model: nn.Module, data: C.SplitData, token_bytes: np.ndarray, T: int,
                    batch: int, device: torch.device, amp) -> tuple[dict[str, Any], np.ndarray]:
    """Score every token of the stream except the first exactly once (stream_batches); returns metrics and the
    per-token nats (float32 [N], NaN at 0). Sprint-1 discipline: fixed eval batch with padded rows, buffers
    snapshotted and restored, reset_state() before every batch if defined, train mode restored."""
    tokens = data.tokens
    N = len(tokens)
    pt = np.full(N, np.nan, dtype=np.float32)
    reset = callable(getattr(raw_model, "reset_state", None))
    snap = snapshot_state(raw_model)
    was_training = model.training
    model.eval()
    max_diff = 0.0
    try:
        for x, y, win, tpos in stream_batches(tokens, T, batch):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if reset:
                raw_model.reset_state()
            with amp:
                loss, aux = model(x, y)
            per = aux.get("per_token_loss") if isinstance(aux, dict) else None
            if not (isinstance(per, torch.Tensor) and per.shape == y.shape):
                raise RuntimeError("[eval] the model must return aux['per_token_loss'] [B, T] in eval mode")
            valid = y >= 0
            if valid.any():
                hp = torch.float32 if per.device.type == "mps" else torch.float64  # MPS has no float64
                d = (per.to(hp)[valid].mean() - loss.to(hp)).abs().item()
                max_diff = max(max_diff, d)
                if d > PT_REL_TOL * max(1.0, abs(loss.item())):
                    raise RuntimeError(f"[eval] per_token_loss mean differs from the loss by {d:.3g}")
            per_np = per.float().cpu().numpy()
            m = tpos >= 0
            pt[tpos[m]] = per_np[m]
    finally:
        mutated = restore_state(raw_model, snap)
        model.train(was_training)
    if np.isnan(pt[1:]).any():
        raise RuntimeError(f"[eval] {int(np.isnan(pt[1:]).sum())} targets were not scored")
    tgt = np.asarray(tokens[1:], dtype=np.int64)
    nats = float(pt[1:].astype(np.float64).sum())
    nbytes = int(token_bytes[tgt].sum())
    src = np.asarray(data.src[1:])
    winit = C.word_initial(np.asarray(data.words))[1:]
    per_source = {}
    for sid, s in enumerate(C.SOURCES):
        msk = src == sid
        if msk.any():
            ns = float(pt[1:][msk].astype(np.float64).sum())
            per_source[s] = {"nats": ns, "tokens": int(msk.sum()), "words": int(winit[msk].sum()),
                             "nats_per_word": ns / max(1, int(winit[msk].sum()))}
    res = {"nats": nats, "tokens_scored": N - 1, "words": data.n_words, "bytes": nbytes,
           "nats_per_token": nats / (N - 1), "nats_per_word": nats / data.n_words,
           "bpb": nats / (nbytes * math.log(2)), "per_source": per_source,
           "eval_batch_size": batch, "max_abs_batch_mean_diff": max_diff, "eval_state_mutated": mutated,
           "fingerprint": data.meta.get("fingerprint", data.meta.get("sha256"))}
    return res, pt


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    t_start = time.time()
    device = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
        raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK is set: unset it so that no op silently runs on the CPU")
    if args.compile and device.type == "mps":
        print("[train] --compile ignored on MPS", file=sys.stderr, flush=True)
        args.compile = False
    amp, precision = autocast_for(device, args.amp_dtype)
    scaler = LossScaler(precision.startswith("fp16"))
    if args.final:
        commit = git_commit()
        if commit.endswith("+dirty") or commit == "unknown":
            raise SystemExit(f"--final refused: git commit is {commit!r}; FINAL may only be scored from a clean tree")

    cfg = make_cfg(args)
    T = cfg.seq_len
    batch_tokens = args.batch_tokens or BATCH_TOKENS[args.size]
    batch_seqs = max(1, batch_tokens // T)
    micro = min(args.micro_batch_size or MICRO[device.type][args.size], batch_seqs)
    while batch_seqs % micro:
        micro -= 1
    eval_bs = args.eval_batch_size or EVAL_BATCH[device.type]
    data_seed = args.seed if args.data_seed is None else args.data_seed

    run_id = args.run_id or f"{args.model}_{args.size}_{args.split}_e{args.epochs:g}_s{args.seed}"
    out = Path(args.out) if args.out else REPO_ROOT / "babylm" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "pt").mkdir(exist_ok=True)
    if args.final:
        bar = "!" * 78
        print(f"\n{bar}\n!!! --final: the protected FINAL split (BabyLM test) will be scored at the end of this run.\n"
              f"!!! Use only for the frozen, pre-registered final comparison. Recorded in result.json and\n"
              f"!!! {out / 'FINAL_SPLIT_USED'}.\n{bar}\n", file=sys.stderr, flush=True)
        (out / "FINAL_SPLIT_USED").write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                          "argv": sys.argv}) + "\n")

    # data
    train = C.load_split(args.split, args.data_dir)
    dev_sub = C.load_split("dev_sub", args.data_dir)
    train_sub = C.load_split("train_sub", args.data_dir)
    token_bytes = C.token_byte_lengths()
    word_init = C.word_initial(np.asarray(train.words))
    N = train.n_tokens
    total_tokens = args.epochs * N
    plan = None
    bpd = None
    cyc = None
    if args.kd_rows != "all" and (not (args.kd_teacher and args.sources) or args.mntp_frac > 0 or args.kd_perm):
        raise SystemExit("--kd_rows synthetic needs --kd_teacher and --sources, and no --mntp_frac or --kd_perm")
    if args.exposure_rule != "r0" and not args.sources:
        raise SystemExit("--exposure_rule t2 applies to a cycle (--sources)")
    if args.bpe_dropout:
        if args.sources:
            raise SystemExit("--bpe_dropout and --sources (A3 and A4) are exclusive")
        from babylm.harness import bpe_dropout as BD
        if args.window_plan or args.epochs != int(args.epochs):
            raise SystemExit("--bpe_dropout needs an integer --epochs and no --window_plan")
        bpd = BD.parse_plan(args.bpe_dropout, int(args.epochs))
        bpd_meta = BD.build_streams(train, args.data_dir, bpd, args.bpe_dropout_scope, args.bpe_dropout_rare_max,
                                    data_seed, args.bpe_dropout_workers,
                                    log=lambda m: print(m, flush=True))
        streams = [BD.load_stream(args.data_dir, m["name"]) if m else (train.tokens, train.words) for m in bpd_meta]
        sampler = BD.MultiStreamSampler(streams, (train.tokens, train.words), T, data_seed)
        bpd_lengths = [len(st_[0]) for st_ in streams]
        total_steps = (args.steps if args.steps is not None
                       else BD.steps_for(bpd_lengths, batch_seqs * T))
    elif args.sources:  # second LR cycle (babylm.harness.cycle): the plan path with a multi-source sampler
        if args.window_plan:
            raise SystemExit("--sources and --window_plan are exclusive (the cycle's window is --cycle_window)")
        srcs = [CY.load_source(x, args.data_dir) for x in args.sources]
        for so in srcs:
            if so.data.meta.get("tokenizer_sha256") != train.meta["tokenizer_sha256"]:
                raise SystemExit(f"--sources {so.label}: tokenizer differs from {args.split}'s")
        spec = CY.parse_window(args.cycle_window, T)
        nl = None
        if spec.fmt == "line":
            from tokenizers import Tokenizer
            nl = LV.newline_ids(Tokenizer.from_file(str(C.TOKENIZER_PATH)))
        sampler = cyc = CY.CycleSampler(srcs, spec, batch_seqs * T, data_seed, bos_id=C.BOS_ID, pad_id=C.BOS_ID,
                                        nl_ids=nl)
        plan = [spec]
        total_steps = cyc.total_steps if args.steps is None else min(args.steps, cyc.total_steps)
    elif args.window_plan:
        if args.epochs != int(args.epochs):
            raise SystemExit("--window_plan needs an integer --epochs")
        plan = LV.parse_plan(args.window_plan, int(args.epochs), T)
        starts_l = None
        if any(sp.fmt == "line" for sp in plan):
            from tokenizers import Tokenizer
            starts_l = LV.line_starts(np.asarray(train.tokens), LV.newline_ids(Tokenizer.from_file(str(C.TOKENIZER_PATH))),
                                      C.BOS_ID)
        sampler = LV.PlanSampler(train.tokens, plan, batch_seqs * T, data_seed, bos_id=C.BOS_ID, pad_id=C.BOS_ID,
                                 starts=starts_l)
        total_steps = sampler.total_steps if args.steps is None else min(args.steps, sampler.total_steps)
    else:
        total_steps = args.steps if args.steps is not None else max(1, math.ceil(total_tokens / (batch_seqs * T)))
        sampler = EpochSampler(N, T, data_seed)

    def epoch_now() -> float:
        return round(st["tokens_seen"] / N, 6) if bpd is None else round(sampler.epoch_float(), 6)
    micro_tokens = micro * T  # tokens per forward stay constant when a plan uses shorter windows

    def micro_for(n_seqs: int, Tw: int) -> int:
        mm = max(1, min(n_seqs, micro_tokens // Tw))
        while n_seqs % mm:
            mm -= 1
        return mm
    milestones = [(m, max(1, int(round(m * args.milestone_scale)))) for m in MILESTONES]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mod = load_model_module(args.model)
    raw_model = mod.build(cfg).to(device)
    raw_model.train()
    if cfg.tie_embeddings:
        if raw_model.lm_head.weight is not raw_model.embed.weight:
            raise SystemExit(f"--tie_embeddings: model {args.model!r} did not tie embed and lm_head")
        with torch.no_grad():  # the tie replaced the head's init by the embedding's N(0, 1): use the head's
            raw_model.embed.weight.normal_(0.0, float(cfg.extra.get("head_init_std", 0.005)))
    init_info = None
    if args.init_from:  # second LR cycle: the checkpoint's weights, before compile and EMA (the shadow starts there)
        sd_init, cfg_init, words_init = CY.load_init(args.init_from, args.init_weights)
        diff = CY.cfg_diff(cfg_init, cfg.to_dict())
        if diff:
            raise SystemExit(f"--init_from: model cfg differs from the checkpoint's (checkpoint, this run): {diff}")
        raw_model.load_state_dict(sd_init)
        del sd_init
        init_info = {"path": str(args.init_from), "sha256": C.sha256_file(args.init_from),
                     "weights": args.init_weights, "words_seen": words_init}
    mntp_on = args.mntp_frac > 0
    if mntp_on:
        if LV.make_switchable(raw_model) == 0:
            raise SystemExit(f"--mntp_frac: model {args.model!r} has no sprint-1 attention to switch")
        if cfg.vocab_size <= C.VOCAB_SIZE:
            raise SystemExit("--mntp_frac needs --vocab_pad > 16384 (MASK = 16384)")
    mask_id = C.VOCAB_SIZE
    mgen = torch.Generator().manual_seed(data_seed + 7_000_003)
    if args.compile:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
    model = torch.compile(raw_model) if args.compile else raw_model
    ema = LV.EMA(raw_model, args.ema_decay) if args.ema_decay > 0 else None
    ev_model, ev_raw = (ema.shadow, ema.shadow) if ema is not None else (model, raw_model)
    kd_teacher, kd_meta, kd_student = None, None, None
    kd_perm_gen = torch.Generator().manual_seed(data_seed + 9_000_011) if args.kd_perm else None
    if args.kd_perm and not args.kd_teacher:
        raise SystemExit("--kd_perm needs --kd_teacher")
    if args.kd_teacher:  # babylm.moonshot.kd (diagnostics): off leaves the port path unchanged
        if not (hasattr(raw_model, "hidden") and hasattr(raw_model, "logits")):
            raise SystemExit(f"--kd_teacher: model {args.model!r} has no hidden()/logits() split API")
        kd_teacher, kd_meta = KD.load_teacher(args.kd_teacher, device, load_model_module)
        if kd_meta.get("tokenizer_sha256") not in (None, train.meta["tokenizer_sha256"]):
            raise SystemExit("--kd_teacher: the teacher was trained with a different tokenizer")
        kd_student = KD.StudentLogits(raw_model)
        if args.compile:
            kd_student = torch.compile(kd_student)
    params_total, params_nonemb = count_params(raw_model)
    fpt, fpt_src = estimate_flops(raw_model, cfg)

    scale = (batch_seqs * T / args.ref_batch_tokens) ** 0.5 if args.lr_batch_scaling == "sqrt" else 1.0
    ls = args.lr_scale
    base_lrs = {"muon": args.muon_lr * ls, "adam_embed": args.lr_embed * scale * ls,
                "adam_head": args.lr_head * scale * ls, "adam_scalar": args.lr_scalar * scale * ls}
    mk = getattr(raw_model, "make_optimizer", None) or getattr(mod, "make_optimizer", None)
    if mk is not None:
        for k, v in dict(muon_momentum=args.muon_momentum, muon_wd=args.muon_wd,
                         adam_betas=[args.adam_beta1, args.adam_beta2], adam_eps=args.adam_eps,
                         adam_wd=args.adam_wd).items():
            cfg.extra.setdefault(k, v)
        optimizer = mk(raw_model, cfg, dict(base_lrs))
        opt_src = "model.make_optimizer"
    else:
        optimizer = build_optimizer(raw_model, base_lrs, muon_momentum=args.muon_momentum, muon_wd=args.muon_wd,
                                    adam_betas=(args.adam_beta1, args.adam_beta2), adam_eps=args.adam_eps,
                                    adam_wd=args.adam_wd)
        opt_src = "harness.muon_adamw"
    if cfg.tie_embeddings:  # tied weight is named embed.weight, so it sits in the embed group: move it to a head group
        tied = raw_model.embed.weight
        adams = [o for o in getattr(optimizer, "optimizers", [optimizer]) if isinstance(o, torch.optim.AdamW)]
        src = [g for o in adams for g in o.param_groups if any(q is tied for q in g["params"])]
        if len(adams) != 1 or len(src) != 1 or any(g.get("name") == "head" for g in adams[0].param_groups):
            raise SystemExit("--tie_embeddings: unexpected optimizer layout")
        src[0]["params"] = [q for q in src[0]["params"] if q is not tied]
        adams[0].param_groups[:] = [g for g in adams[0].param_groups if g["params"]]
        adams[0].add_param_group(dict(params=[tied], lr=base_lrs["adam_head"], initial_lr=base_lrs["adam_head"],
                                      kind="adam", name="head"))
    group_wd = {"head": args.head_wd, "embed": args.embed_wd}
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", g["lr"])
        wd = group_wd.get(g.get("name")) if g.get("kind") == "adam" else None
        if wd is not None:
            g["weight_decay"] = wd
    for k, wd in group_wd.items():
        if wd is not None and not any(g.get("kind") == "adam" and g.get("name") == k for g in optimizer.param_groups):
            raise SystemExit(f"--{k}_wd given but the optimizer has no adam '{k}' group")

    # state (possibly resumed)
    st = {"step": 0, "tokens_seen": 0, "words_seen": 0, "fired": [], "train_time": 0.0, "raw_train_time": 0.0,
          "compile_s": 0.0, "eval_time": 0.0, "sessions": 0, "last_loss": float("nan"), "status": "running"}
    lineage = None
    if init_info is not None or cyc is not None:  # words_seen = lineage exposure: prior training + this run
        prior = args.exposure_prior if args.exposure_prior is not None else \
            int((init_info or {}).get("words_seen") or 0)
        extra_w, extra_from = {}, {}
        for kv in args.exposure_extra or []:
            k, _, v = kv.partition("=")
            if v.startswith("@"):  # a finished run's result.json: its words_seen
                extra_from[k] = {"path": v[1:], "sha256": C.sha256_file(v[1:])}
                v = json.loads(Path(v[1:]).read_text())["words_seen"]
            extra_w[k] = int(float(v))
        st["words_seen"] = prior
        st["fired"] = [m for m, thr in milestones if thr <= prior]
        st["words_by_kind"] = {k: 0 for k in CY.KINDS}
        steps_used = total_steps * batch_seqs
        planned = {k: 0 for k in CY.KINDS}
        if cyc is not None:
            for si, so in enumerate(cyc.sources):
                planned[so.kind] += int(cyc.w_words[:steps_used][cyc.w_src[:steps_used] == si].sum())
        planned_total = prior + sum(extra_w.values()) + sum(planned.values())
        lineage = {"init": init_info, "prior_words": prior, "extra_words": extra_w, "extra_from": extra_from,
                   "planned_cycle_words": planned,
                   "planned_total": planned_total, "cap": args.exposure_cap,
                   "sources": None if cyc is None else [so.label for so in cyc.sources],
                   "source_sha256": None if cyc is None else [so.data.meta.get("sha256") for so in cyc.sources],
                   "window": None if cyc is None else f"{cyc.spec.fmt}:{cyc.spec.T}",
                   "dropped_windows": None if cyc is None else cyc.dropped_windows,
                   "cycle_sha256": C.sha256_file(Path(CY.__file__))}
        if cyc is not None and planned_total > args.exposure_cap and args.exposure_rule == "r0":
            raise SystemExit(f"R0: planned exposure {planned_total:,} words (prior {prior:,} + extra "
                             f"{sum(extra_w.values()):,} + cycle {sum(planned.values()):,}) exceeds the cap "
                             f"{args.exposure_cap:,}")
        if args.exposure_rule != "r0":  # T2 (critic ref-0014): corpus words capped, synthetic words uncapped
            capped = CY.capped_words(args.exposure_rule, prior, extra_w, planned)
            lineage.update({"rule": args.exposure_rule, "tier": CY.RULES[args.exposure_rule], "planned_capped": capped})
            if capped > args.exposure_cap:
                raise SystemExit(f"T2: planned corpus exposure {capped:,} words (prior {prior:,} + extra "
                                 f"{sum(extra_w.values()):,} + real {planned['real']:,}) exceeds the cap "
                                 f"{args.exposure_cap:,}")
    warmup = args.warmup_steps if args.warmup_frac <= 0 else max(1, int(round(args.warmup_frac * total_steps)))
    resume_path = None
    if args.resume:
        resume_path = (out / "resume.pt") if args.resume == "auto" else Path(args.resume)
        if args.resume == "auto" and not resume_path.exists():
            resume_path = None
    if resume_path is not None:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        if ck["total_steps"] != total_steps or ck["split"] != args.split or ck["cfg"] != cfg.to_dict():
            raise SystemExit(f"{resume_path} is for a different run (steps/split/cfg differ)")
        raw_model.load_state_dict(ck.get("model_raw", ck["model"]))
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        sampler.load_state_dict(ck["sampler"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"], device)
        st.update(ck["state"])
        truncate_logs(out, st["step"])
        print(f"[train] resumed from {resume_path} at step {st['step']} ({st['words_seen']:,} words)", flush=True)
    elif (out / "train_log.csv").exists():
        for name in ("train_log.csv", "curves.csv", "log.jsonl"):  # fresh run in a used dir: start clean logs
            (out / name).unlink(missing_ok=True)
    st["sessions"] += 1
    st["status"] = "running"

    config = {"args": vars(args), "model_cfg": cfg.to_dict(), "batch_seqs": batch_seqs, "micro_batch_size": micro,
              "batch_tokens": batch_seqs * T, "base_lrs": base_lrs, "lr_scale": scale, "optimizer": opt_src,
              "total_steps": total_steps, "split": args.split, "split_tokens": N, "split_words": train.n_words,
              "epochs": args.epochs, "data_seed": data_seed, "device": str(device), "torch": torch.__version__,
              "precision": precision, "compile": bool(args.compile), "eval_batch_size": eval_bs,
              "milestones": [m for m, _ in milestones], "milestone_scale": args.milestone_scale,
              "data": {"data_dir": str(Path(args.data_dir or C.DEFAULT_DATA_DIR)),
                       "train_sha256": train.meta["sha256"], "tokenizer_sha256": train.meta["tokenizer_sha256"],
                       "dev_sub_fingerprint": dev_sub.meta["fingerprint"],
                       "train_sub_fingerprint": train_sub.meta["fingerprint"]},
              "levers": {"lr_scale": args.lr_scale, "window_plan": None if plan is None or cyc is not None else
                         [f"{sp.fmt}:{sp.T}" for sp in plan],
                         "dropped_windows": getattr(sampler, "dropped_windows", None),
                         "windows_per_epoch": getattr(sampler, "windows_per_epoch", None),
                         "mntp_frac": args.mntp_frac, "mntp_p": [args.mntp_p_start, args.mntp_p_end],
                         "mntp_random_keep": [args.mntp_random, args.mntp_keep],
                         "mask_id": mask_id if mntp_on else None, "vocab_size": cfg.vocab_size,
                         "tie_embeddings": cfg.tie_embeddings,
                         "ema_decay": args.ema_decay, "head_wd": args.head_wd, "embed_wd": args.embed_wd,
                         "adam_group_wd": {g.get("name"): g.get("weight_decay") for g in optimizer.param_groups
                                           if g.get("kind") == "adam"},
                         "levers_sha256": C.sha256_file(Path(LV.__file__))}}
    if bpd is not None:
        config["bpe_dropout"] = {"plan": bpd, "scope": args.bpe_dropout_scope, "rare_max": args.bpe_dropout_rare_max,
                                 "seed": data_seed, "epoch_tokens": bpd_lengths,
                                 "token_ratio": sum(bpd_lengths) / (len(bpd_lengths) * N),
                                 "streams": [m and {k: m[k] for k in ("name", "tokens", "inflation", "sha256",
                                                                       "sha256_words")} for m in bpd_meta],
                                 "module_sha256": C.sha256_file(Path(BD.__file__))}
    if kd_teacher is not None:
        config["levers"]["kd"] = {"teacher": kd_meta, "alpha": args.kd_alpha, "temp": args.kd_temp,
                                  "perm": args.kd_perm,
                                  "start_epoch": args.kd_start_epoch, "sha256": C.sha256_file(Path(KD.__file__))}
        if args.kd_rows != "all":  # A4 T2: KD on the synthetic rows only (babylm.a4.t2)
            config["levers"]["kd"].update({"rows": args.kd_rows, "t2_sha256": C.sha256_file(Path(T2.__file__))})
    if lineage is not None:
        config["cycle"] = {**lineage, "warmup_steps": warmup}
    write_json(out / "config.json", config)
    print(f"[train] {args.model} size={args.size} params={params_total:,} non-emb={params_nonemb:,} "
          f"flops/token={fpt:.3e} split={args.split} ({N:,} tokens, {train.n_words:,} words) epochs={args.epochs:g} "
          f"batch={batch_seqs}x{T} micro={micro} steps={total_steps} device={device} {precision} out={out}",
          flush=True)

    tlog = CsvLog(out / "train_log.csv", TRAIN_LOG_FIELDS + ([f"words_{k}" for k in CY.KINDS] if lineage else []))
    clog = CsvLog(out / "curves.csv", CURVE_FIELDS)
    jlog = open(out / "log.jsonl", "a", buffering=1)
    evals: dict[str, Any] = {}

    def save_resume() -> None:
        atomic_save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                     "ema": ema.state_dict() if ema is not None else None,
                     "sampler": sampler.state_dict(), "rng": rng_state(device), "state": dict(st),
                     "scaler": scaler.state_dict(),
                     "total_steps": total_steps, "split": args.split, "cfg": cfg.to_dict(),
                     "git_commit": git_commit()}, out / "resume.pt")

    def milestone(label: str, words_nominal: int | None) -> None:
        t0 = time.time()
        row: dict[str, Any] = {"step": st["step"], "tokens_seen": st["tokens_seen"], "words_seen": st["words_seen"],
                               "epoch": epoch_now(), "label": label,
                               "train_loss": st["last_loss"]}
        rec: dict[str, Any] = {}
        for tag, data in (("dev", dev_sub), ("train", train_sub)):
            r, pt = evaluate_stream(ev_model, ev_raw, data, token_bytes, T, eval_bs, device, amp)
            rec[tag] = r
            row[f"{tag}_nats_per_token"] = r["nats_per_token"]
            row[f"{tag}_nats_per_word"] = r["nats_per_word"]
            row[f"{tag}_bpb"] = r["bpb"]
            if not args.no_pt:
                np.save(out / "pt" / f"pt_{tag}_{label}.npy", pt.astype(np.float16))
        row["gap_nats_per_word"] = row["dev_nats_per_word"] - row["train_nats_per_word"]
        st["eval_time"] += time.time() - t0
        row["wallclock"] = time.time() - t_start
        row["train_time"] = st["train_time"]
        clog.row(row)
        jlog.write(json.dumps({"type": "milestone", **row, "words_nominal": words_nominal,
                               "per_source": {k: v["per_source"] for k, v in rec.items()}}) + "\n")
        evals[label] = row
        print(f"[eval] {label} step {st['step']} words {st['words_seen']:,} dev {row['dev_nats_per_word']:.4f} "
              f"nats/word ({row['dev_nats_per_token']:.4f}/tok) train {row['train_nats_per_word']:.4f} "
              f"gap {row['gap_nats_per_word']:+.4f}", flush=True)
        # "final" (a run ending short of its next milestone, e.g. fractional epochs) saves chck_final.pt, so the
        # finished weights are scoreable and the last chck_<n>M.pt is never mistaken for them
        if args.save and (label.endswith("M") or label == "final"):
            ckdir = out / "checkpoints"
            ckdir.mkdir(exist_ok=True)
            atomic_save({"model": {k: v.detach().float().cpu() for k, v in ev_raw.state_dict().items()},
                         "model_raw": None if ema is None else
                         {k: v.detach().float().cpu() for k, v in raw_model.state_dict().items()},
                         "ema_decay": args.ema_decay or None,
                         "mntp_frac": args.mntp_frac, "window_plan": args.window_plan, "lr_scale": args.lr_scale,
                         **({"bpe_dropout": config["bpe_dropout"]} if bpd is not None else {}),
                         "cfg": cfg.to_dict(), "model_name": args.model, "words_seen": st["words_seen"],
                         "words_nominal": words_nominal, "tokens_seen": st["tokens_seen"], "step": st["step"],
                         "label": milestone_label(words_nominal) if words_nominal else label,
                         "tokenizer_sha256": train.meta["tokenizer_sha256"], "git_commit": git_commit(),
                         **({"lineage": {**lineage, "words_by_kind": dict(st["words_by_kind"])}} if lineage else {})},
                        ckdir / f"{milestone_label(words_nominal) if words_nominal else 'chck_final'}.pt")

    session_times: list[float] = []
    session_compile = 0.0
    steps_this_session = 0
    reset = cfg.reset_state_on_doc and hasattr(raw_model, "reset_state")
    has_step_end = hasattr(raw_model, "on_step_end")
    while st["step"] < total_steps:
        t0 = time.time()
        if bpd is not None:
            x_all, y_all, n_words_batch = sampler.next_batch(batch_seqs)
            Tw, nb = T, batch_seqs
        elif plan is None:
            starts = sampler.next_starts(batch_seqs)
            x_all, y_all = gather_windows(train.tokens, starts, T)
            n_words_batch = words_in_targets(word_init, starts, T)
            Tw, nb = T, batch_seqs
        else:
            x_all, y_all, _, Tw, tpos = sampler.next()
            n_words_batch = int(word_init[tpos].sum()) if cyc is None else sum(sampler.last_words.values())
            nb = len(x_all)
        mic = micro if plan is None else micro_for(nb, Tw)
        k_m = LV.mntp_rows(mic, args.mntp_frac)
        p_m = LV.mntp_p(st["step"], total_steps, args.mntp_p_start, args.mntp_p_end)
        while True:
            try:
                loss_acc = torch.zeros((), device=device)
                aux_acc: dict[str, float] = {}
                n_micro = nb // mic
                kd_on = kd_teacher is not None and st["tokens_seen"] / N >= args.kd_start_epoch
                if k_m == 0 and kd_on and args.kd_rows == "synthetic":  # A4 T2: KD on synthetic rows, CE on real
                    syn_all = torch.from_numpy(cyc.last_syn)
                    for i in range(n_micro):
                        xb = x_all[i * mic:(i + 1) * mic].to(device, non_blocking=True)
                        yb = y_all[i * mic:(i + 1) * mic].to(device, non_blocking=True)
                        rb = syn_all[i * mic:(i + 1) * mic].to(device)
                        with amp:
                            zs = kd_student(xb)
                            with torch.no_grad():
                                zt = KD.teacher_logits(kd_teacher, xb[rb]) if bool(rb.any()) else None
                        loss, kst = T2.kd_rows_terms(zs, zt, yb, rb, args.kd_alpha, args.kd_temp)
                        scaler.loss(loss / n_micro).backward()
                        loss_acc += kst["kd_ce"] / n_micro
                        for k, v in kst.items():
                            aux_acc[k] = aux_acc.get(k, 0.0) + v / n_micro
                elif k_m == 0 and kd_on:  # logit distillation (babylm.moonshot.kd), causal rows
                    for i in range(n_micro):
                        xb = x_all[i * mic:(i + 1) * mic].to(device, non_blocking=True)
                        yb = y_all[i * mic:(i + 1) * mic].to(device, non_blocking=True)
                        with amp:
                            zs = kd_student(xb)
                            with torch.no_grad():
                                zt = KD.teacher_logits(kd_teacher, xb, kd_perm_gen)
                        loss, kst = KD.kd_terms(zs, zt, yb, args.kd_alpha, args.kd_temp)
                        scaler.loss(loss / n_micro).backward()
                        loss_acc += kst["kd_ce"] / n_micro
                        for k, v in kst.items():
                            aux_acc[k] = aux_acc.get(k, 0.0) + v / n_micro
                elif k_m == 0:  # the port's step, unchanged
                    for i in range(n_micro):
                        xb = x_all[i * mic:(i + 1) * mic].to(device, non_blocking=True)
                        yb = y_all[i * mic:(i + 1) * mic].to(device, non_blocking=True)
                        if reset:
                            raw_model.reset_state()
                        with amp:
                            loss, aux = model(xb, yb)
                        al = aux.get("aux_loss") if isinstance(aux, dict) else None
                        scaler.loss((loss + al if al is not None else loss) / n_micro).backward()
                        loss_acc += loss.detach().float() / n_micro
                        for k, v in scalarize(aux).items():
                            aux_acc[k] = aux_acc.get(k, 0.0) + v / n_micro
                else:  # MNTP hybrid: the last k_m rows of each micro-batch are masked-next-token, bidirectional
                    mntp_acc = 0.0
                    n_c = mic - k_m
                    for i in range(n_micro):
                        xb, yb = x_all[i * mic:(i + 1) * mic], y_all[i * mic:(i + 1) * mic]
                        tot = torch.zeros((), device=device)
                        if n_c > 0 and kd_on:
                            xc, yc = xb[:n_c].to(device, non_blocking=True), yb[:n_c].to(device, non_blocking=True)
                            with amp:
                                zs = kd_student(xc)
                                with torch.no_grad():
                                    zt = KD.teacher_logits(kd_teacher, xc, kd_perm_gen)
                            loss, kst = KD.kd_terms(zs, zt, yc, args.kd_alpha, args.kd_temp)
                            tot = tot + loss * (n_c / mic)
                            loss_acc += kst["kd_ce"] / n_micro
                            for k, v in kst.items():
                                aux_acc[k] = aux_acc.get(k, 0.0) + v / n_micro
                        elif n_c > 0:
                            with amp:
                                loss, _ = model(xb[:n_c].to(device, non_blocking=True),
                                                yb[:n_c].to(device, non_blocking=True))
                            tot = tot + loss * (n_c / mic)
                            loss_acc += loss.detach().float() / n_micro
                        xm, ym = LV.mntp_corrupt(xb[n_c:], yb[n_c:], p_m, mask_id, C.VOCAB_SIZE, mgen,
                                                 args.mntp_random, args.mntp_keep)
                        with LV.BidirectionalAttention(raw_model), amp:
                            lm, _ = model(xm.to(device, non_blocking=True), ym.to(device, non_blocking=True))
                        tot = tot + lm * (k_m / mic)
                        mntp_acc += float(lm.detach()) / n_micro
                        scaler.loss(tot / n_micro).backward()
                    aux_acc.update({"mntp_loss": mntp_acc, "mntp_p": p_m})
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
                if not is_oom(ex) or micro == 1:
                    raise
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                micro = max(1, micro // 2)
                while batch_seqs % micro:
                    micro -= 1
                micro_tokens = micro * T
                mic = micro if plan is None else micro_for(nb, Tw)
                k_m = LV.mntp_rows(mic, args.mntp_frac)
                config["micro_batch_size"] = micro
                write_json(out / "config.json", config)
                print(f"[train] OOM at step {st['step']}: micro batch -> {micro}", file=sys.stderr, flush=True)
        step = st["step"]
        f = lr_factor(step, total_steps, warmup, args.cooldown_frac, args.lr_floor)
        mfrac = min(1.0, step / args.muon_momentum_warmup_steps) if args.muon_momentum_warmup_steps > 0 else 1.0
        mom = (1 - mfrac) * args.muon_momentum_start + mfrac * args.muon_momentum
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * f
            if g.get("kind") == "muon" and "momentum" in g:
                g["momentum"] = mom
        if scaler.unscale_and_check(raw_model.parameters()):
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(raw_model)
        optimizer.zero_grad(set_to_none=True)
        if has_step_end:
            raw_model.on_step_end(step, total_steps)
        st["last_loss"] = float(loss_acc.item())
        sync(device)
        dt = time.time() - t0
        session_times.append(dt)
        st["raw_train_time"] += dt
        if 2 <= len(session_times) <= 11:
            new_c = max(0.0, session_times[0] - float(np.median(session_times[1:])))
            st["compile_s"] += new_c - session_compile
            session_compile = new_c
        st["train_time"] = st["raw_train_time"] - st["compile_s"]
        st["step"] += 1
        steps_this_session += 1
        st["tokens_seen"] += batch_seqs * T if plan is None else int((y_all >= 0).sum())
        st["words_seen"] += n_words_batch
        rec = {"step": st["step"], "tokens_seen": st["tokens_seen"], "words_seen": st["words_seen"],
               "epoch": epoch_now(), "loss": st["last_loss"], "lr_factor": f, "momentum": mom,
               "step_time": dt, "wallclock": time.time() - t_start}
        if lineage is not None:
            for k, v in (sampler.last_words if cyc is not None else {"real": n_words_batch}).items():
                st["words_by_kind"][k] += v
            rec.update({f"words_{k}": v for k, v in st["words_by_kind"].items()})
        tlog.row(rec)
        jlog.write(json.dumps({"type": "train", **rec, **aux_acc}) + "\n")
        if st["step"] % args.log_every == 0 or st["step"] == 1:
            print(f"[train] step {st['step']}/{total_steps} loss {st['last_loss']:.4f} words {st['words_seen']:,} "
                  f"lr_f {f:.3f} {dt * 1000:.0f}ms/step {batch_seqs * T / dt:,.0f} tok/s", flush=True)
        if not math.isfinite(st["last_loss"]):
            st["status"] = "diverged"
            print(f"[train] non-finite loss at step {st['step']}; stopping", file=sys.stderr, flush=True)
            break
        fired_now = False
        for m, thr in milestones:
            if m not in st["fired"] and st["words_seen"] >= thr:
                st["fired"].append(m)
                milestone(label_words(m), m)
                fired_now = True
        if fired_now and args.save:
            save_resume()
        if (args.max_session_steps is not None and steps_this_session >= args.max_session_steps
                and st["step"] < total_steps):
            st["status"] = "paused"
            save_resume()
            print(f"[train] paused at step {st['step']} (resume with --resume auto)", flush=True)
            break

    result: dict[str, Any] = {}
    if st["status"] == "running":
        st["status"] = "ok"
        pending = [(m, thr) for m, thr in milestones if m not in st["fired"]]
        if pending and st["words_seen"] >= END_FIRE_FRAC * pending[0][1]:
            m = pending[0][0]
            st["fired"].append(m)
            print(f"[train] end of run at {st['words_seen']:,} words >= {END_FIRE_FRAC:.0%} of milestone "
                  f"{milestone_label(m)}: firing it now", flush=True)
            milestone(label_words(m), m)
        if not evals or list(evals.values())[-1]["step"] != st["step"]:
            milestone("final", None)
        if args.save:
            save_resume()
        full_name = "test" if args.final else "dev"
        if not args.no_full_dev or args.final:
            full = C.load_split(full_name, args.data_dir, final=args.final)
            t0 = time.time()
            r, pt = evaluate_stream(ev_model, ev_raw, full, token_bytes, T, eval_bs, device, amp)
            st["eval_time"] += time.time() - t0
            if not args.no_pt:
                np.save(out / "pt" / f"pt_{full_name}full_final.npy", pt.astype(np.float16))
            result["full_eval"] = {"split": full_name, **{k: v for k, v in r.items()}}
            print(f"[eval] FULL {full_name}: {r['nats_per_word']:.4f} nats/word {r['nats_per_token']:.4f} nats/token "
                  f"{r['bpb']:.4f} bpb", flush=True)
    tlog.close()
    clog.close()
    wall = time.time() - t_start
    dev_min = dev_sub_minimum(out / "curves.csv")
    last = list(evals.values())[-1] if evals else {}
    result.update({
        "status": st["status"], "run_id": run_id, "tag": args.tag, "model": args.model, "size": args.size,
        "seed": args.seed, "data_seed": data_seed, "split": args.split, "epochs": args.epochs, "config": config,
        "git_commit": git_commit(), "params_total": params_total, "params_nonembedding": params_nonemb,
        "flops_per_token": fpt, "flops_source": fpt_src, "flops_per_token_harness": harness_flops_per_token(raw_model, cfg),
        "steps": st["step"], "total_steps": total_steps, "tokens_seen": st["tokens_seen"],
        "words_seen": st["words_seen"], "sessions": st["sessions"], "wallclock_s_session": wall,
        "train_time_s": st["train_time"], "raw_train_time_s": st["raw_train_time"], "compile_s": st["compile_s"],
        "eval_time_s": st["eval_time"], "tokens_per_s": st["tokens_seen"] / st["train_time"] if st["train_time"] else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "final_train_loss": st["last_loss"], "precision": precision,
        "fp16_skipped_steps": scaler.skipped if scaler.enabled else None,
        "data": config["data"],
        "eval_fingerprint": {"dev_sub": dev_sub.meta["fingerprint"], "dev_sub_tokens": dev_sub.n_tokens,
                             "dev_sub_words": dev_sub.n_words, "train_sub": train_sub.meta["fingerprint"],
                             "train_sub_tokens": train_sub.n_tokens, "train_sub_words": train_sub.n_words,
                             "eval_batch_size": eval_bs, "seq_len": T},
        "eval_split_final": bool(args.final),
        "last_milestone": last, "milestones_fired": sorted(st["fired"]),
        "dev_sub_min": dev_min,
        **mem_stats(device), })
    if lineage is not None:
        wk = dict(st["words_by_kind"])
        tot = st["words_seen"] + sum(lineage["extra_words"].values())
        result["exposure"] = {**lineage, "cycle_words": wk, "lineage_words": st["words_seen"], "total": tot,
                              "within_cap": tot <= args.exposure_cap, "warmup_steps": warmup}
        if args.exposure_rule != "r0":
            capped = CY.capped_words(args.exposure_rule, lineage["prior_words"], lineage["extra_words"], wk)
            result["exposure"].update({"capped_words": capped, "within_cap": capped <= args.exposure_cap})
    if st["status"] != "paused":
        write_json(out / "result.json", result)
    jlog.write(json.dumps({"type": "end", "status": st["status"], "step": st["step"]}) + "\n")
    jlog.close()
    print(f"[done] {st['status']} {st['step']}/{total_steps} steps, {st['words_seen']:,} words, {wall:.1f}s -> {out}",
          flush=True)
    return result


def cli() -> None:
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        args, _ = build_parser().parse_known_args()
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "error.txt").write_text(tb)
        sys.exit(1)


if __name__ == "__main__":
    cli()
