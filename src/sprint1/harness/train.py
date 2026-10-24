"""Sprint-1 training harness: one model, one seed, one budget -> sprint1/runs/<run_id>/{result.json, log.jsonl}.

    python -m sprint1.harness.train --model baseline --size small --seed 0 --tokens 40_000_000 \
        --out sprint1/runs/baseline_small_s0
    python -m sprint1.harness.train --model baseline --size tiny --steps 20 --synthetic --threads 1   # CPU smoke

Contract: sprint1/harness/model_api.py. The model module sprint1.models.<name> exposes build(cfg).
Backward on loss + aux["aux_loss"]; only `loss` is ever scored. Optimizer: Muon (2-D hidden matrices) +
AdamW (embed / head / scalar groups), unless the model (or its module) defines make_optimizer(model, cfg,
base_lrs), which is then used with base_lrs = {"muon", "adam_embed", "adam_head", "adam_scalar"}.
Schedule: linear warmup (--warmup_steps), constant, linear cooldown over the last --cooldown_frac of steps
to --lr_floor x base; Muon momentum warms up from --muon_momentum_start over --muon_momentum_warmup_steps.
Timing: train_time_s excludes evals and the first step's compile / warm-up overhead (compile_s = step-1 time
minus the median of steps 2..11); --time_budget_s counts train_time only.
Evaluation (DEV split; FINAL only with --final): val_loss_nats, val_bpb, and the new/seen split (target
token absent / present earlier in the same window), see evaluate(). Eval discipline (sprint1/audit/eval_discipline.md
B3-B5, B10-B12): one fixed --eval_batch_size for every arm (never touched by the OOM handler), windows interleaved
across batch rows, no ragged batch (padding rows score nothing), buffers snapshotted / restored around every eval
and reset_state() before every eval batch, --final refused on a dirty tree, the train shard pool pinned and recorded.
"""
from __future__ import annotations

import argparse
import contextlib
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

from sprint1.harness import data as D
from sprint1.harness.model_api import SIZE_LADDER, ModelConfig
from sprint1.harness.optim import build_optimizer

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_BATCH_TOKENS = 524288  # only used with --lr_batch_scaling sqrt
# Defaults = the baseline tuner's anchor "s1_base" (docs/sprint_1_baseline_tuning.md): effective LRs at a
# 65,536-token batch, no batch scaling (no speedrun record cut Adam LRs when it cut the batch).

# Fixed eval batch (sequences per eval forward) for EVERY arm and size (B5): independent of the micro batch and never
# changed by the OOM handler. CPU default stays small so the smoke tests are fast.
EVAL_BATCH_DEFAULT = {"cuda": 32, "cpu": 8}
PROVISIONAL_TOTAL_SENTINEL = 10 ** 9  # on_step_end total before 2 post-compile step times exist (fractions ~ 0)

# Per-size harness defaults (model shapes come from model_api.SIZE_LADDER).
SIZE_DEFAULTS: dict[str, dict[str, Any]] = {
    "tiny":   dict(batch_tokens=2048,   micro_batch_size=8,  eval_tokens=8192,   steps=20),
    "small":  dict(batch_tokens=65536,  micro_batch_size=16, eval_tokens=524288),
    "medium": dict(batch_tokens=65536,  micro_batch_size=8,  eval_tokens=524288),
    "xs":     dict(batch_tokens=65536,  micro_batch_size=32, eval_tokens=524288),
    "large":  dict(batch_tokens=65536,  micro_batch_size=16, eval_tokens=524288),  # A100 40GB; the OOM handler halves it
}


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def parse_value(v: str) -> Any:
    """json-ish typing for --extra values: ints, floats, true/false, none/null, JSON lists/dicts, else str."""
    s = v.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s[:1] in "[{\"":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def parse_extra(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--extra expects key=val, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = parse_value(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="module name under sprint1/models (build(cfg))")
    ap.add_argument("--size", default="small", choices=sorted(SIZE_LADDER))
    ap.add_argument("--seed", type=int, default=0)
    b = ap.add_argument_group("budget")
    b.add_argument("--tokens", type=int, default=None, help="training tokens (steps = ceil(tokens / batch))")
    b.add_argument("--steps", type=int, default=None)
    b.add_argument("--time_budget_s", type=float, default=None,
                   help="training-time budget (evals excluded). Alone: steps are fixed after --calib_steps "
                        "from the measured step time. With --steps/--tokens: a hard stop.")
    b.add_argument("--calib_steps", type=int, default=10)
    b.add_argument("--batch_tokens", type=int, default=None)
    b.add_argument("--micro_batch_size", type=int, default=None, help="sequences per forward (grad accumulation)")
    o = ap.add_argument_group("optimizer")
    o.add_argument("--muon_lr", type=float, default=0.065)
    o.add_argument("--muon_momentum", type=float, default=0.95)
    o.add_argument("--muon_momentum_start", type=float, default=0.85)
    o.add_argument("--muon_momentum_warmup_steps", type=int, default=150)
    o.add_argument("--muon_wd", type=float, default=0.0)
    o.add_argument("--lr_embed", type=float, default=0.78, help="Adam LR for embedding / value-embedding tables")
    o.add_argument("--lr_head", type=float, default=0.0208, help="Adam LR for the lm head")
    o.add_argument("--lr_scalar", type=float, default=0.052, help="Adam LR for scalars/norms")
    o.add_argument("--lr_batch_scaling", default="none", choices=["sqrt", "none"],
                   help="sqrt: Adam LRs x (batch_tokens / ref_batch_tokens)^0.5 (Muon LR is not scaled)")
    o.add_argument("--ref_batch_tokens", type=int, default=REF_BATCH_TOKENS)
    o.add_argument("--adam_beta1", type=float, default=0.8)
    o.add_argument("--adam_beta2", type=float, default=0.95)
    o.add_argument("--adam_eps", type=float, default=1e-10)
    o.add_argument("--adam_wd", type=float, default=0.0)
    o.add_argument("--warmup_steps", type=int, default=0)
    o.add_argument("--cooldown_frac", type=float, default=0.5)
    o.add_argument("--lr_floor", type=float, default=0.1, help="final LR as a fraction of base")
    o.add_argument("--grad_clip", type=float, default=0.0, help="0 = off (speedrun default)")
    m = ap.add_argument_group("model overrides")
    for k, t in (("n_layer", int), ("n_head", int), ("n_embd", int), ("seq_len", int), ("mlp_ratio", float),
                 ("vocab_size", int)):
        m.add_argument(f"--{k}", type=t, default=None)
    m.add_argument("--extra", action="append", nargs="+", default=None, metavar="KEY=VAL",
                   help="cfg.extra entries (repeatable): ints, floats, true/false, none, JSON, strings")
    e = ap.add_argument_group("evaluation")
    e.add_argument("--eval_every", type=int, default=-1, help="steps between DEV-slice evals; -1 = total/8; 0 = off")
    e.add_argument("--eval_tokens", type=int, default=None, help="tokens in the fixed DEV slice for curve evals")
    e.add_argument("--final_eval_tokens", type=int, default=-1, help="-1 = the whole split")
    e.add_argument("--eval_batch_size", type=int, default=None,
                   help=f"sequences per eval forward, the same for every arm (default {EVAL_BATCH_DEFAULT['cuda']} on "
                        f"CUDA, {EVAL_BATCH_DEFAULT['cpu']} on CPU); an eval OOM fails the run, it is never shrunk")
    e.add_argument("--eval_eager", action="store_true",
                   help="evaluate (curve, probe, final) on the uncompiled module; training stays compiled")
    e.add_argument("--no_eval_probe", action="store_true",
                   help="CUDA: skip the one-batch eval probe after step 1 (it fails fast if --eval_batch_size OOMs)")
    e.add_argument("--no_split", action="store_true", help="skip the new/seen split in the final eval")
    e.add_argument("--final", action="store_true", help="evaluate on the FINAL split (protected!)")
    e.add_argument("--save_per_token", action="store_true",
                   help="final eval also writes <out>/per_token.npz (per-target loss/target/window/pos/seen, see "
                        "PerTokenRecorder) for sprint1/measure/slices_from_run.py")
    r = ap.add_argument_group("run")
    r.add_argument("--out", default=None, help="run dir (default sprint1/runs/<run_id>)")
    r.add_argument("--run_id", default=None)
    r.add_argument("--synthetic", action="store_true", help="deterministic fake corpus, no network")
    r.add_argument("--data_seed", type=int, default=None, help="train-batch order seed (default: --seed)")
    r.add_argument("--data_dir", default=None)
    r.add_argument("--train_data_dir", default=None, help="train shards from another dir (e.g. FineWeb-Edu); DEV/FINAL stay in --data_dir")
    r.add_argument("--train_shards", type=int, default=None,
                   help="train pool = the first N fineweb_train_*.bin in sorted order (fewer present: error). Default: "
                        "the number present (the VM downloads exactly job.json's train_shards); always recorded")
    r.add_argument("--train_tokens_limit", type=int, default=None,
                   help="train pool = the first T tokens of the loaded train shards concatenated in sorted order (one "
                        "stream of shards serves every budget as a prefix); windows are drawn uniformly from that "
                        "prefix and never read past it. Default: the whole pool; recorded")
    r.add_argument("--device", default=None)
    r.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU)")
    r.add_argument("--compile", action="store_true")
    r.add_argument("--profile", action="store_true", help="torch.profiler over --profile_steps -> profile.json/md")
    r.add_argument("--profile_start", type=int, default=5)
    r.add_argument("--profile_steps", type=int, default=5)
    r.add_argument("--peak_tflops", type=float, default=121.0, help="for MFU (L4 dense bf16 = 121)")
    r.add_argument("--sampler", default="replace", choices=["replace", "epoch"],
                   help="replace (default): uniformly random windows with replacement (TrainSampler). epoch: one pass "
                        "without replacement over the pool's stride-seq_len windows in a seeded order (EpochSampler); "
                        "more windows than the pool holds is an error. Use --train_tokens_limit = "
                        "data.epoch_pool_tokens(D) = steps x batch_tokens + 1 for rho = 1")
    r.add_argument("--scaling_meta", default=None,
                   help="JSON object (or @file) written to <out>/scaling.json for the scaling-slope judge, with rho, "
                        "train_pool_tokens, size and budget_tokens filled in from this run")
    r.add_argument("--scaling_meta_file", default=None,
                   help="JSON object file merged UNDER --scaling_meta (e.g. the data pipeline's per-cell fields "
                        "<arm dir>/scaling_meta/<size>_<budget>.json); implies writing scaling.json")
    r.add_argument("--save", action="store_true", help="checkpoint model+optimizer+sampler+RNG+source at the end")
    r.add_argument("--save_weights", action="store_true",
                   help="write <out>/weights.pt at the end: model state_dict + cfg only (no optimizer), for a later "
                        "eval-only pass (sprint1.harness.eval_weights, e.g. FINAL once at the end)")
    r.add_argument("--weights_dtype", default="fp32", choices=["fp32", "bf16"],
                   help="--save_weights tensor dtype for floating-point entries (fp32 = the exact trained weights)")
    r.add_argument("--log_every", type=int, default=10, help="stdout progress cadence (the JSONL logs every step)")
    r.add_argument("--tag", default="", help="free-form label stored in result.json")
    return ap


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def lr_factor(step: int, total: int | None, warmup: int, cooldown_frac: float, floor: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total is None or cooldown_frac <= 0:
        return 1.0
    cd = max(1, int(round(cooldown_frac * total)))
    start = total - cd
    if step < start:
        return 1.0
    w = max(0.0, (total - step) / cd)  # 1 -> 0 across the cooldown
    return floor + (1.0 - floor) * w


def seen_mask(idx: torch.Tensor, targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """seen[b, t] = targets[b, t] occurs in idx[b, 0..t] (the visible context). Ignored targets -> False."""
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device).expand(B, T)
    first = torch.full((B, vocab_size), T, dtype=torch.long, device=idx.device)
    first.scatter_reduce_(1, idx, pos, reduce="amin", include_self=True)
    f = first.gather(1, targets.clamp_min(0))
    return (f <= pos) & (targets >= 0)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    emb_ids = {id(mod.weight) for mod in model.modules() if isinstance(mod, nn.Embedding)}
    emb = sum(p.numel() for n, p in model.named_parameters()
              if id(p) in emb_ids or any(k in n.lower() for k in ("embed", "wte", "wpe", "lm_head", "unembed")))
    return total, total - emb


def harness_flops_per_token(model: nn.Module, cfg: ModelConfig) -> float:
    """The harness's own estimate (B12): 6 x non-embedding matrix params + 6 x n_layer x seq_len x n_embd."""
    emb_ids = {id(mod.weight) for mod in model.modules() if isinstance(mod, nn.Embedding)}
    n_mat = sum(p.numel() for p in model.parameters() if p.ndim >= 2 and id(p) not in emb_ids)
    return 6.0 * n_mat + 6.0 * cfg.n_layer * cfg.seq_len * cfg.n_embd


def estimate_flops(model: nn.Module, cfg: ModelConfig) -> tuple[float, str]:
    if hasattr(model, "flops_per_token"):
        try:
            return float(model.flops_per_token()), "model"
        except Exception as ex:  # noqa: BLE001
            print(f"[train] flops_per_token() failed ({ex!r}); using estimate", file=sys.stderr)
    return harness_flops_per_token(model, cfg), "estimate"


def provisional_total_steps(step: int, step_times: list[float], time_budget_s: float) -> int:
    """total_steps handed to on_step_end during --time_budget_s calibration (never -1): time_budget_s / median of
    the post-compile step times so far (>= 2 needed, else PROVISIONAL_TOTAL_SENTINEL), at least step + 1."""
    post = step_times[1:]
    if len(post) < 2:
        return PROVISIONAL_TOTAL_SENTINEL
    return max(step + 1, int(time_budget_s / max(float(np.median(post)), 1e-9)))


def git_commit() -> str:
    env = os.environ.get("HINTON_GIT_COMMIT")
    if env:
        return env
    f = REPO_ROOT / "sprint1" / "GIT_COMMIT"
    if f.exists():
        return f.read_text().strip()
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "sprint1/models",
                                "sprint1/harness"], capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("+dirty" if dirty else "") if sha else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def scalarize(aux: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(aux, dict):
        return out
    for k, v in aux.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                out[k] = float(v.detach().float().item())
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def mem_stats(device: torch.device) -> dict[str, int]:
    s = {"host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}
    if device.type == "cuda":
        s["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated(device))
        s["peak_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    else:
        s["peak_mem_bytes"] = s["host_peak_rss_bytes"]
    return s


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def make_cfg(args) -> ModelConfig:
    d = dict(SIZE_LADDER[args.size])
    if args.size == "tiny" and args.synthetic and args.vocab_size is None:
        d["vocab_size"] = 1024  # CPU smoke: keep the logits small (pass --vocab_size 50304 for the real head)
    for k in ("n_layer", "n_head", "n_embd", "seq_len", "mlp_ratio", "vocab_size"):
        v = getattr(args, k)
        if v is not None:
            d[k] = v
    extra = parse_extra([x for group in (args.extra or []) for x in group])
    return ModelConfig(extra=extra, **d)


# ------------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------------
PT_REL_TOL = 1e-3  # |mean(per_token_loss[valid]) - loss| <= PT_REL_TOL * max(1, |loss|), every batch


def pt_mean_diff(pt: torch.Tensor, valid: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    """|mean(per_token_loss over valid targets) - scored batch loss| (0-d float64 tensor, no host sync)."""
    return (pt.double()[valid].mean() - loss.detach().double()).abs()


def snapshot_state(module: nn.Module) -> dict[str, Any]:
    """Before an eval (B3): clones of every buffer, and (version counter, data_ptr) of every parameter."""
    return {"buffers": {n: b.detach().clone() for n, b in module.named_buffers()},
            "params": {n: (p._version, p.data_ptr()) for n, p in module.named_parameters()}}


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    if torch.equal(a, b):
        return True
    if a.is_floating_point() or a.is_complex():  # NaN != NaN: equal iff same NaN pattern and equal elsewhere
        na, nb = a.isnan(), b.isnan()
        return bool(torch.equal(na, nb)) and torch.equal(a.masked_fill(na, 0), b.masked_fill(nb, 0))
    return False


def restore_state(module: nn.Module, snap: dict[str, Any]) -> list[str]:
    """After an eval: restore every snapshotted buffer that changed (in place when possible) and return the sorted
    names of what changed: buffer names, "+<name>" for a buffer created during eval (left in place), and
    "param:<name>" for a parameter modified in place or replaced (cannot be restored; the run is suspect)."""
    changed: list[str] = []
    cur = dict(module.named_buffers())
    for name, saved in snap["buffers"].items():
        b = cur.get(name)
        if b is not None and _bit_equal(b, saved):
            continue
        changed.append(name)
        if b is not None and b.shape == saved.shape and b.dtype == saved.dtype and b.device == saved.device:
            b.copy_(saved)
        else:
            mod_name, _, leaf = name.rpartition(".")
            try:
                sub = module.get_submodule(mod_name) if mod_name else module
                sub._buffers[leaf] = saved
            except (AttributeError, KeyError) as ex:
                print(f"[eval] could not restore buffer {name}: {ex!r}", file=sys.stderr, flush=True)
    changed += ["+" + n for n in cur if n not in snap["buffers"]]
    changed += ["param:" + n for n, p in module.named_parameters()
                if snap["params"].get(n) != (p._version, p.data_ptr())]
    return sorted(changed)


class PerTokenRecorder:
    """--save_per_token: every scored target of the final eval, in DEV order (window-major, whatever the eval batch
    layout), -> <out>/per_token.npz with loss (float16 nats), target (int32), window (int32 eval-window index),
    pos (int16 position in the window),
    seen (bool: seen_mask, the new/seen split's own definition) and meta (a JSON string).
    Losses: aux["per_token_loss"] where evaluate() uses it (with --no_split: where its batch mean matches the
    scored loss); otherwise raw_model.forward_logits(idx), CE per sequence in fp32, moved to the CPU at once.
    Every batch's mean must match the scored batch loss (1e-3 / 2e-3 relative, as evaluate()'s checks). If it
    does not, or the model has neither source, nothing is written and the reason goes to per_token_error.txt.
    Padding rows (window -1, all targets ignored) contribute nothing.
    The scored loss and the split are untouched; the forward_logits path costs one extra forward per batch."""

    def __init__(self, n_targets: int):
        self.loss = np.zeros(n_targets, np.float16)
        self.target = np.zeros(n_targets, np.int32)
        self.window = np.zeros(n_targets, np.int32)
        self.pos = np.zeros(n_targets, np.int16)
        self.seen = np.zeros(n_targets, np.bool_)
        self.filled = np.zeros(n_targets, np.bool_)
        self.n = self.n_win = 0
        self.nats = 0.0
        self.max_diff = 0.0
        self.methods: set[str] = set()
        self.error: str | None = None

    def add(self, raw_model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, aux: Any,
            use_pt: bool | None, amp, reset: bool, vocab_size: int, win: np.ndarray | None = None) -> None:
        """win: DEV window index of each row (-1 = padding row); None = consecutive windows from n_win."""
        if self.error:
            return
        B, T = y.shape
        if win is None:
            win = np.arange(self.n_win, self.n_win + B, dtype=np.int64)
        valid = y >= 0
        lv = float(loss.item())
        pt = aux.get("per_token_loss") if isinstance(aux, dict) else None
        tol = PT_REL_TOL * max(1.0, abs(lv))
        if use_pt is None and isinstance(pt, torch.Tensor) and pt.shape == y.shape and valid.any():
            use_pt = pt_mean_diff(pt, valid, loss).item() <= tol
        if use_pt:
            ce, method = pt.float(), "per_token_loss"
        elif callable(getattr(raw_model, "forward_logits", None)):
            if reset:
                raw_model.reset_state()
            with amp:
                logits = raw_model.forward_logits(x)
            if not (isinstance(logits, torch.Tensor) and logits.ndim == 3 and logits.shape[:2] == y.shape):
                self.error = f"forward_logits returned {type(logits).__name__}, not a [B, T, V] tensor"
                return
            ce = torch.stack([nn.functional.cross_entropy(logits[b].float(), y[b], ignore_index=-1, reduction="none")
                              for b in range(B)])
            del logits
            method, tol = "forward_logits", 2e-3 * max(1.0, abs(lv))
        else:
            self.error = "no per-token source: no usable aux['per_token_loss'] and no forward_logits(idx)"
            return
        if valid.any():
            diff = pt_mean_diff(ce, valid, loss).item()
            self.max_diff = max(self.max_diff, diff)
            if diff > tol:
                self.error = (f"{method} batch mean differs from the scored loss by {diff:.3g} > {tol:.3g} "
                              f"(windows {win.tolist()[:4]}...)")
                return
        self.methods.add(method)
        w = torch.from_numpy(np.asarray(win, dtype=np.int64))
        v = valid.cpu() & (w >= 0).unsqueeze(1)
        k = int(v.sum())
        wv = w.unsqueeze(1).expand(B, T)[v]
        pv = torch.arange(T).expand(B, T)[v]
        at = (wv * T + pv).numpy()  # DEV-order slot of each scored target
        if k and (at.max() >= len(self.loss) or self.filled[at].any()):
            self.error = f"target slot out of range or written twice (windows {win.tolist()[:4]}...)"
            return
        c = ce.detach().float().cpu()[v]
        self.nats += float(c.double().sum())
        self.loss[at] = c.numpy()
        self.target[at] = y.cpu()[v].numpy()
        self.window[at] = wv.numpy()
        self.pos[at] = pv.numpy()
        self.seen[at] = seen_mask(x, y, vocab_size).cpu()[v].numpy()
        self.filled[at] = True
        self.n += k
        self.n_win += int((w >= 0).sum())

    def save(self, out: Path, meta: dict[str, Any]) -> None:
        if self.error:
            print(f"[eval] --save_per_token: {self.error}; per_token.npz not written", file=sys.stderr, flush=True)
            (out / "per_token_error.txt").write_text(self.error + "\n")
            (out / "per_token.npz").unlink(missing_ok=True)  # never leave a stale file from an earlier run
            return
        n = self.n
        sel = slice(None) if self.filled.all() else self.filled  # DEV order; unfilled slots = unscored targets
        meta = {**meta, "n_targets": n, "n_windows": self.n_win, "method": "+".join(sorted(self.methods)),
                "mean_loss_fp32": self.nats / max(n, 1), "max_abs_batch_mean_diff": self.max_diff,
                "order": "dev"}
        tmp = out / "per_token.tmp.npz"
        np.savez_compressed(tmp, loss=self.loss[sel], target=self.target[sel], window=self.window[sel],
                            pos=self.pos[sel], seen=self.seen[sel], meta=np.array(json.dumps(meta, default=str)))
        tmp.replace(out / "per_token.npz")
        print(f"[eval] per_token.npz: {n:,} targets ({meta['method']}) -> {out / 'per_token.npz'}", flush=True)


@torch.no_grad()
def evaluate(model: nn.Module, raw_model: nn.Module, tokens: np.ndarray, token_bytes: torch.Tensor, cfg: ModelConfig,
             max_tokens: int | None, batch: int, device: torch.device, amp, split: bool,
             per_token: PerTokenRecorder | None = None) -> dict[str, Any]:
    """Mean CE (nats/token) over every target of the slice's non-overlapping windows, bits per byte, and
    (split=True) the new/seen split. Per-position losses come from aux["per_token_loss"] when the model
    provides it (verified against the masked recomputation on the first batch, and its mean against the scored
    loss on every batch); otherwise a second forward with seen targets masked to -1 gives the new-token loss, and
    the seen-token loss is the remainder.
    Discipline (eval_discipline.md B3-B5): batches of exactly `batch` rows in data.EvalLayout order (windows
    interleaved across rows; padding rows have all targets -1 and score nothing); reset_state() before every
    forward if the model defines it; buffers snapshotted before and restored after (what changed is returned in
    eval_state_mutated); the scored loss must not require grad; the train/eval mode is restored. A CUDA OOM is
    re-raised as a RuntimeError naming the batch size (the harness never shrinks it)."""
    lay = D.eval_layout(len(tokens), cfg.seq_len, max_tokens, batch)
    was_training = model.training
    reset = callable(getattr(raw_model, "reset_state", None))
    snap = snapshot_state(raw_model)
    z = lambda: torch.zeros((), dtype=torch.float64, device=device)  # noqa: E731
    nats, ntok, nbytes, nats_new, n_new = z(), z(), z(), z(), z()
    pt_diff = z()  # max over batches of |mean(per_token_loss) - loss| / max(1, |loss|), where the split uses it
    n_pt_checked = 0
    use_pt: bool | None = None
    model.eval()
    try:
        for x, y, win in D.eval_batches(tokens, cfg.seq_len, max_tokens, batch):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if reset:
                raw_model.reset_state()
            with amp:
                loss, aux = model(x, y)
            if loss.requires_grad or torch.is_grad_enabled():
                raise RuntimeError("[eval] the scored loss requires grad / grad is enabled inside evaluate(): "
                                   "evaluation must not build a graph or train (B2/B3)")
            valid = y >= 0
            nv = valid.sum()
            nats += loss.double() * nv
            ntok += nv
            nbytes += token_bytes[y.clamp_min(0)][valid].sum()
            if not split:
                if per_token is not None:
                    per_token.add(raw_model, x, y, loss, aux, None, amp, reset, cfg.vocab_size, win)
                continue
            new = valid & ~seen_mask(x, y, cfg.vocab_size)
            pt = aux.get("per_token_loss") if isinstance(aux, dict) else None
            if use_pt is None:
                use_pt = False
                if isinstance(pt, torch.Tensor) and pt.shape == y.shape:
                    ok_mean = pt_mean_diff(pt, valid, loss).item() <= PT_REL_TOL * max(1.0, abs(loss.item()))
                    if ok_mean and new.any():
                        if reset:
                            raw_model.reset_state()
                        with amp:
                            l_new, _ = model(x, y.masked_fill(~new, -1))
                        ok_new = abs(pt.double()[new].mean().item() - l_new.item()) <= 2e-3 * max(1.0, abs(l_new.item()))
                        use_pt = ok_new
                    else:
                        use_pt = ok_mean
                    if not use_pt:
                        print("[eval] per_token_loss inconsistent with loss; using two-pass split", file=sys.stderr)
            if use_pt:
                if not (isinstance(pt, torch.Tensor) and pt.shape == y.shape):
                    raise RuntimeError(f"[eval] aux['per_token_loss'] missing or mis-shaped on a later batch "
                                       f"(windows {win.tolist()[:4]}...)")
                pt_diff = torch.maximum(pt_diff, pt_mean_diff(pt, valid, loss) / loss.detach().double().abs().clamp_min(1.0))
                n_pt_checked += 1
                nats_new += pt.double()[new].sum()
            elif new.any():
                if reset:
                    raw_model.reset_state()
                with amp:
                    l_new, _ = model(x, y.masked_fill(~new, -1))
                nats_new += l_new.double() * new.sum()
            n_new += new.sum()
            if per_token is not None:
                per_token.add(raw_model, x, y, loss, aux, use_pt, amp, reset, cfg.vocab_size, win)
    except torch.cuda.OutOfMemoryError as ex:
        raise RuntimeError(f"[eval] CUDA OOM at eval_batch_size={batch} (seq_len {cfg.seq_len}). The harness never "
                           f"changes the eval batch: rerun with a smaller --eval_batch_size, the SAME for every arm "
                           f"of the comparison") from ex
    finally:
        mutated = restore_state(raw_model, snap)
        model.train(was_training)
    if model.training != was_training or raw_model.training != was_training:
        raise RuntimeError(f"[eval] train mode not restored (want {was_training})")
    if mutated:
        print(f"[eval] state changed during eval and was restored (B3): {mutated}", file=sys.stderr, flush=True)
    nats_f, ntok_f, nbytes_f = nats.item(), ntok.item(), nbytes.item()
    out: dict[str, Any] = {
        "val_loss_nats": nats_f / max(ntok_f, 1.0),
        "val_bpb": nats_f / (max(nbytes_f, 1.0) * math.log(2)),
        "eval_tokens": int(ntok_f),
        "eval_bytes": int(nbytes_f),
        "eval_batch_size": batch,
        "eval_interleave": lay.interleave,
        "eval_batches": lay.n_batches,
        "eval_padded_rows": lay.padded_rows,
        "eval_reset_state": reset,
        "eval_state_mutated": mutated,
    }
    if split:
        nn_, nnew = nats_new.item(), n_new.item()
        ns_, nseen = nats_f - nn_, ntok_f - nnew
        out["val_split"] = {
            "new": {"loss": nn_ / nnew if nnew else float("nan"), "frac_positions": nnew / max(ntok_f, 1.0),
                    "frac_bits": nn_ / nats_f if nats_f else float("nan")},
            "seen": {"loss": ns_ / nseen if nseen else float("nan"), "frac_positions": nseen / max(ntok_f, 1.0),
                     "frac_bits": ns_ / nats_f if nats_f else float("nan")},
            "method": "per_token_loss" if use_pt else "two_pass",
        }
        if use_pt:
            md = pt_diff.item()
            out["val_split"]["per_token_check"] = {"batches": n_pt_checked, "max_rel_diff": md,
                                                   "ok": md <= PT_REL_TOL}
            if md > PT_REL_TOL:
                print(f"[eval] WARNING: per_token_loss batch mean differs from the scored loss by {md:.3g} (relative) "
                      f"on some batch > {PT_REL_TOL}: the new/seen split of this run is NOT valid", file=sys.stderr,
                      flush=True)
    return out


def run_eval_probe(model: nn.Module, raw_model: nn.Module, tokens: np.ndarray, token_bytes: torch.Tensor,
                   cfg: ModelConfig, batch: int, device: torch.device, amp) -> dict[str, Any]:
    """Fail fast if the fixed eval batch does not fit (B5: it is never shrunk): one evaluate() batch of `batch`
    DEV windows, run by main() after the first optimizer step on CUDA, outside train_time. evaluate() restores the
    buffers and the train mode; the torch RNG (CPU and CUDA) is restored here, so training is unchanged."""
    t0 = time.time()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        evaluate(model, raw_model, tokens, token_bytes, cfg, batch * cfg.seq_len, batch, device, amp, split=False)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    return {"ok": True, "eval_batch_size": batch, "s": time.time() - t0,
            "peak_mem_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    t_start = time.time()
    if args.final:  # B10: FINAL is touched only by a frozen commit
        commit = git_commit()
        if commit.endswith("+dirty") or commit == "unknown":
            raise SystemExit(f"--final refused: git commit is {commit!r}; FINAL may only be scored from a clean, "
                             f"committed tree (sprint1/models and sprint1/harness). Commit first.")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext())

    sd = SIZE_DEFAULTS[args.size]
    cfg = make_cfg(args)
    T = cfg.seq_len
    batch_tokens = args.batch_tokens or sd["batch_tokens"]
    batch_seqs = max(1, batch_tokens // T)
    micro = min(args.micro_batch_size or sd["micro_batch_size"], batch_seqs)
    while batch_seqs % micro:
        micro -= 1
    eval_tokens = args.eval_tokens or sd["eval_tokens"]
    eval_bs = args.eval_batch_size or EVAL_BATCH_DEFAULT["cuda" if device.type == "cuda" else "cpu"]
    if eval_bs < 1:
        raise SystemExit(f"--eval_batch_size must be >= 1, got {eval_bs}")

    run_id = args.run_id or f"{args.model}_{args.size}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(args.out) if args.out else REPO_ROOT / "sprint1" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    if args.final:
        bar = "!" * 78
        print(f"\n{bar}\n!!! --final: this run will be scored on the protected FINAL split.\n"
              f"!!! Use only for the frozen, pre-registered final comparison. This use is recorded in\n"
              f"!!! result.json (eval_split=final) and {out / 'FINAL_SPLIT_USED'}.\n{bar}\n",
              file=sys.stderr, flush=True)
        (out / "FINAL_SPLIT_USED").write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                          "argv": sys.argv}) + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_shards_default = False
    if not args.synthetic and args.train_shards is None:  # B11: pin the pool to an explicit, recorded N
        args.train_shards = len(D.train_shard_names(args.train_data_dir or args.data_dir)) or None
        train_shards_default = True
    corpus = D.load_corpus(args.data_dir, synthetic=args.synthetic, vocab_size=cfg.vocab_size,
                           train_shards=args.train_shards, train_data_dir=args.train_data_dir)
    train_pool_tokens = int(sum(len(t) for t in corpus.train))
    if args.train_tokens_limit is not None:  # prefix of the pool (default None: the pool is untouched)
        corpus.train, train_pool_tokens = D.prefix_pool(corpus.train, args.train_tokens_limit, T,
                                                        drop_short=args.sampler == "replace")
        print(f"[data] --train_tokens_limit {args.train_tokens_limit:,}: train pool = first {train_pool_tokens:,} "
              f"tokens ({len(corpus.train)} shard piece(s))", flush=True)
    train_shards_used = corpus.train_shards_used()
    print(f"[data] train pool: {len(train_shards_used)} shard(s)"
          + (" (--train_shards default = shards present)" if train_shards_default else "")
          + f": {[x['name'] for x in train_shards_used]}", flush=True)
    token_bytes = torch.as_tensor(corpus.token_bytes, dtype=torch.int64, device=device)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    sampler = (D.EpochSampler(corpus.train, T, data_seed) if args.sampler == "epoch"
               else D.TrainSampler(corpus.train, T, data_seed))
    scaling_meta = None
    if args.scaling_meta is not None or args.scaling_meta_file is not None:  # parse before training: fail fast
        scaling_meta = {}
        for sm in (("@" + args.scaling_meta_file) if args.scaling_meta_file else None, args.scaling_meta):
            if sm is None:
                continue
            part = json.loads(Path(sm[1:]).read_text() if sm.startswith("@") else sm)
            if not isinstance(part, dict):
                raise SystemExit("--scaling_meta / --scaling_meta_file must hold a JSON object")
            scaling_meta.update(part)  # the inline --scaling_meta wins over the file

    mod = importlib.import_module(f"sprint1.models.{args.model}")
    raw_model = mod.build(cfg).to(device)
    raw_model.train()
    model = torch.compile(raw_model) if args.compile else raw_model
    eval_model = raw_model if args.eval_eager else model  # --eval_eager: eval path skips torch.compile
    params_total, params_nonemb = count_params(raw_model)
    fpt, fpt_src = estimate_flops(raw_model, cfg)
    fpt_harness = harness_flops_per_token(raw_model, cfg)

    scale = (batch_seqs * T / args.ref_batch_tokens) ** 0.5 if args.lr_batch_scaling == "sqrt" else 1.0
    base_lrs = {"muon": args.muon_lr, "adam_embed": args.lr_embed * scale, "adam_head": args.lr_head * scale,
                "adam_scalar": args.lr_scalar * scale}
    mk = getattr(raw_model, "make_optimizer", None) or getattr(mod, "make_optimizer", None)
    if mk is not None:
        # forward the CLI optimizer knobs to custom optimizers (explicit --extra entries win)
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
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", g["lr"])

    # budget
    total_steps: int | None
    if args.steps is not None:
        total_steps = args.steps
    elif args.tokens is not None:
        total_steps = max(1, math.ceil(args.tokens / (batch_seqs * T)))
    elif args.time_budget_s is not None:
        total_steps = None  # fixed after calibration
    elif "steps" in sd:
        total_steps = sd["steps"]
    else:
        raise SystemExit("give --steps, --tokens or --time_budget_s")
    if args.sampler == "epoch" and total_steps is not None and total_steps * batch_seqs > sampler.n:
        raise SystemExit(f"--sampler epoch: {total_steps} steps x {batch_seqs} windows = {total_steps * batch_seqs:,} > "
                         f"{sampler.n:,} windows in the pool ({sampler.pool_tokens:,} tokens); --train_tokens_limit "
                         f"{total_steps * batch_seqs * T + 1} gives exactly one pass")
    eval_every = args.eval_every
    if eval_every < 0:
        eval_every = max(1, (total_steps or 1000) // 8)

    config = {"args": vars(args), "model_cfg": cfg.to_dict(), "batch_seqs": batch_seqs, "micro_batch_size": micro,
              "batch_tokens": batch_seqs * T, "base_lrs": base_lrs, "lr_scale": scale, "optimizer": opt_src,
              "total_steps": total_steps, "eval_every": eval_every, "eval_tokens": eval_tokens,
              "data": corpus.source, "data_seed": data_seed, "device": str(device), "torch": torch.__version__,
              "precision": "bf16-autocast" if device.type == "cuda" else "fp32", "compile": bool(args.compile),
              "eval_batch_size": eval_bs, "eval_interleave": True, "train_shards": len(train_shards_used),
              "train_shards_default": train_shards_default, "train_shards_used": train_shards_used,
              "train_tokens_limit": args.train_tokens_limit, "train_pool_tokens": train_pool_tokens,
              "sampler": args.sampler, "n_windows": getattr(sampler, "n", None)}
    write_json(out / "config.json", config)
    print(f"[train] {args.model} size={args.size} params={params_total:,} non-emb={params_nonemb:,} "
          f"flops/token={fpt:.3e} ({fpt_src}) batch={batch_seqs}x{T} micro={micro} steps={total_steps} "
          f"optimizer={opt_src} device={device} out={out}", flush=True)

    prof = None
    if args.profile:
        from sprint1.harness.profile import StepProfiler
        prof = StepProfiler(raw_model, out, start=args.profile_start, n_steps=args.profile_steps, device=device,
                            flops_per_token=fpt, tokens_per_step=batch_seqs * T, peak_tflops=args.peak_tflops)
    region = prof.region if prof else (lambda name: contextlib.nullcontext())

    log_f = open(out / "log.jsonl", "a", buffering=1)
    curve: list[dict[str, Any]] = []
    reset = cfg.reset_state_on_doc and hasattr(raw_model, "reset_state")
    has_step_end = hasattr(raw_model, "on_step_end")
    raw_train_time = 0.0  # sum of step times, including the first step's compile / CUDA warm-up overhead
    compile_s = 0.0       # first step time minus the median of steps 2..11; excluded from train_time
    train_time = 0.0
    eval_time = 0.0
    step_times: list[float] = []
    status = "ok"
    step = 0
    tokens_seen = 0
    last_loss = float("nan")
    provisional_seq: list[int] = []  # totals handed to on_step_end during --time_budget_s calibration
    curve_mutated: set[str] = set()
    eval_probe: dict[str, Any] | None = None

    def run_eval(tag: str) -> dict[str, Any]:
        nonlocal eval_time
        t0 = time.time()
        r = evaluate(eval_model, raw_model, corpus.dev, token_bytes, cfg, eval_tokens, eval_bs, device, amp, split=False)
        eval_time += time.time() - t0
        curve_mutated.update(r["eval_state_mutated"])
        pt = {"step": step, "tokens": tokens_seen, "val_loss": r["val_loss_nats"], "val_bpb": r["val_bpb"],
              "wallclock": time.time() - t_start, "train_time": train_time}
        curve.append(pt)
        log_f.write(json.dumps({"type": "eval", "tag": tag, **pt}) + "\n")
        print(f"[eval] step {step} val_loss {r['val_loss_nats']:.4f} bpb {r['val_bpb']:.4f}", flush=True)
        return r

    while total_steps is None or step < total_steps:
        if prof:
            prof.maybe_start(step)
        t0 = time.time()
        with region("data"):
            x_all, y_all = sampler.next(batch_seqs)
        while True:  # retry the step with a smaller micro batch on CUDA OOM
            try:
                loss_acc = torch.zeros((), device=device)
                aux_acc: dict[str, float] = {}
                n_micro = batch_seqs // micro
                for i in range(n_micro):
                    with region("data"):
                        xb = x_all[i * micro:(i + 1) * micro].to(device, non_blocking=True)
                        yb = y_all[i * micro:(i + 1) * micro].to(device, non_blocking=True)
                    if reset:
                        raw_model.reset_state()
                    with region("forward_backward"):
                        with amp:
                            loss, aux = model(xb, yb)
                        al = aux.get("aux_loss") if isinstance(aux, dict) else None
                        total = loss + al if al is not None else loss
                        (total / n_micro).backward()
                    loss_acc += loss.detach().float() / n_micro
                    for k, v in scalarize(aux).items():
                        aux_acc[k] = aux_acc.get(k, 0.0) + v / n_micro
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if micro == 1:
                    raise
                micro = max(1, micro // 2)
                while batch_seqs % micro:
                    micro -= 1
                config["micro_batch_size"] = micro
                print(f"[train] CUDA OOM at step {step}: micro batch -> {micro}", file=sys.stderr, flush=True)
        f = lr_factor(step, total_steps, args.warmup_steps, args.cooldown_frac, args.lr_floor)
        mfrac = min(1.0, step / args.muon_momentum_warmup_steps) if args.muon_momentum_warmup_steps > 0 else 1.0
        mom = (1 - mfrac) * args.muon_momentum_start + mfrac * args.muon_momentum
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * f
            if g.get("kind") == "muon" and "momentum" in g:
                g["momentum"] = mom
        with region("optimizer"):
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if has_step_end:
            if total_steps is not None:
                raw_model.on_step_end(step, total_steps)
            else:  # --time_budget_s calibration: a provisional total, never -1
                provisional_seq.append(provisional_total_steps(step, step_times, args.time_budget_s))
                raw_model.on_step_end(step, provisional_seq[-1])
        last_loss = float(loss_acc.item())  # syncs
        dt = time.time() - t0
        raw_train_time += dt
        step_times.append(dt)
        if 2 <= len(step_times) <= 11:
            compile_s = max(0.0, step_times[0] - float(np.median(step_times[1:])))
        train_time = raw_train_time - (compile_s if len(step_times) >= 2 else 0.0)
        step += 1
        tokens_seen += batch_seqs * T
        if prof:
            prof.maybe_stop(step)
        rec = {"type": "train", "step": step, "loss": last_loss, "lr_factor": f, "momentum": mom, "tokens": tokens_seen,
               "step_time": dt, "wallclock": time.time() - t_start, **mem_stats(device), **aux_acc}
        log_f.write(json.dumps(rec) + "\n")
        if step % args.log_every == 0 or step == 1:
            print(f"[train] step {step}/{total_steps} loss {last_loss:.4f} lr_f {f:.3f} {dt * 1000:.0f}ms/step "
                  f"{batch_seqs * T / dt:,.0f} tok/s", flush=True)
        if not math.isfinite(last_loss):
            status = "diverged"
            print(f"[train] non-finite loss at step {step}; stopping", file=sys.stderr, flush=True)
            break
        if step == 1 and device.type == "cuda" and not args.no_eval_probe:
            eval_probe = run_eval_probe(eval_model, raw_model, corpus.dev, token_bytes, cfg, eval_bs, device, amp)
            eval_time += eval_probe["s"]
        if total_steps is None and step >= args.calib_steps:
            per = float(np.median(step_times[1:] or step_times))
            total_steps = step + max(0, int((args.time_budget_s - train_time) / per))
            config["total_steps"] = total_steps
            config["calibrated_step_time_s"] = per
            if args.eval_every < 0:
                eval_every = max(1, total_steps // 8)
                config["eval_every"] = eval_every
            write_json(out / "config.json", config)
            print(f"[train] calibrated {per * 1000:.0f} ms/step -> total_steps {total_steps}", flush=True)
        if args.time_budget_s is not None and (args.steps is not None or args.tokens is not None) \
                and train_time > args.time_budget_s:
            status = "time_budget_stop"
            print(f"[train] time budget {args.time_budget_s}s reached at step {step}", file=sys.stderr, flush=True)
            break
        if eval_every and step % eval_every == 0 and (total_steps is None or step < total_steps):
            run_eval("curve")

    if prof:
        prof.finish(step)

    # final evaluation
    split_name = "final" if args.final else "dev"
    fe_tokens = None if args.final_eval_tokens is None or args.final_eval_tokens < 0 else args.final_eval_tokens
    pt_rec = (PerTokenRecorder(D.slice_fingerprint(corpus.split(split_name), T, fe_tokens, split_name)["n_tokens"])
              if args.save_per_token and status != "diverged" else None)
    t0 = time.time()
    final = (evaluate(eval_model, raw_model, corpus.split(split_name), token_bytes, cfg, fe_tokens, eval_bs, device, amp,
                      split=not args.no_split, per_token=pt_rec) if status != "diverged" else
             {"val_loss_nats": float("nan"), "val_bpb": float("nan")})
    eval_time += time.time() - t0
    wall = time.time() - t_start
    curve.append({"step": step, "tokens": tokens_seen, "val_loss": final["val_loss_nats"], "val_bpb": final["val_bpb"],
                  "wallclock": wall, "train_time": train_time, "final": True})
    ms = mem_stats(device)
    result = {
        "status": status, "run_id": run_id, "tag": args.tag, "model": args.model, "size": args.size, "seed": args.seed,
        "config": config, "git_commit": git_commit(),
        "params_total": params_total, "params_nonembedding": params_nonemb,
        "flops_per_token": fpt, "flops_source": fpt_src, "flops_per_token_harness": fpt_harness,
        "tokens": tokens_seen, "steps": step, "wallclock_s": wall, "train_time_s": train_time, "eval_time_s": eval_time,
        "compile_s": compile_s, "raw_train_time_s": raw_train_time,
        "tokens_per_s": tokens_seen / train_time if train_time else float("nan"),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "eval_split": split_name, "final_train_loss": last_loss, "data_seed": data_seed,
        "eval_fingerprint": D.slice_fingerprint(corpus.split(split_name), T, fe_tokens, split_name),
        "curve_eval_fingerprint": D.slice_fingerprint(corpus.dev, T, eval_tokens, "dev"),
        "val_loss_nats": final["val_loss_nats"], "val_bpb": final["val_bpb"],
        "val_split": final.get("val_split"), "eval_tokens": final.get("eval_tokens"), "eval_bytes": final.get("eval_bytes"),
        "eval_batch_size": eval_bs, "eval_interleave": True, "eval_batches": final.get("eval_batches"),
        "eval_padded_rows": final.get("eval_padded_rows"), "eval_reset_state": final.get("eval_reset_state"),
        "eval_state_mutated": final.get("eval_state_mutated"), "curve_eval_state_mutated": sorted(curve_mutated),
        "eval_probe": eval_probe,
        "total_steps_provisional": bool(provisional_seq), "total_steps_provisional_seq": provisional_seq or None,
        "train_shards": len(train_shards_used), "train_shards_default": train_shards_default,
        "train_shards_used": train_shards_used,
        "train_tokens_limit": args.train_tokens_limit, "train_pool_tokens": train_pool_tokens,
        "sampler": args.sampler, "n_windows": getattr(sampler, "n", None), "windows_used": step * batch_seqs,
        "rho": tokens_seen / train_pool_tokens,
        **ms, "curve": curve,
    }
    if args.save_weights:  # before result.json: a run with a result always has its weights
        wdt = torch.bfloat16 if args.weights_dtype == "bf16" else torch.float32
        sd_w = {k: (v.detach().to("cpu", wdt) if v.is_floating_point() else v.detach().cpu())
                for k, v in raw_model.state_dict().items()}
        torch.save({"model": sd_w, "model_name": args.model, "size": args.size, "cfg": cfg.to_dict(),
                    "weights_dtype": args.weights_dtype, "step": step, "tokens": tokens_seen, "run_id": run_id,
                    "seed": args.seed, "git_commit": result["git_commit"], "val_loss_nats": result["val_loss_nats"],
                    "status": status}, out / "weights.pt")
        print(f"[train] weights ({args.weights_dtype}) -> {out / 'weights.pt'}", flush=True)
    if scaling_meta is not None:  # sidecar for the scaling-slope judge (judge.py row_from_run), before result.json
        write_json(out / "scaling.json", {**scaling_meta, "rho": result["rho"], "train_pool_tokens": train_pool_tokens,
                                          "size": args.size, "budget_tokens": scaling_meta.get("budget_tokens", tokens_seen),
                                          "sampler": args.sampler})
    write_json(out / "result.json", result)
    log_f.write(json.dumps({"type": "final", **{k: v for k, v in result.items() if k not in ("config", "curve")}}) + "\n")
    log_f.close()
    sp = result["val_split"]
    print(f"[done] {args.model} {split_name} val_loss {result['val_loss_nats']:.4f} bpb {result['val_bpb']:.4f}"
          + (f" new {sp['new']['loss']:.3f} ({sp['new']['frac_positions']:.2f}) seen {sp['seen']['loss']:.3f}"
             if sp else "") + f" | {step} steps {wall:.1f}s -> {out / 'result.json'}", flush=True)
    if pt_rec is not None:
        pt_rec.save(out, {"run_id": run_id, "model": args.model, "split": split_name, "seq_len": T,
                          "vocab_size": cfg.vocab_size, "synthetic": bool(args.synthetic),
                          "val_loss_nats": result["val_loss_nats"], "eval_tokens": result["eval_tokens"],
                          "eval_fingerprint": result["eval_fingerprint"], "git_commit": result["git_commit"]})

    if args.save:
        ckpt = {"model": raw_model.state_dict(), "cfg": cfg.to_dict(), "args": vars(args), "step": step,
                "sampler": sampler.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "git_commit": result["git_commit"],
                "source": {"model": Path(mod.__file__).read_text(), "train.py": Path(__file__).read_text()}}
        try:
            ckpt["optimizer"] = optimizer.state_dict()
        except Exception as ex:  # noqa: BLE001
            ckpt["optimizer_error"] = repr(ex)
        torch.save(ckpt, out / "ckpt.pt")
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
