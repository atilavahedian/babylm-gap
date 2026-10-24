"""Sprint-1 frozen recipe (sprint1/configs/frozen_baseline.json, origin/sprint-1) ported to BabyLM 2025 strict-small.

GPU (one seed, the real run):
    python babylm/recipes/sprint1_port/train.py --seed 0 --out babylm/recipes/sprint1_port/runs/s1port_small_s0
CPU smoke (tiny model, real BabyLM tokens, 30 steps, checkpoints at 1k..10k, 20k, 30k "words"):
    python babylm/recipes/sprint1_port/train.py --smoke --out <scratch>/smoke

What is unchanged (copied byte-for-byte into vendor/ and verified by git blob hash against origin/sprint-1):
model (sprint1/models/baseline.py), ModelConfig + size ladder (sprint1/harness/model_api.py), Muon + AdamW groups
(sprint1/harness/optim.py). The training step (micro-batch accumulation, LR schedule lr_factor, Muon momentum warmup,
bf16 autocast on CUDA, tf32, torch.compile on CUDA) is a line-for-line copy of sprint1/harness/train.py. The frozen
CLI args (FROZEN_ARGS below, copied from frozen_baseline.json) are parsed FIRST; anything on the command line is
parsed after them and overrides.

What changed (see README.md): data = BabyLM train_10M (prep_data.py), sampler = seeded epoch sampler (each epoch a
random phase offset in [0, T) and a random permutation of the non-overlapping T+1-token windows; sprint-1 drew
windows uniformly with replacement, which has no exact epoch), budget = --epochs full epochs (default 10, the BabyLM
cap) -> total_steps = floor(total windows / batch_seqs), held-out = official BabyLM dev split, BabyLM word-milestone
checkpoints (chck_1M..chck_10M, chck_20M..chck_100M by words_seen_est), loss-curve CSV + PNG.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
from sprint1.harness.model_api import SIZE_LADDER, ModelConfig  # noqa: E402  (vendored, verbatim)
from sprint1.harness.optim import build_optimizer  # noqa: E402  (vendored, verbatim)

# frozen_baseline.json "args" (origin/sprint-1, config s2_B_base, frozen 5 Oct): parsed before the user's argv.
FROZEN_ARGS = ["--size", "small", "--batch_tokens", "65536", "--seq_len", "1024", "--lr_batch_scaling", "none",
               "--muon_lr", "0.065", "--lr_embed", "0.78", "--lr_head", "0.0208", "--lr_scalar", "0.052",
               "--muon_momentum", "0.95", "--muon_momentum_start", "0.85", "--muon_momentum_warmup_steps", "150",
               "--adam_beta1", "0.8", "--adam_beta2", "0.95", "--warmup_steps", "0", "--cooldown_frac", "0.5",
               "--lr_floor", "0.1", "--muon_wd", "0", "--adam_wd", "0", "--extra", "softcap=15", "--extra",
               "unet=true", "--compile"]
SIZE_DEFAULTS = {"tiny": dict(batch_tokens=2048, micro_batch_size=8), "small": dict(batch_tokens=65536, micro_batch_size=16),
                 "medium": dict(batch_tokens=65536, micro_batch_size=8)}  # sprint-1 harness values
EVAL_BATCH_DEFAULT = {"cuda": 32, "cpu": 8}  # sprint-1 harness values
REF_BATCH_TOKENS = 524288
EOT = 50256
CSV_FIELDS = ["step", "tokens_seen", "words_seen_est", "epoch", "lr_scale", "train_loss", "dev_loss", "elapsed_s"]


def milestones_words(unit: float) -> list[tuple[str, float]]:
    """BabyLM 2025: every 1M words to 10M, then every 10M to 100M (names chck_<n>M)."""
    ms = [i for i in range(1, 11)] + [i for i in range(20, 101, 10)]
    return [(f"chck_{m}M", m * unit) for m in ms]


# ------------------------------------------------------------------------------------------------
# CLI (sprint-1 flags kept with their sprint-1 defaults; BabyLM flags added)
# ------------------------------------------------------------------------------------------------
def parse_value(v: str) -> Any:
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


def parse_extra(items) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = parse_value(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sprint1.models.baseline", help="module exposing build(cfg) (vendor/ on path)")
    ap.add_argument("--size", default="small", choices=sorted(SIZE_LADDER))
    ap.add_argument("--seed", type=int, default=0, help="model init (torch.manual_seed) and, unless --data_seed, data order")
    ap.add_argument("--data_seed", type=int, default=None)
    b = ap.add_argument_group("budget")
    b.add_argument("--epochs", type=int, default=10, help="full passes over train_10M (BabyLM cap: 10)")
    b.add_argument("--max_steps", type=int, default=None, help="hard cap on steps (smoke); the LR schedule uses it")
    b.add_argument("--batch_tokens", type=int, default=None)
    b.add_argument("--micro_batch_size", type=int, default=None)
    o = ap.add_argument_group("optimizer (sprint-1)")
    o.add_argument("--muon_lr", type=float, default=0.065)
    o.add_argument("--muon_momentum", type=float, default=0.95)
    o.add_argument("--muon_momentum_start", type=float, default=0.85)
    o.add_argument("--muon_momentum_warmup_steps", type=int, default=150)
    o.add_argument("--muon_wd", type=float, default=0.0)
    o.add_argument("--lr_embed", type=float, default=0.78)
    o.add_argument("--lr_head", type=float, default=0.0208)
    o.add_argument("--lr_scalar", type=float, default=0.052)
    o.add_argument("--lr_batch_scaling", default="none", choices=["sqrt", "none"])
    o.add_argument("--ref_batch_tokens", type=int, default=REF_BATCH_TOKENS)
    o.add_argument("--adam_beta1", type=float, default=0.8)
    o.add_argument("--adam_beta2", type=float, default=0.95)
    o.add_argument("--adam_eps", type=float, default=1e-10)
    o.add_argument("--adam_wd", type=float, default=0.0)
    o.add_argument("--warmup_steps", type=int, default=0)
    o.add_argument("--cooldown_frac", type=float, default=0.5)
    o.add_argument("--lr_floor", type=float, default=0.1)
    o.add_argument("--grad_clip", type=float, default=0.0)
    m = ap.add_argument_group("model overrides")
    for k, t in (("n_layer", int), ("n_head", int), ("n_embd", int), ("seq_len", int), ("mlp_ratio", float),
                 ("vocab_size", int)):
        m.add_argument(f"--{k}", type=t, default=None)
    m.add_argument("--extra", action="append", nargs="+", default=None, metavar="KEY=VAL")
    e = ap.add_argument_group("evaluation (official BabyLM dev split)")
    e.add_argument("--eval_every", type=int, default=100, help="steps between dev-subset evals (also at every "
                   "word milestone, at step 0 and at the end); 0 = only milestones/end")
    e.add_argument("--eval_windows", type=int, default=512, help="curve subset: this many evenly spaced dev windows")
    e.add_argument("--final_eval_windows", type=int, default=-1, help="-1 = every non-overlapping dev window")
    e.add_argument("--eval_batch_size", type=int, default=None)
    e.add_argument("--eval_eager", action="store_true")
    e.add_argument("--save_per_token", action="store_true", help="final eval -> per_token.npz")
    r = ap.add_argument_group("run")
    r.add_argument("--data_dir", default=str(HERE / "data"))
    r.add_argument("--out", required=True)
    r.add_argument("--device", default=None)
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--compile", action="store_true", help="torch.compile (frozen recipe: on; applied on CUDA only)")
    r.add_argument("--no_compile", action="store_true")
    r.add_argument("--milestone_unit", type=float, default=1e6, help="words per 'M' in chck_<n>M (smoke: 1000)")
    r.add_argument("--no_milestone_ckpt", action="store_true")
    r.add_argument("--log_every", type=int, default=10)
    r.add_argument("--smoke", action="store_true", help="CPU smoke: tiny model, 30 steps, 16 eval windows, unit 1000")
    return ap


def parse_args(argv: list[str] | None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(FROZEN_ARGS + list(argv))
    if args.smoke:
        if "--size" not in argv:
            args.size = "tiny"
        if "--seq_len" not in argv:
            args.seq_len = SIZE_LADDER[args.size]["seq_len"]
        if "--batch_tokens" not in argv:
            args.batch_tokens = SIZE_DEFAULTS[args.size]["batch_tokens"]
        args.max_steps = args.max_steps or 30
        args.eval_every = 10 if "--eval_every" not in argv else args.eval_every
        args.eval_windows = 16 if "--eval_windows" not in argv else args.eval_windows
        args.final_eval_windows = 32 if "--final_eval_windows" not in argv else args.final_eval_windows
        args.milestone_unit = 1000.0 if "--milestone_unit" not in argv else args.milestone_unit
        args.no_compile = True
        args.threads = args.threads or 1
    return args


# ------------------------------------------------------------------------------------------------
# helpers (lr_factor and seen_mask copied from sprint1/harness/train.py)
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
    w = max(0.0, (total - step) / cd)
    return floor + (1.0 - floor) * w


def seen_mask(idx: torch.Tensor, targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device).expand(B, T)
    first = torch.full((B, vocab_size), T, dtype=torch.long, device=idx.device)
    first.scatter_reduce_(1, idx, pos, reduce="amin", include_self=True)
    f = first.gather(1, targets.clamp_min(0))
    return (f <= pos) & (targets >= 0)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
class BabyLMData:
    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.train = np.fromfile(d / "train.bin", dtype=np.uint16)
        self.dev = np.fromfile(d / "dev.bin", dtype=np.uint16)
        self.dev_domain = np.fromfile(d / "dev_domain.bin", dtype=np.uint8)
        self.token_bytes = np.load(d / "token_bytes.npy")
        assert len(self.train) == self.meta["train"]["tokens"] and len(self.dev) == self.meta["dev"]["tokens"]
        self.words_per_token = self.meta["train"]["words"] / len(self.train)
        self.files = {k: sha256_file(d / k) for k in ("train.bin", "dev.bin", "dev_domain.bin", "token_bytes.npy")}
        for k, mk in (("train.bin", ("train", "bin_sha256")), ("dev.bin", ("dev", "bin_sha256"))):
            assert self.files[k] == self.meta[mk[0]][mk[1]], f"{k} sha256 differs from meta.json"


class EpochSampler:
    """Epoch e: phase offset o_e ~ U{0..T-1}, windows start at o_e + k*T (k < (N-1-o_e)//T, each T+1 tokens: T inputs,
    T targets), visited in a random permutation. Epochs are consumed back to back; a step's batch may straddle an
    epoch boundary. Every target token is seen at most once per epoch -> at most `epochs` times in total."""

    def __init__(self, tokens: np.ndarray, T: int, seed: int, epochs: int):
        self.tokens, self.T = tokens, T
        rng = np.random.Generator(np.random.PCG64(seed + 1_000_003))
        starts, ep, frac = [], [], []
        for e in range(epochs):
            off = int(rng.integers(0, T))
            n = (len(tokens) - 1 - off) // T
            perm = rng.permutation(n)
            starts.append(off + perm.astype(np.int64) * T)
            ep.append(np.full(n, e, dtype=np.int32))
            frac.append((np.arange(n) + 1) / n)
        self.starts = np.concatenate(starts)
        self.epoch_of = np.concatenate(ep)
        self.epoch_pos = np.concatenate(frac)
        self.windows_per_epoch = [len(s) for s in starts]
        self.ptr = 0

    @property
    def total_windows(self) -> int:
        return len(self.starts)

    def next(self, n: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        s = self.starts[self.ptr:self.ptr + n]
        assert len(s) == n, "sampler exhausted (budget exceeds --epochs)"
        buf = np.asarray(self.tokens[s[:, None] + np.arange(self.T + 1)], dtype=np.int64)
        last = self.ptr + n - 1
        self.ptr += n
        t = torch.from_numpy(buf)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous(), float(self.epoch_of[last] + self.epoch_pos[last])

    def state_dict(self) -> dict:
        return {"ptr": self.ptr}


def dev_windows(n_tokens: int, T: int, n: int, mode: str) -> np.ndarray:
    """'subset': n evenly spaced window starts over the whole dev stream (all 6 domains in token proportion),
    round(linspace(0, N-T-1, n)); 'full': every non-overlapping window k*T, k < (N-1)//T (n > 0 caps the count)."""
    if mode == "subset":
        return np.round(np.linspace(0, n_tokens - T - 1, n)).astype(np.int64)
    k = (n_tokens - 1) // T
    if n > 0:
        k = min(k, n)
    return np.arange(k, dtype=np.int64) * T


def fingerprint(tokens: np.ndarray, starts: np.ndarray, T: int) -> str:
    h = hashlib.sha256()
    h.update(starts.tobytes())
    h.update(np.asarray(tokens[starts[0]:starts[0] + T + 1], dtype=np.int64).tobytes())
    h.update(np.asarray(tokens[starts[-1]:starts[-1] + T + 1], dtype=np.int64).tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, raw_model, data: BabyLMData, starts: np.ndarray, T: int, batch: int, device, amp, vocab: int,
             domains: list[str], per_token: bool = False) -> dict[str, Any]:
    """Mean next-token nats over every target of the given dev windows (fixed batch, padding rows score nothing),
    bits per byte, the sprint-1 new/seen split (target absent/present earlier in the window), per-domain loss
    (domain of the target token). The model's eval forward returns aux['per_token_loss'] (sprint-1 contract);
    its batch mean is checked against the scored loss (sprint-1 PT_REL_TOL = 1e-3)."""
    was = model.training
    model.eval()
    tb = torch.as_tensor(data.token_bytes, dtype=torch.int64, device=device)
    nd = len(domains)
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)  # noqa: E731
    nats, ntok, nbytes, nats_new, n_new = z(()), z(()), z(()), z(()), z(())
    dnats, dtok = z(nd), z(nd)
    max_rel = 0.0
    rec = {"loss": [], "target": [], "window": [], "pos": [], "seen": [], "domain": []} if per_token else None
    offs = np.arange(T + 1)
    try:
        for b0 in range(0, len(starts), batch):
            s = starts[b0:b0 + batch]
            k = len(s)
            sp = np.concatenate([s, np.full(batch - k, s[0])]) if k < batch else s
            buf = torch.from_numpy(np.asarray(data.dev[sp[:, None] + offs], dtype=np.int64))
            dom = torch.from_numpy(np.asarray(data.dev_domain[sp[:, None] + offs[1:]], dtype=np.int64)).to(device)
            x, y = buf[:, :-1].to(device), buf[:, 1:].clone().to(device)
            if k < batch:
                y[k:] = -1
            with amp:
                loss, aux = model(x, y)
            pt = aux["per_token_loss"].double()
            valid = y >= 0
            nv = valid.sum()
            rel = abs(pt[valid].mean().item() - loss.item()) / max(1.0, abs(loss.item()))
            max_rel = max(max_rel, rel)
            nats += loss.double() * nv
            ntok += nv
            nbytes += tb[y.clamp_min(0)][valid].sum()
            seen = seen_mask(x, y, vocab)
            new = valid & ~seen
            nats_new += pt[new].sum()
            n_new += new.sum()
            dnats.index_add_(0, dom[valid], pt[valid])
            dtok.index_add_(0, dom[valid], torch.ones_like(pt[valid]))
            if rec is not None:
                v = valid.cpu()
                rec["loss"].append(pt.float().cpu()[v].numpy().astype(np.float16))
                rec["target"].append(y.cpu()[v].numpy().astype(np.int32))
                rec["window"].append(np.broadcast_to((b0 + np.arange(batch))[:, None], (batch, T))[v.numpy()].astype(np.int32))
                rec["pos"].append(np.broadcast_to(np.arange(T)[None], (batch, T))[v.numpy()].astype(np.int16))
                rec["seen"].append(seen.cpu()[v].numpy())
                rec["domain"].append(dom.cpu()[v].numpy().astype(np.uint8))
    finally:
        model.train(was)
    if max_rel > 1e-3:
        raise RuntimeError(f"[eval] per_token_loss mean differs from the scored loss by {max_rel:.3g} (> 1e-3)")
    nf, tf, bf, nn_, nnew = nats.item(), ntok.item(), nbytes.item(), nats_new.item(), n_new.item()
    out = {"dev_loss": nf / tf, "dev_bpb": nf / (bf * math.log(2)), "dev_tokens": int(tf), "dev_bytes": int(bf),
           "windows": int(len(starts)), "fingerprint": fingerprint(data.dev, starts, T), "pt_max_rel_diff": max_rel,
           "split": {"new": {"loss": nn_ / max(nnew, 1), "frac": nnew / tf},
                     "seen": {"loss": (nf - nn_) / max(tf - nnew, 1), "frac": (tf - nnew) / tf}},
           "by_domain": {d: {"loss": dnats[i].item() / max(dtok[i].item(), 1), "tokens": int(dtok[i].item())}
                         for i, d in enumerate(domains)}}
    if rec is not None:
        out["_per_token"] = {k: np.concatenate(v) for k, v in rec.items()}
    return out


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    t_start = time.time()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    use_compile = args.compile and not args.no_compile and device.type == "cuda"

    d = dict(SIZE_LADDER[args.size])
    for k in ("n_layer", "n_head", "n_embd", "seq_len", "mlp_ratio", "vocab_size"):
        if getattr(args, k) is not None:
            d[k] = getattr(args, k)
    cfg = ModelConfig(extra=parse_extra([x for g in (args.extra or []) for x in g]), **d)
    T = cfg.seq_len
    sd = SIZE_DEFAULTS[args.size]
    batch_tokens = args.batch_tokens or sd["batch_tokens"]
    batch_seqs = max(1, batch_tokens // T)
    micro = min(args.micro_batch_size or sd["micro_batch_size"], batch_seqs)
    while batch_seqs % micro:
        micro -= 1
    eval_bs = args.eval_batch_size or EVAL_BATCH_DEFAULT["cuda" if device.type == "cuda" else "cpu"]

    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = BabyLMData(args.data_dir)
    domains = data.meta["dev"]["domains"]
    data_seed = args.seed if args.data_seed is None else args.data_seed
    sampler = EpochSampler(data.train, T, data_seed, args.epochs)
    total_steps = sampler.total_windows // batch_seqs  # never more than --epochs full passes
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    curve_starts = dev_windows(len(data.dev), T, args.eval_windows, "subset")
    final_starts = dev_windows(len(data.dev), T, args.final_eval_windows, "full")

    import importlib
    mod = importlib.import_module(args.model)
    raw_model = mod.build(cfg).to(device)
    raw_model.train()
    model = torch.compile(raw_model) if use_compile else raw_model
    eval_model = raw_model if args.eval_eager else model
    n_params = sum(p.numel() for p in raw_model.parameters())

    scale = (batch_seqs * T / args.ref_batch_tokens) ** 0.5 if args.lr_batch_scaling == "sqrt" else 1.0
    base_lrs = {"muon": args.muon_lr, "adam_embed": args.lr_embed * scale, "adam_head": args.lr_head * scale,
                "adam_scalar": args.lr_scalar * scale}
    optimizer = build_optimizer(raw_model, base_lrs, muon_momentum=args.muon_momentum, muon_wd=args.muon_wd,
                                adam_betas=(args.adam_beta1, args.adam_beta2), adam_eps=args.adam_eps,
                                adam_wd=args.adam_wd)
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", g["lr"])

    wpt = data.words_per_token
    ms = [] if args.no_milestone_ckpt else milestones_words(args.milestone_unit)
    config = {"args": vars(args), "frozen_args": FROZEN_ARGS, "model_cfg": cfg.to_dict(), "batch_seqs": batch_seqs,
              "micro_batch_size": micro, "batch_tokens": batch_seqs * T, "base_lrs": base_lrs, "total_steps": total_steps,
              "epochs": args.epochs, "windows_per_epoch": sampler.windows_per_epoch,
              "budget_tokens": total_steps * batch_seqs * T, "train_tokens": int(len(data.train)),
              "words_per_token_train": wpt, "budget_words_est": total_steps * batch_seqs * T * wpt,
              "frozen_budget_tokens": 152043520, "data_seed": data_seed, "data_files_sha256": data.files,
              "data_meta": {k: data.meta[k] for k in ("tokenizer_sha256", "doc_unit", "join")},
              "curve_eval": {"windows": len(curve_starts), "fingerprint": fingerprint(data.dev, curve_starts, T)},
              "final_eval": {"windows": len(final_starts), "fingerprint": fingerprint(data.dev, final_starts, T)},
              "eval_batch_size": eval_bs, "device": str(device), "torch": torch.__version__,
              "precision": "bf16-autocast" if device.type == "cuda" else "fp32", "compile": use_compile,
              "params": n_params, "git_commit": git_commit(),
              "vendor_sha256": {str(p.relative_to(HERE)): sha256_file(p) for p in sorted((HERE / "vendor").rglob("*.py"))},
              "train_py_sha256": sha256_file(Path(__file__))}
    write_json(out / "config.json", config)
    print(f"[train] {args.model} size={args.size} params={n_params:,} batch={batch_seqs}x{T} micro={micro} "
          f"steps={total_steps} tokens={total_steps * batch_seqs * T:,} (~{total_steps * batch_seqs * T * wpt / 1e6:.2f}M words, "
          f"{total_steps * batch_seqs / (sum(sampler.windows_per_epoch) / args.epochs):.3f} epochs) device={device} "
          f"compile={use_compile} out={out}", flush=True)

    csv_f = open(out / "curve.csv", "w", newline="", buffering=1)
    cw = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    cw.writeheader()
    log_f = open(out / "log.jsonl", "w", buffering=1)
    evals: list[dict[str, Any]] = []
    eval_time = 0.0
    step_times: list[float] = []
    compile_s = 0.0
    tokens_seen = 0
    epoch = 0.0
    last_loss = float("nan")
    status = "ok"
    mi = 0
    saved: list[dict[str, Any]] = []

    def row(step, f, train_loss, dev_loss):
        cw.writerow({"step": step, "tokens_seen": tokens_seen, "words_seen_est": round(tokens_seen * wpt),
                     "epoch": f"{epoch:.5f}", "lr_scale": f"{f:.5f}", "train_loss": "" if train_loss is None else f"{train_loss:.5f}",
                     "dev_loss": "" if dev_loss is None else f"{dev_loss:.5f}", "elapsed_s": f"{time.time() - t_start:.2f}"})

    def run_eval(step, tag):
        nonlocal eval_time
        t0 = time.time()
        r = evaluate(eval_model, raw_model, data, curve_starts, T, eval_bs, device, amp, cfg.vocab_size, domains)
        eval_time += time.time() - t0
        rec = {"type": "eval", "tag": tag, "step": step, "tokens": tokens_seen, "words_est": tokens_seen * wpt,
               "epoch": epoch, **{k: v for k, v in r.items() if k != "_per_token"}}
        evals.append(rec)
        log_f.write(json.dumps(rec) + "\n")
        print(f"[eval] step {step} ({tag}) dev_loss {r['dev_loss']:.4f} bpb {r['dev_bpb']:.4f} "
              f"words {tokens_seen * wpt / 1e6:.2f}M epoch {epoch:.3f}", flush=True)
        return r["dev_loss"]

    def save(name, step, full=False):
        ck = {"model": raw_model.state_dict(), "cfg": cfg.to_dict(), "step": step, "tokens_seen": tokens_seen,
              "words_seen_est": tokens_seen * wpt, "epoch": epoch, "name": name, "model_module": args.model}
        if full:
            ck.update({"optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(), "args": vars(args),
                       "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                       "python_rng": random.getstate(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                       "source": {"train.py": Path(__file__).read_text(),
                                  "model": Path(mod.__file__).read_text()}})
        p = out / "ckpt" / f"{name}.pt"
        torch.save(ck, p)
        saved.append({"name": name, "step": step, "tokens_seen": tokens_seen, "words_seen_est": tokens_seen * wpt,
                      "epoch": epoch, "path": str(p)})

    dl0 = run_eval(0, "init")
    row(0, lr_factor(0, total_steps, args.warmup_steps, args.cooldown_frac, args.lr_floor), None, dl0)
    step = 0
    while step < total_steps:
        t0 = time.time()
        x_all, y_all, ep_after = sampler.next(batch_seqs)
        loss_acc = torch.zeros((), device=device)
        n_micro = batch_seqs // micro
        for i in range(n_micro):
            xb = x_all[i * micro:(i + 1) * micro].to(device, non_blocking=True)
            yb = y_all[i * micro:(i + 1) * micro].to(device, non_blocking=True)
            with amp:
                loss, aux = model(xb, yb)
            al = aux.get("aux_loss") if isinstance(aux, dict) else None
            total = loss + al if al is not None else loss
            (total / n_micro).backward()
            loss_acc += loss.detach().float() / n_micro
        f = lr_factor(step, total_steps, args.warmup_steps, args.cooldown_frac, args.lr_floor)
        mfrac = min(1.0, step / args.muon_momentum_warmup_steps) if args.muon_momentum_warmup_steps > 0 else 1.0
        mom = (1 - mfrac) * args.muon_momentum_start + mfrac * args.muon_momentum
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * f
            if g.get("kind") == "muon" and "momentum" in g:
                g["momentum"] = mom
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss_acc.item())
        dt = time.time() - t0
        step_times.append(dt)
        if 2 <= len(step_times) <= 11:
            compile_s = max(0.0, step_times[0] - float(np.median(step_times[1:])))
        step += 1
        tokens_seen += batch_seqs * T
        epoch = ep_after
        log_f.write(json.dumps({"type": "train", "step": step, "loss": last_loss, "lr_factor": f, "momentum": mom,
                                "tokens": tokens_seen, "epoch": epoch, "step_time": dt}) + "\n")
        if step % args.log_every == 0 or step == 1:
            print(f"[train] step {step}/{total_steps} loss {last_loss:.4f} lr_f {f:.3f} ep {epoch:.3f} "
                  f"{dt * 1000:.0f}ms/step {batch_seqs * T / dt:,.0f} tok/s", flush=True)
        if not math.isfinite(last_loss):
            status = "diverged"
            row(step, f, last_loss, None)
            break
        dev_loss = None
        hit = []
        while mi < len(ms) and tokens_seen * wpt >= ms[mi][1]:
            hit.append(ms[mi][0])
            mi += 1
        if (hit or (args.eval_every and step % args.eval_every == 0)) and step < total_steps:
            dev_loss = run_eval(step, "milestone:" + ",".join(hit) if hit else "curve")
        for name in hit:
            save(name, step)
        if step == total_steps:
            dev_loss = run_eval(step, "end")
        row(step, f, last_loss, dev_loss)

    train_time = sum(step_times) - compile_s
    t0 = time.time()
    final = (evaluate(eval_model, raw_model, data, final_starts, T, eval_bs, device, amp, cfg.vocab_size, domains,
                      per_token=args.save_per_token) if status == "ok" else {"dev_loss": float("nan")})
    eval_time += time.time() - t0
    if status == "ok":
        save("final", step, full=True)
    pt = final.pop("_per_token", None)
    if pt is not None:
        np.savez_compressed(out / "per_token.npz", **pt, domains=np.array(domains),
                            meta=np.array(json.dumps({"windows": "final_eval", "seq_len": T})))
    csv_f.close()
    result = {"status": status, "seed": args.seed, "data_seed": data_seed, "steps": step, "total_steps": total_steps,
              "tokens_seen": tokens_seen, "words_seen_est": tokens_seen * wpt, "epochs_seen": epoch,
              "final_train_loss": last_loss, "final_dev": final, "curve_evals": evals, "checkpoints": saved,
              "train_time_s": train_time, "compile_s": compile_s, "eval_time_s": eval_time,
              "wallclock_s": time.time() - t_start, "tokens_per_s": tokens_seen / train_time if train_time > 0 else None,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "peak_mem_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
              "config": config}
    write_json(out / "result.json", result)
    log_f.write(json.dumps({"type": "final", "dev_loss": final.get("dev_loss"), "steps": step}) + "\n")
    log_f.close()
    try:
        subprocess.run([sys.executable, str(HERE.parent.parent / "tools" / "plot_curve.py"), str(out / "curve.csv"),
                        str(out / "curve.png")], check=True, timeout=120, capture_output=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[plot] failed: {ex!r}", file=sys.stderr)
    print(f"[done] dev_loss (full dev, {final.get('windows')} windows) {final.get('dev_loss', float('nan')):.4f} "
          f"bpb {final.get('dev_bpb', float('nan')):.4f} | {step} steps, {tokens_seen:,} tokens, "
          f"{tokens_seen * wpt / 1e6:.2f}M words, {time.time() - t_start:.0f}s -> {out}", flush=True)
    return result


if __name__ == "__main__":
    main()
