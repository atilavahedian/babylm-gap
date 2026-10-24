"""Agent E CPU micro-screen on real text (NOT the sprint bench; a pre-GPU sanity signal only).

Trains several E arms on identical data order and identical body init (same seed) with plain AdamW, then
reports held-out loss split into context-new vs already-seen target tokens. Data: GPT-2 token streams
(uint16 .bin, e.g. the measurement agent's Gutenberg split under sprint1/measure/data/tokens/).

  .venv-sprint1/bin/python -m sprint1.models.e_screen --steps 400 --arms plain refine refine_null \
      surface surface_null --out /tmp/e_screen.jsonl
GPU (until the shared harness train.py is the official path):
  python -m sprint1.models.e_screen --size small --device cuda --batch 16 --steps 2000 --lr 3e-3 \
      --arms plain refine refine_null refine_linear surface surface_null surface_out --out e_small.jsonl
Plain AdamW on all parameters (no Muon), bf16 autocast on CUDA: arms are comparable with each other,
not with the sprint baseline numbers.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time

import numpy as np
import torch

from sprint1.harness.model_api import SIZE_LADDER, ModelConfig

ARMS = {
    "plain": ("e_refine", {"rf_k": 0}),
    "refine": ("e_refine", {}),
    "refine_null": ("e_refine", {"null": True}),
    "refine_linear": ("e_refine", {"rf_linear": True}),
    "refine_block": ("e_refine", {"rf_mode": "block"}),
    "refine_block_null": ("e_refine", {"rf_mode": "block", "null": True}),
    "surface": ("e_surface", {}),
    "surface_null": ("e_surface", {"null": True}),
    "surface_out": ("e_surface", {"sf_side": "out"}),
}


def batches(arr: np.ndarray, B: int, T: int, seed: int):
    rng = np.random.default_rng(seed)
    while True:
        ix = rng.integers(0, len(arr) - T - 1, size=B)
        x = np.stack([arr[i:i + T + 1] for i in ix]).astype(np.int64)
        yield torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])


def run_arm(name, args, train, val):
    mod, extra = ARMS[name]
    shape = dict(SIZE_LADDER[args.size]) if args.size else dict(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, seq_len=args.seq_len)
    args.seq_len = shape["seq_len"]
    cfg = ModelConfig(**shape, extra=dict(extra, **json.loads(args.extra), tokens_per_step=args.batch * args.seq_len))
    torch.manual_seed(args.seed)
    model = importlib.import_module(f"sprint1.models.{mod}").build(cfg).to(args.device)
    dev = torch.device(args.device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    gen = batches(train, args.batch, args.seq_len, args.seed + 1)
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = (t.to(dev) for t in next(gen))
        with amp:
            loss, aux = model(x, y)
        opt.zero_grad(set_to_none=True)
        (loss + aux.get("aux_loss", 0.0)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if hasattr(model, "on_step_end"):
            model.on_step_end(step, args.steps)
        if step % 100 == 0:
            print(f"  {name} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    vgen = batches(val, args.batch, args.seq_len, 999)
    tot = {"loss": 0.0, "ce_new": 0.0, "ce_seen": 0.0, "frac_new": 0.0}
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = (t.to(dev) for t in next(vgen))
            with amp:
                loss, aux = model(x, y)
            tot["loss"] += loss.item()
            for k in ("ce_new", "ce_seen", "frac_new"):
                tot[k] += float(aux[k])
    res = {k: v / args.eval_batches for k, v in tot.items()}
    res.update(arm=name, steps=args.steps, train_s=round(time.time() - t0, 1),
               params=sum(p.numel() for p in model.parameters()), flops_per_token=model.flops_per_token(),
               cfg=dict(shape, seq_len=args.seq_len, batch=args.batch,
                        lr=args.lr, seed=args.seed, extra=cfg.extra))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "measure", "data", "tokens"))
    ap.add_argument("--arms", nargs="+", default=["plain", "refine", "refine_null", "surface", "surface_null"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--size", default=None, help="tiny|small|medium from SIZE_LADDER (overrides shape args)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--n_embd", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    train = np.fromfile(os.path.join(args.data, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(args.data, "val.bin"), dtype=np.uint16)
    for name in args.arms:
        res = run_arm(name, args, train, val)
        print(json.dumps(res), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
