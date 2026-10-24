"""Peak memory of one training step (forward + backward + optimizer step) at a given micro batch.

    python -m babylm.harness.mem_estimate --size S --micro 1 2 4 [--device auto|cpu|mps|cuda] [--math_sdpa]

Each (size, micro) is measured in a fresh subprocess. Reported: peak bytes on the device (cuda: max_memory_allocated;
mps: torch.mps.driver_allocated_memory() after the step; cpu: growth of the process peak RSS over the RSS after
building the model). --math_sdpa forces the math attention kernel, which stores the [B, H, T, T] attention matrix
(the CPU default kernel does not); use it on the CPU to bound what a non-flash kernel (e.g. MPS) needs.
Peak for micro m ~ fixed (params, grads, optimizer state) + m x per-sequence activations: the tool prints both.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys


def one(size: str, micro: int, device: str, math_sdpa: bool, seq_len: int) -> dict:
    import contextlib

    import torch

    from babylm.harness.train import SIZES, make_cfg, build_parser, pick_device
    from sprint1.harness.optim import build_optimizer
    from sprint1.models import baseline

    dev = pick_device(None if device == "auto" else device)
    args = build_parser().parse_args(["--size", size, "--seq_len", str(seq_len), "--extra", "softcap=15", "unet=true"])
    cfg = make_cfg(args)
    torch.manual_seed(0)
    m = baseline.build(cfg).to(dev)
    opt = build_optimizer(m, {"muon": 0.065, "adam_embed": 0.78, "adam_head": 0.0208, "adam_scalar": 0.052},
                          adam_betas=(0.8, 0.95), adam_eps=1e-10)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    x = torch.randint(0, cfg.vocab_size, (micro, cfg.seq_len), device=dev)
    ctx = contextlib.nullcontext()
    if math_sdpa:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel(SDPBackend.MATH)
    with ctx:
        for _ in range(2):  # the second step has optimizer state allocated
            loss, _ = m(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    out = {"size": size, "micro": micro, "device": str(dev), "math_sdpa": math_sdpa, "seq_len": cfg.seq_len,
           "params": sum(p.numel() for p in m.parameters())}
    if dev.type == "cuda":
        out["peak_bytes"] = int(torch.cuda.max_memory_allocated(dev))
    elif dev.type == "mps":
        torch.mps.synchronize()
        out["peak_bytes"] = int(torch.mps.driver_allocated_memory())
    else:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        out["peak_bytes"] = int(rss - rss0)
        out["note"] = "cpu: peak RSS growth after model build (model weights excluded)"
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", nargs="+", default=["small", "S"])
    ap.add_argument("--micro", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--math_sdpa", action="store_true")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--_one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a._one:
        print(json.dumps(one(a.size[0], a.micro[0], a.device, a.math_sdpa, a.seq_len)))
        return
    rows = []
    for s in a.size:
        for mb in a.micro:
            cmd = [sys.executable, "-m", "babylm.harness.mem_estimate", "--_one", "--size", s, "--micro", str(mb),
                   "--device", a.device, "--seq_len", str(a.seq_len)] + (["--math_sdpa"] if a.math_sdpa else [])
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode:
                print(p.stderr[-2000:], file=sys.stderr)
                continue
            r = json.loads(p.stdout.strip().splitlines()[-1])
            rows.append(r)
            print(f"{s:6s} micro {mb:3d}: {r['peak_bytes'] / 2**30:6.2f} GiB  ({r['device']}, math_sdpa={a.math_sdpa})",
                  flush=True)
    for s in a.size:
        rr = sorted((r for r in rows if r["size"] == s), key=lambda r: r["micro"])
        if len(rr) >= 2:
            per = (rr[-1]["peak_bytes"] - rr[0]["peak_bytes"]) / (rr[-1]["micro"] - rr[0]["micro"])
            fixed = rr[0]["peak_bytes"] - per * rr[0]["micro"]
            print(f"{s}: ~{per / 2**30:.2f} GiB per sequence of {a.seq_len} + ~{fixed / 2**30:.2f} GiB fixed "
                  f"-> micro for 8 GiB: {int((8 * 2**30 - fixed) // per)}, for 12 GiB: {int((12 * 2**30 - fixed) // per)}")


if __name__ == "__main__":
    main()
