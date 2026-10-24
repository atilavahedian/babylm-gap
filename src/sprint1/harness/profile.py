"""Short-run profile of one model: time per step, a breakdown, tokens/s, MFU, peak memory.

    python -m sprint1.harness.profile --model baseline --size small --out sprint1/runs/prof_baseline_small
    (any train.py flag may follow; defaults: 5 warm-up steps, then 5 profiled steps, small final eval)

Writes <out>/profile.json and <out>/profile.md (plus the usual result.json of the short run).

Breakdown sources:
  harness regions   data (sampling + host->device), forward_backward, optimizer  (record_function ranges)
  model regions     forward time of submodules whose name ends in attn/attention (attention), mlp/ffn/moe
                    (mlp), lm_head/head (head), embed/wte/value_embeds (embed), via forward hooks. Backward is
                    not attributed per module: backward = forward_backward - forward (all modules).
  device time (CUDA kernels) when on GPU, CPU time otherwise. MFU = flops_per_token * tokens/s / peak FLOPs
  (--peak_tflops, L4 dense bf16 = 121). On CPU MFU is not reported.
"""
from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

CATEGORIES = {
    "attention": ("attn", "attention", "self_attn"),
    "mlp": ("mlp", "ffn", "moe", "feed_forward"),
    "head": ("lm_head", "head", "unembed"),
    "embed": ("embed", "wte", "value_embeds", "tok_emb"),
}


def _category(name: str) -> str | None:
    last = name.split(".")[-1].lower()
    for cat, keys in CATEGORIES.items():
        if last in keys:
            return cat
    return None


def _dev_time(evt) -> float:
    for attr in ("device_time_total", "cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


class StepProfiler:
    def __init__(self, model: nn.Module, out_dir: Path, *, start: int, n_steps: int, device: torch.device,
                 flops_per_token: float, tokens_per_step: int, peak_tflops: float):
        self.model, self.out, self.start, self.n = model, Path(out_dir), start, max(1, n_steps)
        self.device, self.fpt, self.tps, self.peak = device, flops_per_token, tokens_per_step, peak_tflops
        self.prof = None
        self.active = False
        self.done = False
        self.t0 = 0.0
        self._stack: list[Any] = []
        self._handles = []
        matched: list[str] = []
        for name, mod in model.named_modules():
            cat = _category(name) if name else None
            if cat is None or any(name.startswith(m + ".") for m in matched):
                continue
            matched.append(name)
            self._handles.append(mod.register_forward_pre_hook(self._pre(cat)))
            self._handles.append(mod.register_forward_hook(self._post()))

    def _pre(self, cat: str):
        def hook(mod, inp):
            if self.active and torch.is_grad_enabled():
                rf = torch.autograd.profiler.record_function(f"model::{cat}")
                rf.__enter__()
                self._stack.append(rf)
        return hook

    def _post(self):
        def hook(mod, inp, out):
            if self.active and self._stack and torch.is_grad_enabled():
                self._stack.pop().__exit__(None, None, None)
        return hook

    def region(self, name: str):
        if not self.active:
            return contextlib.nullcontext()
        return torch.autograd.profiler.record_function(f"harness::{name}")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def maybe_start(self, step: int) -> None:
        if self.done or self.active or step != self.start:
            return
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sync()
        self.prof = torch.profiler.profile(activities=acts, record_shapes=False, profile_memory=False)
        self.prof.__enter__()
        self.active = True
        self.t0 = time.time()

    def maybe_stop(self, step: int) -> None:
        if self.active and step >= self.start + self.n:
            self._stop(step)

    def finish(self, step: int) -> None:
        if self.active:
            self._stop(step)
        for h in self._handles:
            h.remove()
        if not self.done:
            print(f"[profile] run ended at step {step} before profiling started (start={self.start})", file=sys.stderr)

    def _stop(self, step: int) -> None:
        self._sync()
        wall = time.time() - self.t0
        self.prof.__exit__(None, None, None)
        self.active = False
        self.done = True
        n_steps = step - self.start
        self.report(self.prof, wall, n_steps)

    def report(self, prof, wall: float, n_steps: int) -> dict[str, Any]:
        cuda = self.device.type == "cuda"
        ka = prof.key_averages()
        t = (lambda e: _dev_time(e)) if cuda else (lambda e: float(e.cpu_time_total))
        per_step = lambda us: us / 1000.0 / max(n_steps, 1)  # noqa: E731  (us total -> ms per step)
        regions = {e.key.split("::", 1)[1]: per_step(t(e)) for e in ka if e.key.startswith("harness::")}
        modules = {e.key.split("::", 1)[1]: per_step(t(e)) for e in ka if e.key.startswith("model::")}
        fwd = sum(modules.values())
        if "forward_backward" in regions:
            modules["backward_and_other (fb - fwd modules)"] = max(0.0, regions["forward_backward"] - fwd)
        ops = sorted((e for e in ka if "::" in e.key and not e.key.startswith(("harness::", "model::"))),
                     key=t, reverse=True)[:15]
        step_s = wall / max(n_steps, 1)
        tok_s = self.tps / step_s
        out: dict[str, Any] = {
            "device": torch.cuda.get_device_name(self.device) if cuda else "cpu",
            "time_basis": "device (CUDA kernel) time" if cuda else "CPU time",
            "profiled_steps": n_steps, "step_time_s": step_s, "tokens_per_step": self.tps, "tokens_per_s": tok_s,
            "flops_per_token": self.fpt, "achieved_tflops": self.fpt * tok_s / 1e12,
            "mfu": (self.fpt * tok_s / (self.peak * 1e12)) if cuda else None, "peak_tflops": self.peak,
            "regions_ms_per_step": regions, "forward_modules_ms_per_step": modules,
            "top_ops_ms_per_step": [{"op": e.key, "ms": per_step(t(e)), "calls": int(e.count)} for e in ops],
            "peak_mem_bytes": int(torch.cuda.max_memory_allocated(self.device)) if cuda else None,
            "peak_mem_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)) if cuda else None,
            "note": "profiler overhead inflates step time; use result.json tokens_per_s for throughput",
        }
        (self.out / "profile.json").write_text(json.dumps(out, indent=2))
        md = [f"# Profile ({out['device']}, {n_steps} steps, {out['time_basis']})", "",
              f"- step time {step_s * 1000:.1f} ms, {tok_s:,.0f} tokens/s, {out['achieved_tflops']:.2f} TFLOP/s"
              + (f", MFU {out['mfu'] * 100:.1f}% of {self.peak:g} TFLOPS" if cuda else ""),
              f"- peak memory: {out['peak_mem_bytes']}", "", "| region | ms/step |", "|---|---:|"]
        md += [f"| {k} | {v:.2f} |" for k, v in regions.items()]
        md += ["", "| forward module | ms/step |", "|---|---:|"]
        md += [f"| {k} | {v:.2f} |" for k, v in modules.items()]
        md += ["", "| top op | ms/step | calls |", "|---|---:|---:|"]
        md += [f"| `{o['op']}` | {o['ms']:.2f} | {o['calls']} |" for o in out["top_ops_ms_per_step"]]
        (self.out / "profile.md").write_text("\n".join(md) + "\n")
        print(f"[profile] {step_s * 1000:.1f} ms/step, {tok_s:,.0f} tok/s -> {self.out / 'profile.md'}", flush=True)
        return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from sprint1.harness import train

    argv = list(sys.argv[1:] if argv is None else argv)
    ns, _ = train.build_parser().parse_known_args(argv)
    extra: list[str] = []
    if ns.steps is None and ns.tokens is None:
        extra += ["--steps", str(ns.profile_start + ns.profile_steps)]
    if "--eval_every" not in argv:
        extra += ["--eval_every", "0"]
    if "--final_eval_tokens" not in argv:
        extra += ["--final_eval_tokens", "65536"]
    return train.main(argv + extra + ["--profile"])


if __name__ == "__main__":
    main()
