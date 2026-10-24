"""Optimizers for the sprint-1 harness: single-process Muon (modded-nanogpt) + AdamW, and the combiner.

Muon (Jordan et al. 2024): SGD-momentum (Nesterov) whose update for each 2-D hidden matrix is replaced by
its approximate polar factor from 5 quintic Newton-Schulz iterations (bf16 on CUDA, fp32 on CPU), scaled
by sqrt(max(1, rows/cols)) (the speedrun convention, identical to sprint1/harness/optim_c.py's "off" arm).
Everything that is not a 2-D hidden matrix (embeddings, value embeddings, lm head, norms, scalars) uses
AdamW with the speedrun's betas (0.8, 0.95) and eps 1e-10, in three groups: embed / head / scalar.

Every param group carries "initial_lr" and "kind" ("muon" | "adam"); the harness LR schedule sets
group["lr"] = group["initial_lr"] * factor(step) and the Muon momentum warmup sets group["momentum"] on
groups with kind == "muon".
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate U V^T for G = U S V^T (quintic NS from the speedrun; batched over leading dims)."""
    assert G.ndim >= 2
    a, b, c = NS_COEFFS
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Single-process Muon. Use only for 2-D (or batched 2-D) hidden weight matrices."""

    def __init__(self, params, lr: float = 0.05, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, kind="muon")
        super().__init__(params, defaults)
        for g in self.param_groups:
            g.setdefault("initial_lr", g["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mom)
                u = g.lerp(buf, mom) if group["nesterov"] else buf
                o = zeropower_via_newtonschulz5(u, steps=group["ns_steps"])
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(o.view_as(p).to(p.dtype), alpha=-lr * max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
        return loss


class CombinedOptimizer:
    """Steps several optimizers as one; exposes the union of their param_groups (same dict objects)."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = [o for o in optimizers if o is not None]

    @property
    def param_groups(self) -> list[dict]:
        return [g for o in self.optimizers for g in o.param_groups]

    def step(self, closure=None):
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd: dict) -> None:
        for o, s in zip(self.optimizers, sd["optimizers"]):
            o.load_state_dict(s)


def default_param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Contract default: 2-D params outside embeddings / lm head -> muon, everything else -> adam."""
    if hasattr(model, "param_groups"):
        g = model.param_groups()
        return {"muon": list(g.get("muon", [])), "adam": list(g.get("adam", []))}
    muon, adam = [], []
    emb_mods = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    for name, p in model.named_parameters():
        low = name.lower()
        if p.ndim == 2 and id(p) not in emb_mods and not any(k in low for k in ("embed", "wte", "wpe", "lm_head", "unembed")):
            muon.append(p)
        else:
            adam.append(p)
    return {"muon": muon, "adam": adam}


def split_adam(model: nn.Module, adam_params: list[nn.Parameter]) -> dict[str, list[nn.Parameter]]:
    """embed / head / scalar buckets (same rule as optim_c: ndim<2 -> scalar, 'head' in name -> head)."""
    name_of = {id(p): n for n, p in model.named_parameters()}
    out: dict[str, list[nn.Parameter]] = {"embed": [], "head": [], "scalar": []}
    for p in adam_params:
        n = name_of.get(id(p), "").lower()
        if p.ndim < 2:
            out["scalar"].append(p)
        elif "head" in n or "unembed" in n:
            out["head"].append(p)
        else:
            out["embed"].append(p)
    return out


def build_optimizer(model: nn.Module, lrs: dict[str, float], *, muon_momentum: float = 0.95,
                    muon_wd: float = 0.0, adam_betas=(0.8, 0.95), adam_eps: float = 1e-10,
                    adam_wd: float = 0.0, ns_steps: int = 5) -> CombinedOptimizer:
    """lrs keys: muon, adam_embed, adam_head, adam_scalar."""
    groups = default_param_groups(model)
    seen: set[int] = set()
    muon_params, adam_params = [], []
    for p in groups["muon"]:
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); muon_params.append(p)
    for p in groups["adam"]:
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); adam_params.append(p)
    # any trainable parameter the model forgot to list goes to adam (never silently frozen)
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); adam_params.append(p)
    opts: list[Any] = []
    if muon_params:
        opts.append(Muon(muon_params, lr=lrs["muon"], momentum=muon_momentum, weight_decay=muon_wd, ns_steps=ns_steps))
    buckets = split_adam(model, adam_params)
    adam_groups = [dict(params=ps, lr=lrs[f"adam_{k}"], initial_lr=lrs[f"adam_{k}"], kind="adam", name=k)
                   for k, ps in buckets.items() if ps]
    if adam_groups:
        fused = all(p.is_cuda for g in adam_groups for p in g["params"])
        opts.append(torch.optim.AdamW(adam_groups, betas=tuple(adam_betas), eps=adam_eps, weight_decay=adam_wd,
                                      fused=fused if fused else None))
    return CombinedOptimizer(opts)
