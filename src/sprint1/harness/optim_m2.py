"""M2 optimizer: Adam with a *linear* (row-factored) second moment for sparse tables.

Theory: sprint1/math/m2/THEORY.md. Per-coordinate Adam updates a table row that occurs once every K steps with a
soft limiter g / sqrt(g^2 + S) whose knee falls with K (Theorem 2); rows with beta2^K << 1 are updated by sign(g),
which by Bussgang's theorem distorts the context-specific part of the gradient. A second moment that is one scalar
per row (mode "row") or a rank-1 row x column factorisation (mode "factored", Adafactor's v) is linear in the current
gradient's direction, so it has no hard-limiter distortion, while keeping Adam's per-row magnitude normalisation.

    RowFactoredAdam(params, lr, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, mode="factored", null="none")

mode   "adam"      per-coordinate second moment (bit-for-bit the AdamW update; the equal-code reference arm)
       "row"       v_hat[v, j] = R[v]                         R[v] = EMA_beta2(mean_j g[v, j]^2)
       "factored"  v_hat[v, j] = R[v] * C[j] / mean(C)        C[j] = EMA_beta2(mean_v g[v, j]^2)
null   "none"      as above
       "global"    the per-row statistic R[v] is replaced by its mean over rows (one second moment per table): same
                   code and cost, still linear, but the per-row information the mechanism uses is destroyed
Non-2-D parameters always use the per-coordinate path. Decoupled weight decay as in AdamW. Every param group keeps
"initial_lr", "kind" ("adam") and "name" for the harness schedule.
"""
from __future__ import annotations

import math

import torch


class RowFactoredAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.8, 0.95), eps: float = 1e-10, weight_decay: float = 0.0,
                 mode: str = "factored", null: str = "none"):
        if mode not in ("adam", "row", "factored"):
            raise ValueError(f"mode must be adam|row|factored, got {mode!r}")
        if null not in ("none", "global"):
            raise ValueError(f"null must be none|global, got {null!r}")
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, mode=mode, null=null,
                        kind="adam")
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
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            mode, null = group["mode"], group["null"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                sdt = p.dtype if p.dtype in (torch.float32, torch.float64) else torch.float32  # state precision
                g = p.grad.to(sdt)
                st = self.state[p]
                factored = mode != "adam" and p.ndim == 2
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, dtype=sdt)
                    if factored:
                        st["row_sq"] = torch.zeros(p.size(0), dtype=sdt, device=p.device)
                        if mode == "factored":
                            st["col_sq"] = torch.zeros(p.size(1), dtype=sdt, device=p.device)
                    else:
                        st["exp_avg_sq"] = torch.zeros_like(p, dtype=sdt)
                st["step"] += 1
                t = st["step"]
                m = st["exp_avg"]
                m.lerp_(g, 1 - b1)
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                g2 = g * g
                if factored:
                    R = st["row_sq"]
                    R.mul_(b2).add_(g2.mean(dim=1), alpha=1 - b2)
                    r = R.mean().expand_as(R) if null == "global" else R
                    vhat = (r / bc2).unsqueeze(1)
                    if mode == "factored":
                        C = st["col_sq"]
                        C.mul_(b2).add_(g2.mean(dim=0), alpha=1 - b2)
                        cf = C / C.mean().clamp_min(1e-38)  # bias factors cancel in the ratio
                        vhat = vhat * cf.unsqueeze(0)
                    denom = vhat.sqrt().add_(eps)
                else:
                    v = st["exp_avg_sq"]
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)  # same rounding as torch's AdamW
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m.to(p.dtype), denom.to(p.dtype), value=-lr / bc1)
        return loss


def build_m2_optimizer(model, lrs: dict, *, m2_mode: str = "factored", m2_groups=("head", "embed"),
                       m2_null: str = "none", m2_beta2: float | None = None, muon_momentum: float = 0.95,
                       muon_wd: float = 0.0, adam_betas=(0.8, 0.95), adam_eps: float = 1e-10, adam_wd: float = 0.0,
                       ns_steps: int = 5):
    """Same split as harness.optim.build_optimizer (Muon for hidden matrices; embed / head / scalar Adam groups),
    with the groups named in m2_groups handled by RowFactoredAdam(m2_mode). lrs keys: muon, adam_embed, adam_head,
    adam_scalar."""
    from sprint1.harness.optim import CombinedOptimizer, Muon, default_param_groups, split_adam

    groups = default_param_groups(model)
    seen: set[int] = set()
    muon_params, adam_params = [], []
    for p in groups["muon"]:
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            muon_params.append(p)
    for p in list(groups["adam"]) + list(model.parameters()):
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            adam_params.append(p)
    opts = []
    if muon_params:
        opts.append(Muon(muon_params, lr=lrs["muon"], momentum=muon_momentum, weight_decay=muon_wd,
                         ns_steps=ns_steps))
    buckets = split_adam(model, adam_params)
    b1, b2 = float(adam_betas[0]), float(adam_betas[1])
    m2_groups = tuple(m2_groups)
    std, fac = [], []
    for k, ps in buckets.items():
        if not ps:
            continue
        gdict = dict(params=ps, lr=lrs[f"adam_{k}"], initial_lr=lrs[f"adam_{k}"], kind="adam", name=k)
        if k in m2_groups:
            gdict["betas"] = (b1, float(m2_beta2) if m2_beta2 is not None else b2)
            fac.append(gdict)
        else:
            std.append(gdict)
    if fac:
        opts.append(RowFactoredAdam(fac, betas=(b1, b2), eps=adam_eps, weight_decay=adam_wd, mode=m2_mode,
                                    null=m2_null))
    if std:
        fused = all(p.is_cuda for g in std for p in g["params"])
        opts.append(torch.optim.AdamW(std, betas=(b1, b2), eps=adam_eps, weight_decay=adam_wd,
                                      fused=fused if fused else None))
    return CombinedOptimizer(opts)


# --------------------------------------------------------------------------------------------------
# closed forms used by the tests (THEORY.md Theorem 1)
# --------------------------------------------------------------------------------------------------
def adam_impulse_displacement(b1: float, b2: float) -> float:
    """D0: total displacement / lr of an isolated impulse under per-coordinate Adam (t0 -> inf, eps -> 0)."""
    rho = b1 / math.sqrt(b2)
    return (1 - b1) / ((1 - rho) * math.sqrt(1 - b2))
