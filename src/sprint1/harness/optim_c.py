"""Invention agent C optimizers (learning-rule territory). Owned by agent C.

One combined optimizer, `COptimizer`, holds both kinds of parameter group used by the sprint-1
recipe ("muon": 2-D hidden matrices with orthogonalised updates; "adam": embeddings, lm head, norms,
scalars). With every C extension off it is plain Muon + AdamW (the equal-code reference arm). Two
extensions, independently switchable and composable:

1. Log-time memory ("lt", idea card sprint1/ideas/C_ltmuon.md).
   A second, slow momentum buffer whose memory horizon GROWS with training time,
       s_t = beta3(t) s_{t-1} + (1 - beta3(t)) g_t,   beta3(t) = 1 - delta / (delta + t)
   is mixed into the step direction with a weight that also grows slowly,
       alpha(t) = alpha0 * (1 + t) ** (1 - kappa).
   Muon group:  update = Orth( nesterov_fast + alpha(t) * s )            (magnitude set by Orth)
   Adam group:  update = (m_hat + alpha(t) * s) / ((1 + alpha(t)) * (sqrt(v_hat) + eps))
   Schedules follow ADana (log-time momentum, Ferbach/Everett/Paquette et al. 2026); the slow-buffer
   mixing follows AdEMAMix (Pagliardini et al. 2024). The Muon (orthogonalised) form is ours.
   lt_mode="logtime"  the bet.
   lt_mode="frozen"   NULL: identical code, buffers, FLOPs and alpha(t), but beta3 frozen at the fast
                      momentum (no growing memory). Isolates the growing horizon as the ingredient.
   lt_mode="fixed"    comparator: AdEMAMix-style constant beta3 (lt_fixed_beta3).

2. Residual-stream-coupled orthogonalisation ("stream", idea card sprint1/ideas/C_streammuon.md).
   Every block writes into the residual stream through an output matrix W_k [d, h_k]; to first order
   the total change of the stream is [dW_1 ... dW_K] [h_1; ...; h_K], one linear map. Orthogonalise
   that concatenation jointly (spectral steepest descent for the whole write map) instead of each W_k
   separately; readers W_k [h_k, d] are stacked along the shared input (stream) side. The joint update
   is rescaled so its total Frobenius energy equals per-matrix Muon's (same LR, different allocation
   across depth and decorrelation across layers); the ratio is measured exactly every
   `stream_calib_every` (25) steps by also running per-matrix Newton-Schulz on that step (~4% extra
   NS cost, charged to the null arms identically). Otherwise same FLOPs as per-matrix Newton-Schulz.
   stream_mode="stream"     writers coupled along d (stream side) + readers coupled along d.
   stream_mode="writers"    writers only.
   stream_mode="transpose"  NULL (equal overhead): same-shape matrices coupled along their hidden
                            side, which is NOT shared between layers (wrong geometry, same mixing).
   stream_mode="alloc"      NULL: per-matrix Orth, energy re-allocated by gradient-norm share
                            (allocation without coupling).
   Closest prior art: TEON (tensorised orthonormalisation across layers, 2026).

Everything is plain PyTorch (bf16 Newton-Schulz on CUDA, fp32 on CPU), no custom kernels.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------------------
# Newton-Schulz orthogonalisation (modded-nanogpt quintic, Jordan et al. 2024)
# --------------------------------------------------------------------------------------------------
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_ns(G: torch.Tensor, steps: int = 5, use_bf16: bool | None = None, eps: float = 1e-7) -> torch.Tensor:
    """Approximate polar factor U V^T of G (last two dims; leading dims are batched)."""
    assert G.ndim >= 2
    if use_bf16 is None:
        use_bf16 = G.is_cuda
    a, b, c = NS_COEFFS
    X = G.bfloat16() if use_bf16 else G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def muon_shape_scale(rows: int, cols: int) -> float:
    """Per-matrix Muon step scale (modded-nanogpt convention): sqrt(max(1, rows/cols))."""
    return max(1.0, rows / cols) ** 0.5


# --------------------------------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------------------------------
def lt_beta3(t: int, mode: str, delta: float, fast_beta: float, fixed_beta3: float, beta3_max: float) -> float:
    """Slow-buffer EMA coefficient at (1-based) step t."""
    if mode == "logtime":
        b = 1.0 - delta / (delta + t)
    elif mode == "frozen":
        b = fast_beta
    elif mode == "fixed":
        b = fixed_beta3
    else:
        raise ValueError(f"unknown lt_mode {mode!r}")
    return min(b, beta3_max)


def lt_alpha(t: int, alpha0: float, kappa: float, alpha_max: float | None) -> float:
    a = alpha0 * (1.0 + t) ** (1.0 - kappa)
    return a if alpha_max is None else min(a, alpha_max)


# --------------------------------------------------------------------------------------------------
# Residual-stream roles
# --------------------------------------------------------------------------------------------------
WRITER_PAT = re.compile(r"(^|\.)(c_proj|proj|o_proj|out_proj|attn_out|fc_out|fc2|w2|down_proj|wo|mlp_out|out)(\.|_|$)")


def infer_stream_roles(model: nn.Module, d_model: int, muon_params: Iterable[nn.Parameter]) -> dict[int, str]:
    """Map id(param) -> "writer" | "reader" | "other" for hidden matrices.

    A model may define `c_stream_roles() -> dict[nn.Parameter, str]` to override. Otherwise names decide
    (output projections are writers), then shapes: [d, h] with h != d -> writer, [h, d] -> reader.
    Batched params (ndim 3, e.g. stacked q/k/v [3, d, d]) get one role for all slices.
    """
    muon_ids = {id(p) for p in muon_params}
    if hasattr(model, "c_stream_roles"):
        return {id(p): r for p, r in model.c_stream_roles().items() if id(p) in muon_ids}
    roles: dict[int, str] = {}
    for name, p in model.named_parameters():
        if id(p) not in muon_ids or p.ndim < 2:
            continue
        rows, cols = p.shape[-2], p.shape[-1]
        if WRITER_PAT.search(name) and rows == d_model:
            roles[id(p)] = "writer"
        elif cols == d_model and rows != d_model:
            roles[id(p)] = "reader"
        elif rows == d_model and cols != d_model:
            roles[id(p)] = "writer"
        elif rows == d_model and cols == d_model:
            roles[id(p)] = "reader"  # square, not named as an output projection: q/k/v style
        else:
            roles[id(p)] = "other"
    return roles


# --------------------------------------------------------------------------------------------------
# The optimizer
# --------------------------------------------------------------------------------------------------
class COptimizer(torch.optim.Optimizer):
    """Muon + AdamW in one torch.optim.Optimizer, with agent C's log-time memory and stream coupling.

    Param groups need "kind" in {"muon", "adam"}; per-group keys (with defaults):
      muon: lr, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=5
      adam: lr, weight_decay=0, betas=(0.8, 0.95), eps=1e-10
      both: lt_mode in {"off","logtime","frozen","fixed"}, lt_delta=8, lt_kappa=0.85, lt_alpha0=1,
            lt_alpha_max=None, lt_fixed_beta3=0.999, lt_beta3_max=0.9999
    Every group gets "initial_lr" (= lr at construction) so external schedules can scale it.
    Stream coupling (muon groups only) is optimizer-wide: stream_mode + roles {id(param): role}.
    """

    def __init__(self, param_groups: list[dict[str, Any]], *, stream_mode: str = "off",
                 roles: dict[int, str] | None = None, ns_bf16: bool | None = None,
                 stream_calib_every: int = 25):
        defaults = dict(lr=1e-3, weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5,
                        betas=(0.8, 0.95), eps=1e-10, lt_mode="off", lt_delta=8.0, lt_kappa=0.85,
                        lt_alpha0=1.0, lt_alpha_max=None, lt_fixed_beta3=0.999, lt_beta3_max=0.9999,
                        kind="adam")
        super().__init__(param_groups, defaults)
        for g in self.param_groups:
            assert g["kind"] in ("muon", "adam"), g["kind"]
            g.setdefault("initial_lr", g["lr"])
            if g["kind"] == "muon":
                for p in g["params"]:
                    assert p.ndim >= 2, "muon groups take matrices (or batched matrices)"
        if stream_mode not in ("off", "stream", "writers", "transpose", "alloc"):
            raise ValueError(f"unknown stream_mode {stream_mode!r}")
        self.stream_mode = stream_mode
        self.roles = roles or {}
        self.ns_bf16 = ns_bf16
        # energy calibration of coupled updates: every `stream_calib_every` steps the per-matrix Muon
        # energy of the coupled set is measured and the joint update is scaled to match it exactly
        # (Newton-Schulz output energy depends on conditioning, so a nominal budget is not enough).
        self.stream_calib_every = int(stream_calib_every)
        self._stream_ratio: dict[str, float] = {}
        self.t = 0  # global step counter shared by all groups (1-based after first step)
        self.total_steps: int | None = None

    # ---------------------------------------------------------------- helpers
    def state_bytes(self) -> int:
        n = 0
        for st in self.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    n += v.numel() * v.element_size()
        return n

    def _slow(self, p: torch.Tensor, g: torch.Tensor, group: dict, t: int) -> tuple[torch.Tensor | None, float]:
        mode = group["lt_mode"]
        if mode == "off":
            return None, 0.0
        st = self.state[p]
        if "slow" not in st:  # fp32 always: (1 - beta3) reaches 1e-4, below bf16 resolution
            st["slow"] = torch.zeros_like(g, dtype=torch.float32)
        fast_beta = group["momentum"] if group["kind"] == "muon" else group["betas"][0]
        b3 = lt_beta3(t, mode, group["lt_delta"], fast_beta, group["lt_fixed_beta3"], group["lt_beta3_max"])
        st["slow"].lerp_(g.float(), 1.0 - b3)
        return st["slow"], lt_alpha(t, group["lt_alpha0"], group["lt_kappa"], group["lt_alpha_max"])

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.t += 1
        t = self.t
        muon_items: list[tuple[torch.Tensor, torch.Tensor, dict]] = []  # (param, direction, group)
        for group in self.param_groups:
            if group["kind"] == "muon":
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(g)
                    buf = st["momentum_buffer"]
                    beta = group["momentum"]
                    buf.lerp_(g, 1.0 - beta)
                    u = g.lerp(buf, beta) if group["nesterov"] else buf.clone()
                    slow, alpha = self._slow(p, g, group, t)
                    if slow is not None:
                        u = u.float().add_(slow, alpha=alpha)
                    muon_items.append((p, u, group))
            else:
                self._adam_group(group, t)
        if muon_items:
            self._apply_muon(muon_items)
        return loss

    def _adam_group(self, group: dict, t: int) -> None:
        b1, b2 = group["betas"]
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "exp_avg" not in st:
                st["exp_avg"] = torch.zeros_like(g)
                st["exp_avg_sq"] = torch.zeros_like(g)
                st["step"] = 0
            st["step"] += 1
            k = st["step"]
            m, v = st["exp_avg"], st["exp_avg_sq"]
            m.lerp_(g, 1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            bc1, bc2 = 1.0 - b1 ** k, 1.0 - b2 ** k
            denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)  # same op order as torch's AdamW
            slow, alpha = self._slow(p, g, group, k)
            if wd:
                p.mul_(1.0 - lr * wd)
            if slow is None:
                p.addcdiv_(m, denom, value=-lr / bc1)
            else:
                num = (m.float() / bc1).add_(slow, alpha=alpha)
                p.addcdiv_(num, denom.float().mul_(1.0 + alpha), value=-lr)

    # ---------------------------------------------------------------- muon application
    def _ns(self, X: torch.Tensor, steps: int) -> torch.Tensor:
        return zeropower_ns(X, steps=steps, use_bf16=self.ns_bf16)

    @staticmethod
    def _apply_update(p: torch.Tensor, upd: torch.Tensor, group: dict, scale: float) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        if wd:
            p.mul_(1.0 - lr * wd)
        p.add_(upd.to(p.dtype).view_as(p), alpha=-lr * scale)

    def _apply_muon(self, items: list[tuple[torch.Tensor, torch.Tensor, dict]]) -> None:
        mode = self.stream_mode
        if mode == "off":
            for p, u, group in items:
                o = self._ns(u, group["ns_steps"])
                self._apply_update(p, o, group, muon_shape_scale(p.size(-2), p.size(-1)))
            return
        # split into coupled sets; everything not covered falls back to per-matrix Muon
        writers = [it for it in items if self.roles.get(id(it[0])) == "writer"]
        readers = [it for it in items if self.roles.get(id(it[0])) == "reader"]
        if mode == "writers":
            readers = []
        coupled_ids = {id(it[0]) for it in writers + readers}
        for p, u, group in items:
            if id(p) not in coupled_ids:
                o = self._ns(u, group["ns_steps"])
                self._apply_update(p, o, group, muon_shape_scale(p.size(-2), p.size(-1)))
        if mode in ("stream", "writers"):
            self._coupled(writers, cat_dim=-1, key="writers")   # [d, sum h]: shared output (stream) side
            self._coupled(readers, cat_dim=-2, key="readers")   # [sum h, d]: shared input (stream) side
        elif mode == "transpose":
            for i, fam in enumerate(_shape_families(writers)):
                self._coupled(fam, cat_dim=-2, key=f"writers{i}")  # stack along d: couples per-layer hidden sides
            for i, fam in enumerate(_shape_families(readers)):
                self._coupled(fam, cat_dim=-1, key=f"readers{i}")
        elif mode == "alloc":
            self._alloc(writers)
            self._alloc(readers)

    def _coupled(self, items: list, cat_dim: int, key: str) -> None:
        """Joint polar factor of the concatenation; Frobenius energy matched to per-matrix Muon."""
        if not items:
            return
        slices: list[torch.Tensor] = []
        owners: list[tuple[int, int]] = []  # (item index, n slices)
        for i, (p, u, _g) in enumerate(items):
            us = u.reshape(-1, u.size(-2), u.size(-1))
            owners.append((i, us.size(0)))
            slices.extend(us.unbind(0))
        steps = items[0][2]["ns_steps"]
        J = torch.cat(slices, dim=cat_dim)
        O = self._ns(J, steps)
        if key not in self._stream_ratio or self.t % self.stream_calib_every == 0:
            e_pm = sum(muon_shape_scale(s.size(-2), s.size(-1)) ** 2 * self._ns(s, steps).float().pow(2).sum()
                       for s in slices)
            self._stream_ratio[key] = float(e_pm / O.float().pow(2).sum().clamp_min(1e-30))
        scale = math.sqrt(self._stream_ratio[key])
        sizes = [s.size(cat_dim) for s in slices]
        parts = list(torch.split(O, sizes, dim=cat_dim))
        k = 0
        for i, n in owners:
            p, _u, group = items[i]
            upd = torch.stack(parts[k:k + n], 0) if p.ndim > 2 else parts[k]
            k += n
            self._apply_update(p, upd, group, scale)

    def _alloc(self, items: list) -> None:
        """Null: per-matrix polar factors, total energy kept, re-allocated by gradient-norm share."""
        if not items:
            return
        norms = torch.stack([u.float().norm() for _p, u, _g in items])
        share = norms / norms.pow(2).sum().sqrt().clamp_min(1e-30)  # unit vector over matrices
        mult = share * math.sqrt(len(items))  # equal shares -> multiplier 1
        for (p, u, group), m in zip(items, mult.tolist()):
            o = self._ns(u, group["ns_steps"])
            self._apply_update(p, o, group, muon_shape_scale(p.size(-2), p.size(-1)) * m)


def _shape_families(items: list) -> list[list]:
    fams: dict[tuple[int, int], list] = {}
    for it in items:
        fams.setdefault(tuple(it[0].shape[-2:]), []).append(it)
    return list(fams.values())


# --------------------------------------------------------------------------------------------------
# Builder used by agent C model files' make_optimizer hook
# --------------------------------------------------------------------------------------------------
DEFAULT_BASE_LRS = {"muon": 0.02, "adam": 0.008}
LT_KEYS = ("lt_mode", "lt_delta", "lt_kappa", "lt_alpha0", "lt_alpha_max", "lt_fixed_beta3", "lt_beta3_max")


def default_param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Contract default: 2-D params outside embeddings / lm head -> muon, everything else -> adam."""
    if hasattr(model, "param_groups"):
        return model.param_groups()
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if p.ndim >= 2 and not any(k in low for k in ("embed", "wte", "wpe", "lm_head", "head", "unembed")):
            muon.append(p)
        else:
            adam.append(p)
    return {"muon": muon, "adam": adam}


ADAM_ALIASES = {
    "adam_embed": ("adam_embed", "embed", "embedding", "embeddings", "wte"),
    "adam_head": ("adam_head", "head", "lm_head", "unembed"),
    "adam_scalar": ("adam_scalar", "scalar", "scalars", "norm", "norms"),
}
KNOWN_LR_KEYS = {"muon", "adam"} | {k for v in ADAM_ALIASES.values() for k in v}


def _lr_lookup(base_lrs: Any, key: str, default: float) -> float:
    if base_lrs is None:
        return default
    if isinstance(base_lrs, (int, float)):
        return float(base_lrs)
    for k in ADAM_ALIASES.get(key, (key,)):
        if k in base_lrs:
            return float(base_lrs[k])
    return default


def build_c_optimizer(model: nn.Module, cfg, base_lrs: Any = None, *, lt_mode: str = "off",
                      stream_mode: str = "off", adam_groups: list[dict] | None = None) -> COptimizer:
    """Construct COptimizer for `model`.

    base_lrs: None | float | dict. Dict keys used: "muon", "adam" and, if present, finer adam keys
      "adam_embed", "adam_head", "adam_scalar" (params are split by name when those are given).
    adam_groups: optional pre-built list of adam param-group dicts (e.g. copied from the baseline's own
      optimizer construction) to guarantee the adam side is identical to the baseline's.
    cfg.extra knobs (all optional): muon_momentum, muon_wd, adam_betas, adam_eps, adam_wd, ns_steps,
      lt_* (see COptimizer), lt_adam (bool, default True: log-time memory on adam groups too),
      stream_mode, ns_bf16.
    """
    ex = dict(getattr(cfg, "extra", {}) or {})
    if isinstance(base_lrs, dict):
        unknown = set(base_lrs) - KNOWN_LR_KEYS
        if unknown:  # never silently fall back to default LRs
            raise ValueError(f"agent C optimizer: unrecognised base_lrs keys {sorted(unknown)}; known {sorted(KNOWN_LR_KEYS)}")
    lt_mode = ex.get("lt_mode", lt_mode)
    stream_mode = ex.get("stream_mode", stream_mode)
    lt_cfg = {k: ex[k] for k in LT_KEYS if k in ex}
    lt_cfg["lt_mode"] = lt_mode
    lt_adam = bool(ex.get("lt_adam", True))
    try:  # use the harness's own grouping so the "off" arm is the baseline optimizer exactly
        from sprint1.harness.optim import default_param_groups as harness_groups
        groups_by_kind = harness_groups(model)
    except ImportError:
        groups_by_kind = default_param_groups(model)
    seen: set[int] = set()
    muon_params, adam_params = [], []
    for p in groups_by_kind.get("muon", []):
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); muon_params.append(p)
    for p in groups_by_kind.get("adam", []):
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); adam_params.append(p)
    for p in model.parameters():  # never silently freeze a parameter the model forgot to list
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); adam_params.append(p)
    groups: list[dict[str, Any]] = []
    if muon_params:
        groups.append(dict(kind="muon", params=muon_params, lr=_lr_lookup(base_lrs, "muon", DEFAULT_BASE_LRS["muon"]),
                           momentum=float(ex.get("muon_momentum", 0.95)), nesterov=bool(ex.get("muon_nesterov", True)),
                           weight_decay=float(ex.get("muon_wd", 0.0)), ns_steps=int(ex.get("ns_steps", 5)), **lt_cfg))
    adam_lt = dict(lt_cfg) if lt_adam else {"lt_mode": "off"}
    if adam_groups is not None:
        for g in adam_groups:
            g = dict(g)
            g["kind"] = "adam"
            g.update(adam_lt)
            groups.append(g)
    elif adam_params:
        betas = tuple(ex.get("adam_betas", (0.8, 0.95)))
        eps = float(ex.get("adam_eps", 1e-10))
        wd = float(ex.get("adam_wd", 0.0))
        fine = isinstance(base_lrs, dict) and any(k in base_lrs for v in ADAM_ALIASES.values() for k in v)
        if fine:
            name_of = {id(p): n for n, p in model.named_parameters()}
            buckets: dict[str, list] = {"adam_embed": [], "adam_head": [], "adam_scalar": []}
            for p in adam_params:
                n = name_of.get(id(p), "").lower()
                if p.ndim < 2:
                    buckets["adam_scalar"].append(p)
                elif "head" in n or "unembed" in n:
                    buckets["adam_head"].append(p)
                else:
                    buckets["adam_embed"].append(p)
            for key, ps in buckets.items():
                if ps:
                    groups.append(dict(kind="adam", params=ps, lr=_lr_lookup(base_lrs, key, _lr_lookup(base_lrs, "adam", DEFAULT_BASE_LRS["adam"])),
                                       betas=betas, eps=eps, weight_decay=wd, name=key[len("adam_"):], **adam_lt))
        else:
            groups.append(dict(kind="adam", params=adam_params, lr=_lr_lookup(base_lrs, "adam", DEFAULT_BASE_LRS["adam"]),
                               betas=betas, eps=eps, weight_decay=wd, **adam_lt))
    roles = None
    if stream_mode != "off":
        d_model = int(getattr(cfg, "n_embd", 0) or 0)
        roles = infer_stream_roles(model, d_model, muon_params)
    return COptimizer(groups, stream_mode=stream_mode, roles=roles, ns_bf16=ex.get("ns_bf16"),
                      stream_calib_every=int(ex.get("stream_calib_every", 25)))
