"""Temporal self-distillation for the BabyLM harness (babylm.harness.train --sd_mode). KNOWN, not new: temporal
ensembling (Laine & Aila 2017) and progressive self-knowledge distillation (PS-KD, Kim et al. 2021) applied to
causal LM pre-training on a fixed 10M-word corpus. Off by default; off leaves train.py's port path untouched.

--sd_mode temporal
    Every training forward already computes the model's next-token distribution q at every target position p of
    the stream. We keep, per position p, a top-k sparse target t_p (idx int16, mass fp16 and a visit count: 4k + 1
    bytes per position, k = 16 at 16M positions = 1.04 GB) built from those same forwards, so there is no extra
    forward and no extra exposure (critic R0). After the forward that visits p (training mode, before the optimizer step):

        first visit:  t_p <- top-k(q)
        later visits: t_p <- top-k over (idx(t_p) U top-k(q)) of  d * t_p(v) + (1 - d) * q(v)

    q(v) is exact for every candidate (the full softmax is there); t_p(v) = 0 outside its stored support, so t_p is
    an EMA over the epochs' coarsened predictions, kept as a sub-distribution whose remaining mass is the tail
    1 - sum t_p. d = 0: last visit's prediction (PS-KD); d = 1: the first visit's prediction is frozen.
    --sd_write_until E stops the merging from epoch E on, so late (memorising) epochs never enter the target.
    From epoch --sd_start_epoch on (tokens_seen / split tokens), positions that have a target add
        alpha * KL(t_p || q),  over k + 1 bins: the k target tokens and one tail bin (exact KL between the coarsened
                               distributions, so the truncation never distorts the target's entropy)
    to the per-token CE (loss = mean over targets of CE + alpha * has_target * KL). The KL for p always uses the
    target from earlier visits (read before the write). Each position is visited once per epoch by both samplers;
    at the stream sampler's epoch rollover a few windows of epoch e and e+1 can share a step, and then the later
    write wins (a negligible fraction of positions).

--sd_mode smooth (the null twin)
    From the same start epoch, loss = (1 + alpha) * CE_label_smoothing(eps). Its gradient is (1 + alpha)(q - m0)
    with m0 = (1 - eps) onehot + eps uniform, the same form as temporal's (1 + alpha)(q - m), m = (onehot + alpha
    t)/(1 + alpha). eps is chosen so that H(m0) = mean H(m) logged by the temporal run (sd_H, the tail spread
    uniformly over the tokens outside the support, as label smoothing spreads it):
        python -m babylm.harness.self_distill --null_eps_from babylm/runs/<temporal run>

--sd_mode smooth_unigram (the second null, critic ref-0060)
    The same, smoothing toward the training split's unigram distribution u instead of uniform:
    loss = (1 + alpha) * ((1 - eps) CE + eps * CE(u, q)); eps matched to mean H(m) with the label drawn from u:
        python -m babylm.harness.self_distill --null_eps_from babylm/runs/<temporal run> --null unigram

Logged per step (log.jsonl): sd_on, sd_ce, sd_kl (mean over positions with a target), sd_cov (share of targets
that had a cached target), sd_H (mean entropy of m), sd_t_label (mean target mass on the true next token), sd_off
(mean mass m moves off the label; sd_mass = mean cached mass sum t_p, so 1 - sd_mass is the tail bin, alpha (1 - t_label) / (1 + alpha); `--match off` calibrates a null to it instead).
The SD forward calls model.hidden() and model.logits() (sprint-1 split API), not model.forward(), so a model's
aux_loss is not added (the frozen baselines have none). Resume: the cache is saved next to resume.pt (sd_cache.pt).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("none", "temporal", "smooth", "smooth_unigram")
Q_TAIL_MAX = 1.0 - 1e-6


# ------------------------------------------------------------------------------------------------
# target positions
# ------------------------------------------------------------------------------------------------
def stream_positions(starts: np.ndarray, T: int) -> np.ndarray:
    """EpochSampler windows: target j of the window starting at s is stream position s + 1 + j."""
    return np.asarray(starts, dtype=np.int64)[:, None] + 1 + np.arange(T, dtype=np.int64)


def plan_positions(y: torch.Tensor, tpos: np.ndarray) -> np.ndarray:
    """PlanSampler batches: tpos lists the stream position of every real target in row-major order (each row's real
    targets are a prefix), so it fills the y >= 0 cells in order; -1 elsewhere."""
    valid = (y >= 0).numpy()
    pos = np.full(valid.shape, -1, dtype=np.int64)
    if int(valid.sum()) != len(tpos):
        raise RuntimeError(f"plan batch has {int(valid.sum())} targets but {len(tpos)} positions")
    pos[valid] = tpos
    return pos


# ------------------------------------------------------------------------------------------------
# math
# ------------------------------------------------------------------------------------------------
def binned_kl(logq: torch.Tensor, t_idx: torch.Tensor, t_p: torch.Tensor) -> torch.Tensor:
    """KL(t || q) per row over k + 1 bins: the k (distinct) target tokens and the tail (everything else).
    logq [M, V] log-probs, t_idx [M, k] long, t_p [M, k] masses with sum <= 1. Returns [M]."""
    t = t_p.float().clamp_min(0.0)
    tail = (1.0 - t.sum(-1)).clamp_min(0.0)
    lq = logq.gather(1, t_idx)
    log_q_tail = torch.log1p(-lq.exp().sum(-1).clamp(max=Q_TAIL_MAX))
    return (torch.special.xlogy(t, t) - t * lq).sum(-1) + torch.special.xlogy(tail, tail) - tail * log_q_tail


def temporal_core(zf: torch.Tensor, yf: torch.Tensor, idx_o: torch.Tensor, p_o: torch.Tensor, has: torch.Tensor,
                  alpha: float, use_kl: bool, k: int):
    """The temporal loss on flat logits zf [M, V] (torch.compile'd under --compile): (loss, CE, KL sum over rows with a
    target, this visit's top-k log-probs and ids, its log-probs at the old target ids); the last three detached."""
    ce = F.cross_entropy(zf, yf, ignore_index=-1)
    logq = F.log_softmax(zf, -1)
    loss, kl_sum = ce, torch.zeros((), device=zf.device)
    if use_kl:
        kl = torch.where(has, binned_kl(logq, idx_o, p_o), torch.zeros((), device=zf.device))
        loss = ce + alpha * kl.sum() / (yf >= 0).sum().clamp_min(1)
        kl_sum = kl.detach().sum()
    lq = logq.detach()
    lp_n, idx_n = lq.topk(k, -1)
    return loss, ce.detach(), kl_sum, lp_n, idx_n, lq.gather(1, idx_o)


def mixed_entropy(y: torch.Tensor, t_idx: torch.Tensor, t_p: torch.Tensor, alpha: float, V: int) -> torch.Tensor:
    """Entropy of m = (onehot(y) + alpha * t~) / (1 + alpha) per row, t~ = the binned target with its tail spread
    uniformly over the V - k tokens outside its support. Returns [M] (float64)."""
    t = t_p.double().clamp_min(0.0)
    k = t.shape[1]
    tail = (1.0 - t.sum(-1)).clamp_min(0.0)
    hit = t_idx == y[:, None]
    m_in = (alpha * t + hit.double()) / (1 + alpha)
    u = tail / max(1, V - k)
    a = alpha * u / (1 + alpha)
    miss = (~hit.any(-1)).double()  # the label sits outside the support and also gets its tail share
    b = (alpha * u + 1) / (1 + alpha)
    xl = torch.special.xlogy
    return -(xl(m_in, m_in).sum(-1) + (V - k - miss) * xl(a, a) + miss * xl(b, b))


def ls_entropy(eps: float, V: int) -> float:
    """Entropy of the label-smoothing target (1 - eps) onehot + eps / V."""
    hi, lo = 1 - eps + eps / V, eps / V
    return -(hi * math.log(hi) if hi > 0 else 0.0) - ((V - 1) * lo * math.log(lo) if lo > 0 else 0.0)


def unigram(tokens: np.ndarray, V: int) -> np.ndarray:
    c = np.bincount(np.asarray(tokens, dtype=np.int64), minlength=V)[:V].astype(np.float64)
    return c / c.sum()


def unigram_ls_entropy(eps: float, u: np.ndarray) -> float:
    """E_{y ~ u} H((1 - eps) onehot(y) + eps u)."""
    xl = lambda a: np.where(a > 0, a * np.log(np.where(a > 0, a, 1.0)), 0.0)  # noqa: E731
    s_all = xl(eps * u).sum()
    hy = 1 - eps + eps * u
    return float((u * -(xl(hy) + s_all - xl(eps * u))).sum())


def unigram_eps_for_entropy(H: float, u: np.ndarray) -> float:
    """Smallest eps with unigram_ls_entropy(eps) = H: the entropy is concave in eps, so bisect on [0, argmax]."""
    if H <= 0:
        return 0.0
    grid = np.linspace(0.0, 1.0, 1001)
    hi = float(grid[int(np.argmax([unigram_ls_entropy(e, u) for e in grid]))])
    lo = 0.0
    if H >= unigram_ls_entropy(hi, u):
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if unigram_ls_entropy(mid, u) < H else (lo, mid)
    return (lo + hi) / 2


def ls_eps_for_entropy(H: float, V: int) -> float:
    """eps in [0, 1 - 1/V] with ls_entropy(eps) = H (monotone there); bisection."""
    if H <= 0:
        return 0.0
    lo, hi = 0.0, 1.0 - 1.0 / V
    if H >= ls_entropy(hi, V):
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2
        if ls_entropy(mid, V) < H:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------------------------------------
# cache
# ------------------------------------------------------------------------------------------------
class TargetCache:
    """Per-position top-k targets: idx int16 [N, k], mass fp16 [N, k], visit count uint8 [N] (saturates at 255)."""

    def __init__(self, n_positions: int, k: int, decay: float, vocab: int, device: torch.device | str = "cpu"):
        if vocab > 32767:
            raise ValueError(f"vocab {vocab} does not fit the int16 index cache")
        if not 1 <= k <= vocab:
            raise ValueError(f"k={k} must be in [1, vocab]")
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay={decay} must be in [0, 1]")
        self.N, self.k, self.decay, self.V = int(n_positions), int(k), float(decay), int(vocab)
        self.device = torch.device(device)
        # one spare row at index N takes the reads and writes of padded targets (no host sync to drop them)
        self.idx = torch.zeros((self.N + 1, self.k), dtype=torch.int16, device=self.device)
        self.p = torch.zeros((self.N + 1, self.k), dtype=torch.float16, device=self.device)
        self.cnt = torch.zeros(self.N + 1, dtype=torch.uint8, device=self.device)

    @property
    def nbytes(self) -> int:
        return self.N * self.k * 4 + self.N

    def read(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """pos [M] long (>= 0) -> (idx [M, k] long, mass [M, k] float32, has [M] bool); rows without a target get
        idx = arange(k) (distinct, so binned_kl stays finite) and mass 0."""
        pos = pos.to(self.device)
        has = self.cnt[pos] > 0
        ar = torch.arange(self.k, device=self.device).expand(len(pos), self.k)
        idx = torch.where(has[:, None], self.idx[pos].long(), ar)
        p = torch.where(has[:, None], self.p[pos].float(), torch.zeros((), device=self.device))
        return idx, p, has

    @torch.no_grad()
    def update(self, pos: torch.Tensor, logq: torch.Tensor, old: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
               | None = None) -> None:
        """Merge this visit's prediction (logq [M, V] log-probs of the rows at pos [M]) into the targets."""
        idx_o, p_o, has = old if old is not None else self.read(pos)
        logq = logq.detach().float()
        idx_o = idx_o.to(logq.device)
        lp_n, idx_n = logq.topk(self.k, -1)
        self.merge(pos, lp_n, idx_n, logq.gather(1, idx_o), (idx_o, p_o, has))

    @torch.no_grad()
    def merge(self, pos: torch.Tensor, lp_n: torch.Tensor, idx_n: torch.Tensor, lq_old: torch.Tensor,
              old: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """update() given this visit's top-k (log-probs lp_n, ids idx_n) and its log-probs at the old ids lq_old."""
        pos = pos.to(self.device)
        dev = lp_n.device
        idx_o, p_o, has = (t.to(dev) for t in old)
        p_n = lp_n.float().exp()
        d = self.decay
        w_old = d * p_o + (1 - d) * lq_old.float().exp()
        dup = (idx_n[:, :, None] == idx_o[:, None, :]).any(-1)
        w_new = torch.where(dup, torch.full_like(p_n, -1.0), (1 - d) * p_n)
        w_old = torch.where(has[:, None], w_old, torch.full_like(w_old, -1.0))  # first visit: the new top-k only
        w_new = torch.where(has[:, None], w_new, p_n)
        w = torch.cat([w_old, w_new], 1)
        ids = torch.cat([idx_o, idx_n], 1)
        wk, j = w.topk(self.k, -1)
        self.idx[pos] = ids.gather(1, j).to(torch.int16).to(self.device)
        self.p[pos] = wk.clamp_min(0.0).to(torch.float16).to(self.device)
        self.cnt[pos] = (self.cnt[pos].int() + 1).clamp(max=255).to(torch.uint8)

    def state_dict(self) -> dict:
        return {"idx": self.idx.cpu(), "p": self.p.cpu(), "cnt": self.cnt.cpu(),
                "meta": {"N": self.N, "k": self.k, "decay": self.decay, "V": self.V}}

    def load_state_dict(self, d: dict) -> None:
        if d["meta"] != {"N": self.N, "k": self.k, "decay": self.decay, "V": self.V}:
            raise ValueError(f"sd cache is for {d['meta']}, not N={self.N} k={self.k} decay={self.decay} V={self.V}")
        self.idx.copy_(d["idx"])
        self.p.copy_(d["p"])
        self.cnt.copy_(d["cnt"])


# ------------------------------------------------------------------------------------------------
# harness object
# ------------------------------------------------------------------------------------------------
def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("temporal self-distillation (babylm.harness.self_distill; off by default)")
    g.add_argument("--sd_mode", default="none", choices=MODES, help="temporal = the attack; smooth = its null twin")
    g.add_argument("--sd_alpha", type=float, default=0.5, help="weight of KL(target || model) (smooth: (1+a) x CE_ls)")
    g.add_argument("--sd_k", type=int, default=16, help="top-k entries cached per position (4k bytes per position)")
    g.add_argument("--sd_decay", type=float, default=0.6, help="EMA decay d of the cached targets over epochs")
    g.add_argument("--sd_start_epoch", type=float, default=1.0,
                   help="add the KL (or the smoothing) from this epoch (tokens_seen / split tokens) on; 1.0 = epoch 2")
    g.add_argument("--sd_write_until", type=float, default=math.inf,
                   help="stop merging predictions into the cache from this epoch on (targets then stay the EMA of the "
                        "earlier epochs' predictions, before late-epoch memorisation)")
    g.add_argument("--sd_smooth_eps", type=float, default=0.0, help="smooth mode: label smoothing eps")
    g.add_argument("--sd_cache_device", default="same", choices=["same", "cpu"],
                   help="where the temporal cache lives (same = the training device)")


class SelfDistill:
    def __init__(self, args, n_positions: int, vocab: int, device: torch.device, compile_: bool = False,
                 tokens: np.ndarray | None = None):
        self.mode, self.alpha = args.sd_mode, float(args.sd_alpha)
        self.start = float(args.sd_start_epoch)
        self.write_until = float(getattr(args, "sd_write_until", math.inf))
        self.eps = float(args.sd_smooth_eps)
        self.V = int(vocab)
        if self.mode not in MODES[1:]:
            raise ValueError(f"sd_mode {self.mode!r}")
        self.u = None
        if self.mode == "smooth_unigram":
            if tokens is None:
                raise ValueError("smooth_unigram needs the training tokens")
            self.u = torch.from_numpy(unigram(tokens, vocab)).float().to(device)
        if self.alpha < 0 or not 0.0 <= self.eps < 1.0:
            raise ValueError("--sd_alpha must be >= 0 and --sd_smooth_eps in [0, 1)")
        self.cache = None
        if self.mode == "temporal":
            cdev = device if args.sd_cache_device == "same" else torch.device("cpu")
            self.cache = TargetCache(n_positions, args.sd_k, args.sd_decay, vocab, cdev)
        self.device = device
        self._logits = None
        self._core = None
        self.compile_ = compile_
        self._stats: dict[str, tuple] = {}

    def describe(self) -> dict:
        return {"mode": self.mode, "alpha": self.alpha, "start_epoch": self.start,
                "write_until": None if math.isinf(self.write_until) else self.write_until,
                "k": self.cache.k if self.cache else None, "decay": self.cache.decay if self.cache else None,
                "smooth_eps": self.eps if self.mode.startswith("smooth") else None,
                "cache_bytes": self.cache.nbytes if self.cache else 0,
                "cache_device": str(self.cache.device) if self.cache else None}

    def _fwd(self, raw_model: nn.Module):
        if self._logits is None:
            f = lambda x: raw_model.logits(raw_model.hidden(x))  # noqa: E731
            self._logits = torch.compile(f) if self.compile_ else f
        return self._logits

    def _acc(self, d: dict) -> None:
        """Per micro-batch: plain values are averaged over micro-batches; (num, den) pairs (tensors allowed, no host
        sync) are summed and divided once per step in pop_stats."""
        for k, v in d.items():
            if isinstance(v, tuple):
                n0, d0 = self._stats.get(k, (0.0, 0.0))
                self._stats[k] = (n0 + v[0], d0 + v[1])
            else:
                s0, c0 = self._stats.get(k, (0.0, 0))
                self._stats[k] = (s0 + v, c0 + 1)

    def pop_stats(self) -> dict[str, float]:
        """The step's statistics (one host sync per step); a ratio with a zero denominator is left out."""
        keys = list(self._stats)
        items = [x for k in keys for x in self._stats[k]]
        dev = next((x.device for x in items if isinstance(x, torch.Tensor)), torch.device("cpu"))
        vals = torch.stack([x.detach().double().reshape(()) if isinstance(x, torch.Tensor)
                            else torch.tensor(float(x), dtype=torch.float64, device=dev) for x in items]).tolist() \
            if items else []
        out = {}
        for i, k in enumerate(keys):
            num, den = vals[2 * i], vals[2 * i + 1]
            if den > 0:
                out[f"sd_{k}"] = num / den
        if "sd_t_label" in out:
            out["sd_off"] = self.alpha * (1 - out["sd_t_label"]) / (1 + self.alpha)  # mass m moves off the label
        self._stats = {}
        return out

    def loss(self, raw_model: nn.Module, x: torch.Tensor, y: torch.Tensor, pos: np.ndarray, epoch: float, amp
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """One causal micro-batch: (loss with graph, plain CE detached for the log). x, y on the device; pos [B, T]
        stream positions (-1 where y < 0). Also merges this forward's predictions into the cache. No host sync
        except for batches with padding (line windows), where the positions are checked against the targets."""
        with amp:
            z = self._fwd(raw_model)(x)
        B, T, V = z.shape
        zf, yf = z.reshape(B * T, V), y.reshape(B * T)
        on = epoch >= self.start
        st: dict = {"on": float(on)}
        valid = yf >= 0
        n_valid = valid.sum().clamp_min(1)
        if self.mode != "temporal":
            ce = F.cross_entropy(zf, yf, ignore_index=-1)
            st["ce"] = ce.detach()
        if self.mode == "smooth":
            loss = (1 + self.alpha) * F.cross_entropy(zf, yf, ignore_index=-1, label_smoothing=self.eps) if on else ce
            self._acc(st)
            return loss, ce.detach()
        if self.mode == "smooth_unigram":
            loss = ce
            if on and self.eps > 0:
                cu = -(F.log_softmax(zf, -1) * self.u).sum(-1)
                cu = torch.where(valid, cu, torch.zeros((), device=z.device)).sum() / n_valid
                loss = (1 + self.alpha) * ((1 - self.eps) * ce + self.eps * cu)
            elif on:
                loss = (1 + self.alpha) * ce
            self._acc(st)
            return loss, ce.detach()
        pf = np.asarray(pos, dtype=np.int64).reshape(-1)
        pad = pf < 0
        if pad.any() and bool((~pad & ~valid.cpu().numpy()).any()):
            raise RuntimeError("sd: target positions do not match the targets")
        # a real target with position -1 has no canonical cache key (babylm.harness.stack: an R2L row, a re-segmented
        # piece, a synthetic token): like padding it reads/writes the spare row, and it keeps its CE but is never distilled
        pw = torch.from_numpy(np.where(pad, self.cache.N, pf))  # padding reads/writes the cache's spare row
        idx_o, p_o, has = (t.to(z.device) for t in self.cache.read(pw))
        has = has & valid
        if pad.any():  # only batches with padding or unkeyed targets pay for this transfer
            has = has & torch.from_numpy(~pad).to(z.device)
        use_kl = on and self.alpha > 0
        if self._core is None:
            self._core = torch.compile(temporal_core) if self.compile_ else temporal_core
        if use_kl:
            loss, ce, kl_sum, lp_n, idx_n, lq_old = self._core(zf, yf, idx_o, p_o, has, self.alpha, True,
                                                               self.cache.k)
        else:  # the port's exact CE graph; the cache inputs come from a gradient-free pass over the same logits
            loss = ce = F.cross_entropy(zf, yf, ignore_index=-1)
            with torch.no_grad():
                _, _, kl_sum, lp_n, idx_n, lq_old = self._core(zf.detach(), yf, idx_o, p_o, has, self.alpha, False,
                                                               self.cache.k)
        st["ce"] = ce.detach()
        n_has = has.sum()
        zero = torch.zeros((), device=z.device)
        if use_kl:
            st["kl"] = (kl_sum, n_has)
        yc = yf.clamp_min(0)
        H = mixed_entropy(yc, idx_o, p_o, self.alpha, V)
        st["H"] = (torch.where(has, H, H.new_zeros(())).sum(), n_has)
        tl = (p_o * (idx_o == yc[:, None])).sum(-1)
        st["t_label"] = (torch.where(has, tl, zero).sum(), n_has)
        st["mass"] = (torch.where(has, p_o.sum(-1), zero).sum(), n_has)  # cached mass (1 - tail): truncation gauge
        st["cov"] = (n_has, valid.sum())
        if epoch < self.write_until:
            self.cache.merge(pw, lp_n, idx_n, lq_old, (idx_o, p_o, has))
        st["write"] = float(epoch < self.write_until)
        self._acc(st)
        return loss, ce.detach()

    def save(self, path: Path) -> None:
        if self.cache is None:
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.cache.state_dict(), tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        if self.cache is None:
            return
        self.cache.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))


# ------------------------------------------------------------------------------------------------
# null-twin calibration
# ------------------------------------------------------------------------------------------------
def null_eps_from_run(run_dir: Path, vocab: int | None = None, null: str = "uniform",
                      data_dir: str | None = None, match: str = "H") -> dict:
    """Mean sd_H over the steps where the temporal KL was on -> the smoothing eps with that entropy (uniform or
    unigram null)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    V = vocab or int(cfg["model_cfg"]["vocab_size"])
    if cfg["args"].get("sd_mode") != "temporal":
        raise SystemExit(f"{run_dir} is not a --sd_mode temporal run")
    key = {"H": "sd_H", "off": "sd_off"}[match]
    H = [r[key] for r in (json.loads(x) for x in (run_dir / "log.jsonl").read_text().splitlines() if x.strip())
         if r.get("type") == "train" and r.get("sd_on") == 1.0 and r.get(key) is not None]
    if not H:
        raise SystemExit("no logged steps with the KL on")
    h = float(np.mean(H))
    # the harness pads V for MNTP; label smoothing spreads over the model's whole output layer, as F.cross_entropy
    if null == "unigram":
        from babylm.data import corpus as C
        tr = C.load_split(cfg["split"], data_dir or cfg["data"]["data_dir"])
        u = unigram(tr.tokens, V)
        # off-label mass of unigram smoothing: eps * (1 - u_y), y ~ u
        eps = unigram_eps_for_entropy(h, u) if match == "H" else min(1.0, h / (1 - float((u * u).sum())))
        mode = "smooth_unigram"
    else:
        eps, mode = (ls_eps_for_entropy(h, V) if match == "H" else min(1.0, h / (1 - 1 / V))), "smooth"
    return {"mean_H" if match == "H" else "mean_off": h, "match": match, "steps": len(H), "vocab": V, "eps": eps, "null": null, "mode": mode,
            "alpha": cfg["args"]["sd_alpha"], "start_epoch": cfg["args"]["sd_start_epoch"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="null-twin calibration for --sd_mode smooth")
    ap.add_argument("--null_eps_from", required=True, help="run dir of a --sd_mode temporal run")
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--null", default="uniform", choices=["uniform", "unigram"])
    ap.add_argument("--data_dir", default=None, help="default: the temporal run's data dir")
    ap.add_argument("--match", default="H", choices=["H", "off"],
                    help="H: same mean target entropy (registered); off: same mean mass moved off the label")
    a = ap.parse_args(argv)
    r = null_eps_from_run(Path(a.null_eps_from), a.vocab, a.null, a.data_dir, a.match)
    print(json.dumps(r, indent=1))
    print(f"null twin: --sd_mode {r['mode']} --sd_alpha {r['alpha']} --sd_start_epoch {r['start_epoch']} "
          f"--sd_smooth_eps {r['eps']:.6f}")


if __name__ == "__main__":
    main()
