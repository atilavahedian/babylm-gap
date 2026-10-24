"""Known levers for the BabyLM attack base (babylm.harness.train flags). Every lever is off by default, and off
leaves the port's code path untouched (the harness only builds these objects when a lever flag is set).

(a) MNTP hybrid (GPT-BERT, Charpentier & Samuel 2024; 2025 BabyLM baseline). A fixed number of the sequences in every
    micro-batch is a masked-next-token sequence: positions j >= 1 of the input are corrupted with probability p
    (80% MASK id, 10% random real token, 10% kept), attention is bidirectional, and the output at j-1 predicts the
    original token at j. The other sequences are ordinary causal LM. One forward pass per sequence, so every window
    still counts once against the exposure cap. Loss = (n_causal * L_causal + n_mntp * L_mntp) / n_seqs per
    micro-batch (each sequence weighs the same; each L is the mean over its own scored targets).
    p falls linearly from p_start to p_end over training (GPT-BERT 2025: 0.3 -> 0.15, random 0.1, keep 0.1).

(b) Window plans. One spec per epoch, "fmt:T" with fmt in {stream, line}, "xN" repeats it:
        "stream:1024"                         the port's windows (same RNG as stream.EpochSampler, so the same
                                              windows in the same order; the port's epoch rollover is kept)
        "line:512"                            every window starts at a line start with BOS
        "line:128x6,line:256x2,line:512x2"    GPT-BERT 2025's 128/256/512 length schedule (60/20/20 %), line-aligned
        "line:512x5,stream:512x5"             mixed formats
    A line window starting at line start s is x = [BOS, t_s .. t_{e-2}], y = [t_s .. t_{e-1}] (padded to T with
    pad / -1), where e is the last line start <= s + T (or s + T when one line is longer than T). Windows of an epoch
    tile the token stream exactly, so every target token is seen exactly once per epoch, as in the port. Each epoch
    shifts the tiling (the first window is cut at a random length) and visits its windows in a random permutation.
    Batch tokens stay constant: batch_seqs = batch_tokens // T. A step never mixes two window lengths: when T changes
    between epochs, the tail windows (< one batch) of the earlier group are dropped and counted in `dropped_windows`.
    Steps = whole batches of the planned windows (never more than `epochs` passes; the port's ceil() can spill a
    final partial batch into an 11th epoch, the plan does not). Stream epoch e draws offset and permutation from
    PCG64([seed, e]) exactly like stream.EpochSampler; line epoch e from PCG64([seed, e, 1]).

(c) EMA of the weights for evaluation: shadow <- d * shadow + (1 - d) * w after every optimizer step, with
    d = min(decay, (1 + step) / (10 + step)). Curve, final and exported weights are the shadow when enabled.

(d) Rare-word loss focusing (attack card A5, docs/babylm/attacks/A5_rare_word_focus.md). --focus_plan E:BETA: from
    epoch E (1-based, by tokens consumed) every causal target outside the focus mask (babylm.data.focus_mask: words
    seen <= 63 times in the split and the 3 words after them) stays in the loss with probability BETA, focus targets
    always; dropped targets become ignore_index (-1) so the loss is the mean over the kept targets. Inputs, data
    order, exposure counters, FLOPs and the model are unchanged; MNTP rows keep their own targets. --focus_null (the
    null twin N1) ignores the mask and keeps every target with the plan's expected kept share. One uniform draw per
    target position from a dedicated generator (saved in resume.pt), identical for arm and twin.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------------------------------------
# (b) window plans
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EpochSpec:
    fmt: str  # "stream" | "line"
    T: int


def parse_plan(plan: str | None, epochs: int, default_T: int) -> list[EpochSpec]:
    """'line:128x6,line:256x2,line:512x2' -> 10 EpochSpecs. A single spec without xN fills every epoch."""
    if not plan:
        return [EpochSpec("stream", default_T)] * epochs
    items = [p.strip() for p in plan.split(",") if p.strip()]
    out: list[EpochSpec] = []
    for it in items:
        m = re.fullmatch(r"(stream|line):(\d+)(?:x(\d+))?", it)
        if not m:
            raise ValueError(f"bad window plan item {it!r} (want fmt:T or fmt:TxN)")
        n = int(m.group(3)) if m.group(3) else (epochs if len(items) == 1 else 1)
        out += [EpochSpec(m.group(1), int(m.group(2)))] * n
    if len(out) != epochs:
        raise ValueError(f"window plan covers {len(out)} epochs, --epochs is {epochs}")
    return out


def newline_ids(tokenizer) -> np.ndarray:
    """Ids whose decoded text ends with a newline (a `tokenizers.Tokenizer`, or anything with .decode(list[int]))."""
    V = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else len(tokenizer)
    out = []
    for i in range(V):
        try:
            s = tokenizer.decode([i])
        except Exception:  # noqa: BLE001
            continue
        if s.endswith("\n"):
            out.append(i)
    return np.asarray(out, dtype=np.int64)


def line_starts(tokens: np.ndarray, nl_ids: np.ndarray, eot_id: int) -> np.ndarray:
    """Positions p that begin a line: p == 0, or tokens[p-1] ends a line / is EOT, and tokens[p] is not EOT
    (a document's EOT stays at the end of the preceding segment)."""
    tok = np.asarray(tokens)
    brk = np.isin(tok, nl_ids) | (tok == eot_id)
    starts = np.flatnonzero(brk[:-1]) + 1
    starts = starts[tok[starts] != eot_id]
    return np.unique(np.concatenate([[0], starts])).astype(np.int64)


def tile_lines(starts: np.ndarray, n_tokens: int, T: int, first_cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy tiling of [0, n_tokens) into windows [s, e) with e - s <= T, e at a line start whenever one exists in
    (s, s + cap]. The first window uses cap = first_cap. Returns (window starts, lengths); windows cover every
    position exactly once."""
    ws, wl = [], []
    s, cap = 0, first_cap
    ends = np.append(starts, n_tokens)
    while s < n_tokens:
        lim = min(s + cap, n_tokens)
        k = np.searchsorted(ends, lim, side="right") - 1  # last boundary <= lim
        e = int(ends[k])
        if e <= s:  # a line longer than cap: cut mid-line
            e = lim
        ws.append(s)
        wl.append(e - s)
        s, cap = e, T
    return np.asarray(ws, dtype=np.int64), np.asarray(wl, dtype=np.int64)


class PlanSampler:
    """Epoch-exact sampler over a window plan. With plan = [stream:T] * E it yields the port's EpochSampler windows
    (same per-epoch RNG, same windows, same order, batches may straddle epochs), minus the port's final spill."""

    def __init__(self, tokens: np.ndarray, plan: list[EpochSpec], batch_tokens: int, seed: int, *,
                 bos_id: int | None = None, pad_id: int = 0, starts: np.ndarray | None = None):
        self.tokens, self.plan = tokens, plan
        self.bos_id, self.pad_id = bos_id, pad_id
        if any(sp.fmt == "line" for sp in plan):
            assert starts is not None and bos_id is not None, "line windows need line starts and a BOS id"
        N = len(tokens)
        self.windows_per_epoch: list[int] = []
        steps: list[tuple[int, int, int]] = []  # (first window index, n windows, T)
        W_s, W_l, W_b, W_ep, W_frac = [], [], [], [], []
        base = 0
        group_T, group_start = None, 0
        self.dropped_windows = 0

        def close_group(end):
            nonlocal group_start
            if group_T is None:
                return
            bs = max(1, batch_tokens // group_T)
            n = end - group_start
            for k in range(n // bs):
                steps.append((group_start + k * bs, bs, group_T))
            self.dropped_windows += n % bs

        for e, sp in enumerate(plan):
            T = sp.T
            if sp.fmt == "stream":
                rng = np.random.Generator(np.random.PCG64([seed, e]))
                off = int(rng.integers(0, T))
                n = (N - 1 - off) // T
                s = off + rng.permutation(n).astype(np.int64) * T
                ln = np.full(n, T, dtype=np.int64)
                bos = np.zeros(n, dtype=bool)
            else:
                rng = np.random.Generator(np.random.PCG64([seed, e, 1]))
                cap = int(rng.integers(1, T + 1))
                s, ln = tile_lines(starts, N, T, cap)
                n = len(s)
                perm = rng.permutation(n)
                s, ln = s[perm], ln[perm]
                bos = np.ones(n, dtype=bool)
            if group_T is not None and T != group_T:
                close_group(base)
                group_start = base
            group_T = T
            W_s.append(s); W_l.append(ln); W_b.append(bos)
            W_ep.append(np.full(n, e, dtype=np.int32)); W_frac.append((np.arange(n) + 1) / n)
            self.windows_per_epoch.append(n)
            base += n
        close_group(base)
        self.w_start = np.concatenate(W_s)
        self.w_len = np.concatenate(W_l)
        self.w_bos = np.concatenate(W_b)
        self.epoch_of = np.concatenate(W_ep)
        self.epoch_pos = np.concatenate(W_frac)
        self.steps = steps
        self.ptr = 0

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def max_T(self) -> int:
        return max(sp.T for sp in self.plan)

    def batch_at(self, i: int) -> tuple[torch.Tensor, torch.Tensor, float, int, np.ndarray]:
        """(x, y, epoch float after the step, T, stream positions of every real target)."""
        w0, n, T = self.steps[i]
        idx = np.arange(w0, w0 + n)
        x = np.full((n, T), self.pad_id, dtype=np.int64)
        y = np.full((n, T), -1, dtype=np.int64)
        tok = self.tokens
        tpos = []
        for r, w in enumerate(idx):
            s, ln = int(self.w_start[w]), int(self.w_len[w])
            if self.w_bos[w]:
                x[r, 0] = self.bos_id
                x[r, 1:ln] = tok[s:s + ln - 1]
                y[r, :ln] = tok[s:s + ln]
                tpos.append(np.arange(s, s + ln))
            else:
                seg = np.asarray(tok[s:s + T + 1], dtype=np.int64)
                x[r] = seg[:-1]
                y[r] = seg[1:]
                tpos.append(np.arange(s + 1, s + T + 1))
        last = w0 + n - 1
        return (torch.from_numpy(x), torch.from_numpy(y), float(self.epoch_of[last] + self.epoch_pos[last]), T,
                np.concatenate(tpos))

    def next(self):
        assert self.ptr < len(self.steps), "sampler exhausted"
        out = self.batch_at(self.ptr)
        self.ptr += 1
        return out

    def target_counts(self) -> np.ndarray:
        """How often each token position is a target over the whole plan (tests and the exposure audit)."""
        c = np.zeros(len(self.tokens), dtype=np.int32)
        for i in range(len(self.steps)):
            np.add.at(c, self.batch_at(i)[4], 1)
        return c

    def state_dict(self) -> dict:
        return {"ptr": self.ptr, "plan": [f"{sp.fmt}:{sp.T}" for sp in self.plan], "N": int(len(self.tokens))}

    def load_state_dict(self, d: dict) -> None:
        if d.get("plan") != [f"{sp.fmt}:{sp.T}" for sp in self.plan] or d.get("N") != len(self.tokens):
            raise ValueError("sampler state is for a different window plan or stream")
        self.ptr = int(d["ptr"])


# ------------------------------------------------------------------------------------------------
# (a) MNTP hybrid
# ------------------------------------------------------------------------------------------------
def mntp_p(step: int, total: int, p_start: float, p_end: float) -> float:
    return p_start + (p_end - p_start) * min(1.0, step / max(1, total - 1))


def mntp_corrupt(x: torch.Tensor, y: torch.Tensor, p: float, mask_id: int, n_real: int, gen: torch.Generator,
                 random_frac: float = 0.1, keep_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """x, y: [B, T] causal inputs/targets with y[:, j-1] == x[:, j] wherever x[:, j] is a real token (y[:, j] >= 0).
    Returns (x_corrupt, y_mntp): positions j >= 1 with a real token are chosen with probability p (at least one per
    row); chosen x[:, j] -> MASK (1 - random - keep), random real token (random), unchanged (keep); y_mntp[:, j-1] =
    original x[:, j] at chosen j, -1 elsewhere."""
    B, T = x.shape
    real = y >= 0
    real[:, 0] = False  # position 0 (BOS or the first stream token) is never a prediction target of j-1
    u = torch.rand((B, T), generator=gen)
    chosen = (u < p) & real
    # at least one chosen position per row (the row's highest-u... lowest-u real position)
    none = ~chosen.any(1)
    if none.any():
        uu = torch.where(real, u, torch.full_like(u, 2.0))
        j = uu.argmin(1)
        fix = none & real.any(1)
        chosen[fix, j[fix]] = True
    r = torch.rand((B, T), generator=gen)
    to_mask = chosen & (r < 1 - random_frac - keep_frac)
    to_rand = chosen & (r >= 1 - random_frac - keep_frac) & (r < 1 - keep_frac)
    xc = x.clone()
    xc[to_mask] = mask_id
    xc[to_rand] = torch.randint(0, n_real, (int(to_rand.sum()),), generator=gen)
    ym = torch.full_like(y, -1)
    ym[:, :-1] = torch.where(chosen[:, 1:], x[:, 1:], torch.full_like(x[:, 1:], -1))
    return xc, ym


def mntp_rows(micro: int, frac: float) -> int:
    """Masked rows per micro-batch (fixed, so compiled shapes are constant); 0 < frac -> at least one."""
    if frac <= 0:
        return 0
    return max(1, min(micro - 1 if frac < 1 else micro, int(round(frac * micro))))


def make_switchable(model: nn.Module) -> int:
    """Swap every sprint-1 CausalSelfAttention for SwitchableAttention (same parameters, same causal output; see
    babylm/models/switchable.py). Returns the number of modules swapped."""
    from babylm.models.switchable import SwitchableAttention
    from sprint1.models.baseline import CausalSelfAttention
    n = 0
    for m in model.modules():
        if type(m) is CausalSelfAttention:
            m.__class__ = SwitchableAttention
            n += 1
    return n


class BidirectionalAttention:
    """Context manager flipping every attention module that has an `is_causal` attribute (make_switchable)."""

    def __init__(self, model: nn.Module, enabled: bool = True):
        self.mods = [m for m in model.modules() if hasattr(m, "is_causal")] if enabled else []

    def __enter__(self):
        for m in self.mods:
            m.is_causal = False
        return self

    def __exit__(self, *exc):
        for m in self.mods:
            m.is_causal = True
        return False


# ------------------------------------------------------------------------------------------------
# (c) EMA
# ------------------------------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        sp = list(self.shadow.parameters())
        mp = [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(sp, mp, 1.0 - d)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)
        self.n += 1
        return d

    def state_dict(self) -> dict:
        return {"decay": self.decay, "n": self.n, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        self.decay, self.n = float(d["decay"]), int(d["n"])
        self.shadow.load_state_dict(d["shadow"])


# ------------------------------------------------------------------------------------------------
# (d) rare-word loss focusing (A5)
# ------------------------------------------------------------------------------------------------
def parse_focus(spec: str, epochs: float) -> tuple[int, float]:
    """'7:0.25' -> (start epoch 7, keep probability 0.25 for non-focus targets)."""
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d*)?|\.\d+)", str(spec).strip())
    if not m:
        raise ValueError(f"bad focus plan {spec!r} (want E:BETA, e.g. 7:0.25)")
    start, beta = int(m.group(1)), float(m.group(2))
    if start < 1 or start > math.ceil(epochs):
        raise ValueError(f"focus plan starts at epoch {start}, the run has {epochs:g} epochs")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"focus plan keep probability {beta} is not in [0, 1]")
    return start, beta


def focus_expected_keep(focus_share: float, beta: float) -> float:
    """Expected share of targets kept: focus targets always, the rest with probability beta."""
    return float(focus_share + beta * (1.0 - focus_share))


def focus_gather(mask: np.ndarray, starts: np.ndarray, T: int) -> torch.Tensor:
    """Bool [B, T] focus flags of the targets (positions start+1 .. start+T) of stream windows."""
    offs = np.arange(1, T + 1, dtype=np.int64)
    return torch.from_numpy(np.asarray(mask[np.asarray(starts)[:, None] + offs], dtype=np.bool_))


def focus_keep(fo: torch.Tensor, real: torch.Tensor, beta: float, gen: torch.Generator,
               null_share: float | None = None) -> torch.Tensor:
    """Bool [B, T]: which real targets stay in the loss. Focus targets always, the others with probability beta;
    with null_share (the twin) the mask is ignored and every target stays with that probability. Exactly one uniform
    draw per position from `gen` in both modes."""
    u = torch.rand(fo.shape, generator=gen)
    kept = (u < null_share) if null_share is not None else (fo | (u < beta))
    return kept & real
