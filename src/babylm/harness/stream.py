"""Epoch sampler, stream evaluation windows and word milestones for the BabyLM harness.

EpochSampler (training): epoch e uses a start offset o_e in [0, T) and a permutation of the windows
k = 0..n_e-1, n_e = (N - 1 - o_e) // T, both drawn from np.random.Generator(PCG64([data_seed, e])). Window k reads
inputs tokens[o_e + kT : o_e + kT + T] and targets tokens[o_e + kT + 1 : o_e + kT + T + 1], so the targets of one
epoch tile [o_e + 1, o_e + n_e T] exactly once. Windows may cross documents (BOS separates them). Batches are taken
in permutation order and roll over into the next epoch. State = (epoch, index in the epoch's permutation).

Stream evaluation: the stream (N tokens) is cut into n = ceil((N - 1) / T) non-overlapping windows; window w reads
inputs tokens[wT : wT + T] and targets tokens[wT + 1 : wT + T + 1], clipped at the end (the tail of the last window
is padding: input BOS, target -1). Every token except the first is a target exactly once. Rows are interleaved as in
sprint-1 (sprint1.harness.data.EvalLayout: row r of batch b = window r * n_batches + b; padding rows score nothing).

Milestones (whitespace words seen): 1M..10M every 1M, 20M..100M every 10M, 200M..1B every 100M (BabyLM's checkpoint
schedule), labelled chck_<n>M.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from sprint1.harness.data import EvalLayout

MILESTONES = ([m * 1_000_000 for m in range(1, 11)] + [m * 10_000_000 for m in range(2, 11)]
              + [m * 100_000_000 for m in range(2, 11)])


def milestone_label(m: int) -> str:
    return f"chck_{m // 1_000_000}M"


class EpochSampler:
    def __init__(self, n_tokens: int, seq_len: int, data_seed: int):
        if n_tokens < seq_len + 2:
            raise ValueError(f"stream of {n_tokens} tokens is shorter than one window of {seq_len}")
        self.N, self.T, self.seed = int(n_tokens), int(seq_len), int(data_seed)
        self.epoch, self.idx = 0, 0
        self._plan_epoch = -1
        self._starts = np.zeros(0, dtype=np.int64)

    def plan(self, epoch: int) -> np.ndarray:
        """Window start positions of `epoch` in sampling order."""
        rng = np.random.Generator(np.random.PCG64([self.seed, epoch]))
        off = int(rng.integers(0, self.T))
        n = (self.N - 1 - off) // self.T
        return off + rng.permutation(n).astype(np.int64) * self.T

    def _ensure(self) -> None:
        if self._plan_epoch != self.epoch:
            self._starts = self.plan(self.epoch)
            self._plan_epoch = self.epoch

    def next_starts(self, n: int) -> np.ndarray:
        out = []
        while n > 0:
            self._ensure()
            take = min(n, len(self._starts) - self.idx)
            out.append(self._starts[self.idx:self.idx + take])
            self.idx += take
            n -= take
            if self.idx >= len(self._starts):
                self.epoch += 1
                self.idx = 0
        return np.concatenate(out)

    def epoch_float(self) -> float:
        self._ensure()
        return self.epoch + self.idx / max(1, len(self._starts))

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "idx": self.idx, "seed": self.seed, "N": self.N, "T": self.T}

    def load_state_dict(self, sd: dict) -> None:
        if (sd["seed"], sd["N"], sd["T"]) != (self.seed, self.N, self.T):
            raise ValueError(f"sampler state is for seed/N/T {(sd['seed'], sd['N'], sd['T'])}, "
                             f"not {(self.seed, self.N, self.T)}")
        self.epoch, self.idx = int(sd["epoch"]), int(sd["idx"])
        self._plan_epoch = -1


def gather_windows(tokens: np.ndarray, starts: np.ndarray, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    offs = np.arange(T + 1, dtype=np.int64)
    buf = np.asarray(tokens[starts[:, None] + offs], dtype=np.int64)
    t = torch.from_numpy(buf)
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()


def words_in_targets(word_init: np.ndarray, starts: np.ndarray, T: int) -> int:
    """Number of word-initial tokens among the targets (positions start+1 .. start+T) of the windows."""
    offs = np.arange(1, T + 1, dtype=np.int64)
    return int(word_init[starts[:, None] + offs].sum())


def stream_layout(n_tokens: int, T: int, batch: int) -> EvalLayout:
    n_win = math.ceil((n_tokens - 1) / T)
    return EvalLayout(n_win, batch, math.ceil(n_win / batch), True)


def stream_batches(tokens: np.ndarray, T: int, batch: int, pad_id: int = 0):
    """Yields (x, y, win, tpos): int64 [batch, T] inputs / targets (-1 = not scored), the window index of each row
    (-1 = padding row) and the stream position of each target (int64 [batch, T], -1 where not scored)."""
    N = len(tokens)
    lay = stream_layout(N, T, batch)
    offs = np.arange(T, dtype=np.int64)
    for b in range(lay.n_batches):
        win = lay.windows(b)
        w = np.where(win >= 0, win, 0)
        ipos = w[:, None] * T + offs                    # input positions
        tpos = ipos + 1                                 # target positions
        valid = (tpos < N) & (win >= 0)[:, None]
        x = np.where(ipos < N, np.asarray(tokens[np.minimum(ipos, N - 1)], dtype=np.int64), pad_id)
        y = np.where(valid, np.asarray(tokens[np.minimum(tpos, N - 1)], dtype=np.int64), -1)
        yield (torch.from_numpy(x), torch.from_numpy(y), win, np.where(valid, tpos, -1))
