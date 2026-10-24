"""N-gram hashing for attack A4: the 8-gram copy block, overlap audits and the 13-gram leakage check.

Hash of an n-gram x_0..x_{n-1} (token ids, or word ids for the leakage check): two polynomial hashes
    h1 = fold(h1 * B1 + x_j + 1) mod P1,   h2 = fold(h2 * B2 + x_j + 1) mod P2,   h = h1 * P2 + h2  (< 2^49)
with P1, P2 the largest primes below 2^24 and 2^25. Every intermediate stays below 2^45, so numpy (CPU) and torch
(CPU or GPU) give identical int64 values with no overflow. A collision only adds a false "seen" (a token wrongly
blocked, an n-gram wrongly counted as overlapping): at 16M distinct contexts the chance per lookup is about 3e-8.

Copy-block keys pack the hash of the (n-1)-token context and the next token: key = ctx_hash << 14 | token (token ids
< 16384 = 2^14, so key < 2^63). All n-grams sharing a context form one contiguous run of the sorted key array, so the
set of blocked next tokens for a batch of contexts is two searchsorted calls and one gather (BlockIndex.mask).
"""
from __future__ import annotations

import numpy as np
import torch

P1, P2 = 16_777_213, 33_554_393
TOKEN_BASES = (16_411, 16_417)      # token ids < 16384
WORD_BASES = (1_000_003, 1_000_033)  # word ids (leakage check vocabularies stay well below P1)
TOKEN_BITS = 14
TOKEN_MASK = (1 << TOKEN_BITS) - 1


def _bases(word: bool) -> tuple[int, int]:
    return WORD_BASES if word else TOKEN_BASES


def ngram_hashes(stream: np.ndarray, n: int, word: bool = False) -> np.ndarray:
    """Hash of every n-gram of a 1-D id array: out[i] = hash(stream[i:i+n]), int64 [len - n + 1] (empty if shorter)."""
    s = np.asarray(stream).astype(np.int64)
    m = len(s) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.int64)
    b1, b2 = _bases(word)
    h1 = np.zeros(m, dtype=np.int64)
    h2 = np.zeros(m, dtype=np.int64)
    for j in range(n):
        x = s[j:j + m] + 1
        h1 = (h1 * b1 + x) % P1
        h2 = (h2 * b2 + x) % P2
    return h1 * P2 + h2


def hash_rows(x: np.ndarray, word: bool = False) -> np.ndarray:
    """Hash of each row of an int array [M, n] (same values as ngram_hashes on the concatenated row)."""
    x = np.asarray(x).astype(np.int64)
    b1, b2 = _bases(word)
    h1 = np.zeros(len(x), dtype=np.int64)
    h2 = np.zeros(len(x), dtype=np.int64)
    for j in range(x.shape[1]):
        h1 = (h1 * b1 + x[:, j] + 1) % P1
        h2 = (h2 * b2 + x[:, j] + 1) % P2
    return h1 * P2 + h2


def hash_rows_torch(x: torch.Tensor) -> torch.Tensor:
    """hash_rows for a token tensor [B, n] on any device (int64 arithmetic, no overflow)."""
    b1, b2 = TOKEN_BASES
    x = x.long()
    h1 = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    h2 = torch.zeros_like(h1)
    for j in range(x.shape[1]):
        h1 = torch.remainder(h1 * b1 + x[:, j] + 1, P1)
        h2 = torch.remainder(h2 * b2 + x[:, j] + 1, P2)
    return h1 * P2 + h2


def block_keys(stream: np.ndarray, n: int = 8) -> np.ndarray:
    """Sorted unique copy-block keys of every n-gram of a token stream: ctx_hash(stream[i:i+n-1]) << 14 | stream[i+n-1]."""
    s = np.asarray(stream).astype(np.int64)
    if len(s) < n:
        return np.zeros(0, dtype=np.int64)
    if s.max() > TOKEN_MASK:
        raise ValueError("block keys need token ids < 2^14")
    ctx = ngram_hashes(s, n - 1)[: len(s) - n + 1]
    return np.unique((ctx << TOKEN_BITS) | s[n - 1:])


def in_sorted(sorted_arr: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Membership of each q in a sorted unique array (bool array shaped like q)."""
    q = np.asarray(q)
    if len(sorted_arr) == 0:
        return np.zeros(q.shape, dtype=bool)
    i = np.searchsorted(sorted_arr, q)
    i = np.minimum(i, len(sorted_arr) - 1)
    return sorted_arr[i] == q


class BlockIndex:
    """Sorted copy-block keys on a device. mask(logits, ctx) sets to -inf every token that would complete an n-gram of
    the indexed stream after the last n-1 tokens of each row's context (ctx [B, n-1])."""

    def __init__(self, keys: np.ndarray | torch.Tensor, n: int, device: str | torch.device = "cpu"):
        self.n = int(n)
        self.keys = torch.as_tensor(np.asarray(keys, dtype=np.int64) if not torch.is_tensor(keys) else keys,
                                    dtype=torch.long).to(device)
        self.fallbacks = 0  # rows whose every allowed token was blocked (never seen in practice; left unblocked)

    @classmethod
    def build(cls, stream: np.ndarray, n: int = 8, device: str | torch.device = "cpu") -> "BlockIndex":
        return cls(block_keys(stream, n), n, device)

    def __len__(self) -> int:
        return int(self.keys.numel())

    def ranges(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[lo, hi) runs of the sorted keys holding each row's blocked next tokens (ctx: [B, >= n-1], last n-1 used)."""
        h = hash_rows_torch(ctx[:, -(self.n - 1):]) << TOKEN_BITS
        lo = torch.searchsorted(self.keys, h, right=False)
        hi = torch.searchsorted(self.keys, h | TOKEN_MASK, right=True)
        return lo, hi

    def blocked_sets(self, ctx: torch.Tensor) -> list[set[int]]:
        lo, hi = self.ranges(ctx)
        return [set((self.keys[a:b] & TOKEN_MASK).tolist()) for a, b in zip(lo.tolist(), hi.tolist())]

    def mask(self, logits: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """In place on logits [B, V]; returns the number of tokens removed per row (int64 [B])."""
        lo, hi = self.ranges(ctx)
        cnt = hi - lo
        K = int(cnt.max().item()) if cnt.numel() else 0
        if K == 0:
            return cnt
        B, V = logits.shape
        ar = torch.arange(K, device=logits.device)
        valid = ar[None, :] < cnt[:, None]
        idx = (lo[:, None] + ar[None, :]).clamp(max=len(self) - 1)
        tok = torch.where(valid, self.keys[idx] & TOKEN_MASK, torch.full_like(idx, V))
        m = torch.zeros(B, V + 1, dtype=torch.bool, device=logits.device)
        m.scatter_(1, tok, True)
        m = m[:, :V]
        before = torch.isfinite(logits)
        dead = ~(before & ~m).any(1)  # a row with nothing left: leave it unblocked
        if dead.any():
            self.fallbacks += int(dead.sum())
            m[dead] = False
        logits.masked_fill_(m, float("-inf"))
        return (m & before).sum(1)
