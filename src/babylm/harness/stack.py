"""Composition of the BabyLM attack flags (babylm.harness.train). Note: docs/babylm/attacks/STACKING.md.

Every attack is off by default and off builds nothing, so any subset of flags is a valid run and the all-off run is
the base bit for bit (babylm/tests/test_stack.py). This module holds the few rules that only exist when attacks
meet: the order of the per-row transforms, the cache key A1 uses under them, and the R0 exposure ledger.

Order inside one causal micro-batch (CPU, before the device; MNTP rows take none of it):
  1. A3 segmentation (--bpe_dropout): epoch e trains on its own re-segmented stream; windows, inputs and labels are
     that stream's pieces (built before training, so this step is just the sampler's choice of stream).
  2. A2 R2L (--noise_r2l): a row is reversed after segmentation, so a reversed row is the reverse of what it trains.
  3. A2 token noise (--noise_mode replace|mask): input ids only, after R2L (the R2L marker is protected); targets clean.
  4. forward; A2 gauss adds embedding noise in a hook on model.embed.
  5. A1 (--sd_mode temporal) reads, distils and writes on that same noised forward: no second (clean) forward, so no
     extra exposure and no extra compute. Its cached target is then an EMA of predictions under independent noise
     draws, as in temporal ensembling (Laine & Aila 2017, which ensembles over augmented inputs).

A1 cache key (canonical position): a target is keyed by its stream position in the canonical (train_10M) token
stream, or by -1 = no key. A -1 target keeps its CE and is neither distilled nor cached. -1 goes to:
  - R2L rows: their predictions condition on the right context, a different distribution from the L2R teacher's;
  - padding and MNTP rows (as before);
  - A3: every piece without a canonical twin of the same byte span (CanonicalKeys); intact tokens keep their key,
    so A1's teacher for an intact token is shared across the epochs' segmentations;
  - A4 (--sources): synthetic rows, and real rows of any split other than the training split (cycle_positions).
    The A1 cache starts empty in a cycle (it is not carried by --init_from), and A1's epoch clock in a cycle is the
    cycle's tokens / training-split tokens.

R0 exposure (critic R0.2/R0.3): words_seen counts word-initial canonical target tokens of the windows fed, before
any transform, once per window per forward. A1 and A2 use the same forward (one view): exposure is unchanged. A3
counts the canonical whitespace words of its streams (word ids carried per piece; never pieces), so A3 changes
tokens_seen and the step count but not words per epoch. A4 keeps its own lineage ledger (result.json exposure);
exposure_r0 restates it as real input words (prior lineage included) + synthetic words + generator words against
--exposure_cap. A3 and A4 do not compose yet (the cycle samples canonical streams): train.py refuses the pair.
"""
from __future__ import annotations

import math

import numpy as np
import torch

ORDER = ("A3 segmentation", "A2 R2L", "A2 token noise", "forward (A2 gauss hook)", "A1 cache read, KL, write")


def active(args) -> list[str]:
    """The attack flags that are on, in ORDER's naming."""
    on = []
    if getattr(args, "noise_r2l", 0.0) > 0 or getattr(args, "noise_mode", "none") != "none":
        on.append(f"A2:{getattr(args, 'noise_mode', 'none')}"
                  + (f"+r2l{args.noise_r2l:g}" if getattr(args, "noise_r2l", 0.0) > 0 else ""))
    if getattr(args, "sd_mode", "none") != "none":
        on.append(f"A1:{args.sd_mode}")
    if getattr(args, "bpe_dropout", None):
        on.append(f"A3:{args.bpe_dropout}:{getattr(args, 'bpe_dropout_scope', 'all')}")
    if getattr(args, "sources", None) or getattr(args, "init_from", None):
        on.append("A4:cycle" + ("+" + ",".join(args.sources) if getattr(args, "sources", None) else ""))
    return on


def describe(args) -> dict:
    return {"attacks": active(args), "order": list(ORDER),
            "sd_cache_on_noised_forward": getattr(args, "sd_mode", "none") == "temporal"
            and (getattr(args, "noise_mode", "none") != "none" or getattr(args, "noise_r2l", 0.0) > 0),
            "sd_skips_r2l_rows": getattr(args, "sd_mode", "none") == "temporal" and getattr(args, "noise_r2l", 0.0) > 0,
            "sd_keys_canonical_tokens_only": getattr(args, "sd_mode", "none") == "temporal"
            and bool(getattr(args, "bpe_dropout", None)),
            "sd_keys_real_training_split_only": getattr(args, "sd_mode", "none") == "temporal"
            and bool(getattr(args, "sources", None))}


def sd_positions(pos: np.ndarray, noise) -> np.ndarray:
    """A1 cache keys of one micro-batch's causal rows after A2: reversed rows get -1 (no key)."""
    rows = None if noise is None else noise.last_r2l
    if rows is None or not bool(rows.any()):
        return pos
    pos = np.array(pos, copy=True)
    pos[rows.numpy() if isinstance(rows, torch.Tensor) else rows] = -1
    return pos


class CanonicalKeys:
    """A1 cache keys for A3 epoch streams: a token of a re-segmented stream keeps the canonical position of the
    canonical token with the same byte span (same start and end in the shared text, hence the same id); a piece of
    a re-segmented word has no canonical twin and gets -1. Every stream decodes to the canonical stream's text
    (bpe_dropout asserts it), so the byte offsets of the two streams index the same text. Keeps the last two
    streams' maps (a batch straddles at most two epochs)."""

    def __init__(self, canon_tokens: np.ndarray, tok_bytes: np.ndarray):
        self.canon = canon_tokens
        self.tb = np.asarray(tok_bytes, dtype=np.int64)
        if (self.tb[np.unique(np.asarray(canon_tokens))] <= 0).any():
            raise ValueError("canonical keys need every token to span at least one byte")
        self.c_start, self.c_end = self._spans(canon_tokens)
        self._maps: dict[int, tuple[object, np.ndarray]] = {}

    def _spans(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        end = np.cumsum(self.tb[np.asarray(tokens, dtype=np.int64)])
        return end - self.tb[np.asarray(tokens, dtype=np.int64)], end

    def keys(self, tokens: np.ndarray) -> np.ndarray:
        """int64 [len(tokens)]: canonical position of each token, -1 for re-segmented pieces."""
        if tokens is self.canon:
            return np.arange(len(tokens), dtype=np.int64)
        hit = self._maps.get(id(tokens))
        if hit is not None and hit[0] is tokens:
            return hit[1]
        start, end = self._spans(tokens)
        i = np.searchsorted(self.c_start, start).clip(max=len(self.c_start) - 1)
        ok = (self.c_start[i] == start) & (self.c_end[i] == end)
        if int(self.c_end[-1]) != int(end[-1]):
            raise ValueError("stream and canonical text differ in length")
        k = np.where(ok, i, -1).astype(np.int64)
        if len(self._maps) >= 2:
            self._maps.pop(next(iter(self._maps)))
        self._maps[id(tokens)] = (tokens, k)
        return k

    def positions(self, sampler, T: int) -> np.ndarray:
        """[B, T] keys of the targets of the rows babylm.harness.bpe_dropout.MultiStreamSampler just returned."""
        out = []
        for e, starts in sampler.last_rows:
            k = self.keys(sampler.stream(e)[0])
            out.append(k[np.asarray(starts, dtype=np.int64)[:, None] + 1 + np.arange(T, dtype=np.int64)])
        return np.concatenate(out) if len(out) > 1 else out[0]


def cycle_positions(cyc, train, y: torch.Tensor) -> np.ndarray:
    """A1 cache keys of the batch babylm.harness.cycle.CycleSampler just returned (A4): rows from a real source that
    IS the training split (same sha256) keep their canonical target positions; synthetic rows and other real splits
    get -1 (no canonical position)."""
    n = cyc.batch_seqs
    i = cyc.ptr - 1
    T = y.shape[1]
    pos = np.full(tuple(y.shape), -1, dtype=np.int64)
    sha = train.meta.get("sha256")
    for r, w in enumerate(range(i * n, (i + 1) * n)):
        so = cyc.sources[int(cyc.w_src[w])]
        if so.kind != "real" or so.data.meta.get("sha256") != sha:
            continue
        s, ln = int(cyc.w_start[w]), int(cyc.w_len[w])
        if cyc.w_bos[w]:
            pos[r, :ln] = s + np.arange(ln)
        else:
            pos[r] = s + 1 + np.arange(T)
    if not np.array_equal(pos >= 0, (y.numpy() >= 0) & (pos >= 0)):
        raise RuntimeError("cycle keys do not match the targets")
    return pos


def exposure_r0(words_seen: int, n_words_split: int, epochs_cap: float = 10.0, synthetic_words: int = 0,
                generator_words: int = 0, views_per_window: int = 1, cap_words: int | None = None) -> dict:
    """R0 exposure ledger of one run: every whitespace word fed as training input, over all passes, objectives,
    synthetic text and the training of any generator (R0.2/R0.3). A1 and A2 add nothing (one view, same forward)."""
    total = int(words_seen) * int(views_per_window) + int(synthetic_words) + int(generator_words)
    cap = int(math.floor(epochs_cap * n_words_split)) if cap_words is None else int(cap_words)
    return {"input_words": int(words_seen), "views_per_window": int(views_per_window),
            "synthetic_words": int(synthetic_words), "generator_words": int(generator_words), "total_words": total,
            "cap_words": cap, "within_cap": total <= cap}
