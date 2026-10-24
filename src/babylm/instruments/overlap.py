"""Verbatim line overlap: which dev-sample words sit in a line found word for word in a training split.

A line counts when it has at least 8 whitespace words and its whitespace-normalised form is a
line of the training split, the rule in the experimenter's babylm/analysis/p2_slices.py (critic ref-0260:
the prize that sizes and ranks bottlenecks is the one with train_100M overlap removed). The
100M model saw those lines and the 10M model may not have, so they flatter the prize.
Writes data/babylm_instruments/overlap_dev_v1.npz (per-word booleans in_train_10M, in_train_100M);
slices.load_feats picks it up.

Usage:
  python -m babylm.instruments.overlap --data-root ~/babylm_data
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from babylm.instruments import corpus as C
from babylm.instruments import slices as S
from babylm.instruments.score import load_rows

MIN_WORDS = 8
SPLITS = ("train_10M", "train_100M")


def line_key(words: list[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest(), "little")


def line_hashes(data_root: Path, split: str) -> np.ndarray:
    """Sorted unique 64-bit hashes of the split's lines with at least 8 words (cached)."""
    cache = S.FEATS / f"linehash_{split}.npy"
    if cache.exists():
        return np.load(cache)
    keys = []
    for path in C.split_files(data_root, split):
        for line in path.read_text(encoding="utf-8").split("\n"):
            ws = line.split()
            if len(ws) >= MIN_WORDS:
                keys.append(line_key(ws))
    h = np.unique(np.array(keys, dtype=np.uint64))
    np.save(cache, h)
    return h


def word_keys(split: str) -> np.ndarray:
    """Per sample word: the hash of its line, 0 for lines under 8 words."""
    keys: list[int] = []
    for r in load_rows(split):
        for line in r["text"].split("\n"):
            ws = line.split()
            keys.extend([line_key(ws) if len(ws) >= MIN_WORDS else 0] * len(ws))
    return np.array(keys, dtype=np.uint64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    args = ap.parse_args()
    keys = word_keys("dev")
    n = len(S.load_feats("dev")["chunk"])
    if len(keys) != n:
        raise SystemExit(f"dev sample has {len(keys)} words by line but {n} in the features")
    out = {f"in_{sp}": (keys != 0) & np.isin(keys, line_hashes(args.data_root, sp)) for sp in SPLITS}
    np.savez(S.FEATS / f"overlap_{C.sname('dev')}_{S.VERSION}.npz", **out)
    for k, v in out.items():
        print(f"{k}: {int(v.sum())} of {n} dev-sample words ({v.mean():.2%})")


if __name__ == "__main__":
    main()
