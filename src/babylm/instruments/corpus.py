"""Corpus helpers shared by the BabyLM bottleneck instruments.

Words are whitespace-separated tokens (the critic's unit). A word's
"normal form" lowercases it and strips surrounding punctuation, so
"Boy?" and "boy" count as one type for frequency and n-gram lookups.
"""
from __future__ import annotations

import hashlib
import os
import string
from pathlib import Path

import numpy as np

SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
PUNCT = string.punctuation + "“”‘’«»…—–"
HASH_P = np.uint64(1_000_003)


def norm(word: str) -> str:
    w = word.lower()
    s = w.strip(PUNCT)
    return s if s else w


# Which word stream the probes read: "v1" = the frozen dev and train samples (prepare.py); "harness" = the experimenter's
# DEV-sub and train-sub, rebuilt by harness.py so the harness's per-word dumps can be sliced without rescoring.
SAMPLE = os.environ.get("BABYLM_SAMPLE", "v1")


def sname(split: str) -> str:
    """File name of a sample split ("dev" or "train") under the active BABYLM_SAMPLE."""
    return {"dev": "devsub", "train": "trainsub"}.get(split, split) if SAMPLE == "harness" else split


def split_files(data_root: Path, split: str) -> list[Path]:
    """Official file layout after unzipping: <root>/<split>/<split>/<source>.<ext>."""
    ext = "dev" if split in ("dev", "test") else "train"
    return [data_root / split / split / f"{s}.{ext}" for s in SOURCES]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_header(line: str) -> bool:
    """CHILDES transcript headers look like '= = = childes/.../x.cha = = ='."""
    return line.startswith("= = =") and line.rstrip().endswith("= = =")


class Vocab:
    """Normal-form word -> integer id, grown on demand."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def encode_words(self, words: list[str]) -> list[int]:
        ids = self.ids
        out = []
        for w in words:
            n = norm(w)
            i = ids.get(n)
            if i is None:
                i = len(ids)
                ids[n] = i
            out.append(i)
        return out

    def __len__(self) -> int:
        return len(self.ids)


def encode_split(vocab: Vocab, files: list[Path]) -> np.ndarray:
    """Whole split as one stream of normal-form ids (headers dropped)."""
    chunks = []
    for path in files:
        buf: list[int] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if is_header(line):
                    continue
                buf.extend(vocab.encode_words(line.split()))
        chunks.append(np.asarray(buf, dtype=np.int64))
    return np.concatenate(chunks)


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Rolling hash of every n-gram in a stream; element j covers ids[j:j+n]."""
    if len(ids) < n:
        return np.zeros(0, dtype=np.uint64)
    u = ids.astype(np.uint64) + np.uint64(1)
    m = len(ids) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * HASH_P + u[k : k + m]
    return h


def ngram_set(ids: np.ndarray, n: int) -> np.ndarray:
    return np.unique(ngram_hashes(ids, n))


def in_sorted(sorted_arr: np.ndarray, q: np.ndarray) -> np.ndarray:
    if len(sorted_arr) == 0 or len(q) == 0:
        return np.zeros(len(q), dtype=bool)
    pos = np.searchsorted(sorted_arr, q)
    pos[pos == len(sorted_arr)] = 0
    return sorted_arr[pos] == q
