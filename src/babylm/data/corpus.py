"""BabyLM corpus layout shared by prepare.py, the harness, the tests and analysis code.

Documents: blank-line-separated blocks of a source file (a line is blank if line.strip() == ""). Exception:
switchboard has no blank line in any split (each file is one block), so its documents are consecutive pieces of
SWB_PIECE_LINES lines, cut at line indices that are multiples of SWB_PIECE_LINES within the file (train_10M was cut
from train_100M in 2000-line chunks aligned the same way, so the pieces of both files coincide).
A document's text is "\\n".join(normalize_line(l) for l in its lines): whitespace inside every line is collapsed
to single spaces and stripped (the GPT-BERT 2025 reference's normalisation; newlines and document boundaries are
kept). Words are the items of str.split(), which this normalisation leaves unchanged.

Binary files in <data_dir>/bin (all little-endian, no header):
  {split}.bin        uint16 token ids; every document is preceded by <|endoftext|> (id 0, BOS / separator)
  {split}.words.bin  uint32 index of the whitespace word each token belongs to (pure-whitespace tokens and BOS
                     attach to the following word; whitespace after the split's last word gets id n_words)
  {split}.src.bin    uint8 source id (index into SOURCES)
  {split}.meta.json  counts and sha256s
Splits: train_10M, train_100M, dev, test (FINAL, protected), train_20M/30M/50M (nested data-ruler splits),
dev_sub / train_sub (the fixed ~1M-word evaluation subsets).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(os.environ.get("HINTON_BABYLM_DATA", REPO_ROOT / "data" / "babylm"))
TOKENIZER_PATH = PKG_DIR / "tok16k_10m" / "tokenizer.json"
MANIFEST_PATH = PKG_DIR / "manifest.json"
SUBSET_DIR = PKG_DIR / "subsets"

SOURCES = ("bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard")
OFFICIAL_SPLITS = {"train_10M": "train", "train_100M": "train", "dev": "dev", "test": "test"}
RULER_SPLITS = {"train_20M": 20_000_000, "train_30M": 30_000_000, "train_50M": 50_000_000}
SUBSETS = {"dev_sub": "dev", "train_sub": "train_10M"}
PROTECTED_SPLITS = frozenset({"test"})
BOS_TOKEN = "<|endoftext|>"
BOS_ID = 0
VOCAB_SIZE = 16384
SWB_PIECE_LINES = 2000


class FinalSplitError(PermissionError):
    """Raised when the protected FINAL split (test) is opened without final=True."""


def raw_path(data_dir: Path, split: str, source: str) -> Path:
    return Path(data_dir) / "raw" / split / f"{source}.{OFFICIAL_SPLITS[split]}"


def bin_dir(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir or DEFAULT_DATA_DIR) / "bin"


def normalize_line(line: str) -> str:
    """Collapse every run of whitespace inside a line to one space and strip the ends: " ".join(line.split())."""
    return " ".join(line.split())


def doc_text(lines: list[str], start: int, end: int) -> str:
    """Normalised text of the document lines[start:end]."""
    return "\n".join(normalize_line(l) for l in lines[start:end])


def doc_ranges(lines: list[str], source: str) -> list[tuple[int, int]]:
    """[start, end) line ranges of the documents of one file (see the module docstring)."""
    out: list[tuple[int, int]] = []
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "":
            if start is not None:
                out.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        out.append((start, len(lines)))
    if source == "switchboard":
        P = SWB_PIECE_LINES
        cut: list[tuple[int, int]] = []
        for s, e in out:
            b = s
            while b < e:
                nxt = min(e, (b // P + 1) * P)
                cut.append((b, nxt))
                b = nxt
        out = cut
    return out


def read_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").split("\n")


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@lru_cache(maxsize=1)
def whitespace_codepoints() -> np.ndarray:
    """Every code point c with chr(c).isspace() (the separators of str.split())."""
    return np.array([c for c in range(0x110000) if chr(c).isspace()], dtype=np.uint32)


def char_word_index(text: str) -> tuple[np.ndarray, np.ndarray, int]:
    """(positions of non-whitespace chars, word index of each of them, number of words) for str.split()."""
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    nonspace = ~np.isin(cp, whitespace_codepoints())
    pos = np.flatnonzero(nonspace)
    if len(pos) == 0:
        return pos, pos, 0
    starts = np.ones(len(pos), dtype=bool)
    starts[1:] = pos[1:] != pos[:-1] + 1
    wid = np.cumsum(starts) - 1
    return pos, wid, int(wid[-1]) + 1


def token_word_ids(text: str, token_starts: np.ndarray) -> tuple[np.ndarray, int]:
    """Word index (local to `text`) of each token: the word of the first non-whitespace char at or after the
    token's start offset (so a token holding word chars gets that word, and a pure-whitespace token gets the
    following word; whitespace after the last word gets n_words). Returns (ids int64, n_words)."""
    pos, wid, n = char_word_index(text)
    k = np.searchsorted(pos, np.asarray(token_starts, dtype=np.int64), side="left")
    out = np.full(len(k), n, dtype=np.int64)
    ok = k < len(pos)
    out[ok] = wid[k[ok]]
    return out, n


def token_byte_lengths(tokenizer_path: str | Path = TOKENIZER_PATH) -> np.ndarray:
    """UTF-8 byte length of each token id (int32 [vocab]). Byte-level BPE: one vocab char = one byte; the BOS
    separator counts as 1 byte (the sprint-1 convention for <|endoftext|>)."""
    tj = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    vocab: dict[str, int] = tj["model"]["vocab"]
    n = max(max(vocab.values()) + 1, VOCAB_SIZE)
    t = np.zeros(n, dtype=np.int32)
    for s, i in vocab.items():
        t[i] = len(s)
    for a in tj.get("added_tokens", []):
        t[a["id"]] = 1
    t[BOS_ID] = 1
    return t


@dataclass
class SplitData:
    name: str
    tokens: np.ndarray   # uint16 memmap
    words: np.ndarray    # uint32 memmap
    src: np.ndarray      # uint8 memmap
    meta: dict[str, Any]

    @property
    def n_tokens(self) -> int:
        return int(len(self.tokens))

    @property
    def n_words(self) -> int:
        return int(self.meta["words"])

    def doc_starts(self) -> np.ndarray:
        """Token index of every document's BOS."""
        return np.flatnonzero(np.asarray(self.tokens) == BOS_ID)


def load_split(name: str, data_dir: str | Path | None = None, final: bool = False) -> SplitData:
    """Memory-map one tokenized split. The FINAL split (test) refuses to load unless final=True."""
    if name in PROTECTED_SPLITS and not final:
        raise FinalSplitError(f"split {name!r} is the protected FINAL split: it may only be read with an explicit "
                              f"--final flag (train.py --final), which warns and writes a FINAL_SPLIT_USED marker")
    d = bin_dir(data_dir)
    meta_p = d / f"{name}.meta.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"{meta_p} missing: run python -m babylm.data.prepare")
    meta = json.loads(meta_p.read_text())
    tok = np.memmap(d / f"{name}.bin", dtype="<u2", mode="r")
    words = np.memmap(d / f"{name}.words.bin", dtype="<u4", mode="r")
    src = np.memmap(d / f"{name}.src.bin", dtype="u1", mode="r")
    if not (len(tok) == len(words) == len(src) == meta["tokens"]):
        raise ValueError(f"{name}: length mismatch tokens {len(tok)} words {len(words)} src {len(src)} "
                         f"meta {meta['tokens']}")
    return SplitData(name, tok, words, src, meta)


def word_initial(words: np.ndarray) -> np.ndarray:
    """uint8 flag per token: 1 if its word id differs from the previous token's (the first token of each word).
    Position 0 is flagged; the harness counts flags at target positions only."""
    w = np.asarray(words)
    out = np.ones(len(w), dtype=np.uint8)
    if len(w) > 1:
        out[1:] = (w[1:] != w[:-1]).astype(np.uint8)
    return out
