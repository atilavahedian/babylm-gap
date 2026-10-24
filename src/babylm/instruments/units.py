"""Document ids for the frozen samples: the unit for bootstrap CIs and for cross-fitted weights.

Chunks of one long document (a Gutenberg book) are correlated, so CIs resample documents and
weights are fitted on even documents and scored on odd ones, and vice versa. Documents follow the
experimenter's babylm/data/corpus.py: blank-line-separated blocks of a source file; Switchboard has no blank
lines, so its documents are pieces of 2000 lines cut at multiples of 2000 within the file.
Writes data/babylm_instruments/units_<split>_v1.npy (per word, ids 0..n-1 in sample order);
slices.load_feats exposes it as "unit" (the chunk id when the file is missing).

Usage:
  python -m babylm.instruments.units --data-root ~/babylm_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from babylm.instruments import slices as S
from babylm.instruments.score import load_rows

SWB_PIECE_LINES = 2000


def doc_starts(lines: list[str], source: str) -> np.ndarray:
    """First line index of every document of one file."""
    if source == "switchboard":
        return np.arange(0, len(lines), SWB_PIECE_LINES)
    starts, prev_blank = [], True
    for i, line in enumerate(lines):
        blank = line.strip() == ""
        if prev_blank and not blank:
            starts.append(i)
        prev_blank = blank
    return np.asarray(starts)


def word_units(data_root: Path, split: str) -> np.ndarray:
    sub = "dev" if split == "dev" else "train_10M"
    starts: dict[str, np.ndarray] = {}
    ids: dict[tuple[str, int], int] = {}
    out: list[int] = []
    for r in load_rows(split):
        if r["file"] not in starts:
            starts[r["file"]] = doc_starts((data_root / sub / r["file"]).read_text(encoding="utf-8").split("\n"), r["source"])
        for i, line in enumerate(r["text"].split("\n")):
            doc = int(np.searchsorted(starts[r["file"]], r["line_start"] + i, side="right") - 1)
            u = ids.setdefault((r["file"], doc), len(ids))
            out.extend([u] * len(line.split()))
    return np.asarray(out, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    args = ap.parse_args()
    from babylm.instruments.corpus import SAMPLE

    if SAMPLE != "v1":
        raise SystemExit("units.py is for the v1 samples; harness rows are whole documents, so their unit is the row")
    for split in ("dev", "train"):
        u = word_units(args.data_root, split)
        n = len(S.load_feats(split)["chunk"])
        if len(u) != n:
            raise SystemExit(f"{split}: {len(u)} words by line but {n} in the features")
        np.save(S.FEATS / f"units_{split}_{S.VERSION}.npy", u)  # v1 samples only; harness rows are whole documents
        print(f"{split}: {u.max() + 1} documents, {len(np.unique(S.load_feats(split)['chunk']))} chunks, {n} words")


if __name__ == "__main__":
    main()
