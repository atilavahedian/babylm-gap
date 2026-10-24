"""Per-word nats from the harness's per-token dumps: small files for the results branch and for slicing.

    python -m babylm.analysis.perword RUN_DIR [RUN_DIR ...] [--data_dir data/babylm] [--epochs_only]

For each RUN_DIR/pt/pt_{dev,train}_<label>.npy (float16 per-token nats in the subset's stream order, NaN at the
unscored first token) this writes RUN_DIR/pw/pw_{dev,train}_<label>.npy: float16 nats per whitespace word of the
subset, i.e. the sum over the word's tokens (the BOS and pure-whitespace tokens that babylm.data.corpus attaches
to a word count toward it). A word with an unscored token is NaN. pt_devfull_final.npy -> pw_devfull_final.npy.
--epochs_only keeps only labels that sit on whole-epoch boundaries of the run's split (1, 2, 3.333, 4, 5, 10 or
20 epochs, within 2%) plus the last label, so a results push stays a few MB per run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from babylm.data.corpus import load_split

SUBSET_OF_TAG = {"dev": "dev_sub", "train": "train_sub", "devfull": "dev"}
EPOCH_POINTS = (1.0, 2.0, 10 / 3, 4.0, 5.0, 10.0, 20.0)
PT_RE = re.compile(r"^pt_(devfull|dev|train)_(.+)\.npy$")


def per_word(pt: np.ndarray, word_ids: np.ndarray, n_words: int) -> np.ndarray:
    """Sum per-token nats into words; NaN for any word holding an unscored (NaN) token."""
    pt = np.asarray(pt, dtype=np.float64)
    w = np.asarray(word_ids, dtype=np.int64)
    if len(pt) != len(w):
        raise ValueError(f"dump has {len(pt)} tokens but the subset has {len(w)}")
    bad = np.isnan(pt)
    sums = np.bincount(w, weights=np.where(bad, 0.0, pt), minlength=n_words + 1)[:n_words]
    nbad = np.bincount(w, weights=bad.astype(np.float64), minlength=n_words + 1)[:n_words]
    sums[nbad > 0] = np.nan
    return sums


def label_words(label: str) -> float | None:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)M", label)
    return float(m.group(1)) * 1e6 if m else None


def keep_label(label: str, split_words: int, last: str) -> bool:
    if label == last or label == "final":
        return True
    w = label_words(label)
    if w is None:
        return False
    e = w / split_words
    return any(abs(e / p - 1.0) <= 0.02 for p in EPOCH_POINTS)


def convert(run_dir: Path, data_dir: str | None, epochs_only: bool) -> list[str]:
    cfg = json.loads((run_dir / "config.json").read_text())
    split = cfg.get("split") or cfg.get("args", {}).get("split")
    split_words = int(load_split(split, data_dir).n_words) if split else 0
    dumps = sorted(p for p in (run_dir / "pt").glob("pt_*.npy") if PT_RE.match(p.name))
    labels = [PT_RE.match(p.name).group(2) for p in dumps if PT_RE.match(p.name).group(1) == "dev"]
    numeric = sorted((label_words(x), x) for x in labels if label_words(x) is not None)
    last = numeric[-1][1] if numeric else ""
    out_dir = run_dir / "pw"
    out_dir.mkdir(exist_ok=True)
    cache: dict[str, tuple[np.ndarray, int]] = {}
    written = []
    for p in dumps:
        tag, label = PT_RE.match(p.name).groups()
        if epochs_only and tag != "devfull" and split_words and not keep_label(label, split_words, last):
            continue
        sub = SUBSET_OF_TAG[tag]
        if sub not in cache:
            d = load_split(sub, data_dir)
            cache[sub] = (np.asarray(d.words), d.n_words)
        w, n = cache[sub]
        pw = per_word(np.load(p), w, n)
        q = out_dir / f"pw_{tag}_{label}.npy"
        np.save(q, pw.astype(np.float16))
        written.append(str(q))
    (out_dir / "index.json").write_text(json.dumps({"split": split, "split_words": split_words,
                                                     "files": [Path(x).name for x in written]}, indent=1))
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--epochs_only", action="store_true")
    a = ap.parse_args(argv)
    for r in a.run_dirs:
        files = convert(Path(r), a.data_dir, a.epochs_only)
        print(f"{r}: {len(files)} per-word files")


if __name__ == "__main__":
    main()
