"""Phase 2 slices on DEV-sub (default) or the full DEV split from per-word dumps (babylm.analysis.perword output).

    python -m babylm.analysis.p2_slices --runs NAME=RUN_DIR[:LABEL] ... [--pairs A-B ...] [--out DIR] [--subset dev]

Each --runs item names a run's per-word file: RUN_DIR/pw/pw_dev_<LABEL>.npy for DEV-sub (LABEL defaults to the run's
last label), RUN_DIR/pw/pw_devfull_final.npy for --subset dev. Words are whitespace words in the harness's stream
order (sources in corpus.SOURCES order, documents in index order; DEV-sub = the documents listed in
babylm/data/subsets/dev_sub.json, full DEV = every document). Every word gets: source, document, position in its document,
first occurrence in its document or a repeat, its whitespace-word counts in train_10M and train_100M, and whether
its line (>= 8 words, whitespace-normalised) appears verbatim as a line of train_100M / train_10M (overlap buckets).
Writes slices.csv (mean nats/word, word share and share of total nats per bucket and run) and, for every
--pairs A-B, delta.csv (A minus B per bucket, and each bucket's share of the total difference).
--drop_overlap removes every word of a line found verbatim in train_100M from all buckets first (critic ref-0260:
the prize that sizes and ranks bottlenecks is the overlap-removed one).
Word counts are cached in <data_dir>/analysis/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from babylm.data.corpus import DEFAULT_DATA_DIR, SOURCES, SUBSET_DIR, doc_ranges, raw_path

OVERLAP_MIN_WORDS = 8
FREQ_EDGES = [(0, 0), (1, 1), (2, 10), (11, 100), (101, 1000), (1001, 10000), (10001, 10**12)]
POS_EDGES = [(0, 9), (10, 49), (50, 199), (200, 999), (1000, 10**12)]


def word_counts(split: str, data_dir: Path) -> Counter:
    cache = data_dir / "analysis" / f"wordcounts_{split}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    c: Counter = Counter()
    for s in SOURCES:
        c.update(raw_path(data_dir, split, s).read_text(encoding="utf-8").split())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(c))
    return c


def line_key(words: list[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest(), "little")


def line_hashes(split: str, data_dir: Path) -> np.ndarray:
    """Sorted unique 64-bit hashes of the whitespace-normalised lines of a training split with >= 8 words (cached)."""
    cache = data_dir / "analysis" / f"linehash_{split}.npy"
    if cache.exists():
        return np.load(cache)
    keys = []
    for s in SOURCES:
        for line in raw_path(data_dir, split, s).read_text(encoding="utf-8").split("\n"):
            ws = line.split()
            if len(ws) >= OVERLAP_MIN_WORDS:
                keys.append(line_key(ws))
    h = np.unique(np.array(keys, dtype=np.uint64))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, h)
    return h


def devsub_words(data_dir: Path, subset: str = "dev_sub") -> dict[str, np.ndarray]:
    """Word strings and per-word features of DEV-sub (or the full DEV split), in the harness's stream order."""
    if subset == "dev_sub":
        sub = json.loads((SUBSET_DIR / "dev_sub.json").read_text())
    else:
        sub = {"docs": {}, "words": json.loads((data_dir / "bin" / "dev.meta.json").read_text())["words"]}
    words, src, doc, pos, first, lkey = [], [], [], [], [], []
    d = 0
    for sid, s in enumerate(SOURCES):
        lines = raw_path(data_dir, "dev", s).read_text(encoding="utf-8").split("\n")
        ranges = doc_ranges(lines, s)
        for k in (sub["docs"][s] if subset == "dev_sub" else range(len(ranges))):
            a, b = ranges[k]
            seen: set[str] = set()
            i = 0
            for line in lines[a:b]:
                lw = line.split()
                key = line_key(lw) if len(lw) >= OVERLAP_MIN_WORDS else 0
                for w in lw:
                    words.append(w)
                    src.append(sid)
                    doc.append(d)
                    pos.append(i)
                    first.append(w not in seen)
                    lkey.append(key)
                    seen.add(w)
                    i += 1
            d += 1
    if len(words) != sub["words"]:
        raise SystemExit(f"{subset} has {len(words)} words here but {sub['words']} in its manifest")
    return {"word": np.array(words, dtype=object), "src": np.array(src), "doc": np.array(doc),
            "pos": np.array(pos), "first": np.array(first), "lkey": np.array(lkey, dtype=np.uint64)}


def bucket_masks(f: dict[str, np.ndarray], c10: np.ndarray, c100: np.ndarray,
                 ovl: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    m: dict[str, np.ndarray] = {"all": np.ones(len(c10), dtype=bool)}
    for lo, hi in FREQ_EDGES:
        key = f"c10={lo}" if lo == hi else f"c10={lo}-{hi if hi < 10**12 else 'inf'}"
        m[key] = (c10 >= lo) & (c10 <= hi)
    m["absent10_in100"] = (c10 == 0) & (c100 > 0)
    m["absent10_absent100"] = (c10 == 0) & (c100 == 0)
    m["c10<=10"] = c10 <= 10
    m["c10<=63"] = c10 <= 63  # the instruments' rare-word band (seen under 64 times)
    for lo, hi in POS_EDGES:
        m[f"pos={lo}-{hi if hi < 10**12 else 'inf'}"] = (f["pos"] >= lo) & (f["pos"] <= hi)
    m["first_in_doc"] = f["first"]
    m["repeat_in_doc"] = ~f["first"]
    m["rare_repeat"] = (c10 <= 10) & ~f["first"]
    for sid, s in enumerate(SOURCES):
        m[f"src={s}"] = f["src"] == sid
    for name, o in (ovl or {}).items():
        m[f"line_in_{name}"] = o
        m[f"line_not_in_{name}"] = ~o
        for sid, s in enumerate(SOURCES):
            m[f"src={s}&line_not_in_{name}"] = (f["src"] == sid) & ~o
    return m


def load_pw(spec: str, subset: str = "dev_sub") -> tuple[str, np.ndarray]:
    name, rest = spec.split("=", 1)
    run, _, label = rest.partition(":")
    pw_dir = Path(run) / "pw"
    if subset == "dev":
        return name, np.load(pw_dir / f"pw_devfull_{label or 'final'}.npy").astype(np.float64)
    if not label:
        idx = json.loads((pw_dir / "index.json").read_text())
        devs = [x[len("pw_dev_"):-4] for x in idx["files"] if x.startswith("pw_dev_")]
        label = max(devs, key=lambda x: float(x[:-1]) if x[:-1].replace(".", "").isdigit() else -1)
    return name, np.load(pw_dir / f"pw_dev_{label}.npy").astype(np.float64)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out", default="p2_slices")
    ap.add_argument("--subset", default="dev_sub", choices=["dev_sub", "dev"])
    ap.add_argument("--drop_overlap", action="store_true", help="drop words of lines found verbatim in train_100M")
    a = ap.parse_args(argv)
    data_dir, out = Path(a.data_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = devsub_words(data_dir, a.subset)
    c10m, c100m = word_counts("train_10M", data_dir), word_counts("train_100M", data_dir)
    c10 = np.array([c10m.get(w, 0) for w in f["word"]])
    c100 = np.array([c100m.get(w, 0) for w in f["word"]])
    has_key = f["lkey"] != 0
    ovl = {sp.replace("train_", ""): has_key & np.isin(f["lkey"], line_hashes(sp, data_dir))
           for sp in ("train_100M", "train_10M")}
    masks = bucket_masks(f, c10, c100, ovl)
    runs = dict(load_pw(s, a.subset) for s in a.runs)
    n = len(c10)
    for name, v in runs.items():
        if len(v) != n:
            raise SystemExit(f"{name}: {len(v)} words, {a.subset} has {n}")
    ok = np.all([~np.isnan(v) for v in runs.values()], axis=0)
    if a.drop_overlap:
        ok &= ~ovl["100M"]
    with open(out / "slices.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "words", "word_share", "run", "nats_per_word", "share_of_total_nats"])
        for b, m in masks.items():
            mm = m & ok
            for name, v in runs.items():
                w.writerow([b, int(mm.sum()), round(mm.sum() / ok.sum(), 6), name,
                            round(float(v[mm].mean()), 5) if mm.any() else "",
                            round(float(v[mm].sum() / v[ok].sum()), 6) if mm.any() else ""])
    if a.pairs:
        with open(out / "delta.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "bucket", "words", "word_share", "delta_nats_per_word_in_bucket",
                        "delta_contribution_per_dev_word", "share_of_total_delta"])
            for p in a.pairs:
                x, y = p.split("-", 1)
                d = runs[x] - runs[y]
                tot = d[ok].sum()
                for b, m in masks.items():
                    mm = m & ok
                    if not mm.any():
                        continue
                    w.writerow([p, b, int(mm.sum()), round(mm.sum() / ok.sum(), 6), round(float(d[mm].mean()), 5),
                                round(float(d[mm].sum() / ok.sum()), 5), round(float(d[mm].sum() / tot), 4)])
    print(f"wrote {out}/slices.csv" + (f" and {out}/delta.csv" if a.pairs else ""))


if __name__ == "__main__":
    main()
