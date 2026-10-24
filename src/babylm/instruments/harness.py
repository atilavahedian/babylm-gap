"""Run the instruments on the harness's per-word dumps: the experimenter's DEV-sub and train-sub, about 1M words each.

The experimenter's harness writes, per run, pw/pw_{dev,train}_<label>.npy: float16 nats per whitespace word of DEV-sub and
train-sub in stream order (babylm.analysis.perword on the experimenter's branch), at every saved epoch. This module
  features  rebuilds both subsets as rows, one per document (samples/harness_subsets/*.json, copied unchanged from
            the experimenter's babylm/data/subsets at ref-0396), and their word features (prepare.word_features, with the
            same vocabulary and train_10M / train_100M counts as the v1 samples);
  import    copies a run's dumps to data/babylm_scores/<run>/chck_<label>/{devsub,trainsub}.npz (word_nll only;
            meta.json records the source under splits.devsub / splits.trainsub).
Every probe then reads them with BABYLM_SAMPLE=harness (dev -> devsub, train -> trainsub); give it its own --tag.
Differences from the v1 sample: whole documents (1,199 dev documents, 9 of them Gutenberg books), the harness's
text format, and the BOS before each document is a scored target counted in the document's first word.
Token-level probes (copy cache within a chunk, token mixtures, span recall) are not available on dumps.

Usage:
  python -m babylm.instruments.harness features --data-root ~/babylm_data
  python -m babylm.instruments.harness import --pack shared/babylm/p2/results_pack/port_S_10M_e10_s0
  BABYLM_SAMPLE=harness python -m babylm.instruments.stall --tag harness_v1 --run port_S_10M_e10_s0 \\
      --ref grid_S_100M_e10_s0/chck_1000M
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from babylm.instruments import corpus as C
from babylm.instruments import prepare as P

SUBSETS = Path(__file__).resolve().parent / "samples" / "harness_subsets"
OUT = {"dev_sub": "devsub", "train_sub": "trainsub"}
LABEL = re.compile(r"^pw_(dev|train)_(.+)\.npy$")


def subset_rows(data_root: Path, subset: str) -> list[dict]:
    """One row per document, in the harness's stream order (source order, then document index)."""
    spec = json.loads((SUBSETS / f"{subset}.json").read_text())
    split = spec["split"]
    rows = []
    for s, path in zip(C.SOURCES, C.split_files(data_root, split)):
        lines = path.read_text(encoding="utf-8").split("\n")
        for k, (a, b) in sorted(zip(spec["docs"].get(s, []), spec["line_ranges"].get(s, []))):
            text = "\n".join(lines[a:b])
            rows.append({"id": f"{subset}-{s}-{k}", "split": split, "source": s, "file": f"{split}/{path.name}",
                         "line_start": a, "line_end": b, "truncated": False, "text": text,
                         "n_words": len(text.split()), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    n = sum(r["n_words"] for r in rows)
    if n != spec["words"]:
        raise SystemExit(f"{subset}: {n} words rebuilt but {spec['words']} in its manifest")
    return rows


def features(data_root: Path) -> None:
    t0 = time.time()
    files = {s: C.split_files(data_root, s) for s in ("train_10M", "train_100M")}
    vocab = C.Vocab()
    ids100 = C.encode_split(vocab, files["train_100M"])
    ids10 = C.encode_split(vocab, files["train_10M"])
    rows = {sub: subset_rows(data_root, sub) for sub in OUT}
    for rs in rows.values():
        for r in rs:
            P.chunk_ids(vocab, r["text"])
    counts10, counts100 = np.bincount(ids10, minlength=len(vocab)), np.bincount(ids100, minlength=len(vocab))
    print(f"encoded ({time.time() - t0:.0f}s)", flush=True)
    # one pass for both subsets: the n-gram tables of train_10M / train_100M are the slow, memory-heavy part
    feats = P.word_features(rows["dev_sub"] + rows["train_sub"], vocab, counts10, counts100, ids10, ids100)
    del ids10, ids100
    nd_rows, nd_words = len(rows["dev_sub"]), sum(r["n_words"] for r in rows["dev_sub"])
    for sub, name in OUT.items():
        with open(P.REPO / "data" / "babylm_instruments" / f"{name}_sample_{P.VERSION}.jsonl", "w", encoding="utf-8") as f:
            for r in rows[sub]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        part = {k: (v[:nd_words] if sub == "dev_sub" else v[nd_words:]) for k, v in feats.items()}
        if sub == "train_sub":
            part["chunk"] = part["chunk"] - nd_rows
        np.savez_compressed(P.REPO / "data" / "babylm_instruments" / f"features_{name}_{P.VERSION}.npz", **part)
        print(f"{name}: {len(rows[sub])} documents, {len(part['chunk'])} words ({time.time() - t0:.0f}s)", flush=True)


def import_pack(pack: Path, run: str | None) -> None:
    """pack = a run directory of the experimenter's results pack (pw/ inside)."""
    run = run or pack.name
    n_words = {"dev": json.loads((SUBSETS / "dev_sub.json").read_text())["words"],
               "train": json.loads((SUBSETS / "train_sub.json").read_text())["words"]}
    for p in sorted((pack / "pw").glob("pw_*.npy")):
        m = LABEL.match(p.name)
        if not m:
            continue
        split, label = m.groups()
        a = np.load(p).astype(np.float32)
        if len(a) != n_words[split]:
            raise SystemExit(f"{p}: {len(a)} words, subset has {n_words[split]}")
        rev = label if label.startswith("chck_") else f"chck_{label}"
        d = P.REPO / "data" / "babylm_scores" / run / rev
        d.mkdir(parents=True, exist_ok=True)
        nan = np.isnan(a)
        np.savez(d / f"{'devsub' if split == 'dev' else 'trainsub'}.npz", word_nll=np.where(nan, 0.0, a).astype(np.float32))
        meta_p = d / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        meta.setdefault("run", run)  # the directory may also hold this checkpoint's own v1 scores (score.py)
        meta.setdefault("splits", {})[split + "sub"] = {"imported_from": str(pack), "file": p.name, "words": int(len(a)),
                                                         "nan_words_set_to_0": int(nan.sum()), "sha256": C.sha256_file(p),
                                                         "nats_per_word": float(np.nanmean(a.astype(np.float64)))}
        meta_p.write_text(json.dumps(meta, indent=1))
        print(f"{run}/{rev} {split}sub: {np.nanmean(a.astype(np.float64)):.4f} nats/word, {int(nan.sum())} NaN", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("features")
    f.add_argument("--data-root", type=Path, required=True)
    i = sp.add_parser("import")
    i.add_argument("--pack", type=Path, required=True, action="append")
    i.add_argument("--run", help="run name (default: the pack directory name; only with one --pack)")
    args = ap.parse_args()
    if args.cmd == "features":
        features(args.data_root)
    else:
        for pack in args.pack:
            import_pack(pack, args.run if len(args.pack) == 1 else None)


if __name__ == "__main__":
    main()
