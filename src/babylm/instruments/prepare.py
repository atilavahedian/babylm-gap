"""Build the frozen slicing samples and their model-independent word features.

One run produces, under --out (default data/babylm_instruments):
  dev_sample_v1.jsonl, train_sample_v1.jsonl   chunk text (not committed)
  features_{dev,train}_v1.npz                 one row per sample word
and, under babylm/instruments/samples (committed):
  {dev,train}_sample_v1.index.tsv             file, line range, sha256 per chunk
  manifest_v1.json                            zip/file hashes, params, overlap rates

Chunks are runs of consecutive lines (at most --max-words words, at least
--min-words), never crossing a CHILDES transcript header. Sampling is
stratified by source in proportion to each source's word share. Dev chunks
whose 13-gram overlap with train_100M exceeds --max-overlap are dropped
(document-overlap rule from the critic, 2026-10-09T18:34Z).

Usage:
  python -m babylm.instruments.prepare --data-root /path/to/babylm_data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from babylm.instruments import corpus as C

REPO = Path(__file__).resolve().parents[2]
VERSION = "v1"
MAX_N = 8  # longest n-gram order for the suffix-match feature
PRONOUN_I = {"i", "i'm", "i'll", "i've", "i'd"}


def candidate_chunks(path: Path, max_words: int, min_words: int) -> list[dict]:
    out, cur, cur_n, start = [], [], 0, None
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    def flush(end_line: int) -> None:
        nonlocal cur, cur_n, start
        if cur and cur_n >= min_words:
            out.append({"line_start": start, "line_end": end_line, "lines": cur})
        cur, cur_n, start = [], 0, None

    for i, line in enumerate(lines):
        if C.is_header(line):
            flush(i)
            continue
        n = len(line.split())
        if n == 0:
            continue
        if cur_n + n > max_words:
            flush(i)
            if n > max_words:  # one long paragraph: keep its first max_words words
                words = line.split()[:max_words]
                out.append({"line_start": i, "line_end": i + 1, "lines": [" ".join(words)], "truncated": True})
                continue
        if start is None:
            start = i
        cur.append(line)
        cur_n += n
    flush(len(lines))
    return out


def sample_split(split: str, files: list[Path], n_chunks: int, rng, args, reject=None) -> tuple[list[dict], dict]:
    per_source, words = {}, {}
    for src, path in zip(C.SOURCES, files):
        cands = candidate_chunks(path, args.max_words, args.min_words)
        per_source[src] = cands
        words[src] = sum(len(l.split()) for c in cands for l in c["lines"])
    total = sum(words.values())
    quota = {s: max(1, round(n_chunks * words[s] / total)) for s in C.SOURCES}
    chosen, stats = [], {"quota": quota, "rejected_overlap": {}, "candidates": {s: len(v) for s, v in per_source.items()}}
    for src, path in zip(C.SOURCES, files):
        order = rng.permutation(len(per_source[src]))
        got, rej = 0, 0
        for j in order:
            if got >= quota[src]:
                break
            c = per_source[src][j]
            text = "\n".join(c["lines"])
            if reject is not None and reject(text):
                rej += 1
                continue
            chosen.append({
                "split": split, "source": src, "file": f"{split}/{path.name}",
                "line_start": c["line_start"], "line_end": c["line_end"],
                "truncated": bool(c.get("truncated", False)), "text": text,
            })
            got += 1
        stats["rejected_overlap"][src] = rej
    chosen.sort(key=lambda r: (C.SOURCES.index(r["source"]), r["line_start"]))
    for k, r in enumerate(chosen):
        r["id"] = f"{split}-{k:04d}"
        r["n_words"] = len(r["text"].split())
        r["sha256"] = hashlib.sha256(r["text"].encode("utf-8")).hexdigest()
    return chosen, stats


def chunk_ids(vocab: C.Vocab, text: str) -> np.ndarray:
    return np.asarray(vocab.encode_words(text.split()), dtype=np.int64)


def word_features(rows: list[dict], vocab: C.Vocab, counts10, counts100, ids10, ids100) -> dict:
    """Per-word, model-independent features in sample order."""
    chunks = [chunk_ids(vocab, r["text"]) for r in rows]
    feats = {k: [] for k in ["chunk", "source", "widx", "line_start", "freq10", "freq100",
                             "ctx_prev", "ctx_bigram", "cap", "punct"]}
    for ci, (r, ids) in enumerate(zip(rows, chunks)):
        words = r["text"].split()
        n = len(ids)
        line_starts = np.zeros(n, dtype=bool)
        k = 0
        for line in r["text"].split("\n"):
            m = len(line.split())
            if m:
                line_starts[k] = True
            k += m
        prev = np.zeros(n, dtype=np.int16)
        seen: dict[int, int] = {}
        bigr = np.zeros(n, dtype=bool)
        seen_bi: set[tuple[int, int]] = set()
        for i, t in enumerate(ids.tolist()):
            prev[i] = min(seen.get(t, 0), 32767)
            seen[t] = seen.get(t, 0) + 1
            if i > 0:
                key = (int(ids[i - 1]), t)
                bigr[i] = key in seen_bi
                seen_bi.add(key)
        feats["chunk"].append(np.full(n, ci, dtype=np.int32))
        feats["source"].append(np.full(n, C.SOURCES.index(r["source"]), dtype=np.int8))
        feats["widx"].append(np.arange(n, dtype=np.int32))
        feats["line_start"].append(line_starts)
        feats["freq10"].append(counts10[ids].astype(np.int64))
        feats["freq100"].append(counts100[ids].astype(np.int64))
        feats["ctx_prev"].append(prev)
        feats["ctx_bigram"].append(bigr)
        feats["cap"].append(np.array([w[:1].isupper() and C.norm(w) not in PRONOUN_I for w in words], dtype=bool) & ~line_starts)
        feats["punct"].append(np.array([not any(ch.isalnum() for ch in w) for w in words], dtype=bool))
    out = {k: np.concatenate(v) for k, v in feats.items()}
    # longest n-gram ending at word i (inside its chunk) that is attested in train
    for name, ids_train in (("match10", ids10), ("match100", ids100)):
        match = [np.zeros(len(c), dtype=np.int8) for c in chunks]
        alive = [np.ones(len(c), dtype=bool) for c in chunks]
        for order in range(1, MAX_N + 1):
            s = C.ngram_set(ids_train, order)
            for c, mt, al in zip(chunks, match, alive):
                hit = np.zeros(len(c), dtype=bool)
                hit[order - 1:] = C.in_sorted(s, C.ngram_hashes(c, order))
                al &= hit
                mt[al] = order
            del s
        out[name] = np.concatenate(match)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "babylm_instruments")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--dev-chunks", type=int, default=420)
    ap.add_argument("--train-chunks", type=int, default=280)
    ap.add_argument("--max-words", type=int, default=360)
    ap.add_argument("--min-words", type=int, default=120)
    ap.add_argument("--max-overlap", type=float, default=0.05)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    files = {s: C.split_files(args.data_root, s) for s in ("dev", "train_10M", "train_100M")}
    zips = {p.name: C.sha256_file(p) for p in sorted((args.data_root / "zips").glob("*.zip"))}
    file_hashes = {f"{s}/{p.name}": C.sha256_file(p) for s in files for p in files[s]}

    vocab = C.Vocab()
    ids100 = C.encode_split(vocab, files["train_100M"])
    ids10 = C.encode_split(vocab, files["train_10M"])
    print(f"encoded train_100M {len(ids100):,} words, train_10M {len(ids10):,}, vocab {len(vocab):,} ({time.time()-t0:.0f}s)", flush=True)
    set100_13 = C.ngram_set(ids100, 13)
    set10_13 = C.ngram_set(ids10, 13)

    def overlap(text: str, s13: np.ndarray) -> float:
        h = C.ngram_hashes(chunk_ids(vocab, text), 13)
        return float(C.in_sorted(s13, h).mean()) if len(h) else 0.0

    rng = np.random.default_rng(args.seed)
    dev_rows, dev_stats = sample_split("dev", files["dev"], args.dev_chunks, rng, args,
                                       reject=lambda t: overlap(t, set100_13) > args.max_overlap)
    train_rows, train_stats = sample_split("train_10M", files["train_10M"], args.train_chunks, rng, args)

    def rate(rows, s13):
        hit = tot = 0
        for r in rows:
            h = C.ngram_hashes(chunk_ids(vocab, r["text"]), 13)
            hit += int(C.in_sorted(s13, h).sum())
            tot += len(h)
        return hit / max(tot, 1)

    overlap_rates = {
        "dev_vs_train_10M_13gram": rate(dev_rows, set10_13),
        "dev_vs_train_100M_13gram": rate(dev_rows, set100_13),
        "dev_max_chunk_vs_train_100M_13gram": max(overlap(r["text"], set100_13) for r in dev_rows),
        "train_sample_vs_train_10M_13gram": rate(train_rows, set10_13),
    }
    del set100_13, set10_13
    print("overlap", overlap_rates, f"({time.time()-t0:.0f}s)", flush=True)

    # Encode sample words into the same vocab before counting (unseen words get count 0).
    for r in dev_rows + train_rows:
        chunk_ids(vocab, r["text"])
    V = len(vocab)
    counts10 = np.bincount(ids10, minlength=V)
    counts100 = np.bincount(ids100, minlength=V)

    sample_dir = REPO / "babylm" / "instruments" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION, "seed": args.seed, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {k: getattr(args, k) for k in ("dev_chunks", "train_chunks", "max_words", "min_words", "max_overlap")},
        "zip_sha256": zips, "file_sha256": file_hashes, "overlap": overlap_rates,
        "word_normal_form": "lowercase, strip surrounding punctuation (babylm.instruments.corpus.norm)",
        "splits": {},
    }
    for name, rows, stats in (("dev", dev_rows, dev_stats), ("train", train_rows, train_stats)):
        jl = args.out / f"{name}_sample_{VERSION}.jsonl"
        with open(jl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(sample_dir / f"{name}_sample_{VERSION}.index.tsv", "w") as f:
            f.write("id\tsource\tfile\tline_start\tline_end\tn_words\ttruncated\tsha256\n")
            for r in rows:
                f.write(f"{r['id']}\t{r['source']}\t{r['file']}\t{r['line_start']}\t{r['line_end']}\t{r['n_words']}\t{int(r['truncated'])}\t{r['sha256']}\n")
        feats = word_features(rows, vocab, counts10, counts100, ids10, ids100)
        print(f"features {name} done ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(args.out / f"features_{name}_{VERSION}.npz", **feats)
        manifest["splits"][name] = {
            "jsonl_sha256": C.sha256_file(jl), "chunks": len(rows), "words": int(sum(r["n_words"] for r in rows)),
            "words_by_source": {s: int(sum(r["n_words"] for r in rows if r["source"] == s)) for s in C.SOURCES},
            "features_sha256": C.sha256_file(args.out / f"features_{name}_{VERSION}.npz"), **stats,
        }
    with open(sample_dir / f"manifest_{VERSION}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["splits"], indent=1), f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
