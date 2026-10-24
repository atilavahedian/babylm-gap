"""Audits for attack A4, run before any screen number is reported (card "Audits"; critic amendments 4 and 6).

    python -m babylm.a4.audit leak SYN_DIR [SYN_DIR ...] --eval_dir W/osf/evaluation_data --out OUT.json
    python -m babylm.a4.audit copy SYN_DIR --probe PROBE.jsonl.gz --out OUT.json [--docs 2000]

leak (amendment 4). Words = the text lowercased with every run of non-alphanumerics (Unicode \\W and _) made a space.
  Targets: every string of every .jsonl / .json / .tsv / .csv / .txt file under --eval_dir (an item; other files are
  listed as skipped), and DEV. An item of >= 13 words is hit by a corpus if one of its 13-grams occurs in the corpus;
  an item of 6-12 words if the whole item occurs (the secondary short-item exact match: BLiMP-style items are short).
  Corpora (documents decoded from their tokenized splits; n-grams never cross a document): each synthetic split, and
  train_10M as the base rate (the generator saw train_10M, so what matters is text that only the synthetic data has).
  Per eval file: items, items hit by the synthetic text, by train_10M, by the synthetic text only (listed, up to
  --list). DEV: the share of its distinct 13-grams found in the synthetic text, in train_10M, in the synthetic only.
copy (amendment 6). Token 6- and 8-gram overlap with train_10M (share of the n-grams of the split that occur in the
  train_10M stream; n-grams holding a BOS excluded), with DEV's as the base rate. Nearest neighbour of --docs sampled
  documents (>= 32 tokens, seed 0): candidate regions = up to 8 train_10M diagonals (train position - document
  position) with the most shared 6-grams (6-grams occurring more than 256 times in train_10M are skipped), each read
  with a margin of a quarter of the length; distance = the semi-global token edit distance (Myers' bit-vector
  algorithm: the whole document against its best-matching stretch of a candidate region), minimised over candidates;
  a document with no shared 6-gram gets its length. Near-copy = distance <= 10% of the length. A document within 10%
  keeps a run of >= 6 untouched tokens when it has >= 20 tokens, so the 6-gram anchors cannot miss one. Baselines:
  DEV chunks of the same lengths (seed 0) and the probe (--probe: unblocked continuations from the same generator
  and seeds, never trained on). Writes OUT.json, OUT_nn.csv (one row per document) and OUT_nn.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from babylm.a4 import data as D
from babylm.a4 import ngram as NG
from babylm.data import corpus as C

WORD_RE = re.compile(r"[\W_]+")
LEAK_N = 13
SHORT_MIN = 6
TEXT_SUFFIXES = (".jsonl", ".json", ".tsv", ".csv", ".txt")


def log(msg: str) -> None:
    print(f"[a4.audit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------
# leak
# ------------------------------------------------------------------------------------------------
def norm_words(text: str) -> list[str]:
    return WORD_RE.sub(" ", text.lower()).split()


class Vocab:
    """Word -> id (id 0 = the document separator, never a word)."""

    def __init__(self):
        self.d: dict[str, int] = {"": 0}

    def ids(self, words: list[str]) -> np.ndarray:
        d = self.d
        return np.fromiter((d.setdefault(w, len(d)) for w in words), dtype=np.int64, count=len(words))


def split_texts(data: C.SplitData, tok) -> list[str]:
    b = D.doc_bounds(data)
    t = np.asarray(data.tokens)
    return tok.decode_batch([t[b[k] + 1:b[k + 1]].tolist() for k in range(len(b) - 1)], skip_special_tokens=False)


def corpus_ids(texts: list[str], vocab: Vocab) -> np.ndarray:
    """All documents' word ids, each followed by the separator 0 (no n-gram of a target spans one)."""
    parts = []
    for t in texts:
        parts.append(vocab.ids(norm_words(t)))
        parts.append(np.zeros(1, dtype=np.int64))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def word_ngrams(ids: np.ndarray, n: int) -> np.ndarray:
    """Hashes of the n-grams of a word-id stream that hold no separator."""
    h = NG.ngram_hashes(ids, n, word=True)
    cz = np.concatenate([[0], np.cumsum(ids == 0)])
    return h[(cz[n:n + len(h)] - cz[:len(h)]) == 0]


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def eval_items(eval_dir: str | Path) -> tuple[dict[str, list[str]], list[str]]:
    """{relative file: distinct strings} for every text file under eval_dir, and the skipped files."""
    root = Path(eval_dir)
    items: dict[str, list[str]] = {}
    skipped = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        suf = p.suffix.lower()
        if suf not in TEXT_SUFFIXES:
            skipped.append(rel)
            continue
        got: list[str] = []
        try:
            if suf == ".jsonl":
                with p.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            got += _strings(json.loads(line))
            elif suf == ".json":
                got += _strings(json.loads(p.read_text(encoding="utf-8")))
            elif suf in (".tsv", ".csv"):
                with p.open(encoding="utf-8", newline="") as f:
                    for row in csv.reader(f, delimiter="\t" if suf == ".tsv" else ","):
                        got += row
            else:
                got += p.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as ex:
            skipped.append(f"{rel} ({type(ex).__name__})")
            continue
        items[rel] = list(dict.fromkeys(got))
    return items, skipped


class Targets:
    """Probe n-grams of the eval items: (item index, n, hash) for 13-grams of long items and whole short items."""

    def __init__(self, items: dict[str, list[str]], vocab: Vocab):
        self.files, self.texts = [], []
        q_item, q_n, q_h = [], [], []
        for f, strs in items.items():
            for s in strs:
                ids = vocab.ids(norm_words(s))
                n = LEAK_N if len(ids) >= LEAK_N else len(ids)
                if n < SHORT_MIN:
                    continue
                i = len(self.texts)
                self.files.append(f)
                self.texts.append(s)
                h = NG.ngram_hashes(ids, n, word=True)
                q_item.append(np.full(len(h), i, dtype=np.int64))
                q_n.append(np.full(len(h), n, dtype=np.int64))
                q_h.append(h)
        cat = (lambda xs: np.concatenate(xs) if xs else np.zeros(0, dtype=np.int64))
        self.q_item, self.q_n, self.q_h = cat(q_item), cat(q_n), cat(q_h)

    def hits(self, ids: np.ndarray) -> np.ndarray:
        """bool per item: the corpus (word ids with separators) contains one of its probe n-grams."""
        hit = np.zeros(len(self.texts), dtype=bool)
        for n in np.unique(self.q_n).tolist():
            sel = self.q_n == n
            h = np.unique(NG.ngram_hashes(ids, n, word=True))
            found = NG.in_sorted(h, self.q_h[sel])
            hit[self.q_item[sel][found]] = True
        return hit


def leak(syn_dirs: list[str], eval_dir: str, data_dir: str | None, out: str | Path, max_list: int = 200) -> dict:
    from babylm.data import prepare as P
    tok = P.load_tokenizer()
    vocab = Vocab()
    items, skipped = eval_items(eval_dir)
    tg = Targets(items, vocab)
    log(f"{len(tg.texts):,} eval items of >= {SHORT_MIN} words in {len(items)} files ({len(skipped)} skipped)")
    dev_ids = corpus_ids(split_texts(C.load_split("dev", data_dir), tok), vocab)
    dev_h = np.unique(word_ngrams(dev_ids, LEAK_N))
    del dev_ids
    tr = C.load_split("train_10M", data_dir)
    tr_ids = corpus_ids(split_texts(tr, tok), vocab)
    tr_hit = tg.hits(tr_ids)
    tr_dev = NG.in_sorted(np.unique(NG.ngram_hashes(tr_ids, LEAK_N, word=True)), dev_h)
    del tr_ids
    files = list(items)
    pos = {f: i for i, f in enumerate(files)}
    fidx = np.asarray([pos[f] for f in tg.files], dtype=np.int64)

    def per_file(mask: np.ndarray) -> dict[str, int]:
        c = np.bincount(fidx[mask], minlength=len(files)) if len(fidx) else np.zeros(len(files), dtype=np.int64)
        return {f: int(c[i]) for i, f in enumerate(files)}

    n_items = per_file(np.ones(len(tg.texts), dtype=bool))
    rep = {"rule": f"{LEAK_N}-gram on lowercased words (non-alphanumerics -> space); items of {SHORT_MIN}-"
                   f"{LEAK_N - 1} words: whole-item exact match", "eval_dir": str(eval_dir),
           "eval_files": len(files), "eval_items": len(tg.texts), "skipped_files": skipped,
           "train_10M": {"sha256": tr.meta["sha256"], "items_hit": int(tr_hit.sum()),
                         "dev_13gram_share": float(tr_dev.mean()) if len(dev_h) else None},
           "dev_distinct_13grams": int(len(dev_h)), "synthetic": {}}
    for sd in syn_dirs:
        data = C.load_split("synth", sd)
        ids = corpus_ids(split_texts(data, tok), vocab)
        hit = tg.hits(ids)
        sy_dev = NG.in_sorted(np.unique(NG.ngram_hashes(ids, LEAK_N, word=True)), dev_h)
        only = hit & ~tr_hit
        rows = {f: {"items": n_items[f], "hit_synthetic": a, "hit_train_10M": b, "hit_synthetic_only": c}
                for f, a, b, c in zip(files, per_file(hit).values(), per_file(tr_hit).values(),
                                      per_file(only).values())}
        rep["synthetic"][str(sd)] = {
            "sha256": data.meta["sha256"], "words": data.n_words, "items_hit": int(hit.sum()),
            "items_hit_synthetic_only": int(only.sum()),
            "dev_13gram_share": float(sy_dev.mean()) if len(dev_h) else None,
            "dev_13gram_share_synthetic_only": float((sy_dev & ~tr_dev).mean()) if len(dev_h) else None,
            "per_file": {f: r for f, r in rows.items() if r["hit_synthetic"] or r["hit_train_10M"]},
            "synthetic_only_items": [{"file": tg.files[i], "text": tg.texts[i]}
                                     for i in np.flatnonzero(only)[:max_list].tolist()]}
        log(f"{sd}: {int(hit.sum()):,} items hit ({int(only.sum()):,} not in train_10M; train_10M hits "
            f"{int(tr_hit.sum()):,}); DEV 13-grams {float(sy_dev.mean()):.2e} (train_10M {float(tr_dev.mean()):.2e})")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


# ------------------------------------------------------------------------------------------------
# copy
# ------------------------------------------------------------------------------------------------
def myers_distance(pattern, text) -> int:
    """min over substrings s of text of the edit distance (unit costs) between the whole pattern and s (Myers 1999,
    Hyyro's formulation; Python ints as bit vectors of any length)."""
    m = len(pattern)
    if m == 0:
        return 0
    peq: dict = defaultdict(int)
    for i, c in enumerate(pattern):
        peq[c] |= 1 << i
    mask = (1 << m) - 1
    top = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    best = m
    for c in text:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & mask) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = (ph << 1) & mask  # a free start in the text: row 0 stays 0
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
        if score < best:
            best = score
    return best


class TrainIndex:
    """Sorted 6-gram hashes of the train_10M stream with their positions."""

    def __init__(self, tokens: np.ndarray, n: int = 6, max_occ: int = 256):
        self.tokens = np.asarray(tokens).astype(np.int64)
        h = NG.ngram_hashes(self.tokens, n)
        self.order = np.argsort(h, kind="stable")
        self.h = h[self.order]
        self.n, self.max_occ = n, max_occ

    def candidates(self, q: np.ndarray, k: int = 8) -> list[int]:
        """Up to k diagonals (train start of the document) with the most shared n-grams, at least m/4 apart."""
        m = len(q)
        qh = NG.ngram_hashes(q, self.n)
        lo = np.searchsorted(self.h, qh, "left")
        hi = np.searchsorted(self.h, qh, "right")
        cnt = hi - lo
        ok = (cnt > 0) & (cnt <= self.max_occ)
        if not ok.any():
            return []
        lo, cnt, qi = lo[ok], cnt[ok], np.flatnonzero(ok)
        start = np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(cnt.sum())
        diag = self.order[start] - np.repeat(qi, cnt)
        d, votes = np.unique(diag, return_counts=True)
        out: list[int] = []
        for j in np.lexsort((d, -votes)).tolist():
            if all(abs(int(d[j]) - x) > m // 4 for x in out):
                out.append(int(d[j]))
                if len(out) == k:
                    break
        return out

    def nn(self, q: np.ndarray, k: int = 8) -> tuple[int, int]:
        """(distance, number of candidate regions)."""
        m = len(q)
        cands = self.candidates(q, k)
        pad = max(8, m // 4)
        best = m
        ql = q.tolist()
        for dg in cands:
            a, b = max(0, dg - pad), min(len(self.tokens), dg + m + pad)
            best = min(best, myers_distance(ql, self.tokens[a:b].tolist()))
            if best == 0:
                break
        return best, len(cands)


def ngram_overlap(tokens: np.ndarray, train_sorted: dict[int, np.ndarray], bos_id: int = C.BOS_ID) -> dict[str, float]:
    t = np.asarray(tokens).astype(np.int64)
    cb = np.concatenate([[0], np.cumsum(t == bos_id)])
    out = {}
    for n, ts in train_sorted.items():
        h = NG.ngram_hashes(t, n)
        clean = (cb[n:n + len(h)] - cb[:len(h)]) == 0
        out[f"{n}gram"] = float(NG.in_sorted(ts, h[clean]).mean()) if clean.any() else None
    return out


def split_docs(data: C.SplitData) -> list[np.ndarray]:
    b = D.doc_bounds(data)
    t = np.asarray(data.tokens).astype(np.int64)
    return [t[b[k] + 1:b[k + 1]] for k in range(len(b) - 1)]


def copy_audit(syn_dir: str, probe: str | None, data_dir: str | None, out: str | Path, n_docs: int = 2000,
               min_len: int = 32, seed: int = 0) -> dict:
    t0 = time.time()
    tr = C.load_split("train_10M", data_dir)
    dev = C.load_split("dev", data_dir)
    syn = C.load_split("synth", syn_dir)
    trt = np.asarray(tr.tokens)
    train_sorted = {n: np.unique(NG.ngram_hashes(trt, n)) for n in (6, 8)}
    rep = {"syn_dir": str(syn_dir), "syn_sha256": syn.meta["sha256"], "train_10M_sha256": tr.meta["sha256"],
           "overlap": {"synthetic": ngram_overlap(syn.tokens, train_sorted),
                       "dev": ngram_overlap(dev.tokens, train_sorted)}}
    log(f"token n-gram overlap with train_10M: synthetic {rep['overlap']['synthetic']} dev {rep['overlap']['dev']}")
    del train_sorted
    rng = np.random.Generator(np.random.PCG64(seed))
    sdocs = [d for d in split_docs(syn) if len(d) >= min_len]
    sdocs = [sdocs[i] for i in sorted(rng.choice(len(sdocs), min(n_docs, len(sdocs)), replace=False).tolist())]
    sets = {"synthetic": sdocs}
    ddocs = [d for d in split_docs(dev) if len(d) > min_len]
    chunks = []
    for d in sdocs:  # a DEV chunk of the same length from a random long-enough DEV document
        pool = [x for x in (ddocs[i] for i in rng.integers(0, len(ddocs), 64)) if len(x) > len(d)]
        if pool:
            x = pool[0]
            s = int(rng.integers(0, len(x) - len(d) + 1))
            chunks.append(x[s:s + len(d)])
    sets["dev_chunks"] = chunks
    if probe:
        from babylm.data import prepare as P
        texts = [t for t in (D.clean_doc(r["text"]) for r in D.read_shard(probe)) if t]
        dt = P.tokenize_docs(P.load_tokenizer(), texts)
        pdocs = [x.astype(np.int64) for x in dt.ids if len(x) >= min_len]
        sets["probe"] = [pdocs[i] for i in sorted(rng.choice(len(pdocs), min(n_docs, len(pdocs)),
                                                             replace=False).tolist())] if pdocs else []
    idx = TrainIndex(trt)
    rows, summ = [], {}
    for name, docs in sets.items():
        dist = []
        for i, q in enumerate(docs):
            dd, nc = idx.nn(q)
            dist.append(dd / len(q))
            rows.append({"set": name, "i": i, "tokens": len(q), "distance": dd, "norm": round(dd / len(q), 5),
                         "candidates": nc})
        a = np.asarray(dist)
        summ[name] = {"docs": int(len(a)), "near_copy_rate": float((a <= 0.1).mean()) if len(a) else None,
                      "exact_rate": float((a == 0).mean()) if len(a) else None,
                      "mean_norm_distance": float(a.mean()) if len(a) else None,
                      "quantiles_norm_distance": {q: float(np.quantile(a, q)) for q in (0.01, 0.05, 0.1, 0.5)}
                      if len(a) else None}
        log(f"NN {name}: {summ[name]}")
    rep["nearest_neighbour"] = {"rule": "semi-global token edit distance to train_10M around the best 6-gram "
                                        "diagonals; near-copy = distance <= 10% of the length", **summ}
    rep["seconds"] = time.time() - t0
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    stem = str(out).removesuffix(".json")
    with open(stem + "_nn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["set"])
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in sets:
            v = [r["norm"] for r in rows if r["set"] == name]
            if v:
                ax.hist(v, bins=50, range=(0, 1), histtype="step", label=f"{name} (near-copy "
                        f"{summ[name]['near_copy_rate']:.3f})")
        ax.axvline(0.1, color="k", lw=0.6, ls="--")
        ax.set_xlabel("NN edit distance to train_10M / length (tokens)")
        ax.set_ylabel("documents")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(stem + "_nn.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass
    return rep


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("leak")
    lk.add_argument("syn_dirs", nargs="+")
    lk.add_argument("--eval_dir", required=True)
    lk.add_argument("--data_dir", default=None)
    lk.add_argument("--out", required=True)
    lk.add_argument("--list", type=int, default=200)
    cp = sub.add_parser("copy")
    cp.add_argument("syn_dir")
    cp.add_argument("--probe", default=None)
    cp.add_argument("--data_dir", default=None)
    cp.add_argument("--out", required=True)
    cp.add_argument("--docs", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "leak":
        return leak(a.syn_dirs, a.eval_dir, a.data_dir, a.out, a.list)
    return copy_audit(a.syn_dir, a.probe, a.data_dir, a.out, a.docs)


if __name__ == "__main__":
    sys.exit(main() and None)
