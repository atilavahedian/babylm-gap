"""Data for attack A4: document halves (A4b), seeds, the synthetic-split writer and the N2 sentence shuffle.

    python -m babylm.a4.data halves [--data_dir D] [--seed 4242]                # D/bin/train_10M_hA, train_10M_hB
    python -m babylm.a4.data build OUT_DIR SHARD.jsonl.gz ... [--words W]      # generated shards -> OUT_DIR/bin/synth.*
    python -m babylm.a4.data shuffle SRC_DIR OUT_DIR [--seed 0]                # N2: words shuffled within sentences

Halves: per source (corpus.SOURCES order, one generator PCG64(seed)), the documents of train_10M in a seeded
permutation are dealt one by one to the half holding fewer of that source's words, so each half has about half of
every source's words. Each half is written like prepare.py writes a split (stream order, BOS before each document,
word ids re-based, source ids kept), its document list in the meta.

Seeds (seed_starts): prompt_len-token windows that start at a line start (a document's first line starts at its BOS)
and hold no other document's BOS, tiling the stream greedily (each starts at the first allowed position at or after
the previous one's end); seed_order shuffles them with PCG64(seed). The generator uses them in that order and, if the
word target needs more rows, again in the same order (a new pass; every continuation is an independent draw).

Synthetic split (write_synth_split): every document's lines normalised (corpus.normalize_line), empty lines and empty
documents dropped, documents shuffled with PCG64(seed), then tokenized and laid out exactly as prepare.py lays out a
split (source id per document; 6 = "self" for N1's unseeded text) into <out>/bin/synth.{bin,words.bin,src.bin,
meta.json}: what babylm.harness.cycle loads for --sources synthetic:<out>:EPOCHS. <out>/docs.txt.gz keeps the exact
text (documents separated by a blank line).

N2 (shuffle_text): per line, a leading speaker label ("*CHI:", "A:") stays first; the rest is cut into sentences after
every word ending in . ! or ? (closing quotes or brackets allowed after it); per sentence, a final punctuation-only
word stays last, a punctuation run ending the last word is moved to whatever word ends up last, and the words are
shuffled. Word count per line is exact; the multiset of words is exact up to that moved punctuation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

from babylm.data import corpus as C
from babylm.data import prepare as P
from babylm.harness import levers as LV

SOURCES_EXT = tuple(C.SOURCES) + ("self",)
SELF_SRC = len(C.SOURCES)
HALF_NAMES = ("train_10M_hA", "train_10M_hB")
HALF_SEED = 4242


# ------------------------------------------------------------------------------------------------
# documents and halves
# ------------------------------------------------------------------------------------------------
def doc_bounds(data: C.SplitData) -> np.ndarray:
    """[n_docs + 1] token offsets: document k = tokens[b[k]:b[k+1]], its BOS first."""
    return np.append(data.doc_starts(), data.n_tokens).astype(np.int64)


def doc_words(data: C.SplitData, b: np.ndarray) -> np.ndarray:
    first = np.asarray(data.words).astype(np.int64)[b[:-1]]
    return np.append(first[1:], data.n_words) - first


def split_halves(data: C.SplitData, seed: int = HALF_SEED) -> tuple[list[int], list[int]]:
    b = doc_bounds(data)
    nw = doc_words(data, b)
    src = np.asarray(data.src)[b[:-1]]
    rng = np.random.Generator(np.random.PCG64(seed))
    halves: tuple[list[int], list[int]] = ([], [])
    for sid in range(len(C.SOURCES)):
        idx = np.flatnonzero(src == sid)
        load = [0, 0]
        for k in idx[rng.permutation(len(idx))]:
            h = 0 if load[0] <= load[1] else 1
            halves[h].append(int(k))
            load[h] += int(nw[k])
    return sorted(halves[0]), sorted(halves[1])


def subset_split(data: C.SplitData, docs: list[int], name: str, data_dir: Path, extra: dict) -> dict:
    """Write documents `docs` (indices, stream order) of a split as split `name`, word ids re-based to 0."""
    b = doc_bounds(data)
    nw = doc_words(data, b)
    tok, wid, src = np.asarray(data.tokens), np.asarray(data.words).astype(np.int64), np.asarray(data.src)
    pt, pw, ps = [], [], []
    per: dict[str, dict[str, int]] = {}
    off = 0
    for k in docs:
        a, e = int(b[k]), int(b[k + 1])
        pt.append(tok[a:e])
        pw.append(wid[a:e] - wid[a] + off)
        ps.append(src[a:e])
        st = per.setdefault(C.SOURCES[int(src[a])], {"tokens": 0, "words": 0, "docs": 0})
        st["tokens"] += e - a
        st["words"] += int(nw[k])
        st["docs"] += 1
        off += int(nw[k])
    return P.write_split(name, np.concatenate(pt), np.concatenate(pw), np.concatenate(ps), per, extra, data_dir,
                         data.meta["tokenizer_sha256"])


def make_halves(data_dir: str | Path | None = None, seed: int = HALF_SEED) -> list[dict]:
    data_dir = Path(data_dir or C.DEFAULT_DATA_DIR)
    data = C.load_split("train_10M", data_dir)
    out = []
    for name, docs, other in zip(HALF_NAMES, split_halves(data, seed), HALF_NAMES[::-1]):
        out.append(subset_split(data, docs, name, data_dir,
                                {"subset_of": "train_10M", "train_10M_sha256": data.meta["sha256"], "half_seed": seed,
                                 "complement": other, "doc_indices": docs,
                                 "rule": "per source: seeded permutation dealt to the half with fewer words"}))
    return out


# ------------------------------------------------------------------------------------------------
# seeds
# ------------------------------------------------------------------------------------------------
def seed_starts(tokens: np.ndarray, nl_ids: np.ndarray, prompt_len: int = 64, bos_id: int = C.BOS_ID) -> np.ndarray:
    tok = np.asarray(tokens)
    N = len(tok)
    st = LV.line_starts(tok, nl_ids, bos_id)
    q = np.unique(np.where((st > 0) & (tok[np.maximum(st - 1, 0)] == bos_id), st - 1, st))
    bos = np.append(np.flatnonzero(tok == bos_id), N)
    nxt = bos[np.searchsorted(bos, q, side="right")]  # the first BOS after q (N if none)
    cand = q[(q + prompt_len <= N) & (nxt >= q + prompt_len)]
    out, end = [], -1
    for x in cand.tolist():
        if x >= end:
            out.append(x)
            end = x + prompt_len
    return np.asarray(out, dtype=np.int64)


def seed_order(starts: np.ndarray, seed: int = 0) -> np.ndarray:
    return starts[np.random.Generator(np.random.PCG64(seed)).permutation(len(starts))]


# ------------------------------------------------------------------------------------------------
# synthetic splits
# ------------------------------------------------------------------------------------------------
def clean_doc(text: str) -> str:
    text = text.replace(C.BOS_TOKEN, " ")  # never a BOS inside a document
    return "\n".join(ln for ln in (C.normalize_line(x) for x in text.split("\n")) if ln)


def write_synth_split(texts: list[str], srcs: list[int], out_dir: str | Path, extra: dict, seed: int = 0,
                      shuffle: bool = True) -> dict:
    docs = [(d, int(s)) for d, s in ((clean_doc(t), s) for t, s in zip(texts, srcs)) if d]
    if shuffle:
        docs = [docs[i] for i in np.random.Generator(np.random.PCG64(seed)).permutation(len(docs))]
    dt = P.tokenize_docs(P.load_tokenizer(), [d for d, _ in docs])
    n_tok = sum(len(x) + 1 for x in dt.ids)
    tokens = np.empty(n_tok, dtype=np.uint16)
    words = np.empty(n_tok, dtype=np.uint32)
    src = np.empty(n_tok, dtype=np.uint8)
    per: dict[str, dict[str, int]] = {}
    p = woff = 0
    for (_, sid), ids, wid, n in zip(docs, dt.ids, dt.wids, dt.nwords):  # prepare.assemble's layout
        L = len(ids) + 1
        tokens[p] = C.BOS_ID
        words[p] = woff
        tokens[p + 1:p + L] = ids
        words[p + 1:p + L] = wid.astype(np.int64) + woff
        src[p:p + L] = sid
        st = per.setdefault(SOURCES_EXT[sid], {"tokens": 0, "words": 0, "docs": 0})
        st["tokens"] += L
        st["words"] += int(n)
        st["docs"] += 1
        p += L
        woff += int(n)
    assert p == n_tok
    out_dir = Path(out_dir)
    meta = P.write_split("synth", tokens, words, src, per, {"doc_shuffle_seed": seed if shuffle else None, **extra},
                         out_dir, C.sha256_file(C.TOKENIZER_PATH))
    with gzip.open(out_dir / "docs.txt.gz", "wt", encoding="utf-8") as f:
        for d, _ in docs:
            f.write(d + "\n\n")
    return meta


def read_shard(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def shard_stats_path(shard: str | Path) -> Path:
    return Path(str(shard).removesuffix(".jsonl.gz") + ".json")


def build_from_shards(out_dir: str | Path, shards: list[str | Path], seed: int = 0, words: int | None = None) -> dict:
    """One synthetic split from generator shards (babylm.a4.generate). words: each shard contributes its rows in
    order until words / len(shards) kept words (None = every row)."""
    texts, srcs, stats = [], [], []
    for sh in shards:
        rows = read_shard(sh)
        if words is not None:
            quota, tot, keep = words / len(shards), 0, 0
            while keep < len(rows) and tot < quota:
                tot += len(clean_doc(rows[keep]["text"]).split())
                keep += 1
            if tot < quota:
                raise SystemExit(f"{sh}: {tot:,} words < quota {quota:,.0f}")
            rows = rows[:keep]
        texts += [r["text"] for r in rows]
        srcs += [r["src"] for r in rows]
        sp = shard_stats_path(sh)
        stats.append({"shard": str(sh), "rows_used": len(rows),
                      "gen": json.loads(sp.read_text()) if sp.exists() else None})
    return write_synth_split(texts, srcs, out_dir, {"words_target": words, "shards": stats}, seed)


# ------------------------------------------------------------------------------------------------
# N2
# ------------------------------------------------------------------------------------------------
SPEAKER = re.compile(r"^\*?[A-Za-z][A-Za-z0-9_]{0,7}:$")
SENT_END = re.compile(r"[.!?][\"'”’)\]]*$")
PUNCT_ONLY = re.compile(r"^[^\w]+$")
TRAIL = re.compile(r"[.!?,;:\"'”’)\]]+$")


def _shuffle_sentence(ws: list[str], rng: np.random.Generator) -> list[str]:
    if len(ws) < 2:
        return ws
    tail = []
    if PUNCT_ONLY.match(ws[-1]):
        tail, ws = ws[-1:], ws[:-1]
    m = TRAIL.search(ws[-1])
    punct = ""
    if m and m.start() > 0:
        punct, ws = m.group(0), ws[:-1] + [ws[-1][:m.start()]]
    out = [ws[i] for i in rng.permutation(len(ws))]
    out[-1] += punct
    return out + tail


def shuffle_text(text: str, rng: np.random.Generator) -> str:
    lines = []
    for line in text.split("\n"):
        ws = line.split()
        out = []
        if ws and SPEAKER.match(ws[0]):
            out, ws = ws[:1], ws[1:]
        cur: list[str] = []
        for w in ws:
            cur.append(w)
            if SENT_END.search(w):
                out += _shuffle_sentence(cur, rng)
                cur = []
        out += _shuffle_sentence(cur, rng)
        lines.append(" ".join(out))
    return "\n".join(lines)


def read_synth_docs(src_dir: str | Path) -> tuple[list[str], list[int]]:
    with gzip.open(Path(src_dir) / "docs.txt.gz", "rt", encoding="utf-8") as f:
        texts = [t for t in f.read().split("\n\n") if t]
    data = C.load_split("synth", src_dir)
    srcs = np.asarray(data.src)[data.doc_starts()].tolist()
    if len(srcs) != len(texts):
        raise SystemExit(f"{src_dir}: {len(texts)} texts but {len(srcs)} documents")
    return texts, srcs


def make_shuffled(src_dir: str | Path, out_dir: str | Path, seed: int = 0) -> dict:
    texts, srcs = read_synth_docs(src_dir)
    rng = np.random.Generator(np.random.PCG64([seed, 2]))
    sh = [shuffle_text(t, rng) for t in texts]
    if any(len(a.split()) != len(b.split()) for a, b in zip(texts, sh)):
        raise SystemExit("N2 changed a document's word count")
    meta = json.loads((C.bin_dir(src_dir) / "synth.meta.json").read_text())
    return write_synth_split(sh, srcs, out_dir, {"null": "N2: words shuffled within sentences", "shuffle_seed": seed,
                                                 "of": str(src_dir), "of_sha256": meta["sha256"]},
                             seed, shuffle=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("halves")
    h.add_argument("--data_dir", default=None)
    h.add_argument("--seed", type=int, default=HALF_SEED)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("shards", nargs="+")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--words", type=float, default=None)
    s = sub.add_parser("shuffle")
    s.add_argument("src_dir")
    s.add_argument("out_dir")
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "halves":
        metas = make_halves(a.data_dir, a.seed)
    elif a.cmd == "build":
        metas = [build_from_shards(a.out_dir, a.shards, a.seed, None if a.words is None else int(a.words))]
    else:
        metas = [make_shuffled(a.src_dir, a.out_dir, a.seed)]
    for m in metas:
        print(json.dumps({k: m[k] for k in ("split", "tokens", "words", "docs", "sha256")}))


if __name__ == "__main__":
    sys.exit(main())
