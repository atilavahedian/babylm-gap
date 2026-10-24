"""Mini-ruler splits for fast falsification: nested subsets of train_10M (train_1M c train_2M c train_5M c train_10M).

    python -m babylm.moonshot.mini_ruler [--data_dir data/babylm] [--sizes 1,2,5] [--piece_tokens 2048]

train_10M is cut into pieces of at most `piece_tokens` tokens: every document is split at word-initial tokens, and a
piece that does not start a document gets a BOS (id 0) in front (BOS attaches to the piece's first word, as in
corpus.py). Per source, one seeded permutation of its pieces (PCG64([seed, source index])) is taken as a prefix whose
word total is closest to size x (the source's share of train_10M words); prefixes of one permutation make the subsets
nested by construction. Output: <data_dir>/bin/train_<n>M.{bin,words.bin,src.bin,meta.json} in the corpus format,
word ids renumbered to 0..W-1 in stream order. These subsets are a proxy for falsification only (labels, never
verdicts): the official ruler is train_10M/20M/30M/50M/100M.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from babylm.data import corpus as C

SEED = 20261010


def pieces_of(tokens: np.ndarray, words: np.ndarray, piece_tokens: int) -> list[tuple[int, int]]:
    """[start, end) token ranges: documents (from each BOS) cut into pieces of <= piece_tokens at word starts."""
    bos = np.flatnonzero(tokens == C.BOS_ID)
    ends = np.append(bos[1:], len(tokens))
    winit = C.word_initial(words)
    out: list[tuple[int, int]] = []
    for s, e in zip(bos.tolist(), ends.tolist()):
        b = s
        while b < e:
            if e - b <= piece_tokens:
                out.append((b, e))
                break
            # last word start in (b, b + piece_tokens]
            cand = np.flatnonzero(winit[b + 1:b + piece_tokens + 1]) + b + 1
            cut = int(cand[-1]) if len(cand) else b + piece_tokens
            out.append((b, cut))
            b = cut
    return out


def build(data_dir: Path, sizes_m: list[int], piece_tokens: int = 2048, seed: int = SEED) -> dict:
    tr = C.load_split("train_10M", data_dir)
    tok, wrd, src = np.asarray(tr.tokens), np.asarray(tr.words), np.asarray(tr.src)
    pcs = pieces_of(tok, wrd, piece_tokens)
    p_src = np.array([int(src[s]) for s, _ in pcs])
    p_words = np.array([len(np.unique(wrd[s:e])) for s, e in pcs])
    total_words = tr.n_words
    share = {i: tr.meta["per_source"][name]["words"] / total_words for i, name in enumerate(C.SOURCES)}
    order = {}
    for i in range(len(C.SOURCES)):
        idx = np.flatnonzero(p_src == i)
        rng = np.random.Generator(np.random.PCG64([seed, i]))
        order[i] = idx[rng.permutation(len(idx))]
    bdir = C.bin_dir(data_dir)
    report = {}
    for n in sorted(sizes_m):
        target = n * 1_000_000
        chosen = []
        for i in range(len(C.SOURCES)):
            quota = target * share[i]
            cum = np.cumsum(p_words[order[i]])
            k = int(np.argmin(np.abs(cum - quota))) + 1 if len(cum) else 0
            chosen.extend(order[i][:k].tolist())
        chosen.sort()  # keep train_10M stream order
        t_parts, w_parts, s_parts = [], [], []
        next_w = 0
        for j in chosen:
            s, e = pcs[j]
            t = tok[s:e].copy()
            w = wrd[s:e].astype(np.int64)
            sr = src[s:e].copy()
            if t[0] != C.BOS_ID:
                t = np.concatenate([[C.BOS_ID], t]).astype(tok.dtype)
                w = np.concatenate([[w[0]], w])
                sr = np.concatenate([[sr[0]], sr]).astype(src.dtype)
            _, inv = np.unique(w, return_inverse=True)  # renumber words of this piece, stream order kept
            w = inv + next_w
            next_w = int(w.max()) + 1
            t_parts.append(t)
            w_parts.append(w)
            s_parts.append(sr)
        T = np.concatenate(t_parts).astype("<u2")
        W = np.concatenate(w_parts).astype("<u4")
        S = np.concatenate(s_parts).astype("u1")
        name = f"train_{n}M"
        T.tofile(bdir / f"{name}.bin")
        W.tofile(bdir / f"{name}.words.bin")
        S.tofile(bdir / f"{name}.src.bin")
        per = {}
        for i, sname in enumerate(C.SOURCES):
            m = S == i
            per[sname] = {"tokens": int(m.sum()), "words": int(len(np.unique(W[m])))}
        meta = {"split": name, "tokens": int(len(T)), "words": int(next_w), "docs": int((T == C.BOS_ID).sum()),
                "tokens_per_word": float(len(T) / max(1, next_w)), "per_source": per,
                "parent": "train_10M", "parent_sha256": tr.meta["sha256"], "tokenizer_sha256": tr.meta["tokenizer_sha256"], "piece_tokens": piece_tokens,
                "seed": seed, "pieces": len(chosen), "sha256": C.sha256_array(T),
                "format": tr.meta["format"], "label": "mini-ruler proxy (falsification only)"}
        (bdir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
        report[name] = {k: meta[k] for k in ("tokens", "words", "docs", "pieces", "sha256")}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--sizes", default="1,2,5")
    ap.add_argument("--piece_tokens", type=int, default=2048)
    a = ap.parse_args()
    rep = build(Path(a.data_dir) if a.data_dir else C.DEFAULT_DATA_DIR, [int(x) for x in a.sizes.split(",")],
                a.piece_tokens)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
