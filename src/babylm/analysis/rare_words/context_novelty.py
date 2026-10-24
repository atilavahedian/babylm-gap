"""For each dev-sample word: does its (prev, word) bigram / (prev2, prev1, word) trigram of normal forms occur in
train_10M and in train_100M? Then the gap per word by band x first/repeat x local-context novelty."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rare_words as RW  # noqa: E402

OUT, RAW = Path(sys.argv[1]), Path(sys.argv[2])
DROP = np.load(sys.argv[3]) if len(sys.argv) > 3 else None
TAG = "_filtered" if DROP is not None else ""
M10 = "babylm-baseline-10m-gpt-bert-causal-focus"
M100 = "babylm-baseline-100m-gpt-bert-causal-focus"
rows = [json.loads(l) for l in (OUT / "sample.jsonl").read_text().splitlines()]
lines = (OUT / "words.tsv").read_text(encoding="utf-8").split("\n")
hdr = lines[0].split("\t")
recs = [l.split("\t") for l in lines[1:] if l]
col = {h: i for i, h in enumerate(hdr)}
norm = [r[col["norm"]] for r in recs]
c10 = np.array([int(r[col["c10"]]) for r in recs])
rep300 = np.array([int(r[col["repeat300"]]) for r in recs])
N = len(recs)
piece_len = [len(r["text"].split()) for r in rows]
# dev bigrams / trigrams (within piece)
bi = [None] * N
tri = [None] * N
i = 0
for L in piece_len:
    for j in range(L):
        if j >= 1:
            bi[i + j] = (norm[i + j - 1], norm[i + j])
        if j >= 2:
            tri[i + j] = (norm[i + j - 2], norm[i + j - 1], norm[i + j])
    i += L
bi_set = set(b for b in bi if b is not None)
tri_set = set(t for t in tri if t is not None)
found = {}
for split in ("train_10M", "train_100M"):
    fb, ft = set(), set()
    for s in RW.SOURCES:
        with open(RAW / split / f"{s}.train", encoding="utf-8") as f:
            for ln in f:
                if RW.is_header(ln):
                    continue
                w = [RW.norm(x) for x in ln.split()]
                for k in range(1, len(w)):
                    b = (w[k - 1], w[k])
                    if b in bi_set:
                        fb.add(b)
                    if k >= 2:
                        t = (w[k - 2], w[k - 1], w[k])
                        if t in tri_set:
                            ft.add(t)
    found[split] = (fb, ft)
    print(split, "dev bigrams found", len(fb), "of", len(bi_set), "trigrams", len(ft), "of", len(tri_set), flush=True)
nll10 = np.load(OUT / f"{M10}.npz")["word_nll"].astype(np.float64)
nll100 = np.load(OUT / f"{M100}.npz")["word_nll"].astype(np.float64)
gap = nll10 - nll100
keepw = np.ones(N, bool) if DROP is None else ~DROP
band = np.where(c10 == 0, "0", np.where(c10 <= 63, "1-63", np.where(c10 < 8192, "64-8k", "8k+")))
b10 = np.array([b is not None and b in found["train_10M"][0] for b in bi])
b100 = np.array([b is not None and b in found["train_100M"][0] for b in bi])
t10 = np.array([t is not None and t in found["train_10M"][1] for t in tri])
t100 = np.array([t is not None and t in found["train_100M"][1] for t in tri])
res = {}
tot = gap[keepw].sum()
for k in ("0", "1-63", "64-8k", "8k+"):
    for fr, lab in ((0, "first"), (1, "repeat")):
        base = (band == k) & (rep300 == fr) & keepw
        for cname, cm in (("bigram in 10M", b10), ("bigram only in 100M", b100 & ~b10), ("bigram in neither", ~b100),
                          ("trigram in 10M", t10), ("trigram only in 100M", t100 & ~t10), ("trigram in neither", ~t100)):
            m = base & cm
            if m.sum() == 0:
                continue
            res[f"{k}|{lab}|{cname}"] = dict(n=int(m.sum()), share_of_cell=float(m.sum() / base.sum()), gap_per_word=float(gap[m].mean()),
                                            nll10=float(nll10[m].mean()), nll100=float(nll100[m].mean()), gap_share_total=float(gap[m].sum() / tot))
(OUT / f"context_novelty{TAG}.json").write_text(json.dumps(res, indent=1))
print(f"{'cell':55s} {'n':>6s} {'share':>6s} {'gap/w':>6s} {'nll10':>6s} {'nll100':>6s} {'gapsh':>6s}")
for k, v in res.items():
    print(f"{k:55s} {v['n']:6d} {v['share_of_cell']:6.1%} {v['gap_per_word']:6.2f} {v['nll10']:6.2f} {v['nll100']:6.2f} {v['gap_share_total']:6.1%}")
