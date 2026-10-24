"""Matched own-count comparison: each model's loss on words it saw k times in ITS OWN training set, minus the
unigram surprise -log(k / N_own). If the 100M model's 'knowledge beyond frequency' at matched own-count equals the
10M model's, the rare-word gap is pure occurrence count; if larger, scale buys better learning per occurrence."""
import json, sys
from pathlib import Path
import numpy as np
OUT = Path(sys.argv[1]); DROP = np.load(sys.argv[2])
M10 = "babylm-baseline-10m-gpt-bert-causal-focus"; M100 = "babylm-baseline-100m-gpt-bert-causal-focus"
lines = (OUT / "words.tsv").read_text(encoding="utf-8").split("\n"); hdr = lines[0].split("\t")
recs = [l.split("\t") for l in lines[1:] if l]; col = {h: i for i, h in enumerate(hdr)}
c10 = np.array([int(r[col["c10"]]) for r in recs]); c100 = np.array([int(r[col["c100"]]) for r in recs])
cat = np.array([r[col["cat"]] for r in recs]); rep = np.array([int(r[col["repeat300"]]) for r in recs])
z10 = np.load(OUT / f"{M10}.npz"); z100 = np.load(OUT / f"{M100}.npz")
nll10 = z10["word_nll"].astype(float); nll100 = z100["word_nll"].astype(float)
def split_tokens(z):
    tw = z["tok_word"].astype(np.int64); tn = z["tok_nll"].astype(float)
    first = np.ones(len(tw), bool); first[1:] = tw[1:] != tw[:-1]
    n = len(z["word_nll"])
    return np.bincount(tw[first], weights=tn[first], minlength=n), np.bincount(tw[~first], weights=tn[~first], minlength=n), np.bincount(tw, minlength=n)
f10, k10, nt10 = split_tokens(z10); f100, k100, nt100 = split_tokens(z100)
keep = ~DROP
N10, N100 = 9_843_976, 98_000_000  # train_10M tokens (whitespace words); train_100M ~ 98M words (10x)
import collections
# own-count bands
def band(c):
    return np.where(c == 0, "0", np.where(c <= 3, "1-3", np.where(c <= 15, "4-15", np.where(c <= 63, "16-63", np.where(c <= 255, "64-255", np.where(c <= 1023, "256-1k", np.where(c < 8192, "1k-8k", "8k+")))))))
b10, b100 = band(c10), band(c100)
print("own-count band | 10M model: n, nll, -log p_uni, nll-uni, first, cont, tok/w | 100M model: n, nll, -log p_uni, nll-uni, first, cont, tok/w")
for k in ["1-3", "4-15", "16-63", "64-255", "256-1k", "1k-8k", "8k+"]:
    m1 = keep & (b10 == k) & (rep == 0); m2 = keep & (b100 == k) & (rep == 0)
    u1 = -np.log(c10[m1] / N10); u2 = -np.log(c100[m2] / N100)
    print(f"{k:8s} | {m1.sum():6d} {nll10[m1].mean():6.2f} {u1.mean():6.2f} {(nll10[m1]-u1).mean():6.2f} {f10[m1].mean():5.2f} {k10[m1].mean():5.2f} {nt10[m1].mean():4.2f} | "
          f"{m2.sum():6d} {nll100[m2].mean():6.2f} {u2.mean():6.2f} {(nll100[m2]-u2).mean():6.2f} {f100[m2].mean():5.2f} {k100[m2].mean():5.2f} {nt100[m2].mean():4.2f}")
print("\n(first uses only; nll-uni = loss minus own-corpus unigram surprise = what context + knowledge buy)")
# same words, both models: words with c10 in 1-63 AND c100 in 1-63 (the 100M model saw them about as rarely) -> is it still better?
m = keep & (c10 >= 1) & (c10 <= 63) & (c100 <= 63) & (rep == 0)
print(f"\nwords with c10 1-63 and c100 <= 63 (n={m.sum()}): nll10 {nll10[m].mean():.2f} nll100 {nll100[m].mean():.2f} gap {(nll10-nll100)[m].mean():.2f}; median c10 {np.median(c10[m]):.0f} c100 {np.median(c100[m]):.0f}")
for lo, hi in ((1, 3), (4, 15), (16, 63)):
    m = keep & (c10 >= 1) & (c10 <= 63) & (c100 >= lo) & (c100 <= hi) & (rep == 0)
    if m.sum():
        print(f"  c100 in {lo}-{hi}: n={m.sum()} nll10 {nll10[m].mean():.2f} nll100 {nll100[m].mean():.2f} gap {(nll10-nll100)[m].mean():.2f} (median c10 {np.median(c10[m]):.0f})")
# non-ascii never-seen words: tokens per word and first/cont split per model
m = keep & (cat == "non-ascii letters")
print(f"\nnon-ascii never-seen (n={m.sum()}): tok/word 10M {nt10[m].mean():.2f} vs 100M {nt100[m].mean():.2f}; nll10 first/cont {f10[m].mean():.2f}/{k10[m].mean():.2f}; nll100 first/cont {f100[m].mean():.2f}/{k100[m].mean():.2f}; per-token nll 10M {(nll10[m].sum()/nt10[m].sum()):.2f} 100M {(nll100[m].sum()/nt100[m].sum()):.2f}")
for k in ["capitalized novel word (name-like)", "lowercase novel word", "inflection of seen word", "derivation of seen word", "closed compound of seen words", "hyphen/slash compound of seen parts"]:
    m = keep & (cat == k)
    print(f"{k:40s} n={m.sum():5d} tok/w {nt10[m].mean():.2f}/{nt100[m].mean():.2f} nll10 f/c {f10[m].mean():5.2f}/{k10[m].mean():5.2f} nll100 f/c {f100[m].mean():5.2f}/{k100[m].mean():5.2f} per-tok {(nll10[m].sum()/nt10[m].sum()):.2f}/{(nll100[m].sum()/nt100[m].sum()):.2f}")
# 1-63 band: gap by number of pieces (10M tokenizer)
print("\n1-63 band first uses by 10M pieces per word: n, share, gap/word, gap first, gap cont, nll10 cont, nll100 cont")
m0 = keep & (c10 >= 1) & (c10 <= 63) & (rep == 0)
for p in (1, 2, 3, 4):
    m = m0 & ((nt10 == p) if p < 4 else (nt10 >= 4))
    print(f"  pieces {'>=4' if p == 4 else p}: n={m.sum():5d} share {m.sum()/m0.sum():.1%} gap {(nll10-nll100)[m].mean():5.2f} first {(f10-f100)[m].mean():5.2f} cont {(k10-k100)[m].mean():5.2f} nll10c {k10[m].mean():5.2f} nll100c {k100[m].mean():5.2f}")
