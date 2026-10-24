"""Decompose the GPT-BERT 10M vs 100M dev gap by rare-word type, first use vs repeat, first token vs spelling,
and rare words as context. Reads analysis/score_v1/{words.tsv,sample.jsonl,<model>.npz}; writes decomp.json/.md."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rare_words as RW  # noqa: E402

OUT = Path(sys.argv[1])
RAW = Path(sys.argv[2])
DROP = np.load(sys.argv[3]) if len(sys.argv) > 3 else None  # bool per word: drop (verbatim train_100M lines)
TAG = "_filtered" if DROP is not None else ""
M10 = "babylm-baseline-10m-gpt-bert-causal-focus"
M100 = "babylm-baseline-100m-gpt-bert-causal-focus"
BANDS = ["0", "1-63", "64-8k", "8k+"]
rng = np.random.default_rng(0)


def band(c):
    return np.where(c == 0, "0", np.where(c <= 63, "1-63", np.where(c < 8192, "64-8k", "8k+")))


# ---- load
rows = [json.loads(l) for l in (OUT / "sample.jsonl").read_text().splitlines()]
piece_len = np.array([len(r["text"].split()) for r in rows])
piece_of_word = np.repeat(np.arange(len(rows)), piece_len)
lines = (OUT / "words.tsv").read_text(encoding="utf-8").split("\n")
hdr = lines[0].split("\t")
recs = [l.split("\t") for l in lines[1:] if l]
assert len(recs) == piece_len.sum(), (len(recs), piece_len.sum())
col = {h: i for i, h in enumerate(hdr)}
source = np.array([r[col["source"]] for r in recs])
norm = np.array([r[col["norm"]] for r in recs])
c10 = np.array([int(r[col["c10"]]) for r in recs])
c100 = np.array([int(r[col["c100"]]) for r in recs])
cat = np.array([r[col["cat"]] for r in recs])
rep_piece = np.array([int(r[col["repeat_piece"]]) for r in recs])
rep300 = np.array([int(r[col["repeat300"]]) for r in recs])
N = len(recs)
z10 = np.load(OUT / f"{M10}.npz")
z100 = np.load(OUT / f"{M100}.npz")
nll10 = z10["word_nll"].astype(np.float64)
nll100 = z100["word_nll"].astype(np.float64)
assert len(nll10) == N == len(nll100)
if DROP is not None:
    keep = ~DROP
    assert len(keep) == N
    source, norm, c10, c100, cat, rep_piece, rep300 = (a[keep] for a in (source, norm, c10, c100, cat, rep_piece, rep300))
    nll10, nll100, piece_of_word = nll10[keep], nll100[keep], piece_of_word[keep]
    recs = [r for r, k in zip(recs, keep) if k]
    z10 = {"word_nll": z10["word_nll"], "tok_nll": z10["tok_nll"], "tok_word": z10["tok_word"], "keep": keep}
    z100 = {"word_nll": z100["word_nll"], "tok_nll": z100["tok_nll"], "tok_word": z100["tok_word"], "keep": keep}
    N = int(keep.sum())
gap = nll10 - nll100
b = band(c10)

# ---- source reweighting to the official dev split word shares
dev_tot = {}
for s in RW.SOURCES:
    t = 0
    for ln in (RAW / "dev" / f"{s}.dev").read_text(encoding="utf-8").split("\n"):
        if not RW.is_header(ln):
            t += len(ln.split())
    dev_tot[s] = t
T = sum(dev_tot.values())
samp_share = {s: float((source == s).mean()) for s in RW.SOURCES}
w = np.array([(dev_tot[s] / T) / samp_share[s] for s in source])
w = w / w.mean()


def stats(mask, weights=w):
    m = mask.astype(bool)
    ww = weights[m]
    n = ww.sum()
    return dict(share=float(n / weights.sum()), n=int(m.sum()), nll10=float((nll10[m] * ww).sum() / n) if n else None,
                nll100=float((nll100[m] * ww).sum() / n) if n else None,
                gap_per_word=float((gap[m] * ww).sum() / n) if n else None,
                gap_share=float((gap[m] * ww).sum() / (gap * weights).sum()))


def boot(mask, fn, B=600):
    """Piece bootstrap of fn(mask_idx) where fn returns a weighted mean of gap over mask."""
    pieces = np.arange(len(rows))
    vals = []
    for _ in range(B):
        samp = rng.choice(pieces, size=len(pieces), replace=True)
        cnt = np.bincount(samp, minlength=len(pieces))
        ww = w * cnt[piece_of_word]
        m = mask.astype(bool)
        if (ww[m]).sum() == 0:
            continue
        vals.append(fn(m, ww))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def gpw(m, ww):
    return (gap[m] * ww[m]).sum() / ww[m].sum()


def gshare(m, ww):
    return (gap[m] * ww[m]).sum() / (gap * ww).sum()


res = {"n_words": N, "n_pieces": len(rows), "sources": {s: dict(dev_share=dev_tot[s] / T, sample_share=samp_share[s]) for s in RW.SOURCES}}
res["total"] = dict(nll10=float((nll10 * w).mean()), nll100=float((nll100 * w).mean()), gap=float((gap * w).mean()),
                    gap_raw=float(gap.mean()), gap_ci=boot(np.ones(N, bool), gpw))
res["bands"] = {k: stats(b == k) for k in BANDS}
for k in ("0", "1-63"):
    res["bands"][k]["gap_per_word_ci"] = boot(b == k, gpw)
    res["bands"][k]["gap_share_ci"] = boot(b == k, gshare)

# never-seen by category and by 100M count
res["never_seen_by_cat"] = {}
m0 = b == "0"
for k in sorted(set(cat[m0])):
    m = m0 & (cat == k)
    d = stats(m)
    d["share_of_band"] = float(w[m].sum() / w[m0].sum())
    d["share_of_band_gap"] = float((gap[m] * w[m]).sum() / (gap[m0] * w[m0]).sum())
    res["never_seen_by_cat"][k] = d
c100b = np.where(c100 == 0, "0", np.where(c100 <= 9, "1-9", np.where(c100 <= 63, "10-63", "64+")))
res["never_seen_by_c100"] = {}
for k in ["0", "1-9", "10-63", "64+"]:
    m = m0 & (c100b == k)
    d = stats(m)
    d["share_of_band"] = float(w[m].sum() / w[m0].sum())
    d["share_of_band_gap"] = float((gap[m] * w[m]).sum() / (gap[m0] * w[m0]).sum())
    res["never_seen_by_c100"][k] = d
# decomposable vs not
decomp = np.isin(cat, ["inflection of seen word", "derivation of seen word", "two-affix derivation of seen word",
                       "closed compound of seen words", "hyphen/slash compound of seen parts"])
res["never_seen_decomposable"] = {"yes": stats(m0 & decomp), "no": stats(m0 & ~decomp)}
for k, m in (("yes", m0 & decomp), ("no", m0 & ~decomp)):
    res["never_seen_decomposable"][k]["share_of_band_gap"] = float((gap[m] * w[m]).sum() / (gap[m0] * w[m0]).sum())

# 1-63 by sub-band and by 100M count
m1 = b == "1-63"
sub = np.where(c10 == 1, "1", np.where(c10 <= 3, "2-3", np.where(c10 <= 7, "4-7", np.where(c10 <= 15, "8-15", np.where(c10 <= 31, "16-31", "32-63")))))
res["rare_by_c10"] = {}
for k in ["1", "2-3", "4-7", "8-15", "16-31", "32-63"]:
    m = m1 & (sub == k)
    d = stats(m)
    d["share_of_band"] = float(w[m].sum() / w[m1].sum())
    d["share_of_band_gap"] = float((gap[m] * w[m]).sum() / (gap[m1] * w[m1]).sum())
    d["median_c100_over_c10"] = float(np.median(c100[m] / np.maximum(c10[m], 1)))
    res["rare_by_c10"][k] = d
ratio = c100 / np.maximum(c10, 1)
rb = np.where(ratio < 3, "<3x", np.where(ratio < 7, "3-7x", np.where(ratio < 15, "7-15x", "15x+")))
res["rare_by_c100_ratio"] = {}
for k in ["<3x", "3-7x", "7-15x", "15x+"]:
    m = m1 & (rb == k)
    d = stats(m)
    d["share_of_band"] = float(w[m].sum() / w[m1].sum())
    d["share_of_band_gap"] = float((gap[m] * w[m]).sum() / (gap[m1] * w[m1]).sum())
    res["rare_by_c100_ratio"][k] = d

# first use vs repeat, per band
res["first_vs_repeat"] = {}
for k in BANDS:
    mb = b == k
    out = {}
    for name, flag in (("repeat300", rep300), ("repeat_piece", rep_piece)):
        for v, lab in ((0, "first"), (1, "repeat")):
            m = mb & (flag == v)
            d = stats(m)
            d["share_of_band"] = float(w[m].sum() / w[mb].sum())
            d["share_of_band_gap"] = float((gap[m] * w[m]).sum() / (gap[mb] * w[mb]).sum())
            if k in ("0", "1-63") and name == "repeat300":
                d["gap_per_word_ci"] = boot(m, gpw)
            out[f"{name}:{lab}"] = d
    res["first_vs_repeat"][k] = out

# capitalised vs lowercase in 1-63 band
surf = np.array([r[col["surface"]] for r in recs])
capm = np.array([s.strip(RW.PUNCT)[:1].isupper() for s in surf])
res["rare_by_case"] = {"cap": stats(m1 & capm), "lower": stats(m1 & ~capm)}
for k, m in (("cap", m1 & capm), ("lower", m1 & ~capm)):
    res["rare_by_case"][k]["share_of_band_gap"] = float((gap[m] * w[m]).sum() / (gap[m1] * w[m1]).sum())

# first token vs continuation tokens, per model (own tokenizer)
def split_tokens(z):
    tw = z["tok_word"].astype(np.int64)
    tn = z["tok_nll"].astype(np.float64)
    first = np.ones(len(tw), bool)
    first[1:] = tw[1:] != tw[:-1]
    n_all = len(z["word_nll"])
    nf = np.bincount(tw[first], weights=tn[first], minlength=n_all)
    nc = np.bincount(tw[~first], weights=tn[~first], minlength=n_all)
    ntok = np.bincount(tw, minlength=n_all)
    if isinstance(z, dict) and "keep" in z:
        nf, nc, ntok = nf[z["keep"]], nc[z["keep"]], ntok[z["keep"]]
    return nf, nc, ntok


f10, c10n, nt10 = split_tokens(z10)
f100, c100n, nt100 = split_tokens(z100)
res["first_token_vs_spelling"] = {}
for k in BANDS:
    m = b == k
    ww = w[m]
    n = ww.sum()
    res["first_token_vs_spelling"][k] = dict(
        tok_per_word_10m=float((nt10[m] * ww).sum() / n), tok_per_word_100m=float((nt100[m] * ww).sum() / n),
        nll10_first=float((f10[m] * ww).sum() / n), nll10_cont=float((c10n[m] * ww).sum() / n),
        nll100_first=float((f100[m] * ww).sum() / n), nll100_cont=float((c100n[m] * ww).sum() / n),
        gap_first=float(((f10 - f100)[m] * ww).sum() / n), gap_cont=float(((c10n - c100n)[m] * ww).sum() / n),
        gap_first_share=float(((f10 - f100)[m] * ww).sum() / (gap[m] * ww).sum()))

# rare words as context: frequent targets by distance to the last rare (c10<=63) word in the same piece
def dist_to_last(flag):
    d = np.full(N, 10**6, dtype=np.int64)
    last = -10**9
    cur_piece = -1
    for i in range(N):
        if piece_of_word[i] != cur_piece:
            cur_piece = piece_of_word[i]
            last = -10**9
        d[i] = i - last
        if flag[i]:
            last = i
    return d


freq = c10 >= 64
res["context_side"] = {}
for name, flag in (("rare<=63", c10 <= 63), ("never_seen", c10 == 0)):
    d = dist_to_last(flag)
    out = {}
    for lab, lo, hi in (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-6", 4, 6), ("7-15", 7, 15), ("16-50", 16, 50), (">50/none", 51, 10**9)):
        m = freq & (d >= lo) & (d <= hi)
        s = stats(m)
        s["share_of_freq_words"] = float(w[m].sum() / w[freq].sum())
        out[lab] = s
    # excess gap on frequent words within 3 words after a rare word, vs frequent words >50 away
    near = freq & (d <= 3)
    far = freq & (d > 50)
    base = (gap[far] * w[far]).sum() / w[far].sum()
    excess = ((gap[near] - base) * w[near]).sum() / w.sum()
    out["excess_within3_nats_per_word_overall"] = float(excess)
    out["excess_within3_ci"] = boot(near, lambda m, ww: ((gap[m] - base) * ww[m]).sum() / ww.sum())
    res["context_side"][name] = out

# by source
res["by_source"] = {s: dict(**stats(source == s), rare_share=float(w[(source == s) & (c10 <= 63)].sum() / w[source == s].sum()),
                            rare_gap_share=float((gap[(source == s) & (c10 <= 63)] * w[(source == s) & (c10 <= 63)]).sum() / (gap * w).sum()))
                    for s in RW.SOURCES}
# top words by gap mass in 1-63 band (what are they)
mass = collections.Counter()
cnt = collections.Counter()
for i in np.where(m1)[0]:
    mass[norm[i]] += gap[i] * w[i]
    cnt[norm[i]] += 1
res["top_rare_words_by_gap_mass"] = [(k, round(v, 1), cnt[k], int(c10[norm == k][0]), int(c100[norm == k][0])) for k, v in mass.most_common(40)]
(OUT / f"decomp{TAG}.json").write_text(json.dumps(res, indent=1))

# ---- markdown summary
L = []
t = res["total"]
L.append(f"# GPT-BERT 10M vs 100M, dev sample {N:,} words / {len(rows)} pieces (reweighted to dev source shares){' | words in verbatim train_100M lines dropped' if DROP is not None else ''}\n")
L.append(f"nats/word: 10M {t['nll10']:.3f}, 100M {t['nll100']:.3f}, gap {t['gap']:.3f} (CI {t['gap_ci'][0]:.3f}-{t['gap_ci'][1]:.3f}); raw gap {t['gap_raw']:.3f}\n")
L.append("## Bands (train_10M count)\n| band | word share | gap/word | gap share | gap nats/word overall |\n|---|---|---|---|---|")
for k in BANDS:
    d = res["bands"][k]
    L.append(f"| {k} | {d['share']:.3%} | {d['gap_per_word']:.2f} | {d['gap_share']:.1%} | {d['gap_share'] * t['gap']:.3f} |")
L.append("\n## Never-seen words by type (share of band words / share of band gap / gap per word)\n| type | share | gap share | gap/word | nll10 | nll100 |\n|---|---|---|---|---|---|")
for k, d in sorted(res["never_seen_by_cat"].items(), key=lambda kv: -kv[1]["share_of_band_gap"]):
    L.append(f"| {k} | {d['share_of_band']:.1%} | {d['share_of_band_gap']:.1%} | {d['gap_per_word']:.2f} | {d['nll10']:.2f} | {d['nll100']:.2f} |")
L.append("\n## Never-seen words by train_100M count\n| c100 | share | gap share | gap/word | nll10 | nll100 |\n|---|---|---|---|---|---|")
for k, d in res["never_seen_by_c100"].items():
    L.append(f"| {k} | {d['share_of_band']:.1%} | {d['share_of_band_gap']:.1%} | {d['gap_per_word']:.2f} | {d['nll10']:.2f} | {d['nll100']:.2f} |")
L.append("\n## 1-63 band by train_10M count\n| c10 | share | gap share | gap/word | nll10 | nll100 | median c100/c10 |\n|---|---|---|---|---|---|---|")
for k, d in res["rare_by_c10"].items():
    L.append(f"| {k} | {d['share_of_band']:.1%} | {d['share_of_band_gap']:.1%} | {d['gap_per_word']:.2f} | {d['nll10']:.2f} | {d['nll100']:.2f} | {d['median_c100_over_c10']:.1f} |")
L.append("\n## 1-63 band by c100/c10 ratio\n| ratio | share | gap share | gap/word |\n|---|---|---|---|")
for k, d in res["rare_by_c100_ratio"].items():
    L.append(f"| {k} | {d['share_of_band']:.1%} | {d['share_of_band_gap']:.1%} | {d['gap_per_word']:.2f} |")
L.append("\n## First use vs repeat (repeat = same normal form within the previous 300 words of the piece)\n| band | split | share of band | gap share of band | gap/word | nll10 | nll100 |\n|---|---|---|---|---|---|---|")
for k in BANDS:
    for lab in ("repeat300:first", "repeat300:repeat"):
        d = res["first_vs_repeat"][k][lab]
        ci = d.get("gap_per_word_ci")
        L.append(f"| {k} | {lab.split(':')[1]} | {d['share_of_band']:.1%} | {d['share_of_band_gap']:.1%} | {d['gap_per_word']:.2f}{'' if not ci else f' ({ci[0]:.2f}-{ci[1]:.2f})'} | {d['nll10']:.2f} | {d['nll100']:.2f} |")
L.append("\n## First token vs spelling (continuation tokens), each model with its own tokenizer\n| band | tok/word 10M | tok/word 100M | gap first | gap cont | first share | nll10 first/cont | nll100 first/cont |\n|---|---|---|---|---|---|---|---|")
for k in BANDS:
    d = res["first_token_vs_spelling"][k]
    L.append(f"| {k} | {d['tok_per_word_10m']:.2f} | {d['tok_per_word_100m']:.2f} | {d['gap_first']:.2f} | {d['gap_cont']:.2f} | {d['gap_first_share']:.0%} | {d['nll10_first']:.2f}/{d['nll10_cont']:.2f} | {d['nll100_first']:.2f}/{d['nll100_cont']:.2f} |")
L.append("\n## Rare words as context: frequent (64+) targets by distance to the last rare word\n| rare def | distance | share of freq words | gap/word |\n|---|---|---|---|")
for name, out in res["context_side"].items():
    for lab, d in out.items():
        if isinstance(d, dict):
            L.append(f"| {name} | {lab} | {d['share_of_freq_words']:.1%} | {d['gap_per_word']:.2f} |")
    L.append(f"| {name} | excess within 3 words, nats/word overall | {out['excess_within3_nats_per_word_overall']:.3f} | CI {out['excess_within3_ci'][0]:.3f}-{out['excess_within3_ci'][1]:.3f} |")
L.append("\n## 1-63 band by case\n| case | share of band | gap share of band | gap/word |\n|---|---|---|---|")
for k, d in res["rare_by_case"].items():
    L.append(f"| {k} | {d['share'] / res['bands']['1-63']['share']:.1%} | {d['share_of_band_gap']:.1%} | {d['gap_per_word']:.2f} |")
L.append("\n## By source\n| source | dev share | gap/word | gap share | rare word share | rare gap share (of total) |\n|---|---|---|---|---|---|")
for s, d in res["by_source"].items():
    L.append(f"| {s} | {res['sources'][s]['dev_share']:.1%} | {d['gap_per_word']:.2f} | {d['gap_share']:.1%} | {d['rare_share']:.1%} | {d['rare_gap_share']:.1%} |")
L.append("\n## Top 1-63 words by gap mass (word, mass, n in sample, c10, c100)\n" + ", ".join(f"{k}({v}, n{n}, {a}/{bb})" for k, v, n, a, bb in res["top_rare_words_by_gap_mass"]))
(OUT / f"decomp{TAG}.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
