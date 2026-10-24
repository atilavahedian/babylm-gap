"""One-page anatomy of the 10M-to-100M gap ("the prize") from the pre-read tables (CSV + PNG).

For each 10M/100M pair of finals: where the gap lives (word frequency), whether it shows on
text both models trained on (train sample) or only on new text (that part split into words seen
under 64 and 64+ times in train_10M), how much a count model of
the 100M words can supply (an oracle for local word statistics), and what cheap, legal 10M
add-ons recover (count model of the 10M words, copy cache, temperature, a 2-model ensemble).
Run after slices.py, anatomy.py and ngram.py. With --data-root the total is also reweighted to the
official dev split's source word shares (critic ref-0255: the sample under-weights simple_wiki), and
the gap without speaker labels (*CHI:, A:) is reported; after overlap.py, also the gap with lines found
verbatim in train_100M removed (critic ref-0260); and the gap without words carrying CHILDES conversation-analysis
markup, which train_10M never contains (slices.CHAT_MARKUP; experimenter, 2026-10-10).

Usage:
  python -m babylm.instruments.prize --tag preread_v1 --data-root ~/babylm_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from babylm.instruments import corpus as C
from babylm.instruments import slices as S

PAIRS = {"GPT-BERT causal": ("gptbert_cf_10m/chck_100M", "gptbert_cf_100m/chck_1000M"),
         "GPT-2": ("gpt2_10m/chck_100M", "gpt2_100m/chck_1000M")}
# Headline text format per family where it differs from the first-scored runs (preread_formats.sh): GPT-BERT with
# whitespace collapse (its reference code; the probe agrees), GPT-2 with one tab rule for all sources (critic ref-0259).
# Used when that pair is in summary.json; slices, epochs and add-ons stay on the first-scored runs, which differ only
# on words after leading or repeated spaces (GPT-BERT, 0.4% of dev words) or on Switchboard (GPT-2, 1.5%).
HEADLINE = {"GPT-BERT causal": ("gptbert_cf_10m_collapse/chck_100M", "gptbert_cf_100m_collapse/chck_1000M"),
            "GPT-2": ("gpt2_10m_tabspace/chck_100M", "gpt2_100m_tabspace/chck_1000M")}
FORMAT_ROWS = {"GPT-BERT causal": [("whitespace collapsed (reference code; headline)", HEADLINE["GPT-BERT causal"]),
                                   ("whitespace as in the files (first scored)", PAIRS["GPT-BERT causal"])],
               "GPT-2": [("all speaker-label tabs to spaces (headline)", HEADLINE["GPT-2"]),
                         ("per-source tabs (chosen after scoring; slice tables)", PAIRS["GPT-2"]),
                         ("all tabs kept (code-literal)", ("gpt2_10m_tabkeep/chck_100M", "gpt2_100m_tabkeep/chck_1000M"))]}
GROUPS = [("never seen in 10M", ["0"]), ("seen 1-63 times", ["1", "2-3", "4-7", "8-15", "16-63"]),
          ("seen 64-8k times", ["64-255", "256-1023", "1k-8k"]), ("seen 8k+ times", ["8k+"])]


def dev_source_shares(data_root: Path) -> np.ndarray:
    n = []
    for path in C.split_files(data_root, "dev"):
        with open(path, encoding="utf-8") as f:
            n.append(sum(len(line.split()) for line in f if not C.is_header(line)))
    return np.asarray(n, float) / sum(n)


def strat_weights(stratum: np.ndarray, reps: int, seed: int = 0) -> np.ndarray:
    """Bootstrap counts per document (reps x documents), resampling each source's documents among themselves, so every
    source keeps its documents in every replicate (DEV-sub has one Switchboard document)."""
    rng = np.random.default_rng(seed)
    W = np.zeros((reps, len(stratum)))
    for st in np.unique(stratum):
        idx = np.flatnonzero(stratum == st)
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=reps)
    return W


def reweighted(d: np.ndarray, src: np.ndarray, chunk: np.ndarray, share: np.ndarray, reps: int = 1000) -> tuple[float, float, float]:
    """Sum over sources of dev share x mean gap in that source, with a document bootstrap stratified by source."""
    nc = int(chunk.max()) + 1
    csrc = np.zeros(nc, int)
    csrc[chunk] = src
    Sd, Sn = np.bincount(chunk, weights=d, minlength=nc), np.bincount(chunk, minlength=nc).astype(float)
    W = strat_weights(csrc, reps)
    point, boot = 0.0, np.zeros(reps)
    for i, w in enumerate(share):
        m = csrc == i
        point += w * Sd[m].sum() / Sn[m].sum()
        boot += w * (W[:, m] @ Sd[m]) / (W[:, m] @ Sn[m])
    return float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def group_reps(d: np.ndarray, grp: np.ndarray, src: np.ndarray, chunk: np.ndarray, share: np.ndarray | None,
               keep: np.ndarray, K: int, reps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap replicates (row 0: the point estimate) of each group's contribution (sum of d over its words / all kept
    words) and word share, every source weighted to its share (share=None: the sample as it is). grp -1: in no group, but
    counted in the denominator. Document bootstrap: Poisson for the sample as it is, stratified by source when
    reweighted (the same replicate weights on every call, so ratios across calls are paired)."""
    nc = int(chunk.max()) + 1
    if share is None:
        src, share = np.zeros_like(src), np.ones(1)
    csrc = np.zeros(nc, int)
    csrc[chunk] = src
    ok = keep & (grp >= 0)
    Sd, Nk = np.zeros((nc, K)), np.zeros((nc, K))
    np.add.at(Sd, (chunk[ok], grp[ok]), d[ok])
    np.add.at(Nk, (chunk[ok], grp[ok]), 1)
    N = np.bincount(chunk[keep], minlength=nc).astype(float)
    W = np.vstack([np.ones(nc), np.random.default_rng(0).poisson(1.0, size=(reps, nc)) if len(share) == 1 else strat_weights(csrc, reps)])
    contrib, words = np.zeros((reps + 1, K)), np.zeros((reps + 1, K))
    for i, w in enumerate(share):
        m = csrc == i
        n = (W[:, m] @ N[m])[:, None]
        contrib += w * (W[:, m] @ Sd[m]) / n
        words += w * (W[:, m] @ Nk[m]) / n
    return contrib, words


def ci(v: np.ndarray) -> np.ndarray:
    """[point, lo, hi] along the last axis from replicates (row 0: the point estimate)."""
    return np.stack([v[0], np.nanpercentile(v[1:], 2.5, axis=0), np.nanpercentile(v[1:], 97.5, axis=0)], -1)


def band_shares(d: np.ndarray, grp: np.ndarray, src: np.ndarray, chunk: np.ndarray, share: np.ndarray | None,
                keep: np.ndarray, reps: int = 1000) -> dict[str, np.ndarray]:
    """Per frequency group: word share, gap share, gap per word and contribution, each [point, lo, hi], over the kept
    words, with every source reweighted to its share (share=None: the sample as it is); see group_reps."""
    contrib, words = group_reps(d, grp, src, chunk, share, keep, len(GROUPS), reps)
    out = {"word_share": words, "gap_share": contrib / contrib.sum(1, keepdims=True), "gap_per_word": contrib / words,
           "contrib": contrib}
    return {k: ci(v) for k, v in out.items()}


def new_text_parts(d: np.ndarray, grp: np.ndarray, unit: np.ndarray, dt: np.ndarray, grpt: np.ndarray, unitt: np.ndarray,
                   groups: list[list[int]], reps: int = 1000) -> np.ndarray:
    """Part of the dev gap that does not show on text both models trained on, per group of frequency bands: the sum over
    the group's bands of dev word share x (gap per dev word - gap per train-sample word). grp, grpt: band per dev / train
    word (-1: none). [point, lo, hi] per group; dev and train documents are resampled independently (Poisson)."""
    K = len(GROUPS)

    def sums(x, g, u, seed):
        nc = int(u.max()) + 1
        Sx, Nx = np.zeros((nc, K)), np.zeros((nc, K))
        ok = g >= 0
        np.add.at(Sx, (u[ok], g[ok]), x[ok])
        np.add.at(Nx, (u[ok], g[ok]), 1)
        W = np.vstack([np.ones(nc), np.random.default_rng(seed).poisson(1.0, size=(reps, nc))])
        return W @ Sx, W @ Nx, W @ np.bincount(u, minlength=nc).astype(float)

    Sd, Nd, Td = sums(d, grp, unit, 0)
    St, Nt, _ = sums(dt, grpt, unitt, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        band = (Nd / Td[:, None]) * (Sd / Nd - St / Nt)
    part = np.stack([np.nansum(band[:, g], 1) for g in groups], 1)
    return np.stack([part[0], np.percentile(part[1:], 2.5, axis=0), np.percentile(part[1:], 97.5, axis=0)], 1)


def freq_groups(F: dict, train_like: bool = False) -> np.ndarray:
    grp = np.full(len(F["freq10"]), -1)
    f10b = S.dims(F, train_like)["freq10"][0]
    for gi, (_, labels) in enumerate(GROUPS):
        grp[np.isin(f10b, [S.FREQ_LABELS.index(x) for x in labels])] = gi
    return grp


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--pair", action="append", default=[], help="name=10M_run/rev:100M_run/rev, replacing the published pairs")
    args = ap.parse_args()
    pairs = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split(":")) for x in args.pair} or PAIRS
    F = S.load_feats("dev")
    share = dev_source_shares(args.data_root) if args.data_root else None
    out = S.REPO / "results" / "babylm" / args.tag
    summ = json.loads((out / "summary.json").read_text())["gaps"]
    read = lambda name: pd.read_csv(out / name) if (out / name).exists() else None
    ngram, cache, temps, mix = read("ngram_mix.csv"), read("cache.csv"), read("temperature.csv"), read("mixtures.csv")
    rows, band_rows = [], []
    pkey = lambda a, b: f"{a.replace('/', '-')}__{b.replace('/', '-')}"
    for fam, (a0, b0) in pairs.items():
        a, b = HEADLINE[fam] if fam in HEADLINE and pkey(*HEADLINE[fam]) in summ else (a0, b0)
        key = pkey(a, b)
        if key not in summ:
            continue
        g = summ[key]
        add = lambda part, what, val, lo=np.nan, hi=np.nan: rows.append(
            {"family": fam, "panel": part, "item": what, "nats": val, "lo": lo, "hi": hi})
        add("total", "10M final minus 100M final (dev)", g["total_nats"], *g["total_ci95"])
        for what, (fa, fb) in FORMAT_ROWS.get(fam, []):
            if pkey(fa, fb) in summ:
                add("format", what, summ[pkey(fa, fb)]["total_nats"], *summ[pkey(fa, fb)]["total_ci95"])
        sa, sb = S.load_score(*a.split("/"), "dev"), S.load_score(*b.split("/"), "dev")
        d = sa["word_nll"].astype(np.float64) - sb["word_nll"].astype(np.float64)
        if share is not None:
            add("required labels", "total reweighted to dev source shares", *reweighted(d, F["source"].astype(int), F["unit"], share))
        tag = F["speaker_tag"]
        ci = S.bootstrap_gap(np.where(tag, 0.0, d), F["unit"], np.where(tag, -1, 0), 1)[2:4]
        add("required labels", "per word, speaker labels left out", float(d[~tag].mean()), float(ci[0][0]), float(ci[1][0]))
        add("required labels", "carried by speaker labels (*CHI:, A:)", float(d[tag].sum() / len(d)))
        if "in_train_100M" in F:
            o = F["in_train_100M"]
            ci = S.bootstrap_gap(np.where(o, 0.0, d), F["unit"], np.where(o, -1, 0), 1)[2:4]
            add("required labels", f"lines verbatim in train_100M removed ({o.mean():.2%} of words)", float(d[~o].mean()),
                float(ci[0][0]), float(ci[1][0]))
        mk = F["markup"]
        if mk.any():  # CHILDES conversation-analysis markup: none in train_10M (S.CHAT_MARKUP)
            ci = S.bootstrap_gap(np.where(mk, 0.0, d), F["unit"], np.where(mk, -1, 0), 1)[2:4]
            add("required labels", f"CHILDES markup words left out ({mk.mean():.2%} of words, none in train_10M)",
                float(d[~mk].mean()), float(ci[0][0]), float(ci[1][0]))
            add("required labels", "carried by CHILDES markup words", float(d[mk].sum() / len(d)))
            if share is not None and "in_train_100M" in F:
                k = ~F["in_train_100M"]
                add("required labels", "reweighted, overlap removed",
                    *reweighted(d[k], F["source"].astype(int)[k], F["unit"][k], share))
                k = ~F["in_train_100M"] & ~mk
                add("required labels", "reweighted, overlap and CHILDES markup removed",
                    *reweighted(d[k], F["source"].astype(int)[k], F["unit"][k], share))
        gt = summ.get(key + "_train")
        if gt:
            na, nb = (S.words_seen(x.split("/")[1]) / w for x, w in ((a, 1e7), (b, 1e8)))
            seen = "both saw 10 times" if na == nb == 10 else f"seen {na:g} times by the 10M model, {nb:g} by the 100M"
            add("seen vs new text", f"on train text {seen}", gt["total_nats"], *gt["total_ci95"])
            add("seen vs new text", "extra on new text only", g["total_nats"] - gt["total_nats"])
            ta, tb = S.load_score(*a.split("/"), "train"), S.load_score(*b.split("/"), "train")
            if ta is not None and tb is not None:
                FT = S.load_feats("train")
                dt = ta["word_nll"].astype(np.float64) - tb["word_nll"].astype(np.float64)
                parts = new_text_parts(d, freq_groups(F), F["unit"], dt, freq_groups(FT, True), FT["unit"], [[0, 1], [2, 3]])
                for what, p in zip(("  of which words seen <64 times", "  of which words seen 64+ times"), parts):
                    add("seen vs new text", what, *p)
        gap = pd.read_csv(out / f"gap_{key}.csv")
        f = gap[gap.dim == "freq10"]
        for name, labels in GROUPS:
            add("by word frequency in train_10M", name, float(f[f.label.isin(labels)].contrib.sum()))
        if ngram is not None:
            for corpus, what in (("train_100M", "count model of the 100M words (oracle, not legal)"),
                                 ("train_10M", "count model of the same 10M words")):
                r = ngram[(ngram.model == a0) & (ngram.count_model == corpus)]
                if len(r):
                    part = "local statistics" if corpus == "train_100M" else "cheap legal 10M add-ons"
                    add(part, what, float(r.gain.iloc[0]), float(r.gain_lo.iloc[0]), float(r.gain_hi.iloc[0]))
        if cache is not None:
            r = cache[cache.model == a0]
            if len(r):
                add("cheap legal 10M add-ons", "copy cache of earlier words", float(r.gain.iloc[0]), float(r.gain_lo.iloc[0]), float(r.gain_hi.iloc[0]))
        if temps is not None:
            r = temps[temps.model == a0]
            if len(r):
                add("cheap legal 10M add-ons", "one softmax temperature", float(r.global_gain.iloc[0]))
        if mix is not None and fam.startswith("GPT-BERT"):
            r = mix[mix.mixture == "gpt2_10m/chck_100M+gptbert_cf_10m/chck_100M"]
            if len(r):
                add("cheap legal 10M add-ons", "ensemble with GPT-2 10M", float(r.gain_vs_best.iloc[0]), float(r.gain_lo.iloc[0]), float(r.gain_hi.iloc[0]))
        # where the prize sits by train_10M frequency: the sample as it is, and with train_100M overlap removed and
        # sources reweighted to the dev split (critics ref-0260, ref-0255)
        grp = freq_groups(F)
        variants = {"dev sample as it is": (None, np.ones(len(d), bool))}
        if share is not None and "in_train_100M" in F:
            variants["overlap removed, reweighted to dev sources"] = (share, ~F["in_train_100M"])
            if F["markup"].any():
                variants["overlap and CHILDES markup removed, reweighted"] = (share, ~F["in_train_100M"] & ~F["markup"])
        for vname, (sh, keep) in variants.items():
            bs = band_shares(d, grp, F["source"].astype(int), F["unit"], sh, keep)
            for gi, (gname, _) in enumerate(GROUPS):
                band_rows.append({"family": fam, "variant": vname, "band": gname,
                                  **{f"{k}{suf}": float(v[gi, j]) for k, v in bs.items() for j, suf in enumerate(("", "_lo", "_hi"))}})
    t = pd.DataFrame(rows)
    t.to_csv(out / "prize_anatomy.csv", index=False)
    if band_rows:
        pd.DataFrame(band_rows).to_csv(out / "prize_bands.csv", index=False)
    fams = [f for f in pairs if f in set(t.family)]
    panels = ["seen vs new text", "by word frequency in train_10M", "local statistics", "cheap legal 10M add-ons"]
    fig, axes = plt.subplots(len(panels), len(fams), figsize=(7.6 * len(fams), 10), facecolor=S.SURFACE, squeeze=False)
    for j, fam in enumerate(fams):
        tf = t[t.family == fam]
        tot = tf[tf.panel == "total"].iloc[0]
        shown = tf[tf.panel.isin(panels)][["nats", "lo", "hi"]].to_numpy()
        # room for negative rows (a 10M model ahead on its own train text) and rows past the total, plus value labels
        x0, x1 = min(0.0, np.nanmin(shown) * 1.1), max(tot.nats * 1.05, np.nanmax(shown) * 1.3)
        for i, part in enumerate(panels):
            ax = axes[i, j]
            tp = tf[tf.panel == part]
            req = tf[tf.panel == "required labels"].set_index("item").nats
            short = {"total reweighted": "reweighted to dev sources", "per word, speaker": "without speaker labels",
                     "lines verbatim": "train_100M overlap removed", "CHILDES markup words left out": "without CHILDES markup"}
            notes = [f"{v:.2f} {name}" for k, v in req.items() for p, name in short.items() if k.startswith(p)]
            rws = "".join("\n" + "; ".join(notes[k:k + 2]) for k in range(0, len(notes), 2))
            head = f"{fam}{': ' if len(fam) < 30 else chr(10)}gap {tot.nats:.2f} nats/word (95% CI {tot.lo:.2f} to {tot.hi:.2f})"
            title = f"{head}{rws}\n{part}" if i == 0 else part
            S.setup_axes(ax, title, "nats / word" if i == len(panels) - 1 else "", "")
            ax.grid(True, axis="x", color=S.GRID, linewidth=0.6)
            ax.grid(False, axis="y")
            y = np.arange(len(tp))[::-1]
            ax.barh(y, tp.nats, color=S.CAT[0] if part != "cheap legal 10M add-ons" else S.CAT[2], height=0.6)
            err = tp.dropna(subset=["lo"])
            if len(err):
                ye = y[[list(tp.index).index(k) for k in err.index]]
                ax.errorbar(err.nats, ye, xerr=[err.nats - err.lo, err.hi - err.nats], fmt="none", ecolor=S.INK2, lw=1)
            for yy, (_, r) in zip(y, tp.iterrows()):
                x_end = max(r.nats, 0, r.hi if np.isfinite(r.hi) else 0)
                ax.text(x_end + 0.012 * (x1 - x0), yy, f"{r.nats:.3f}  ({r.nats / tot.nats:.0%} of gap)", va="center", fontsize=7.5, color=S.INK)
            ax.set_yticks(y, tp["item"], fontsize=8)
            ax.set_xlim(x0, x1)
    what = "BabyLM 2025 baselines, dev sample" if pairs is PAIRS else f"{args.tag}, {'DEV-sub' if C.SAMPLE == 'harness' else 'dev sample'}"
    fig.suptitle(f"Anatomy of the 10M-to-100M gap ({what}, nats per word)", color=S.INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "prize_anatomy.png", dpi=150)
    plt.close(fig)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
