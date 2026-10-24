"""Epoch-vs-data and exposure analyses for one 10M/100M model pair (CSV + PNG).

For a 10M-track run (checkpoints chck_10M..chck_100M = epochs 1..10) and the
same recipe on the 100M track, per slice bin:
  epoch_gain_k   dev loss drop from epoch k-1 to k of the 10M run
  late_gain      drop over the last three epochs (7 -> 10)
  data_gain      10M-run final minus 100M-run final (the prize)
  fresh_gain     10M run at 100M words minus 100M run at 100M words (same
                 exposure; repeated vs fresh text; the 100M run's learning
                 rate has not decayed yet, so this understates fresh data)
and the same numbers as contributions (bin mean x word share) that sum to the
totals. exposure.csv puts both runs on one axis: how many times the word's
normal form has been seen (10M run: epochs x train_10M count; 100M run in its
first epoch: train_100M count x words seen / 97.97M).

Usage:
  python -m babylm.instruments.bottleneck --tag preread_v1 --ten gpt2_10m --hundred gpt2_100m
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from babylm.instruments import slices as S

TRAIN100_WORDS = 97_968_519


def effective_words(ws: np.ndarray, losses: np.ndarray, target: float) -> tuple[float, str]:
    """Fresh words the 100M run needed to reach `target` loss (log-linear interpolation of
    its running-minimum curve). Flags '<' or '>' when the target lies outside the scored range."""
    o = np.argsort(ws)
    w, l = np.asarray(ws, float)[o], np.minimum.accumulate(np.asarray(losses, float)[o])
    if target >= l[0]:
        return float(w[0]), "<=" if target > l[0] else "="
    if target <= l[-1]:
        return float(w[-1]), ">=" if target < l[-1] else "="
    j = int(np.flatnonzero(l <= target)[0])
    lw0, lw1 = np.log(w[j - 1]), np.log(w[j])
    f = (l[j - 1] - target) / max(l[j - 1] - l[j], 1e-12)
    return float(np.exp(lw0 + f * (lw1 - lw0))), "="


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--ten", required=True)
    ap.add_argument("--hundred", required=True)
    ap.add_argument("--no-equal-exposure", action="store_true",
                    help="skip analyses that compare the two runs at equal words seen (use when the runs differ in "
                         "model size, batch size or sequence-length schedule, e.g. the GPT-BERT baselines)")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    F = S.load_feats("dev")
    D = S.dims(F)
    chunk = F["unit"]
    ten_revs = [r for r in S.scored(args.ten) if 10e6 <= S.words_seen(r) <= 100e6 and S.words_seen(r) % 10e6 == 0]
    ep = {int(S.words_seen(r) // 10e6): S.load_score(args.ten, r, "dev")["word_nll"].astype(np.float64) for r in ten_revs}
    final10 = ep.get(10)
    h_iso = S.load_score(args.hundred, "chck_100M", "dev")
    h_fin = S.load_score(args.hundred, "chck_1000M", "dev")
    if final10 is None or h_fin is None:
        raise SystemExit("need the 10M run at chck_100M and the 100M run at chck_1000M")
    h_fin = h_fin["word_nll"].astype(np.float64)
    h_iso = h_iso["word_nll"].astype(np.float64) if h_iso is not None and not args.no_equal_exposure else None
    N = len(final10)
    summary = {"pair": [args.ten, args.hundred], "epochs_scored": sorted(ep), "totals": {}}

    def tot(x):
        return float(x.mean())

    summary["totals"] = {
        "final10_nats": tot(final10), "final100_nats": tot(h_fin), "data_gain": tot(final10 - h_fin),
        "fresh_gain": tot(final10 - h_iso) if h_iso is not None else None,
        "epoch1_to_10_gain": tot(ep[1] - final10) if 1 in ep else None,
        "late_gain_7_to_10": tot(ep[7] - final10) if 7 in ep else None,
    }
    for name, d in (("data_gain", final10 - h_fin), ("fresh_gain", None if h_iso is None else final10 - h_iso),
                    ("late_gain_7_to_10", None if 7 not in ep else ep[7] - final10)):
        if d is not None:
            ci = S.bootstrap_gap(d, chunk, np.zeros(N, dtype=int), 1)[4]
            summary["totals"][name + "_ci95"] = [float(ci[0]), float(ci[1])]

    rows = []
    for dim, (b, labels) in D.items():
        for k, lab in enumerate(labels):
            m = b == k
            n = int(m.sum())
            if n == 0:
                continue
            r = {"dim": dim, "bin": k, "label": lab, "n": n, "share": n / N}
            for e in sorted(ep):
                r[f"dev_ep{e}"] = float(ep[e][m].mean())
            for e in sorted(ep):
                if e - 1 in ep:
                    r[f"epoch_gain_{e}"] = float((ep[e - 1][m] - ep[e][m]).mean())
            if 7 in ep:
                r["late_gain"] = float((ep[7][m] - final10[m]).mean())
                r["late_contrib"] = r["late_gain"] * n / N
            r["final100"] = float(h_fin[m].mean())
            r["data_gain"] = float((final10[m] - h_fin[m]).mean())
            r["data_contrib"] = r["data_gain"] * n / N
            if h_iso is not None:
                r["iso100"] = float(h_iso[m].mean())
                r["fresh_gain"] = float((final10[m] - h_iso[m]).mean())
                r["fresh_contrib"] = r["fresh_gain"] * n / N
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / f"epoch_vs_data_{args.ten}__{args.hundred}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=S.SURFACE)
    for ax, dim in zip(axes, ("freq10", "bigram_cov", "position")):
        tt = t[t.dim == dim]
        x = np.arange(len(tt))
        S.setup_axes(ax, f"{dim}: contribution to total nats/word", "", "nats / word")
        w = 0.27
        ax.bar(x - w, tt.data_contrib, w, color=S.CAT[0], label="10x data (final vs final)")
        if "fresh_contrib" in tt:
            ax.bar(x, tt.fresh_contrib, w, color=S.CAT[1], label="fresh vs repeated, same 100M words")
        if "late_contrib" in tt:
            ax.bar(x + w, tt.late_contrib, w, color=S.CAT[2], label="epochs 7 to 10 of the 10M run")
        ax.set_xticks(x, tt.label, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color=S.INK2, lw=0.8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{args.ten} vs {args.hundred}: where 10x data, fresh text and late epochs pay off", color=S.INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"epoch_vs_data_{args.ten}__{args.hundred}.png", dpi=150)
    plt.close(fig)

    # per-epoch dev curves by frequency band for the 10M run, with the 100M run's final as reference
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=S.SURFACE)
    S.setup_axes(ax, f"{args.ten}: dev nats/word by train-10M frequency per epoch (dots = {args.hundred} final)", "epoch", "nats / word")
    tf = t[t.dim == "freq10"]
    es = sorted(ep)
    for j, r in enumerate(tf.itertuples()):
        ax.plot(es, [getattr(r, f"dev_ep{e}") for e in es], color=S.RAMP[j], lw=2, label=r.label)
        ax.plot([10.3], [r.final100], "o", color=S.RAMP[j], ms=5)
    ax.legend(fontsize=7, frameon=False, ncol=2, title="train-10M count", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"epochs_by_freq_{args.ten}.png", dpi=150)
    plt.close(fig)

    # exposure-matched curve
    bins = np.r_[0, 2.0 ** np.arange(0, 21)]
    erows = []
    for e in es:
        exp = e * F["freq10"]
        for k in range(len(bins) - 1):
            m = (exp >= bins[k]) & (exp < bins[k + 1])
            if m.sum() >= 50:
                erows.append({"run": args.ten, "words_seen": e * 10e6, "exp_lo": bins[k], "n": int(m.sum()), "mean_nats": float(ep[e][m].mean())})
    for rev in [] if args.no_equal_exposure else S.scored(args.hundred):
        ws = S.words_seen(rev)
        if ws > TRAIN100_WORDS * 1.001:
            continue
        nll = S.load_score(args.hundred, rev, "dev")["word_nll"]
        exp = F["freq100"] * ws / TRAIN100_WORDS
        for k in range(len(bins) - 1):
            m = (exp >= bins[k]) & (exp < bins[k + 1])
            if m.sum() >= 50:
                erows.append({"run": args.hundred, "words_seen": ws, "exp_lo": bins[k], "n": int(m.sum()), "mean_nats": float(nll[m].mean())})
    ex = pd.DataFrame(erows)
    ex.to_csv(out / f"exposure_{args.ten}__{args.hundred}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=S.SURFACE)
    S.setup_axes(ax, "Dev nats/word vs times the word was seen in training", "exposures (log2 bins)", "nats / word", logx=True)
    for i, run in enumerate((args.ten, args.hundred)):
        for ws, g in ex[ex.run == run].groupby("words_seen"):
            if run == args.ten and ws not in (10e6, 30e6, 100e6):
                continue
            if run == args.hundred and ws not in (10e6, 30e6, 100e6, 20e6, 50e6):
                continue
            ls = {10e6: ":", 20e6: "-.", 30e6: "-.", 50e6: "--", 100e6: "-"}.get(ws, "-")
            ax.plot(g.exp_lo.clip(lower=0.5), g.mean_nats, ls, color=S.CAT[i], lw=2, label=f"{run} @ {ws/1e6:.0f}M words")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"exposure_{args.ten}__{args.hundred}.png", dpi=150)
    plt.close(fig)
    # repetition penalty: both runs at the same words seen (same number of steps); the 10M run
    # repeats its data, the 100M run sees fresh text. Its learning rate is lower at equal W
    # (shorter schedule), which favours the repeating run, so this is a lower bound.
    prow = []
    for W in () if args.no_equal_exposure else (10, 20, 30, 50, 70, 100):
        rev = f"chck_{W}M"
        a10, a100 = S.load_score(args.ten, rev, "dev"), S.load_score(args.hundred, rev, "dev")
        if a10 is None or a100 is None:
            continue
        d = a10["word_nll"].astype(np.float64) - a100["word_nll"].astype(np.float64)
        lo, hi = S.bootstrap_gap(d, chunk, np.zeros(N, dtype=int), 1)[4]
        prow.append({"words_seen_M": W, "dim": "all", "label": "all", "n": N, "share": 1.0, "penalty": float(d.mean()),
                     "contrib": float(d.mean()), "ci_lo": lo, "ci_hi": hi})
        for dim in ("freq10", "ngram10", "position", "context", "bigram_cov", "source"):
            b, labels = D[dim]
            for k, lab in enumerate(labels):
                m = b == k
                if m.sum():
                    prow.append({"words_seen_M": W, "dim": dim, "label": lab, "n": int(m.sum()), "share": float(m.mean()),
                                 "penalty": float(d[m].mean()), "contrib": float(d[m].sum() / N)})
    if prow:
        pr = pd.DataFrame(prow)
        pr.to_csv(out / f"repetition_penalty_{args.ten}__{args.hundred}.csv", index=False)
        summary["repetition_penalty"] = {int(r.words_seen_M): [round(r.penalty, 4), round(r.ci_lo, 4), round(r.ci_hi, 4)]
                                         for r in pr[pr.dim == "all"].itertuples()}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=S.SURFACE)
        S.setup_axes(axes[0], "Repeated minus fresh data at equal words seen, by train-10M frequency", "words seen (M)", "nats / word (mean in band)")
        S.setup_axes(axes[1], "Same, as contribution to all-text nats/word", "words seen (M)", "nats / word")
        pf = pr[pr.dim == "freq10"]
        for j, lab in enumerate(S.FREQ_LABELS):
            g = pf[pf.label == lab].sort_values("words_seen_M")
            axes[0].plot(g.words_seen_M, g.penalty, color=S.RAMP[j], lw=2, marker="o", ms=3, label=lab)
            axes[1].plot(g.words_seen_M, g.contrib, color=S.RAMP[j], lw=2, marker="o", ms=3, label=lab)
        ga = pr[pr.dim == "all"].sort_values("words_seen_M")
        axes[1].plot(ga.words_seen_M, ga.contrib, color=S.INK, lw=2, label="all")
        for ax in axes:
            ax.axhline(0, color=S.INK2, lw=0.8)
        axes[0].legend(fontsize=7, frameon=False, ncol=2, title="train-10M count", title_fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"repetition_penalty_{args.ten}__{args.hundred}.png", dpi=150)
        plt.close(fig)
    # effective data: how many fresh words (on the 100M run's own curve) each epoch of the
    # 10M run is worth, overall and per frequency band. The 100M run is not annealed at its
    # early checkpoints, so these are upper bounds on the value of repetition.
    h_revs = [] if args.no_equal_exposure else [r for r in S.scored(args.hundred) if S.words_seen(r) <= 1000e6]
    if len(h_revs) >= 3:
        hw = np.array([S.words_seen(r) for r in h_revs])
        hl = {r: S.load_score(args.hundred, r, "dev")["word_nll"].astype(np.float64) for r in h_revs}
        erow = []
        for e in es:
            w_eff, flag = effective_words(hw, np.array([hl[r].mean() for r in h_revs]), float(ep[e].mean()))
            erow.append({"epoch": e, "words_seen": e * 10e6, "dim": "all", "label": "all", "effective_words": w_eff, "flag": flag})
        fbins, flabels = D["freq10"]
        for k, lab in enumerate(flabels):
            m = fbins == k
            if m.sum() < 200:
                continue
            for e in es:
                w_eff, flag = effective_words(hw, np.array([hl[r][m].mean() for r in h_revs]), float(ep[e][m].mean()))
                erow.append({"epoch": e, "words_seen": e * 10e6, "dim": "freq10", "label": lab, "effective_words": w_eff, "flag": flag})
        ef = pd.DataFrame(erow)
        ef.to_csv(out / f"effective_data_{args.ten}__{args.hundred}.csv", index=False)
        summary["effective_words_M"] = {int(r.epoch): f"{r.flag}{r.effective_words / 1e6:.1f}" for r in ef[ef.dim == "all"].itertuples()}
        summary["effective_words_final_by_freq10_M"] = {r.label: f"{r.flag}{r.effective_words / 1e6:.1f}"
                                                         for r in ef[(ef.dim == "freq10") & (ef.epoch == max(es))].itertuples()}
        fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=S.SURFACE)
        S.setup_axes(ax, f"{args.ten}: each epoch's dev loss in fresh-word equivalents on {args.hundred}'s curve",
                     "words seen by the 10M run (M)", "fresh words reaching the same loss (M)")
        ga = ef[ef.dim == "all"]
        ax.plot([10, 100], [10, 100], ":", color=S.INK2, lw=1, label="repetition as good as fresh text")
        ax.plot(ga.words_seen / 1e6, ga.effective_words / 1e6, color=S.INK, lw=2, marker="o", ms=4, label="all words")
        gf = ef[ef.dim == "freq10"]
        for j, lab in enumerate(S.FREQ_LABELS):
            g = gf[gf.label == lab]
            if len(g):
                ax.plot(g.words_seen / 1e6, g.effective_words / 1e6, color=S.RAMP[j], lw=1.5, label=f"train-10M count {lab}")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"effective_data_{args.ten}__{args.hundred}.png", dpi=150)
        plt.close(fig)
    (out / f"bottleneck_{args.ten}__{args.hundred}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
