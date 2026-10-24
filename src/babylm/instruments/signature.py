"""Lever signatures: which words does each kind of improvement help (CSV + PNG)?

A lever is a pair of scored checkpoints, weaker:stronger (10x data, more epochs, fresh vs
repeated text, a recipe change, later an attack vs its base). Per slice it reports the
share of the weaker model's loss that the stronger one removes, and two tilts with
document-bootstrap CIs:
  rare tilt     share removed on words seen 1 to 15 times in train_10M divided by the
                share removed on words seen 64 or more times
  novelty tilt  share removed where the trigram ending in the word is in neither train_10M
                nor train_100M (the word itself is in train_10M) divided by the share removed
                where the trigram is in train_10M; trigrams only in train_100M are left out,
                since the 100M model saw them and a 10M attack cannot
Extra epochs help contexts already in the data more than new ones (novelty tilt under 1);
more data does not. An attack placed on the same map shows whether its gain looks like
more data or like more passes over the same words.

Usage:
  python -m babylm.instruments.signature --tag preread_v1                      # default levers
  python -m babylm.instruments.signature --tag preread_v1 \
      --lever "my attack=base_run/chck_100M:attack_run/chck_100M"            # add ';pos<128' to restrict
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from babylm.instruments import slices as S

DEFAULT = [
    "10x data (GPT-BERT)=gptbert_cf_10m/chck_100M:gptbert_cf_100m/chck_1000M",
    "10x data (GPT-2)=gpt2_10m/chck_100M:gpt2_100m/chck_1000M",
    "fresh vs repeated text (GPT-2, 100M words seen)=gpt2_10m/chck_100M:gpt2_100m/chck_100M",
    "recipe GPT-2 to GPT-BERT (10M)=gpt2_10m/chck_100M:gptbert_cf_10m/chck_100M",
    "epochs 1 to 10 (GPT-2 10M)=gpt2_10m/chck_10M:gpt2_10m/chck_100M",
    "epochs 7 to 10 (GPT-2 10M)=gpt2_10m/chck_70M:gpt2_10m/chck_100M",
    "epochs 2 to 10 (GPT-BERT 10M)=gptbert_cf_10m/chck_20M:gptbert_cf_10m/chck_100M;pos<128",
    "epochs 7 to 10 (GPT-BERT 10M)=gptbert_cf_10m/chck_70M:gptbert_cf_10m/chck_100M;pos<128",
]
RARE, FREQ = ["1", "2-3", "4-7", "8-15"], ["64-255", "256-1023", "1k-8k", "8k+"]


def parse(spec: str) -> tuple[str, str, str, int]:
    label, rest = spec.split("=", 1)
    pair, _, cond = rest.partition(";")
    a, b = pair.split(":")
    return label, a, b, int(cond.split("<")[1]) if cond.startswith("pos<") else 0


def ratio_ci(d: np.ndarray, a: np.ndarray, num: np.ndarray, den: np.ndarray, chunk: np.ndarray,
             reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(sum d / sum a over num) / (sum d / sum a over den), with a document-cluster Poisson bootstrap."""
    nc = int(chunk.max()) + 1
    agg = lambda m, x: np.bincount(chunk[m], weights=x[m], minlength=nc)
    dn, an, dd, ad = agg(num, d), agg(num, a), agg(den, d), agg(den, a)
    point = (dn.sum() / an.sum()) / (dd.sum() / ad.sum())
    W = np.random.default_rng(seed).poisson(1.0, size=(reps, nc)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = ((W @ dn) / (W @ an)) / ((W @ dd) / (W @ ad))
    return float(point), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from babylm.instruments.stall import word_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--lever", action="append", default=[], help="label=run/rev:run/rev[;pos<N] (weaker:stronger)")
    ap.add_argument("--no-defaults", action="store_true")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    F = S.load_feats("dev")
    D = S.dims(F)
    f10, fl = D["freq10"]
    cov, cl = D["trigram_cov"]
    chunk = F["unit"]
    rare = np.isin(f10, [fl.index(x) for x in RARE])
    freq = np.isin(f10, [fl.index(x) for x in FREQ])
    novel = cov == cl.index("in neither")
    seen = cov == cl.index("seen in 10M")
    rows, tilts = [], []
    for spec in ([] if args.no_defaults else DEFAULT) + args.lever:
        label, a, b, maxpos = parse(spec)
        sa, sb = S.load_score(*a.split("/"), "dev"), S.load_score(*b.split("/"), "dev")
        if sa is None or sb is None:
            print(f"skip {label}: missing scores")
            continue
        keep = word_mask(sa, len(chunk), maxpos)
        la = sa["word_nll"].astype(np.float64)
        d = la - sb["word_nll"].astype(np.float64)
        rows.append({"lever": label, "weaker": a, "stronger": b, "dim": "all", "label": "all", "n": int(keep.sum()),
                     "loss_weaker": la[keep].mean(), "gain": d[keep].mean(), "share_removed": d[keep].sum() / la[keep].sum()})
        for dim in ("freq10", "trigram_cov", "context"):
            bins, labels = D[dim]
            for k, lab in enumerate(labels):
                m = keep & (bins == k)
                if m.any():
                    rows.append({"lever": label, "weaker": a, "stronger": b, "dim": dim, "label": lab, "n": int(m.sum()),
                                 "loss_weaker": la[m].mean(), "gain": d[m].mean(), "share_removed": d[m].sum() / la[m].sum()})
        rt = ratio_ci(d, la, keep & rare, keep & freq, chunk)
        nt = ratio_ci(d, la, keep & novel, keep & seen, chunk)
        tilts.append({"lever": label, "weaker": a, "stronger": b, "max_tok_pos": maxpos, "gain": float(d[keep].mean()),
                      "rare_tilt": rt[0], "rare_lo": rt[1], "rare_hi": rt[2],
                      "novelty_tilt": nt[0], "novelty_lo": nt[1], "novelty_hi": nt[2]})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in tilts[-1].items()}), flush=True)
    if not tilts:
        return
    t, tl = pd.DataFrame(rows), pd.DataFrame(tilts)
    t.to_csv(out / "signature.csv", index=False)
    tl.to_csv(out / "signature_tilts.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=S.SURFACE, gridspec_kw={"width_ratios": [1.2, 1.1, 1]})
    S.setup_axes(axes[0], "Share of the weaker model's loss removed, by train-10M count", "times the word occurs in train_10M", "share of loss removed")
    S.setup_axes(axes[1], "Same, by whether the trigram ending in the word is in train_10M", "", "share of loss removed")
    S.setup_axes(axes[2], "Lever map (95% CI): extra epochs sit below 1 on novelty", "rare tilt (words seen 1-15 vs 64+ times)", "novelty tilt (trigram in neither corpus vs in train_10M)")
    levers = list(tl.lever)
    w = 0.8 / len(levers)
    for i, lev in enumerate(levers):
        col = S.CAT[i % len(S.CAT)]
        g = t[(t.lever == lev) & (t.dim == "freq10")]
        axes[0].plot(np.arange(len(g)), g.share_removed, color=col, lw=2, marker="o", ms=3, label=lev)
        g = t[(t.lever == lev) & (t.dim == "trigram_cov")]
        axes[1].bar(np.arange(len(g)) - 0.4 + w * (i + 0.5), g.share_removed, w * 0.9, color=col)
        r = tl[tl.lever == lev].iloc[0]
        axes[2].errorbar(r.rare_tilt, r.novelty_tilt, xerr=[[r.rare_tilt - r.rare_lo], [r.rare_hi - r.rare_tilt]],
                         yerr=[[r.novelty_tilt - r.novelty_lo], [r.novelty_hi - r.novelty_tilt]], fmt="o", color=col, ms=6, lw=1)
        axes[2].annotate(lev, (r.rare_tilt, r.novelty_tilt), xytext=(6, 7 if i % 2 == 0 else -13), textcoords="offset points",
                         fontsize=7, color=S.INK)
    axes[0].set_xticks(np.arange(len(S.FREQ_LABELS)), S.FREQ_LABELS, rotation=45, ha="right", fontsize=7)
    axes[1].set_xticks(np.arange(len(cl)), cl, rotation=20, ha="right", fontsize=7)
    axes[2].axvline(1, color=S.INK2, lw=0.8, ls=":")
    axes[2].axhline(1, color=S.INK2, lw=0.8, ls=":")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Lever signatures on the dev sample (BabyLM 2025 baselines; pre-read)", color=S.INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "signature.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
