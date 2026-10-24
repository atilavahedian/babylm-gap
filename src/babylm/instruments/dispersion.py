"""Contexts per type: is the gap larger for words whose train_10M occurrences are bunched together?

For every train_10M word type (normal form): its count, its range (number of 1000-word
blocks it occurs in) and its dispersion = range / the range expected if its occurrences
were spread at random over the B blocks, B (1 - (1 - 1/B)^count). Bursty words (character
names, topic words) have dispersion well below 1. A second view is left diversity: distinct
preceding words per occurrence. Within each train-10M frequency band, dev words that are the
first occurrence of their type in the chunk (so in-context copying cannot help) are split
into terciles of each measure, and the gap of each pair is compared across terciles.

Usage:
  python -m babylm.instruments.dispersion --tag preread_v1 --data-root ~/babylm_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from babylm.instruments import corpus as C
from babylm.instruments import slices as S
from babylm.instruments.score import load_rows
from babylm.instruments.signature import ratio_ci

PAIRS = {"10x data (GPT-BERT)": ("gptbert_cf_10m/chck_100M", "gptbert_cf_100m/chck_1000M"),
         "10x data (GPT-2)": ("gpt2_10m/chck_100M", "gpt2_100m/chck_1000M"),
         "epochs 1 to 10 (GPT-2 10M)": ("gpt2_10m/chck_10M", "gpt2_10m/chck_100M")}
BANDS = ["8-15", "16-63", "64-255", "256-1023"]
TERC = ["lowest third", "middle third", "highest third"]
NOTE = {"dispersion": "low = bursty (bunched in few blocks)", "left diversity": "low = few distinct preceding words per occurrence"}


def type_stats(data_root: Path, block: int = 1000) -> tuple[C.Vocab, np.ndarray, np.ndarray, np.ndarray]:
    vocab = C.Vocab()
    ids = C.encode_split(vocab, C.split_files(data_root, "train_10M"))
    V, B = len(vocab), -(-len(ids) // block)
    count = np.bincount(ids, minlength=V).astype(np.float64)
    occ = np.unique(ids * B + np.arange(len(ids)) // block)
    rng = np.bincount(occ // B, minlength=V).astype(np.float64)
    expected = B * (1 - np.power(1 - 1 / B, count))
    with np.errstate(invalid="ignore", divide="ignore"):
        disp = rng / expected
        left = np.bincount(np.unique(ids[:-1] * V + ids[1:]) % V, minlength=V) / count
    return vocab, count, disp, left


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--data-root", type=Path, required=True)
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    F = S.load_feats("dev")
    vocab, count, disp, left = type_stats(args.data_root)
    wid = np.array([vocab.ids.get(C.norm(w), -1) for r in load_rows("dev") for w in r["text"].split()])
    cnt = np.where(wid >= 0, count[np.maximum(wid, 0)], 0)
    mismatch = int((cnt != F["freq10"]).sum())
    f10, fl = S.dims(F)["freq10"]
    first = F["ctx_prev"] == 0
    measures = {"dispersion": np.where(wid >= 0, disp[np.maximum(wid, 0)], np.nan),
                "left diversity": np.where(wid >= 0, left[np.maximum(wid, 0)], np.nan)}
    rows, summ = [], {"count_mismatch_vs_features": mismatch, "pooled": {}}
    terc = {}
    for mname, x in measures.items():
        t = np.full(len(x), -1)
        for band in BANDS:
            m = first & (f10 == fl.index(band)) & np.isfinite(x)
            q = np.quantile(x[m], [1 / 3, 2 / 3])
            t[m] = np.digitize(x[m], q)
        terc[mname] = t
    for pname, (a, b) in PAIRS.items():
        sa, sb = S.load_score(*a.split("/"), "dev"), S.load_score(*b.split("/"), "dev")
        if sa is None or sb is None:
            print(f"skip {pname}: missing scores")
            continue
        la = sa["word_nll"].astype(np.float64)
        d = la - sb["word_nll"].astype(np.float64)
        for mname, t in terc.items():
            for band in ["all bands"] + BANDS:
                inband = t >= 0 if band == "all bands" else (t >= 0) & (f10 == fl.index(band))
                for k, lab in enumerate(TERC):
                    m = inband & (t == k)
                    rows.append({"pair": pname, "measure": mname, "band": band, "tercile": lab, "n": int(m.sum()),
                                 "measure_median": float(np.median(measures[mname][m])), "loss_weaker": la[m].mean(),
                                 "gap": d[m].mean(), "share_removed": d[m].sum() / la[m].sum()})
            ones = np.ones(len(d))
            r = ratio_ci(d, ones, t == 0, t == 2, F["unit"])
            rs = ratio_ci(d, la, t == 0, t == 2, F["unit"])
            summ["pooled"][f"{pname} | {mname}"] = {
                "gap_ratio_bursty_over_spread": [round(v, 3) for v in r],
                "share_removed_ratio_bursty_over_spread": [round(v, 3) for v in rs]}
            print(pname, "|", mname, summ["pooled"][f"{pname} | {mname}"], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "dispersion.csv", index=False)
    (out / "dispersion.json").write_text(json.dumps(summ, indent=1))

    pairs = list(dict.fromkeys(t.pair))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=S.SURFACE, squeeze=False)
    for ax, mname in zip(axes[0], measures):
        S.setup_axes(ax, f"Gap per word by {mname} tercile ({NOTE[mname]})",
                     "first occurrences of words seen 8-1023 times in train_10M; terciles within each band",
                     "nats / word")
        w = 0.8 / len(pairs)
        for i, pname in enumerate(pairs):
            g = t[(t.pair == pname) & (t.measure == mname) & (t.band == "all bands")]
            ax.bar(np.arange(3) - 0.4 + w * (i + 0.5), g.gap, w * 0.9, color=S.CAT[i], label=pname)
        ax.set_xticks(np.arange(3), TERC, fontsize=8)
        ax.set_ylim(0, t[(t.measure == mname) & (t.band == "all bands")].gap.max() * 1.3)
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(out / "dispersion.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
