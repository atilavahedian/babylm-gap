"""Where a 10M run's late epochs still pay, by word frequency: DEV-sub loss per train_10M frequency band at every
dumped epoch, the gap to a 100M reference per band, and the change between chosen epochs per band, with document
CIs. Never-seen words are also shown without CHILDES markup (slices.CHAT_MARKUP: none in train_10M, so no 10M-only
method can learn it).

  BABYLM_SAMPLE=harness python -m babylm.instruments.bands --tag base_r1 --ref grid_S_100M_e10_s0/chck_1000M \
      --change 50M:100M --change 70M:100M RUN
Every run needs `harness import` first. Writes results/babylm/<tag>/bands_<RUN>.csv (loss per band per epoch),
bands_<RUN>_change.csv (change per band between epochs; below 0: the loss fell) and bands_<RUN>.png.
"""
from __future__ import annotations

import argparse

import numpy as np

from babylm.instruments import slices as S
from babylm.instruments.cells import epoch, mean_ci

# train_10M count bands (slices.dims freq10 labels)
BANDS = [("8k+", ["8k+"]), ("64-8k", ["64-255", "256-1023", "1k-8k"]), ("1-63", ["1", "2-3", "4-7", "8-15", "16-63"]),
         ("never seen", ["0"])]


def masks(F, D) -> dict[str, np.ndarray]:
    f10, fl = D["freq10"]
    mk = F["markup"]
    m = {name: np.isin(f10, [fl.index(x) for x in labs]) for name, labs in BANDS}
    m["never seen, no markup"] = m["never seen"] & ~mk
    m["all"] = np.ones(len(f10), bool)
    m["all, no markup"] = ~mk
    return m


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", required=True, help="100M reference run/rev")
    ap.add_argument("--change", action="append", default=[], help="A:B in words seen, e.g. 70M:100M")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    F = S.load_feats("dev")
    M, unit = masks(F, S.dims(F)), F["unit"]
    nll = lambda r, v: S.load_score(r, v, "dev")["word_nll"].astype(np.float64)
    ref = nll(*args.ref.split("/"))
    revs = S.scored(args.run)
    if not revs:
        raise SystemExit(f"{args.run}: no DEV-sub scores (run harness import first)")
    loss = {v: nll(args.run, v) for v in revs}
    rows = []
    for v in revs:
        for b, m in M.items():
            g = mean_ci((loss[v] - ref)[m], unit[m])
            rows.append({"run": args.run, "rev": v, "epoch": epoch(v), "band": b, "words": int(m.sum()),
                         "word_share": m.mean(), "loss": loss[v][m].mean(), "ref_loss": ref[m].mean(),
                         "gap": g[0], "gap_lo": g[1], "gap_hi": g[2],
                         "gap_contrib": (loss[v] - ref)[m].sum() / len(ref)})
    t = pd.DataFrame(rows)
    t.to_csv(out / f"bands_{args.run}.csv", index=False)
    ch = []
    for spec in args.change:
        a, b = (f"chck_{x}" for x in spec.split(":"))
        if a not in loss or b not in loss:
            print(f"skip {spec}: no dump")
            continue
        n = epoch(b) - epoch(a)
        for band, m in M.items():
            c = mean_ci((loss[b] - loss[a])[m], unit[m])
            ch.append({"run": args.run, "from_epoch": epoch(a), "to_epoch": epoch(b), "band": band,
                       "change": c[0], "lo": c[1], "hi": c[2], "per_epoch": c[0] / n,
                       "contrib": (loss[b] - loss[a])[m].sum() / len(m)})
    tc = pd.DataFrame(ch)
    tc.to_csv(out / f"bands_{args.run}_change.csv", index=False)

    show = [b for b, _ in BANDS] + ["never seen, no markup"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=S.SURFACE)
    S.setup_axes(axes[0], f"DEV-sub loss by train_10M count (dashed: {args.ref.split('/')[0]})", "epochs of train_10M",
                 "nats / word")
    S.setup_axes(axes[1], "Gap to the reference by band (95% CI)", "epochs of train_10M", "nats / word")
    for i, b in enumerate(show):
        g = t[t.band == b].sort_values("epoch")
        col = S.CAT[i % len(S.CAT)]
        ls = ":" if b == "never seen, no markup" else "-"
        axes[0].plot(g.epoch, g.loss, color=col, lw=2, ls=ls, marker="o", ms=3, label=f"{b} ({g.word_share.iloc[0]:.1%} of words)")
        if ls == "-":
            axes[0].axhline(g.ref_loss.iloc[0], color=col, lw=1, ls="--")
        axes[1].plot(g.epoch, g.gap, color=col, lw=2, ls=ls, marker="o", ms=3, label=b)
        axes[1].fill_between(g.epoch, g.gap_lo, g.gap_hi, color=col, alpha=0.15, lw=0)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10], ["0.1", "0.2", "0.5", "1", "2", "5", "10"])
        ax.legend(fontsize=7.5, frameon=False)
    axes[1].axhline(0, color=S.INK2, lw=0.8)
    fig.suptitle(f"{args.run} on DEV-sub, by how often each word occurs in train_10M", color=S.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"bands_{args.run}.png", dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 250)
    last = t[t.rev == revs[-1]]
    print(last.drop(columns=["run", "rev"]).round(3).to_string(index=False))
    if len(tc):
        print(tc.drop(columns=["run"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
