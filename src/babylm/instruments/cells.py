"""Compare 10M runs that dump per-word scores at each epoch (a recipe sweep such as the base builder's round 1) on
DEV-sub, against a 100M reference and one or more base 10M runs (the port, a sweep's plain control). Per run:
  - loss at each dumped epoch, and the gap to the reference at the run's best epoch and final (DEV-sub as it is,
    document CIs: the same numbers as the port's 1.014 at epoch 2), also without CHILDES markup words;
  - whether it still overfits, and where: the gain and the rare and novelty tilts (signature.py) of epochs 1 to 2,
    2 to 5 (2 to 4 for a 4-epoch run) and 5 to 10, and the gain on words whose trigram is in train_10M ("seen") and
    in neither corpus ("new"). A loss rise whose novelty tilt is above 1 lands on new contexts first (the port: 3.40
    from epoch 2 to 5); a negative tilt means seen contexts improve while new ones get worse;
  - what it buys over each base run, at the finals and at the best epochs, with the same tilts.

  BABYLM_SAMPLE=harness python -m babylm.instruments.cells --tag base_r1 --ref grid_S_100M_e10_s0/chck_1000M \
      --base port=port_S_10M_e10_s0 --base control=base_S_wd2p0_lr0p5_final RUN [RUN ...]
Every run needs `harness import` first. Writes results/babylm/<tag>/cells.csv, cells_levers.csv and cells.png.
"""
from __future__ import annotations

import argparse

import numpy as np

from babylm.instruments import slices as S
from babylm.instruments.signature import FREQ, RARE, ratio_ci

EPOCH_LEVERS = [("chck_10M", "chck_20M"), ("chck_20M", "chck_50M"), ("chck_20M", "chck_40M"), ("chck_50M", "chck_100M")]


def epoch(rev: str) -> float:
    return S.words_seen(rev) / 1e7


def mean_ci(d: np.ndarray, unit: np.ndarray, reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of d over words, with a document-cluster Poisson bootstrap."""
    nc = int(unit.max()) + 1
    s, n = np.bincount(unit, weights=d, minlength=nc), np.bincount(unit, minlength=nc).astype(float)
    W = np.random.default_rng(seed).poisson(1.0, size=(reps, nc)).astype(float)
    boot = (W @ s) / (W @ n)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", required=True, help="100M reference run/rev")
    ap.add_argument("--base", action="append", default=[], help="name=run: a 10M run each run is compared against")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    F = S.load_feats("dev")
    D = S.dims(F)
    f10, fl = D["freq10"]
    cov, cl = D["trigram_cov"]
    unit = F["unit"]
    rare, freq = np.isin(f10, [fl.index(x) for x in RARE]), np.isin(f10, [fl.index(x) for x in FREQ])
    novel, seen = cov == cl.index("in neither"), cov == cl.index("seen in 10M")
    mk = F["markup"]  # CHILDES markup words, none in train_10M: gaps are also reported without them
    nll = lambda r, v: S.load_score(r, v, "dev")["word_nll"].astype(np.float64)
    ref = nll(*args.ref.split("/"))
    bases = dict(b.split("=", 1) for b in args.base)
    runs = list(dict.fromkeys(list(bases.values()) + args.runs))
    revs = {r: S.scored(r) for r in runs}
    for r in runs:
        if not revs[r]:
            raise SystemExit(f"{r}: no DEV-sub scores (run harness import first)")
    loss = {r: {v: nll(r, v) for v in revs[r]} for r in runs}
    best = {r: min(revs[r], key=lambda v: loss[r][v].mean()) for r in runs}
    final = {r: revs[r][-1] for r in runs}

    def lever(run, label, a, b, la, lb):
        d = la - lb
        g, rt, nt = mean_ci(d, unit), ratio_ci(d, la, rare, freq, unit), ratio_ci(d, la, novel, seen, unit)
        gs, gn = mean_ci(d[seen], unit[seen]), mean_ci(d[novel], unit[novel])
        return {"run": run, "lever": label, "weaker": a, "stronger": b, "gain": g[0], "gain_lo": g[1], "gain_hi": g[2],
                "gain_seen": gs[0], "gain_seen_lo": gs[1], "gain_seen_hi": gs[2],
                "gain_new": gn[0], "gain_new_lo": gn[1], "gain_new_hi": gn[2],
                "share_removed": d.sum() / la.sum(), "rare_tilt": rt[0], "rare_lo": rt[1], "rare_hi": rt[2],
                "novelty_tilt": nt[0], "novelty_lo": nt[1], "novelty_hi": nt[2]}

    rows, levers = [], []
    for r in runs:
        row = {"run": r, "base": next((k for k, v in bases.items() if v == r), "")}
        for v in revs[r]:
            row[f"loss_e{epoch(v):g}"] = loss[r][v].mean()
        for which, v in (("best", best[r]), ("final", final[r])):
            g, gm = mean_ci(loss[r][v] - ref, unit), mean_ci((loss[r][v] - ref)[~mk], unit[~mk])
            row.update({f"{which}_epoch": epoch(v), f"{which}_loss": loss[r][v].mean(),
                        f"gap_{which}": g[0], f"gap_{which}_lo": g[1], f"gap_{which}_hi": g[2],
                        f"gap_{which}_nomarkup": gm[0], f"gap_{which}_nomarkup_lo": gm[1], f"gap_{which}_nomarkup_hi": gm[2]})
        rows.append(row)
        for a, b in EPOCH_LEVERS:
            if a in loss[r] and b in loss[r] and not (b == "chck_40M" and "chck_50M" in loss[r]):
                levers.append(lever(r, f"epochs {epoch(a):g} to {epoch(b):g}", f"{r}/{a}", f"{r}/{b}", loss[r][a], loss[r][b]))
        for name, br in bases.items():
            if br != r:
                for which, vb, vr in (("finals", final[br], final[r]), ("best epochs", best[br], best[r])):
                    levers.append(lever(r, f"over {name}, {which}", f"{br}/{vb}", f"{r}/{vr}", loss[br][vb], loss[r][vr]))
        levers.append(lever(r, "10x data, best epoch", f"{r}/{best[r]}", args.ref, loss[r][best[r]], ref))
    t, tl = pd.DataFrame(rows), pd.DataFrame(levers)
    t.to_csv(out / "cells.csv", index=False)
    tl.to_csv(out / "cells_levers.csv", index=False)

    # one row per run: bases first, then runs by their gap at the best epoch
    order = [r for r in runs if r in bases.values()] + sorted((r for r in runs if r not in bases.values()),
                                                                key=lambda r: t.set_index("run").gap_best[r])
    y = np.arange(len(order))[::-1]
    T = t.set_index("run").loc[order]
    L = tl.set_index(["run", "lever"])
    fig, axes = plt.subplots(1, 4, figsize=(21, 1.8 + 0.42 * len(order)), facecolor=S.SURFACE, sharey=True)
    a0, a1, a2, a3 = axes
    S.setup_axes(a0, f"Gap to {args.ref.split('/')[0]} at the best epoch", "nats / word (95% CI)", "")
    a0.barh(y, T.gap_best, color=S.CAT[0], height=0.6)
    a0.errorbar(T.gap_best, y, xerr=[T.gap_best - T.gap_best_lo, T.gap_best_hi - T.gap_best], fmt="none", ecolor=S.INK2, lw=1)
    for yy, (r, row) in zip(y, T.iterrows()):
        a0.text(row.gap_best_hi, yy, f"  {row.gap_best:.3f} (epoch {row.best_epoch:g})", va="center", fontsize=7.5, color=S.INK)
    a0.set_xlim(0, T.gap_best_hi.max() * 1.45)
    S.setup_axes(a1, "Loss change, epoch 5 to 10 (above 0: overfitting)", "nats / word (95% CI)", "")
    for yy, r in zip(y, order):
        if (r, "epochs 5 to 10") in L.index:
            q = L.loc[(r, "epochs 5 to 10")]
            a1.errorbar(-q.gain, yy, xerr=[[q.gain_hi - q.gain], [q.gain - q.gain_lo]], fmt="o", color=S.INK, ms=5, lw=1)
            a1.text(-q.gain_lo, yy, f"  {-q.gain:+.3f}", va="center", fontsize=7.5, color=S.INK)
        else:
            a1.text(0, yy, "  no epoch 5 and 10 dumps", va="center", fontsize=7.5, color=S.INK2)
    for ax, lev in ((a2, "epochs 2 to 5"), (a3, "epochs 5 to 10")):
        S.setup_axes(ax, f"Loss change, {lev}, by trigram (above 0: worse)", "nats / word (95% CI)", "")
        for k, (part, name) in enumerate((("seen", "trigram in train_10M"), ("new", "trigram in neither corpus"))):
            pts = [(yy, L.loc[(r, lev)]) for yy, r in zip(y, order) if (r, lev) in L.index]
            for i, (yy, q) in enumerate(pts):
                g, lo, hi = q[f"gain_{part}"], q[f"gain_{part}_lo"], q[f"gain_{part}_hi"]
                ax.errorbar(-g, yy + (0.14 - 0.28 * k), xerr=[[hi - g], [g - lo]], fmt="o", color=S.CAT[k], ms=5, lw=1,
                            label=name if i == 0 else None)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for ax in axes[1:]:
        ax.axvline(0, color=S.INK2, lw=0.8)
    a0.set_yticks(y, order, fontsize=7.5)
    for ax in axes:
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
    fig.suptitle(f"10M runs on DEV-sub vs {args.ref} (document-bootstrap CIs)", color=S.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "cells.png", dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(tl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
