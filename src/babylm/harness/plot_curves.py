"""Plot a BabyLM run's curves: python -m babylm.harness.plot_curves RUN_DIR [RUN_DIR ...] [--out DIR]

Writes into each run dir (or --out for several runs overlaid):
  train_loss.png   train loss (nats/token) vs words seen (log x)
  eval_curves.png  DEV-sub and TRAIN-sub nats/word vs words seen (top) and their gap DEV - TRAIN (bottom)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_csv(p: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(p.open())) if p.exists() else []


def col(rows, k):
    return [float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")]


def plot(runs: list[Path], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    many = len(runs) > 1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in runs:
        rows = read_csv(r / "train_log.csv")
        if rows:
            w, l = col(rows, "words_seen"), col(rows, "loss")
            ax.plot([max(x, 1) for x in w], l, lw=0.8, label=r.name if many else "train loss")
    ax.set_xscale("log")
    ax.set_xlabel("words seen")
    ax.set_ylabel("train loss (nats/token)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title("train loss" + ("" if many else f": {runs[0].name}"), fontsize=9)
    fig.tight_layout()
    p1 = out / "train_loss.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for r in runs:
        rows = read_csv(r / "curves.csv")
        if not rows:
            continue
        w = col(rows, "words_seen")
        pre = f"{r.name} " if many else ""
        l1, = a1.plot(w, col(rows, "dev_nats_per_word"), "o-", ms=3, label=pre + "DEV-sub")
        a1.plot(w, col(rows, "train_nats_per_word"), "s--", ms=3, color=l1.get_color(), alpha=0.7,
                label=pre + "TRAIN-sub")
        a2.plot(w, col(rows, "gap_nats_per_word"), "o-", ms=3, color=l1.get_color(), label=pre + "DEV - TRAIN")
    a1.set_xscale("log")
    a1.set_ylabel("nats / word")
    a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=7)
    a1.set_title("held-out (DEV-sub) vs training (TRAIN-sub) loss" + ("" if many else f": {runs[0].name}"),
                 fontsize=9)
    a2.axhline(0, color="k", lw=0.5)
    a2.set_xlabel("words seen")
    a2.set_ylabel("gap (nats / word)")
    a2.grid(alpha=0.3, which="both")
    a2.legend(fontsize=7)
    fig.tight_layout()
    p2 = out / "eval_curves.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    return [p1, p2]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None, help="output dir (default: the run dir; required for several runs)")
    a = ap.parse_args(argv)
    runs = [Path(r) for r in a.runs]
    if len(runs) > 1 and not a.out:
        raise SystemExit("--out is required with several runs")
    for p in plot(runs, Path(a.out) if a.out else runs[0]):
        print(p)


if __name__ == "__main__":
    main()
