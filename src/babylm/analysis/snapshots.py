"""Snapshot mixtures: a run's final next-token distribution mixed with an earlier epoch's, at zero extra exposure.

    python -m babylm.analysis.snapshots RUN_DIR [RUN_DIR ...] [--out DIR] [--weights 0.1,0.2,0.3,0.4,0.5]

Reads RUN_DIR/pt/pt_dev_<label>.npy (per-token DEV-sub nats in stream order, NaN at the unscored first token;
babylm.analysis.post_runs packs every label from one epoch on for --pt_runs). A token-level mixture gives the observed
token the mixture of the two snapshots' probabilities, so per-token nats suffice: -log((1 - w) e^-final + w e^-label).
The snapshot label and w are picked on the even DEV-sub documents and reported on the odd ones (as
babylm.analysis.knn_diag; a BOS target opens its document); the odd-document oracle is printed beside.
nats/word = summed token nats / DEV-sub words with a scored token in those documents.
Writes OUT/snapshots.csv (run, label, w, nats/token and nats/word on even and odd documents) and OUT/summary.json
(per run: final alone, the pick, its held-out gain). Known idea (snapshot ensembles, Huang et al. 2017); the
instruments measured +0.040 to +0.048 nats/word for the published GPT-BERT 10M (epoch 7 or 5 mixed into the final).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from babylm.analysis import perword
from babylm.data import corpus as C


def mix(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Per-token nats of the mixture (1 - w) p_a + w p_b, given per-token nats a and b (NaN stays NaN)."""
    with np.errstate(invalid="ignore"):
        return -np.logaddexp(math.log1p(-w) - a, math.log(w) - b)


def snapshot_table(run: Path, weights: list[float], data_dir: Path | None) -> tuple[list[dict], dict]:
    d = C.load_split("dev_sub", data_dir)
    toks = np.asarray(d.tokens)
    starts = np.flatnonzero(toks == C.BOS_ID)
    doc = np.searchsorted(starts, np.arange(len(toks)), side="right") - 1
    words = np.asarray(d.words)
    dumps = {perword.label_words(f.name[len("pt_dev_"):-4]): f for f in (run / "pt").glob("pt_dev_*.npy")}
    dumps = {w: f for w, f in dumps.items() if w is not None}
    if len(dumps) < 2:
        raise SystemExit(f"{run}: needs at least two pt_dev_<label>.npy snapshots")
    pts = {w: np.load(f).astype(np.float64) for w, f in sorted(dumps.items())}
    if any(len(v) != len(toks) for v in pts.values()):
        raise SystemExit(f"{run}: dump length differs from DEV-sub's {len(toks)} tokens")
    last = max(pts)
    fin = pts.pop(last)
    ok = ~np.isnan(fin) & np.all([~np.isnan(v) for v in pts.values()], axis=0)
    groups = {"even": ok & (doc % 2 == 0), "odd": ok & (doc % 2 == 1)}
    nwords = {g: np.unique(words[m]).size for g, m in groups.items()}

    def stats(x: np.ndarray) -> dict[str, float]:
        return {f"{q}_{g}": float(x[m].sum() / (m.sum() if q == "npt" else nwords[g]))
                for g, m in groups.items() for q in ("npt", "npw")}

    base = stats(fin)
    rows = [{"run": run.name, "label": f"{last / 1e6:g}M", "w": 0.0, **base}]
    for lab, x in pts.items():
        for w in weights:
            rows.append({"run": run.name, "label": f"{lab / 1e6:g}M", "w": w, **stats(mix(fin, x, w))})
    pick = min(rows[1:], key=lambda r: r["npt_even"])
    oracle = min(rows[1:], key=lambda r: r["npt_odd"])
    summary = {"final": rows[0]["label"], "final_npt_odd": base["npt_odd"], "final_npw_odd": base["npw_odd"],
               "pick": {k: pick[k] for k in ("label", "w")}, "pick_npt_odd": pick["npt_odd"],
               "pick_npw_odd": pick["npw_odd"], "gain_npt_odd": base["npt_odd"] - pick["npt_odd"],
               "gain_npw_odd": base["npw_odd"] - pick["npw_odd"],
               "oracle_odd": {k: oracle[k] for k in ("label", "w")},
               "oracle_gain_npw_odd": base["npw_odd"] - oracle["npw_odd"],
               "tokens": {g: int(m.sum()) for g, m in groups.items()}, "words": nwords}
    return rows, summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="snapshots")
    ap.add_argument("--weights", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--data_dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = [float(x) for x in a.weights.split(",")]
    rows, summ = [], {}
    for r in a.runs:
        t, s = snapshot_table(Path(r), weights, Path(a.data_dir) if a.data_dir else None)
        rows += t
        summ[Path(r).name] = s
        print(Path(r).name, json.dumps({k: s[k] for k in ("pick", "gain_npw_odd", "oracle_gain_npw_odd")}))
    with open(out / "snapshots.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")


if __name__ == "__main__":
    main()
