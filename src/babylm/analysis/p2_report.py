"""Phase 2 report from a results pack (babylm.analysis.post_runs): grid table, registered predictions, slices, plots.

    python -m babylm.analysis.p2_report --pack DIR [--out DIR] [--size S] [--grid p2|v2]

Cells (run names in the pack, size S by default): 10M x {1,2,4,10,20} epochs (10M x 10 = port_<size>_10M_e10_s0),
30M x {3.333,10}, 100M x {1,10}, seeds 0-2 of 10M x 10. Missing cells are skipped. --grid v2 reads the v2 grid at
the tuned recipe instead (v2_<size>_<data>_e<epochs>_s<seed>; critic ref-0052) and adds the pair best10-best100:
the 10M and 100M cells (<= 10 epochs) with the lowest registered DEV nats/word. Both arms are picked on DEV, so every
number on that pair is labelled DEV-selected; 100M x 10 is the strict-track ceiling and only the prize anchor (critic
ref-0448).
Writes to --out:
  grid.csv          per run: data, epochs, words seen, throughput, registered DEV (dev_bpw: full dev, whitespace
                    collapsed) nats/word and nats/token, DEV-sub and TRAIN-sub nats/word at the end, their gap;
                    the harness's own full-DEV nats/word on all words and overlap-removed; label "overlap-inflated"
                    on every run trained beyond train_10M (critic ref-0260)
  predictions.csv   P1, P2, P3, P5, P6 as registered (critic_log 18:56Z): value, interval, hit
  prize.csv         overlap-removed gaps (critic ref-0260: every word of a DEV line found verbatim in train_100M
                    dropped; harness full-DEV per-word nats, one word set for every model): nats/word, nats/token,
                    nats/word reweighted to full-DEV domain word shares, and per domain. The prize that sizes and
                    ranks bottlenecks is prize / npw_ovr_rw. Per pair also the gap on text both models
                    trained on (TRAIN-sub, inside train_10M, which is inside train_100M) next to the gap on new
                    text (DEV-sub), both at each run's last milestone: seen_text_npw, new_text_devsub_npw, and the
                    new-text gap with DEV-sub lines found verbatim in train_100M dropped (new_text_devsub_npw_ovr).
                    Column label: DEV-selected (pair best10-best100), overlap-inflated (a run trained beyond train_10M
                    scored on words that include overlapping lines).
  slices_dev/       babylm.analysis.p2_slices on the full DEV split: prize (10M x10 - 100M x10), iso-exposure
                    (10M x10 - 100M x1), extra epochs (10M x1 - 10M x10), 10x data (10M x1 - 100M x1)
  slices_dev_ovr/   the same with overlapping lines dropped (the ranking input)
  p2_curves.png/csv DEV nats/word vs epochs at 10M, vs data at 10 epochs, and at 100M exposures; TRAIN-DEV gap
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from babylm.analysis import p2_slices, perword
from babylm.data.corpus import DEFAULT_DATA_DIR, SOURCES, load_split

LN2 = math.log(2)
# v2 predictions, registered with the critic before the v2 grid launched: (name, quantity, lo, hi, unit).
# quantity: "<pair>:<prize.csv quantity>", "<pair>:seen_over_new", "share:<bucket>" (best10-best100, overlap
# removed), "knn:<headline key>" (knn_p7 on v2 10M x 10 s0), "ens" (3 seeds, full DEV), "diff:<x>-<y>:<grid key>".
V2_PREDS = [
    ("V1 prize best10-best100, overlap-removed, domain-reweighted", "prize_best:npw_ovr_rw", 0.50, 0.95,
     "DEV nats/word"),
    ("V2 iso 10Mx10 - 100Mx1, overlap-removed, domain-reweighted", "iso:npw_ovr_rw", 0.20, 0.65, "DEV nats/word"),
    # V3 is not scored unless the DEV-sub new-text gap is positive (critic ref-0448: a negative ratio would pass)
    ("V3 seen-text gap / new-text gap, best10-best100", "prize_best:seen_over_new", -math.inf, 0.50, "ratio"),
    ("V4a share of the overlap-removed prize on c10<=63", "share:c10<=63", 0.25, 0.50, "share"),
    ("V4b share of the overlap-removed prize on absent10_in100", "share:absent10_in100", 0.05, 0.20, "share"),
    ("V5 kNN gain beyond temperature, held-out DEV-sub", "knn:gain_knn_beyond_temp", 0.03, 0.15, "nats/token"),
    ("V6 3-seed prob-average gain over the mean seed", "ens", 0.03, 0.10, "DEV nats/token"),
    ("V7a epochs 4 -> 10 gain (10M_e4 - 10M_e10)", "diff:10M_e4-10M_e10:dev_npw", -0.05, 0.20, "DEV nats/word"),
    ("V7b epochs 1 -> 4 gain (10M_e1 - 10M_e4)", "diff:10M_e1-10M_e4:dev_npw", 0.20, 0.70, "DEV nats/word"),
    # matched exposure (100M words either way): the DEV-selected best single 10M cell minus the probability average
    # of two 5-epoch seeds; positive = the ensemble is better
    ("V8 best single 10M cell - 2x5-epoch prob-average ensemble", "ens2x5", -0.02, 0.06, "DEV nats/token"),
]
PAIRS = {"prize": ("10M_e10", "100M_e10"), "iso": ("10M_e10", "100M_e1"), "epochs": ("10M_e1", "10M_e10"),
         "data": ("10M_e1", "100M_e1")}


def cells(size: str, grid: str = "p2") -> dict[str, str]:
    if grid == "v2":
        keys = ("10M_e1", "10M_e2", "10M_e4", "10M_e5", "10M_e10", "30M_e3p333", "30M_e10", "100M_e1", "100M_e10")
        return {k: f"v2_{size}_{k}_s0" for k in keys}
    c = {f"10M_e{e}": f"grid_{size}_10M_e{e}_s0" for e in ("1", "2", "4", "20")}
    c["10M_e10"] = f"port_{size}_10M_e10_s0"
    c.update({"30M_e3p333": f"grid_{size}_30M_e3p333_s0", "30M_e10": f"grid_{size}_30M_e10_s0",
              "100M_e1": f"grid_{size}_100M_e1_s0", "100M_e10": f"grid_{size}_100M_e10_s0"})
    return c


def seed_runs(size: str, grid: str = "p2") -> list[str]:
    return [f"{'v2' if grid == 'v2' else 'port'}_{size}_10M_e10_s{s}" for s in (0, 1, 2)]


def last_pw(run: Path, tag: str) -> np.ndarray | None:
    """Per-word nats of the run's last milestone on DEV-sub (tag dev) or TRAIN-sub (tag train)."""
    labs = [(perword.label_words(f.name[len(f"pw_{tag}_"):-4]), f) for f in (run / "pw").glob(f"pw_{tag}_*.npy")]
    labs = [(w, f) for w, f in labs if w is not None]
    return np.load(max(labs)[1]).astype(np.float64) if labs else None


def seen_vs_new(pack: Path, c: dict[str, str], x: str, y: str, ovl: np.ndarray | None = None) -> dict[str, float]:
    """x minus y on TRAIN-sub (text both trained on) and on DEV-sub (new text), nats/word, last milestones; with
    ovl (DEV-sub words of lines found verbatim in train_100M) also the new-text gap without them."""
    out = {}
    for q, tag in (("seen_text_npw", "train"), ("new_text_devsub_npw", "dev")):
        a, b = last_pw(pack / c[x], tag), last_pw(pack / c[y], tag)
        if a is not None and b is not None and len(a) == len(b):
            ok = ~np.isnan(a) & ~np.isnan(b)
            out[q] = float((a - b)[ok].mean())
            if tag == "dev" and ovl is not None and len(ovl) == len(a):
                out[q + "_ovr"] = float((a - b)[ok & ~ovl].mean())
    return out


def load(run: Path) -> dict | None:
    if not (run / "result.json").exists():
        return None
    r = json.loads((run / "result.json").read_text())
    d = json.loads((run / "dev_bpw.json").read_text()) if (run / "dev_bpw.json").exists() else {}
    last = r.get("last_milestone", {})
    return {"run": run.name, "split": r.get("split"), "epochs": r.get("epochs"), "words_seen": r.get("words_seen"),
            "steps": r.get("steps"), "tokens_per_s": round(r.get("tokens_per_s") or 0), "train_time_s":
            round(r.get("train_time_s") or 0), "gpu": r.get("gpu"), "dev_npw": d.get("nats_per_word"),
            "dev_npt": d.get("nats_per_token"), "devsub_npw": last.get("dev_nats_per_word"),
            "devsub_npt": last.get("dev_nats_per_token"), "trainsub_npw": last.get("train_nats_per_word"),
            "gap_npw": last.get("gap_nats_per_word")}


def overlap_removed(pack: Path, c: dict[str, str], keys: list[str], data_dir: Path) -> dict[str, dict[str, float]]:
    """Per cell: harness full-DEV nats/word and nats/token on all words and with the words of DEV lines found verbatim
    in train_100M dropped (one word set for every cell), the dropped version reweighted to full-DEV domain word
    shares, and per domain."""
    have = [k for k in keys if (pack / c[k] / "pw" / "pw_devfull_final.npy").exists()]
    if not have:
        return {}
    f = p2_slices.devsub_words(data_dir, "dev")
    keep = ~((f["lkey"] != 0) & np.isin(f["lkey"], p2_slices.line_hashes("train_100M", data_dir)))
    n = len(keep)
    tpw = np.bincount(np.asarray(load_split("dev", data_dir).words, dtype=np.int64), minlength=n + 1)[:n]
    pw = {k: np.load(pack / c[k] / "pw" / "pw_devfull_final.npy").astype(np.float64) for k in have}
    ok = np.all([~np.isnan(v) for v in pw.values()], axis=0)
    m = ok & keep
    share = {s: float((f["src"] == i).mean()) for i, s in enumerate(SOURCES)}
    out: dict[str, dict[str, float]] = {}
    for k, v in pw.items():
        r = {"npw_all": v[ok].mean(), "npt_all": v[ok].sum() / tpw[ok].sum(), "npw_ovr": v[m].mean(),
             "npt_ovr": v[m].sum() / tpw[m].sum(), "kept_word_share": m.sum() / ok.sum()}
        for i, s in enumerate(SOURCES):
            ms = m & (f["src"] == i)
            r[f"npw_ovr_{s}"] = v[ms].mean()
        r["npw_ovr_rw"] = sum(share[s] * r[f"npw_ovr_{s}"] for s in SOURCES)
        out[k] = {q: float(x) for q, x in r.items()}
    return out


def single_minus_ens(pack: Path, members: list[str], single: str) -> float | None:
    """Full-DEV nats/token of one run minus the probability average of the member runs, on tokens all of them score."""
    f = [pack / r / "pt" / "pt_devfull_final.npy" for r in (*members, single)]
    if not all(x.exists() for x in f):
        return None
    x = np.stack([np.load(q).astype(np.float64) for q in f])
    x = x[:, ~np.isnan(x).any(0)]
    ens = -(np.logaddexp.reduce(-x[:-1], axis=0) - math.log(len(members)))
    return float(x[-1].mean() - ens.mean())


def ensemble_gain(pts: list[np.ndarray]) -> tuple[float, float, float]:
    """Mean-seed and probability-averaged nats/token over tokens scored by every seed; returns (mean, ens, gain)."""
    x = np.stack([p.astype(np.float64) for p in pts])
    ok = ~np.isnan(x).any(0)
    x = x[:, ok]
    mean_seed = float(x.mean())
    ens = float(-(np.logaddexp.reduce(-x, axis=0) - math.log(len(pts))).mean())
    return mean_seed, ens, mean_seed - ens


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", default="p2_report")
    ap.add_argument("--size", default="S")
    ap.add_argument("--grid", default="p2", choices=["p2", "v2"])
    ap.add_argument("--data_dir", default=None)
    a = ap.parse_args(argv)
    pack, out = Path(a.pack), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = cells(a.size, a.grid)
    rows = {k: load(pack / v) for k, v in c.items()}
    rows = {k: v for k, v in rows.items() if v}
    seeds = [load(pack / r) for r in seed_runs(a.size, a.grid)]
    seeds = [s for s in seeds if s]
    pairs = dict(PAIRS)
    if a.grid == "v2":  # each arm at its best legal epoch count, chosen on the registered DEV number
        for k, run in (("v1_100M_e10", f"grid_{a.size}_100M_e10_s0"), ("v1_100M_e1", f"grid_{a.size}_100M_e1_s0")):
            r = load(pack / run)  # the WD 0 100M cells are the other 100M reference
            if r:
                c[k], rows[k] = run, r
        for arm, pre in (("best10", ("10M_e",)), ("best100", ("100M_e", "v1_100M_e"))):
            cand = [k for k in rows for q in pre if k.startswith(q)
                    and float(k[len(q):].replace("p", ".")) <= 10 and rows[k]["dev_npw"] is not None]
            if cand:
                best = min(cand, key=lambda k: rows[k]["dev_npw"])
                c[arm], rows[arm] = c[best], {**rows[best], "chosen_on_dev": best}
        pairs["prize_best"] = ("best10", "best100")
    data_dir = Path(a.data_dir) if a.data_dir else DEFAULT_DATA_DIR
    ovr = overlap_removed(pack, c, list(rows), data_dir)
    fs = p2_slices.devsub_words(data_dir, "dev_sub") if ovr else None
    ovl_sub = None if fs is None else (fs["lkey"] != 0) & np.isin(fs["lkey"], p2_slices.line_hashes("train_100M", data_dir))
    gen = {name: seen_vs_new(pack, c, x, y, ovl_sub) for name, (x, y) in pairs.items() if x in rows and y in rows}
    for k, v in rows.items():
        v["label"] = "overlap-inflated" if v["split"] != "train_10M" else ""
        v.update({q: round(ovr[k][q], 5) for q in ("npw_all", "npw_ovr", "npt_ovr")} if k in ovr else {})
    fields = ["cell", "run", "split", "epochs", "words_seen", "steps", "tokens_per_s", "train_time_s", "gpu",
              "dev_npw", "dev_npt", "label", "npw_all", "npw_ovr", "npt_ovr", "devsub_npw", "devsub_npt",
              "trainsub_npw", "gap_npw"]
    with open(out / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        for k, v in rows.items():
            if k not in ("best10", "best100"):
                w.writerow({"cell": k, **v})
        for s in seeds[1:]:
            w.writerow({"cell": "10M_e10_seed", **s})
        if a.grid == "v2" and (s := load(pack / f"v2_{a.size}_10M_e5_s1")):
            w.writerow({"cell": "10M_e5_seed", **s})
    def label(name: str, x: str, y: str, q: str) -> str:
        t = ["DEV-selected"] if name == "prize_best" else []
        if q in ("npw_all", "npt_all", "new_text_devsub_npw") and {rows[x]["split"], rows[y]["split"]} != {"train_10M"}:
            t.append("overlap-inflated")
        return "; ".join(t)

    with open(out / "prize.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "quantity", "value", "label"])
        for name, (x, y) in pairs.items():
            if x in ovr and y in ovr:
                for q in ovr[x]:
                    if q != "kept_word_share":
                        w.writerow([f"{name} {x}-{y}", q, round(ovr[x][q] - ovr[y][q], 5), label(name, x, y, q)])
            for q, v in gen.get(name, {}).items():
                w.writerow([f"{name} {x}-{y}", q, round(v, 5), label(name, x, y, q)])
            if name == "prize_best":
                w.writerow([f"{name} {x}-{y}", "chosen_on_dev",
                            f"{rows[x]['chosen_on_dev']} vs {rows[y]['chosen_on_dev']}", "DEV-selected"])
        if ovr:
            w.writerow(["all cells", "kept_word_share", round(next(iter(ovr.values()))["kept_word_share"], 5), ""])

    preds = []

    def pred(name: str, value: float | None, lo: float, hi: float, unit: str, note: str = "") -> None:
        hit = None if value is None else (lo <= value <= hi)
        preds.append({"prediction": name, "value": None if value is None else round(value, 4), "lo": lo, "hi": hi,
                      "unit": unit, "hit": hit, "note": note})

    def diff(x: str, y: str, key: str) -> float | None:
        if x in rows and y in rows and rows[x][key] is not None and rows[y][key] is not None:
            return rows[x][key] - rows[y][key]
        return None

    def ovr_note(x: str, y: str) -> str:
        if x in ovr and y in ovr:
            return (f"; overlap-inflated; overlap-removed {ovr[x]['npt_ovr'] - ovr[y]['npt_ovr']:.4f} nats/token, "
                    f"{ovr[x]['npw_ovr_rw'] - ovr[y]['npw_ovr_rw']:.4f} nats/word (domain-reweighted)")
        return "; overlap-inflated"

    registered = a.grid == "p2"  # P1-P6 belong to the v1 cells; v2 predictions are registered separately
    if registered:
        pred("P1 prize 10Mx10 - 100Mx10", diff("10M_e10", "100M_e10", "dev_npt"), 0.30, 0.50, "DEV nats/token",
             f"nats/word {diff('10M_e10', '100M_e10', 'dev_npw')}" + ovr_note("10M_e10", "100M_e10"))
        pred("P2 iso 10Mx10 - 100Mx1", diff("10M_e10", "100M_e1", "dev_npt"), -math.inf, 0.10, "DEV nats/token",
             f"nats/word {diff('10M_e10', '100M_e1', 'dev_npw')}" + ovr_note("10M_e10", "100M_e1"))
    pts = []
    for r in seed_runs(a.size, a.grid):
        f = pack / r / "pt" / "pt_devfull_final.npy"
        if f.exists():
            pts.append(np.load(f))
    ens_gain = None
    if len(pts) == 3:
        m, e, g = ensemble_gain(pts)
        ens_gain = g
        pred("P3 3-seed prob-average vs mean seed" if registered else "3-seed prob-average vs mean seed", g,
             0.03, 0.08, "DEV nats/token", f"mean seed {m:.4f}, ensemble {e:.4f}")
    elif registered:
        pred("P3 3-seed prob-average vs mean seed", None, 0.03, 0.08, "DEV nats/token", f"{len(pts)} of 3 seeds")

    have = sorted({k for p in pairs.values() for k in p if k in rows
                   and (pack / c[k] / "pw" / "pw_devfull_final.npy").exists()})
    shares: dict[str, dict[str, float]] = {"slices_dev": {}, "slices_dev_ovr": {}}
    if have:
        spec = [f"{k}={pack / c[k]}" for k in have]
        prs = [f"{x}-{y}" for x, y in pairs.values() if x in have and y in have]
        for sub, extra in (("slices_dev", []), ("slices_dev_ovr", ["--drop_overlap"])):
            sd = out / sub
            p2_slices.main(["--runs", *spec, "--subset", "dev", "--out", str(sd), *extra,
                            *(["--pairs", *prs] if prs else []), "--data_dir", str(data_dir)])
            if (sd / "delta.csv").exists():
                for r in csv.DictReader(open(sd / "delta.csv")):
                    if r["pair"] == ("best10-best100" if a.grid == "v2" else "10M_e10-100M_e10"):
                        shares[sub][r["bucket"]] = float(r["share_of_total_delta"])
    if not registered:  # v2 predictions (V2_PREDS)
        extra: dict[str, str] = {}

        def value(q: str) -> float | None:
            kind, _, rest = q.partition(":")
            if kind == "share":
                return shares["slices_dev_ovr"].get(rest)
            if kind == "knn":
                f = pack / c["10M_e10"] / "knn_p7" / "summary.json"
                return json.loads(f.read_text())["headline_held_out_nats_per_token"].get(rest) if f.exists() else None
            if kind == "ens":
                return ens_gain
            if kind == "ens2x5":
                members = [f"v2_{a.size}_10M_e5_s{i}" for i in (0, 1)]
                return single_minus_ens(pack, members, c["best10"]) if "best10" in c else None
            if kind == "diff":
                xy, _, key = rest.partition(":")
                return diff(*xy.split("-"), key)
            pair = kind
            x, y = pairs.get(pair, ("", ""))
            if rest == "seen_over_new":  # not scored unless the new-text gap is positive (critic ref-0448)
                g = gen.get(pair, {})
                seen, new, new_o = (g.get(k) for k in ("seen_text_npw", "new_text_devsub_npw", "new_text_devsub_npw_ovr"))
                if seen is not None and new_o is not None:
                    extra[q] = (f"; overlap-inflated; overlap-removed ratio {seen / new_o:.4f}" if new_o > 0
                                else "; overlap-removed new-text gap <= 0")
                if new is not None and new <= 0:
                    extra[q] = "; not scored: DEV-sub new-text gap <= 0" + extra.get(q, "")
                return seen / new if seen is not None and new is not None and new > 0 else None
            return ovr[x][rest] - ovr[y][rest] if x in ovr and y in ovr else None
        preds.clear()
        for name, q, lo, hi, unit in V2_PREDS:
            v = value(q)  # best10 and best100 are both picked on DEV
            sel = q.startswith(("prize_best:", "share:", "ens2x5"))
            pred(name, v, lo, hi, unit, q + ("; DEV-selected" if sel else "") + extra.get(q, ""))
    for name, b, lo, hi in (("P5 share of prize on c10<=10", "c10<=10", 0.30, 1.0),
                            ("P6 share of prize on absent10_in100", "absent10_in100", 0.10, 0.25)):
        o = shares["slices_dev_ovr"].get(b)
        if registered:
            pred(name, shares["slices_dev"].get(b), lo, hi, "share of DEV nats/word",
                 "" if o is None else f"overlap-removed share {o:.4f}")
    if preds:
        with open(out / ("predictions.csv" if registered else "v2_predictions.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, list(preds[0]))
            w.writeheader()
            w.writerows(preds)
    for p in preds:
        print(p)
    plot(rows, pack, c, out)


def plot(rows: dict, pack: Path, c: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    pts = []
    ep = [(float(k.split("_e")[1].replace("p", ".")), rows[k]["dev_npw"]) for k in rows
          if k.startswith("10M_") and rows[k]["dev_npw"]]
    if ep:
        ep.sort()
        ax[0].plot([x for x, _ in ep], [y for _, y in ep], "o-", label="10M words")
        pts += [("10M", x, y) for x, y in ep]
    for k in ("100M_e1", "100M_e10", "30M_e3p333", "30M_e10"):
        if k in rows and rows[k]["dev_npw"]:
            x = float(k.split("_e")[1].replace("p", "."))
            ax[0].plot([x], [rows[k]["dev_npw"]], "s", label=k)
            pts.append((k.split("_")[0], x, rows[k]["dev_npw"]))
    ax[0].set(xscale="log", xlabel="epochs", ylabel="DEV nats/word (registered protocol)", title="epochs")
    ax[0].axvline(10, color="gray", lw=0.6, ls=":")
    ax[0].legend(fontsize=7)
    d10 = [(m, rows[f"{m}M_e10"]["dev_npw"]) for m in (10, 30, 100) if f"{m}M_e10" in rows and rows[f"{m}M_e10"]["dev_npw"]]
    iso = [(m, rows[k]["dev_npw"]) for m, k in ((10, "10M_e10"), (30, "30M_e3p333"), (100, "100M_e1"))
           if k in rows and rows[k]["dev_npw"]]
    if d10:
        ax[1].plot(*zip(*d10), "o-", label="10 epochs")
    if iso:
        ax[1].plot(*zip(*iso), "s--", label="100M words exposure")
    ax[1].set(xscale="log", xlabel="unique words (M)", ylabel="DEV nats/word", title="data")
    ax[1].legend(fontsize=7)
    for k in ("10M_e10", "100M_e10", "100M_e1"):
        f = pack / c.get(k, "") / "curves.csv"
        if k in rows and f.exists():
            r = list(csv.DictReader(open(f)))
            ax[2].plot([float(x["words_seen"]) / 1e6 for x in r], [float(x["gap_nats_per_word"]) for x in r], label=k)
    ax[2].set(xscale="log", xlabel="words seen (M)", ylabel="DEV-sub minus TRAIN-sub nats/word", title="memorisation gap")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "p2_curves.png", dpi=120)
    with open(out / "p2_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["data", "epochs", "dev_nats_per_word"])
        w.writerows(pts)


if __name__ == "__main__":
    main()
