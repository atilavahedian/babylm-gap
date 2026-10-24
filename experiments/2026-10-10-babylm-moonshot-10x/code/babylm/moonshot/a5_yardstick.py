"""A5 (critic ref-0354): the 10x yardstick Y and data multipliers x = 10 ** (g / Y), all on calibrated_t.

    python -m babylm.moonshot.a5_yardstick --config A5.json --out a5_out.json

A5.json:
    {"root": DIR (optional; dump paths relative to it),
     "y": {"pool": [DUMP...], "t100": DUMP or [DUMP...], "t100_label": "..."},
     "gains": {NAME: {"arm": [[DUMP, DUMP], ...], "ref": [DUMP...], "ref_sd": [DUMP...], "labels": ["..."]}}}

Every side is read through the pinned instrument babylm/tools/calib_control.py on v2 twin dumps (dev_bpw_dump --twin
--temps): a mixture's members are tempered at one shared T and then mixed (token-level probability average), as in
ceiling_report --calib. Legs full and guarded, each on all of DEV ("all") and overlap-removed ("ovr": words of DEV
lines found verbatim in train_100M dropped, critic ref-0260, ceiling_report.keep_mask), passed to calib_control as its
token inclusion mask with the word counts restricted to it. g and Y are both taken on the same view.

Y = mean(pool) - mean(T100); "t100" is one dump or a list of seeds. With several seeds the 95% half-width is Welch's:
t(Welch-Satterthwaite df) sqrt(sd_p^2/n_p + sd_t^2/n_t) (critic ref-0189); with one, T100's unknown seed SD is the pool's
(sd_p sqrt(1/n_p + 1), df n_p - 1), labelled "Y single-seed (assumed SD)". A single T100 dump without temps has no
calibrated_t: Y is then bracketed by T100's tempering gain lying between 0 and the pool's, labelled provisional.
g = mean(ref) - mean(arm) over the arm's mixtures (n_a) and the reference runs (n_r); SE = sqrt(sd_a^2/n_a +
sd_r^2/n_r), a single-seed reference taking sd_r = max(SD of "ref_sd" (the pool at its size), SD of the Y pool (S E6))
(critic ref-0361); 95% t interval at df 2,
carried through 10 ** (g / Y). Any "N x" sentence uses headline = the smaller lower bound of the two views.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from babylm.moonshot import ceiling_report as R

LEGS, VIEWS = ("full", "guarded"), ("all", "ovr")
DF_GAIN = 2
LABELS = ["DEV-loss data-equivalent multiplier (log-linear)"]
T100_UNTUNED = "T100 untuned, 1 seed: x likely overstated"


def calib_tool():
    spec = importlib.util.spec_from_file_location("calib_control", R.CALIB_TOOL)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    return cc


class Reader:
    """Reads a side (one dump, or a mixture of several) on both views and legs: {view: {leg: {raw, cal_t, t}}}."""

    def __init__(self, cc, keep_word: np.ndarray | None = None):
        self.cc, self.cache = cc, {}
        self.S, self.tok = cc.shared(), cc.tokenizer()
        if keep_word is None:
            d, files = cc.dev_files()
            keep_word = R.keep_mask(files, Path(d))
        assert keep_word.size == self.S["W"], f"keep mask {keep_word.size} words vs DEV {self.S['W']}"
        self.keep = keep_word

    def __call__(self, members: list[str]) -> dict:
        key = tuple(members)
        if key not in self.cache:
            self.cache[key] = self._read(members)
        return self.cache[key]

    def _read(self, members: list[str]) -> dict:
        cc, S = self.cc, self.S
        loaded = [cc.load_v2(p, S, self.tok) for p in members]
        _, ids, words, ok, drop, *_ = loaded[0]
        has_t = all(z[6] is not None for z in loaded)
        mix = (lambda cols: R.ensemble_tokens(cols)) if len(loaded) > 1 else (lambda cols: cols[0])
        x = mix([z[0] for z in loaded])
        xt = temps = None
        if has_t:
            temps = loaded[0][7]
            assert all(z[7] == temps for z in loaded), "members' temps differ"
            xt = np.stack([mix([z[6][:, j].astype(np.float64) for z in loaded]) for j in range(len(temps))], 1)
        out = {}
        for view in VIEWS:
            s, m = (S, ok) if view == "all" else (
                dict(S, W=int(self.keep.sum()), bad=S["bad"] & self.keep), ok & self.keep[words])
            raw = cc.calibrate(x, ids, words, m, drop, s)
            ct = cc.calibrate_t(x, xt, temps, ids, words, m, drop, s) if has_t else None
            out[view] = {leg: {"raw": raw[leg]["raw"], "cal_t": ct[leg]["calibrated_t"] if ct else None,
                               "t": [ct[leg]["t_even"], ct[leg]["t_odd"], ct[leg]["t_edge"]] if ct else None}
                         for leg in LEGS}
        return out


def _mean_sd(v: list[float]) -> tuple[float, float]:
    a = np.asarray(v, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else float("nan")


def yardstick(pool: list[dict], t100: list[dict], view: str, leg: str) -> dict:
    ct = [p[view][leg]["cal_t"] for p in pool]
    if None in ct:
        raise SystemExit("A5 Y is cal_t only: every pool dump needs --temps (critic ref-0354)")
    n = len(ct)
    m, sd = _mean_sd(ct)
    tv = [t[view][leg]["cal_t"] for t in t100]
    if len(t100) > 1:
        if None in tv:
            raise SystemExit("A5 Y is cal_t only: every T100 seed needs --temps")
        mt, sdt = _mean_sd(tv)
        a, b = sd ** 2 / n, sdt ** 2 / len(tv)
        df = (a + b) ** 2 / (a ** 2 / (n - 1) + b ** 2 / (len(tv) - 1))  # Welch (critic ref-0189)
        hw = float(stats.t.ppf(0.975, df) * math.sqrt(a + b))
        ys, prov, lab = [m - mt], False, f"Y Welch over {n} pool vs {len(tv)} T100 seeds, df {df:.1f}"
    else:
        hw = float(stats.t.ppf(0.975, n - 1) * sd * math.sqrt(1 / n + 1)) if n > 1 else float("nan")
        lab = f"Y single-seed (assumed SD), df {n - 1}"
        if tv[0] is not None:
            ys, prov = [m - tv[0]], False
        else:  # T100 without temps: its tempering gain lies in [0, the pool's mean gain]
            gain = _mean_sd([p[view][leg]["raw"] for p in pool])[0] - m
            ys, prov = sorted([m - t100[0][view][leg]["raw"], m - t100[0][view][leg]["raw"] + gain]), True
    return {"pool_mean": m, "pool_sd": sd, "n_pool": n, "t100": tv, "Y": ys, "provisional": prov,
            "Y_ci95_halfwidth": hw, "Y_ci_label": lab, "two_x_nats": [math.log10(2) * y for y in ys]}


def gain(arm: list[dict], ref: list[dict], ref_sd: list[dict], view: str, leg: str, sd_floor: float = 0.0) -> dict:
    a = [r[view][leg]["cal_t"] for r in arm]
    rv = [r[view][leg]["cal_t"] for r in ref]
    if None in a or None in rv:
        raise SystemExit("A5 gains are cal_t only: every arm and reference dump needs --temps")
    ma, sda = _mean_sd(a)
    mr, sdr = _mean_sd(rv)
    if len(rv) == 1:  # single-seed menu reference: max(its size's pool SD, the S E6 pool SD) (critic ref-0361)
        sdr = max(_mean_sd([r[view][leg]["cal_t"] for r in ref_sd])[1] if len(ref_sd) > 1 else 0.0, sd_floor)
    se = math.sqrt((sda ** 2 if len(a) > 1 else 0.0) / len(a) + sdr ** 2 / len(rv))
    g, h = mr - ma, float(stats.t.ppf(0.975, DF_GAIN) * se)
    return {"arm_mean": ma, "arm_sd": sda, "n_arm": len(a), "ref_mean": mr, "ref_sd": sdr, "n_ref": len(rv),
            "g": g, "se": se, "g_ci95": [g - h, g + h], "df": DF_GAIN}


def multiplier(g: dict, y: dict) -> dict:
    lo, hi = g["g_ci95"]
    return {"x": sorted(10 ** (g["g"] / v) for v in y["Y"]),
            "x_ci95": [10 ** (lo / max(y["Y"])) if lo >= 0 else 10 ** (lo / min(y["Y"])), 10 ** (hi / min(y["Y"]))]}


def report(cfg: dict, read) -> dict:
    root = cfg.get("root")
    if root:  # dump paths relative to "root"
        read = (lambda r: lambda members: r([str(Path(root) / m) for m in members]))(read)
    yc = cfg["y"]
    t100s = yc["t100"] if isinstance(yc["t100"], list) else [yc["t100"]]
    pool, t100 = [read([p]) for p in yc["pool"]], [read([p]) for p in t100s]
    base_labels = LABELS + [yc.get("t100_label", T100_UNTUNED)]
    out = {"Y": {f"{v}/{l}": yardstick(pool, t100, v, l) for v in VIEWS for l in LEGS}, "gains": {}}
    for name, gc in cfg.get("gains", {}).items():
        arm = [read(list(m)) for m in gc["arm"]]
        ref = [read([p]) for p in gc["ref"]]
        ref_sd = [read([p]) for p in gc.get("ref_sd", [])]
        rows = {}
        for v in VIEWS:
            for l in LEGS:
                g = gain(arm, ref, ref_sd, v, l, out["Y"][f"{v}/{l}"]["pool_sd"])
                rows[f"{v}/{l}"] = g | multiplier(g, out["Y"][f"{v}/{l}"])
        headline = {l: min(rows[f"{v}/{l}"]["x_ci95"][0] for v in VIEWS) for l in LEGS}
        out["gains"][name] = {"rows": rows, "headline_x_lower": headline,
                              "labels": base_labels + list(gc.get("labels", []))}
    if any(y["provisional"] for y in out["Y"].values()):
        out["labels"] = ["provisional: T100 cal_t bracketed (no temps dump)"]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    read = Reader(calib_tool())
    res = report(cfg, read)
    res["reads"] = {"+".join(k): v for k, v in read.cache.items()}
    res["ovr_words_dropped"], res["config"] = int((~read.keep).sum()), cfg
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, y in res["Y"].items():
        print("Y", k, [round(v, 4) for v in y["Y"]], "+-", round(y["Y_ci95_halfwidth"], 4), "(provisional)" * y["provisional"])
    for name, gr in res["gains"].items():
        for k, r in gr["rows"].items():
            print(name, k, "g", round(r["g"], 4), [round(v, 4) for v in r["g_ci95"]],
                  "x", [round(v, 3) for v in r["x"]], "ci", [round(v, 3) for v in r["x_ci95"]])
        print(name, "headline x lower bound", {l: round(v, 3) for l, v in gr["headline_x_lower"].items()}, gr["labels"])


if __name__ == "__main__":
    main()
