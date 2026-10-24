"""Job lists for the R4.4 reference menu (card PE section "FLOPs and the R4.4 plan"; critic ref-0135(4), ref-0318,
ref-0079): plain base-family models at >= F x the base's FLOPs/token, trained on train_10M at their own best epoch count
E* from {2, 4, 5, 6, 8, 10}, WD re-tuned, every run a counted trial. An ensemble-shaped arm at F x (legal split K = 2:
2.000; PE3: 2.463; K = 3: 3.000) must beat the lowest-loss of the base E* pool and every eligible entry:

    python -m babylm.moonshot.make_menu_jobs BASE_JOBS_JSON:JOB_NAME --f 2.0 --epochs 4,5,6 [--wd 0.4] [--seeds 0]
        [--entries deep,28L384d] [--out FILE]

Entries (FLOPs/token = 6 (L 12 d^2 + V d) + 6 L T d, the card's formula; x base 193,462,272):
  F 2.000: deep 28L/384d (2.073), wide 12L/592d (2.092; 8 heads of 74)
  F 2.463: deep 35L/384d (2.543), wide 12L/672d (2.614; 7 heads of 96)
  F 3.000: deep 42L/384d (3.012), wide 12L/736d (3.074; 23 heads of 32)
  F 4.146 (the 28L/384d legal split K = 2: 2 x 2.073; experimenter 20:32Z, proposed to the critic): deep 60L/384d (4.220),
          wide 12L/880d (4.241; 11 heads of 80), balanced 36L/512d (4.358; 8 heads of 64)
  M 12L/768d (3.317) belongs to every F's menu. --f F lists the entries registered at F and at every larger F (a smaller
  F inherits them; "--entries" narrows it). The run name records entry, E and WD: ms_menu_<L>L<d>d_e<E>_wd<WD>, plus
  _s<seed> for seeds other than 0 (claim replicates; seed 0 keeps the registered menu's names).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babylm.moonshot.make_ceiling_jobs import base_job, override

V, T = 16384, 1024
BASE_FLOPS = 193_462_272
MENU = {  # registered F -> [(name, n_layer, n_embd, n_head)]
    2.0: [("deep", 28, 384, 6), ("wide", 12, 592, 8)],
    2.463: [("deep", 35, 384, 6), ("wide", 12, 672, 7)],
    3.0: [("deep", 42, 384, 6), ("wide", 12, 736, 23)],
    4.146: [("deep", 60, 384, 6), ("wide", 12, 880, 11), ("balanced", 36, 512, 8)],
}
SIZE_M = ("M", 12, 768, 12)
E_GRID = (2, 4, 5, 6, 8, 10)
LABEL = "R4.4 reference menu (plain base family on train_10M, own E*, counted trial)"


def flops(n_layer: int, d: int) -> int:
    return 6 * (n_layer * 12 * d * d + V * d) + 6 * n_layer * T * d


def entries(f: float) -> list[tuple[str, int, int, int]]:
    out = [e for g, es in sorted(MENU.items()) if g >= f - 1e-9 for e in es] + [SIZE_M]
    return [e for e in out if flops(e[1], e[2]) >= f * BASE_FLOPS]


def build(spec: str, f: float, epochs: list[float], wd: float | None = None, only: set[str] | None = None,
          seeds: list[int] | None = None) -> dict:
    b = base_job(spec)
    if bad := [e for e in epochs if e not in E_GRID]:
        raise SystemExit(f"epochs {bad} are outside the registered grid {E_GRID}")
    a0 = b["args"]
    wd = float(a0[a0.index("--muon_wd") + 1]) if wd is None else wd
    jobs = []
    for name, L, d, H in entries(f):
        tag = f"{L}L{d}d"
        if only and name not in only and tag not in only:
            continue
        x = flops(L, d) / BASE_FLOPS
        for s in seeds or [0]:
            for e in epochs:
                run = f"ms_menu_{tag}_e{e:g}_wd{wd:g}".replace(".", "p") + (f"_s{s}" if s else "")
                a = list(a0)
                for flag, v in {"--split": "train_10M", "--seed": str(s), "--epochs": f"{e:g}", "--tag": run,
                                "--n_layer": str(L), "--n_embd": str(d), "--n_head": str(H),
                                "--muon_wd": f"{wd:g}"}.items():
                    a = override(a, flag, v)
                jobs.append({"name": run, "module": b["module"], "args": a, "flops_x_base": round(x, 3),
                             "label": f"{LABEL}: {name} {tag} ({x:.3f}x FLOPs), E {e:g}, WD {wd:g}, seed {s}",
                             "timeout_s": int(600 + 240 * e * x)})
    return {"description": f"R4.4 menu for F = {f:g} on base job {spec} (train_10M, seeds {seeds or [0]}; pool: per the "
                           "ticket)",
            "note": "Solo runs (claims run solo). After each: babylm/harness/score_trial.sh babylm/runs/<name> <eval "
                    "workdir> and python -m babylm.moonshot.dev_bpw_dump --twin --temps 0.9,0.95,1.05,1.1,1.15,1.2 "
                    "into .../_dumps_v2/<name>.npz (the calibrated control applies to the menu too). Every run is a "
                    "counted trial; frozen and hashed before the arm's first promoted score (ref-0086).",
            "f": f, "epochs": epochs, "wd": wd, "jobs": jobs}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="BASE_JOBS_JSON:JOB_NAME")
    ap.add_argument("--f", type=float, required=True)
    ap.add_argument("--epochs", default="4,5,6")
    ap.add_argument("--wd", type=float, default=None, help="muon WD (default: the base's)")
    ap.add_argument("--entries", default=None, help="comma list of entry names or tags (deep, wide, M, 28L384d, ...)")
    ap.add_argument("--seeds", default="0", help="comma list of seeds (non-zero seeds get an _s<seed> suffix)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = build(a.base, a.f, [float(x) for x in a.epochs.split(",")], a.wd,
              set(a.entries.split(",")) if a.entries else None, [int(x) for x in a.seeds.split(",")])
    txt = json.dumps(d, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(txt if not a.out else f"wrote {a.out}: {len(d['jobs'])} jobs")


if __name__ == "__main__":
    main()
