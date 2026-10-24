"""Job lists for moonshot card PE (shared-trunk particle ensemble, babylm/models/particles.py), built from the base job
so every arm shares its recipe, seed and data order:

    python -m babylm.moonshot.make_particle_jobs BASE_JOBS_JSON:JOB_NAME --gate {2,3} [--reuse_f1] [--epochs E]
                                                 [--out FILE]

Gate 2 (mini-ruler proxy, train_1M x the base's epochs; labels "mini-ruler proxy", at most "killed (proxy)"):
  B (the attack code at K = 1: the base model bit for bit) at seeds 0, 1, 2 (I3 = the 3-seed ensemble, "K x exposure"),
  PE3 (K 3, Ls 4), NJ (same net, mixture objective), F-deep (35L/384d) and F-M (size M, 12L/768d) at the base's WD.
Gate 3 (10M screen, train_10M, seed 0): B, PE3, PE3w (0.5x Muon WD on the private blocks), NJ, F-deep, F-M.
--reuse_f1 (gate 2 only) drops the arms card F1 stage 1 already runs with identical arguments: B s0-s2 are ms_f1_m0-m2
(the baseline at the same seed, split and recipe; K = 1 is the base bit for bit, test_particles) and F-M is ms_f1_FM.
--epochs E (gate 3) runs every arm at E epochs instead of the base's 10: the base is past its optimum at e10 (experimenter
08:50Z: s(E) is flat at E 4-5), so PE is screened at the base's E* (critic ref-0079), names gaining _e<E>.
The bootstrap arms (PE3b / PE3r) are not built: the card gates them on Gate 2 and expects 0 to -0.01.
Run only after Gate 0 (ceiling-map C3: g(8) >= 0.10 and interpolated g(3) >= 0.06) and the critic's acceptance.
Every number is read with the registered scorer (score_trial.sh -> dev_bpw.py on the HF export); nothing is fitted on
DEV. Exposure: every arm is one connected network fed the base's words (critic 00:40Z Q1, Ls = 4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babylm.moonshot.make_ceiling_jobs import base_job, override

PE = ["--extra", "pe_k=3", "--extra", "pe_shared=4"]
ARMS = {  # name: (overrides, extra args, label, timeout_s at gate 2, timeout_s at gate 3 (about 2x the card's A100 est.))
    "B": ({"--model": "particles"}, [], "B: attack code at K = 1 (the base, bit for bit)", 900, 1200),
    "PE3": ({"--model": "particles"}, PE, "PE3: K 3 particles on a 4-block shared trunk (2.463x FLOPs)", 1500, 2700),
    "PE3w": ({"--model": "particles"}, PE + ["--extra", "pe_private_wd_scale=0.5"],
             "PE3w: PE3 with 0.5x Muon WD on the private blocks", 1500, 2700),
    "NJ": ({"--model": "particles"}, PE + ["--extra", "pe_objective=mixture"],
           "NJ: PE3's network trained on the mixture CE (binding R6 null)", 1500, 2700),
    "Fdeep": ({"--model": "baseline", "--n_layer": "35"}, [], "F-deep: 35L/384d at base WD (2.543x FLOPs)", 1500,
              2700),
    "FM": ({"--model": "baseline", "--size": "M"}, [], "F-M: size M 12L/768d at base WD (3.317x FLOPs)", 1800, 3600),
}
F1_REUSE = {"ms_pe_g2_B_s0": "ms_f1_m0", "ms_pe_g2_B_s1": "ms_f1_m1", "ms_pe_g2_B_s2": "ms_f1_m2",
            "ms_pe_g2_FM_s0": "ms_f1_FM"}
GATES = {2: (["B", "PE3", "NJ", "Fdeep", "FM"], (0, 1, 2), "train_1M"),
         3: (["B", "PE3", "PE3w", "NJ", "Fdeep", "FM"], (0,), "train_10M")}


def job(b: dict, gate: int, arm: str, seed: int, split: str, epochs: float | None = None) -> dict:
    ov, extra, label, t2, t3 = ARMS[arm]
    name = f"ms_pe_g{gate}_{arm}_s{seed}" + (f"_e{epochs:g}" if epochs else "")
    a = list(b["args"])
    for flag, v in {**ov, "--split": split, "--seed": str(seed), "--tag": name,
                    **({"--epochs": f"{epochs:g}"} if epochs else {})}.items():
        a = override(a, flag, v)
    lab = f"{label}; {'mini-ruler proxy' if gate == 2 else '10M screen'}" + (f"; {epochs:g} epochs (base E*)" if epochs else "")
    if arm == "B" and seed:
        lab += "; I3 member (K x exposure)"
    return {"name": name, "module": b["module"], "args": a + extra, "label": lab,
            "timeout_s": t2 if gate == 2 else t3}


def build(spec: str, gate: int, reuse_f1: bool = False, epochs: float | None = None) -> dict:
    if reuse_f1 and gate != 2:
        raise SystemExit("--reuse_f1 applies to gate 2 only")
    if epochs is not None and (gate != 3 or not 0 < epochs <= 10):
        raise SystemExit("--epochs applies to gate 3, within (0, 10]")
    b = base_job(spec)
    arms, seeds, split = GATES[gate]
    jobs = [job(b, gate, arm, s, split, epochs) for arm in arms for s in (seeds if arm == "B" else seeds[:1])]
    reused = {j["name"]: F1_REUSE[j["name"]] for j in jobs if reuse_f1 and j["name"] in F1_REUSE}
    jobs = [j for j in jobs if j["name"] not in reused]
    return {"description": f"moonshot card PE gate {gate} on base job {spec} ({split}; pool: moonshot)",
            "note": "Run only after Gate 0 (C3 g(8) >= 0.10, g(3) >= 0.06) and the critic's acceptance of card PE. "
                    + ("Gate 2 needs the mini-ruler splits: python -m babylm.moonshot.mini_ruler. " if gate == 2
                       else "") + "After each run: babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>; keep the "
                    "dev_bpw dumps (python -m babylm.moonshot.dev_bpw_dump) for J and rho.",
            **({"reused_from_f1": reused} if reused else {}), "base_job": spec, "gate": gate, "jobs": jobs}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="BASE_JOBS_JSON:JOB_NAME")
    ap.add_argument("--gate", type=int, choices=sorted(GATES), required=True)
    ap.add_argument("--reuse_f1", action="store_true", help="gate 2: read B s0-s2 and F-M from card F1's runs")
    ap.add_argument("--epochs", type=float, default=None, help="gate 3: every arm at E epochs (the base's E*)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = build(a.base, a.gate, a.reuse_f1, a.epochs)
    txt = json.dumps(d, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(txt if not a.out else f"wrote {a.out}: {len(d['jobs'])} jobs")


if __name__ == "__main__":
    main()
