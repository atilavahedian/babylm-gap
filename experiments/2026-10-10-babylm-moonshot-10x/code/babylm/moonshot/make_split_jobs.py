"""Job lists for the legal-split family (experimenter 07:10Z; the readout is the moonshot's): ensembles whose members together
see at most 10 epochs of train_10M (<= 100M words, R0), built from the base job so every member shares its recipe:

    python -m babylm.moonshot.make_split_jobs BASE_JOBS_JSON:JOB_NAME --plan split --k 3 [--out FILE]
    python -m babylm.moonshot.make_split_jobs BASE_JOBS_JSON:JOB_NAME --plan branch --trunk 4 --k 2 [--out FILE]

split:  K independent runs at seeds 0..K-1, 10/K epochs each, the base's own schedule (flat, then its cooldown_frac).
        K = 2 is the experimenter's v2_S_10M_e5_s0 / e5_s1.
branch: a trunk at the base's flat LR (cooldown_frac 0) for T epochs, saved; K children start from its resume.pt
        (weights and optimizer moments, harness --init_from), each with a new data seed (101, 102, ...) and
        (10 - T) / K epochs under its own cooldown (--child_cooldown of its steps; default all), no momentum warm-up.
        Exposure T + K x (10 - T) / K = 10 epochs, the trunk counted once (each child records its words in
        levers.init_from).
Readout: python -m babylm.moonshot.ceiling_report c3 <member v2 dumps> --ref <10-epoch base seed dumps> --guard --calib
(vs_ref = the base mean minus the all-member mixture). A submission would be one HF model computing the members'
probability average (the PE export path), at K x the forward FLOPs. Known ideas, never called new: fixed-budget
independent ensembles; branch-train-merge (Li et al. 2022); snapshot ensembles (Huang et al. 2017).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babylm.moonshot.make_ceiling_jobs import base_job, override

TOTAL_EPOCHS = 10.0
CHILD_DATA_SEED = 100
LABEL = "legal split (members' summed exposure <= 10 epochs of train_10M = 100M words)"


def timeout(epochs: float) -> int:
    return int(600 + 240 * epochs)  # the base's 10 epochs train in ~8 min on an A100 at size S; scoring rides on top


def fmt(epochs: float) -> str:
    return f"{epochs:.6f}".rstrip("0").rstrip(".")


def job(b: dict, name: str, label: str, epochs: float, flags: dict[str, str], extra: list[str] | None = None) -> dict:
    a = list(b["args"])
    for flag, v in {"--epochs": fmt(epochs), "--tag": name, **flags}.items():
        a = override(a, flag, v)
    return {"name": name, "module": b["module"], "args": a + (extra or []), "label": f"{label}; {LABEL}",
            "timeout_s": timeout(epochs)}


def build(spec: str, plan: str, k: int, trunk: float = 4.0, child_cooldown: float = 1.0) -> dict:
    b = base_job(spec)
    if "--split" in b["args"] and b["args"][b["args"].index("--split") + 1] != "train_10M":
        raise SystemExit("the legal split is defined on train_10M")
    if k < 2:
        raise SystemExit("--k must be >= 2")
    if plan == "split":
        e = TOTAL_EPOCHS / k
        jobs = [job(b, f"ms_ls_k{k}_s{s}", f"split member {s + 1}/{k}: {fmt(e)} epochs, seed {s}", e,
                    {"--seed": str(s)}) for s in range(k)]
    elif plan == "branch":
        if not 0 < trunk < TOTAL_EPOCHS:
            raise SystemExit("--trunk must be inside (0, 10) epochs")
        e = (TOTAL_EPOCHS - trunk) / k
        tname = f"ms_ls_br{fmt(trunk)}k{k}_trunk"
        jobs = [job(b, tname, f"branch trunk: {fmt(trunk)} epochs at flat LR, seed 0", trunk,
                    {"--seed": "0", "--cooldown_frac": "0"}, ["--save"] if "--save" not in b["args"] else [])]
        for c in range(1, k + 1):
            jobs.append({**job(b, f"ms_ls_br{fmt(trunk)}k{k}_c{c}",
                               f"branch child {c}/{k}: {fmt(e)} epochs from the trunk, data seed {CHILD_DATA_SEED + c}",
                               e, {"--seed": "0", "--data_seed": str(CHILD_DATA_SEED + c),
                                   "--cooldown_frac": f"{child_cooldown:g}", "--muon_momentum_warmup_steps": "0"},
                               ["--init_from", f"babylm/runs/{tname}/resume.pt"]), "after": [tname]})
    else:
        raise SystemExit(f"unknown plan {plan!r}")
    epochs = sum(float(j["args"][j["args"].index("--epochs") + 1]) for j in jobs)
    assert epochs <= TOTAL_EPOCHS + 1e-9, epochs
    return {"description": f"legal split '{plan}' K={k} on base job {spec} (train_10M; summed exposure "
                           f"{epochs:.6f} epochs <= 10; pool: per the experimenter)",
            "note": "Solo runs (claims run solo, critic ref-0001). After each member: babylm/harness/score_trial.sh "
                    "babylm/runs/<name> <eval workdir>, the twin json and python -m babylm.moonshot.dev_bpw_dump --twin "
                    "--check into shared/babylm/moonshot/dumps_v2/legal_split/<name>/dev_v2.npz (with "
                    "dev_bpw_twin.json beside it). " + ("Children run after the trunk and read its resume.pt."
                                                        if plan == "branch" else ""),
            "base_job": spec, "plan": plan, "k": k, "summed_epochs": epochs, "jobs": jobs}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="BASE_JOBS_JSON:JOB_NAME")
    ap.add_argument("--plan", choices=["split", "branch"], required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--trunk", type=float, default=4.0, help="branch: trunk epochs")
    ap.add_argument("--child_cooldown", type=float, default=1.0, help="branch: cooldown share of each child's steps")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = build(a.base, a.plan, a.k, a.trunk, a.child_cooldown)
    txt = json.dumps(d, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(txt if not a.out else f"wrote {a.out}: {len(d['jobs'])} jobs, {d['summed_epochs']:.6f} epochs")


if __name__ == "__main__":
    main()
