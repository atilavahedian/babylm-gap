"""GPU job lists for the experimenter's Phase 2 grids.

    python -m babylm.configs.make_gpu_jobs                      -> babylm/configs/jobs_gpu_p2.json (v1, port recipe)
    python -m babylm.configs.make_gpu_jobs --v2 --commit SHA --set muon_wd=0.4 --set lr_scale=1 \
        [--set model=baseline_reg --append "--extra dropout=0.1"]
                                                                -> babylm/configs/jobs_gpu_p2v2.json and
                                                                   babylm/configs/gcp/lead_p2v2{a,b}.json

v1: the frozen port args (--save --resume auto) with A100-sized safety timeouts. One list for all grid VMs; each VM
picks its runs with `run_jobs jobs_gpu_p2.json --only NAME` and runs babylm.analysis.post_runs after each.
port_S_10M_e10_s0 doubles as the grid's 10M x 10 cell (jobs_b's grid_S_10M_e10_s0 is not run).
VM a = P1/P2 cells, VM b = seeds 1-2 and the rest of the S grid, VM m = size M (P1 at M).

v2 (critic ref-0052: the port recipe overfits after epoch 2, so the bottleneck ranking uses a grid at the base
sweep's tuned recipe): the same cells named v2_S_<data>_e<epochs>_s<seed>, every run = frozen port args with the
--set FLAG=VALUE overrides, replaced in place (e.g. the s1 winner's muon_wd and lr_scale, or model=baseline_reg),
plus the raw --append args (one quoted string for repeatable flags, e.g. "--extra dropout=0.1"). The 100M arms use the recipe tuned on 10M (if anything this understates the
prize; the WD 0 100M cells from v1 are the other reference). VM v2a = 100M x 10 (trial 1 of the critic's Y-a
reference, our recipe at 100M) and 100M x 1; VM v2b = the 10M epoch curve with a full anneal at each length (the
critic's ref-0423 screening tiebreak), a second 10M x 5 seed (V8: a 2-member ensemble at the same 100M-word exposure
as one 10-epoch run) and the 30M cells. 10M x 10 seeds 0-2 are not here: seed 0 is the round-1 winner run itself and
seeds 1-2 are the floor seeds (jobs_base_seeds_*), with identical args. The P7 kNN cells are deferred to the ceiling
map; the v1 port model they used is no longer the base.
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from babylm.configs.make_base_jobs import override
from babylm.configs.make_jobs import job

HERE = Path(__file__).resolve().parent
VMS = {
    "a": [("port_S_10M_e10_s0", "S", "train_10M", "10", 0, 1.5), ("grid_S_100M_e10_s0", "S", "train_100M", "10", 0, 4),
          ("grid_S_100M_e1_s0", "S", "train_100M", "1", 0, 1)],
    "b": [("port_S_10M_e10_s1", "S", "train_10M", "10", 1, 1.5), ("port_S_10M_e10_s2", "S", "train_10M", "10", 2, 1.5),
          ("grid_S_10M_e1_s0", "S", "train_10M", "1", 0, 0.5), ("grid_S_10M_e2_s0", "S", "train_10M", "2", 0, 0.5),
          ("grid_S_10M_e4_s0", "S", "train_10M", "4", 0, 1), ("grid_S_30M_e3p333_s0", "S", "train_30M", "3.333", 0, 1.5),
          ("grid_S_30M_e10_s0", "S", "train_30M", "10", 0, 3), ("grid_S_10M_e20_s0", "S", "train_10M", "20", 0, 2)],
    "m": [("size_M_10M_e10_s0", "M", "train_10M", "10", 0, 1.5), ("size_M_100M_e1_s0", "M", "train_100M", "1", 0, 1.5),
          ("size_M_100M_e10_s0", "M", "train_100M", "10", 0, 6)],
}
# v2 cells: (data split, epochs, seed, timeout h); timeouts from the measured A100 rate (364k tokens/s at size S)
V2 = {
    "a": [("100M", "10", 0, 3.0), ("100M", "1", 0, 0.75)],
    "b": [("10M", "1", 0, 0.5), ("10M", "2", 0, 0.5), ("10M", "4", 0, 0.5), ("10M", "5", 0, 0.5),
          ("10M", "5", 1, 0.5), ("30M", "3.333", 0, 0.75), ("30M", "10", 0, 1.5)],
}
V2_MAXDUR = {"a": "3h", "b": "2h30m"}
PY = "/opt/p2venv/bin/python"
SETUP = ("set -e; python3 -m venv /opt/p2venv; /opt/p2venv/bin/pip install -q torch==2.7.0 numpy tokenizers==0.21.1 "
         "transformers==4.51.3 safetensors matplotlib pytest; /opt/p2venv/bin/python -m babylm.data.prepare --download "
         "--splits train_10M,train_100M,dev; /opt/p2venv/bin/python -m pytest -q -x babylm/tests; if [ -n "
         "\"${RESTORE_URI:-}\" ]; then mkdir -p babylm/runs; gsutil -m -q rsync -r \"$RESTORE_URI\" babylm/runs; fi")
KNN_GRID = ("--device cuda --temps 1,3,10,30,100 "
            "--lambdas 0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8")


def v2_name(data: str, epochs: str, seed: int) -> str:
    return f"v2_S_{data}_e{epochs.replace('.', 'p')}_s{seed}"


def v2_jobs(sets: list[tuple[str, str]], append: list[str]) -> list[dict]:
    out = []
    for vm, cells in V2.items():
        for data, epochs, seed, th in cells:
            j = job(v2_name(data, epochs, seed), "S", f"train_{data}", epochs, seed, th)
            args = j["args"]
            for flag, value in sets:
                args = override(args, flag, value)
            tag_at = args.index("--tag")
            j["args"] = args[:tag_at] + list(append) + args[tag_at:]
            out.append({**j, "vm": f"v2{vm}"})
    return out


def knn_cmd(run: str) -> str:
    out = f"babylm/runs/{run}/knn_p7"
    slices = (f"{PY} -m babylm.analysis.p2_slices --runs lm={out}:lm knn={out}:knn lmtemp={out}:lmtemp "
              f"knntemp={out}:knntemp --pairs lm-knn lm-lmtemp lmtemp-knntemp --out {out}/slices")
    return (f"{PY} -m babylm.analysis.knn_diag --ckpt babylm/runs/{run}/checkpoints/chck_100M.pt --out {out} "
            f"{KNN_GRID} > babylm/runs/{run}/knn_p7.log 2>&1; rc=$?; {slices} >> babylm/runs/{run}/knn_p7.log 2>&1; "
            f"mkdir -p babylm/results_pack/{run}; cp -r {out} babylm/results_pack/{run}/ 2>/dev/null; exit $rc")


def gcp_files(commit: str, jobs: list[dict]) -> dict[str, dict]:
    pt = " ".join(v2_name(d, e, s) for cs in V2.values() for d, e, s, _ in cs if d == "10M")  # seed + 2x5 ensembles
    files = {}
    for vm in V2:
        cmds = []
        for j in (j for j in jobs if j["vm"] == f"v2{vm}"):
            cmds.append({"name": j["name"], "timeout_s": j["timeout_s"],
                         "cmd": f"{PY} -m babylm.harness.run_jobs babylm/configs/jobs_gpu_p2v2.json --only {j['name']}; "
                                f"rc=$?; {PY} -m babylm.analysis.post_runs --pt_runs {pt}; exit $rc"})
        files[f"lead_p2v2{vm}"] = {
            "name": f"lead_p2v2{vm}", "commit": commit, "paths": ["babylm", "sprint1"], "setup": SETUP,
            "env": {"RESTORE_URI": ""}, "jobs": cmds, "outputs": ["babylm/runs", "babylm/results_pack"],
            "keep_checkpoints": True, "continue_on_error": True, "suggested_max_run_duration": V2_MAXDUR[vm],
            "note": "A100 spot. Pool: promotion (Y-a trial 1)." if vm == "a" else "A100 spot. Pool: floor.",
        }
    return files


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--commit", default="", help="v2: the code commit the GCP job files pin")
    ap.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE", help="v2: e.g. muon_wd=0.4")
    ap.add_argument("--append", default="", help="v2: one quoted string of raw args added to every run (before --tag)")
    a = ap.parse_args(argv)
    if not a.v2:
        jobs = [{**job(*spec), "vm": vm} for vm, specs in VMS.items() for spec in specs]
        out = {"description": "Phase 2 grid on A100 spot VMs (a: P1/P2 cells; b: seeds 1-2 + rest of the S grid; "
                              "m: size M)",
               "note": "Per VM: run_jobs jobs_gpu_p2.json --only <name>, then babylm.analysis.post_runs after each run.",
               "jobs": jobs}
        (HERE / "jobs_gpu_p2.json").write_text(json.dumps(out, indent=1) + "\n")
        print(HERE / "jobs_gpu_p2.json", len(jobs), "jobs")
        return
    if not a.commit or not a.set:
        raise SystemExit("--v2 needs --commit and at least one --set (the tuned recipe)")
    sets = [("--" + s.split("=", 1)[0].lstrip("-"), s.split("=", 1)[1]) for s in a.set]
    append = shlex.split(a.append)
    jobs = v2_jobs(sets, append)
    recipe = " ".join(f"{f} {v}" for f, v in sets) + (" " + " ".join(append) if append else "")
    out = {"description": f"Phase 2 v2 grid at the tuned recipe ({recipe}); size S; critic ref-0052",
           "note": "Per VM: run_jobs jobs_gpu_p2v2.json --only <name>, then babylm.analysis.post_runs after each.",
           "jobs": jobs}
    (HERE / "jobs_gpu_p2v2.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, f in gcp_files(a.commit, jobs).items():
        (HERE / "gcp" / f"{name}.json").write_text(json.dumps(f, indent=1) + "\n")
    print(HERE / "jobs_gpu_p2v2.json", len(jobs), "jobs; gcp files", ", ".join(gcp_files(a.commit, jobs)))


if __name__ == "__main__":
    main()
