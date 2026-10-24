"""Job list for the moonshot's ceiling-map diagnostics (docs/babylm/moonshot/ceiling_map.md), built from the base job
the floor seeds use, so the C3 seeds match it exactly:

    python -m babylm.moonshot.make_ceiling_jobs BASE_JOBS_JSON:JOB_NAME [--seed1_run NAME] [--out FILE]

C3: the base job at seeds 3..7 (seeds 0-2 are the base's own run and the two floor seeds).
C1-self:   seed 0, KD alpha 1 / tau 1 from the floor seed-1 run's final checkpoint (babylm/runs/<seed1_run>/...;
           default <job name>_s1, as make_base_jobs.stage_seeds names it); its job carries "after": <seed1_run>.
C1-oracle: seed 0, KD alpha 1 / tau 1 from T100 (p2 grid_S_100M_e10_s0 chck_1000M; port recipe, 1B words seen;
           out of track). The executor stages it at T100_PATH and checks T100_SHA256 before the run.
Every job keeps --save/--resume auto and gets its own --tag; labels go in "label". Diagnostics only, never verdicts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

T100_SRC = "shared/babylm/p2/ckpt/grid_S_100M_e10_s0/chck_1000M.pt"
T100_PATH = "babylm/teachers/T100_grid_S_100M_e10_s0_chck_1000M.pt"
T100_SHA256 = "2bf1cea08ce2253dc54406fce6b509f02f0f6300ea1c003b904340cafb7e99db"
C3_SEEDS = (3, 4, 5, 6, 7)


def override(args: list[str], flag: str, value: str) -> list[str]:
    a = list(args)
    if flag in a:
        a[a.index(flag) + 1] = value
    else:
        a += [flag, value]
    return a


def base_job(spec: str) -> dict:
    path, name = spec.rsplit(":", 1)
    jobs = json.loads(Path(path).read_text())["jobs"]
    hit = [j for j in jobs if j["name"] == name]
    if len(hit) != 1:
        raise SystemExit(f"{spec}: expected one job named {name!r}, found {len(hit)}")
    return hit[0]


def variant(base: dict, name: str, label: str, seed: int, extra: list[str], timeout_mult: float = 1.0) -> dict:
    a = override(base["args"], "--seed", str(seed))
    a = override(a, "--tag", name)
    return {"name": name, "module": base["module"], "args": a + extra, "label": label,
            "timeout_s": int(base.get("timeout_s", 10800) * timeout_mult)}


def build(spec: str, seed1_run: str | None = None) -> dict:
    b = base_job(spec)
    short = b["name"]
    seed1_run = seed1_run or f"{short}_s1"
    jobs = [variant(b, f"ms_c3_{short}_s{s}", "C3 member (K x exposure, out of track)", s, []) for s in C3_SEEDS]
    kd = ["--kd_alpha", "1.0", "--kd_temp", "1.0"]
    self_ck = f"babylm/runs/{seed1_run}/checkpoints/chck_100M.pt"
    jobs.append(variant(b, f"ms_c1self_{short}", "C1-self (teacher = 10M-trained base seed 1; 2x exposure, "
                        "diagnostic)", 0, ["--kd_teacher", self_ck, *kd], 1.6))
    jobs[-1]["after"] = seed1_run
    jobs.append(variant(b, f"ms_c1oracle_{short}", "C1-oracle (oracle teacher, 100M words; out of track)", 0,
                        ["--kd_teacher", T100_PATH, *kd], 1.6))
    return {"description": f"moonshot ceiling map on base job {spec}: C3 seeds {list(C3_SEEDS)}, C1-self, C1-oracle "
                           "(size S, train_10M x 10, diagnostics only; pool: moonshot)",
            "note": (f"Before C1-oracle: copy {T100_SRC} to {T100_PATH} and check sha256 {T100_SHA256}. "
                     f"C1-self needs {self_ck} (the floor seed-1 run) to exist first. After each run: "
                     "babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>; keep pt/pt_devfull_final.npy."),
            "base_job": spec, "t100": {"src": T100_SRC, "path": T100_PATH, "sha256": T100_SHA256},
            "jobs": jobs}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="BASE_JOBS_JSON:JOB_NAME")
    ap.add_argument("--seed1_run", default=None, help="run name of the floor seed-1 job (default <job>_s1)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = build(a.base, a.seed1_run)
    txt = json.dumps(d, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(txt if not a.out else f"wrote {a.out}: {len(d['jobs'])} jobs")


if __name__ == "__main__":
    main()
