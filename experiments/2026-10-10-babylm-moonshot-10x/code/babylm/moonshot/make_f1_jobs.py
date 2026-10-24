"""Job list for card F1 stage 1 (closed-system Bayes by brute force; design agent card, draft PR #67; critic entry 04:40Z):
a "T1 diagnostic, not BabyLM-legal" on the mini-ruler, built from the base job so every run shares its recipe:

    python -m babylm.moonshot.make_f1_jobs BASE_JOBS_JSON:JOB_NAME [--out FILE]

All on train_1M x the base's epochs (python -m babylm.moonshot.mini_ruler builds the split):
  members  ms_f1_m{0..7}   the base at seeds 0-7 (m0 is base@1M; G1 = DEV(base@1M) - DEV(base@10M, 5.0367))
  student  ms_f1_stu       seed 0, KD alpha 0.5 / tau 1 from the 8 members' probability average (real text only)
  N-single ms_f1_nsingle   seed 0, KD from member 1 alone (soft targets without diversity)
  N-perm   ms_f1_nperm     the student's 8 teachers with their distributions permuted across positions (--kd_perm)
  F-M      ms_f1_FM        size M (12L/768d), seed 0, plain CE: the FLOP reference
The three KD runs carry "after": the eight member runs (they read babylm/runs/ms_f1_m<k>/checkpoints/chck_10M.pt).
Binding comparators (critic): the better of N-single, N-perm and F-M. Readouts: g(8) at 1M (ceiling_report c3 over the
member dumps), phi = (DEV(base@1M) - DEV(x)) / G1 for the 8-ensemble and the student. Kill if phi_stu < 0.20 or the
student is not ahead of every comparator by 2x the 1M seed floor (SD of the 8 members; a mini-ruler label only).
Exposure (label T1 diagnostic): 8 x 1M x 10 members + 1M x 10 student = 90M at the 1M scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babylm.moonshot.make_ceiling_jobs import base_job, override

MEMBERS = tuple(range(8))
KD = ["--kd_alpha", "0.5", "--kd_temp", "1.0"]
LABEL = "F1 stage 1, T1 diagnostic, not BabyLM-legal (mini-ruler proxy)"


def ck(name: str) -> str:
    return f"babylm/runs/{name}/checkpoints/chck_10M.pt"


def job(b: dict, name: str, seed: int, label: str, timeout: int, ov: dict | None = None,
        extra: list[str] | None = None) -> dict:
    a = list(b["args"])
    for flag, v in {"--split": "train_1M", "--seed": str(seed), "--tag": name, **(ov or {})}.items():
        a = override(a, flag, v)
    return {"name": name, "module": b["module"], "args": a + (extra or []), "label": f"{label}; {LABEL}",
            "timeout_s": timeout}


def build(spec: str) -> dict:
    b = base_job(spec)
    members = [f"ms_f1_m{s}" for s in MEMBERS]
    jobs = [job(b, n, s, "member" + (" (= base@1M)" if s == 0 else ""), 900) for n, s in zip(members, MEMBERS)]
    teachers = ",".join(ck(n) for n in members)
    for name, label, extra in (
            ("ms_f1_stu", "student: KD from the 8-member probability average", ["--kd_teacher", teachers, *KD]),
            ("ms_f1_nsingle", "null N-single: KD from member 1 alone", ["--kd_teacher", ck(members[1]), *KD]),
            ("ms_f1_nperm", "null N-perm: 8 teachers, distributions permuted across positions",
             ["--kd_teacher", teachers, "--kd_perm", *KD])):
        jobs.append({**job(b, name, 0, label, 1800, extra=extra), "after": members})
    jobs.append(job(b, "ms_f1_FM", 0, "F-M: size M FLOP reference (3.317x)", 1500, {"--size": "M"}))
    return {"description": f"card F1 stage 1 on base job {spec} (train_1M; T1 diagnostic, not BabyLM-legal; pool: "
                           "moonshot)",
            "note": "Needs the mini-ruler splits (python -m babylm.moonshot.mini_ruler). Run the 8 members (and F-M) "
                    "first; the three KD jobs read the members' chck_10M.pt. After each run: "
                    "babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>, and a dev_bpw_dump of every "
                    "member for g(8). Atila's line: <message> ('yes run', "
                    "04:36:52Z, answering the coordinator post that names this diagnostic).",
            "base_job": spec, "jobs": jobs}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="BASE_JOBS_JSON:JOB_NAME")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = build(a.base)
    txt = json.dumps(d, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(txt if not a.out else f"wrote {a.out}: {len(d['jobs'])} jobs")


if __name__ == "__main__":
    main()
