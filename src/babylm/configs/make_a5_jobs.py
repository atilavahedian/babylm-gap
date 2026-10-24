"""Job list for attack A5 (late-epoch rare-word loss focusing; docs/babylm/attacks/A5_rare_word_focus.md).

    python -m babylm.configs.make_a5_jobs --base babylm/configs/jobs_base_r1_S.json base_S_wd0p4_lr0p5_r1_ve0_tied \
        [--only a5 a5_null] [--label diagnostic]

Takes the base winner's job (its exact flags) and writes jobs_a5_<size>.json with up to three runs, same seed and
data order as the base:
  <stem>_a5          arm 1: --focus_plan 7:0.25 (focus mask = words seen <= 63 times in train_10M + 3 words after)
  <stem>_a5_null     N1 (binding twin): the same plan with --focus_null (random targets, same expected kept share)
  <stem>_a5_a0       N2 (mechanism): focus mask without the aftermath words (--focus_mask focus_a0)
--only picks a subset; the experimenter's 03:01Z decision is the two-run set a5 a5_null, labelled "diagnostic" (critic
ref-0267: a diagnostic run set cannot be promoted). Before the runs the mask(s) must exist on the machine (not a
train job, so not in this list):
  python -m babylm.data.focus_mask --split train_10M
  python -m babylm.data.focus_mask --split train_10M --after 0 --tag focus_a0      (only for a5_a0)
The base run itself is the control (already scored). Verdict per the card: full-DEV nats/word (score_trial.sh) plus
per-word dumps by band (babylm/analysis/perword.py keeps pw_devfull_final.npy); the exposure counters in result.json
must equal the base's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = "7:0.25"
RUNS = ("a5", "a5_null", "a5_a0")
MASK_CMD = "python -m babylm.data.focus_mask --split train_10M"
MASK_A0_CMD = MASK_CMD + " --after 0 --tag focus_a0"


def a5_jobs(base_job: dict, plan: str = PLAN, only: list[str] | None = None) -> list[dict]:
    args = list(base_job["args"])
    tag_i = args.index("--tag")
    stem = args[tag_i + 1]

    def mk(suffix: str, extra: list[str]) -> dict:
        a = args[:tag_i] + extra + ["--tag", f"{stem}_{suffix}"] + args[tag_i + 2:]
        return {**base_job, "name": f"{stem}_{suffix}", "args": a}

    jobs = {"a5": mk("a5", ["--focus_plan", plan]),
            "a5_null": mk("a5_null", ["--focus_plan", plan, "--focus_null"]),
            "a5_a0": mk("a5_a0", ["--focus_plan", plan, "--focus_mask", "focus_a0"])}
    return [jobs[s] for s in (only or RUNS)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", nargs=2, metavar=("JOBFILE", "NAME"), required=True,
                    help="job list and job name of the base winner (control)")
    ap.add_argument("--plan", default=PLAN, help="focus plan E:BETA (the card fixes 7:0.25)")
    ap.add_argument("--only", nargs="+", choices=RUNS, default=None, help="subset of the runs (default: all three)")
    ap.add_argument("--label", default="attack", help="'diagnostic' = the run set cannot be promoted (critic ref-0267)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.base[0]).read_text())
    base = next(j for j in spec["jobs"] if j["name"] == a.base[1])
    size = base["args"][base["args"].index("--size") + 1]
    jobs = a5_jobs(base, a.plan, a.only)
    names = [j["name"] for j in jobs]
    masks = [MASK_CMD] + ([MASK_A0_CMD] if any(n.endswith("_a5_a0") for n in names) else [])
    out = Path(a.out) if a.out else HERE / f"jobs_a5_{size}.json"
    out.write_text(json.dumps({
        "description": f"A5 {a.label} at the base winner {a.base[1]}: focus plan {a.plan}; runs {' '.join(names)}",
        "label": a.label,
        "note": "Build the mask(s) first: " + " ; ".join(masks) + " . After each run: "
                "babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir> and "
                "python -m babylm.analysis.perword babylm/runs/<name> --epochs_only (keeps pw_devfull_final.npy).",
        "control": a.base[1], "jobs": jobs}, indent=1) + "\n")
    print("wrote", out, names)


if __name__ == "__main__":
    main()
