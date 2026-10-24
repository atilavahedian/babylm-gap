"""Job lists for attack A1 (temporal self-distillation; docs/babylm/attacks/A1_temporal_self_distill.md).

    # stage 1: the base re-run (--sd_mode none) and the five arms on a base job's exact argv
    python -m babylm.configs.make_a1_jobs --base_jobs babylm/configs/jobs_base_r1_S.json \
        --base_name base_S_wd2p0_lr0p5_r1_ewd0p005 --out babylm/configs/jobs_a1_S.json
    # stage 2, after stage 1: the four nulls, calibrated on the best arm (the card's rule picks it)
    python -m babylm.configs.make_a1_jobs --base_jobs ... --base_name ... --nulls babylm/runs/<best arm> \
        --out babylm/configs/jobs_a1_nulls_S.json

Every job is the base job's argv with its --tag/--out replaced and the SD flags appended (seed and data order are the
base's). A base that sets an SD flag itself is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = {
    "A1a": ["--sd_mode", "temporal", "--sd_alpha", "0.5", "--sd_decay", "0.6", "--sd_start_epoch", "1"],
    "A1b": ["--sd_mode", "temporal", "--sd_alpha", "0.5", "--sd_decay", "0.85", "--sd_start_epoch", "1"],
    "A1c": ["--sd_mode", "temporal", "--sd_alpha", "1.0", "--sd_decay", "0.85", "--sd_start_epoch", "1"],
    "A1d": ["--sd_mode", "temporal", "--sd_alpha", "0.5", "--sd_decay", "0.0", "--sd_start_epoch", "1"],
    "A1e": ["--sd_mode", "temporal", "--sd_alpha", "0.5", "--sd_decay", "0.6", "--sd_start_epoch", "1",
            "--sd_write_until", "5"],
}
NULLS = {"A1nullH": ("uniform", "H"), "A1nullUH": ("unigram", "H"), "A1nullO": ("uniform", "off"),
         "A1nullUO": ("unigram", "off")}


def strip_flag(args: list[str], flag: str) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def derive(base: dict, name: str, extra: list[str]) -> dict:
    if any(a.startswith("--sd_") for a in base["args"]):
        raise SystemExit(f"base job {base['name']} already sets an --sd_ flag")
    args = strip_flag(strip_flag(base["args"], "--tag"), "--out")
    timeout = int(base.get("timeout_s", 10800) * 1.5)  # the SD path is slower than the fused port step
    return {"name": name, "module": base.get("module", "babylm.harness.train"),
            "args": args + extra + ["--tag", name], "timeout_s": timeout}


def stage1(base: dict, prefix: str) -> list[dict]:
    jobs = [derive(base, f"{prefix}sd_base", ["--sd_mode", "none"])]
    jobs += [derive(base, f"{prefix}{k}", v) for k, v in ARMS.items()]
    return jobs


def stage2(base: dict, prefix: str, best_run: Path, data_dir: str | None = None) -> list[dict]:
    from babylm.harness.self_distill import null_eps_from_run
    jobs = []
    for k, (null, match) in NULLS.items():
        r = null_eps_from_run(best_run, null=null, match=match, data_dir=data_dir)
        jobs.append(derive(base, f"{prefix}{k}", ["--sd_mode", r["mode"], "--sd_alpha", str(r["alpha"]),
                                                   "--sd_start_epoch", str(r["start_epoch"]),
                                                   "--sd_smooth_eps", f"{r['eps']:.6f}"]))
    return jobs


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_jobs", required=True)
    ap.add_argument("--base_name", required=True)
    ap.add_argument("--prefix", default="", help="job name prefix, e.g. 'p0_' for a provisional base")
    ap.add_argument("--nulls", default=None, metavar="BEST_RUN_DIR", help="stage 2: the four nulls from this arm")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.base_jobs).read_text())
    base = next((j for j in spec["jobs"] if j["name"] == a.base_name), None)
    if base is None:
        raise SystemExit(f"no job {a.base_name} in {a.base_jobs}")
    jobs = stage2(base, a.prefix, Path(a.nulls), a.data_dir) if a.nulls else stage1(base, a.prefix)
    out = {"description": f"attack A1 {'nulls' if a.nulls else 'screen'} on {a.base_name} ({a.base_jobs})",
           "note": "card docs/babylm/attacks/A1_temporal_self_distill.md; after each run: post_runs / dev_bpw.py on "
                   "the final checkpoint (the kill metric); report tokens_per_s",
           "jobs": jobs}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(jobs)} jobs -> {a.out}")
    return out


if __name__ == "__main__":
    main()
