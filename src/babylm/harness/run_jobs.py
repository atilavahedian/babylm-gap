"""Run a babylm job list sequentially on this machine: python -m babylm.harness.run_jobs LIST.json [--only NAME ...]

Format (as sprint1/gcp/run_jobs.py): {"jobs": [{"name": ..., "module": "babylm.harness.train", "args": [...],
"timeout_s": ...}]}. Modules are allowlisted; argv lists never go through a shell; --out defaults to
babylm/runs/<name>; a job whose <out>/result.json exists is skipped (jobs pass --resume auto, so a rerun after a
crash or preemption continues from resume.pt). One line per job is appended to babylm/runs/jobs.jsonl.
Device-agnostic: --device {auto,cuda,mps,cpu} and --extra_args '...' are appended to every train job (later flags
win in argparse), e.g. on the Mac: --device mps (the train harness drops --compile on MPS).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from babylm.data.corpus import REPO_ROOT

ALLOWED = ("babylm.harness.train", "babylm.harness.export_hf", "babylm.harness.plot_curves")
RUNS = REPO_ROOT / "babylm" / "runs"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_list")
    ap.add_argument("--only", nargs="*", default=None, help="run only these job names")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"],
                    help="appended to every train job (e.g. mps on the Mac; --compile is ignored there)")
    ap.add_argument("--extra_args", default="", help="more train flags for every job, e.g. '--micro_batch_size 4'")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.job_list).read_text())
    rc_all = 0
    RUNS.mkdir(parents=True, exist_ok=True)
    for i, j in enumerate(spec["jobs"]):
        name = j["name"]
        if a.only and name not in a.only:
            continue
        module = j.get("module", "babylm.harness.train")
        if module not in ALLOWED:
            print(f"[jobs] refusing module {module}", flush=True)
            rc_all = 1
            continue
        args = [str(x) for x in j["args"]]
        if module == "babylm.harness.train":
            if a.device:
                args += ["--device", a.device]
            args += shlex.split(a.extra_args)
            if "--out" not in args:
                args += ["--out", str(RUNS / name)]
            out = Path(args[args.index("--out") + 1])
            if (out / "result.json").exists():
                print(f"[jobs] skip {name}: result exists", flush=True)
                continue
        cmd = [sys.executable, "-m", module, *args]
        print(f"[jobs] {i + 1}/{len(spec['jobs'])} {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, timeout=j.get("timeout_s")).returncode
        except subprocess.TimeoutExpired:
            rc = -9
        rc_all = rc_all or rc
        with open(RUNS / "jobs.jsonl", "a") as f:
            f.write(json.dumps({"name": name, "module": module, "args": args, "rc": rc, "seconds": time.time() - t0,
                                "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        print(f"[jobs] {name} rc={rc} in {time.time() - t0:.0f}s", flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
