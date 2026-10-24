"""Accuracy summary of an official fast zero-shot run: python -m babylm.harness.fast_summary EVAL_DIR [--json OUT]

EVAL_DIR is what score_trial.sh copies to RUN/score/eval (the pipeline's results/<name> tree). Reads every
zero_shot/<backend>/<task>/<dataset>/best_temperature_report.txt ('### AVERAGE' then the value, as the ruler's
babylm/eval/collect.py) and reading/report.txt (eye-tracking and self-paced-reading scores). Prints and returns
{"<task>/<dataset>": value, ...}; predictions.json files are not needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def avg_from_report(path: str) -> float | None:
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("### AVERAGE") and i + 1 < len(lines):
            try:
                return float(lines[i + 1].strip())
            except ValueError:
                return None
    return None


def summarize(eval_dir: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for rep in sorted(glob.glob(os.path.join(eval_dir, "**", "zero_shot", "*", "*", "*", "best_temperature_report.txt"),
                                recursive=True)):
        parts = rep.split(os.sep)
        out[f"{parts[-3]}/{parts[-2]}"] = avg_from_report(rep)
    for rep in sorted(glob.glob(os.path.join(eval_dir, "**", "zero_shot", "*", "reading", "report.txt"), recursive=True)):
        for m in re.finditer(r"^(EYE TRACKING|SELF-PACED READING) SCORE:\s*([-\d.]+)", open(rep).read(), re.M):
            out["reading/" + m.group(1).lower().replace(" ", "_").replace("-", "_")] = float(m.group(2))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    s = summarize(a.eval_dir)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=1)
    for k, v in s.items():
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
