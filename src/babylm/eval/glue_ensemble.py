"""Average GLUE class probabilities over fine-tuned members (critic 44552922).

    python glue_ensemble.py --pipeline EVAL2025_DIR --out RESULTS_DIR/NAME/main/finetune MEMBER_FT_DIR [MEMBER_FT_DIR ...]

Each MEMBER_FT_DIR is an official results tree <results>/<model>/main/finetune whose 7 task dirs hold the official
results.txt plus valid_probs.json (glue_probs_run.py). A K=2 mixture's members are its two models, each fine-tuned
once; a single model's matched ensemble is the same model fine-tuned with two seeds. Per task: (1) each member's
valid_probs.json must reproduce that member's official results.txt through the official Trainer.calculate_metrics
(proves the dump is the official final evaluation), (2) labels must match across members, (3) the mean of the member
probabilities is scored with the same official function and metric list. Writes <out>/<task>/results.txt in the
official format (so babylm/eval/collect.py reads it like any model) and <out>/ensemble.json (members, per-member and
ensemble metrics). Any check failing -> exit 1, nothing written for that task.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

TASK_METRICS = {"boolq": ["accuracy", "f1", "mcc"], "multirc": ["accuracy", "f1", "mcc"], "rte": ["accuracy", "f1", "mcc"],
                "wsc": ["accuracy", "f1", "mcc"], "mrpc": ["accuracy", "f1", "mcc"], "qqp": ["accuracy", "f1", "mcc"],
                "mnli": ["accuracy"]}   # = the --metrics of each task in the official eval_finetuning.sh


def read_results(p: Path) -> dict:
    return {k.strip(): float(v) for k, v in (ln.split(":", 1) for ln in p.read_text().splitlines() if ":" in ln)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("members", nargs="+")
    a = ap.parse_args()
    if len(a.members) < 2:
        sys.exit("need at least 2 members")
    sys.path.insert(0, a.pipeline)
    from evaluation_pipeline.finetune.trainer import Trainer
    out, summary, ok = Path(a.out), {"members": a.members, "tasks": {}}, True
    for task, mets in TASK_METRICS.items():
        probs, labels, per = [], None, []
        for m in a.members:
            d = Path(m) / task
            if not (d / "valid_probs.json").is_file() or not (d / "results.txt").is_file():
                print(f"[ens] {task}: missing valid_probs.json/results.txt in {d}"); ok = False; break
            v = json.loads((d / "valid_probs.json").read_text())
            p, y = torch.tensor(v["probs"], dtype=torch.float64), torch.tensor(v["labels"])
            mine = Trainer.calculate_metrics(torch.tensor(v["logits"]), y, mets)   # the member's own official argmax
            off = read_results(d / "results.txt")
            if any(abs(float(mine[k]) - off[k]) > 1e-9 for k in mets):
                print(f"[ens] {task}: {d} valid_probs.json does not reproduce results.txt: {mine} vs {off}"); ok = False; break
            if labels is not None and not torch.equal(labels, y):
                print(f"[ens] {task}: label order differs between members"); ok = False; break
            labels = y; probs.append(p); per.append({k: float(mine[k]) for k in mets})
        else:
            ens = Trainer.calculate_metrics(torch.stack(probs).mean(0), labels, mets)
            ens = {k: float(v) for k, v in ens.items()}
            (out / task).mkdir(parents=True, exist_ok=True)
            (out / task / "results.txt").write_text("\n".join(f"{k}: {v}" for k, v in ens.items()))
            summary["tasks"][task] = {"members": per, "ensemble": ens, "n": int(labels.numel())}
            print(f"[ens] {task}: members {per} -> ensemble {ens}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ensemble.json").write_text(json.dumps(summary, indent=1))
    return 0 if ok and len(summary["tasks"]) == len(TASK_METRICS) else 1


if __name__ == "__main__":
    sys.exit(main())
