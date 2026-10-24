"""Run the official GLUE finetune (evaluation_pipeline.finetune.run, unchanged) and also save the final validation
class probabilities, for ensembles (critic 44552922: a K=2 mixture's GLUE = members fine-tuned separately, member
probabilities averaged; every single reference gets the same 2-fine-tune-seed ensemble).

Called by the GLUE python shim (vm_job.sh) in place of `python -m evaluation_pipeline.finetune.run ARGS` when
GLUE_DUMP_PROBS=1, from the pipeline checkout as cwd. It wraps the static Trainer.calculate_metrics(logits, labels,
metrics) to record logits, softmax(logits) and labels on every call, then runs the official module as __main__. run.py's last
call is evaluate(evaluate_best_model=True), the one written to results.txt, so the record left at exit is exactly the
validation set the official score uses. Written to <results_dir>/<model>/<revision>/finetune/<task>/valid_probs.json
with the metrics recomputed from it (glue_ensemble.py checks they equal results.txt). Nothing else changes: no RNG is
drawn, the logits are copied, and results.txt / predictions.json are the official ones.
"""
import json
import os
import pathlib
import runpy
import sys

sys.path.insert(0, os.getcwd())   # the pipeline checkout (python -m would put cwd first)
from evaluation_pipeline.finetune import trainer as _tr  # noqa: E402

_orig = _tr.Trainer.calculate_metrics
_last = {}


def _calc(logits, labels, metrics_to_calculate):
    out = _orig(logits, labels, metrics_to_calculate)
    lg = logits.detach().float().cpu()
    _last.update(logits=lg.tolist(), probs=lg.softmax(-1).tolist(), labels=labels.detach().cpu().tolist(),
                 metrics={k: float(v) for k, v in out.items()})
    return out


_tr.Trainer.calculate_metrics = staticmethod(_calc)


def _arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


try:
    sys.argv = [os.path.join(os.getcwd(), "evaluation_pipeline", "finetune", "run.py")] + sys.argv[1:]
    runpy.run_module("evaluation_pipeline.finetune.run", run_name="__main__", alter_sys=True)
finally:
    if _last:
        task = _arg("--task")
        if task == "mnli":   # run.py renames mnli from the valid file's stem
            task = pathlib.Path(_arg("--valid_data") or _arg("--predict_data")).stem.split(".")[0]
        model = pathlib.Path(_arg("--model_name_or_path")).stem
        out = pathlib.Path(_arg("--results_dir", "results")) / model / (_arg("--revision_name") or "main") / "finetune" / task
        out.mkdir(parents=True, exist_ok=True)
        (out / "valid_probs.json").write_text(json.dumps({"task": task, "seed": int(_arg("--seed", 42)), **_last}))
