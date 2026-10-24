"""CPU test of glue_probs_run.py + glue_ensemble.py (a few minutes).

    python test_glue_probs.py EVAL2025_DIR HF_TEMPLATE_DIR TOKENIZER_DIR

Builds a tiny random Hinton export (vocab = the tokenizer's) and fine-tunes it on a small RTE subset for 1 epoch:
(1) via the official `python -m evaluation_pipeline.finetune.run` and (2) via glue_probs_run.py with the same args and
seed: results.txt and predictions.json must be byte-identical, and valid_probs.json must reproduce results.txt.
Then a second seed, and glue_ensemble.py over the two seeds on all 7 task names (the other 6 tasks are copies of RTE's
outputs under their names, which exercises the per-task loop and metric lists; mnli gets accuracy only) must write
results.txt files equal to calculate_metrics on the mean probabilities.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

pipe, tpl, tokd = map(Path, sys.argv[1:4])
here = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
work = tmp / "eval2025"; work.mkdir()
(work / "evaluation_pipeline").symlink_to(pipe / "evaluation_pipeline")
data = work / "data"; data.mkdir()
for split, n in (("train", 48), ("valid", 40)):
    lines = (pipe / f"evaluation_data/full_eval/glue_filtered/rte.{split}.jsonl").read_text().splitlines()[:n]
    (data / f"rte.{split}.jsonl").write_text("\n".join(lines) + "\n")

model = tmp / "tiny"; model.mkdir()
for f in ("configuration_hinton.py", "modeling_hinton.py"):
    shutil.copy(tpl / f, model / f)
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
    if (tokd / f).exists():
        shutil.copy(tokd / f, model / f)
V = len(AutoTokenizer.from_pretrained(model))
cfg = {"model_type": "hinton_gpt", "architectures": ["HintonForCausalLM"],
       "auto_map": {"AutoConfig": "configuration_hinton.HintonConfig", "AutoModel": "modeling_hinton.HintonModel",
                    "AutoModelForCausalLM": "modeling_hinton.HintonForCausalLM"},
       "vocab_size": V, "n_layer": 2, "n_head": 2, "n_embd": 32, "seq_len": 128, "automodel_pool": "last"}
(model / "config.json").write_text(json.dumps(cfg))
torch.manual_seed(0)
m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model, trust_remote_code=True), trust_remote_code=True)
m.save_pretrained(model)
for f in ("configuration_hinton.py", "modeling_hinton.py"):
    shutil.copy(tpl / f, model / f)
(model / "config.json").write_text(json.dumps(cfg))


def args(results, seed):
    return ["--model_name_or_path", str(model), "--train_data", "data/rte.train.jsonl", "--valid_data",
            "data/rte.valid.jsonl", "--predict_data", "data/rte.valid.jsonl", "--task", "rte", "--num_labels", "2",
            "--batch_size", "16", "--learning_rate", "3e-4", "--num_epochs", "1", "--sequence_length", "128",
            "--results_dir", results, "--metrics", "accuracy", "f1", "mcc", "--metric_for_valid", "accuracy",
            "--seed", str(seed)]


env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
run = lambda cmd: subprocess.run(cmd, cwd=work, env=env, check=True, capture_output=True, text=True)
run([sys.executable, "-m", "evaluation_pipeline.finetune.run"] + args("res_official", 42))
run([sys.executable, str(here / "glue_probs_run.py")] + args("res_probs", 42))
run([sys.executable, str(here / "glue_probs_run.py")] + args("res_probs43", 43))
o, p = work / "res_official/tiny/main/finetune/rte", work / "res_probs/tiny/main/finetune/rte"
for f in ("results.txt", "predictions.json"):
    assert (o / f).read_bytes() == (p / f).read_bytes(), f"{f} differs with the probability dump"
v = json.loads((p / "valid_probs.json").read_text())
assert v["task"] == "rte" and v["seed"] == 42 and len(v["labels"]) == 40, (v["task"], v["seed"], len(v["labels"]))

members = []
for name in ("res_probs", "res_probs43"):
    ft = work / name / "tiny/main/finetune"
    for t in ("boolq", "multirc", "wsc", "mrpc", "qqp", "mnli"):
        shutil.copytree(ft / "rte", ft / t)
    vm = json.loads((ft / "mnli/valid_probs.json").read_text())   # mnli: accuracy only, as the official script
    (ft / "mnli/results.txt").write_text(f"accuracy: {vm['metrics']['accuracy']}")
    members.append(str(ft))
r = subprocess.run([sys.executable, str(here / "glue_ensemble.py"), "--pipeline", str(pipe), "--out",
                    str(work / "res_ens/ens/main/finetune")] + members, env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr
sys.path.insert(0, str(pipe))
from evaluation_pipeline.finetune.trainer import Trainer  # noqa: E402
pa = torch.tensor(json.loads((work / "res_probs/tiny/main/finetune/rte/valid_probs.json").read_text())["probs"], dtype=torch.float64)
pb = torch.tensor(json.loads((work / "res_probs43/tiny/main/finetune/rte/valid_probs.json").read_text())["probs"], dtype=torch.float64)
y = torch.tensor(v["labels"])
want = Trainer.calculate_metrics((pa + pb) / 2, y, ["accuracy", "f1", "mcc"])
got = dict(ln.split(": ") for ln in (work / "res_ens/ens/main/finetune/rte/results.txt").read_text().splitlines())
assert all(abs(float(got[k]) - float(want[k])) < 1e-12 for k in want), (got, want)
assert set(json.loads((work / "res_ens/ens/main/finetune/ensemble.json").read_text())["tasks"]) == \
    {"boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"}
bad = work / "res_probs43/tiny/main/finetune/wsc/results.txt"   # a dump that does not match results.txt must fail
bad.write_text("accuracy: 0.0\nf1: 0.0\nmcc: 0.0")
r = subprocess.run([sys.executable, str(here / "glue_ensemble.py"), "--pipeline", str(pipe), "--out",
                    str(work / "res_ens2/ens/main/finetune")] + members, env=env, capture_output=True, text=True)
assert r.returncode == 1 and "does not reproduce" in r.stdout, r.stdout
print("glue probs OK: official outputs byte-identical with the dump; ensemble = official metric on mean probabilities;",
      "mismatched dump refused")
