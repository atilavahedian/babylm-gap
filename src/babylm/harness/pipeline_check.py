"""Check an exported model through the official BabyLM 2025 pipeline (evaluation-pipeline-2025 @ bf55c1131e53654c2a87418f5629c26959acd710).

    python -m babylm.harness.pipeline_check --export EXPORT_DIR --ckpt RUN/checkpoints/chck_XM.pt \
        --pipeline PIPELINE_REPO --blimp_dir .../fast_eval/blimp_fast [--n_batches 4] [--run_official]

1. Official loader equivalence: batches from evaluation_pipeline.sentence_zero_shot.dataset.get_dataloader (backend
   causal: AutoProcessor, BOS prepended, right padding, phrase mask) scored with the compute_causal_results arithmetic
   on the exported model must equal, sentence by sentence, sum_{t>=1} log p(tok_t | tok_<t) of [BOS] + tokens from
   the TRAINING module (sprint1.models.baseline) -- i.e. every word including the first is scored, nothing else.
2. --run_official: python -m evaluation_pipeline.sentence_zero_shot.run --backend causal --task blimp on blimp_dir
   (unmodified, as a subprocess in the pipeline repo) and print its accuracy report.
The pipeline repo is third-party code: it is imported only by this checker, never by the harness.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from babylm.data import corpus as C


@torch.no_grad()
def loader_equivalence(export: Path, ckpt: Path, pipeline: Path, blimp_dir: Path, n_batches: int) -> dict:
    sys.path.insert(0, str(pipeline))
    from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader  # noqa: E402
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    from babylm.harness.export_hf import harness_model

    args = SimpleNamespace(data_path=blimp_dir, task="blimp", model_path_or_name=str(export), backend="causal",
                           revision_name=None, images_path=None, image_split=None, image_template=None,
                           full_sentence_scores=False, batch_size=16, non_causal_batch_size=16)
    dl = get_dataloader(args)
    hf = AutoModelForCausalLM.from_pretrained(str(export), trust_remote_code=True).eval()
    ref, _ = harness_model(ckpt)
    tok = Tokenizer.from_file(str(C.TOKENIZER_PATH))
    max_diff, n, bos_ok = 0.0, 0, True
    for b, (raw, sd, labels, metas, uids, images) in enumerate(dl):
        if b >= n_batches:
            break
        for k in range(len([x for x in sd if x.endswith("attn_mask")])):
            p = f"sentence_{k}"
            logits = hf(input_ids=sd[f"{p}_inputs"], attention_mask=sd[f"{p}_attn_mask"]).logits
            lp = F.log_softmax(logits, dim=-1)
            tgt = torch.gather(lp, -1, sd[f"{p}_targets"].unsqueeze(-1)).squeeze(-1)
            official = (tgt * sd[f"{p}_phrase_mask"]).sum(1)
            bos_ok &= bool((sd[f"{p}_inputs"][:, 0] == C.BOS_ID).all())
            for i, r in enumerate(raw):
                s = r["sentences"][k]
                ids = torch.tensor([[C.BOS_ID] + tok.encode(s, add_special_tokens=False).ids])
                z = ref.logits(ref.hidden(ids))
                mine = F.log_softmax(z[0, :-1], -1).gather(-1, ids[0, 1:, None]).sum()
                max_diff = max(max_diff, abs(float(mine) - float(official[i])))
                n += 1
    return {"sentences": n, "max_abs_logprob_diff": max_diff, "bos_first": bos_ok}


def run_official(export: Path, pipeline: Path, blimp_dir: Path, out_dir: Path) -> str:
    cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", str(export),
           "--backend", "causal", "--task", "blimp", "--data_path", str(blimp_dir), "--save_predictions",
           "--output_dir", str(out_dir)]
    p = subprocess.run(cmd, cwd=pipeline, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"official run failed:\n{p.stderr[-3000:]}")
    rep = next(out_dir.rglob("best_temperature_report.txt"))
    return rep.read_text()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--blimp_dir", required=True)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--run_official", action="store_true")
    ap.add_argument("--out", default=None, help="results dir for --run_official (default EXPORT/../pipeline_results)")
    a = ap.parse_args(argv)
    export, pipeline, blimp = Path(a.export).resolve(), Path(a.pipeline).resolve(), Path(a.blimp_dir).resolve()
    res = loader_equivalence(export, Path(a.ckpt), pipeline, blimp, a.n_batches)
    print(json.dumps(res), flush=True)
    ok = res["max_abs_logprob_diff"] < 1e-3 and res["bos_first"]
    if a.run_official:
        out = Path(a.out).resolve() if a.out else export.parent / "pipeline_results"
        print(run_official(export, pipeline, blimp, out), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
