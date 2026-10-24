"""Verify an HF export against the training model and the official BabyLM loader (CPU, tiny, 1 thread).

    python babylm/recipes/sprint1_port/verify_export.py --ckpt <run>/ckpt/final.pt --scratch <dir>

Checks (all printed and written to <scratch>/verify_export.json):
  1. logits: training module (vendored sprint1/models/baseline.py, eval mode) vs AutoModelForCausalLM.from_pretrained(
     export, trust_remote_code=True), max |diff| on dev windows of seq_len tokens and on BOS-prefixed sentences;
     and the training forward's scored loss vs CE from the HF logits;
  2. right padding: logits of real tokens unchanged when a batch is right-padded with the pad id (official collate);
  3. tokenizer through AutoProcessor (the official loader): BOS = <|endoftext|> prepended with offset (0, 0);
  4. official dataloader: per-sentence phrase log-prob from evaluation_pipeline.sentence_zero_shot (dataset.get_dataloader
     + the compute_causal_results arithmetic) equals sum_{t>=1} log p_train(tok_t | tok_<t) from the training module;
  5. the official run.py (python -m evaluation_pipeline.sentence_zero_shot.run --backend causal --task blimp) on a
     5-pair BLiMP-format file written to <scratch> (NOT the real evaluation data) completes and writes its report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
EVAL_ROOT = Path("~/babylm/eval2025")

BLIMP_SMOKE = [
    ("The cats annoy Tim.", "The cats annoys Tim."),
    ("Who should Derek hug after shocking Richard?", "Who should Derek hug Richard after shocking?"),
    ("Many girls insulted themselves.", "Many girls insulted herself."),
    ("These books were dropped.", "This books were dropped."),
    ("Rachel was not ever happy.", "Rachel was ever not happy."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--data_dir", default=str(HERE / "data"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    from export_hf import export
    exp = export(Path(a.ckpt), scratch / "hf_export")

    from sprint1.harness.model_api import ModelConfig
    from sprint1.models import baseline
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"])
    ref = baseline.build(cfg)
    ref.load_state_dict(ck["model"], strict=True)
    ref.eval()

    from transformers import AutoModelForCausalLM, AutoProcessor
    hf = AutoModelForCausalLM.from_pretrained(str(exp), trust_remote_code=True)
    hf.eval()
    report["hf_class"] = type(hf).__name__
    report["hf_n_params"] = sum(p.numel() for p in hf.parameters())
    report["ref_n_params"] = sum(p.numel() for p in ref.parameters())

    # 1. dev windows
    dev = np.fromfile(Path(a.data_dir) / "dev.bin", dtype=np.uint16)
    T = cfg.seq_len
    starts = np.array([0, len(dev) // 3, 2 * len(dev) // 3, len(dev) - T - 2])
    buf = torch.from_numpy(np.asarray(dev[starts[:, None] + np.arange(T + 1)], dtype=np.int64))
    x, y = buf[:, :-1], buf[:, 1:]
    with torch.no_grad():
        lr = ref.logits(ref.hidden(x))
        loss_ref, _ = ref(x, y)
        lh = hf(input_ids=x, attention_mask=torch.ones_like(x)).logits
    ce_hf = torch.nn.functional.cross_entropy(lh.reshape(-1, lh.size(-1)), y.reshape(-1)).item()
    report["dev_windows"] = {"shape": list(x.shape), "max_abs_logit_diff": (lr - lh).abs().max().item(),
                             "train_forward_loss": loss_ref.item(), "hf_ce": ce_hf,
                             "loss_abs_diff": abs(loss_ref.item() - ce_hf)}

    from transformers import AutoModel
    base = AutoModel.from_pretrained(str(exp), trust_remote_code=True).eval()
    with torch.no_grad():
        hb = base(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
        report["automodel_max_abs_hidden_diff"] = (hb - ref.hidden(x)).abs().max().item()
    del base

    # 3. tokenizer via the official loader's call
    proc = AutoProcessor.from_pretrained(str(exp), padding_side="right", trust_remote_code=True)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    enc = proc(text="The cats annoy Tim.", return_offsets_mapping=True)
    report["tokenizer"] = {"class": type(tok).__name__, "ids": enc["input_ids"], "offsets": enc["offset_mapping"],
                           "bos": tok.bos_token_id, "eos": tok.eos_token_id, "pad": tok.pad_token_id,
                           "padding_side": tok.padding_side,
                           "bos_prepended": enc["input_ids"][0] == 50256 and tuple(enc["offset_mapping"][0]) == (0, 0)}

    # 2. right padding
    sents = [s for pair in BLIMP_SMOKE for s in pair]
    ids = [torch.tensor(proc(text=s)["input_ids"]) for s in sents]
    pad = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=tok.pad_token_id)
    am = torch.nn.utils.rnn.pad_sequence([torch.ones_like(t) for t in ids], batch_first=True, padding_value=0)
    with torch.no_grad():
        lp = hf(input_ids=pad, attention_mask=am).logits
        diffs, ref_diffs = [], []
        for i, t in enumerate(ids):
            single = hf(input_ids=t[None]).logits[0]
            diffs.append((lp[i, :len(t)] - single).abs().max().item())
            ref_diffs.append((ref.logits(ref.hidden(t[None]))[0] - single).abs().max().item())
    report["right_padding_max_abs_diff"] = max(diffs)
    report["sentences_max_abs_logit_diff_vs_train_module"] = max(ref_diffs)

    # 4 + 5. official pipeline on a 5-pair BLiMP-format file
    bdir = scratch / "blimp_smoke"
    bdir.mkdir(exist_ok=True)
    with open(bdir / "smoke_items.jsonl", "w") as f:
        for i, (g, b) in enumerate(BLIMP_SMOKE):
            f.write(json.dumps({"sentence_good": g, "sentence_bad": b, "field": "smoke", "linguistics_term": "smoke",
                                "UID": "smoke_items", "pair_id": i}) + "\n")
    sys.path.insert(0, str(EVAL_ROOT))
    cwd = os.getcwd()
    os.chdir(EVAL_ROOT)
    try:
        from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader
        args = SimpleNamespace(backend="causal", model_path_or_name=str(exp), revision_name=None, data_path=bdir,
                               task="blimp", images_path=None, image_template=None, image_split=None,
                               full_sentence_scores=False, batch_size=64)
        dl = get_dataloader(args)
        worst = 0.0
        first_word_scored = True
        rows = []
        with torch.no_grad():
            for raw, sd, labels, metas, uids, images in dl:
                n_s = len([k for k in sd if k.endswith("attn_mask")])
                for si in range(n_s):
                    inp, tgt, pm = sd[f"sentence_{si}_inputs"], sd[f"sentence_{si}_targets"], sd[f"sentence_{si}_phrase_mask"]
                    lg = hf(input_ids=inp, attention_mask=sd[f"sentence_{si}_attn_mask"]).logits
                    lpp = torch.log_softmax(lg, -1).gather(-1, tgt[..., None])[..., 0]
                    pipe = (lpp * pm).sum(1)
                    for r in range(inp.size(0)):
                        s = raw[r]["sentences"][si]
                        t = torch.tensor(proc(text=s)["input_ids"])
                        lt = torch.log_softmax(ref.logits(ref.hidden(t[None]))[0, :-1], -1)
                        manual = lt.gather(-1, t[1:, None])[:, 0].sum()
                        worst = max(worst, abs(manual.item() - pipe[r].item()))
                        first_word_scored &= bool(pm[r, 0] == 1)
                        rows.append({"sentence": s, "pipeline_logprob": pipe[r].item(), "train_module_logprob": manual.item()})
        report["official_dataloader"] = {"max_abs_logprob_diff": worst, "first_token_after_bos_scored": first_word_scored,
                                         "examples": rows[:4]}
    finally:
        os.chdir(cwd)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run", "--data_path", str(bdir),
           "--task", "blimp", "--model_path_or_name", str(exp), "--backend", "causal",
           "--output_dir", str(scratch / "official_out"), "--batch_size", "8"]
    p = subprocess.run(cmd, cwd=EVAL_ROOT, env=env, capture_output=True, text=True, timeout=900)
    reports = list((scratch / "official_out").rglob("best_temperature_report.txt"))
    report["official_run"] = {"returncode": p.returncode, "report": reports[0].read_text() if reports else None,
                              "stderr_tail": p.stderr[-800:] if p.returncode else ""}
    (scratch / "verify_export.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
