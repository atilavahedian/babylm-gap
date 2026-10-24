"""Eval-only pass over saved weights (train.py --save_weights) with the harness's own evaluate().

    python -m sprint1.harness.eval_weights --weights sprint1/runs/<run>/weights.pt --split dev --out <dir>
    python -m sprint1.harness.eval_weights --weights ${BUCKET}/runs/<run>/weights.pt --split final --out <dir>

The model is rebuilt from the saved cfg (sprint1.models.<model_name>.build), the state_dict is loaded strictly, and
the split (DEV or FINAL of --data_dir's fineweb_val_000000.bin, cut at N // 2 exactly as train.py; or with
--token_file every window of any shard file, e.g. TUNE-B for the narrowing check) is scored by
train.evaluate() with the same layout, eval batch default, bf16 autocast on CUDA and new/seen split. --split final
follows train.py's rule B10 (refused on a dirty or unknown commit) and writes FINAL_SPLIT_USED. A GCS weights path
is copied to <out>/weights.pt with gsutil first. Writes <out>/result.json (and per_token.npz with --save_per_token).
"""
from __future__ import annotations
GCS_SCHEME = "gs:" + "//"  # Google Cloud Storage URI scheme

import argparse
import contextlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from sprint1.harness import data as D
from sprint1.harness import train as TR
from sprint1.harness.model_api import ModelConfig


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="weights.pt from train.py --save_weights (local or GCS)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "final", "token_file"])
    ap.add_argument("--token_file", default=None,
                    help="score every window of this shard file (data.py header format, e.g. TUNE-B), whole file; "
                         "sets --split token_file. Bytes table from --data_dir")
    ap.add_argument("--data_dir", default=None, help="dir with fineweb_val_000000.bin (default HINTON_DATA_DIR)")
    ap.add_argument("--synthetic", action="store_true", help="the synthetic corpus's val split (CPU tests)")
    ap.add_argument("--final_eval_tokens", type=int, default=-1, help="-1 = the whole split")
    ap.add_argument("--eval_batch_size", type=int, default=None)
    ap.add_argument("--no_split", action="store_true")
    ap.add_argument("--save_per_token", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tag", default="")
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.token_file:
        args.split = "token_file"
    elif args.split == "token_file":
        raise SystemExit("--split token_file needs --token_file PATH")
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = TR.git_commit()
    if args.split == "final":  # B10, as train.py --final
        if commit.endswith("+dirty") or commit == "unknown":
            raise SystemExit(f"--split final refused: git commit is {commit!r}; FINAL may only be scored from a clean, "
                             f"committed tree. Commit first.")
        print("!" * 78 + "\n!!! eval_weights --split final: scoring the protected FINAL split.\n" + "!" * 78,
              file=sys.stderr, flush=True)
        (out / "FINAL_SPLIT_USED").write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                          "argv": sys.argv, "weights": args.weights}) + "\n")
    wpath = args.weights
    if wpath.startswith(GCS_SCHEME):
        local = out / "weights.pt"
        if subprocess.run(["gsutil", "-q", "cp", wpath, str(local)]).returncode != 0:
            raise SystemExit(f"could not copy {wpath}")
        wpath = str(local)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

    w = torch.load(wpath, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(w["cfg"])
    mod = importlib.import_module(f"sprint1.models.{w['model_name']}")
    model = mod.build(cfg)
    sd = {k: (v.float() if v.is_floating_point() and v.dtype == torch.bfloat16 else v) for k, v in w["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).train()  # evaluate() switches to eval and restores the mode

    if args.token_file:
        tokens = D.load_shard(args.token_file)
        token_bytes_np = (D.synthetic_bytes_table(cfg.vocab_size) if args.synthetic
                          else D.token_bytes_table(D.resolve_data_dir(args.data_dir), cfg.vocab_size))
    elif args.synthetic:
        corpus = D.load_corpus(None, synthetic=True, vocab_size=cfg.vocab_size)
        tokens, token_bytes_np = corpus.split(args.split), corpus.token_bytes
    else:
        ddir = D.resolve_data_dir(args.data_dir)
        val = D.load_shard(ddir / D.VAL_NAME)
        h = len(val) // 2
        tokens = val[:h] if args.split == "dev" else val[h:]
        token_bytes_np = D.token_bytes_table(ddir, cfg.vocab_size)
    token_bytes = torch.as_tensor(token_bytes_np, dtype=torch.int64, device=device)
    eval_bs = args.eval_batch_size or TR.EVAL_BATCH_DEFAULT["cuda" if device.type == "cuda" else "cpu"]
    fe_tokens = None if args.final_eval_tokens is None or args.final_eval_tokens < 0 else args.final_eval_tokens
    fp = D.slice_fingerprint(tokens, cfg.seq_len, fe_tokens, args.split)
    pt_rec = TR.PerTokenRecorder(fp["n_tokens"]) if args.save_per_token else None
    r = TR.evaluate(model, model, tokens, token_bytes, cfg, fe_tokens, eval_bs, device, amp, split=not args.no_split,
                    per_token=pt_rec)
    result = {"status": "ok", "kind": "eval_weights", "tag": args.tag, "weights": args.weights,
              "model": w["model_name"], "size": w.get("size"), "seed": w.get("seed"), "run_id": w.get("run_id"),
              "train_git_commit": w.get("git_commit"), "train_val_loss_nats": w.get("val_loss_nats"),
              "weights_dtype": w.get("weights_dtype"), "tokens": w.get("tokens"), "steps": w.get("step"),
              "git_commit": commit, "eval_split": args.split, "eval_fingerprint": fp,
              "data": args.token_file or ("synthetic" if args.synthetic else str(D.resolve_data_dir(args.data_dir))),
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "precision": "bf16-autocast" if device.type == "cuda" else "fp32", "wallclock_s": time.time() - t0,
              **r}
    TR.write_json(out / "result.json", result)
    if pt_rec is not None:
        pt_rec.save(out, {"run_id": w.get("run_id"), "model": w["model_name"], "split": args.split,
                          "seq_len": cfg.seq_len, "vocab_size": cfg.vocab_size, "synthetic": bool(args.synthetic),
                          "val_loss_nats": r["val_loss_nats"], "eval_tokens": r["eval_tokens"],
                          "eval_fingerprint": fp, "git_commit": commit})
    print(f"[eval_weights] {w['model_name']} {args.split} val_loss {r['val_loss_nats']:.4f} bpb {r['val_bpb']:.4f} "
          f"-> {out / 'result.json'}", flush=True)
    return result


if __name__ == "__main__":
    main()
