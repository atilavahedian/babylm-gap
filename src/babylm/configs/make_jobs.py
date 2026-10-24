"""Regenerate babylm/configs/frozen_port.json and the job lists: python -m babylm.configs.make_jobs

Job-list format = sprint1/gcp/run_jobs.py's ({"jobs": [{"name", "module", "args", "timeout_s"}]}), with module
babylm.harness.train; run with python -m babylm.harness.run_jobs <list> (or add babylm.harness.train to
sprint1/gcp/run_jobs.py ALLOWED). Every job resumes from its own resume.pt after a preemption (--resume auto).
timeout_s is a generous safety stop (a timed-out job continues from its last milestone when the list is rerun).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = [  # sprint1/configs/frozen_baseline.json "args", minus --size (set per job)
    "--batch_tokens", "65536", "--seq_len", "1024", "--lr_batch_scaling", "none",
    "--muon_lr", "0.065", "--lr_embed", "0.78", "--lr_head", "0.0208", "--lr_scalar", "0.052",
    "--muon_momentum", "0.95", "--muon_momentum_start", "0.85", "--muon_momentum_warmup_steps", "150",
    "--adam_beta1", "0.8", "--adam_beta2", "0.95", "--warmup_steps", "0", "--cooldown_frac", "0.5",
    "--lr_floor", "0.1", "--muon_wd", "0", "--adam_wd", "0", "--extra", "softcap=15", "--extra", "unet=true",
    "--compile",
]
RUN = ["--save", "--resume", "auto"]


def job(name: str, size: str, split: str, epochs: str, seed: int, timeout_h: float) -> dict:
    return {"name": name, "module": "babylm.harness.train",
            "args": ["--model", "baseline", "--size", size, "--split", split, "--epochs", epochs,
                     "--seed", str(seed), *FROZEN, *RUN, "--tag", name],
            "timeout_s": int(timeout_h * 3600)}


def ep_tag(e: str) -> str:
    return e.replace(".", "p")


def main() -> None:
    frozen = {
        "config": "frozen_port",
        "derived_from": "sprint1/configs/frozen_baseline.json (s2_B_base, frozen 5 Oct by the experimenter)",
        "args": ["--model", "baseline", "--size", "small", *FROZEN, "--split", "train_10M", "--epochs", "10"],
        "unchanged": "model code (sprint1.models.baseline), optimizer code (sprint1.harness.optim), LR schedule "
                     "(sprint1.harness.train.lr_factor), every optimizer/schedule value, size small 6L/384d, seq 1024, "
                     "batch 65536 tokens, softcap 15, unet, bf16 autocast + torch.compile on CUDA",
        "changed": {"data": "BabyLM 2025 train_10M (was FineWeb GPT-2 shards)",
                    "tokenizer": "byte-level BPE 16384 trained on train_10M (was GPT-2 50257, padded 50304)",
                    "vocab_size": 16384,
                    "budget": "10 epochs of the split, sampled as a fresh permutation of tiled windows per epoch "
                              "(was a single pass over uniformly random windows of FineWeb)",
                    "evaluation": "DEV-sub / TRAIN-sub (~1M words each) at word milestones + full BabyLM dev at the "
                                  "end; nats/word, nats/token, bits/byte (was FineWeb val first half)"},
    }
    (HERE / "frozen_port.json").write_text(json.dumps(frozen, indent=1) + "\n")

    a = [job(f"port_{sz}_10M_e10_s{s}", sz, "train_10M", "10", s, 12 if sz == "small" else 24)
         for sz in ("small", "S") for s in (0, 1, 2)]
    b = [job(f"grid_S_10M_e{ep_tag(e)}_s0", "S", "train_10M", e, 0, 48) for e in ("1", "2", "4", "10", "20")]
    b += [job(f"grid_S_100M_e{ep_tag(e)}_s0", "S", "train_100M", e, 0, 96) for e in ("1", "10")]
    b += [job(f"grid_S_30M_e{ep_tag(e)}_s0", "S", "train_30M", e, 0, 48) for e in ("3.333", "10")]
    c = [job("size_M_10M_e10_s0", "M", "train_10M", "10", 0, 48),
         job("size_M_100M_e10_s0", "M", "train_100M", "10", 0, 168)]
    note = ("Prerequisite on the VM: python -m babylm.data.prepare --download (verifies md5s, tokenizes with the "
            "committed tokenizer, checks sha256 against babylm/data/manifest.json). Run: python -m "
            "babylm.harness.run_jobs <this file>. Runs land in babylm/runs/<name>.")
    for fname, jobs, what in (("jobs_a_frozen_port.json", a, "(a) frozen port, small and S, train_10M x 10 epochs, seeds 0-2"),
                              ("jobs_b_phase2_grid.json", b, "(b) Phase-2 grid at S, seed 0: epochs x data"),
                              ("jobs_c_size_M.json", c, "(c) size M: train_10M x 10 and train_100M x 10")):
        (HERE / fname).write_text(json.dumps({"description": what, "note": note, "jobs": jobs}, indent=1) + "\n")
        print(HERE / fname, len(jobs), "jobs")


if __name__ == "__main__":
    main()
