"""CPU tests for the GPU job lists (no torch needed)."""
from __future__ import annotations

import json

from babylm.configs import make_gpu_jobs as MG
from babylm.configs.make_base_jobs import HERE

RUN_KEYS = ("--split", "--epochs", "--seed", "--tag")


def config(args):
    """(flag, value) pairs of a job's args minus the per-run keys, order-free; switches get value None."""
    pairs, i = [], 0
    while i < len(args):
        nxt = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else None
        if args[i] not in RUN_KEYS:
            pairs.append((args[i], nxt))
        i += 2 if nxt is not None else 1
    return sorted(pairs, key=str)


def test_v2_cells_match_round1_winner():
    """The v2 grid at the winner's recipe trains the winner's exact config: only split, epochs, seed and tag differ.
    10M x 10 seeds 0-2 are left out because the winner run and the floor seeds already are those cells."""
    win = next(j for j in json.loads((HERE / "jobs_base_r1_S.json").read_text())["jobs"]
               if j["name"] == "base_S_wd0p4_lr0p5_r1_ve0_tied")["args"]
    cells = MG.v2_jobs([("--muon_wd", "0.4"), ("--lr_scale", "0.5")], ["--extra", "n_value_embeds=0", "--tie_embeddings"])
    shapes = {tuple(j["name"].split("_")[2:4]) for j in cells}
    assert {("100M", "e10"), ("10M", "e2"), ("30M", "e10")} <= shapes and ("10M", "e10") not in shapes
    for j in cells:
        assert config(j["args"]) == config(win), j["name"]
