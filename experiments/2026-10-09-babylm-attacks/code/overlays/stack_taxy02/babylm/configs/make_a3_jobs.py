"""Screen job list for attack A3 (BPE-dropout, docs/babylm/attacks/A3_bpe_dropout.md):

    python -m babylm.configs.make_a3_jobs --base_flags "<base winner flags>" [--epochs 10] [--size S] [--seed 0]

Every job = make_base_jobs.job (frozen port, train_10M, --save --resume auto) + the base winner's flags + the arm's
flags (argparse: later flags win, --extra accumulates). The base arm runs in the same batch, through the A3 sampler
at p = 0 (critic ref-0063; bit-identical to the port path on CPU, re-checked on GPU by this arm). Anneal = the last
max(1, round(0.2 E)) epochs canonical. The null's p (0.1139) matches A3a's token count on train_10M
(python -m babylm.harness.bpe_dropout match --p 0.1 --scope frequent --target_scope all: 1.1731 vs 1.1732 per epoch).
After each run: babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>.
"""
from __future__ import annotations

import argparse
import json
import shlex

from babylm.configs.make_base_jobs import HERE, job

NULL_P = "0.1139"


def arms(E: int) -> list[tuple[str, list[str]]]:
    k = max(1, round(0.2 * E))
    ann = lambda p: f"{p}x{E - k},0x{k}"  # noqa: E731
    return [("base", ["--bpe_dropout", "0"]),  # critic ref-0063: the base runs through the stream builder at p=0
            ("a3a_all_p0p1_anneal", ["--bpe_dropout", ann("0.1")]),
            ("a3b_all_p0p1_const", ["--bpe_dropout", f"0.1x{E}"]),
            ("a3c_rare_p0p2_anneal", ["--bpe_dropout", ann("0.2"), "--bpe_dropout_scope", "rare"]),
            ("a3n_freq_p0p1139_anneal", ["--bpe_dropout", ann(NULL_P), "--bpe_dropout_scope", "frequent"])]


def screen(base_flags: list[str], E: int, size: str, seed: int, label: str) -> dict:
    jobs = []
    for name, fl in arms(E):
        j = job(f"a3_{size}_{label}_s{seed}_{name}", size, {"--epochs": str(E), "--seed": str(seed)}, 3.0)
        tail = j["args"][-5:]  # --save --resume auto --tag NAME
        j["args"] = j["args"][:-5] + base_flags + fl + tail
        jobs.append(j)
    return {"description": f"attack A3 screen (BPE-dropout), size {size}, train_10M x {E}, seed {seed}, base flags "
                           f"'{shlex.join(base_flags)}' ({label}); arms: base, A3a all p0.1 anneal, A3b all p0.1 "
                           f"constant, A3c rare (<=63) p0.2 anneal, null: frequent (>63) p{NULL_P} anneal "
                           "(token count matched to A3a)",
            "note": "Run base first. The dropout arms build their epoch streams on the CPU before training "
                    "(cached in <data_dir>/bin/bpedrop/, ~2 min table + ~10-20 s per epoch on 12 vCPU); "
                    "train_time excludes it. After each run: babylm/harness/score_trial.sh babylm/runs/<name> "
                    "<eval workdir>.",
            "jobs": jobs}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_flags", required=True, help="the base winner's trainer flags beyond the frozen port")
    ap.add_argument("--label", required=True, help="short tag for the base recipe, e.g. r1win")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--size", default="S")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    spec = screen(shlex.split(a.base_flags), a.epochs, a.size, a.seed, a.label)
    f = HERE / f"jobs_a3_screen_{a.size}_{a.label}.json"
    f.write_text(json.dumps(spec, indent=1) + "\n")
    print("wrote", f)


if __name__ == "__main__":
    main()
