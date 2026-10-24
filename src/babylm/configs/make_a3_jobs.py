"""Screen job list for attack A3 (BPE-dropout, docs/babylm/attacks/A3_bpe_dropout.md):

    python -m babylm.configs.make_a3_jobs --base_flags "<base winner flags>" --label L [--epochs 10] [--size S]
        [--seeds 0] [--arms a3a] [--out FILE]

Every job = make_base_jobs.job (frozen port, train_10M, --save --resume auto) + the base winner's flags + the arm's
flags (argparse: later flags win, --extra accumulates). The base arm runs in the same batch, through the A3 sampler
at p = 0 (critic ref-0063; bit-identical to the port path on CPU, re-checked on GPU by this arm). Anneal = the last
max(1, round(0.2 E)) epochs canonical. The null's p (0.1185) matches A3a's token count on train_10M
exactly (epoch-0 streams: 18,957,525 vs 18,957,645 tokens; the first p, 0.1139, came from a biased 3-chunk estimate and
gave 18,836,214).
After each run: babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>.
"""
from __future__ import annotations

import argparse
import json
import shlex

from babylm.configs.make_base_jobs import HERE, job

NULL_P = "0.1185"


def arms(E: int) -> list[tuple[str, list[str]]]:
    k = max(1, round(0.2 * E))
    ann = lambda p: f"{p}x{E - k},0x{k}"  # noqa: E731
    return [("base", ["--bpe_dropout", "0"]),  # critic ref-0063: the base runs through the stream builder at p=0
            ("a3a_all_p0p1_anneal", ["--bpe_dropout", ann("0.1")]),
            ("a3b_all_p0p1_const", ["--bpe_dropout", f"0.1x{E}"]),
            ("a3c_rare_p0p2_anneal", ["--bpe_dropout", ann("0.2"), "--bpe_dropout_scope", "rare"]),
            ("a3n_freq_p0p1185_anneal", ["--bpe_dropout", ann(NULL_P), "--bpe_dropout_scope", "frequent"])]


def screen(base_flags: list[str], E: int, size: str, seeds: list[int], label: str, only: list[str] | None = None) -> dict:
    jobs = []
    for seed in seeds:
        for name, fl in arms(E):
            if only and not any(name.startswith(o) for o in only):
                continue
            j = job(f"a3_{size}_{label}_e{E}_s{seed}_{name}" if E != 10 else f"a3_{size}_{label}_s{seed}_{name}", size,
                    {"--epochs": str(E), "--seed": str(seed)}, 3.0)
            tail = j["args"][-5:]  # --save --resume auto --tag NAME
            j["args"] = j["args"][:-5] + base_flags + fl + tail
            jobs.append(j)
    seed = ",".join(map(str, seeds))
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
    ap.add_argument("--seeds", default="0", help="comma list, e.g. 1,2")
    ap.add_argument("--arms", default=None, help="comma list of arm-name prefixes to keep, e.g. a3a (default: all)")
    ap.add_argument("--out", default=None, help="file name under babylm/configs (default jobs_a3_screen_SIZE_LABEL.json)")
    a = ap.parse_args(argv)
    spec = screen(shlex.split(a.base_flags), a.epochs, a.size, [int(s) for s in a.seeds.split(",")], a.label,
                  a.arms.split(",") if a.arms else None)
    f = HERE / (a.out or f"jobs_a3_screen_{a.size}_{a.label}.json")
    f.write_text(json.dumps(spec, indent=1) + "\n")
    print("wrote", f)


if __name__ == "__main__":
    main()
