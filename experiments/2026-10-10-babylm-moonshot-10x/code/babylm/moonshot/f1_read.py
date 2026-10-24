"""Card F1 stage 1 readout (T1 diagnostic, not BabyLM-legal; mini-ruler proxy; babylm/moonshot/make_f1_jobs.py), on v2
twin dumps:

    python -m babylm.moonshot.f1_read --members M0..M7 --stu DUMP --nsingle DUMP --nperm DUMP --fm DUMP
        --base10m DUMP [DUMP...] --out DIR [--guard [TOK_DIR]]

m0 is base@1M; G1 = DEV(m0) - DEV(base@10M) (the mean of the --base10m dumps, same metric and leg);
phi(x) = (DEV(m0) - DEV(x)) / G1 for the 8-member probability average and the student; g(8) = mean member - mixture;
floor_1M = SD of the 8 members. Kill if phi_stu < 0.20 or the student is not ahead of each of N-single, N-perm and F-M
by 2 floor_1M; the 10M student is built only if phi_stu >= 0.30 (and no kill). Full leg (registered) and, with --guard,
dev_bpw_twin's ex_unseen_chars leg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from babylm.data import corpus as C
from babylm.moonshot.ceiling_report import DEFAULT_TOK, Dump, dev, ensemble_tokens, guard_tokens, keep_mask

LABEL = "F1 stage 1, T1 diagnostic, not BabyLM-legal (mini-ruler proxy)"
COMPARATORS = ("nsingle", "nperm", "fm")


def read(members: list[Dump], stu: Dump, comp: dict[str, Dump], base10m: list[Dump], keep: np.ndarray,
         q: str = "npw_raw") -> dict:
    ref = members[0]
    for d in [*members[1:], stu, *comp.values(), *base10m]:
        if d.files != ref.files or not np.array_equal(d.word, ref.word):
            raise SystemExit(f"{d.path}: token stream differs from {ref.path}")

    def one(d: Dump) -> float:
        return dev(d.per_word(), keep)[q]

    m = [one(d) for d in members]
    ens = dev(ref.per_word(ensemble_tokens([d.nats for d in members])), keep)[q]
    b10 = float(np.mean([one(d) for d in base10m]))
    g1 = m[0] - b10
    s, floor = one(stu), float(np.std(m, ddof=1))
    c = {k: one(d) for k, d in comp.items()}
    ahead = {k: v - s for k, v in c.items()}
    phi_stu = (m[0] - s) / g1 if g1 > 0 else float("nan")
    killed = not (phi_stu >= 0.20) or any(v < 2 * floor for v in ahead.values())
    return {"members": m, "base_1M": m[0], "base_10M": b10, "G1": g1, "ens8": ens, "g8": float(np.mean(m)) - ens,
            "phi_ens8": (m[0] - ens) / g1 if g1 > 0 else float("nan"), "student": s, "phi_stu": phi_stu,
            "floor_1M": floor, "comparators": c, "student_ahead_by": ahead, "bar": 2 * floor, "killed": killed,
            "build_10M_student": (not killed) and phi_stu >= 0.30}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs=8, required=True)
    ap.add_argument("--stu", required=True)
    for k in COMPARATORS:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--base10m", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev_dir", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--guard", nargs="?", const=DEFAULT_TOK, default=None, metavar="TOK_DIR")
    a = ap.parse_args(argv)
    dd = Path(a.data_dir) if a.data_dir else C.DEFAULT_DATA_DIR
    dev_dir = Path(a.dev_dir) if a.dev_dir else dd / "raw" / "dev"
    mem, stu, b10 = [Dump(x) for x in a.members], Dump(a.stu), [Dump(x) for x in a.base10m]
    comp = {k: Dump(getattr(a, k)) for k in COMPARATORS}
    keep = keep_mask(mem[0].files, dev_dir, dd)
    res = {"label": LABEL, "full": read(mem, stu, comp, b10, keep), "full_ovr": read(mem, stu, comp, b10, keep, "npw_ovr")}
    if a.guard:
        mask, wu = guard_tokens(mem[0], dev_dir, a.guard)
        z = (lambda d: d.zeroed(mask, d.n_words - wu))
        res["guarded"] = read([z(d) for d in mem], z(stu), {k: z(d) for k, d in comp.items()}, [z(d) for d in b10], keep)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
