"""Card PE Gate 2 readout (mini-ruler proxy, 1M x 10; card section "Kill rule"), on v2 twin dumps:

    python -m babylm.moonshot.pe_gate2_read --pe3 MEMBER_DUMP x3 --pe3_ens DUMP --i3 DUMP x3 --f FDEEP FM --out DIR
        [--nj_ens DUMP] [--f_json TWIN_JSON...] [--guard [TOK_DIR]]

--pe3: PE3's member dumps (babylm.moonshot.particle_members, each scored like any model); --pe3_ens: PE3's own dump,
whose DEV must equal the member mixture within 1e-4 (critic ref-0417). --i3: the three independent 1M base seeds
(F1's ms_f1_m0-m2, reused as B, critic ref-0296). --f: F-deep and F-M. --nj_ens: NJ's own dump (reported).
--f_json: an F run with no dump, read from its dev_bpw_twin.json (nats_per_word on the full leg,
ex_unseen_chars_nats_per_word on the guarded leg; not on the overlap-removed diagnostic).
    J(X) = mean member DEV - p_bar DEV;  rho = J(PE3) / J(I3);  floor_1M = SD of the I3 singles
    (a) killed (proxy) unless p_bar(PE3) <= better F - 2 floor_1M
    (b) shared trunk killed (proxy) if rho < 0.5
    (c) the bootstrap arms are not built, so (c) is not run.
Every number on the full leg (registered) and, with --guard, on dev_bpw_twin's ex_unseen_chars leg. Labels: "mini-ruler
proxy"; outcomes at most "killed (proxy)". Calibration binds at promotion and claim, not here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from babylm.data import corpus as C
from babylm.moonshot.ceiling_report import DEFAULT_TOK, Dump, dev, ensemble_tokens, guard_tokens, keep_mask

LABEL = "mini-ruler proxy (1M x 10); outcomes at most 'killed (proxy)'"


def read(pe3: list[Dump], pe3_ens: Dump, i3: list[Dump], f: list[Dump], nj: Dump | None, keep: np.ndarray,
         q: str = "npw_raw", f_values: dict[str, float] | None = None) -> dict:
    ref = pe3[0]
    for d in [*pe3[1:], pe3_ens, *i3, *f, *([nj] if nj else [])]:
        if d.files != ref.files or not np.array_equal(d.word, ref.word):
            raise SystemExit(f"{d.path}: token stream differs from {ref.path}")

    def one(d: Dump) -> float:
        return dev(d.per_word(), keep)[q]

    def mix(ds: list[Dump]) -> float:
        return dev(ds[0].per_word(ensemble_tokens([d.nats for d in ds])), keep)[q]

    pe_m, i3_m = [one(d) for d in pe3], [one(d) for d in i3]
    pbar, pbar_own, i3_mix = mix(pe3), one(pe3_ens), mix(i3)
    fs = {d.path: one(d) for d in f} | (f_values or {})
    floor = float(np.std(i3_m, ddof=1))
    j_pe, j_i3 = float(np.mean(pe_m)) - pbar, float(np.mean(i3_m)) - i3_mix
    rho = j_pe / j_i3 if j_i3 > 0 else float("nan")
    best_f = min(fs.values())
    out = {"pe3_members": pe_m, "pe3_pbar_member_mixture": pbar, "pe3_pbar_own_dump": pbar_own,
           "pe3_mixture_check": abs(pbar - pbar_own) <= 1e-4, "J_pe3": j_pe,
           "i3_singles": i3_m, "B_mean": float(np.mean(i3_m)), "floor_1M": floor, "i3_mixture": i3_mix, "J_i3": j_i3,
           "rho": rho, "F": fs, "better_F": best_f,
           "pe3_vs_better_F": best_f - pbar_own, "pe3_vs_B_mean": float(np.mean(i3_m)) - pbar_own,
           "a_bar": 2 * floor, "a_killed": not (pbar_own <= best_f - 2 * floor), "b_killed": not (rho >= 0.5)}
    if nj:
        out["nj_pbar_own_dump"] = one(nj)
        out["pe3_vs_nj"] = out["nj_pbar_own_dump"] - pbar_own
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pe3", nargs="+", required=True)
    ap.add_argument("--pe3_ens", required=True)
    ap.add_argument("--i3", nargs=3, required=True)
    ap.add_argument("--f", nargs="*", default=[])
    ap.add_argument("--f_json", nargs="*", default=[], help="F runs without a dump: their dev_bpw_twin.json")
    ap.add_argument("--nj_ens", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev_dir", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--guard", nargs="?", const=DEFAULT_TOK, default=None, metavar="TOK_DIR")
    a = ap.parse_args(argv)
    dd = Path(a.data_dir) if a.data_dir else C.DEFAULT_DATA_DIR
    dev_dir = Path(a.dev_dir) if a.dev_dir else dd / "raw" / "dev"
    pe3, i3, f = [Dump(x) for x in a.pe3], [Dump(x) for x in a.i3], [Dump(x) for x in a.f]
    pe3_ens, nj = Dump(a.pe3_ens), Dump(a.nj_ens) if a.nj_ens else None
    if not f and not a.f_json:
        raise SystemExit("give at least one F reference (--f or --f_json)")
    tw = {p: json.loads(Path(p).read_text()) for p in a.f_json}
    keep = keep_mask(pe3[0].files, dev_dir, dd)
    res = {"label": LABEL, "full": read(pe3, pe3_ens, i3, f, nj, keep,
                                        f_values={p: t["nats_per_word"] for p, t in tw.items()})}
    if f:
        res["full_ovr"] = read(pe3, pe3_ens, i3, f, nj, keep, "npw_ovr")
    if a.guard:
        mask, wu = guard_tokens(pe3[0], dev_dir, a.guard)
        z = (lambda d: d.zeroed(mask, d.n_words - wu))
        res["guarded"] = read([z(d) for d in pe3], z(pe3_ens), [z(d) for d in i3], [z(d) for d in f],
                              z(nj) if nj else None, keep,
                              f_values={p: t["ex_unseen_chars_nats_per_word"] for p, t in tw.items()})
    if not res["full"]["pe3_mixture_check"]:
        raise SystemExit(f"PE3 member mixture {res['full']['pe3_pbar_member_mixture']:.6f} vs its own dump "
                         f"{res['full']['pe3_pbar_own_dump']:.6f}: differ by more than 1e-4")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pe_gate2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
