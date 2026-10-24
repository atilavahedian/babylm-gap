"""Screen verdict of attack A4 by its registered kill rule (card "Kill rule"; critic amendments 1, 3, 4 and 5).

    python -m babylm.a4.verdict --base_score BASE/score.json --floor F --blimp_sigma S \
        [--runs_root babylm/runs] [--data babylm/runs/a4_data] [--out babylm/runs/a4_data/verdict.json]

F = the critic's registered seed floor of full-DEV nats/word, S = the seed sigma of BLiMP-fast accuracy (the fast
suite's units). Without both there is no verdict (amendment 5), only the audits and the table.
DEV = score.json's dev_nats_per_word (dev_bpw.py, full DEV, the scored checkpoint); gain over X = DEV(X) - DEV(arm).
The judged arm is the one of A4a (a4_a) and A4b (a4_b) with the lower DEV (amendment 5: one arm, picked on DEV). Its
references: its N0' twin, the round-1 base, its N1 and N2, and a FLOP-matched reference when its FLOPs exceed 1.25x
its twin's (registered: N0'_a for A4b). The comparator (amendment 1) = the better of N0' and the base.
KILL if a gain over any reference is below 2F, if BLiMP-fast is >= 2S below the comparator's, or if the arm breaks
R0. A missing score = no verdict. The other arm's numbers are reported, not judged.
Audits first (amendment 4): no table without a leak audit covering every synthetic split of the arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babylm.a4 import ledger as LG
from babylm.harness import cycle as CY

FAMILIES = {"a": {"arm": "a4_a", "n0": "a4_n0a", "n1": "a4_n1a", "n2": "a4_n2a"},
            "b": {"arm": "a4_b", "n0": "a4_n0b", "n1": "a4_n1b", "n2": "a4_n2b"}}
FLOP_MATCHED = {"b": "a4_n0a"}  # card, run registration: N0'_a has about A4b's training FLOPs
BLIMP = "blimp/blimp_fast"


def score(run: Path) -> dict | None:
    p = run / "score" / "score.json"
    if not p.exists():
        p = run / "score.json"  # a synced run dir (small files only)
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {"dev": s.get("dev_nats_per_word"), "blimp": (s.get("fast") or {}).get(BLIMP),
            "checkpoint": s.get("checkpoint")}


def synth_splits(run: Path) -> dict[str, str]:
    """{sha256: source label} of the run's synthetic sources (the split's token hash, as the leak audit records it)."""
    ex = json.loads((run / "result.json").read_text()).get("exposure") or {}
    return {h: s for s, h in zip(ex.get("sources") or [], ex.get("source_sha256") or [])
            if CY.parse_source(s)[0] == "synthetic"}


def audits(data: Path, used: dict[str, str]) -> dict:
    """Leak and copy summaries; raises when the leak audit misses a synthetic split the arms used (amendment 4)."""
    leak = {}
    for f in sorted((data / "audit").glob("leak*.json")):
        leak.update(json.loads(f.read_text())["synthetic"])
    audited = {v["sha256"] for v in leak.values()}
    missing = [lab for h, lab in used.items() if h not in audited]
    if missing:
        raise SystemExit(f"amendment 4: no leak audit for {missing}: run babylm.a4.audit leak before any number")
    out = {"leak": {Path(k).name: {"words": v["words"], "items_hit": v["items_hit"],
                                   "items_hit_synthetic_only": v["items_hit_synthetic_only"],
                                   "dev_13gram_share": v["dev_13gram_share"],
                                   "dev_13gram_share_synthetic_only": v["dev_13gram_share_synthetic_only"]}
                    for k, v in leak.items()}, "copy": {}}
    for f in sorted((data / "audit").glob("copy_*.json")):
        c = json.loads(f.read_text())
        nn = c["nearest_neighbour"]
        out["copy"][f.stem.removeprefix("copy_")] = {
            **{f"overlap_{n}": c["overlap"]["synthetic"].get(n) for n in ("6gram", "8gram")},
            "dev_overlap_6gram": c["overlap"]["dev"].get("6gram"),
            **{f"near_copy_{k}": (nn.get(k) or {}).get("near_copy_rate") for k in ("synthetic", "dev_chunks", "probe")}}
    return out


def judge(fam: str, runs: Path, base: dict, floor: float | None, sigma: float | None, rows: dict) -> dict:
    f = FAMILIES[fam]
    arm = score(runs / f["arm"])
    refs = {"N0'": f["n0"], "N1": f["n1"], "N2": f["n2"]}
    sc = {k: score(runs / v) for k, v in refs.items()}
    sc["base"] = base
    fl = rows.get(f["arm"], {}).get("flops_total")
    twin_fl = rows.get(f["n0"], {}).get("flops_total")
    ratio = fl / twin_fl if fl and twin_fl else None
    if fam in FLOP_MATCHED and (ratio is None or ratio > LG.FLOP_RULE):
        sc["FLOP-matched " + FLOP_MATCHED[fam]] = score(runs / FLOP_MATCHED[fam])
    out = {"arm": f["arm"], "dev": arm and arm["dev"], "blimp_fast": arm and arm["blimp"], "flops_vs_twin": ratio,
           "within_R0": rows.get(f["arm"], {}).get("within_R0"), "refs": {}, "kill": [], "missing": []}
    for k, s in sc.items():
        if not s or s["dev"] is None or not arm or arm["dev"] is None:
            out["missing"].append(k)
            continue
        out["refs"][k] = {"dev": s["dev"], "gain": s["dev"] - arm["dev"], "blimp_fast": s["blimp"]}
    have = [k for k in ("N0'", "base") if k in out["refs"]]
    if len(have) == 2:
        comp = min(have, key=lambda k: out["refs"][k]["dev"])
        out["comparator"] = comp
        cb = out["refs"][comp]["blimp_fast"]
        out["blimp_drop_vs_comparator"] = None if cb is None or out["blimp_fast"] is None else cb - out["blimp_fast"]
    if floor is None or sigma is None:
        out["verdict"] = "NO VERDICT: the floor is not registered (amendment 5)"
        return out
    if out["missing"]:
        out["verdict"] = f"NO VERDICT: missing {out['missing']}"
        return out
    for k, r in out["refs"].items():
        if r["gain"] < 2 * floor:
            out["kill"].append(f"gain over {k} {r['gain']:+.4f} < 2F = {2 * floor:.4f}")
    d = out.get("blimp_drop_vs_comparator")
    if d is None:
        out["kill"].append("no BLiMP-fast score")
    elif d >= 2 * sigma:
        out["kill"].append(f"BLiMP-fast {d:.2f} below the comparator >= 2S = {2 * sigma:.2f}")
    if out["within_R0"] is False:
        out["kill"].append("over the R0 cap")
    out["verdict"] = "KILL" if out["kill"] else "PASS"
    return out


def table(fams: dict) -> str:
    lines = ["| family | arm DEV | reference | ref DEV | gain | BLiMP-fast arm / ref |", "|---|---|---|---|---|---|"]
    for fam, j in fams.items():
        for k, r in j["refs"].items():
            bl = " / ".join("?" if x is None else f"{x:.2f}" for x in (j["blimp_fast"], r["blimp_fast"]))
            lines.append(f"| {fam} ({j['arm']}) | {j['dev']:.4f} | {k} | {r['dev']:.4f} | {r['gain']:+.4f} | {bl} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_score", required=True, help="score.json of the round-1 base run (seed 0)")
    ap.add_argument("--floor", type=float, default=None, help="F: the critic's registered DEV seed floor")
    ap.add_argument("--blimp_sigma", type=float, default=None, help="S: the seed sigma of BLiMP-fast")
    ap.add_argument("--runs_root", default="babylm/runs")
    ap.add_argument("--data", default=None, help="A4's data dir (audit/, ledger.json); default RUNS_ROOT/a4_data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs = Path(a.runs_root)
    data = Path(a.data) if a.data else runs / "a4_data"
    present = [r for f in FAMILIES.values() for r in f.values() if (runs / r / "result.json").exists()]
    used = {h: lab for r in present for h, lab in synth_splits(runs / r).items()}
    rep = {"audits": audits(data, used), "floor": a.floor, "blimp_sigma": a.blimp_sigma}
    lg = data / "ledger.json"
    rows = {r["run"]: r for r in json.loads(lg.read_text())["arms"]} if lg.exists() else \
        {r: LG.arm_row(runs / r, None) for r in present}
    bs = json.loads(Path(a.base_score).read_text())
    base = {"dev": bs.get("dev_nats_per_word"), "blimp": (bs.get("fast") or {}).get(BLIMP), "run": bs.get("run")}
    fams = {k: judge(k, runs, base, a.floor, a.blimp_sigma, rows) for k in FAMILIES}
    scored = {k: j for k, j in fams.items() if j["dev"] is not None}
    best = min(scored, key=lambda k: scored[k]["dev"]) if scored else None
    rep["families"] = fams
    rep["judged"] = best
    rep["verdict"] = fams[best]["verdict"] if best else "NO VERDICT: no scored arm"
    rep["promote"] = fams[best]["arm"] if best and rep["verdict"] == "PASS" else None
    rep["table"] = table(fams)
    print(json.dumps(rep["audits"], indent=1))
    print(rep["table"])
    print(f"judged arm: {best and fams[best]['arm']} -> {rep['verdict']}"
          + (f" ({'; '.join(fams[best]['kill'])})" if best and fams[best]["kill"] else ""))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        Path(a.out).with_suffix(".md").write_text(rep["table"] + f"\n\nJudged arm: {best} -> {rep['verdict']}\n")
    return rep


if __name__ == "__main__":
    main()
