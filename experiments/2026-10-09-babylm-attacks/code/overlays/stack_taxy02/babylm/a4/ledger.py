"""Exposure (R0) and FLOP ledger of attack A4's arms (critic amendments 2 and 3).

    python -m babylm.a4.ledger RUN_DIR [RUN_DIR ...] [--base_result BASE/result.json] [--out OUT.json]

Per student run (a second-cycle run of babylm.harness.train; its result.json "exposure" block):
  words = the generator G's training (the --init_from checkpoint's words_seen) + the other trainings in the arm
          (--exposure_extra: the half-generators) + every word fed in the cycle, per kind (real / synthetic);
          seeds read at inference count nothing. R0: total <= the cap (100M).
  FLOPs = training FLOPs (flops_per_token x tokens_seen, the harness's estimate) of G, of every extra run and of the
          cycle, + the generation FLOPs of every synthetic source (babylm.a4.generate stats; N2 inherits its source's).
  Amendment 3: total FLOPs above 1.25 x the base round-1 run's (--base_result) => a FLOP-matched reference is needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from babylm.harness import cycle as CY

FLOP_RULE = 1.25


def run_dir_of(ckpt: str | Path) -> Path:
    p = Path(ckpt)
    return p.parent.parent if p.parent.name == "checkpoints" else p.parent


def train_flops(result: dict) -> float:
    return float(result["flops_per_token"]) * float(result["tokens_seen"])


def gen_flops(syn_dir: str | Path) -> tuple[float, list[str]]:
    """Generation FLOPs behind a synthetic split (following an N2 split to the split it shuffles)."""
    meta = json.loads((Path(syn_dir) / "bin" / f"{CY.SYNTH_SPLIT}.meta.json").read_text())
    if meta.get("of"):
        return gen_flops(meta["of"])
    shards = meta.get("shards") or []
    return float(sum((s.get("gen") or {}).get("flops", 0.0) for s in shards)), [s["shard"] for s in shards]


def arm_row(run: str | Path, base_flops: float | None) -> dict:
    run = Path(run)
    r = json.loads((run / "result.json").read_text())
    ex = r.get("exposure")
    if ex is None:
        raise SystemExit(f"{run}: not a second-cycle run (no exposure block in result.json)")
    row = {"run": run.name, "status": r["status"], "words": {}, "flops": {}}
    init = ex.get("init")
    g_res = None
    if init:
        gp = run_dir_of(init["path"]) / "result.json"
        g_res = json.loads(gp.read_text()) if gp.exists() else None
    row["words"]["G"] = int(ex["prior_words"])
    row["flops"]["G_train"] = train_flops(g_res) if g_res else None
    for k, v in ex["extra_words"].items():
        row["words"][k] = int(v)
        src = (ex.get("extra_from") or {}).get(k)
        row["flops"][f"{k}_train"] = train_flops(json.loads(Path(src["path"]).read_text())) if src else None
    for k, v in ex["cycle_words"].items():
        row["words"][f"cycle_{k}"] = int(v)
    row["flops"]["generation"] = 0.0
    for s in ex.get("sources") or []:
        kind, where, _ = CY.parse_source(s)
        if kind == "synthetic":
            row["flops"]["generation"] += gen_flops(where)[0]
    row["flops"]["cycle_train"] = train_flops(r)
    row["words_total"] = sum(row["words"].values())
    row["within_R0"] = row["words_total"] <= int(ex["cap"])
    known = [v for v in row["flops"].values() if v is not None]
    row["flops_total"] = sum(known) if len(known) == len(row["flops"]) else None
    row["flops_vs_base"] = row["flops_total"] / base_flops if base_flops and row["flops_total"] else None
    row["needs_flop_matched_reference"] = None if row["flops_vs_base"] is None else row["flops_vs_base"] > FLOP_RULE
    row["dev_nats_per_word_harness"] = (r.get("full_eval") or {}).get("nats_per_word")
    sc = run / "score" / "score.json"
    row["dev_nats_per_word"] = json.loads(sc.read_text()).get("dev_nats_per_word") if sc.exists() else None
    return row


def table(rows: list[dict]) -> str:
    keys = sorted({k for r in rows for k in r["words"]}, key=lambda k: (k != "G", k.startswith("cycle"), k))
    head = ["arm", *keys, "words total", "R0", "FLOPs total", "x base", "DEV nats/word"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        f = r["flops_total"]
        out.append("| " + " | ".join([r["run"], *(f"{r['words'].get(k, 0) / 1e6:.2f}M" for k in keys),
                                      f"{r['words_total'] / 1e6:.2f}M", "ok" if r["within_R0"] else "OVER",
                                      "?" if f is None else f"{f:.3e}",
                                      "?" if r["flops_vs_base"] is None else f"{r['flops_vs_base']:.2f}",
                                      "?" if r["dev_nats_per_word"] is None else f"{r['dev_nats_per_word']:.4f}"])
                   + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base_result", default=None, help="result.json of the base round-1 run (FLOP reference)")
    ap.add_argument("--base_epochs", type=float, default=None,
                    help="without --base_result: the base = G's recipe at this many epochs (G's FLOPs scaled)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs = [r for r in a.runs if (Path(r) / "result.json").exists()]
    base = train_flops(json.loads(Path(a.base_result).read_text())) if a.base_result else None
    if base is None and a.base_epochs and runs:
        init = json.loads((Path(runs[0]) / "result.json").read_text())["exposure"]["init"]
        g = json.loads((run_dir_of(init["path"]) / "result.json").read_text())
        base = train_flops(g) * a.base_epochs / float(g["epochs"])
    rows = [arm_row(r, base) for r in runs]
    rep = {"base_result": a.base_result, "base_epochs": a.base_epochs, "base_flops": base, "flop_rule": FLOP_RULE,
           "arms": rows,
           "table": table(rows)}
    print(rep["table"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        Path(a.out).with_suffix(".md").write_text(rep["table"] + "\n")
    return rep


if __name__ == "__main__":
    sys.exit(main() and None)
