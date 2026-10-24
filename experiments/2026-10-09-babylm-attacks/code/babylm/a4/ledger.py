"""Exposure (R0) and FLOP ledger of attack A4's arms (critic amendments 2 and 3).

    python -m babylm.a4.ledger RUN_DIR [RUN_DIR ...] [--base_result BASE/result.json] [--out OUT.json]

Per student run (a second-cycle run of babylm.harness.train; its result.json "exposure" block):
  words = the generator G's training (the --init_from checkpoint's words_seen) + the other trainings in the arm
          (--exposure_extra: the half-generators) + every word fed in the cycle, per kind (real / synthetic);
          seeds read at inference count nothing. R0: total <= the cap (100M).
  FLOPs = training FLOPs (flops_per_token x tokens_seen, the harness's estimate) of G, of every extra run and of the
          cycle, + the generation FLOPs of every synthetic source (babylm.a4.generate stats; N2 inherits its source's).
  Amendment 3: total FLOPs above 1.25 x the base round-1 run's (--base_result) => a FLOP-matched reference is needed.
  G's training FLOPs scale to the checkpoint the cycle starts from (a mid-run checkpoint counts its share), and a run
  with a KD teacher adds its forwards (kd_teacher: flops_per_token / 3 per token it read, from log.jsonl; same model).
T2 (--exposure_rule t2; critic ref-0014): the cap covers corpus words only (G + extras + the cycle's real words):
  words_capped / within_cap; within_R0 stays the every-word count, so a T2 arm reads "outside R0.2".
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


def kd_teacher_flops(run: Path, r: dict, kd: dict) -> float | None:
    """Teacher forwards of a KD run: flops_per_token / 3 (forward only) per token the teacher read: per logged step,
    its new tokens x syn_tok (the synthetic share, babylm.a4.t2) or x 1 (KD on every row) when the step ran KD.
    None when the teacher is not the student's architecture or the log is missing."""
    t, cfg = kd.get("teacher") or {}, (r.get("config") or {}).get("model_cfg") or {}
    if any(t.get(k) != cfg.get(k) for k in ("n_layer", "n_embd")) or not (run / "log.jsonl").exists():
        return None
    tok, prev = 0.0, 0
    for line in (run / "log.jsonl").read_text().splitlines():
        x = json.loads(line)
        if x.get("type") != "train":
            continue
        new, prev = x["tokens_seen"] - prev, x["tokens_seen"]
        if "kd_kl" in x:
            tok += new * float(x.get("syn_tok", 1.0))
    return float(r["flops_per_token"]) / 3.0 * tok


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
    g_words = (init or {}).get("words_seen")  # G may be a mid-run checkpoint of a longer run
    share = min(1.0, g_words / g_res["words_seen"]) if g_res and g_words and g_res.get("words_seen") else 1.0
    row["flops"]["G_train"] = train_flops(g_res) * share if g_res else None
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
    kd = ((r.get("config") or {}).get("levers") or {}).get("kd")
    if kd:
        row["flops"]["kd_teacher"] = kd_teacher_flops(run, r, kd)
    row["words_total"] = sum(row["words"].values())
    row["within_R0"] = row["words_total"] <= int(ex["cap"])
    row["rule"] = ex.get("rule", "r0")
    if row["rule"] != "r0":
        row["tier"] = ex.get("tier")
        row["words_capped"] = CY.capped_words(row["rule"], ex["prior_words"], ex["extra_words"], ex["cycle_words"])
        row["within_cap"] = row["words_capped"] <= int(ex["cap"])
    known = [v for v in row["flops"].values() if v is not None]
    row["flops_total"] = sum(known) if len(known) == len(row["flops"]) else None
    row["flops_vs_base"] = row["flops_total"] / base_flops if base_flops and row["flops_total"] else None
    row["needs_flop_matched_reference"] = None if row["flops_vs_base"] is None else row["flops_vs_base"] > FLOP_RULE
    row["dev_nats_per_word_harness"] = (r.get("full_eval") or {}).get("nats_per_word")
    sc = run / "score" / "score.json"
    s = json.loads(sc.read_text()) if sc.exists() else {}
    row["dev_nats_per_word"], row["dev_nats_per_word_twin"] = s.get("dev_nats_per_word"), s.get("dev_nats_per_word_twin")
    return row


def cap_cell(r: dict) -> str:
    if r.get("rule", "r0") == "r0":
        return "R0 ok" if r["within_R0"] else "R0 OVER"
    return (f"{r['rule'].upper()} {'ok' if r['within_cap'] else 'OVER'}: corpus {r['words_capped'] / 1e6:.2f}M "
            "(outside R0.2)")


def table(rows: list[dict]) -> str:
    keys = sorted({k for r in rows for k in r["words"]}, key=lambda k: (k != "G", k.startswith("cycle"), k))
    head = ["arm", *keys, "words total", "cap", "FLOPs total", "x base", "DEV v1", "DEV v2 (twin)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        f = r["flops_total"]
        out.append("| " + " | ".join([r["run"], *(f"{r['words'].get(k, 0) / 1e6:.2f}M" for k in keys),
                                      f"{r['words_total'] / 1e6:.2f}M", cap_cell(r),
                                      "?" if f is None else f"{f:.3e}",
                                      "?" if r["flops_vs_base"] is None else f"{r['flops_vs_base']:.2f}",
                                      *("?" if r.get(k) is None else f"{r[k]:.4f}"
                                        for k in ("dev_nats_per_word", "dev_nats_per_word_twin"))])
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
