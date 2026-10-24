"""Attack A4's T2 arm: closed-system augmentation past 100M words (official FAQ tier, outside R0.2; critic ref-0014,
F1 T2 allocation accepted 04:52Z and V_syn fixed at ref-0108; card docs/babylm/attacks/A4_reexpression.md, "T2 arm").

The student continues the generator G on a fresh LR cycle over real and synthetic rows (babylm.harness.cycle). Real
rows train on plain cross-entropy. Synthetic rows train on the moonshot's KD loss (babylm.moonshot.kd.kd_terms) to
the frozen G, which wrote them:
    loss = [ sum over real-row targets of CE
             + sum over synthetic-row targets of ((1 - alpha) CE + alpha tau^2 KL(p_tau || q_tau)) ] / all targets
with p = softmax(G logits / tau), q = softmax(student logits / tau) over the full vocabulary (the teacher's vocabulary
a prefix of the student's), exactly kd_terms on the synthetic rows. The teacher runs on the synthetic rows only.
babylm.harness.train --kd_teacher G --kd_rows synthetic --sources ... --exposure_rule t2.

Readouts (babylm.configs.make_a4_t2_jobs runs the jobs; audits first, amendment 4; DEV = score.json only):
    python -m babylm.a4.t2 twin RUN [RUN ...]       (on the VM: merge score/dev_bpw_twin.json into score.json)
    python -m babylm.a4.t2 gate --runs_root R --data R/a4_t2g_data --oracle_score BASE10M/score.json|--oracle_dev X \
        --ref_instrument oracle=solo [--packed] [--metric v2|v1] [--lead_min L] [--out G.json]
    python -m babylm.a4.t2 verdict --base_score BASE/score.json --runs_root R --data R/a4_t2_data \
        --ref_instrument base=solo [flop_ref=solo] [estar_raw=solo] \
        [--flop_ref_score M/score.json] [--metric v2|v1] [--pass_gain P --kill_gain K] [--packed] [--data_dir D] \
        [--calib CALIB.json] [--estar_raw ESTAR_RAW/score.json] [--vm_evidence INSTRUMENTS.tsv] [--estar NOTE] \
        [--out V.json]
DEV metric (babylm.a4.verdict): v2 = the blank-line twin is primary (critic 05:02Z; its interim floor ref-0019), v1
  reported beside it; a --metric v1 read is information only, with the blank-line guard (ref-0020: v2 gain > 0 and
  within 0.0058 of the v1 gain, else "protocol-artifact"), which holds by construction under v2.
gate (mini-ruler, labels only; ruling ref-0110): G1 = DEV(base@1M) - DEV(base@10M, the oracle, same metric); phi_stu =
  (DEV(base@1M) - DEV(arm)) / G1; the binding comparator = the lowest DEV of base@1M, N0', N1, N2 and the size-M FLOP
  reference at 1M. "killed (proxy)" if phi_stu < 0.20 or the arm leads the binding comparator by < 0.016 nats/word
  (the registered bar, kept on the twin: experimenter 05:19Z); else the 10M stage may run (a pass is a label, never a verdict).
  G@1M, the teacher, is reported, not compared.
verdict (10M): references = the round-1 base (exposure-matched), N0', N1, N2 and the size-M FLOP reference
  (--flop_ref_score); per reference gain = DEV(ref) - DEV(arm) against the metric's floor (v2 ref-0019: >= 0.010
  pass, 0.0074-0.010 inconclusive, < 0.0074 kill; against the nulls N0', N1, N2 a pass needs >= 0.020, critic
  ref-0028); a BLiMP-fast drop above 2.0 below the binding comparator (the
  lowest-DEV reference) kills, as does a corpus exposure over the T2 cap. A kill (or a protocol-artifact) is final as
  soon as it shows; else missing references leave it PENDING (batch 2 runs N1 and N2 only once the arm clears the base
  and N0'). Promotion (3 seeds, size M, official eval, Y-a for any 10x wording) needs a gain >= 0.26 over the binding
  comparator and the promotion gates of babylm.a4.verdict (calibrated_t over every reference, ref-0011 / ref-0029;
  raw over the base at E*_raw and calibrated_t over the base at E*_cal, ref-0030 / ref-0031; --calib, --estar_raw;
  each on its own metric's kill line, ref-0032: calibrated_t 0.0095 full / 0.0094 guarded solo, raw 0.0074; every
  compared run's instrument corroborated from VM evidence, ref-0035: --vm_evidence, or a solo run's complete
  runner sidecar, ref-0036);
  a pass below 0.26 is reported as a reproduction at strict-small, never a win. G, the teacher, is
  reported, not compared. Every number carries the tier label. Never-seen-character leg (critic ref-0001; instrument
  ref-0023 / ref-0025): every gain is also read on the v2 twin @ ref-0026's ex_unseen_chars number (DEV without the
  10,277 words holding a character absent from train_10M), v2 against v2; a pass needs both legs, a missing leg is
  PENDING. The harness per-word cross-check (babylm.a4.unseen, 10,275 words) is reported, never binding. Same
  condition (ref-0027 / ref-0001): the student, N0', N1, N2 and the base run under one condition: the 10M screen runs
  SOLO against the solo seed-0 base (--packed: a packed base and the packed floor). The gate may run packed (labels
  only; --packed, with a packed oracle). Gate and verdict refuse while any score they read is VOID
  (babylm.a4.verdict.STEP_RULE: the scored checkpoint's step must be the run's final step) or crosses environments
  (babylm.a4.verdict.ENV_RULE, critic ref-0037, 15:34Z, ref-0038 and ref-0039: the torch/CUDA build, the GPU model and
  the packing instrument): the batch's runs and its G share one environment, and the oracle, the base, the size-M
  reference and the E*_raw base are in it (the E10 base: v2e, torch 2.7, solo; the 0.002 single-pair admission is
  retracted), each with its instrument declared from its ticket (--ref_instrument). --oracle_dev, a bare number,
  carries no build record and is refused: pass --oracle_score. A verdict is a screen only (babylm.a4.verdict.SCOPE).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

TIER = "closed-system augmentation (official FAQ), outside R0.2"
LABELS = (TIER, "known ideas, reproduction at strict-small", "compute-heavy", "never a strict-small (R0) entry")
GATE_RUNS = {"arm": "a4_t2g", "base@1M": "a4_t2g_base", "N0'": "a4_t2g_n0", "N1": "a4_t2g_n1", "N2": "a4_t2g_n2",
             "FLOP ref (size M @1M)": "a4_t2g_M"}
GATE_G = ("G@1M (teacher)", "a4_t2g_G")
GATE = {"ref": "F1 card Part C T2 allocation (ref-0109), critic ref-0108, G annealed (experimenter 04:47Z); mini-ruler labels "
               "only (ref-0110)", "phi_min": 0.20, "lead_min": 0.016}  # 0.016 kept on the twin (experimenter 05:19Z)
STAGE_RUNS = {"arm": "a4_t2", "N0'": "a4_t2_n0", "N1": "a4_t2_n1", "N2": "a4_t2_n2"}
STAGE_G = ("G (teacher)", "a4_t2_G")
FLOP_REF = "FLOP ref (size M)"
PROMOTE_GAIN = 0.26


def kd_rows_terms(zs: torch.Tensor, zt_rows: torch.Tensor | None, y: torch.Tensor, rows: torch.Tensor, alpha: float,
                  tau: float) -> tuple[torch.Tensor, dict[str, float]]:
    """zs [B, T, Vs] student logits; rows [B] bool, True = a synthetic row (KD term); zt_rows [R, T, Vt] the teacher's
    logits on those rows only (R = rows.sum(), Vt <= Vs; None when R = 0); y [B, T] targets (-1 = unscored).
    Returns the mean loss over all scored targets and detached stats: kd_ce = CE over all scored targets (the plain
    path's loss, for the train log), ce_real / ce_syn per kind, kd_kl, kd_t_label and kd_t_ent over synthetic targets,
    syn_tok = the synthetic share of the scored targets."""
    zs = zs.float()
    B, T, Vs = zs.shape
    rows = rows.to(device=zs.device, dtype=torch.bool)
    m = y >= 0
    n = m.sum().clamp_min(1)
    ce_tok = F.cross_entropy(zs.view(B * T, Vs), y.reshape(B * T), ignore_index=-1, reduction="none").view(B, T)
    syn = m & rows[:, None]
    real = m & ~rows[:, None]
    ce_real, ce_syn = (ce_tok * real).sum(), (ce_tok * syn).sum()
    n_syn = int(syn.sum())
    st = {"kd_ce": float((ce_real + ce_syn).detach() / n), "ce_real": float(ce_real.detach() / real.sum().clamp_min(1)),
          "ce_syn": float(ce_syn.detach() / max(n_syn, 1)), "syn_tok": float(n_syn / float(n)),
          "kd_kl": 0.0, "kd_t_label": 0.0, "kd_t_ent": 0.0}
    if n_syn == 0 or alpha <= 0:
        return (ce_real + ce_syn) / n, st
    if zt_rows is None or zt_rows.shape[0] != int(rows.sum()):
        raise ValueError("zt_rows must hold the teacher's logits of exactly the synthetic rows")
    zt = zt_rows.float()
    Vt = zt.shape[-1]
    if Vt > Vs:
        raise ValueError(f"teacher vocab {Vt} > student vocab {Vs}")
    zs_m, zt_m = zs[syn], zt[m[rows]]               # [n_syn, V], both in (row, position) order
    lq = F.log_softmax(zs_m / tau, dim=-1)[:, :Vt]  # student log-probs on the teacher's ids (full normaliser)
    lp = F.log_softmax(zt_m / tau, dim=-1)
    kl = (lp.exp() * (lp - lq)).sum(-1).sum()
    loss = (ce_real + (1.0 - alpha) * ce_syn + alpha * (tau ** 2) * kl) / n
    with torch.no_grad():
        lpt1 = F.log_softmax(zt_m, dim=-1)
        ys = y[syn]
        st["kd_t_label"] = float(lpt1.gather(1, ys.clamp_max(Vt - 1).unsqueeze(1)).exp().mean())
        st["kd_t_ent"] = float(-(lpt1.exp() * lpt1).sum(-1).mean())
    st["kd_kl"] = float(kl.detach() / n_syn)
    return loss, st


# ------------------------------------------------------------------------------------------------------------------
# readouts
# ------------------------------------------------------------------------------------------------------------------
TWIN_SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "dev_bpw_twin.py"  # ref-0049 here; ref-0026 on the VM


def merge_twin(run: str | Path) -> float | None:
    """score.json += dev_nats_per_word_twin (and, from the twin @ ref-0026, dev_nats_per_word_twin_ex_unseen) from
    score/dev_bpw_twin.json, the keys the executor merges at copy-back; returns v2, or None when the twin did not
    run."""
    from babylm.a4 import verdict as VD
    sd = Path(run) / "score"
    tw, sc = sd / "dev_bpw_twin.json", sd / "score.json"
    if not (tw.exists() and sc.exists()):
        print(f"[twin] {run}: no {'score.json' if tw.exists() else 'dev_bpw_twin.json'}")
        return None
    t, s = json.loads(tw.read_text()), json.loads(sc.read_text())
    s[VD.METRICS["v2"]] = t["nats_per_word"]
    if t.get(VD.GUARDED["twin_key"]) is not None:
        s[VD.GUARDED["key"]] = t[VD.GUARDED["twin_key"]]
    s["dev_twin"] = {"protocol": t.get("protocol"), "separators": t.get("separators"),
                     "script_sha256": hashlib.sha256(TWIN_SCRIPT.read_bytes()).hexdigest(),
                     **{k: t.get(k) for k in ("unseen_chars_count", "unseen_chars_words", "unseen_chars_sha256")}}
    sc.write_text(json.dumps(s, indent=1))
    print(f"[twin] {Path(run).name}: v1 {s.get(VD.METRICS['v1'])} v2 {t['nats_per_word']} "
          f"v2 without never-seen-char words {s.get(VD.GUARDED['key'])}")
    return t["nats_per_word"]


def _used_splits(runs: Path, names: list[str]) -> dict[str, str]:
    from babylm.a4 import verdict as VD
    return {h: lab for n in names if (runs / n / "result.json").exists() for h, lab in VD.synth_splits(runs / n).items()}


def _rows(runs: Path, data: Path, names: list[str]) -> dict:
    from babylm.a4 import ledger as LG
    lg = data / "ledger.json"
    if lg.exists():
        return {r["run"]: r for r in json.loads(lg.read_text())["arms"]}
    return {n: LG.arm_row(runs / n, None) for n in names if (runs / n / "result.json").exists()}


def _teacher(runs: Path, g: tuple[str, str], metric: str) -> dict:
    from babylm.a4 import verdict as VD
    s = VD.score(runs / g[1], metric)
    return {"name": g[0], "run": g[1], **({k: s[k] for k in ("dev", "v1", "v2", "blimp")} if s else {"dev": None})}


def _home(runs: Path, scores: dict[str, dict | None], g: tuple[str, str]) -> dict[str, dict]:
    """The batch's environments for ENV_RULE: its scored runs and its G (the lineage every student starts from)."""
    from babylm.a4 import verdict as VD
    return {**{k: s["env"] for k, s in scores.items() if s},
            **({g[0]: VD.run_env(runs / g[1])} if (runs / g[1]).is_dir() else {})}


def gate(runs_root: str | Path, data: str | Path, oracle: str | Path | float, metric: str | None = None,
         lead_min: float | None = None, instrument: str = "solo", declared: dict[str, str] | None = None) -> dict:
    """oracle: the 10M base's score.json (a number has no build record: ENV_RULE refuses it); instrument: the gate
    batch's (its ticket's); declared: {"oracle": its instrument}, from its ticket (ENV_RULE)."""
    from babylm.a4 import verdict as VD
    metric = metric or VD.PRIMARY
    th = {**GATE, "metric": metric, **({} if lead_min is None else {"lead_min": lead_min})}
    lead_min = th["lead_min"]
    runs, data = Path(runs_root), Path(data)
    names = [*GATE_RUNS.values(), GATE_G[1]]
    rep = {"label": "mini-ruler proxy (labels only, critic ref-0110); " + TIER, "rule": th,
           "audits": VD.audits(data, _used_splits(runs, names)), "teacher": _teacher(runs, GATE_G, metric)}
    sc = {k: VD.score(runs / v, metric) for k, v in GATE_RUNS.items()}
    o = {"dev": float(oracle)} if isinstance(oracle, (int, float)) else VD.score_file(oracle, metric)
    rep["step_check"] = VD.void_check({**sc, "oracle": o, GATE_G[0]: VD.score(runs / GATE_G[1], metric)})
    rep["env_check"] = VD.env_check(_home(runs, sc, GATE_G), {"oracle": o}, instrument, declared)
    rep["dev"] = {k: None if s is None else s["dev"] for k, s in sc.items()}
    rep["dev_v1"] = {k: None if s is None else s["v1"] for k, s in sc.items()}
    rep["blimp_fast"] = {k: None if s is None else s["blimp"] for k, s in sc.items()}
    rep["oracle_dev"] = o["dev"] if o else None
    missing = [k for k, d in rep["dev"].items() if d is None] + ([] if rep["oracle_dev"] else ["oracle"])
    kill = []
    arm, b1 = rep["dev"]["arm"], rep["dev"]["base@1M"]
    if arm is not None and b1 is not None and rep["oracle_dev"]:
        g1 = b1 - rep["oracle_dev"]
        rep["G1"] = g1
        rep["phi"] = {k: (b1 - d) / g1 for k, d in rep["dev"].items() if d is not None and k != "base@1M"}
        if round(rep["phi"]["arm"], 6) < th["phi_min"]:
            kill.append(f"phi_stu {rep['phi']['arm']:.3f} < {th['phi_min']:.2f}")
    comps = {k: d for k, d in rep["dev"].items() if k != "arm" and d is not None}
    if arm is not None and comps:
        comp = min(comps, key=comps.get)
        lead = comps[comp] - arm
        rep.update({"binding_comparator": comp, "lead": lead, "lead_over": {k: d - arm for k, d in comps.items()},
                    "guard": VD.guard(sc["arm"], sc[comp], metric)})
        if round(experimenter, 6) < lead_min:
            kill.append(f"leads the binding comparator ({comp}) by {lead:+.4f} < {lead_min:.4f}")
    rep["kill"], rep["missing"] = kill, missing
    ok = (rep.get("guard") or {}).get("ok")
    rep["decision"] = ("killed (proxy)" if kill else
                       f"NO READOUT: missing {missing}" if missing else
                       "protocol-artifact (guard ref-0020): no proceed" if ok is False else
                       "NO READOUT: the guard needs v1 and v2" if ok is None else
                       "proceed to the 10M stage (mini-ruler proxy pass: a label, not a verdict)")
    return rep


def verdict(runs_root: str | Path, data: str | Path, base_score: str | Path, flop_ref_score: str | Path | None = None,
            th: dict | None = None, data_dir: str | None = None, calib: str | Path | None = None,
            estar_raw: str | Path | None = None, declared: dict[str, str] | None = None,
            vm_evidence: str | Path | None = None) -> dict:
    """th: babylm.a4.verdict.thresholds(...) (default: the primary metric's registered floor, solo; its condition is
    the batch's instrument); data_dir: raw/ and bin/dev.* for the harness cross-check of the never-seen-character leg;
    calib, estar_raw: the promotion gates' inputs (babylm.a4.verdict.gates: calib_control's json, names base,
    estar_cal, flop_ref and the runs' dir names); declared: {reference: its instrument}, from its ticket (ENV_RULE);
    vm_evidence: executor 3's instrument table (babylm.a4.verdict.CORROBORATION), which promotion needs."""
    from babylm.a4 import verdict as VD
    th = th or VD.thresholds()
    metric = th["metric"]
    runs, data = Path(runs_root), Path(data)
    names = list(STAGE_RUNS.values())
    rep = {"labels": list(LABELS), "thresholds": th, "audits": VD.audits(data, _used_splits(runs, names)),
           "teacher": _teacher(runs, STAGE_G, metric)}
    rows = _rows(runs, data, names)
    arm = VD.score(runs / STAGE_RUNS["arm"], metric)
    refs = {"base": VD.score_file(base_score, metric),
            **{k: VD.score(runs / v, metric) for k, v in STAGE_RUNS.items() if k != "arm"},
            FLOP_REF: VD.score_file(flop_ref_score, metric) if flop_ref_score else None}
    es = VD.score_file(estar_raw, metric) if estar_raw else None
    if estar_raw and es is None:
        raise SystemExit(f"no E*_raw base score at {estar_raw}")
    rep["step_check"] = VD.void_check({"arm": arm, **refs, "estar_raw": es,
                                       STAGE_G[0]: VD.score(runs / STAGE_G[1], metric)})
    own = {"arm": arm, **{k: refs[k] for k in STAGE_RUNS if k != "arm"}}
    home, rs = _home(runs, own, STAGE_G), {"base": refs["base"], FLOP_REF: refs[FLOP_REF], "estar_raw": es}
    rep["env_check"] = VD.env_check(home, rs, th["condition"], declared)
    named = {**STAGE_RUNS, STAGE_G[0]: STAGE_G[1]}  # the VM evidence names runs by their dir names
    rep["instrument_check"] = VD.corroborate({**{named[k]: e for k, e in home.items()},
                                              **{VD.ref_run(k, s): s["env"] for k, s in rs.items() if s}},
                                             th["condition"], VD.vm_evidence(vm_evidence))
    rep["scope"] = VD.SCOPE
    row = rows.get(STAGE_RUNS["arm"], {})
    out = {"arm": STAGE_RUNS["arm"], "dev": arm and arm["dev"], "dev_v1": arm and arm["v1"],
           "dev_v2": arm and arm["v2"], "dev_v2x": arm and arm["v2x"], "blimp_fast": arm and arm["blimp"],
           "flops_vs_base": row.get("flops_vs_base"),
           "words_capped": row.get("words_capped"), "within_cap": row.get("within_cap"), "refs": {}, "kill": [],
           "inconclusive": [], "artifact": [], "guard_pending": [], "leg_pending": [], "missing": [], "gates": None,
           "promotion": None}
    if arm is None or arm["dev"] is None:
        out["verdict"] = f"NO VERDICT: the arm has no {metric} score"
        return {**rep, **out}
    for k, s in refs.items():
        if not s or s["dev"] is None:
            out["missing"].append(k)
            continue
        VD.grade(out, k, s, arm, th, null=k in ("N0'", "N1", "N2"))  # stream-randomized nulls (critic ref-0028)
    if out["refs"]:  # the never-seen-character leg (critic ref-0001): a pass needs both legs
        VD.apply_leg(out, {"arm": arm, **{k: refs[k] for k in out["refs"]}}, th, data_dir)
        order = sorted(out["refs"], key=lambda k: out["refs"][k]["dev"])
        out["binding_comparator"] = order[0]
        with_b = [k for k in order if out["refs"][k]["blimp_fast"] is not None]  # the BLiMP line: best DEV with a score
        if out["blimp_fast"] is None:
            out["kill"].append("the arm has no BLiMP-fast score")
        elif with_b:
            d = out["refs"][with_b[0]]["blimp_fast"] - out["blimp_fast"]
            out["blimp_ref"], out["blimp_drop"] = with_b[0], d
            if round(d, 6) > th["blimp_kill"]:
                out["kill"].append(f"BLiMP-fast {d:.2f} below {with_b[0]} > {th['blimp_kill']:.1f}")
    if out["within_cap"] is False:
        out["kill"].append("corpus exposure over the T2 cap")
    stage1 = all(out["refs"].get(k, {}).get("status") in ("pass", "inconclusive") for k in ("base", "N0'"))
    out["clears_base_and_n0"] = stage1
    v = VD.decide(out, th)  # a kill, a protocol-artifact or numbers-only shows before every reference is in
    if out["missing"] and v not in ("KILL", "PROTOCOL-ARTIFACT") and not v.startswith("NUMBERS"):
        nxt = " (launch batch 2: gcp/a4_t2_b2.json)" if stage1 and {"N1", "N2"} & set(out["missing"]) else ""
        v = f"PENDING: missing {out['missing']}{nxt}"
    out["verdict"] = v
    names = {"arm": STAGE_RUNS["arm"], **{k: v for k, v in STAGE_RUNS.items() if k != "arm"},
             "base": VD.CALIB_NAMES["base"], FLOP_REF: "flop_ref"}  # calib json names
    VD.gates(out, names, {"arm": arm, **{k: refs[k] for k in out["refs"]}}, th, VD.calib_runs(calib), es,
             rep["instrument_check"]["missing"])
    gain = out["refs"][out["binding_comparator"]]["gain"] if out.get("binding_comparator") else None
    out["gain_over_binding"] = gain
    big = gain is not None and round(gain, 6) >= PROMOTE_GAIN
    out["promote"] = out["verdict"] == "PASS" and big and out["promotion"] == "promote"
    if out["verdict"] == "PASS":
        out["next"] = (f"screen pass below the promotion bar ({PROMOTE_GAIN}): report as a reproduction at "
                       "strict-small, never a win" if not big else
                       "promotion: 3 seeds, size M, official eval (Y-a for any 10x wording)" if out["promote"] else
                       f"promotion gates: {out['promotion']}")
    return {**rep, **out}


def _f(x: float | None, f: str = "+.4f") -> str:
    return "?" if x is None else format(x, f)


def _table(r: dict) -> str:
    m = r["thresholds"]["metric"]
    lines = [f"| reference | ref DEV ({m}) | arm DEV | gain | gain v1 / v2 | gain without never-seen-char words: v2 "
             "twin / harness cross-check | status | BLiMP-fast arm / ref |", "|---|---|---|---|---|---|---|---|"]
    for k, x in r.get("refs", {}).items():
        bl = " / ".join(_f(v, ".2f") for v in (r["blimp_fast"], x["blimp_fast"]))
        lines.append(f"| {k} | {x['dev']:.4f} | {r['dev']:.4f} | {x['gain']:+.4f} | {_f(x['gain_v1'])} / "
                     f"{_f(x['gain_v2'])} | {_f(x.get('gain_ex_unseen'))} / {_f(x.get('gain_ex_unseen_harness'))} | "
                     f"{x['status'] or '-'} | {bl} |")
    t = r.get("teacher") or {}
    lines.append(f"| {t.get('name', 'G')} (not compared) | {_f(t.get('dev'), '.4f')} | | | | | | "
                 f"{_f(t.get('blimp'), '.2f')} |")
    if r.get("gates"):
        from babylm.a4 import verdict as VD
        lines += ["", "| arm | promotion gate (its metric's kill line, both legs) | gain full | gain without "
                  "never-seen-char words | lam-only full / without |", "|---|---|---|---|---|",
                  *VD.gate_rows(r["arm"], r["gates"])]
    return "\n".join(lines)


def _gate_table(r: dict) -> str:
    m = r["rule"]["metric"]
    lines = [f"| run | DEV ({m}) | DEV v1 | phi | lead of the arm | BLiMP-fast |", "|---|---|---|---|---|---|"]
    for k, d in r["dev"].items():
        phi = (r.get("phi") or {}).get(k)
        lead = (r.get("lead_over") or {}).get(k)
        lines.append(f"| {k} | {_f(d, '.4f')} | {_f(r['dev_v1'].get(k), '.4f')} | {_f(phi, '.3f')} | "
                     f"{_f(experimenter)} | {_f(r['blimp_fast'].get(k), '.2f')} |")
    t = r.get("teacher") or {}
    lines.append(f"| {t.get('name', 'G')} (not compared) | {_f(t.get('dev'), '.4f')} | {_f(t.get('v1'), '.4f')} | "
                 f"| | {_f(t.get('blimp'), '.2f')} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    from babylm.a4 import verdict as VD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("twin")
    t.add_argument("runs", nargs="+")
    g = sub.add_parser("gate")
    g.add_argument("--runs_root", default="babylm/runs")
    g.add_argument("--data", default=None, help="default RUNS_ROOT/a4_t2g_data")
    o = g.add_mutually_exclusive_group(required=True)
    o.add_argument("--oracle_score", help="score.json of the E10 base at 10M (the 10x oracle) in the gate's "
                                          "environment: v2e (torch 2.7; base_r1 s0 is refused, ENV_RULE)")
    o.add_argument("--oracle_dev", type=float, help="its DEV as a bare number: no build record, refused by ENV_RULE "
                                                    "(kept for the pinned executor files' on-VM readout)")
    ins = dict(nargs="+", action="extend", default=[], metavar="NAME=solo|packed")
    g.add_argument("--ref_instrument", **ins, help="the oracle's packing instrument, from its ticket (ENV_RULE, "
                   "critic ref-0039): oracle=solo for v2e")
    g.add_argument("--packed", action="store_true", help="the gate batch ran packed (labels only); its oracle must "
                   "be packed too (ENV_RULE)")
    g.add_argument("--metric", choices=sorted(VD.METRICS), default=VD.PRIMARY)
    g.add_argument("--lead_min", type=float, default=None, help=f"default {GATE['lead_min']}")
    g.add_argument("--out", default=None)
    v = sub.add_parser("verdict")
    v.add_argument("--runs_root", default="babylm/runs")
    v.add_argument("--data", default=None, help="default RUNS_ROOT/a4_t2_data")
    v.add_argument("--base_score", required=True, help="score.json of the E10 base, seed 0 (exposure-matched), in "
                                                       "the batch's environment: v2e (torch 2.7, solo; ENV_RULE)")
    v.add_argument("--ref_instrument", **ins, help="each reference's packing instrument, from its ticket (ENV_RULE, "
                   "critic ref-0039): base=solo, flop_ref=solo with --flop_ref_score, estar_raw=solo with --estar_raw")
    v.add_argument("--flop_ref_score", default=None, help="score.json of the size-M FLOP reference at 10M")
    v.add_argument("--metric", choices=sorted(VD.METRICS), default=VD.PRIMARY)
    v.add_argument("--pass_gain", type=float, default=None, help="default: the metric's registered floor")
    v.add_argument("--kill_gain", type=float, default=None)
    v.add_argument("--blimp_kill", type=float, default=None)
    v.add_argument("--packed", action="store_true", help="packed screen with a packed base (critic ref-0001 packed "
                   "floor); the T2 screen runs solo against the solo seed-0 base")
    v.add_argument("--data_dir", default=None, help="raw/ and bin/dev.* for the leg's harness cross-check")
    v.add_argument("--calib", default=None, help="calib_control.py @ ref-0033 --format v2 --json output: base=<E10 "
                   "base> estar_cal=<base at E*_cal> flop_ref=<size-M reference> and each run by its dir name")
    v.add_argument("--estar_raw", default=None, help="score.json of the same-seed base at E*_raw")
    v.add_argument("--estar", default=None, help="which E*_raw / E*_cal were picked and by whom (recorded)")
    v.add_argument("--vm_evidence", default=None, help="executor 3's instrument table (critic ref-0035; "
                   "babylm.a4.verdict.EVIDENCE columns), one row per compared run: promotion needs it, or a solo "
                   "run's complete instrument.json sidecar (critic ref-0036)")
    v.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "twin":
        return {r: merge_twin(r) for r in a.runs}
    runs = Path(a.runs_root)
    if a.cmd == "gate":
        r = gate(runs, a.data or runs / "a4_t2g_data", a.oracle_dev if a.oracle_dev is not None else a.oracle_score,
                 a.metric, a.lead_min, "packed" if a.packed else "solo",
                 VD.instruments(a.ref_instrument, {"oracle": "oracle"}))
        text = _gate_table(r) + f"\n\nT2 gate ({a.metric}): {r['decision']}" \
            + (f" ({'; '.join(r['kill'])})" if r["kill"] else "") + f" [{r['label']}]" \
            + f"\n{VD.env_line(r['env_check'])}"
    else:
        th = VD.thresholds(a.metric, a.pass_gain, a.kill_gain, a.blimp_kill, a.packed)
        r = verdict(runs, a.data or runs / "a4_t2_data", a.base_score, a.flop_ref_score, th, a.data_dir, a.calib,
                    a.estar_raw, VD.instruments(a.ref_instrument, {"base": "base", "flop_ref": FLOP_REF,
                                                                  "estar_raw": "estar_raw"}), a.vm_evidence)
        r["estar"] = {"note": a.estar, "raw_base": a.estar_raw, "calib": a.calib}
        why = r["kill"] or r["artifact"] or r["inconclusive"]
        text = _table(r) + f"\n\nT2 arm {r['arm']}: {r['verdict']} [{th['ref']}; {th['condition']}]" \
            + (f" ({'; '.join(why)})" if why else "") + (f"\n{r['next']}" if r.get("next") else "") \
            + f"\nLabels: {'; '.join(r['labels'])}; FLOPs x base {r.get('flops_vs_base')}" \
            + (f"\nE*: {a.estar}" if a.estar else "") + f"\n{VD.env_line(r['env_check'])}" \
            + f"\n{VD.evidence_line(r['instrument_check'])}\nscope: {VD.SCOPE}"
    print(json.dumps(r["audits"], indent=1))
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))
        Path(a.out).with_suffix(".md").write_text(text + "\n")
    return r


if __name__ == "__main__":
    main()
