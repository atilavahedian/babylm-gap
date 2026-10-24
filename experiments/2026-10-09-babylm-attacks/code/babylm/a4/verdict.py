"""Screen verdict of attack A4 by its registered kill rule (card "Kill rule"; critic amendments 1, 3, 4 and 5).

    python -m babylm.a4.verdict --base_score BASE/score.json --ref_instrument base=solo [estar_raw=solo] \
        [--metric v2|v1] [--pass_gain G] [--kill_gain G] [--blimp_kill 2.0] [--packed] [--runs_root babylm/runs] \
        [--data babylm/runs/a4_data] [--data_dir DATA] [--calib CALIB.json] [--estar_raw ESTAR_RAW/score.json] \
        [--vm_evidence INSTRUMENTS.tsv] [--estar NOTE] [--out .../verdict.json]

Metric v2 is primary (critic 05:02Z, rule ref-0018): DEV = dev_nats_per_word_twin, the blank-line twin
(babylm/eval/dev_bpw_twin.py: dev_bpw.py on the fp32 HF export, full DEV, the scored checkpoint, with each run
of blank lines scored as one document break; read from score.json or the dev_bpw_twin.json beside it; never result.json's
full_eval); gain over X = DEV(X) - DEV(arm). A screen is one arm seed against one base seed, on the interim v2 screen
floor (critic ref-0019, 0.0036): a gain >= 0.010 nats/word passes, 0.0074 <= gain < 0.010 is inconclusive (it may buy 2
more seeds of the arm, once per attack, as R3 trials; the claim then reads the arm's 3-seed solo mean against solo
base pools in the arms' environment, raw and calibrated, on the claim floors: NEXT_SEEDS), below 0.0074 kills; a
BLiMP-fast drop above 2.0 points kills. Null comparisons use the same thresholds. The blank-line guard (ref-0020)
holds by construction under v2 (critic ref-0021).
v1 = dev_nats_per_word (dev_bpw.py) stays reported beside it. --metric v1 reads under v1 for information only, on the
noise floor v1 (critic ref-0022; per-seed SD of the round-1 winner, seeds 0-2: 0.0058 nats/word, BLiMP-fast 0.71: pass
>= 0.016, inconclusive from 0.012), and there the guard applies: a gain counts only if its v2 gain is > 0 and within
0.0058 of its v1 gain, else "protocol-artifact" (no pass, no promotion; the number is still reported).
Never-seen-character leg (critic ref-0001, binding on every screen): each gain is also read on DEV without the words
that hold a character absent from the train_10M raw text. Canonical instrument (critic ref-0023 / ref-0024 / ref-0025,
experimenter 06:43Z): the v2 twin @ ref-0026's ex_unseen_chars_nats_per_word (the literal 85-character rule over 10,277 words,
the span token rule, (N - NU) / (W - WU)), v2 against v2, read from score.json (dev_nats_per_word_twin_ex_unseen, merged
at copy-back) or the dev_bpw_twin.json beside it; a twin json counting other characters or words is not read. A pass
must clear the bar on both legs; a leg below the kill line kills and a leg in the band makes the reading inconclusive;
a missing leg reads PENDING (critic ref-0025). The harness per-word cross-check (babylm.a4.unseen = unseen_chars.py,
10,275 words, RUN/pw/pw_devfull_final.npy; --data_dir holds raw/ and bin/dev.*) is reported beside it, never binding.
Same condition (critic ref-0027 / ref-0001): arm, nulls and base run under one execution condition. A4 runs SOLO (one
job per GPU), against the solo seed-0 base, on the solo floor above; --packed reads a packed screen (with a packed base)
on the packed floor: pass >= 0.0144, inconclusive 0.0102-0.0144, kill below 0.0102. Claims run solo only. The
instrument (solo or packed) is part of the environment (ENV_RULE).
The judged arm is the one of A4a (a4_a) and A4b (a4_b) with the lower DEV (amendment 5: one arm, picked on DEV). Its
references: its N0' twin, the round-1 base (same seed), its N1 and N2, and a FLOP-matched reference when its FLOPs
exceed 1.25x its twin's (registered: N0'_a for A4b). The comparator (amendment 1) = the better of N0' and the base:
the BLiMP-fast line is read against it. KILL if any reference kills, BLiMP-fast drops too far or the arm breaks R0;
else INCONCLUSIVE if any reference is in the band; else PASS. A missing score = no verdict. The other arm's numbers
are reported, not judged. Null bar (critic ref-0028): against a null (N0', N1, N2, N0'_a: stream-randomized, noisy) a
single-seed pass needs a gain >= 0.020 on each leg; below it, down to the kill line, the reading is inconclusive and 2
more seeds of the arm and that null decide.
Promotion gates (critic ref-0011 amendment 2 / ref-0029; ref-0030 (3) / ref-0031), read after the screen and never
changing its status, every number on both legs, each on the kill line of its own metric (critic ref-0032, CAL_T: a bar
read on calibrated_t is the raw bar x the leg's calibrated_t/raw SD ratio): (1) calibration: the calibrated_t gain (T x
lam, cross-fitted on DEV document parity; calib_control.py @ ref-0033 --format v2 --json over dev_bpw_dump.py @ ref-0034
--twin --temps 0.9,0.95,1.05,1.1,1.15,1.2 dumps of full DEV, base and arm on one GPU type: --calib) over every graded
reference; below the calibrated_t kill line (solo 0.0095 full / 0.0094 guarded, packed 0.0131 provisional) a pass is
"calibration-equivalent". (2) Epochs: the raw gain over the same-seed base at E*_raw (--estar_raw; the raw kill line
0.0074, packed 0.0102) and the calibrated_t gain over the same-seed base at E*_cal (calib name estar_cal; the
calibrated_t kill line); below its line a pass is "epoch-equivalent". Either label means no promotion; a missing number
or a T pick at the grid edge (ref-0011 amendment 1: the grid is extended for every model first) leaves it pending, as
does (3) an instrument not corroborated from VM evidence (CORROBORATION, critic ref-0035; --vm_evidence, or a complete
runner sidecar, ref-0036). The calib json names the E10 base "base" and every run by its dir name; each raw value must
equal its twin's number. The lam-only calibrated gain is reported beside.
Audits first (amendment 4): no table without a leak audit covering every synthetic split of the arms.
Step rule (critic, experimenter 14:48Z; the K3 bug): a score is VOID unless its scored checkpoint's step (its curves.csv row) is
the run's final step (result.json steps); the readout refuses while any score it reads is VOID (STEP_RULE).
Environment rule (critic ref-0037, ref-0038, 76151841, ref-0039; experimenter 15:07Z, 19:19Z; ENV_RULE): the torch/CUDA build
(config.json "torch", e.g. 2.7.0+cu126), the GPU model (result.json "gpu", the device name torch reports; critic 15:34Z:
another model is another environment, another instance of the same model is not) and the packing instrument (solo or
packed; critic 48436034 / ref-0039) are part of the recipe. The A4 runs and the generators they start from share one
environment, and every reference (--base_score, the E10 base: v2e, the E10 s0 run in torch 2.7, solo; --estar_raw) is in
it, or the readout refuses. Run files do not record packing: the batch's instrument is its ticket's (solo; --packed for
a packed ticket) and each reference's is declared from its ticket (--ref_instrument base=solo estar_raw=solo; executor
3's run table names each run's VM, and a packed ticket's VM name carries -p2-); a reference with no declared instrument
or another one is refused. The ticket is enough for a screen; promotion and every claim also need each compared run's
instrument corroborated by executor 3 from VM evidence (critic ref-0035: --vm_evidence, one row per run: the VM's job
list or startup.log showing one job at a time, its machine type and GPU name); for a solo run a complete instrument.json
sidecar (the runner's record, executor 3 @ ref-0040) stands in for the row (critic ref-0036, SIDECAR_RULE: written on
the VM by startup.sh with its commit, never regenerated, solo stated positively and, critic ref-0041, through the whole
training window, its VM, machine type and GPUs). Any disagreement between a row, a sidecar, the ticket and the run's GPU
record VOIDs the read. The 0.002 single-pair admission is retracted (critic ref-0038: same-seed reruns differ by up to
0.006); a cross-environment read would have to clear its bar + ENV_SYS (+ PACK_SYS across instruments) on both legs, raw
and calibrated_t, so this readout reads within one environment and one instrument only. E*_cal's environment is checked
through --estar_raw (the calib json's estar_cal must be that run: E*_raw = E*_cal = 6, experimenter 14:28Z), else promotion
stays pending. A read against the single-seed E10 base is a screen only (SCOPE): a claim needs 3 seeds per side in the
arms' environment (experimenter 19:19Z) and a solo E10 pool there of 3 runs or more (torch 2.7: v2e, v2f and one more; critic
ref-0039) for its floors; the claim prints both comparisons, against that E10 pool and against the E* pool (E6 s0-s4),
and the E* pool binds: a gain that clears the E10 pool but not the E* pool is epoch-equivalent and not promoted (experimenter
19:55Z; NEXT_SEEDS).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from babylm.a4 import ledger as LG
from babylm.a4 import unseen as UN
from babylm.harness import cycle as CY

FAMILIES = {"a": {"arm": "a4_a", "n0": "a4_n0a", "n1": "a4_n1a", "n2": "a4_n2a"},
            "b": {"arm": "a4_b", "n0": "a4_n0b", "n1": "a4_n1b", "n2": "a4_n2b"}}
FLOP_MATCHED = {"b": "a4_n0a"}  # card, run registration: N0'_a has about A4b's training FLOPs
BLIMP = "blimp/blimp_fast"
METRICS = {"v1": "dev_nats_per_word", "v2": "dev_nats_per_word_twin"}
PRIMARY = "v2"  # critic 2026-10-10 05:02Z: the blank-line twin is primary, v1 stays reported
FLOOR_V1 = {"ref": "critic ref-0022, noise floor v1", "pass_gain": 0.016, "kill_gain": 0.012, "blimp_kill": 2.0}
FLOOR_V2 = {"ref": "critic ref-0019, interim v2 screen floor 0.0036", "pass_gain": 0.010, "kill_gain": 0.0074,
            "blimp_kill": 2.0}
FLOOR_V2_PACKED = {"ref": "critic ref-0001, interim packed v2 screen floor 0.0051", "pass_gain": 0.0144,
                   "kill_gain": 0.0102, "blimp_kill": 2.0}
FLOORS = {"v1": FLOOR_V1, "v2": FLOOR_V2}
GUARDED = {"key": "dev_nats_per_word_twin_ex_unseen", "twin_key": "ex_unseen_chars_nats_per_word", "chars": 85,
           "words": 10_277, "ref": "v2 twin @ ref-0026 ex_unseen_chars, v2 against v2 (critic ref-0023 / ref-0024 / "
           "ref-0025, experimenter 06:43Z)"}  # the never-seen-character leg's binding number
GUARD_TOL = 0.0058  # critic ref-0020 (v1 reads): v2 gain > 0 and within 1x floor v1 of the v1 gain
NEXT_SEEDS = ("inconclusive (critic ref-0019): may buy 2 more seeds of the arm, once per attack, as R3 trials "
              "(make_a4_jobs --seed 1|2 --arm X), then the arm's 3-seed solo mean is judged against solo base pools "
              "of 3 runs or more in the arms' environment and instrument (experimenter 19:19Z; critic ref-0039), and the "
              "claim prints both comparisons (experimenter 19:55Z): the E10 pool (torch 2.7: v2e, v2f and one more) and the "
              "pool at E* (E6 s0-s4; critic ref-0042 / ref-0030 / ref-0031: raw vs the E*_raw pool, calibrated_t vs "
              "the E*_cal pool), raw and calibrated gains both on both legs (ref-0011), each on the claim floor of its "
              "own metric and pool (critic 76151841 / ref-0039 / ref-0032: 2x that pool's SD on the metric being "
              "gated; the torch 2.9 pool's SDs are not borrowed and the E10 floors do not transfer to E6) and on the "
              "arm-variance bar 2 sqrt(sd_arm^2 / 3 + sd_pool^2 / n_pool) on the same metric's SDs (critic 14:30Z / "
              "ref-0043, every claim); the E* pool binds: a gain that clears the E10 pool but not the E* pool is "
              "epoch-equivalent and not promoted (experimenter 19:55Z); the BLiMP-fast 1.4 line reads the 3-seed mean against "
              "the better (higher-BLiMP) of the two pools' means, both reported (critic ref-0044); a null in the band "
              "at the screen has its 3 seeds read as a pool on the same rule (critic ref-0044); every run's "
              "instrument corroborated from VM evidence (critic ref-0035)")
CALIB = {"ref": "critic ref-0011 / ref-0029: calibrated_t binds (softmax(z / T) mixed with lam x unigram, T and lam "
                "picked jointly, cross-fitted on DEV document parity); the lam-only value is reported",
         "tool": "babylm/tools/calib_control.py @ ref-0033 --format v2 --json",
         "dump": "babylm/moonshot/dev_bpw_dump.py @ ref-0034 --twin --temps 0.9,0.95,1.05,1.1,1.15,1.2"}
ESTAR = ("critic ref-0030 / ref-0031: raw gain vs the same-seed base at E*_raw (v2 twin), calibrated_t gain vs the "
         "same-seed base at E*_cal; the screen keeps its registered bars vs E10")
CALIB_NAMES = {"base": "base", "estar_cal": "estar_cal"}  # names in the calib json besides the runs' dir names
NULL_BAR = {"pass_gain": 0.020, "ref": "critic ref-0028: stream-randomized nulls are noisy (the A3 nulls differed by "
            "0.024), so a single-seed arm-vs-null pass below 0.020 reads inconclusive and 2 more seeds of the arm and "
            "the null decide; the kill line stays"}
LEGS = (("full", "v2"), ("guarded", "v2x"))  # calib_control leg -> score_file key
STEP_RULE = ("critic, experimenter 14:48Z (the K3 bug: a run ending short of a milestone was scored from a mid-anneal "
             "checkpoint): a score is VOID unless its scored checkpoint's step is the run's final step; re-score from "
             "resume.pt (EMA off) or rerun at a commit with ref-0045 (chck_final.pt)")
ENV_RULE = ("critic ref-0037 (registered 14:50Z, map 14:54Z; GPU model binding 15:34Z; instrument 48436034 / "
            "ref-0039), experimenter 15:07Z: the torch/CUDA build, the GPU model and the packing instrument (solo or packed) "
            "are part of the recipe; every compared run (arm, nulls, references, the generators the runs start from) "
            "shares one environment (the reference: p2venv torch 2.7.0+cu126 on an NVIDIA A100-SXM4-40GB, solo; "
            "another instance of the same GPU model is the same environment). Run files do not record packing: the "
            "batch's instrument is its ticket's (--packed for a packed one) and each reference's is declared from its "
            "ticket (--ref_instrument NAME=solo|packed). The 0.002 single-pair admission is retracted (critic "
            "ref-0038); a cross-environment read stands only if its margin clears its bar + ENV_SYS on both legs, raw "
            "and calibrated_t (critic 48436034 / 76151841: raw 0.014 full / 0.013 guarded, calibrated_t 0.014 / "
            "0.014), plus PACK_SYS across instruments (critic ref-0046: 0.012 / 0.010), else it needs a "
            "within-environment run. This readout reads within one environment and one instrument only: use a "
            "reference run in the arms' environment (the E10 base: v2e, solo)")
INSTRUMENTS = ("solo", "packed")
CAL_T = {"ref": "critic ref-0032: a bar read on calibrated_t (the promotion gates, every claim) is the raw bar x the "
                "leg's calibrated_t/raw solo E10 SD ratio (full 0.0037 / 0.0029 = 1.28, guarded 0.0038 / 0.0030 = "
                "1.27); packed x1.28 on both legs, provisional until a packed calibrated_t SD exists",
         "solo": {"full": 1.28, "guarded": 1.27}, "packed": {"full": 1.28, "guarded": 1.28}}
SCOPE = ("a screen only (critic ref-0039): no claim until the arms' environment has a solo E10 pool of 3 runs or more "
         "(torch 2.7: v2e, v2f and one more), whose SD on the gated metric sets the claim floors; the torch 2.9 pool's "
         "SD is not borrowed")
EVIDENCE = ("run", "vm", "machine_type", "gpu", "instrument", "evidence")  # --vm_evidence columns (critic ref-0035)
SIDECAR = ("run", "source", "startup_commit", "instrument", "pack_env", "pack_partners", "train_procs_at_launch", "vm",
           "machine_type", "gpu", "gpu_procs_max", "gpu_proc_log", "train_window_unix")  # what the readouts keep of the
# runner's RUN/instrument.json (executor 3 @ ref-0040, ref-0047; critic ref-0036, ref-0041)
SIDECAR_SOURCE = "babylm/gcp/startup.sh write_instrument"
CORROBORATION = ("critic ref-0035: a screen reads on the tickets' declared instruments; promotion and every claim also "
                 "need each compared run's instrument corroborated by executor 3 from VM evidence (the VM's job list "
                 "or startup.log showing one job at a time, solo, or two, packed; its machine type and GPU name): "
                 "--vm_evidence, one row per run, or for a solo run its complete instrument.json sidecar "
                 "(SIDECAR_RULE). Any disagreement between a row, a sidecar, the ticket and the run's GPU record VOIDs "
                 "the read; a run without a complete row or sidecar leaves promotion pending")
SIDECAR_RULE = ("critic ref-0036: the runner's sidecar stands in for a solo run's row only when complete: (a) written "
                "on the VM during the run by startup.sh, which records its own commit (source, startup_commit, not "
                "+dirty); (b) copied back with the run's outputs, never regenerated (one naming another run "
                "conflicts); (c) solo stated positively: PACK recorded as unset or 1 (pack_env), no pack partners, "
                "one training process at launch (train_procs_at_launch), as no pack lines alone is not solo; (d) its "
                "VM, machine type and nvidia-smi GPUs, the run's result.json GPU among them. Runs from before "
                "ref-0040 still need the row. Critic ref-0041: solo means exactly 1 GPU compute process in every "
                "sample of the training window, first train step to final checkpoint write (train_window_unix over "
                "gpu_proc_log; later scoring is excluded by time), and a second process in it VOIDs the read; a "
                "sidecar that cannot isolate the window needs gpu_procs_max = 1 over the job, else the row decides")
LINEAGE = ("a4_G", "a4_GA", "a4_GB")  # G (every A4 run starts from it) and the half-data generators (A4b's text)


def score_file(p: str | Path, metric: str = PRIMARY) -> dict | None:
    """A score.json's numbers: dev (under metric), v1, v2 and v2x (the never-seen-character leg, GUARDED; score.json's
    merged keys, else the dev_bpw_twin.json beside it, whose character and word counts come along), BLiMP-fast, and
    the harness per-word dump for the cross-check."""
    p = Path(p)
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    tw = [t for t in (p.parent / "dev_bpw_twin.json", p.parent / "score" / "dev_bpw_twin.json") if t.exists()]
    t = json.loads(tw[0].read_text()) if tw else {}
    v2 = s.get(METRICS["v2"])
    v2 = t.get("nats_per_word") if v2 is None else v2
    v2x = s.get(GUARDED["key"])
    v2x = t.get(GUARDED["twin_key"]) if v2x is None else v2x
    inst = [t.get("unseen_chars_count"), t.get("unseen_chars_words")] if GUARDED["twin_key"] in t else None
    run = run_of(p)
    out = {"v1": s.get(METRICS["v1"]), "v2": v2, "v2x": v2x, "v2x_counts": inst,
           "blimp": (s.get("fast") or {}).get(BLIMP), "checkpoint": s.get("checkpoint"), "run": s.get("run"),
           "pw": UN.find(p), "steps": step_check(p, s), "env": run_env(run), "run_dir": run and str(run)}
    out["dev"] = out[metric]
    return out


def run_of(p: Path) -> Path | None:
    """The run dir behind a score.json: its folder, or the one above score/ (the first holding config.json)."""
    return next((d for d in (p.parent, p.parent.parent) if (d / "config.json").exists()), None)


def run_env(run: Path | None) -> dict:
    """A run's environment: config.json "torch" (version and CUDA build, e.g. 2.7.0+cu126) and result.json "gpu" (the
    GPU model, torch.cuda.get_device_name), plus the runner's instrument.json sidecar when the run carries one
    (executor 3 @ ref-0040: solo or packed from the job's pack log, its VM, machine type and nvidia-smi GPUs; checked by
    corroborate); "why" when either record is missing."""
    if run is None or not (run / "config.json").exists():
        return {"torch": None, "gpu": None, "why": f"no config.json in {run}" if run else "no config.json beside it"}
    r = run / "result.json"
    out = {"torch": json.loads((run / "config.json").read_text()).get("torch"),
           "gpu": json.loads(r.read_text()).get("gpu") if r.exists() else None}
    if (run / "instrument.json").exists():
        j = json.loads((run / "instrument.json").read_text())
        out["sidecar"] = {k: j[k] for k in SIDECAR if k in j}  # a key it lacks is not recorded (pack_env null is)
    why = [w for w, v in ((f"{run}/config.json records no torch build", out["torch"]),
                          (f"{run}/result.json records no GPU model", out["gpu"])) if not v]
    return {**out, "why": "; ".join(why)} if why else out


def env_name(e: dict) -> str:
    return f"torch {e['torch']} / {e['gpu']}"


def env_check(home: dict[str, dict], refs: dict[str, dict | None], instrument: str = "solo",
              declared: dict[str, str] | None = None) -> dict:
    """ENV_RULE over a readout: home = {name: run_env} of the batch's runs (scored runs and their generators), which
    must share one environment (build and GPU model) and run under instrument (its ticket's: solo or packed); refs =
    {name: score_file dict} of the references from outside the batch, each in that environment and declared (declared:
    {name: instrument}, from its ticket; run files do not record packing) under that instrument. Refuses on a missing
    build or GPU record, a cross-environment read, an undeclared reference instrument or a cross-instrument read;
    {"home" (the build), "gpu" (the GPU model), "instrument", "ref_instruments", "builds", "gpus"}."""
    bare = {"torch": None, "gpu": None, "why": "a bare number: no run, no build record"}
    builds = {**home, **{k: s.get("env") or bare for k, s in refs.items() if s}}
    bad = [f"{k}: {e['why']}" for k, e in builds.items() if e.get("why")]
    if bad:
        raise SystemExit(f"VOID read: {'; '.join(bad)}. {ENV_RULE}")
    he = sorted({(e["torch"], e["gpu"]) for e in home.values()})
    if len(he) > 1:
        raise SystemExit("VOID read: the batch spans environments (" + "; ".join(
            f"{k} {env_name(e)}" for k, e in home.items()) + f"). {ENV_RULE}")
    h = dict(zip(("torch", "gpu"), he[0])) if he else None
    cross = [f"{k} ran on {env_name(s['env'])}" for k, s in refs.items()
             if s and h is not None and (s["env"]["torch"], s["env"]["gpu"]) != he[0]]
    if cross:
        raise SystemExit(f"VOID cross-environment read: {'; '.join(cross)}, the batch on {env_name(h)}. {ENV_RULE}")
    declared = declared or {}
    miss = [k for k, s in refs.items() if s and k not in declared]
    if miss:
        raise SystemExit(f"VOID read: no packing instrument declared for {', '.join(miss)} (run files do not record "
                         "packing: declare each reference's from its ticket, --ref_instrument NAME=solo|packed; critic "
                         f"ref-0039). {ENV_RULE}")
    other = [f"{k} ran {declared[k]}" for k, s in refs.items() if s and declared[k] != instrument]
    if other:
        raise SystemExit(f"VOID cross-instrument read: {'; '.join(other)}, the batch {instrument}. {ENV_RULE}")
    return {"rule": ENV_RULE, "home": h and h["torch"], "gpu": h and h["gpu"], "instrument": instrument,
            "ref_instruments": {k: declared[k] for k, s in refs.items() if s}, "builds": builds,
            "gpus": sorted({e["gpu"] for e in builds.values() if e.get("gpu")})}


def instruments(items: list[str], names: dict[str, str]) -> dict[str, str]:
    """--ref_instrument NAME=solo|packed ... -> {reference key: instrument}; names maps each NAME to its key in the
    readout's references (a later NAME overrides an earlier one)."""
    out = {}
    for it in items:
        n, _, v = it.partition("=")
        if n not in names or v not in INSTRUMENTS:
            raise SystemExit(f"--ref_instrument {it!r}: expected NAME=solo|packed, NAME one of {', '.join(names)}")
        out[names[n]] = v
    return out


def env_line(chk: dict) -> str:
    return (f"environment: torch {chk['home']} / {chk['gpu']} / {chk['instrument']} (ENV_RULE, critic ref-0037 / "
            "ref-0038 / ref-0039)")


def cal_t_bar(bar: float, condition: str, leg: str) -> float:
    """A raw bar read on calibrated_t (CAL_T): solo pass 0.010 / kill 0.0074 -> 0.0128 / 0.0095 full, 0.0127 / 0.0094
    guarded; packed 0.0144 / 0.0102 -> 0.0184 / 0.0131."""
    return round(bar * CAL_T[condition][leg], 4)


def vm_evidence(p: str | Path | None) -> dict[str, dict] | None:
    """Executor 3's instrument table (--vm_evidence; CORROBORATION): tab-separated, '#' lines are comments, a header
    naming EVIDENCE's columns, one row per run (the VM whose copy is scored); {run: row}."""
    if p is None:
        return None
    rd = csv.DictReader([ln for ln in Path(p).read_text().splitlines() if ln.strip() and not ln.startswith("#")],
                        delimiter="\t")
    miss = [c for c in EVIDENCE if c not in (rd.fieldnames or [])]
    if miss:
        raise SystemExit(f"{p}: no {', '.join(miss)} column; expected {', '.join(EVIDENCE)} (critic ref-0035)")
    rows = {}
    for r in rd:
        r = {k: (v or "").strip() for k, v in r.items() if k}
        if r["run"] in rows:
            raise SystemExit(f"{p}: two rows for {r['run']}: keep the one of the VM whose copy is scored")
        if r["instrument"] not in INSTRUMENTS:
            raise SystemExit(f"{p}: {r['run']}: instrument {r['instrument']!r}, expected solo|packed")
        rows[r["run"]] = r
    return rows


def gpu_name(name: str | None) -> str:
    """A GPU name as compared across records (case and the NVIDIA prefix aside: nvidia-smi and torch may differ)."""
    n = " ".join((name or "").lower().split())
    return n.removeprefix("nvidia ")


def sidecar_lacks(sc: dict) -> list[str]:
    """What keeps a runner sidecar from standing in for a solo run's row (SIDECAR_RULE); [] when complete. (b) is the
    runner's (its writer skips a dir that has one); a GPU outside its list and packing_signs conflict instead. The
    commit is launch.sh's last startup.sh commit, "+dirty" when the copy sent differs from it (executor 3 @ ref-0047):
    a dirty copy is no commit."""
    pack, commit = sc.get("pack_env", ""), str(sc.get("startup_commit") or "")
    return [w for w, ok in (("startup.sh as its writer", sc.get("source") == SIDECAR_SOURCE),
                            ("startup.sh's commit", commit),
                            ("a committed startup.sh (it ran +dirty)", not commit.endswith("+dirty")),
                            ("PACK recorded as unset or 1", pack is None or str(pack) == "1"),
                            ("an empty pack-partner list", sc.get("pack_partners") == []),
                            ("one training process at launch", sc.get("train_procs_at_launch") == 1),
                            window_lack(sc),
                            ("its VM", sc.get("vm")), ("its machine type", sc.get("machine_type")),
                            ("its nvidia-smi GPUs", sc.get("gpu"))) if not ok]


def window_counts(sc: dict) -> list[int] | None:
    """critic ref-0041: the GPU compute-process counts sampled inside the run's training window (train_window_unix:
    [first train step, final checkpoint write], over the job's gpu_proc_log of [unix s, count]); None when the
    sidecar cannot isolate the window (either missing, or no sample inside it)."""
    w, log = sc.get("train_window_unix"), sc.get("gpu_proc_log")
    if not (isinstance(w, (list, tuple)) and len(w) == 2 and None not in w and log):
        return None
    return [int(n) for t, n in log if w[0] <= t <= w[1]] or None


def window_lack(sc: dict) -> tuple[str, bool]:
    """sidecar_lacks' solo-throughout item (critic ref-0041): exactly 1 process in every sample of the training window
    (a 0 is a gap, not a second job; more than 1 is packing_signs'), else, with no window, gpu_procs_max = 1."""
    inw = window_counts(sc)
    if inw is None:
        return "a GPU-process maximum of 1 over the job (no training window)", sc.get("gpu_procs_max") == 1
    return "exactly 1 GPU process in every sample of the training window", all(n == 1 for n in inw)


def packing_signs(sc: dict) -> list[str]:
    """A sidecar's record of a second job on the GPU: PACK other than unset or 1, pack partners, more than one training
    process at launch (critic ref-0036) or in the training window (ref-0041). A solo ticket and any of them disagree."""
    pack, n, inw = sc.get("pack_env"), sc.get("train_procs_at_launch"), window_counts(sc) or [0]
    return ([f"PACK={pack}"] if pack not in (None, "") and str(pack) != "1" else []) \
        + ([f"pack partners {', '.join(map(str, sc['pack_partners']))}"] if sc.get("pack_partners") else []) \
        + ([f"{n} training processes at launch"] if isinstance(n, int) and n > 1 else []) \
        + ([f"{max(inw)} GPU processes in the training window"] if max(inw) > 1 else [])


def corroborate(runs: dict[str, dict], instrument: str, rows: dict[str, dict] | None) -> dict:
    """CORROBORATION over a read: runs = {run name: its run_env} (every run the read compares, generators included),
    instrument = the read's (its tickets', ENV_RULE), rows = vm_evidence's. Refuses (VOID) when a row or the run's
    instrument.json sidecar (run_env) says another instrument than the ticket (or, solo, the sidecar records packing),
    when either names a GPU other than the run's result.json GPU, when the row and the sidecar name other VMs or machine
    types, or when the sidecar was written for another run; {"rule", "instrument", "compared" (how many runs), "rows",
    "missing": runs with neither a complete row (vm, machine type, GPU name and evidence) nor, solo, a complete sidecar
    (SIDECAR_RULE), "by_sidecar": runs a complete sidecar corroborates without a complete row, "sidecars": runs carrying
    one}."""
    rows = rows or {}
    bad, miss, side = [], [], []
    for run, e in runs.items():
        sc, r = e.get("sidecar"), rows.get(run)
        signs = packing_signs(sc) if sc and instrument == "solo" else []
        if sc and sc.get("instrument") != instrument:
            bad.append(f"{run}: its ticket says {instrument}, the runner's instrument.json says {sc.get('instrument')} "
                       f"({sc.get('vm')}, pack partners {sc.get('pack_partners')})")
            continue
        if signs:
            bad.append(f"{run}: its ticket says solo, the runner's instrument.json records {', '.join(signs)} "
                       f"({sc.get('vm')})")
            continue
        if sc and sc.get("run") and sc["run"] != run:
            bad.append(f"{run}: the scored copy's instrument.json was written for {sc['run']}")
            continue
        if sc and e.get("gpu") and sc.get("gpu") and gpu_name(e["gpu"]) not in {gpu_name(g) for g in sc["gpu"]}:
            bad.append(f"{run}: result.json records {e['gpu']}, the runner's instrument.json names {sc.get('gpu')}")
            continue
        clash = [f"{w}{r[k]}, the scored copy's instrument.json {sc[k]}"
                 for k, w in (("vm", ""), ("machine_type", "machine type "))
                 if sc and r is not None and r[k] and sc.get(k) and r[k] != sc[k]]
        if clash:
            bad.append(f"{run}: the VM evidence row names {'; '.join(clash)}")
            continue
        row_ok = r is not None and all(r[k] for k in ("vm", "machine_type", "gpu", "evidence"))
        if r is not None and r["instrument"] != instrument:
            bad.append(f"{run}: its ticket says {instrument}, the VM evidence says {r['instrument']} ({r['vm']}: "
                       f"{r['evidence']})")
        elif r is not None and r["gpu"] and e.get("gpu") and gpu_name(r["gpu"]) != gpu_name(e["gpu"]):
            bad.append(f"{run}: result.json records {e['gpu']}, the VM evidence names {r['gpu']} ({r['vm']})")
        elif row_ok:
            continue
        elif sc and instrument == "solo" and not sidecar_lacks(sc):
            side.append(run)
        elif r is not None:
            miss.append(f"{run} (incomplete row)")
        elif not sc:
            miss.append(run)
        else:
            miss.append(f"{run} (no row; " + (f"its instrument.json lacks {', '.join(sidecar_lacks(sc))})"
                                              if instrument == "solo" else "a sidecar stands in for a solo run only)"))
    if bad:
        raise SystemExit(f"VOID read: {'; '.join(bad)}. {CORROBORATION}")
    return {"rule": CORROBORATION, "sidecar_rule": SIDECAR_RULE, "instrument": instrument, "compared": len(runs),
            "rows": {k: rows[k] for k in runs if k in rows}, "missing": miss, "by_sidecar": side,
            "sidecars": sorted(k for k, e in runs.items() if e.get("sidecar"))}


def ref_run(key: str, s: dict) -> str:
    """A reference's run name for the VM evidence: score.json's "run" (the run dir's name on the VM), else the dir's."""
    return s.get("run") or (Path(s["run_dir"]).name if s.get("run_dir") else key)


def evidence_line(chk: dict) -> str:
    m, n, sc, bs = chk["missing"], chk["compared"], chk.get("sidecars") or [], chk.get("by_sidecar") or []
    return (f"instrument evidence (critic ref-0035): {n - len(m)} of {n} compared runs corroborated "
            f"{chk['instrument']} from VM evidence"
            + (f" ({len(bs)} by a complete instrument.json sidecar, critic ref-0036)" if bs else "")
            + (f"; not corroborated: {', '.join(m)}" if m else "")
            + (f"; {len(sc)} instrument.json sidecar(s) agree" if sc else ""))


def step_check(p: Path, s: dict) -> dict | None:
    """STEP_RULE for one score: the scored checkpoint's step (its curves.csv row, chck_<label>.pt <-> label) against
    the run's final step (result.json steps; the run dir is the score's folder or the one above score/). None when
    score.json names no checkpoint (not written by score_trial.sh)."""
    ck = s.get("checkpoint")
    if not ck:
        return None
    out = {"checkpoint": ck, "ckpt_step": None, "final_step": None, "last_row": None, "ok": False}
    run = next((d for d in (p.parent, p.parent.parent) if (d / "result.json").exists()), None)
    if run is None or not (run / "curves.csv").exists():
        return {**out, "why": "no result.json beside the score" if run is None else f"no curves.csv in {run}"}
    out["final_step"] = json.loads((run / "result.json").read_text()).get("steps")
    with (run / "curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    label = Path(ck).stem.removeprefix("chck_")
    hit = [int(float(r["step"])) for r in rows if r.get("label") == label]
    out["last_row"] = [rows[-1].get("label"), int(float(rows[-1]["step"]))] if rows else None
    out["ckpt_step"] = hit[-1] if hit else None
    out["ok"] = out["ckpt_step"] is not None and out["ckpt_step"] == out["final_step"]
    if not out["ok"]:
        out["why"] = f"scored {ck} at step {out['ckpt_step']}, the run ends at step {out['final_step']}"
    return out


def void_check(named: dict[str, dict | None]) -> dict:
    """STEP_RULE over every score a readout reads: refuses while any is VOID; {name: its step check} (None: no
    checkpoint named, unchecked)."""
    chk = {k: s.get("steps") for k, s in named.items() if s}
    bad = [f"{k}: {c['why']}" for k, c in chk.items() if c and not c["ok"]]
    if bad:
        raise SystemExit(f"VOID score(s): {'; '.join(bad)}. {STEP_RULE}")
    return chk


def score(run: Path, metric: str = PRIMARY) -> dict | None:
    for p in (run / "score" / "score.json", run / "score.json"):  # a run dir, or a synced one (small files only)
        if p.exists():
            return score_file(p, metric)
    return None


def unseen_rests(dumps: dict, data_dir: str | None = None) -> dict:
    """The harness cross-check: {name: DEV nats/word without the 10,275 flagged words, or None}
    (babylm.a4.unseen.rests); {"_info": {"error": ...}} when it cannot run here (no data, other counts). Reported, never
    binding."""
    try:
        return UN.rests(dumps, data_dir)
    except SystemExit as e:
        return {**{k: None for k in dumps}, "_info": {"instrument": UN.TOOL, "error": str(e)}}


def guarded(scores: dict[str, dict]) -> tuple[dict, list[str]]:
    """({name: v2x or None}, names whose twin json counts other characters or words than GUARDED's)."""
    vals, bad = {}, []
    for k, s in scores.items():
        c = s.get("v2x_counts")
        if c is not None and list(c) != [GUARDED["chars"], GUARDED["words"]]:
            vals[k] = None
            bad.append(k)
        else:
            vals[k] = s.get("v2x")
    return vals, bad


def thresholds(metric: str = PRIMARY, pass_gain: float | None = None, kill_gain: float | None = None,
               blimp_kill: float | None = None, packed: bool = False) -> dict:
    """The metric's registered floor for the execution condition unless overridden; ref says which (None thresholds =
    numbers only)."""
    if packed and metric != "v2":
        raise SystemExit("a packed screen has a registered floor on v2 only (critic ref-0001)")
    reg = (FLOOR_V2_PACKED if packed else FLOORS[metric]) or {}
    th = {"metric": metric, "key": METRICS[metric], "condition": "packed" if packed else "solo",
          "pass_gain": reg.get("pass_gain") if pass_gain is None else pass_gain,
          "kill_gain": reg.get("kill_gain") if kill_gain is None else kill_gain,
          "blimp_kill": FLOOR_V1["blimp_kill"] if blimp_kill is None else blimp_kill}
    if th["pass_gain"] is None or th["kill_gain"] is None:
        th["ref"] = f"no registered {metric} floor yet: numbers only (critic ref-0018)"
    elif reg and all(th[k] == reg[k] for k in ("pass_gain", "kill_gain", "blimp_kill")):
        th["ref"] = reg["ref"]
    else:
        th["ref"] = "custom thresholds (not registered)"
    th["numbers_only"] = th["pass_gain"] is None or th["kill_gain"] is None
    return th


def status(gain: float, th: dict) -> str | None:
    if th["numbers_only"]:
        return None
    g = round(gain, 6)  # scores carry 4-6 decimals; no float dust at a threshold
    return "pass" if g >= th["pass_gain"] else "inconclusive" if g >= th["kill_gain"] else "kill"


def guard(arm: dict, ref: dict, metric: str = PRIMARY) -> dict:
    """Both gains over ref and the blank-line guard (ref-0020): by construction under v2 (critic ref-0021); under v1,
    v2 gain > 0 and within GUARD_TOL of the v1 gain (ok None when either side lacks v1 or v2)."""
    g = {f"gain_{m}": None if arm.get(m) is None or ref.get(m) is None else ref[m] - arm[m] for m in METRICS}
    if metric == "v2":
        return {**g, "ok": True}
    if g["gain_v1"] is None or g["gain_v2"] is None:
        return {**g, "ok": None}
    return {**g, "ok": round(g["gain_v2"], 6) > 0 and round(abs(g["gain_v2"] - g["gain_v1"]), 6) <= GUARD_TOL}


def null_bar(th: dict, null: bool) -> dict:
    """th, with the pass line raised to NULL_BAR's for an arm-vs-null reading (critic ref-0028)."""
    if not null or th["numbers_only"]:
        return th
    return {**th, "pass_gain": max(th["pass_gain"], NULL_BAR["pass_gain"])}


def grade(out: dict, k: str, s: dict, arm: dict, th: dict, null: bool = False) -> None:
    """out["refs"][k] for reference score s: gain under the metric, guard, status; kill / band / artifact lists. A null
    reference (null=True) passes only at NULL_BAR's line."""
    r = {"dev": s["dev"], "gain": s["dev"] - arm["dev"], "blimp_fast": s["blimp"], **guard(arm, s, th["metric"]),
         "null": null}
    th = null_bar(th, null)
    r["status"] = status(r["gain"], th)
    if r["status"] == "kill":
        out["kill"].append(f"gain over {k} {round(r['gain'], 6):+.4f} < {th['kill_gain']:.4f}")
    elif r["status"] in ("pass", "inconclusive"):
        if r["ok"] is False:
            r["status"] = "protocol-artifact"
            out["artifact"].append(f"gain over {k}: v2 {r['gain_v2']:+.4f}, v1 {r['gain_v1']:+.4f} (guard ref-0020)")
        elif r["ok"] is None:
            out["guard_pending"].append(k)
        if r["status"] == "inconclusive":
            out["inconclusive"].append(f"gain over {k} {round(r['gain'], 6):+.4f} in [{th['kill_gain']:.4f}, "
                                       f"{th['pass_gain']:.4f})" + (" (null bar, critic ref-0028)" if null else ""))
    out["refs"][k] = r


def apply_leg(out: dict, scores: dict[str, dict], th: dict, data_dir: str | None = None) -> None:
    """The never-seen-character leg on every graded reference (critic ref-0001; instrument ref-0023 / ref-0025): gain =
    v2x(ref) - v2x(arm) (GUARDED), on the same bar as the full-DEV leg (the v2 floor of the condition under a v1 read).
    A pass needs both legs: a leg below the kill line kills, one in the band turns a pass inconclusive, a missing leg
    leaves a pass or an inconclusive pending. scores: {"arm": ..., ref: ...}, score_file's dicts. The harness per-word
    cross-check (gain_ex_unseen_harness) is reported beside it and never changes a status."""
    lt = th if th["metric"] == "v2" else thresholds("v2", packed=th["condition"] == "packed")
    vals, bad = guarded(scores)
    cross = unseen_rests({k: s.get("pw") for k, s in scores.items()}, data_dir)
    arm, carm = vals.get("arm"), cross.get("arm")
    out["unseen_leg"] = {"instrument": GUARDED["ref"], "arm": arm, "other_counts": bad,
                         "cross_check": {**cross.get("_info", {}), "arm_rest": carm, "binding": False}}
    out.setdefault("leg_pending", [])
    for k, r in out["refs"].items():
        ref, cref = vals.get(k), cross.get(k)
        g = None if arm is None or ref is None else ref - arm
        lk = null_bar(lt, r.get("null", False))
        st = None if g is None else status(g, lk)
        r["gain_ex_unseen"], r["status_ex_unseen"] = g, st
        r["gain_ex_unseen_harness"] = None if carm is None or cref is None else cref - carm
        if r["status"] not in ("pass", "inconclusive"):
            continue
        if g is None:
            out["leg_pending"].append(k)
        elif st == "kill":
            r["status"] = "kill"
            out["kill"].append(f"gain over {k} without never-seen-character words {round(g, 6):+.4f} < "
                               f"{lt['kill_gain']:.4f}")
        elif st == "inconclusive" and r["status"] == "pass":
            r["status"] = "inconclusive"
            out["inconclusive"].append(f"gain over {k} without never-seen-character words {round(g, 6):+.4f} in "
                                       f"[{lk['kill_gain']:.4f}, {lk['pass_gain']:.4f})"
                                       + (" (null bar, critic ref-0028)" if r.get("null") else ""))


def decide(out: dict, th: dict) -> str:
    """KILL > numbers only > PROTOCOL-ARTIFACT > PENDING (guard needs both metrics; the never-seen-character leg needs
    the dumps) > INCONCLUSIVE > PASS."""
    if out["kill"]:
        return "KILL"
    if th["numbers_only"]:
        return f"NUMBERS ONLY: {th['ref']}"
    if out["artifact"]:
        return "PROTOCOL-ARTIFACT"
    if out["guard_pending"]:
        return f"PENDING: the guard needs v1 and v2 for {out['guard_pending']}"
    if out.get("leg_pending"):
        bad = (out.get("unseen_leg") or {}).get("other_counts")
        return (f"PENDING: the never-seen-character leg ({GUARDED['twin_key']}) is missing for {out['leg_pending']}"
                + (f"; twin json with other counts: {bad}" if bad else ""))
    return "INCONCLUSIVE" if out["inconclusive"] else "PASS"


def calib_runs(p: str | Path | None) -> dict | None:
    """calib_control's --json output: {name: {"full": {...}, "guarded": {...}}} (v2 dumps only)."""
    if p is None:
        return None
    j = json.loads(Path(p).read_text())
    if j.get("format") != "v2":
        raise SystemExit(f"{p}: the calibrated control reads v2 (--twin --temps) dumps (critic ref-0029)")
    return j["runs"]


def gates(out: dict, names: dict[str, str], scores: dict[str, dict], th: dict, calib: dict | None = None,
          estar_raw: dict | None = None, uncorroborated: list[str] | None = None) -> None:
    """Promotion gates on a screen (module docstring): sets out["gates"] (every number) and out["promotion"] (None
    unless the screen passed; else "promote", "not promoted: <label>" or "pending: <what>"). names: {"arm" or a graded
    reference: its calib json name}; scores: score_file's dicts under the same keys (a calib raw value that differs from
    its twin number is another dump: refused); uncorroborated: the compared runs whose instrument no VM evidence
    corroborates (corroborate's "missing"; None: no check, every run counted as missing)."""
    lt = th if th["metric"] == "v2" else thresholds("v2", packed=th["condition"] == "packed")
    calib, arm = calib or {}, names["arm"]
    kill = {"raw": {leg: lt["kill_gain"] for leg, _ in LEGS},  # each gate on its own metric's kill line (ref-0032)
            "calibrated_t": {leg: cal_t_bar(lt["kill_gain"], lt["condition"], leg) for leg, _ in LEGS}}
    vals, _ = guarded({k: s or {} for k, s in scores.items()})
    for k, s in scores.items():
        c = calib.get(names.get(k)) or {}
        for leg, m in LEGS:
            raw, ref = (c.get(leg) or {}).get("raw"), (s or {}).get("v2") if m == "v2" else vals.get(k)
            if raw is not None and ref is not None and abs(raw - ref) > 1e-5 * abs(ref):
                raise SystemExit(f"calib json {names[k]!r}: {leg} raw {raw:.6f} is not {k}'s twin {ref:.6f}")
    g = {"kill_gain": lt["kill_gain"], "kill_lines": kill, "cal_t": CAL_T["ref"], "calib": CALIB, "estar": ESTAR,
         "calibrated": {}, "blocked": [], "pending": []}

    def val(n, leg, key):
        return ((calib.get(n) or {}).get(leg) or {}).get(key)

    def gain(n, key="calibrated_t"):
        return {leg: None if val(n, leg, key) is None or val(arm, leg, key) is None
                else val(n, leg, key) - val(arm, leg, key) for leg, _ in LEGS}

    def check(label, what, gains, tnames=(), metric="calibrated_t"):
        if any(val(n, leg, "t_edge") for n in tnames for leg, _ in LEGS):
            g["pending"].append(f"{what}: T picked at the grid edge, extend the grid for every model (ref-0011)")
            return
        miss = [leg for leg, _ in LEGS if gains[leg] is None]
        if miss:
            g["pending"].append(f"{what}: no number ({', '.join(miss)})")
        for leg, _ in LEGS:
            k = kill[metric][leg]
            if gains[leg] is not None and round(gains[leg], 6) < k:
                g["blocked"].append(f"{label}: {what} ({leg}) {gains[leg]:+.4f} < {k:.4f}")

    if not calib:
        g["pending"].append("no calibrated control (--calib: calib_control @ ref-0033 over the temps dumps)")
    for k in out["refs"]:
        n = names.get(k)
        g["calibrated"][k] = {"name": n, **gain(n), **{f"{leg}_lam": v for leg, v in gain(n, "calibrated").items()}}
        if calib:
            check("calibration-equivalent", f"calibrated_t gain over {k}", g["calibrated"][k], (arm, n))
    e = estar_raw or {}
    ev, _ = guarded({"arm": scores.get("arm") or {}, "e": e})
    a2 = (scores.get("arm") or {}).get("v2")
    g["estar_raw"] = {"run": e.get("run"), "full": None if e.get("v2") is None or a2 is None else e["v2"] - a2,
                      "guarded": None if ev["e"] is None or ev["arm"] is None else ev["e"] - ev["arm"]}
    if estar_raw is None:
        g["pending"].append("no same-seed base at E*_raw (--estar_raw)")
    else:
        check("epoch-equivalent", "raw gain over the same-seed base at E*_raw", g["estar_raw"], metric="raw")
    g["estar_cal"] = {"name": CALIB_NAMES["estar_cal"], **gain(CALIB_NAMES["estar_cal"])}
    ec, er = calib.get(CALIB_NAMES["estar_cal"]) or {}, {"full": e.get("v2"), "guarded": ev["e"]}
    g["estar_cal"]["build_via"] = "--estar_raw" if all(  # ENV_RULE: E*_cal's build is --estar_raw's (same run)
        (ec.get(leg) or {}).get("raw") is not None and er[leg] is not None
        and abs(ec[leg]["raw"] - er[leg]) <= 1e-5 * abs(er[leg]) for leg, _ in LEGS) else None
    if calib and CALIB_NAMES["estar_cal"] not in calib:
        g["pending"].append(f"no same-seed base at E*_cal in the calib json (name {CALIB_NAMES['estar_cal']})")
    elif calib and g["estar_cal"]["build_via"] is None:
        g["pending"].append("the same-seed base at E*_cal: no build record (the calib json's estar_cal is not the "
                            "--estar_raw run; ENV_RULE)")
    elif calib:
        check("epoch-equivalent", "calibrated_t gain over the same-seed base at E*_cal", g["estar_cal"],
              (arm, CALIB_NAMES["estar_cal"]))
    u = ["(no VM evidence check)"] if uncorroborated is None else uncorroborated
    g["uncorroborated"] = u
    if u:
        g["pending"].append(f"the instrument is not corroborated from VM evidence for {', '.join(u)} (critic "
                            "ref-0035 / ref-0036: --vm_evidence or a complete runner sidecar)")
    out["gates"] = g
    if out.get("verdict") != "PASS":
        out["promotion"] = None
    elif g["blocked"]:
        out["promotion"] = "not promoted: " + " and ".join(sorted({b.split(":")[0] for b in g["blocked"]}))
    elif g["pending"]:
        out["promotion"] = "pending: " + "; ".join(g["pending"])
    else:
        out["promotion"] = "promote"


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


def judge(fam: str, runs: Path, base: dict, th: dict, rows: dict, data_dir: str | None = None,
          calib: dict | None = None, estar_raw: dict | None = None, uncorroborated: list[str] | None = None) -> dict:
    f = FAMILIES[fam]
    arm = score(runs / f["arm"], th["metric"])
    refs = {"N0'": f["n0"], "N1": f["n1"], "N2": f["n2"]}
    sc = {k: score(runs / v, th["metric"]) for k, v in refs.items()}
    sc["base"] = base
    names = {"arm": f["arm"], **refs, "base": CALIB_NAMES["base"]}  # calib json names
    fl = rows.get(f["arm"], {}).get("flops_total")
    twin_fl = rows.get(f["n0"], {}).get("flops_total")
    ratio = fl / twin_fl if fl and twin_fl else None
    if fam in FLOP_MATCHED and (ratio is None or ratio > LG.FLOP_RULE):
        sc["FLOP-matched " + FLOP_MATCHED[fam]] = score(runs / FLOP_MATCHED[fam], th["metric"])
        names["FLOP-matched " + FLOP_MATCHED[fam]] = FLOP_MATCHED[fam]
    out = {"arm": f["arm"], "dev": arm and arm["dev"], "dev_v1": arm and arm["v1"], "dev_v2": arm and arm["v2"],
           "dev_v2x": arm and arm["v2x"], "blimp_fast": arm and arm["blimp"], "flops_vs_twin": ratio,
           "within_R0": rows.get(f["arm"], {}).get("within_R0"), "refs": {}, "kill": [], "inconclusive": [],
           "artifact": [], "guard_pending": [], "leg_pending": [], "missing": []}
    for k, s in sc.items():
        if not s or s["dev"] is None or not arm or arm["dev"] is None:
            out["missing"].append(k)
            continue
        grade(out, k, s, arm, th, null=k != "base")  # N0', N1, N2 and N0'_a are stream-randomized nulls
    if arm and out["refs"]:
        apply_leg(out, {"arm": arm, **{k: sc[k] for k in out["refs"]}}, th, data_dir)
    have = [k for k in ("N0'", "base") if k in out["refs"]]
    if len(have) == 2:
        comp = min(have, key=lambda k: out["refs"][k]["dev"])
        out["comparator"] = comp
        cb = out["refs"][comp]["blimp_fast"]
        out["blimp_drop_vs_comparator"] = None if cb is None or out["blimp_fast"] is None else cb - out["blimp_fast"]
    if out["missing"]:
        out["verdict"] = f"NO VERDICT: missing {out['missing']}"
    else:
        d = out.get("blimp_drop_vs_comparator")
        if d is None:
            out["kill"].append("no BLiMP-fast score")
        elif round(d, 6) > th["blimp_kill"]:
            out["kill"].append(f"BLiMP-fast {d:.2f} below the comparator > {th['blimp_kill']:.1f}")
        if out["within_R0"] is False:
            out["kill"].append("over the R0 cap")
        out["verdict"] = decide(out, th)
    out["gates"], out["promotion"] = None, None
    if arm:
        gates(out, names, {"arm": arm, **{k: sc[k] for k in out["refs"]}}, th, calib, estar_raw, uncorroborated)
    return out


def fmt(x: float | None, f: str = "+.4f") -> str:
    return "?" if x is None else format(x, f)


def table(fams: dict, metric: str = PRIMARY) -> str:
    lines = [f"| family | arm DEV ({metric}) | reference | ref DEV | gain | gain v1 / v2 | gain without never-seen-char "
             "words: v2 twin / harness cross-check | status | BLiMP-fast arm / ref |",
             "|---|---|---|---|---|---|---|---|---|"]
    for fam, j in fams.items():
        for k, r in j["refs"].items():
            bl = " / ".join(fmt(x, ".2f") for x in (j["blimp_fast"], r["blimp_fast"]))
            lines.append(f"| {fam} ({j['arm']}) | {j['dev']:.4f} | {k} | {r['dev']:.4f} | {r['gain']:+.4f} | "
                         f"{fmt(r['gain_v1'])} / {fmt(r['gain_v2'])} | {fmt(r.get('gain_ex_unseen'))} / "
                         f"{fmt(r.get('gain_ex_unseen_harness'))} | {r.get('status') or '-'} | {bl} |")
    return "\n".join(lines)


def gate_rows(label: str, g: dict | None) -> list[str]:
    """Table rows of gates()'s numbers: calibrated_t over each reference (lam-only beside), then the two E* gains."""
    if not g:
        return []
    rows = [(f"calibrated_t over {k}", c, f"{fmt(c.get('full_lam'))} / {fmt(c.get('guarded_lam'))}")
            for k, c in g["calibrated"].items()]
    rows += [("raw over the same-seed base at E*_raw", g["estar_raw"], ""),
             ("calibrated_t over the same-seed base at E*_cal", g["estar_cal"], "")]
    return [f"| {label} | {w} | {fmt(c.get('full'))} | {fmt(c.get('guarded'))} | {lam} |" for w, c, lam in rows]


def gate_table(fams: dict) -> str:
    lines = ["| family | promotion gate (its metric's kill line, both legs) | gain full | gain without never-seen-char "
             "words | lam-only full / without |", "|---|---|---|---|---|"]
    for fam, j in fams.items():
        lines += gate_rows(f"{fam} ({j['arm']})", j.get("gates"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_score", required=True, help="score.json of the E10 base, seed 0, in the arms' "
                    "environment: v2e (v2_S_10M_e10_s0, torch 2.7, solo; experimenter 19:19Z). base_r1 s0 (torch 2.9.1) is "
                    "another environment and refused (ENV_RULE)")
    ap.add_argument("--ref_instrument", nargs="+", action="extend", default=[], metavar="NAME=solo|packed",
                    help="each reference's packing instrument, from its ticket (run files do not record it; ENV_RULE, "
                    "critic ref-0039): base=solo, and estar_raw=solo with --estar_raw")
    ap.add_argument("--metric", choices=sorted(METRICS), default=PRIMARY, help="DEV metric of the verdict (v2 primary)")
    ap.add_argument("--pass_gain", type=float, default=None, help="screen pass, nats/word (default: the metric's floor)")
    ap.add_argument("--kill_gain", type=float, default=None, help="below it a gain kills (default: the metric's floor)")
    ap.add_argument("--blimp_kill", type=float, default=None, help="BLiMP-fast drop that kills (default 2.0)")
    # defaults: the metric's registered floor (v2: ref-0019 pass 0.010, kill below 0.0074; v1: ref-0022, information)
    ap.add_argument("--packed", action="store_true", help="a packed screen with a packed base: the packed v2 floor "
                    "(critic ref-0001: pass 0.0144, kill below 0.0102); A4 runs solo")
    ap.add_argument("--data_dir", default=None, help="raw/ and bin/dev.* for the harness cross-check of the "
                    "never-seen-character leg (default: HINTON_BABYLM_DATA or data/babylm)")
    ap.add_argument("--runs_root", default="babylm/runs")
    ap.add_argument("--data", default=None, help="A4's data dir (audit/, ledger.json); default RUNS_ROOT/a4_data")
    ap.add_argument("--calib", default=None, help="calib_control.py @ ref-0033 --format v2 --json output over the "
                    "temps dumps: base=<E10 base, same seed> estar_cal=<base at E*_cal, same seed> <run>=... per run")
    ap.add_argument("--estar_raw", default=None, help="score.json of the same-seed base at E*_raw (v2 twin, both legs)")
    ap.add_argument("--estar", default=None, help="which E*_raw / E*_cal were picked and by whom (recorded)")
    ap.add_argument("--vm_evidence", default=None, help="executor 3's instrument table (critic ref-0035): "
                    "tab-separated run, vm, machine_type, gpu, instrument, evidence, one row per compared run; "
                    "promotion needs every run corroborated (a solo run also by its complete instrument.json "
                    "sidecar, critic ref-0036); a screen reads without it")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs = Path(a.runs_root)
    data = Path(a.data) if a.data else runs / "a4_data"
    present = [r for f in FAMILIES.values() for r in f.values() if (runs / r / "result.json").exists()]
    used = {h: lab for r in present for h, lab in synth_splits(runs / r).items()}
    th = thresholds(a.metric, a.pass_gain, a.kill_gain, a.blimp_kill, a.packed)
    rep = {"audits": audits(data, used), "thresholds": th}
    lg = data / "ledger.json"
    rows = {r["run"]: r for r in json.loads(lg.read_text())["arms"]} if lg.exists() else \
        {r: LG.arm_row(runs / r, None) for r in present}
    base = score_file(a.base_score, a.metric)
    if base is None:
        raise SystemExit(f"no base score at {a.base_score}")
    estar_raw = score_file(a.estar_raw, a.metric) if a.estar_raw else None
    if a.estar_raw and estar_raw is None:
        raise SystemExit(f"no E*_raw base score at {a.estar_raw}")
    own = {r: score(runs / r, a.metric) for f in FAMILIES.values() for r in f.values()}
    rep["step_check"] = void_check({"base": base, "estar_raw": estar_raw, **own})
    home = {**{r: s["env"] for r, s in own.items() if s},
            **{g: run_env(runs / g) for g in LINEAGE if (runs / g).is_dir()}}
    refs = {"base": base, "estar_raw": estar_raw}
    rep["env_check"] = env_check(home, refs, th["condition"],
                                 instruments(a.ref_instrument, {"base": "base", "estar_raw": "estar_raw"}))
    rep["instrument_check"] = corroborate({**home, **{ref_run(k, s): s["env"] for k, s in refs.items() if s}},
                                          th["condition"], vm_evidence(a.vm_evidence))
    rep["scope"] = SCOPE
    calib = calib_runs(a.calib)
    fams = {k: judge(k, runs, base, th, rows, a.data_dir, calib, estar_raw, rep["instrument_check"]["missing"])
            for k in FAMILIES}
    scored = {k: j for k, j in fams.items() if j["dev"] is not None}
    best = min(scored, key=lambda k: scored[k]["dev"]) if scored else None
    rep["families"] = fams
    rep["judged"] = best
    rep["verdict"] = fams[best]["verdict"] if best else "NO VERDICT: no scored arm"
    rep["promotion"] = fams[best]["promotion"] if best else None
    rep["promote"] = fams[best]["arm"] if best and rep["promotion"] == "promote" else None
    rep["estar"] = {"note": a.estar, "raw_base": a.estar_raw, "calib": a.calib}
    rep["next"] = NEXT_SEEDS if rep["verdict"] == "INCONCLUSIVE" and a.metric == "v2" else None
    band = [k for k, r in fams[best]["refs"].items() if r.get("null") and "inconclusive" in
            (r.get("status"), r.get("status_ex_unseen"))] if rep["next"] else []
    if band:  # critic ref-0028: arm and null seeds decide (make_a4_jobs --seed K writes N0' twins; N1/N2 need adding)
        rep["next"] += f"; the 2 extra seeds also run for the null(s) in the band: {', '.join(band)} (critic ref-0028)"
    rep["table"] = table(fams, a.metric) + "\n\n" + gate_table(fams)
    print(json.dumps(rep["audits"], indent=1))
    print(rep["table"])
    why = (fams[best]["kill"] or fams[best]["artifact"] or fams[best]["inconclusive"]) if best else []
    line = (f"judged arm: {best and fams[best]['arm']} -> {rep['verdict']} [{th['ref']}; {th['condition']}]"
            + (f" ({'; '.join(why)})" if why else "")
            + (f"\npromotion: {rep['promotion']}" if rep["promotion"] else "")
            + f"\n{env_line(rep['env_check'])}\n{evidence_line(rep['instrument_check'])}\nscope: {SCOPE}")
    print(line)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        Path(a.out).with_suffix(".md").write_text(rep["table"] + f"\n\n{line}\n" + (f"E*: {a.estar}\n" if a.estar
                                                                                      else "")
                                                  + (f"\n{rep['next']}\n" if rep["next"] else ""))
    return rep


if __name__ == "__main__":
    main()
