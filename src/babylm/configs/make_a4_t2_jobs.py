"""Jobs for attack A4's T2 arm, closed-system augmentation past 100M words (official FAQ tier, outside R0.2; card
docs/babylm/attacks/A4_reexpression.md "T2 arm"; allocation ref-0109, critic ref-0014 and ref-0108; G annealed, lead
04:47Z):

    python -m babylm.configs.make_a4_t2_jobs --base_job [REV:]FILE.json:NAME [--commit SHA]

Writes babylm/configs/jobs_a4_t2.json (trainer jobs for babylm.harness.run_jobs) and three GPU executor files:
  gcp/a4_t2_gate.json  the mini-ruler T2 gate at the 1M scale (labels only, ruling ref-0110): train_1M
                       (babylm.moonshot.mini_ruler, checked against TRAIN_1M), G@1M (a4_t2g_G: the base args on
                       train_1M at 5 epochs, annealed), 10M words of G@1M text (A4a: train_1M seeds and copy-block)
                       and of unseeded text (N1), the N2 shuffle, the audits, base@1M x 10 (a4_t2g_base), the size-M
                       FLOP reference at 1M (a4_t2g_M), the students a4_t2g, a4_t2g_n0, a4_t2g_n1, a4_t2g_n2, every
                       score with its blank-line twin, the ledger and the gate readout (v2).
  gcp/a4_t2.json       the 10M stage, batch 1 (after a gate pass): G (a4_t2_G: the base args at 5 epochs, annealed =
                       the experimenter's v2_S_10M_e5_s0 args), 100M words of G's text (A4a) and the probe, the split, the
                       audits, the students a4_t2 and a4_t2_n0, the scores (G's too) with twins and the ledger.
  gcp/a4_t2_b2.json    batch 2, only if a4_t2 clears the base and N0' (babylm.a4.t2 verdict says so): 100M unseeded
                       words (N1), the N2 shuffle of a4_t2's split, audits, the students a4_t2_n1 and a4_t2_n2.
Students = the base job's args without --epochs, + --init_from G/resume.pt --warmup_frac 0.02 --exposure_rule t2
--sources real:<split>:5 [synthetic:<dir>:1 --kd_teacher G/resume.pt --kd_rows synthetic --kd_alpha 0.5 --kd_temp
1.0]; N0' has no synthetic source and no teacher. The gate's runs add --split train_1M (and its students
--exposure_cap CAP_1M). Corpus exposure (the trainer refuses a plan above the cap): 10M stage G 5 epochs + real 5
epochs = 99.5M <= 100M; the gate G@1M 5 + real 5 epochs of train_1M = 10.0M <= CAP_1M = 100M x 1,001,443 / 9,953,773
words (the 100M cap scaled to the subset). The base recipe has no EMA, so G's resume.pt holds its evaluated weights
(the KD teacher loads them). Each score runs the blank-line twin (babylm/eval/dev_bpw_twin.py) and merges
dev_nats_per_word_twin (metric v2, primary since the critic's 05:02Z switch) and dev_nats_per_word_twin_ex_unseen (the
never-seen-character leg, critic ref-0001) into score.json: the setup puts the twin @ ref-0026 over the vendored
ref-0049 copy (the executor's bucket copy, its sha256 checked; same headline to the bit, plus ex_unseen_chars, the
canonical second leg: critic ref-0023, experimenter 06:43Z). Each 10M student also writes its harness per-word dumps
(babylm.analysis.perword: pw/pw_devfull_final.npy, the leg's cross-check). The 10M files run SOLO (same condition as
the solo seed-0 base, critic ref-0027 / ref-0001); the gate may run packed (labels only). The A4 thread reads the gate
and the verdicts after copy-back (babylm.a4.t2 gate / verdict).
The executor archives --commit and runs that commit's jobs_a4_t2.json: commit and push the trainer file, then re-run
with --commit <that sha> for the executor files (refused otherwise).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from babylm.configs import make_a4_jobs as MJ

HERE = MJ.HERE
RUNS, EVAL_W, RULER_SHA = MJ.RUNS, MJ.EVAL_W, MJ.RULER_SHA
TRAINER = "jobs_a4_t2.json"
D2 = f"{RUNS}/a4_t2_data"   # the 10M stage's synthetic splits, audits, ledger
DG = f"{RUNS}/a4_t2g_data"  # the gate's
G_EPOCHS = 5                # G = the base args at 5 epochs, annealed (experimenter 04:47Z: the v2_S_10M_e5_s0 args)
G_RUN, G1_RUN = "a4_t2_G", "a4_t2g_G"
G_CKPT, G1_CKPT = f"{RUNS}/{G_RUN}/resume.pt", f"{RUNS}/{G1_RUN}/resume.pt"
REAL_EPOCHS = 5
V_SYN = 100_000_000         # fixed before any run (critic ref-0108)
V_SYN_1M = 10_000_000       # the gate's: V_syn at the 1M scale
TRAIN_1M = {"words": 1_001_443, "tokens": 1_601_735,  # babylm.moonshot.mini_ruler --sizes 1, seed 20261010, pieces 2048
            "sha256": "c6acf2aaeebd793986ddd5dfb4783f5cef7aeb01ec636306c9f782d56888cd83",
            "parent_sha256": "30be8c16ad4b2147d6c09d63f9463f2222313cd87a0e7ea11084084b0d74cfe4"}
CAP_1M = MJ.CAP * TRAIN_1M["words"] // MJ.TRAIN_WORDS
ORACLE_DEV = 5.002263348908206  # round-1 base seed 0 at 10M, score.json dev_nats_per_word_twin (the gate's oracle, v2)
KD = ["--kd_rows", "synthetic", "--kd_alpha", "0.5", "--kd_temp", "1.0"]
GATE_REFS = (G1_RUN, "a4_t2g_base", "a4_t2g_M")
GATE_RUNS = (*GATE_REFS, "a4_t2g", "a4_t2g_n0", "a4_t2g_n1", "a4_t2g_n2")
LABEL = "A4 T2 arm: closed-system augmentation (official FAQ), outside R0.2; never a strict-small entry; pool: moonshot"
TWIN = {"uri": "__BUCKET__/code/dev_bpw_twin_ref-0026.py",  # instruments ref-0026: v2 + ex_unseen_chars (ref-0023)
        "sha256": "ad43c239fa2a47605c246748fee795a3070e0ad09f68cdc18eb0a880208bab07"}


def setup(gate: bool) -> str:
    s = ("set -e; pip install -q numpy tokenizers matplotlib transformers==4.51.3; "
         f"T={EVAL_W}/ruler_tools_{RULER_SHA}; mkdir -p $T; gsutil -q cp \"$RULER_TOOLS_URI\" /tmp/rt.tgz; "
         "tar -xzf /tmp/rt.tgz -C $T; python -m babylm.data.prepare --download --splits train_10M,dev; "
         f"$T/babylm/eval/setup.sh {EVAL_W}; "
         f"if [ -n \"${{RESTORE_URI:-}}\" ]; then mkdir -p {RUNS}; gsutil -m -q rsync -r \"$RESTORE_URI\" {RUNS}; fi; "
         f"gsutil -q cp {TWIN['uri']} babylm/eval/dev_bpw_twin.py; "
         f"echo '{TWIN['sha256']}  babylm/eval/dev_bpw_twin.py' | sha256sum -c -")
    if not gate:
        return s
    return s + ("; python -m babylm.moonshot.mini_ruler --sizes 1; python -c \"from babylm.data import corpus as C; "
                f"m = C.load_split('train_1M').meta; want = {TRAIN_1M!r}; got = {{k: m[k] for k in want}}; "
                "assert got == want, ('train_1M differs from the committed subset', got)\"")


def trainer_jobs(args: list[str], to: int) -> list[dict]:
    if "--ema_decay" in args:
        raise SystemExit("the base sets --ema_decay: G's resume.pt would hold raw weights, not the evaluated ones")
    one = MJ.set_flag(MJ.set_flag(args, "--split", "train_1M"), "--epochs", "10")
    cyc1 = MJ.drop_flag(one, "--epochs") + ["--init_from", G1_CKPT, "--warmup_frac", "0.02", "--exposure_rule", "t2",
                                            "--exposure_cap", str(CAP_1M)]
    cyc = MJ.drop_flag(args, "--epochs") + ["--init_from", G_CKPT, "--warmup_frac", "0.02", "--exposure_rule", "t2"]
    r1, r10 = f"real:train_1M:{REAL_EPOCHS}", f"real:train_10M:{REAL_EPOCHS}"
    kd1, kd10 = ["--kd_teacher", G1_CKPT, *KD], ["--kd_teacher", G_CKPT, *KD]
    return [MJ.mk(G1_RUN, MJ.set_flag(one, "--epochs", str(G_EPOCHS)), to), MJ.mk("a4_t2g_base", one, to),
            MJ.mk("a4_t2g_M", MJ.set_flag(one, "--size", "M"), to),
            MJ.mk("a4_t2g", [*cyc1, "--sources", r1, f"synthetic:{DG}/syn_t2:1", *kd1], to),
            MJ.mk("a4_t2g_n0", [*cyc1, "--sources", r1], to),
            MJ.mk("a4_t2g_n1", [*cyc1, "--sources", r1, f"synthetic:{DG}/syn_n1:1", *kd1], to),
            MJ.mk("a4_t2g_n2", [*cyc1, "--sources", r1, f"synthetic:{DG}/syn_n2:1", *kd1], to),
            MJ.mk(G_RUN, MJ.set_flag(args, "--epochs", str(G_EPOCHS)), to),
            MJ.mk("a4_t2", [*cyc, "--sources", r10, f"synthetic:{D2}/syn_t2:1", *kd10], to),
            MJ.mk("a4_t2_n0", [*cyc, "--sources", r10], to),
            MJ.mk("a4_t2_n1", [*cyc, "--sources", r10, f"synthetic:{D2}/syn_n1:1", *kd10], to),
            MJ.mk("a4_t2_n2", [*cyc, "--sources", r10, f"synthetic:{D2}/syn_n2:1", *kd10], to)]


def gen(d: str, out: str, ckpt: str, *flags: str) -> str:
    return f"[ -f {d}/gen/{out}.json ] || python -m babylm.a4.generate --ckpt {ckpt} --out {d}/gen/{out} " + " ".join(flags)


def built(d: str, name: str) -> str:
    return f"[ -f {d}/{name}/bin/synth.meta.json ] ||"


def run(names: str, to: int) -> dict:
    return {"name": names.split()[0], "cmd": f"python -m babylm.harness.run_jobs babylm/configs/{TRAINER} --device cuda "
                                             f"--only {names}", "timeout_s": to}


def score(name: str, pw: bool = False) -> str:
    """score_trial.sh (DEV v1 + the fast suite), then the blank-line twin @ ref-0026 (v2 and the never-seen-character
    leg) merged into score.json; pw: the harness per-word dumps (babylm.analysis.perword: pw/pw_devfull_final.npy, the
    leg's cross-check)."""
    d = f"{RUNS}/{name}/score"
    pw_cmd = (f"; [ -f {RUNS}/{name}/pt/pt_devfull_final.npy ] && {{ [ -f {RUNS}/{name}/pw/pw_devfull_final.npy ] || "
              f"python -m babylm.analysis.perword {RUNS}/{name} --epochs_only; }}") if pw else ""
    return (f"{{ [ -f {d}/score.json ] || bash babylm/harness/score_trial.sh {RUNS}/{name} {EVAL_W}; }}; "
            f"[ -d {d}/hf ] && {{ [ -f {d}/dev_bpw_twin.json ] || {EVAL_W}/venv/bin/python babylm/eval/dev_bpw_twin.py "
            f"{d}/hf {EVAL_W}/osf/text_data/dev/dev --slice full --ws collapse --line_join newline --out "
            f"{d}/dev_bpw_twin.json > {d}/dev_bpw_twin.log 2>&1 || echo '[twin] {name} failed'; }}; "
            f"python -m babylm.a4.t2 twin {RUNS}/{name}" + pw_cmd)


def student(name: str, to: int, pw: bool = False) -> dict:
    cmd = (f"python -m babylm.harness.run_jobs babylm/configs/{TRAINER} --only {name} --device cuda; "
           f"[ -f {RUNS}/{name}/result.json ] && {{ {score(name, pw)}; }}")
    return {"name": name, "cmd": cmd, "timeout_s": to + 3600}


def scores(job: str, names: tuple[str, ...] | list[str]) -> dict:
    return {"name": job, "timeout_s": 3600 * len(names), "cmd": "; ".join(
        f"[ -f {RUNS}/{n}/result.json ] && {{ {score(n)}; }}" for n in names)}


def audits(name: str, d: str, splits: list[str], copies: list[str], probe: str, tag: str = "") -> dict:
    return {"name": name, "timeout_s": 10800, "cmd": f"mkdir -p {d}/audit; " + "; ".join([
        f"[ -f {d}/audit/leak{tag}.json ] || python -m babylm.a4.audit leak " + " ".join(f"{d}/{s}" for s in splits)
        + f" --eval_dir {EVAL_W}/osf/evaluation_data --out {d}/audit/leak{tag}.json",
        *(f"[ -f {d}/audit/copy_{s}.json ] || python -m babylm.a4.audit copy {d}/syn_{s} --probe {probe} "
          f"--out {d}/audit/copy_{s}.json" for s in copies)])}


def report(d: str, runs: list[str], ledger: str, extra: str = "") -> dict:
    cmd = (f"mkdir -p {d}/plots; for r in {' '.join(runs)}; do [ -f {RUNS}/$r/curves.csv ] && "
           f"python -m babylm.harness.plot_curves {RUNS}/$r; done; "
           f"python -m babylm.harness.plot_curves " + " ".join(f"{RUNS}/{r}" for r in runs) + f" --out {d}/plots/all; "
           f"python -m babylm.a4.ledger " + " ".join(f"{RUNS}/{r}" for r in runs if r not in GATE_REFS + (G_RUN,))
           + f" {ledger} --out {d}/ledger.json" + extra)
    return {"name": "a4_t2g_report" if d == DG else "a4_t2_report", "cmd": cmd, "timeout_s": 1800}


def executor_files(commit: str, to: int, label: str) -> tuple[dict, dict, dict]:
    env = {"RULER_TOOLS_URI": f"__BUCKET__/code/ruler_tools_{RULER_SHA}.tar.gz", "RESTORE_URI": ""}
    common = {"commit": commit, "paths": ["babylm", "sprint1"], "outputs": [RUNS], "keep_checkpoints": True,
              "continue_on_error": True}
    tail = ("Preempted => relaunch the same file with env RESTORE_URI=<bucket>/vms/<old VM>/out/babylm/runs (every step "
            "skips what exists). Afterwards copy each run's small files (result.json, config.json, curves.csv, "
            "train_log.csv, log.jsonl, *.png, score/*.json incl. dev_bpw_twin.json{pw}) and "
            "{d}/{{audit,plots,ledger.*,gate*}} and {d}/gen/*.json to shared/babylm/attacks/A4/t2/<run>/; "
            "keep the run dirs (G's resume.pt) in GCS; tell the A4 thread (it reads the audits first, then the verdict).")
    pw = (", pw/ (the students' per-word dumps: the never-seen-character leg, critic ref-0001) and the students' "
          "per-token v2 dumps as for A4 batch 1 (experimenter 06:20Z)")
    solo = ("SOLO: every job alone on its GPU, never packed (critic ref-0027 / ref-0001: the screen's comparator is the "
            "solo seed-0 base). ")
    g_syn = ("--seed_split train_1M --block_split train_1M --words", str(V_SYN_1M))
    gate_jobs = [
        run(G1_RUN, to),
        {"name": "a4_t2g_generate", "timeout_s": 7200, "cmd": f"mkdir -p {DG}/gen; " + "; ".join([
            gen(DG, "t2", G1_CKPT, *g_syn), gen(DG, "probe", G1_CKPT, "--seed_split train_1M --rows 2000"),
            gen(DG, "n1", G1_CKPT, "--selfseed --block_split train_1M --words", str(V_SYN_1M))])},
        {"name": "a4_t2g_build", "timeout_s": 3600, "cmd": "; ".join([
            f"{built(DG, 'syn_t2')} python -m babylm.a4.data build {DG}/syn_t2 {DG}/gen/t2.jsonl.gz --words {V_SYN_1M}",
            f"{built(DG, 'syn_n1')} python -m babylm.a4.data build {DG}/syn_n1 {DG}/gen/n1.jsonl.gz --words {V_SYN_1M}",
            f"{built(DG, 'syn_n2')} python -m babylm.a4.data shuffle {DG}/syn_t2 {DG}/syn_n2"])},
        audits("a4_t2g_audit", DG, ["syn_t2", "syn_n1", "syn_n2"], ["t2", "n1"], f"{DG}/gen/probe.jsonl.gz"),
        run("a4_t2g_base", to), run("a4_t2g_M", to),
        *(student(n, to) for n in GATE_RUNS[len(GATE_REFS):]),
        scores("a4_t2g_score_refs", GATE_REFS),
        report(DG, list(GATE_RUNS), f"--base_result {RUNS}/a4_t2g_base/result.json",
               f"; python -m babylm.a4.t2 gate --runs_root {RUNS} --data {DG} --oracle_dev {ORACLE_DEV} "
               f"--out {DG}/gate.json || true")]
    b1 = [
        run(G_RUN, to),
        {"name": "a4_t2_generate", "timeout_s": 14400, "cmd": f"mkdir -p {D2}/gen; " + "; ".join([
            gen(D2, "t2", G_CKPT, "--seed_split train_10M --block_split train_10M --words", str(V_SYN)),
            gen(D2, "probe", G_CKPT, "--seed_split train_10M --rows 2000")])},
        {"name": "a4_t2_build", "timeout_s": 7200, "cmd":
            f"{built(D2, 'syn_t2')} python -m babylm.a4.data build {D2}/syn_t2 {D2}/gen/t2.jsonl.gz --words {V_SYN}"},
        audits("a4_t2_audit", D2, ["syn_t2"], ["t2"], f"{D2}/gen/probe.jsonl.gz"),
        *(student(n, to, True) for n in ("a4_t2", "a4_t2_n0")),
        scores("a4_t2_score_G", (G_RUN,)),
        report(D2, [G_RUN, "a4_t2", "a4_t2_n0"], "--base_epochs 10")]
    b2 = [
        {"name": "a4_t2_generate_b2", "timeout_s": 14400, "cmd": f"mkdir -p {D2}/gen; " + gen(
            D2, "n1", G_CKPT, "--selfseed --block_split train_10M --words", str(V_SYN))},
        {"name": "a4_t2_build_b2", "timeout_s": 7200, "cmd": "; ".join([
            f"{built(D2, 'syn_n1')} python -m babylm.a4.data build {D2}/syn_n1 {D2}/gen/n1.jsonl.gz --words {V_SYN}",
            f"{built(D2, 'syn_n2')} python -m babylm.a4.data shuffle {D2}/syn_t2 {D2}/syn_n2"])},
        audits("a4_t2_audit_b2", D2, ["syn_n1", "syn_n2"], ["n1"], f"{D2}/gen/probe.jsonl.gz", "_b2"),
        *(student(n, to, True) for n in ("a4_t2_n1", "a4_t2_n2")),
        report(D2, [G_RUN, "a4_t2", "a4_t2_n0", "a4_t2_n1", "a4_t2_n2"], "--base_epochs 10")]
    fg = {"name": "a4_t2_gate", **common, "setup": setup(True), "env": dict(env), "jobs": gate_jobs, "note": (
        f"{LABEL}. Mini-ruler T2 gate at the 1M scale (labels only; {label}): one A100, sequential, about 1.2 h "
        "(scoring dominates): maxdur 2 h; may run packed (labels only, critic ref-0066). " + tail.format(d=DG, pw=""))}
    f1 = {"name": "a4_t2", **common, "solo": True, "setup": setup(False), "env": dict(env), "jobs": b1, "note": (
        f"{LABEL}. 10M stage batch 1, only after the A4 thread reads the gate as 'proceed' ({label}): G = the base "
        f"args at {G_EPOCHS} epochs, annealed (a4_t2_G); 100M synthetic words, the arm and N0'. One A100, sequential, "
        "about 2.7 h: maxdur 4 h. " + solo + tail.format(d=D2, pw=pw))}
    f2 = {"name": "a4_t2_b2", **common, "solo": True, "setup": setup(False), "env": dict(env), "jobs": b2, "note": (
        f"{LABEL}. 10M stage batch 2, only if the A4 thread says a4_t2 cleared the base and N0': set RESTORE_URI to "
        "batch 1's <bucket>/vms/<VM>/out/babylm/runs (it holds a4_t2_G/resume.pt, a4_t2_data/syn_t2 and the two "
        "students). About 2 h: maxdur 3 h. " + solo + tail.format(d=D2, pw=pw))}
    return fg, f1, f2


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_job", default=None, help="[REV:]FILE.json:NAME of the base round-1 winner's job")
    ap.add_argument("--commit", default=None, help="code commit for the executor files (default: HEAD)")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args(argv)
    args, label, to, e_star = MJ.base_args(a.base_job)
    if e_star != 10:
        raise SystemExit(f"T2's allocation is on the 10-epoch base (G at 5 of its epochs), not E* = {e_star}")
    jobs = trainer_jobs(args, to)
    out = Path(a.out_dir) if a.out_dir else HERE
    spec = {"description": f"attack A4, T2 arm (closed-system augmentation, official FAQ, outside R0.2): the 1M gate "
                           f"and the 10M stage; {label}",
            "note": "Run through babylm/configs/gcp/a4_t2_gate.json, a4_t2.json and a4_t2_b2.json (generation, splits "
                    "and audits sit between the trainer jobs).",
            "allocation": {"G": f"the base args at {G_EPOCHS} epochs, annealed (experimenter 04:47Z; v2_S_10M_e5_s0 args)",
                           "real_epochs": REAL_EPOCHS, "V_syn": V_SYN,
                           "kd": "alpha 0.5, tau 1, synthetic rows only", "gate_V_syn": V_SYN_1M, "gate_cap": CAP_1M,
                           "train_1M": TRAIN_1M, "oracle_dev": ORACLE_DEV},
            "jobs": jobs}
    text = json.dumps(spec, indent=1) + "\n"
    (out / TRAINER).write_text(text)
    commit = a.commit or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=HERE).stdout.strip()
    if a.out_dir is None:
        MJ.check_pinned(commit, text, TRAINER)
    fg, f1, f2 = executor_files(commit, to, label)
    (out / "gcp").mkdir(exist_ok=True)
    for f in (fg, f1, f2):
        (out / "gcp" / f"{f['name']}.json").write_text(json.dumps(f, indent=1) + "\n")
    print(f"wrote {out / TRAINER} ({len(jobs)} trainer jobs) and gcp/a4_t2{{_gate,,_b2}}.json @ {commit}")
    return {"trainer": spec, "gate": fg, "batch1": f1, "batch2": f2}


if __name__ == "__main__":
    main()
