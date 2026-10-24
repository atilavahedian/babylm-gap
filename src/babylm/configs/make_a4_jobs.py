"""Screen jobs for attack A4, closed-system re-expression (card docs/babylm/attacks/A4_reexpression.md):

    python -m babylm.configs.make_a4_jobs --base_job [REV:]FILE.json:NAME [--g_epochs 4] [--gen_epochs 2] [--commit SHA]

Writes babylm/configs/jobs_a4_screen.json (trainer jobs for babylm.harness.run_jobs) and the GPU executor's job files
babylm/configs/gcp/a4_screen.json (batch 1) and a4_screen_b2.json (batch 2: N1 and N2 for A4b, launched only if A4b
has the lower full-DEV nats/word of the two arms). Every trainer job takes the base job's trainer args verbatim (the
base round-1 winner, sent by the experimenter):
  a4_G          the base args at --epochs E_G (registered: 4, since the winner trains 10; the base stays the comparator)
  a4_GA, a4_GB  the same at --split train_10M_hA / train_10M_hB (babylm.a4.data halves), --epochs E_g (registered: 2)
  students      the base args without --epochs, plus --init_from a4_G/resume.pt, --warmup_frac 0.02 and --sources:
    a4_n0a real:train_10M:2R_a      a4_a, a4_n1a, a4_n2a  real:train_10M:R_a + synthetic:<syn_a4a | syn_n1a | syn_n2a>:1
    a4_n0b real:train_10M:2R_b      a4_b, a4_n1b, a4_n2b  real:train_10M:R_b + synthetic:<syn_a4b | syn_n1b | syn_n2b>:1
  R_a = (10 - E_G) // 2 and R_b = (10 - E_G - E_g) // 2 whole epochs (10 epochs of train_10M = 99.5M words, the R0
  budget), each synthetic target = the real part's words (R x 9,953,773), and the b family carries the
  half-generators in --exposure_extra, so every arm stays inside 100M words (the trainer refuses a plan above it).
REV: read FILE.json at that git revision (the base builder's branch need not be checked out). --provisional marks
the files as built on a provisional base. Without --base_job the PROVISIONAL stand-in is the port at size S with
muon_wd 2.0, lr_scale 0.5: regenerate before launch. --g_uri: G is an existing run (its GCS run dir, restored into
babylm/runs/a4_G) instead of trained here.
The executor archives --commit and runs that commit's jobs_a4_screen.json, so writing into the repo is two steps:
commit and push the trainer file, then re-run with --commit <that sha> for the executor files (refused otherwise).
--seed K (K > 0, --arm a|b|ab) writes a seed-K replicate of the promoted arm(s): every run and data directory becomes
its _sK twin (a4_G_sK, a4_GA_sK, a4_GB_sK, a4_data_sK/syn_a4{a,b}, a4_{n0a,a,n0b,b}_sK: the layout make_stack_jobs
--own_g reads). G_K, the half-generators and the students train at --seed K, generation and the split build use seed
K, and the halves stay the registered ones. Files: jobs_a4_sK.json and gcp/a4_sK.json (one batch: G_K, generation,
build, audits, then the arm's student and its N0' twin; --data_only stops after the audits, for the stack's runs).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from babylm.configs.make_base_jobs import job

HERE = Path(__file__).resolve().parent
TRAIN_WORDS = 9_953_773  # train_10M, babylm/data/manifest.json
CAP = 100_000_000
RUNS = "babylm/runs"
D = f"{RUNS}/a4_data"
EVAL_W = "/opt/babylm"  # score_trial.sh / ruler setup.sh work dir on the executor's VMs
RULER_SHA = "c09ef37ac16bbe0a9f3ab9def66ad000ea40cef5"


def sfx(seed: int) -> str:
    """Run and data directory suffix of a seed: seed 0 = the screen's own directories, seed k = their _sk twins."""
    return f"_s{seed}" if seed else ""


def setup(seed: int = 0) -> str:
    g = f"{RUNS}/a4_G{sfx(seed)}"
    return ("set -e; pip install -q numpy tokenizers matplotlib transformers==4.51.3; "
            f"T={EVAL_W}/ruler_tools_{RULER_SHA}; mkdir -p $T; gsutil -q cp \"$RULER_TOOLS_URI\" /tmp/rt.tgz; "
            "tar -xzf /tmp/rt.tgz -C $T; python -m babylm.data.prepare --download --splits train_10M,dev; "
            f"$T/babylm/eval/setup.sh {EVAL_W}; "
            f"if [ -n \"${{RESTORE_URI:-}}\" ]; then mkdir -p {RUNS}; gsutil -m -q rsync -r \"$RESTORE_URI\" {RUNS}; fi; "
            f"if [ -n \"${{G_URI:-}}\" ] && [ ! -f {g}/result.json ]; then mkdir -p {g}; "
            f"gsutil -m -q rsync -r \"$G_URI\" {g}; fi; python -m babylm.a4.data halves")


SETUP = setup()


def base_args(spec: str | None) -> tuple[list[str], str, int, int]:
    """(trainer args without --tag, base label, timeout_s, the base's epochs E*)."""
    if spec is None:
        j = job("provisional", "S", {"--muon_wd": "2.0", "--lr_scale": "0.5"})
        label = "PROVISIONAL stand-in (port, S, muon_wd 2.0, lr_scale 0.5): regenerate with --base_job before launch"
    else:
        path, name = spec.rsplit(":", 1)
        if ":" in path:  # REV:FILE
            rev, rel = path.split(":", 1)
            text = subprocess.run(["git", "show", f"{rev}:{rel}"], capture_output=True, text=True, check=True,
                                  cwd=HERE).stdout
        else:
            text = Path(path).read_text()
        j = {x["name"]: x for x in json.loads(text)["jobs"]}[name]
        label = f"base job {name} from {path}"
    a = [str(x) for x in j["args"]]
    if "--tag" in a:
        i = a.index("--tag")
        del a[i:i + 2]
    for bad in ("--window_plan", "--sources", "--init_from", "--steps", "--final"):
        if bad in a:
            raise SystemExit(f"the base job sets {bad}: A4's jobs need a plain-epochs base (ask the A4 thread)")
    if a[a.index("--seed") + 1] != "0" or a[a.index("--split") + 1] != "train_10M":
        raise SystemExit("the A4 screen is seed 0 on train_10M")
    for f in ("--save",):
        if f not in a:
            a.append(f)
    if "--resume" not in a:
        a += ["--resume", "auto"]
    return a, label, int(j.get("timeout_s", 10800)), int(float(a[a.index("--epochs") + 1]))


def set_flag(a: list[str], flag: str, value: str) -> list[str]:
    a = list(a)
    a[a.index(flag) + 1] = value
    return a


def drop_flag(a: list[str], flag: str) -> list[str]:
    a = list(a)
    i = a.index(flag)
    del a[i:i + 2]
    return a


def mk(name: str, args: list[str], timeout_s: int) -> dict:
    return {"name": name, "module": "babylm.harness.train", "args": [*args, "--tag", name], "timeout_s": timeout_s}


def plan(e_star: int, e_g: int, e_gen: int) -> dict:
    budget = CAP // TRAIN_WORDS  # whole epochs of train_10M inside R0
    r_a, r_b = (budget - e_g) // 2, (budget - e_g - e_gen) // 2
    if r_a < 1 or r_b < 1:
        raise SystemExit(f"no room for a cycle: E* {e_star}, E_G {e_g}, E_g {e_gen}")
    p = {"E_star": e_star, "E_G": e_g, "E_g": e_gen, "R_a": r_a, "R_b": r_b,
         "syn_words_a": r_a * TRAIN_WORDS, "syn_words_b": r_b * TRAIN_WORDS}
    p["words_a"] = (e_g + 2 * r_a) * TRAIN_WORDS
    p["words_b"] = (e_g + e_gen + 2 * r_b) * TRAIN_WORDS
    if max(p["words_a"], p["words_b"]) > CAP:
        raise SystemExit(f"plan over R0: {p}")
    return p


def trainer_jobs(args: list[str], to: int, p: dict, g_trained: bool, seed: int = 0, arms: str = "ab",
                 students: bool = True) -> list[dict]:
    """Seed 0: the screen (G, both half-generators, both families with their nulls). Seed k: the _sk twins of G, the
    half-generators (arm b) and, with students, each arm's student and N0' twin, all at --seed k."""
    s, d = sfx(seed), f"{D}{sfx(seed)}"
    if seed:
        args = set_flag(args, "--seed", str(seed))
    g_args = set_flag(args, "--epochs", str(p["E_G"]))
    h_args = set_flag(set_flag(args, "--epochs", str(p["E_g"])), "--split", "train_10M_hA")
    cyc = drop_flag(args, "--epochs") + ["--init_from", f"{RUNS}/a4_G{s}/resume.pt", "--warmup_frac", "0.02"]
    gens = ["--exposure_extra", f"G_A=@{RUNS}/a4_GA{s}/result.json", f"G_B=@{RUNS}/a4_GB{s}/result.json"]
    ra, rb = p["R_a"], p["R_b"]
    fams = {"a": (("a4_n0a", [f"real:train_10M:{2 * ra}"], []),
                  ("a4_a", [f"real:train_10M:{ra}", f"synthetic:{d}/syn_a4a:1"], []),
                  ("a4_n1a", [f"real:train_10M:{ra}", f"synthetic:{d}/syn_n1a:1"], []),
                  ("a4_n2a", [f"real:train_10M:{ra}", f"synthetic:{d}/syn_n2a:1"], [])),
            "b": (("a4_n0b", [f"real:train_10M:{2 * rb}"], []),
                  ("a4_b", [f"real:train_10M:{rb}", f"synthetic:{d}/syn_a4b:1"], gens),
                  ("a4_n1b", [f"real:train_10M:{rb}", f"synthetic:{d}/syn_n1b:1"], gens),
                  ("a4_n2b", [f"real:train_10M:{rb}", f"synthetic:{d}/syn_n2b:1"], gens))}
    jobs = [mk(f"a4_G{s}", g_args, to)] if g_trained else []
    if "b" in arms:
        jobs += [mk(f"a4_GA{s}", h_args, to), mk(f"a4_GB{s}", set_flag(h_args, "--split", "train_10M_hB"), to)]
    for f in "ab":
        for name, src, extra in (fams[f] if f in arms and students else ()):
            if seed and name[3:5] in ("n1", "n2"):  # the nulls are the screen's (seed 0) only
                continue
            jobs.append(mk(name + s, [*cyc, "--sources", *src, *extra], to))
    return jobs


def gen(out: str, ckpt: str, *flags: str, seed: int = 0) -> str:
    d = f"{D}{sfx(seed)}"
    return f"[ -f {d}/gen/{out}.json ] || python -m babylm.a4.generate --ckpt {RUNS}/{ckpt}{sfx(seed)}/resume.pt " \
           f"--out {d}/gen/{out} " + " ".join(flags) + (f" --seed {seed}" if seed else "")


def built(name: str, seed: int = 0) -> str:
    return f"[ -f {D}{sfx(seed)}/{name}/bin/synth.meta.json ] ||"


def trainer_file(seed: int) -> str:
    return f"jobs_a4_s{seed}.json" if seed else "jobs_a4_screen.json"


def student(name: str, to: int, seed: int = 0) -> dict:
    cmd = (f"python -m babylm.harness.run_jobs babylm/configs/{trainer_file(seed)} --only {name} --device cuda; "
           f"[ -f {RUNS}/{name}/result.json ] && {{ [ -f {RUNS}/{name}/score/score.json ] || "
           f"bash babylm/harness/score_trial.sh {RUNS}/{name} {EVAL_W}; }}")
    return {"name": name, "cmd": cmd, "timeout_s": to + 3600}


def report(e_star: int) -> dict:
    singles = "a4_G a4_GA a4_GB a4_n0a a4_a a4_n1a a4_n2a a4_n0b a4_b a4_n1b a4_n2b"
    arms = " ".join(f"{RUNS}/{r}" for r in singles.split()[3:])
    cmd = (f"mkdir -p {D}/plots; for r in {singles}; do [ -f {RUNS}/$r/curves.csv ] && "
           f"python -m babylm.harness.plot_curves {RUNS}/$r; done; "
           f"python -m babylm.harness.plot_curves {RUNS}/a4_G {RUNS}/a4_n0a {RUNS}/a4_a {RUNS}/a4_n1a {RUNS}/a4_n2a "
           f"--out {D}/plots/arm_a; "
           f"python -m babylm.harness.plot_curves {RUNS}/a4_G {RUNS}/a4_n0b {RUNS}/a4_b {RUNS}/a4_n1b {RUNS}/a4_n2b "
           f"--out {D}/plots/arm_b; python -m babylm.a4.ledger {arms} --base_epochs {e_star} --out {D}/ledger.json")
    return {"name": "a4_report", "cmd": cmd, "timeout_s": 1800}


def executor_files(commit: str, p: dict, to: int, g_trained: bool, label: str) -> tuple[dict, dict]:
    W, half_a, half_b = TRAIN_WORDS, p["syn_words_b"] // 2, p["syn_words_b"] - p["syn_words_b"] // 2
    common = {"commit": commit, "paths": ["babylm", "sprint1"], "setup": SETUP,
              "env": {"RULER_TOOLS_URI": f"__BUCKET__/code/ruler_tools_{RULER_SHA}.tar.gz", "RESTORE_URI": "",
                      "G_URI": ""},
              "outputs": [RUNS], "keep_checkpoints": True, "continue_on_error": True}
    tr = "python -m babylm.harness.run_jobs babylm/configs/jobs_a4_screen.json --device cuda --only"
    b1 = [{"name": "a4_G", "cmd": f"{tr} a4_G", "timeout_s": to}] if g_trained else []
    b1 += [
        {"name": "a4_halfgens", "cmd": f"{tr} a4_GA a4_GB", "timeout_s": to},
        {"name": "a4_generate", "timeout_s": 10800, "cmd": f"mkdir -p {D}/gen; " + "; ".join([
            gen("a4a", "a4_G", "--seed_split train_10M --block_split train_10M --words", str(p["syn_words_a"])),
            gen("probe", "a4_G", "--seed_split train_10M --rows 2000"),
            gen("n1a", "a4_G", "--selfseed --block_split train_10M --words", str(p["syn_words_a"])),
            gen("a4b_A", "a4_GA", "--seed_split train_10M_hB --words", str(half_a)),
            gen("a4b_B", "a4_GB", "--seed_split train_10M_hA --words", str(half_b))])},
        {"name": "a4_build", "timeout_s": 7200, "cmd": "; ".join([
            f"{built('syn_a4a')} python -m babylm.a4.data build {D}/syn_a4a {D}/gen/a4a.jsonl.gz --words {p['syn_words_a']}",
            f"{built('syn_n1a')} python -m babylm.a4.data build {D}/syn_n1a {D}/gen/n1a.jsonl.gz --words {p['syn_words_a']}",
            f"{built('syn_n2a')} python -m babylm.a4.data shuffle {D}/syn_a4a {D}/syn_n2a",
            f"{built('syn_a4b')} python -m babylm.a4.data build {D}/syn_a4b {D}/gen/a4b_A.jsonl.gz "
            f"{D}/gen/a4b_B.jsonl.gz --words {p['syn_words_b']}"])},
        {"name": "a4_audit", "timeout_s": 10800, "cmd": f"mkdir -p {D}/audit; " + "; ".join([
            f"[ -f {D}/audit/leak.json ] || python -m babylm.a4.audit leak {D}/syn_a4a {D}/syn_n1a {D}/syn_n2a "
            f"{D}/syn_a4b --eval_dir {EVAL_W}/osf/evaluation_data --out {D}/audit/leak.json",
            *(f"[ -f {D}/audit/copy_{s}.json ] || python -m babylm.a4.audit copy {D}/syn_{s} --probe "
              f"{D}/gen/probe.jsonl.gz --out {D}/audit/copy_{s}.json" for s in ("a4a", "n1a", "a4b"))])},
        *(student(n, to) for n in ("a4_n0a", "a4_a", "a4_n1a", "a4_n2a", "a4_n0b", "a4_b")),
        report(p["E_star"])]
    b2 = [
        {"name": "a4_generate_b2", "timeout_s": 10800, "cmd": f"mkdir -p {D}/gen; " + "; ".join([
            gen("n1b_A", "a4_GA", "--selfseed --words", str(half_a)),
            gen("n1b_B", "a4_GB", "--selfseed --words", str(half_b))])},
        {"name": "a4_build_b2", "timeout_s": 7200, "cmd": "; ".join([
            f"{built('syn_n1b')} python -m babylm.a4.data build {D}/syn_n1b {D}/gen/n1b_A.jsonl.gz "
            f"{D}/gen/n1b_B.jsonl.gz --words {p['syn_words_b']}",
            f"{built('syn_n2b')} python -m babylm.a4.data shuffle {D}/syn_a4b {D}/syn_n2b"])},
        {"name": "a4_audit_b2", "timeout_s": 10800, "cmd": "; ".join([
            f"[ -f {D}/audit/leak_b2.json ] || python -m babylm.a4.audit leak {D}/syn_n1b {D}/syn_n2b --eval_dir "
            f"{EVAL_W}/osf/evaluation_data --out {D}/audit/leak_b2.json",
            f"[ -f {D}/audit/copy_n1b.json ] || python -m babylm.a4.audit copy {D}/syn_n1b --probe "
            f"{D}/gen/probe.jsonl.gz --out {D}/audit/copy_n1b.json"])},
        *(student(n, to) for n in ("a4_n1b", "a4_n2b")),
        report(p["E_star"])]
    note = (f"A4 screen ({label}). Plan {p}. One A100, sequential, about 2 h (batch 2 about 45 min): maxdur 3 h "
            "(batch 2: 1.5 h). Preempted => relaunch the same file with env "
            "RESTORE_URI=<bucket>/vms/<old VM>/out/babylm/runs (every step skips what exists). Afterwards copy each "
            f"run's small files (result.json, config.json, curves.csv, train_log.csv, *.png, score/*.json) and "
            f"{D}/{{audit,plots,ledger.*}} and {D}/gen/*.json to shared/babylm/attacks/A4/<run>/; tell the "
            "A4 thread (it reports the audits first, then the screen).")
    f1 = {"name": "a4_screen", **common, "jobs": b1, "note": note}
    f2 = {"name": "a4_screen_b2", **common, "jobs": b2,
          "note": "Batch 2, only if A4b has the lower full-DEV nats/word: set RESTORE_URI to batch 1's "
                  "<bucket>/vms/<VM>/out/babylm/runs. " + note}
    return f1, f2


def seed_executor_file(commit: str, p: dict, to: int, g_trained: bool, label: str, seed: int, arms: str,
                       students: bool) -> dict:
    """One batch for seed k: G_k (and the half-generators for arm b), the arm's generation (+ the probe), its split,
    the audits, then (with students) the arm's student and its N0' twin, scored, and the ledger."""
    s, d = sfx(seed), f"{D}{sfx(seed)}"
    half_a, half_b = p["syn_words_b"] // 2, p["syn_words_b"] - p["syn_words_b"] // 2
    tr = f"python -m babylm.harness.run_jobs babylm/configs/{trainer_file(seed)} --device cuda --only"
    gens, builds, splits = [], [], []
    if "a" in arms:
        gens.append(gen("a4a", "a4_G", "--seed_split train_10M --block_split train_10M --words", str(p["syn_words_a"]),
                        seed=seed))
        builds.append(f"{built('syn_a4a', seed)} python -m babylm.a4.data build {d}/syn_a4a {d}/gen/a4a.jsonl.gz "
                      f"--words {p['syn_words_a']} --seed {seed}")
        splits.append("a4a")
    gens.append(gen("probe", "a4_G", "--seed_split train_10M --rows 2000", seed=seed))
    if "b" in arms:
        gens += [gen("a4b_A", "a4_GA", "--seed_split train_10M_hB --words", str(half_a), seed=seed),
                 gen("a4b_B", "a4_GB", "--seed_split train_10M_hA --words", str(half_b), seed=seed)]
        builds.append(f"{built('syn_a4b', seed)} python -m babylm.a4.data build {d}/syn_a4b {d}/gen/a4b_A.jsonl.gz "
                      f"{d}/gen/a4b_B.jsonl.gz --words {p['syn_words_b']} --seed {seed}")
        splits.append("a4b")
    jobs = [{"name": "a4_G", "cmd": f"{tr} a4_G{s}", "timeout_s": to}] if g_trained else []
    if "b" in arms:
        jobs.append({"name": "a4_halfgens", "cmd": f"{tr} a4_GA{s} a4_GB{s}", "timeout_s": to})
    jobs += [
        {"name": "a4_generate", "timeout_s": 10800, "cmd": f"mkdir -p {d}/gen; " + "; ".join(gens)},
        {"name": "a4_build", "timeout_s": 7200, "cmd": "; ".join(builds)},
        {"name": "a4_audit", "timeout_s": 10800, "cmd": f"mkdir -p {d}/audit; " + "; ".join([
            f"[ -f {d}/audit/leak.json ] || python -m babylm.a4.audit leak "
            + " ".join(f"{d}/syn_{x}" for x in splits)
            + f" --eval_dir {EVAL_W}/osf/evaluation_data --out {d}/audit/leak.json",
            *(f"[ -f {d}/audit/copy_{x}.json ] || python -m babylm.a4.audit copy {d}/syn_{x} --probe "
              f"{d}/gen/probe.jsonl.gz --out {d}/audit/copy_{x}.json" for x in splits)])}]
    runs = [f"a4_{n}{f}{s}" for f in "ab" if f in arms for n in ("n0", "")] if students else []
    if students:
        jobs += [student(r, to, seed) for r in runs]
        singles = " ".join([f"a4_G{s}"] + ([f"a4_GA{s}", f"a4_GB{s}"] if "b" in arms else []) + runs)
        jobs.append({"name": "a4_report", "timeout_s": 1800, "cmd": (
            f"mkdir -p {d}/plots; for r in {singles}; do [ -f {RUNS}/$r/curves.csv ] && "
            f"python -m babylm.harness.plot_curves {RUNS}/$r; done; "
            + "; ".join(f"python -m babylm.harness.plot_curves {RUNS}/a4_G{s} {RUNS}/a4_n0{f}{s} {RUNS}/a4_{f}{s} "
                        f"--out {d}/plots/arm_{f}" for f in "ab" if f in arms)
            + f"; python -m babylm.a4.ledger " + " ".join(f"{RUNS}/{r}" for r in runs)
            + f" --base_epochs {p['E_star']} --out {d}/ledger.json")})
    what = "the arm's student and N0' twin" if students else "data only (G, generation, split, audits)"
    note = (f"A4 seed-{seed} replicate, arm(s) {arms}, {what} ({label}). Plan {p}. One A100, sequential, about "
            f"{'1 h' if students else '40 min'}: maxdur 2 h. Preempted => relaunch the same file with env "
            "RESTORE_URI=<bucket>/vms/<old VM>/out/babylm/runs (every step skips what exists). Afterwards copy each "
            f"run's small files and {d}/{{audit,plots,ledger.*}} and {d}/gen/*.json to "
            "shared/babylm/attacks/A4/<run>/; tell the A4 thread.")
    return {"name": f"a4_s{seed}", "commit": commit, "paths": ["babylm", "sprint1"], "setup": setup(seed),
            "env": {"RULER_TOOLS_URI": f"__BUCKET__/code/ruler_tools_{RULER_SHA}.tar.gz", "RESTORE_URI": "",
                    "G_URI": ""},
            "outputs": [RUNS], "keep_checkpoints": True, "continue_on_error": True, "jobs": jobs, "note": note}


def check_pinned(commit: str, text: str, name: str = "jobs_a4_screen.json") -> None:
    """The executor archives the pinned commit and runs that commit's trainer file: it must be the one just written."""
    r = subprocess.run(["git", "show", f"{commit}:babylm/configs/{name}"], capture_output=True, text=True, cwd=HERE)
    if r.returncode or r.stdout != text:
        raise SystemExit(f"wrote {name} only: commit {commit} carries a different one and the executor runs the "
                         "pinned commit's. Commit and push it, then re-run with --commit <that sha>.")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_job", default=None, help="[REV:]FILE.json:NAME of the base round-1 winner's job")
    ap.add_argument("--provisional", action="store_true", help="the base is provisional: say so in the files")
    ap.add_argument("--g_epochs", type=int, default=4, help="E_G (registered: 4)")
    ap.add_argument("--gen_epochs", type=int, default=2, help="E_g of each half-generator (registered: 2)")
    ap.add_argument("--g_uri", default=None, help="G is this existing run (GCS run dir); no a4_G training job")
    ap.add_argument("--commit", default=None, help="code commit for the executor files (default: HEAD)")
    ap.add_argument("--seed", type=int, default=0, help="K > 0: a seed-K replicate of --arm (the _sK twins)")
    ap.add_argument("--arm", choices=("a", "b", "ab"), default=None, help="with --seed K: the promoted arm(s)")
    ap.add_argument("--data_only", action="store_true", help="with --seed K: G_K and its data only (no students)")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args(argv)
    if (a.seed > 0) != (a.arm is not None) or (a.data_only and not a.seed) or a.seed < 0:
        raise SystemExit("--seed K > 0 takes --arm a|b|ab (and optionally --data_only); seed 0 is the screen")
    args, label, to, e_star = base_args(a.base_job)
    if a.provisional:
        label = "PROVISIONAL " + label + ": regenerate on the frozen round-1 winner before launch"
    if a.g_uri and a.g_epochs != e_star:
        raise SystemExit(f"--g_uri: G is the base run itself, so --g_epochs must be its epochs ({e_star})")
    p = plan(e_star, a.g_epochs, a.gen_epochs)
    if a.seed:
        return seed_main(a, args, label, to, p)
    jobs = trainer_jobs(args, to, p, a.g_uri is None)
    out = Path(a.out_dir) if a.out_dir else HERE
    spec = {"description": f"attack A4 screen: G, half-generators, students and nulls (second LR cycle); {label}",
            "note": "Run through babylm/configs/gcp/a4_screen.json (data, generation and audits sit between the "
                    "trainer jobs). a4_n1b/a4_n2b are batch 2 (a4_screen_b2.json).",
            "plan": p, "jobs": jobs}
    text = json.dumps(spec, indent=1) + "\n"
    (out / "jobs_a4_screen.json").write_text(text)
    commit = a.commit or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=HERE).stdout.strip()
    if a.out_dir is None:
        check_pinned(commit, text)
    f1, f2 = executor_files(commit, p, to, a.g_uri is None, label)
    if a.g_uri:
        f1["env"]["G_URI"] = f2["env"]["G_URI"] = a.g_uri
    (out / "gcp").mkdir(exist_ok=True)
    (out / "gcp" / "a4_screen.json").write_text(json.dumps(f1, indent=1) + "\n")
    (out / "gcp" / "a4_screen_b2.json").write_text(json.dumps(f2, indent=1) + "\n")
    print(f"wrote {out / 'jobs_a4_screen.json'} ({len(jobs)} trainer jobs) and gcp/a4_screen{{,_b2}}.json @ {commit}")
    return {"trainer": spec, "batch1": f1, "batch2": f2}


def seed_main(a: argparse.Namespace, args: list[str], label: str, to: int, p: dict) -> dict:
    out = Path(a.out_dir) if a.out_dir else HERE
    jobs = trainer_jobs(args, to, p, a.g_uri is None, a.seed, a.arm, not a.data_only)
    tf = trainer_file(a.seed)
    spec = {"description": f"attack A4 seed-{a.seed} replicate of arm(s) {a.arm}"
                           f"{' (data only)' if a.data_only else ''}; {label}",
            "note": f"Run through babylm/configs/gcp/a4_s{a.seed}.json.", "plan": p, "jobs": jobs}
    text = json.dumps(spec, indent=1) + "\n"
    (out / tf).write_text(text)
    commit = a.commit or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=HERE).stdout.strip()
    if a.out_dir is None:
        check_pinned(commit, text, tf)
    f = seed_executor_file(commit, p, to, a.g_uri is None, label, a.seed, a.arm, not a.data_only)
    if a.g_uri:
        f["env"]["G_URI"] = a.g_uri
    (out / "gcp").mkdir(exist_ok=True)
    (out / "gcp" / f"a4_s{a.seed}.json").write_text(json.dumps(f, indent=1) + "\n")
    print(f"wrote {out / tf} ({len(jobs)} trainer jobs) and gcp/a4_s{a.seed}.json @ {commit}")
    return {"trainer": spec, "batch": f}


if __name__ == "__main__":
    main()
