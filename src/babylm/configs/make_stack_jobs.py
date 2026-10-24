"""Stack jobs: the promoted attack arms combined on the base round-1 winner (docs/babylm/attacks/STACKING.md).

    python -m babylm.configs.make_stack_jobs \\
        --arm A1=babylm/configs/jobs_a1_r1_S.json:r1_A1b --arm A2=babylm/configs/jobs_a2_screen.json:a2_A2b \\
        [--arm A3=FILE:NAME] [--arm A4=FILE:NAME] [--seeds 0 1 2] [--loo] [--out FILE]

Each arm is named by its promoted screen job. Only its attack flags are taken (A1 --sd_*, A2 --noise_*, A3
--bpe_dropout*, A4 --init_from/--sources/... ); every other trainer setting must parse to the winner's (checked with
train.py's own parser), so a stack job differs from the winner only in attack flags, --seed, --tag and, when an arm
needs it, --vocab_pad. A padded vocab is applied to every job of the file (stacks, leave-one-out stacks and the base
re-runs written then), so all of them share one init stream (critic ref-0062). --loo also writes every leave-one-out
stack (each promoted arm removed in turn) for the attribution in the final report.
A4 makes a job a second LR cycle (--init_from its G, --sources ...): --epochs is dropped, as in A4's own jobs, and G is
A4's seed-0 generator for every seed (a screen). Claim runs pass --own_g: seed k then reads every A4 run directory as its
`_s{k}` twin (a4_G_s1/, a4_data_s1/, ...), which the A4 thread builds first. A3 + A4 is refused (train.py refuses it). Pool: promotion. No GPU is used here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WINNER = "babylm/configs/jobs_base_r1_S.json:base_S_wd0p4_lr0p5_r1_ve0_tied"
WINNER_COMMIT = "ref-0065"
ORDER = ("A1", "A2", "A3", "A4")
A4_FLAGS = ("--init_from", "--init_weights", "--sources", "--cycle_window", "--warmup_frac", "--exposure_prior",
            "--exposure_extra", "--exposure_cap")
ON = {"A1": lambda ns: ns["sd_mode"] != "none", "A2": lambda ns: ns["noise_mode"] != "none",
      "A3": lambda ns: ns["bpe_dropout"] is not None, "A4": lambda ns: bool(ns["sources"])}
FREE = ("seed", "tag", "vocab_pad")  # settings a stack job may change besides attack flags


def attack_of(flag: str) -> str | None:
    if flag.startswith("--sd_"):
        return "A1"
    if flag.startswith("--noise_"):
        return "A2"
    if flag.startswith("--bpe_dropout"):
        return "A3"
    if flag in A4_FLAGS:
        return "A4"
    return None


def attack_dest(dest: str) -> str | None:
    return attack_of("--" + dest)


def attack_dests(ns: dict) -> list[str]:
    return [k for k in ns if attack_dest(k)]


def groups(args: list[str]) -> list[tuple[str, list[str]]]:
    """Trainer args as (flag, values) in order; values never start with '--' in these job files."""
    out: list[tuple[str, list[str]]] = []
    for a in args:
        if a.startswith("--"):
            out.append((a, []))
        else:
            if not out:
                raise ValueError(f"value {a!r} before any flag")
            out[-1][1].append(a)
    return out


def flat(gs: list[tuple[str, list[str]]]) -> list[str]:
    return [t for f, v in gs for t in (f, *v)]


def load_job(spec: str) -> dict:
    path, name = spec.rsplit(":", 1)
    p = Path(path) if Path(path).is_absolute() else REPO / path  # job files name repo-relative paths
    jobs = {j["name"]: j for j in json.loads(p.read_text())["jobs"]}
    if name not in jobs:
        raise SystemExit(f"{name} not in {path}")
    return jobs[name]


def parsed(args: list[str]) -> dict:
    from babylm.harness import train as TR
    return vars(TR.build_parser().parse_args(args))


def non_attack(ns: dict, extra_free: tuple[str, ...] = ()) -> dict:
    return {k: v for k, v in ns.items() if attack_dest(k) is None and k not in FREE + extra_free}


def arm_flags(name: str, job: dict, winner_ns: dict) -> list[tuple[str, list[str]]]:
    """The arm's attack flags (last occurrence of each), after checking every other setting equals the winner's."""
    a = job["args"]
    ns = parsed(a)
    free = ("epochs",) if name == "A4" else ()
    diff = {k: (winner_ns.get(k), v) for k, v in non_attack(ns, free).items()
            if non_attack(winner_ns, free).get(k) != v}
    if diff:
        raise SystemExit(f"arm {name} ({job['name']}) differs from the winner outside attack flags: {diff}")
    own: dict[str, list[str]] = {}
    for f, v in groups(a):
        att = attack_of(f)
        if att is None:
            continue
        if att != name:
            raise SystemExit(f"arm {name} ({job['name']}) carries {att} flag {f}")
        own[f] = v
    if not own or not ON[name](ns):
        raise SystemExit(f"arm {name} ({job['name']}) sets no {name} flags (is it the screen's base arm?)")
    return list(own.items())


RUNS = "babylm/runs/"


def own_g(flags: list[tuple[str, list[str]]], seed: int) -> list[tuple[str, list[str]]]:
    """A4 flags for a claim run at `seed`: every run directory they name (G, its synthetic data, half-generators) becomes
    that directory's `_s{seed}` twin, so seed k starts from its own G_k. Seed 0 keeps the screen's own directories."""
    if seed == 0:
        return flags

    def sub(x: str) -> str:
        i = x.find(RUNS)
        if i < 0:
            return x
        j = i + len(RUNS)
        k = x.find("/", j)
        k = len(x) if k < 0 else k
        return f"{x[:k]}_s{seed}{x[k:]}"
    return [(f, [sub(v) for v in vs]) for f, vs in flags]


def g_dirs(flags: list[tuple[str, list[str]]]) -> list[str]:
    out = []
    for _, vs in flags:
        for v in vs:
            i = v.find(RUNS)
            if i >= 0:
                d = v[i:].split("/")[:3]
                out.append("/".join(d))
    return sorted(set(out))


def stack_job(tag: str, seed: int, winner_args: list[str], arms: dict[str, list], pad: int | None,
              timeout_s: int, own: bool = False) -> dict:
    gs = [(f, v) for f, v in groups(winner_args) if f not in ("--tag", "--seed", "--vocab_pad")]
    if "A4" in arms:
        gs = [(f, v) for f, v in gs if f != "--epochs"]  # a cycle's length is its --sources plan
    for k in ORDER:
        gs += own_g(arms[k], seed) if (own and k == "A4" and k in arms) else arms.get(k, [])
    gs += [("--seed", [str(seed)])] + ([("--vocab_pad", [str(pad)])] if pad else []) + [("--tag", [tag])]
    return {"name": tag, "module": "babylm.harness.train", "args": flat(gs), "timeout_s": timeout_s}


def build(arm_specs: dict[str, str], seeds: list[int], loo: bool, winner_spec: str = WINNER,
          own: bool = False, singles: bool = False) -> dict:
    if own and "A4" not in arm_specs:
        raise SystemExit("--own_g needs an A4 arm")
    if len(arm_specs) < 2:
        raise SystemExit("a stack needs at least two promoted arms")
    if "A3" in arm_specs and "A4" in arm_specs:
        raise SystemExit("A3 + A4 do not compose yet (train.py refuses --bpe_dropout with --sources)")
    w = load_job(winner_spec)
    wns = parsed(w["args"])
    if any(attack_of(f) for f, _ in groups(w["args"])):
        raise SystemExit("the winner job already sets attack flags")
    jobs_by_arm = {k: load_job(v) for k, v in arm_specs.items()}
    arms = {k: arm_flags(k, j, wns) for k, j in jobs_by_arm.items()}
    pads = {int(p) for p in (parsed(j["args"]).get("vocab_pad") for j in jobs_by_arm.values()) if p}
    if len(pads) > 1:
        raise SystemExit(f"arms use different --vocab_pad values {sorted(pads)}")
    pad = pads.pop() if pads else None
    if pad and wns.get("vocab_pad") and int(wns["vocab_pad"]) != pad:
        raise SystemExit("the winner and the arms use different --vocab_pad values")
    to = max(int(w.get("timeout_s", 10800)), *(int(j.get("timeout_s", 10800)) for j in jobs_by_arm.values()))
    names = [k for k in ORDER if k in arms]
    subsets = [names] + ([[k for k in names if k != drop] for drop in names] if loo else [])
    # a stack is judged against the best single promoted arm too (critic ref-0021): an arm whose own runs used another
    # --vocab_pad has no same-init 3-seed comparator, so write it alone at the stack's settings (--singles: every arm)
    own_pad = {k: parsed(j["args"]).get("vocab_pad") for k, j in jobs_by_arm.items()}
    subsets += [[k] for k in names if (singles or (own_pad[k] or None) != pad) and [k] not in subsets]
    jobs = []
    for sub in subsets:
        label = "stack_" + "".join(k.lower() for k in sub)
        for s in seeds:
            jobs.append(stack_job(f"{label}_s{s}", s, w["args"], {k: arms[k] for k in sub}, pad, to, own))
    if pad:  # the winner's own seeds are unpadded: re-run the base at the stack's init stream
        for s in seeds:
            jobs.append(stack_job(f"stack_base_s{s}", s, w["args"], {}, pad, to))
    desc = (f"attack stack {'+'.join(names)} on {w['name']} ({winner_spec.rsplit(':', 1)[0]} @ {WINNER_COMMIT}), "
            f"seeds {seeds}" + (", with leave-one-out stacks" if loo else "")
            + (f", every job at --vocab_pad {pad} (base re-run included)" if pad else ""))
    return {"description": desc, "pool": "promotion", "solo": True,
            "note": "Pool: promotion. Arms: " + "; ".join(f"{k}={arm_specs[k]}" for k in names)
                    + ". Verdicts read score.json dev_nats_per_word_twin (critic ref-0021; v1 dev_nats_per_word for"
                    " information only), never result.json full_eval. Judged as a claim (critic ref-0019): 3-seed means"
                    " against both the base and the best single promoted arm, at the claim floor (pooled SD of n >= 5"
                    " plain base seeds) registered before any claim is read; BLiMP-fast not down by more than 1.4."
                    " The bar must hold on v2 full DEV and on v2 DEV without its never-seen-character words"
                    " (unseen_chars.py). Claims run SOLO, one job per GPU, never packed (critic ref-0001)."
                    + ((" Claim run: seed k starts from its own G_k (--own_g); the A4 thread must first build "
                        + ", ".join(d for s in seeds if s for d in g_dirs(own_g(arms["A4"], s))) + "."
                        + " G_k reuses the seed-0 card's recipe, generation settings and mix: no per-seed tuning, no picking among G_k"
                        " (critic ref-0117)." if own else " Shared-G screen: A4 cycles start from A4's seed-0 G for every"
                        " seed, and G_0's full exposure is charged to each run; a claim needs --own_g (critic ref-0117).") if "A4" in arms else ""),
            "arms": {k: {"job": arm_specs[k], "flags": flat(arms[k])} for k in names},
            "jobs": jobs}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="AK=FILE.json:NAME",
                    help="a promoted arm: its attack (A1..A4) and its screen job")
    ap.add_argument("--winner", default=WINNER, help="FILE.json:NAME of the base round-1 winner's job")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--loo", action="store_true", help="also write every leave-one-out stack")
    ap.add_argument("--singles", action="store_true",
                    help="also write every arm alone at the stack's settings (always done for arms whose own runs used "
                         "another --vocab_pad)")
    ap.add_argument("--own_g", action="store_true",
                    help="claim runs: seed k's A4 cycle starts from its own G_k (babylm/runs/<dir>_s<k>/ for k > 0)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    specs = {}
    for x in a.arm:
        k, _, v = x.partition("=")
        if k not in ORDER or not v or k in specs:
            raise SystemExit(f"bad --arm {x!r} (want A1..A4=FILE.json:NAME, each attack once)")
        specs[k] = v
    spec = build(specs, a.seeds, a.loo, a.winner, a.own_g, a.singles)
    out = Path(a.out) if a.out else HERE / ("jobs_stack_" + "".join(k.lower() for k in ORDER if k in specs) + ".json")
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(f"wrote {out} ({len(spec['jobs'])} jobs)")
    return spec


if __name__ == "__main__":
    main()
