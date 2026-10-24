"""Screen jobs for attack A2, input-noise augmentation (card docs/babylm/attacks/A2_input_noise.md):

    python -m babylm.configs.make_a2_jobs --base_job babylm/configs/jobs_base_reg_S.json:NAME   # arms + base arm
    python -m babylm.configs.make_a2_jobs --base_job ... --nulls P DIST                         # the two null twins
    python -m babylm.configs.make_a2_jobs --base_job ... --batch2 A2a                           # claim seeds + nulls + refine
    python -m babylm.configs.make_a2_jobs --base_job ... --solo A2a                             # the same, SOLO (critic ref-0001)

Every job copies the base job's trainer args verbatim (the base round-1 winner, sent by the experimenter) and appends only
--noise_* flags, so the base arm is the base recipe run by the A2 code at default flags (critic ref-0060 amendment 2:
re-run inside the same batch). Every job runs at --vocab_pad 16512 so the base arm, the replace arms, the mask
and R2L arms and the nulls share one init stream (critic ref-0062). Without --base_job the PROVISIONAL stand-in is the port at size S with muon_wd 2.0,
lr_scale 0.5 (the reg stage's starting cell); regenerate before launch. Size S, train_10M x 10, seed 0, --save.
After each run: babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir> (full-DEV dev_bpw = kill metric).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babylm.configs.make_base_jobs import job

HERE = Path(__file__).resolve().parent
ARMS = {  # name: noise flags (card table)
    "A2a": ["--noise_mode", "replace", "--noise_p", "0.05", "--noise_dist", "uniform"],
    "A2b": ["--noise_mode", "replace", "--noise_p", "0.15", "--noise_dist", "uniform"],
    "A2c": ["--noise_mode", "replace", "--noise_p", "0.30", "--noise_dist", "uniform"],
    "A2d": ["--noise_mode", "replace", "--noise_p", "0.15", "--noise_dist", "unigram"],
    "A2e": ["--noise_mode", "replace", "--noise_p", "0.05", "--noise_dist", "uniform", "--noise_r2l", "0.5"],
}


def base_args(spec: str | None) -> tuple[list[str], str, int]:
    """(trainer args without --tag, base label, timeout_s)."""
    if spec is None:
        j = job("provisional", "S", {"--muon_wd": "2.0", "--lr_scale": "0.5"})
        label = "PROVISIONAL stand-in (port, S, muon_wd 2.0, lr_scale 0.5): regenerate with --base_job before launch"
    else:
        path, name = spec.rsplit(":", 1)
        jobs = {j["name"]: j for j in json.loads(Path(path).read_text())["jobs"]}
        j = jobs[name]
        label = f"base job {name} from {path}"
    a = list(j["args"])
    if "--tag" in a:
        i = a.index("--tag")
        del a[i:i + 2]
    if any(x.startswith("--noise_") for x in a):
        raise SystemExit("the base job already sets --noise_* flags")
    if "--seed" not in a or a[a.index("--seed") + 1] != "0" or a[a.index("--size") + 1] != "S":
        raise SystemExit("the A2 screen is size S, seed 0")
    if "--vocab_pad" not in a:  # critic ref-0062: every arm (base, replace, mask, R2L, nulls) shares one init stream
        a += ["--vocab_pad", "16512"]
    elif a[a.index("--vocab_pad") + 1] != "16512":
        raise SystemExit("the A2 screen runs every arm at --vocab_pad 16512 (critic ref-0062)")
    return a, label, int(j.get("timeout_s", 10800))


REFINE = {"p025": "0.025", "p035": "0.035"}  # labelled refinement below A2a (experimenter 05:53Z); not part of the A2a claim


def with_seed(args: list[str], seed: int) -> list[str]:
    a = list(args)
    a[a.index("--seed") + 1] = str(seed)
    return a


def unpadded(args: list[str]) -> list[str]:
    """The base job's own vocab (no --vocab_pad): init- and code-identical to the solo base seed pool."""
    a = list(args)
    i = a.index("--vocab_pad")
    del a[i:i + 2]
    return a


def mk(name: str, args: list[str], noise: list[str], timeout_s: int) -> dict:
    return {"name": name, "module": "babylm.harness.train", "args": [*args, *noise, "--tag", name],
            "timeout_s": timeout_s}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_job", default=None, help="FILE.json:NAME of the base round-1 winner's job")
    ap.add_argument("--nulls", nargs=2, metavar=("P", "DIST"), default=None,
                    help="write the null twins at the best arm's noise_p and noise_dist instead of the arms")
    ap.add_argument("--batch2", default=None, metavar="ARM",
                    help="write batch 2 for the best ARM: claim seeds 1-2 of ARM and base, its nulls, refine pair")
    ap.add_argument("--solo", default=None, metavar="ARM",
                    help="write the SOLO batch for ARM: seeds 0-2, nulls, refine pair, read against the solo base pool")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    args, label, to = base_args(a.base_job)
    if a.solo:
        arm = ARMS[a.solo]
        if arm[:4] != ["--noise_mode", "replace", "--noise_p", arm[3]] or "--noise_r2l" in arm:
            raise SystemExit("the solo batch expects a replace arm without R2L")
        p, dist = arm[3], arm[5]
        u = unpadded(args)  # replace and gauss need no pad ids: same vocab, init and code path as the solo base pool
        jobs = [mk(f"a2s_{a.solo}_s{sd}", with_seed(u, sd), arm, to) for sd in (0, 1, 2)]
        jobs += [mk("a2s_null_gauss", u, ["--noise_mode", "gauss", "--noise_p", p, "--noise_dist", dist], to),
                 mk("a2s_null_mask", args, ["--noise_mode", "mask", "--noise_p", p], to),
                 mk("a2s_base_pad_s0", args, [], to)]  # mask needs the MASK id (pad 16512): its own padded solo base
        jobs += [mk(f"a2s_refine_{k}", u, ["--noise_mode", "replace", "--noise_p", v, "--noise_dist", dist], to)
                 for k, v in REFINE.items()]
        spec = {"description": f"attack A2 SOLO batch for {a.solo} (p {p}, {dist}), one run per GPU: {a.solo} seeds 0-2, "
                               f"gauss null s0 and refine pair (p {', '.join(REFINE.values())}, s0, exploratory) at the "
                               f"base job's own vocab, read against the solo base seed pool; mask null s0 and its padded "
                               f"solo base s0 at --vocab_pad 16512; {label}",
                "note": "Run SOLO (no packing), priority in listed order; refine pair only if slots allow. Score with "
                        "score_trial.sh plus the masked and twin dev_bpw.",
                "jobs": jobs}
        out = Path(a.out) if a.out else HERE / "jobs_a2_solo.json"
    elif a.batch2:
        arm = ARMS[a.batch2]
        if arm[:4] != ["--noise_mode", "replace", "--noise_p", arm[3]] or "--noise_r2l" in arm:
            raise SystemExit("batch 2 expects a replace arm without R2L")
        p, dist = arm[3], arm[5]
        jobs = []
        for sd in (1, 2):  # packed pairs (arm, base) per seed
            jobs += [mk(f"a2_{a.batch2}_s{sd}", with_seed(args, sd), arm, to), mk(f"a2_base_s{sd}", with_seed(args, sd), [], to)]
        jobs += [mk("a2_null_mask", args, ["--noise_mode", "mask", "--noise_p", p], to),
                 mk("a2_null_gauss", args, ["--noise_mode", "gauss", "--noise_p", p, "--noise_dist", dist], to)]
        jobs += [mk(f"a2_refine_{k}", args, ["--noise_mode", "replace", "--noise_p", v, "--noise_dist", dist], to)
                 for k, v in REFINE.items()]
        spec = {"description": f"attack A2 batch 2 for {a.batch2} (p {p}, {dist}): claim seeds 1-2 of {a.batch2} and the "
                               f"base, the two null twins (seed 0), and a labelled refinement pair (p "
                               f"{', '.join(REFINE.values())}, seed 0, not part of the claim); {label}",
                "note": "Pack two per GPU in listed order: (arm_s1, base_s1), (arm_s2, base_s2), (mask, gauss), "
                        "(refine_p025, refine_p035); packed vs packed only (critic ref-0027). Score with score_trial.sh "
                        "plus the masked and twin dev_bpw.",
                "jobs": jobs}
        out = Path(a.out) if a.out else HERE / "jobs_a2_batch2.json"
    elif a.nulls:
        p, dist = a.nulls
        jobs = [mk("a2_null_mask", args, ["--noise_mode", "mask", "--noise_p", p], to),
                mk("a2_null_gauss", args, ["--noise_mode", "gauss", "--noise_p", p, "--noise_dist", dist], to)]
        spec = {"description": f"attack A2 null twins at p {p}, dist {dist} (the best arm on full DEV); {label}",
                "note": "Same base, seed and data order as jobs_a2_screen.json. Score with score_trial.sh.",
                "jobs": jobs}
        out = Path(a.out) if a.out else HERE / "jobs_a2_nulls.json"
    else:
        jobs = [mk("a2_base", args, [], to)] + [mk(f"a2_{k}", args, v, to) for k, v in ARMS.items()]
        spec = {"description": f"attack A2 screen: base arm + arms A2a-e (input-noise augmentation), size S, "
                               f"train_10M x 10, seed 0; {label}",
                "note": "Run all 6 in one batch on one GPU type. Then score each with score_trial.sh; the best arm "
                        "(lowest full-DEV nats/word) sets --nulls P DIST for jobs_a2_nulls.json.",
                "jobs": jobs}
        out = Path(a.out) if a.out else HERE / "jobs_a2_screen.json"
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(f"wrote {out} ({len(spec['jobs'])} jobs)")
    return spec


if __name__ == "__main__":
    main()
