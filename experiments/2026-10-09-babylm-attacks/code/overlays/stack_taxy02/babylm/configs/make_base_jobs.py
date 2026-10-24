"""Job lists for the attack-base sweep (base builder thread): python -m babylm.configs.make_base_jobs

Every job = the frozen port (make_jobs.FROZEN) + one cell's overrides, train_10M x 10 epochs, seed 0, --save, so
the final checkpoint can be exported and scored by babylm/harness/score_trial.sh (DEV nats/word via dev_bpw.py and
the official fast suite). Greedy: each later stage starts from the previous stage's winner (recorded in
babylm/configs/base_sweep_log.md with every trial's score; the trial count is reported to the critic at freeze).

Stage s1 (size S): muon_wd x lr_scale. The four WD values at lr_scale 1 are also the experimenter's registered P4 cells.
Stage s2 (sizes small and M): the same joint grid (Kim, Kotha, Liang & Hashimoto 2025: the best WD grows with size
and LR must be tuned with it). Size L follows once the S and M optima show the trend (cost: ~2.5x M per run).
Stage reg (after s1 showed wd0 overfitting from epoch 2: DEV-sub minimum 5.48 at 20M words, 9.70 at 100M), at the
s1 winner's (muon_wd, lr_scale): epochs {2, 4} with the full schedule each (fewer than 10 is legal; the honest
early-stopping baseline), and AdamW decay on the untied head {0.03, 0.1, 0.3} and embeddings {0.001} (muon_wd only
touches matrices), EMA of weights {0.99, 0.998} (evals and
exports use the EMA), and dropout {0.1, 0.2} (babylm.models.baseline_reg; Xue et al. 2023). 10 trials. Written by: python -m babylm.configs.make_base_jobs --reg SIZE WD LR. Every trial's result.json
logs dev_sub_min (minimum and its epoch); selection stays on the final model.
Stage cap (token-keyed lookup capacity: at S the input embedding, 3 value-embedding tables and the untied head hold
~31.5M params against ~21M in the blocks, for 16M training tokens), at the reg winner: n_value_embeds {0, 1}
(default 3) and tied input/output embeddings (Press & Wolf 2017; tied weight initialised and optimised as the head).
Written by: python -m babylm.configs.make_base_jobs --cap SIZE WD LR [--base_flags "<reg winner flags>"]. 3 trials.
"""
from __future__ import annotations

import json
from pathlib import Path

from babylm.configs.make_jobs import FROZEN, RUN

HERE = Path(__file__).resolve().parent


def override(args: list[str], flag: str, value: str) -> list[str]:
    a = list(args)
    if flag in a:
        a[a.index(flag) + 1] = value
    else:
        a += [flag, value]
    return a


def job(name: str, size: str, cell: dict[str, str], timeout_h: float = 3.0, extra: tuple[str, ...] = ()) -> dict:
    """extra: KEY=VAL model extras appended as further --extra flags (FROZEN already carries two)."""
    args = ["--model", "baseline", "--size", size, "--split", "train_10M", "--epochs", "10", "--seed", "0", *FROZEN]
    for k, v in cell.items():
        args = override(args, k, v)
    for kv in extra:
        args += ["--extra", kv]
    return {"name": name, "module": "babylm.harness.train", "args": [*args, *RUN, "--tag", name],
            "timeout_s": int(timeout_h * 3600)}


def tag(x: str) -> str:
    return x.replace(".", "p")


def wd_lr_grid(size: str, timeout_h: float = 3.0, lrs=("1", "0.5"), wds=("0", "0.1", "0.4", "1.0", "2.0")) -> list[dict]:
    jobs = []
    for lr in lrs:
        for wd in wds:
            jobs.append(job(f"base_{size}_wd{tag(wd)}_lr{tag(lr)}", size, {"--muon_wd": wd, "--lr_scale": lr},
                            timeout_h))
    return jobs


def stage_s1() -> dict:
    jobs = wd_lr_grid("S")
    return {"description": "attack-base stage s1: size S, muon_wd {0,0.1,0.4,1,2} x lr_scale {1,0.5}, seed 0, "
                           "train_10M x 10 epochs; P4 cells = the lr_scale 1 rows with wd 0/0.1/0.4/1.0",
            "note": "Order: lr_scale 1 first. After each run: babylm/harness/score_trial.sh babylm/runs/<name> "
                    "<eval workdir>.",
            "jobs": jobs}


def stage_s2(size: str) -> dict:
    return {"description": f"attack-base stage s2: size {size}, muon_wd {{0,0.1,0.4,1,2}} x lr_scale {{1,0.5}}, "
                           "seed 0, train_10M x 10 epochs",
            "note": "After each run: babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>.",
            "jobs": wd_lr_grid(size, 3.0 if size == "small" else 6.0)}


def stage_reg(size: str, wd: str, lr: str, timeout_h: float = 3.0) -> dict:
    base = {"--muon_wd": wd, "--lr_scale": lr}
    cells = [(f"ep{e}", {"--epochs": e}) for e in ("2", "4")]
    cells += [(f"hwd{tag(h)}", {"--head_wd": h}) for h in ("0.03", "0.1", "0.3")]
    cells += [("ewd0p001", {"--embed_wd": "0.001"})]
    # EMA of weights for eval (the published GPT-BERT 10M gains +0.048 nats/word from mixing in epoch-5 predictions):
    # horizons ~0.4 and ~2 epochs at 244 steps/epoch
    cells += [(f"ema{tag(d)}", {"--ema_decay": d}) for d in ("0.99", "0.998")]
    # MNTP hybrid (GPT-BERT's regulariser / data multiplier), share of sequences per micro-batch
    cells += [(f"mntp{tag(m)}", {"--mntp_frac": m}) for m in ("0.25", "0.5")]
    jobs = [job(f"base_{size}_wd{tag(wd)}_lr{tag(lr)}_{n}", size, {**base, **c}, timeout_h) for n, c in cells]
    # dropout (Xue et al. 2023: the most effective fix for multi-epoch degradation): babylm.models.baseline_reg,
    # embedding + residual dropout; p = 0 is exactly the frozen model
    jobs += [job(f"base_{size}_wd{tag(wd)}_lr{tag(lr)}_do{tag(d)}", size, {**base, "--model": "baseline_reg"},
                 timeout_h, (f"dropout={d}",)) for d in ("0.1", "0.2")]
    return {"description": f"attack-base stage reg: size {size} at muon_wd {wd} lr_scale {lr}; epochs "
                           "{2,4} (full schedule each), head AdamW wd {0.03,0.1,0.3}, embed AdamW wd {0.001}, "
                           "EMA {0.99,0.998}, MNTP {0.25,0.5}, dropout {0.1,0.2} (baseline_reg)",
            "note": "After each run: babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>. "
                    "result.json dev_sub_min logs the DEV-sub minimum and its epoch.",
            "jobs": jobs}


def stage_cap(size: str, wd: str, lr: str, base_flags: list[str], timeout_h: float = 3.0) -> dict:
    base = {"--muon_wd": wd, "--lr_scale": lr}
    flags = list(base_flags)
    model = "baseline"
    if "--model" in flags:
        i = flags.index("--model")
        model = flags[i + 1]
        del flags[i:i + 2]
    extras = tuple(flags[i + 1] for i, a in enumerate(flags) if a == "--extra")
    pairs = [a for i, a in enumerate(flags) if a != "--extra" and (i == 0 or flags[i - 1] != "--extra")]
    cell = dict(base, **{"--model": model})
    k = None
    for a in pairs:  # --flag value / --switch
        if a.startswith("--"):
            k = a
            cell[k] = ""
        elif k is not None:
            cell[k] = a
            k = None
    switches = [f for f, v in cell.items() if v == ""]
    cell = {f: v for f, v in cell.items() if v != ""}
    stem = f"base_{size}_wd{tag(wd)}_lr{tag(lr)}_cap"
    jobs = [job(f"{stem}_ve{n}", size, cell, timeout_h, extras + (f"n_value_embeds={n}",)) for n in ("0", "1")]
    jobs.append(job(f"{stem}_tied", size, cell, timeout_h, extras))
    for j in jobs:
        j["args"] = j["args"][:-len(RUN) - 2] + switches + (["--tie_embeddings"] if j["name"].endswith("_tied") else []) \
            + j["args"][-len(RUN) - 2:]
    return {"description": f"attack-base stage cap: size {size} at the reg winner (muon_wd {wd}, lr_scale {lr}, "
                           f"{' '.join(base_flags) or 'no reg flags'}): n_value_embeds {{0,1}} (default 3), tied "
                           "embeddings",
            "note": "After each run: babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>.",
            "jobs": jobs}


def stage_round1(size: str, wd: str, lr: str, timeout_h: float = 3.0, all_wds=("0", "0.4")) -> dict:
    """Round 1 (experimenter 23:40Z): s1 decays only the Muon matrices, while the AdamW lookup tables (input embedding,
    untied head, value embeddings; ~60% of the parameters at S) run at adam_wd 0 and are the likely memoriser.
    One cell per lever plus pairs and an all-on cell, at the s1 winner."""
    base = {"--muon_wd": wd, "--lr_scale": lr}
    stem = f"base_{size}_wd{tag(wd)}_lr{tag(lr)}_r1"
    # order = launch priority (experimenter 00:10Z): executor 3 gives VM A the first 6 and VM B the last 5, so all-on and
    # embed wd 0.005 experimenter on A and ve0+tied leads on B (before the two lighter-WD all-on cells are inserted below)
    cells = [  # (name, cell, extras, switches)
        # tied + no value embeds leaves no 'embed' AdamW group: the one lookup table is the tied weight, decayed by
        # --head_wd (an --embed_wd here makes the trainer refuse to start)
        ("all", {"--model": "baseline_reg", "--head_wd": "0.1", "--mntp_frac": "0.25"},
         ("dropout=0.1", "n_value_embeds=0"), ("--tie_embeddings",)),
        ("ewd0p005", {"--embed_wd": "0.005"}, (), ()),
        ("ve0", {}, ("n_value_embeds=0",), ()),
        ("tied", {}, (), ("--tie_embeddings",)),
        ("ewd0p001", {"--embed_wd": "0.001"}, (), ()),
        ("hwd0p1", {"--head_wd": "0.1"}, (), ()),
        ("ve0_tied", {}, ("n_value_embeds=0",), ("--tie_embeddings",)),
        ("do0p1", {"--model": "baseline_reg"}, ("dropout=0.1",), ()),
        ("ewd0p001_hwd0p1", {"--embed_wd": "0.001", "--head_wd": "0.1"}, (), ()),
        ("mntp0p25", {"--mntp_frac": "0.25"}, (), ()),
        ("ep4", {"--epochs": "4"}, (), ()),
    ]
    def mk(stem_, base_, n, c, extras, switches):
        j = job(f"{stem_}_{n}", size, {**base_, **c}, timeout_h, extras)
        j["args"] = j["args"][:-len(RUN) - 2] + list(switches) + j["args"][-len(RUN) - 2:]
        return j

    jobs = [mk(stem, base, *c) for c in cells]
    # all-on at lighter Muon WD (experimenter 00:12Z): wd 2 won s1 only while the tables were unregularised. Inserted third
    # on each VM as launched 00:30Z (A = first 7 jobs, B = the next 6; the 2 embed-wd 0.02 cells came later)
    allc = cells[0]
    lighter = [mk(f"base_{size}_wd{tag(w)}_lr{tag(lr)}_r1", {"--muon_wd": w, "--lr_scale": lr}, *allc)
               for w in all_wds]
    for k, j in zip((2, 9), lighter):
        jobs.insert(k, j)
    # short full schedules at muon_wd 0 (01:20Z): the port's DEV-sub minimum (5.43 at epoch 2, lr 0.5) beats every
    # 10-epoch cell, and a run that ends at its minimum gets the LR cooldown there too; plus embed wd 0.005 at wd 0
    w0 = {"--muon_wd": "0", "--lr_scale": lr}
    stem0 = f"base_{size}_wd0_lr{tag(lr)}_r1"
    jobs += [mk(stem0, w0, f"ep{e}", {"--epochs": e}, (), ()) for e in ("2", "3")]
    jobs += [mk(stem0, w0, "ewd0p005", {"--embed_wd": "0.005"}, (), ()),
             mk(stem0, w0, "ewd0p005_ep4", {"--embed_wd": "0.005", "--epochs": "4"}, (), ())]
    # 01:20Z: ve0_tied at wd 2 ended at its minimum (5.32 full DEV) and still falling: the same cell at lighter Muon WD
    # 01:35Z (instruments): wd2/lr0.5 costs ~0.5 nats/word early; ve0_tied at muon_wd 0 and at lr 1
    for w, l in (("0.4", lr), ("1.0", lr), ("0", lr), ("0.4", "1")):
        jobs.append(mk(f"base_{size}_wd{tag(w)}_lr{tag(l)}_r1", {"--muon_wd": w, "--lr_scale": l}, "ve0_tied", {},
                       ("n_value_embeds=0",), ("--tie_embeddings",)))
    return {"description": f"attack-base round 1: size {size} at muon_wd {wd} lr_scale {lr}: AdamW-table levers "
                           "(no value embeds, tied, embed wd {0.001,0.005}, head wd 0.1, both), dropout 0.1, "
                           "MNTP 0.25 (mask p 0.3->0.15), 4 epochs, and all-on (tied+ve0+hwd+do+MNTP), also at muon_wd "
                           f"{{{','.join(all_wds)}}}",
            "note": "60-step CUDA smoke first (MNTP/dropout/tied compiled on CUDA). After each run: "
                    "babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>.",
            "jobs": gpu_smoke()["jobs"] + jobs}


def stage_seeds(jobfile: str, name: str, seeds=("1", "2")) -> dict:
    """Seeds 1 and 2 of a chosen trial (critic noise floor): the job unchanged except --seed (data order follows
    the seed) and the name/tag."""
    src = next(j for j in json.loads(Path(jobfile).read_text())["jobs"] if j["name"] == name)
    jobs = []
    for sd in seeds:
        a = override(list(src["args"]), "--seed", sd)
        a = override(a, "--tag", f"{name}_s{sd}")
        jobs.append({**src, "name": f"{name}_s{sd}", "args": a})
    return {"description": f"seeds {', '.join(seeds)} of {name} (from {Path(jobfile).name})",
            "note": "After each run: babylm/harness/score_trial.sh babylm/runs/<name> <eval workdir>.", "jobs": jobs}


def gpu_smoke() -> dict:
    """60 steps at size small with every lever on, compiled, on CUDA: catches compile/MNTP/EMA/plan problems before
    the lever stages use real compute. Not a tuning trial (no score, not counted)."""
    lev = {"--window_plan": "line:128x6,line:256x2,line:512x2", "--mntp_frac": "0.25", "--ema_decay": "0.999",
           "--steps": "60", "--milestone_scale": "0.01"}
    j = job("base_gpu_smoke_levers", "small", lev, 0.5)
    j["args"] = [a for a in j["args"] if a not in ("--save",)] + ["--no_pt"]
    return {"description": "CUDA smoke of all base levers (60 steps, small, compiled); not a tuning trial",
            "jobs": [j]}


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", nargs=3, metavar=("SIZE", "WD", "LR"), default=None,
                    help="write only jobs_base_reg_SIZE.json at the given winner")
    ap.add_argument("--cap", nargs=3, metavar=("SIZE", "WD", "LR"), default=None,
                    help="write only jobs_base_cap_SIZE.json at the reg winner")
    ap.add_argument("--r1", nargs=3, metavar=("SIZE", "WD", "LR"), default=None,
                    help="write only jobs_base_r1_SIZE.json (round 1) at the s1 winner")
    ap.add_argument("--seeds", nargs=2, metavar=("JOBFILE", "NAME"), default=None,
                    help="write only jobs_base_seeds_NAME.json: seeds 1 and 2 of trial NAME from JOBFILE")
    ap.add_argument("--base_flags", default="", help="the reg winner's extra trainer flags, e.g. '--head_wd 0.1'")
    ap.add_argument("--grid", default=None, metavar="SIZE",
                    help="write only jobs_base_grid_SIZE.json: the --wd x --lr cells (e.g. extending s1 upward)")
    ap.add_argument("--wd", nargs="+", default=["4.0", "8.0"])
    ap.add_argument("--lr", nargs="+", default=["1", "0.5"])
    a = ap.parse_args(argv)
    if a.seeds:
        f = HERE / f"jobs_base_seeds_{a.seeds[1]}.json"
        f.write_text(json.dumps(stage_seeds(*a.seeds), indent=1) + "\n")
        print("wrote", f)
        return
    if a.r1:
        size, wd, lr = a.r1
        f = HERE / f"jobs_base_r1_{size}.json"
        f.write_text(json.dumps(stage_round1(size, wd, lr), indent=1) + "\n")
        print("wrote", f)
        return
    if a.grid:
        f = HERE / f"jobs_base_grid_{a.grid}.json"
        spec = {"description": f"attack-base WD x LR extension: size {a.grid}, muon_wd {a.wd} x lr_scale {a.lr}, seed 0, "
                               "train_10M x 10 epochs", "jobs": wd_lr_grid(a.grid, 3.0, tuple(a.lr), tuple(a.wd))}
        f.write_text(json.dumps(spec, indent=1) + "\n")
        print("wrote", f)
        return
    if a.cap:
        import shlex
        size, wd, lr = a.cap
        f = HERE / f"jobs_base_cap_{size}.json"
        f.write_text(json.dumps(stage_cap(size, wd, lr, shlex.split(a.base_flags)), indent=1) + "\n")
        print("wrote", f)
        return
    if a.reg:
        size, wd, lr = a.reg
        f = HERE / f"jobs_base_reg_{size}.json"
        spec = stage_reg(size, wd, lr)
        # the 60-step CUDA smoke first: the EMA and MNTP cells are the first compiled CUDA runs of those levers
        spec["jobs"] = gpu_smoke()["jobs"] + spec["jobs"]
        f.write_text(json.dumps(spec, indent=1) + "\n")
        print("wrote", f)
        return
    for name, spec in (("s1", stage_s1()), ("s2_small", stage_s2("small")), ("s2_M", stage_s2("M")),
                       ("gpu_smoke", gpu_smoke())):
        (HERE / f"jobs_base_{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
        print("wrote", HERE / f"jobs_base_{name}.json")


if __name__ == "__main__":
    main()
