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
    jobs = [job(f"base_{size}_wd{tag(wd)}_lr{tag(lr)}_{n}", size, {**base, **c}, timeout_h) for n, c in cells]
    # dropout (Xue et al. 2023: the most effective fix for multi-epoch degradation): babylm.models.baseline_reg,
    # embedding + residual dropout; p = 0 is exactly the frozen model
    jobs += [job(f"base_{size}_wd{tag(wd)}_lr{tag(lr)}_do{tag(d)}", size, {**base, "--model": "baseline_reg"},
                 timeout_h, (f"dropout={d}",)) for d in ("0.1", "0.2")]
    return {"description": f"attack-base stage reg: size {size} at s1 winner muon_wd {wd} lr_scale {lr}; epochs "
                           "{2,4} (full schedule each), head AdamW wd {0.03,0.1,0.3}, embed AdamW wd {0.001}, "
                           "EMA {0.99,0.998}, dropout {0.1,0.2} (baseline_reg)",
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
    ap.add_argument("--base_flags", default="", help="the reg winner's extra trainer flags, e.g. '--head_wd 0.1'")
    ap.add_argument("--grid", default=None, metavar="SIZE",
                    help="write only jobs_base_grid_SIZE.json: the --wd x --lr cells (e.g. extending s1 upward)")
    ap.add_argument("--wd", nargs="+", default=["4.0", "8.0"])
    ap.add_argument("--lr", nargs="+", default=["1", "0.5"])
    a = ap.parse_args(argv)
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
        f.write_text(json.dumps(stage_reg(size, wd, lr), indent=1) + "\n")
        print("wrote", f)
        return
    for name, spec in (("s1", stage_s1()), ("s2_small", stage_s2("small")), ("s2_M", stage_s2("M")),
                       ("gpu_smoke", gpu_smoke())):
        (HERE / f"jobs_base_{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
        print("wrote", HERE / f"jobs_base_{name}.json")


if __name__ == "__main__":
    main()
