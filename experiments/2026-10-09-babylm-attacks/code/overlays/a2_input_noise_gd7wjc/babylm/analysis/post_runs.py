"""Post-steps for finished harness runs (GPU VM or local), then a compact results pack.

    python -m babylm.analysis.post_runs [--runs_root babylm/runs] [--pack babylm/results_pack] [--pt_runs NAME ...]

Every run dir under --runs_root with a result.json gets, once each (a step is skipped when its output exists):
  pw/            per-word nats (babylm.analysis.perword --epochs_only)
  hf/            HF export of the last milestone checkpoint (babylm.harness.export_hf --verify)
  dev_bpw.json   the registered DEV nats/word: babylm/eval/dev_bpw.py (byte-identical to the ruler's ref-0050; 10850895 gives the same numbers for this format),
                 full dev split, in this recipe's committed training format babylm/recipes/harness/format.json
                 (critic ref-0051: whitespace collapsed, plain newlines)
  train_loss.png, eval_curves.png   (babylm.harness.plot_curves)
Then --pack/<run>/ receives the small files (config.json, result.json, curves.csv, train_log.csv, the two PNGs,
dev_bpw.json, hf/config.json, hf/hinton_export.json, pw/*) and, for runs named in --pt_runs, the per-token dumps of
the full DEV split and of DEV-sub at the run's last label (for probability-averaged seed ensembles), plus DEV-sub at
every milestone from one epoch on (snapshot mixtures across epochs). Every step's outcome is appended to
--pack/post_log.jsonl; a failed step does not stop the other steps or runs. Exit code 1 if any step failed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from babylm.analysis import perword
from babylm.data.corpus import DEFAULT_DATA_DIR, REPO_ROOT

CK_RE = re.compile(r"^chck_(\d+)M\.pt$")
FORMAT = json.loads((REPO_ROOT / "babylm" / "recipes" / "harness" / "format.json").read_text())
SMALL = ("config.json", "result.json", "curves.csv", "train_log.csv", "train_loss.png", "eval_curves.png",
         "dev_bpw.json", "hf/config.json", "hf/hinton_export.json")


def last_ckpt(run: Path) -> Path:
    cks = [(int(m.group(1)), p) for p in (run / "checkpoints").glob("chck_*M.pt") if (m := CK_RE.match(p.name))]
    if not cks:
        raise FileNotFoundError(f"no checkpoints/chck_*M.pt in {run}")
    return max(cks)[1]


def sh(*cmd: object) -> None:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=REPO_ROOT)
    if r.returncode:
        raise RuntimeError(f"rc={r.returncode}: {' '.join(map(str, cmd))}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_root", default=str(REPO_ROOT / "babylm" / "runs"))
    ap.add_argument("--pack", default=str(REPO_ROOT / "babylm" / "results_pack"))
    ap.add_argument("--pt_runs", nargs="*", default=[], help="runs whose final per-token DEV dumps go in the pack")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    a = ap.parse_args(argv)
    pack, dev_dir = Path(a.pack), Path(a.data_dir) / "raw" / "dev"
    pack.mkdir(parents=True, exist_ok=True)
    failed = 0
    with open(pack / "post_log.jsonl", "a") as log:
        def step(run: Path, name: str, done: bool, fn: Callable[[], object]) -> None:
            nonlocal failed
            if done:
                return
            t0, err = time.time(), ""
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - one bad run must not stop the others
                err, failed = f"{type(e).__name__}: {e}"[-3000:], failed + 1
            rec = {"run": run.name, "step": name, "ok": not err, "s": round(time.time() - t0, 1),
                   **({"error": err} if err else {})}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("[post]", json.dumps(rec), flush=True)

        for run in sorted(p for p in Path(a.runs_root).iterdir() if (p / "result.json").exists()):
            step(run, "perword", (run / "pw" / "index.json").exists(),
                 lambda: perword.convert(run, a.data_dir, epochs_only=True))
            step(run, "export_hf", (run / "hf" / "config.json").exists(),
                 lambda: sh(sys.executable, "-m", "babylm.harness.export_hf", "--ckpt", last_ckpt(run),
                            "--out", run / "hf", "--verify"))
            step(run, "dev_bpw", (run / "dev_bpw.json").exists() or not (run / "hf" / "config.json").exists(),
                 lambda: sh(sys.executable, REPO_ROOT / "babylm" / "eval" / "dev_bpw.py", run / "hf", dev_dir,
                            "--slice", "full", "--ws", FORMAT["ws"], "--line_join", FORMAT["line_join"],
                            "--out", run / "dev_bpw.json"))
            step(run, "plot", (run / "eval_curves.png").exists(),
                 lambda: sh(sys.executable, "-m", "babylm.harness.plot_curves", run))

            def pack_run() -> None:
                dst = pack / run.name
                for rel in SMALL:
                    if (run / rel).exists():
                        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(run / rel, dst / rel)
                if (run / "pw").is_dir():
                    shutil.copytree(run / "pw", dst / "pw", dirs_exist_ok=True)
                if run.name in a.pt_runs:
                    last = json.loads((run / "pw" / "index.json").read_text())["files"]
                    labels = [f[len("pw_dev_"):-4] for f in last if f.startswith("pw_dev_")]
                    lab = max(labels, key=lambda x: perword.label_words(x) or -1)
                    (dst / "pt").mkdir(parents=True, exist_ok=True)
                    for f in (f"pt_dev_{lab}.npy", "pt_devfull_final.npy"):
                        shutil.copy2(run / "pt" / f, dst / "pt" / f)
                    r = json.loads((run / "result.json").read_text())
                    per_epoch = r["words_seen"] / r["epochs"]  # also every DEV-sub snapshot from epoch 1 on (mixtures)
                    for f in (run / "pt").glob("pt_dev_*.npy"):
                        w = perword.label_words(f.name[len("pt_dev_"):-4])
                        if w is not None and w >= 0.98 * per_epoch:
                            shutil.copy2(f, dst / "pt" / f.name)
            step(run, "pack", False, pack_run)
    print(f"[post] done, {failed} failed step(s); pack at {pack}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
