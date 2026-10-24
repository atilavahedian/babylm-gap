"""Registered DEV metric vs its blank-line twin (babylm/eval/dev_bpw_twin.py) on whole sampled DEV documents, per
model and source, and the twin's full-DEV value estimated as the model's registered full-DEV score plus the sampled
difference pooled with each file's share of DEV words (switchboard has no blank line: difference 0).

  python -m babylm.instruments.twin_sample --tag base_r1 --data-root DATA [--frac 0.02] s0=RUN/hf s1=RUN_s1/hf ...
Registered full-DEV scores are read from each export's run dir (RUN/score.json dev_nats_per_word) when present.
Writes results/babylm/<tag>/twin_sample.csv (per source) and twin_sample_models.csv (pooled, with the estimate).
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np

from babylm.instruments import slices as S
from babylm.instruments.scorers import run_jobs, sliding_jobs, texts


def main() -> None:
    import pandas as pd
    import torch

    from babylm.instruments import score as SC

    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path of a harness HF export")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.02, help="share of each dev file's words (whole documents)")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((args.data_root / "dev" / "dev").glob("*.dev"))
    total = {f.stem: texts(f, 1.0)[3] for f in files}
    data = [(f.stem, *texts(f, args.frac)) for f in files if f.stem != "switchboard"]
    rows, models = [], []
    for spec in args.models:
        name, path = spec.split("=", 1)
        model, tok, _, _ = SC.load_model(Path(path), torch.float32)
        model.eval()
        bos = tok.bos_token_id
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        for src, reg_text, docs, words, file_words in data:
            reg = enc(reg_text)
            twin = [t for d in docs for t in [bos] + enc(d)][1:]  # the window's BOS stands in for the first document's
            n = {k: float(np.nansum(run_jobs(model, sliding_jobs(ids, bos), len(ids), contextlib.nullcontext)))
                 for k, ids in (("registered", reg), ("twin", twin))}
            rows.append({"model": name, "source": src, "words": words, "file_words": file_words, "docs": len(docs),
                         "separators": len(docs) - 1, "registered": n["registered"] / words, "twin": n["twin"] / words})
            print(rows[-1], flush=True)
        sc = Path(path).parent / "score.json"
        reg_full = json.loads(sc.read_text()).get("dev_nats_per_word") if sc.exists() else None
        models.append({"model": name, "path": path, "registered_full": reg_full})
        del model
    t = pd.DataFrame(rows)
    t["diff"] = t.twin - t.registered
    t["dev_share"] = t.file_words / sum(total.values())
    t.to_csv(out / "twin_sample.csv", index=False)
    m = pd.DataFrame(models).set_index("model")
    m["twin_minus_registered"] = (t["diff"] * t.dev_share).groupby(t.model).sum()
    m["twin_full_estimate"] = m.registered_full + m.twin_minus_registered
    m.to_csv(out / "twin_sample_models.csv")
    pd.set_option("display.width", 200)
    print(t.drop(columns=["file_words"]).round(4).to_string(index=False))
    print(m.drop(columns=["path"]).round(4).to_string())
    if len(m) > 1:
        print("SD across models:\n" + m[["registered_full", "twin_full_estimate"]].std().round(4).to_string())


if __name__ == "__main__":
    main()
