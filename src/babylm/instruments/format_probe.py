"""Infer a checkpoint's training text format from its loss on TRAIN text only.

Held-out loss on BabyLM text swings by up to 4 nats/word with formatting the model
never saw (a space after each newline; tabs after CHILDES/Switchboard speaker labels).
This probe scores a fixed subset of the train sample (never dev) under each candidate
format and prints the arguments score.py should use:
  --line-join  newline | newline_space   (chosen on all sources)
  --tab-space  sources whose tabs become spaces (chosen per source)
  --collapse   every line rewritten as ' '.join(line.split()) (chosen on all sources, against the
               line join and tab handling chosen above; critic ref-0056 candidate set)
Where a model's training code fixes the format, the code wins and a disagreeing probe is flagged.

Usage:
  python -m babylm.instruments.format_probe --repo BabyLM-community/babylm-baseline-10m-gpt2 --revision main
  python -m babylm.instruments.format_probe --local-path DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from babylm.instruments import score as S

TAB_SOURCES = ("childes", "switchboard")


def probe(model, tok, family: str, rows: list[dict], per_source: int = 6, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    sub = {s: [rs[i] for i in rng.choice(len(rs), min(per_source, len(rs)), replace=False)] for s, rs in by.items()}
    allrows = [r for rs in sub.values() for r in rs]
    res = {"line_join": {}, "tab": {}}
    for lj in sorted(S.LINE_JOINS):
        res["line_join"][lj] = float(S.score_split(model, tok, allrows, family, line_join=lj, tab_space=TAB_SOURCES)["word_nll"].mean())
    best_lj = min(res["line_join"], key=res["line_join"].get)
    tab_space = []
    for src in TAB_SOURCES:
        keep = float(S.score_split(model, tok, sub[src], family, line_join=best_lj)["word_nll"].mean())
        space = float(S.score_split(model, tok, sub[src], family, line_join=best_lj, tab_space=(src,))["word_nll"].mean())
        res["tab"][src] = {"keep": keep, "space": space}
        if space < keep:
            tab_space.append(src)
    base = float(S.score_split(model, tok, allrows, family, line_join=best_lj, tab_space=tuple(tab_space))["word_nll"].mean())
    col = float(S.score_split(model, tok, allrows, family, line_join=best_lj, collapse=True)["word_nll"].mean())
    res["collapse"] = {"no": base, "yes": col}
    res["args"] = {"line_join": best_lj, "tab_space": ",".join(tab_space), "collapse": col < base}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--local-path", type=Path)
    ap.add_argument("--cache", type=Path, default=S.REPO / "data" / "babylm_ckpt_tmp_probe")
    args = ap.parse_args()
    if args.local_path:
        path = args.local_path
    else:
        path, _ = S.fetch(args.repo, args.revision, args.cache / args.repo.split("/")[-1] / args.revision)
    model, tok, family, _ = S.load_model(path, torch.bfloat16)
    res = probe(model, tok, family, S.load_rows("train"))
    print(json.dumps(res, indent=1))
    if not args.local_path:
        shutil.rmtree(args.cache / args.repo.split("/")[-1] / args.revision, ignore_errors=True)


if __name__ == "__main__":
    main()
