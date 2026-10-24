"""Where a model's rare and never-seen vocabulary rows sit: for HF exports, the mean row norm of the tied embedding and
the mean capped logit, pre-softcap logit and log-probability on a fixed train_10M text, for three row sets: ids that
never occur in the train_10M stream, the byte tokens of the CHILDES conversation-analysis markup (⌈ ⌉ ⌊ ⌋ ↗ → ⁎ ∆ ∙;
rare but seen), and ids seen over 100 times. Built to compare packed (two per GPU under MPS) and solo runs.

  python -m babylm.instruments.rowprobe --tag base_r1 --data-root DATA s0=RUN/hf s3p=RUN/score/hf ...
Writes results/babylm/<tag>/rowprobe.csv (one row per model and row set).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from babylm.instruments import slices as S
from babylm.instruments.scorers import doc_ranges

MARKUP = "⌈⌉⌊⌋↗→⁎∆∙"
SOFTCAP = 15.0


def train_counts(tk, data_root: Path) -> np.ndarray:
    """Occurrences of each token id in the train_10M stream (documents tokenized on their own, as prepare.py)."""
    cnt = np.zeros(tk.get_vocab_size(), np.int64)
    for f in sorted((data_root / "train_10M" / "train_10M").glob("*.train")):
        lines = f.read_text(encoding="utf-8").split("\n")
        norm = [" ".join(x.split()) for x in lines]
        docs = ["\n".join(norm[a:b]) for a, b in doc_ranges(lines, f.stem)]
        for e in tk.encode_batch(docs, add_special_tokens=False):
            cnt += np.bincount(np.asarray(e.ids), minlength=len(cnt))
    return cnt


def main() -> None:
    import pandas as pd
    import torch
    from tokenizers import Tokenizer

    from babylm.instruments import score as SC

    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path of a harness HF export")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=32, help="1024-token rows of probe text (open_subtitles train)")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    first = Path(args.models[0].split("=", 1)[1])
    tk = Tokenizer.from_file(str(first / "tokenizer.json"))
    cnt = train_counts(tk, args.data_root)
    V = len(cnt)
    sets = {"never in train_10M": np.flatnonzero(cnt == 0)[1:] if cnt[0] == 0 else np.flatnonzero(cnt == 0),
            "CA markup bytes": np.array(sorted({i for c in MARKUP for i in tk.encode(c, add_special_tokens=False).ids})),
            "seen over 100 times": np.flatnonzero(cnt > 100)}
    lines = (args.data_root / "train_10M" / "train_10M" / "open_subtitles.train").read_text(encoding="utf-8").split("\n")
    ids = tk.encode("\n".join(" ".join(x.split()) for x in lines[:40000]), add_special_tokens=False).ids
    ids = ids[: args.rows * 1023]
    x = torch.cat([torch.zeros(args.rows, 1, dtype=torch.long), torch.tensor(ids).view(args.rows, 1023)], 1)
    rows = []
    for spec in args.models:
        name, path = spec.split("=", 1)
        model = SC.load_model(Path(path), torch.float32)[0].eval()
        E = model.state_dict()["embed.weight"][:V]
        with torch.inference_mode():
            lg = model(input_ids=x).logits.float()[..., :V]  # padded-vocab rows are left out
        pre = SOFTCAP * torch.atanh((lg / SOFTCAP).clamp(-1 + 1e-7, 1 - 1e-7))
        lse = torch.logsumexp(lg, -1, keepdim=True)
        for lab, idx in sets.items():
            t = torch.as_tensor(idx)
            rows.append({"model": name, "path": path, "row_set": lab, "rows": len(idx),
                         "row_norm": float(E[t].norm(dim=1).mean()), "logit": float(lg[..., t].mean()),
                         "precap_logit": float(pre[..., t].mean()), "log_p": float((lg[..., t] - lse).mean())})
        print(pd.DataFrame(rows[-3:]).drop(columns="path").round(3).to_string(index=False), flush=True)
        del model
    pd.DataFrame(rows).to_csv(out / "rowprobe.csv", index=False)


if __name__ == "__main__":
    main()
