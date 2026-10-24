"""Share per-token and per-word dev-sample scores with other threads (no rescoring): one npz of per-token arrays per
model, one gzipped CSV of dev words with their slice features and each model's nats, and train_10M word counts.

  python -m babylm.instruments.export_dumps --out shared/babylm/preread_v1/dumps --data-root ~/babylm_data \
      gptbert_cf_10m_collapse/chck_100M gptbert_cf_100m_collapse/chck_1000M
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from babylm.instruments import corpus as C
from babylm.instruments import slices as S


def main() -> None:
    import pandas as pd

    from babylm.instruments.rare_split import tokenizer_for, train_counts
    from babylm.instruments.score import load_rows, tokenize_chunks

    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="run/rev with dev scores")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows("dev")
    F = S.load_feats("dev")
    words = [w for r in rows for w in r["text"].split()]
    t = pd.DataFrame({"chunk": F["chunk"], "document": F["unit"], "source": [C.SOURCES[i] for i in F["source"].astype(int)],
                      "position_in_chunk": F["widx"], "word": words, "normal_form": [C.norm(w) for w in words],
                      "train_10M_count": F["freq10"], "train_100M_count": F["freq100"],
                      "first_in_chunk": (S.dims(F)["context"][0] == 0), "line_in_train_100M": F["in_train_100M"]})
    for m in args.models:
        run, rev = m.split("/")
        sc = S.load_score(run, rev, "dev")
        tok, sha, fmt = tokenizer_for(run, rev)
        ids = np.concatenate([np.asarray(c[0][1:]) for c in tokenize_chunks(tok, rows, *fmt)])
        assert len(ids) == len(sc["tok_nll"]), (m, len(ids), len(sc["tok_nll"]))
        np.savez_compressed(args.out / f"tokens_{run}_{rev}.npz", tok_nll=sc["tok_nll"].astype(np.float32),
                            tok_word=sc["tok_word"], tok_chunk=sc["tok_chunk"], tok_pos=sc["tok_pos"],
                            tok_id=ids.astype(np.int32), tok_train_10M_count=train_counts(tok, sha, fmt, args.data_root)[ids])
        (args.out / f"tokens_{run}_{rev}.json").write_text(json.dumps(
            {"tokenizer_sha256_16": sha, "line_join": fmt[0], "tab_space": list(fmt[1]), "collapse": fmt[2],
             "meta": json.loads((S.SCORES / run / rev / "meta.json").read_text())}, indent=1))
        t[f"nats_{run}_{rev}"] = sc["word_nll"].astype(np.float32)
    t.to_csv(args.out / "words_dev_sample_v1.csv.gz", index_label="word_index")
    cnt = Counter()
    for path in C.split_files(args.data_root, "train_10M"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not C.is_header(line):
                    cnt.update(C.norm(w) for w in line.split())
    pd.DataFrame(cnt.most_common(), columns=["normal_form", "count"]).to_csv(args.out / "train_10M_word_counts.csv.gz", index=False)
    print(f"wrote {args.out}: {len(t):,} words, {len(cnt):,} train_10M types")


if __name__ == "__main__":
    main()
