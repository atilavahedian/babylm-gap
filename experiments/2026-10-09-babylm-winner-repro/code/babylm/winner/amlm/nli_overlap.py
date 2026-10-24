"""Exact-sentence overlap between a pretraining text file and GLUE MNLI/RTE splits (nyu-mll/glue parquet).
usage: python nli_overlap.py TRAIN_TXT GLUE_DIR   (GLUE_DIR holds <task>_<split>-00000-of-00001.parquet)
Lines/sentences under 4 words are ignored; matching is whitespace- and case-normalised."""
import re, sys, pandas as pd
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
train = set(norm(l) for l in open(sys.argv[1], errors="replace") if len(l.split()) >= 4)
print("distinct train lines (>=4 words):", len(train))
for task, split, cols in [("mnli", "train", ("premise", "hypothesis")), ("mnli", "validation_matched", ("premise", "hypothesis")),
                          ("mnli", "validation_mismatched", ("premise", "hypothesis")), ("mnli", "test_matched", ("premise", "hypothesis")),
                          ("rte", "train", ("sentence1", "sentence2")), ("rte", "validation", ("sentence1", "sentence2"))]:
    df = pd.read_parquet(f"{sys.argv[2]}/{task}_{split}-00000-of-00001.parquet")
    parts = []
    for c in cols:
        xs = list({norm(x) for x in df[c] if len(x.split()) >= 4})
        hit = sum(x in train for x in xs); parts.append(f"{c} {hit}/{len(xs)} = {hit/len(xs):.1%}")
    both = sum(norm(p) in train and norm(h) in train for p, h in zip(df[cols[0]], df[cols[1]]))
    print(f"{task}_{split}: " + "; ".join(parts) + f"; pair both present {both}/{len(df)} = {both/len(df):.1%}")
