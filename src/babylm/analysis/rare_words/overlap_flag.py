"""Flag sample words whose dev line appears verbatim (whitespace-collapsed) in train_100M, as the instruments'
bottleneck protocol drops them (the 100M model may have memorised the line). Writes line_in_100M.npy (bool per word)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rare_words as RW  # noqa: E402

OUT, RAW = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in (OUT / "sample.jsonl").read_text().splitlines()]


def h(line: str) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(line.split()).encode("utf-8"), digest_size=8).digest(), "little")


dev_lines = {}
for r in rows:
    for ln in r["text"].split("\n"):
        c = " ".join(ln.split())
        if c:
            dev_lines[h(c)] = False
n_short = 0
for s in RW.SOURCES:
    with open(RAW / "train_100M" / f"{s}.train", encoding="utf-8") as f:
        for ln in f:
            if RW.is_header(ln):
                continue
            k = h(ln)
            if k in dev_lines:
                dev_lines[k] = True
flags = []
for r in rows:
    for ln in r["text"].split("\n"):
        c = " ".join(ln.split())
        if not c:
            continue
        found = dev_lines[h(c)]
        flags.extend([found] * len(c.split()))
flags = np.array(flags, dtype=bool)
np.save(OUT / "line_in_100M.npy", flags)
print(f"words {len(flags):,}; in a train_100M line: {flags.mean():.2%}; lines found {sum(dev_lines.values()):,} of {len(dev_lines):,}")
