"""Score GPT-BERT 10M and 100M (published causal-focus finals) on a seeded dev subsample and slice the gap by
rare-word type. Reuses the instruments' loader/scorer (score.py, vendored GPT-BERT, fast attention).

Usage: python3 -I score_rare.py <raw_root> <ckpt_dir> <out_dir> --words 120000 [--threads 4] [--only-counts]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "instr"))
sys.path.insert(0, str(HERE))
import rare_words as RW  # noqa: E402
from babylm.instruments import score as SC  # noqa: E402

SOURCES = RW.SOURCES
MAX_DOC_WORDS = 2500
MAX_PIECES_PER_DOC = 2  # more documents, not longer ones: the models see <=512 tokens of context anyway


def build_rows(root: Path, target_words: int, seed: int) -> list[dict]:
    """Whole dev documents per source (seeded permutation), word total per source ~ its dev share."""
    rng = np.random.default_rng(seed)
    per_source = {}
    totals = {}
    for s in SOURCES:
        lines = (root / "dev" / f"{s}.dev").read_text(encoding="utf-8").split("\n")
        dl = [(did, [ln for ln in d if not RW.is_header(ln)]) for did, d in RW.docs(lines, s)]
        dl = [(did, d) for did, d in dl if d]
        per_source[s] = dl
        totals[s] = sum(len(ln.split()) for _, d in dl for ln in d)
    T = sum(totals.values())
    rows = []
    for s in SOURCES:
        quota = target_words * totals[s] / T
        order = rng.permutation(len(per_source[s]))
        got = 0
        for k in order:
            if got >= quota:
                break
            did, d = per_source[s][k]
            # cut long documents into pieces of <= MAX_DOC_WORDS words at line boundaries (pieces keep the doc id)
            piece, pw, pi = [], 0, 0
            for ln in d:
                n = len(ln.split())
                if piece and pw + n > MAX_DOC_WORDS:
                    rows.append({"id": f"{s}/{did}/{pi}", "source": s, "doc": f"{s}/{did}", "text": "\n".join(piece)})
                    got += pw
                    piece, pw, pi = [], 0, pi + 1
                    if pi >= MAX_PIECES_PER_DOC:
                        break
                piece.append(ln)
                pw += n
            if piece and pi < MAX_PIECES_PER_DOC:
                rows.append({"id": f"{s}/{did}/{pi}", "source": s, "doc": f"{s}/{did}", "text": "\n".join(piece)})
                got += pw
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--words", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--models", default="babylm-baseline-10m-gpt-bert-causal-focus,babylm-baseline-100m-gpt-bert-causal-focus")
    ap.add_argument("--only-counts", action="store_true")
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.raw_root, args.words, args.seed)
    n_words = sum(len(r["text"].split()) for r in rows)
    print(f"sample: {len(rows)} pieces, {n_words:,} words", flush=True)
    (args.out / "sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # word features (same text the models see: collapse keeps words unchanged)
    print("counting train_10M / train_100M normal forms", flush=True)
    c10 = RW.count_split(args.raw_root, "train_10M", "train")
    c100 = RW.count_split(args.raw_root, "train_100M", "train")
    feats = []
    for r in rows:  # repeat flags reset at each piece: a piece is one scoring chunk (<=512-token windows inside it)
        seen_doc: set[str] = set()
        recent = collections.deque(maxlen=300)
        rc: collections.Counter = collections.Counter()
        for w in r["text"].split():
            n = RW.norm(w)
            k10, k100 = c10.get(n, 0), c100.get(n, 0)
            cat, det = RW.classify(w, n, c10, c100, None) if k10 == 0 else ("", "")
            feats.append((r["source"], r["doc"], w, n, k10, k100, cat, int(n in seen_doc), int(rc[n] > 0)))
            seen_doc.add(n)
            if len(recent) == recent.maxlen:
                old = recent[0]
                rc[old] -= 1
                if rc[old] <= 0:
                    del rc[old]
            recent.append(n)
            rc[n] += 1
    assert len(feats) == n_words
    with open(args.out / "words.tsv", "w", encoding="utf-8") as f:
        f.write("source\tdoc\tsurface\tnorm\tc10\tc100\tcat\trepeat_piece\trepeat300\n")
        for t in feats:
            f.write("\t".join(str(x) for x in t) + "\n")
    if args.only_counts:
        return
    for name in args.models.split(","):
        path = args.ckpt_dir / name
        t0 = time.time()
        model, tok, family, n_params = SC.load_model(path, torch.float32)
        print(f"{name}: {family} {n_params / 1e6:.1f}M params loaded in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        res = SC.score_split(model, tok, rows, family, max_len=512, stride=256, batch_tokens=8192, line_join="newline", collapse=True)
        dt = time.time() - t0
        assert len(res["word_nll"]) == n_words, (len(res["word_nll"]), n_words)
        print(f"{name}: scored {len(res['tok_nll']):,} tokens in {dt:.0f}s ({len(res['tok_nll']) / dt:.0f} tok/s); "
              f"nats/word {res['word_nll'].sum() / n_words:.4f}", flush=True)
        np.savez_compressed(args.out / f"{name}.npz", word_nll=res["word_nll"], word_ntok=res["word_ntok"],
                            tok_nll=res["tok_nll"], tok_word=res["tok_word"], tok_ent=res["tok_ent"])


if __name__ == "__main__":
    main()
