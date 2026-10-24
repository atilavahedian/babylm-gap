"""Lexicon-trie oracle: how many nats would a word-completion prior over the train_10M lexicon save when mixed with
the model, position by position? Mixture is linear in the target probability, so it is exact from the model's
target probability + its mass on word-initial tokens (from the calib pass).

P_trie(next token t | tokens of the current word so far P) = count(words with token prefix P+t) / count(prefix P)
P_trie(word ends | P) = count(words whose tokenization is exactly P) / count(prefix P)
p_mix(cont target t) = lam * P_trie(t|P) + (1-lam) * p_model(t)
p_mix(word-initial target t) = p_model(t) * (lam * P_trie(end|P) / init_mass + (1-lam))
Positions whose prefix P is not in the lexicon use the model alone.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "instr"))
sys.path.insert(0, str(HERE))
import rare_words as RW  # noqa: E402
from babylm.instruments import score as SC  # noqa: E402

RAW, CKPT, OUT, NAME = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
DROP = np.load(sys.argv[5]) if len(sys.argv) > 5 else None
TAG = "_filtered" if DROP is not None else ""
LAMS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def load_tok(path: Path):
    from transformers import PreTrainedTokenizerFast

    special = {}
    for k, v in json.loads((path / "special_tokens_map.json").read_text()).items():
        if isinstance(v, (str, dict)):
            special[k] = v["content"] if isinstance(v, dict) else v
    return PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"), **special)


tok = load_tok(CKPT / NAME)
# lexicon: surface whitespace words of train_10M (collapsed text), case kept
lex = collections.Counter()
for s in RW.SOURCES:
    for ln in (RAW / "train_10M" / f"{s}.train").read_text(encoding="utf-8").split("\n"):
        if not RW.is_header(ln):
            lex.update(ln.split())
print(f"lexicon types {len(lex):,}, tokens {sum(lex.values()):,}", flush=True)
types = list(lex.keys())
prefix = collections.Counter()
exact = collections.Counter()
# tokenize each type as a mid-line word: encode "the <word>" and drop the tokens of "the" (GPT-BERT's normalizer
# prepends its own space, the harness tokenizer does not; this path matches in-context tokenization for both)
lead = tuple(tok("the", add_special_tokens=False)["input_ids"])
B = 5000
n_bad = 0
for i in range(0, len(types), B):
    enc = tok(["the " + t for t in types[i:i + B]], add_special_tokens=False)["input_ids"]
    for t, ids in zip(types[i:i + B], enc):
        ids = tuple(ids)
        if ids[:len(experimenter)] != lead:
            n_bad += 1
            continue
        ids = ids[len(experimenter):]
        c = lex[t]
        exact[ids] += c
        for k in range(1, len(ids) + 1):
            prefix[ids[:k]] += c
print(f"trie: {len(prefix):,} prefixes; lead tokens {lead}; {n_bad} types skipped", flush=True)

rows = [json.loads(l) for l in (OUT / "sample.jsonl").read_text().splitlines()]
chunks = SC.tokenize_chunks(tok, rows, "newline", (), True)
cal = np.load(OUT / f"{NAME}.calib.npz")
tok_nll = cal["tok_nll"].astype(np.float64)
init_mass = cal["init_mass"].astype(np.float64)
tok_init = cal["tok_init"].astype(bool)
p_model = np.exp(-tok_nll)
N = len(tok_nll)
p_trie = np.zeros(N)       # P_trie(target | P) for continuation targets, P_trie(end | P) for word-initial targets
covered = np.zeros(N, bool)
pos = 0
for ids, widx, _ in chunks:
    targets = ids[1:]
    cur = []  # tokens of the current word so far
    for j, t in enumerate(targets):
        if j > 0 and cur:
            P = tuple(cur)
            if P in prefix:
                covered[pos] = True
                if tok_init[pos]:
                    p_trie[pos] = exact.get(P, 0) / prefix[P]
                else:
                    p_trie[pos] = prefix.get(P + (t,), 0) / prefix[P]
        if tok_init[pos] or j == 0:
            cur = [t]
        else:
            cur.append(t)
        pos += 1
assert pos == N
print(f"coverage: init targets {covered[tok_init].mean():.3f}, continuation targets {covered[~tok_init].mean():.3f}; "
      f"mean p_trie(end) at covered init {p_trie[covered & tok_init].mean():.3f} vs init_mass {init_mass[covered & tok_init].mean():.3f}; "
      f"mean p_trie(cont) at covered cont {p_trie[covered & ~tok_init].mean():.3f} vs p_model {p_model[covered & ~tok_init].mean():.3f}", flush=True)
# per-word bands from words.tsv
lines = (OUT / "words.tsv").read_text(encoding="utf-8").split("\n")
hdr = lines[0].split("\t")
c10 = np.array([int(l.split("\t")[hdr.index("c10")]) for l in lines[1:] if l])
tw = cal["tok_word"].astype(np.int64)
wb = np.where(c10 == 0, 0, np.where(c10 <= 63, 1, np.where(c10 < 8192, 2, 3)))
tb = wb[tw]
keep_w = np.ones(len(c10), bool) if DROP is None else ~DROP
kt = keep_w[tw]
tok_nll_full = tok_nll.copy()
n_words = int(keep_w.sum())
res = {"covered_share": float(covered[kt].mean()), "n_words": n_words, "lams": {}}
for lam in LAMS:
    pm = np.where(~covered, p_model,
                  np.where(tok_init, p_model * (lam * p_trie / np.maximum(init_mass, 1e-9) + (1 - lam)),
                           lam * p_trie + (1 - lam) * p_model))
    nll = -np.log(np.maximum(pm, 1e-12))
    d = {"nats_per_word": float(nll[kt].sum() / n_words), "gain_nats_per_word": float((tok_nll - nll)[kt].sum() / n_words)}
    for k, lab in enumerate(["0", "1-63", "64-8k", "8k+"]):
        m = (tb == k) & kt
        d[f"gain_band_{lab}_overall"] = float((tok_nll[m] - nll[m]).sum() / n_words)
        d[f"gain_band_{lab}_per_word"] = float((tok_nll[m] - nll[m]).sum() / max(1, ((wb == k) & keep_w).sum()))
    d["gain_cont_positions_overall"] = float((tok_nll - nll)[kt & ~tok_init].sum() / n_words)
    d["gain_init_positions_overall"] = float((tok_nll - nll)[kt & tok_init].sum() / n_words)
    res["lams"][str(lam)] = d
# per-band best lambda (oracle upper bound, 4 params)
best = {}
for k, lab in enumerate(["0", "1-63", "64-8k", "8k+"]):
    m = (tb == k) & kt
    bl, bg = 0.0, 0.0
    for lam in np.linspace(0, 0.9, 46):
        pm = np.where(~covered, p_model,
                      np.where(tok_init, p_model * (lam * p_trie / np.maximum(init_mass, 1e-9) + (1 - lam)),
                               lam * p_trie + (1 - lam) * p_model))
        g = float((tok_nll[m] + np.log(np.maximum(pm[m], 1e-12))).sum() / n_words)
        if g > bg:
            bl, bg = float(lam), g
    best[lab] = {"lam": bl, "gain_overall": bg}
res["per_band_best"] = best
res["per_band_best_total_overall"] = float(sum(v["gain_overall"] for v in best.values()))
(OUT / f"{NAME}.trie{TAG}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
