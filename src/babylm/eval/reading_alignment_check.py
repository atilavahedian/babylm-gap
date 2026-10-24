"""Reading-task target alignment check (critic 81702719, condition i) for the amended Reading column.

    python reading_alignment_check.py HF_TOKENIZER_DIR READING_DATA_CSV [--out report.json]

The official reading task (evaluation-pipeline-2025 @ bf55c1131e53654c2a87418f5629c26959acd710, reading/evaluation_functions.py:get_p2) scores a target
word as: first piece = tokenizer(word)[0]; every further piece t of tokenizer(word) is re-encoded as
tokenizer(decode(t))[0] (via get_p). For each stimulus (word given item, and prev_word given prev_item) this compares,
for the export tokenizer as is (add_prefix_space=False) and for a copy with add_prefix_space=True:
  critic  : tokenizer(word, add_special_tokens=False) ids == the word's ids inside the whole-sentence encoding
  pipeline: the ids get_p2 actually scores (above)         == the word's ids inside the whole-sentence encoding
The whole-sentence reference is the AS-IS tokenizer on item + " " + word (+ the rest of `sentence` when it continues it),
taking the tokens whose character offsets fall in the word's span (the leading space belongs to the word's first token).
Verdict per critic: STOP if more than 1% of targets mismatch.
"""
import argparse
import csv
import json

from transformers import AutoTokenizer


def in_sentence_ids(tok, item, word, sentence):
    text = item + " " + word
    if sentence.startswith(text):
        text = sentence
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    lo, hi = len(item), len(item) + 1 + len(word)
    ids = [i for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]) if a < hi and b > lo]
    return ids


def pipeline_ids(tok, word):
    t = tok(word, add_special_tokens=False)["input_ids"]
    return t[:1] + [tok(tok.decode(x), add_special_tokens=False)["input_ids"][0] for x in t[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tok_dir")
    ap.add_argument("data")
    ap.add_argument("--out")
    a = ap.parse_args()
    base = AutoTokenizer.from_pretrained(a.tok_dir, add_prefix_space=False)
    ps = AutoTokenizer.from_pretrained(a.tok_dir, add_prefix_space=True)
    rows = list(csv.DictReader(open(a.data)))
    targets = []
    for r in rows:
        item = r["item"] if r["item"] else "None"   # run.py: df["item"].fillna("None")
        targets.append(("word", r[""], item, r["word"], r["sentence"]))
        if r.get("prev_item"):
            targets.append(("prev_word", r[""], r["prev_item"], r["prev_word"], r["sentence"]))
    res = {}
    for name, tk in (("as_is", base), ("prefix_space", ps)):
        crit, pipe, multi = [], [], 0
        for kind, idx, item, word, sent in targets:
            ref = in_sentence_ids(base, item, word, sent)
            c = tk(word, add_special_tokens=False)["input_ids"]
            p = pipeline_ids(tk, word)
            multi += len(c) > 1
            if c != ref:
                crit.append({"row": idx, "kind": kind, "word": word, "ref": base.convert_ids_to_tokens(ref), "got": tk.convert_ids_to_tokens(c)})
            if p != ref:
                pipe.append({"row": idx, "kind": kind, "word": word, "ref": base.convert_ids_to_tokens(ref), "got": tk.convert_ids_to_tokens(p)})
        n = len(targets)
        res[name] = {"targets": n, "multi_piece": multi,
                     "critic_mismatch": len(crit), "critic_mismatch_pct": 100 * len(crit) / n,
                     "pipeline_mismatch": len(pipe), "pipeline_mismatch_pct": 100 * len(pipe) / n,
                     "critic_examples": crit[:40], "pipeline_examples": pipe[:40]}
        print(f"{name}: {n} targets ({multi} multi-piece); critic-definition mismatch {len(crit)} "
              f"({100 * len(crit) / n:.2f}%); pipeline-effective mismatch {len(pipe)} ({100 * len(pipe) / n:.2f}%)")
    v = res["prefix_space"]["pipeline_mismatch_pct"]
    res["verdict"] = "PROCEED" if v <= 1.0 else "STOP"
    print(f"verdict (prefix_space, pipeline-effective, threshold 1%): {res['verdict']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
