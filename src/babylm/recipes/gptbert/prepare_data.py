"""Tokenise the official BabyLM 2025 corpora for the GPT-BERT port.

Follows gpt-bert/corpus_tokenization/tokenize_corpus.py (@ref-0493): every *document* is `text.strip()`, encoded with the
published tokenizer.json with add_special_tokens=False, stored as an int16 tensor; `<split>_tokenized.bin` is
`torch.save(list_of_document_tensors)` - exactly what official/dataset.py torch.load()s.

The official repo starts from a .jsonl with one document per line and does not ship the script that built it from the
BabyLM files. Our document rule (deterministic):
  * files are read in sorted order (bnc_spoken, childes, gutenberg, open_subtitles, simple_wiki, switchboard);
  * whitespace inside a line is collapsed: line -> " ".join(line.split()) (tabs / runs of spaces become one space);
  * a document is a maximal run of non-blank lines joined with "\\n"; blank lines separate documents.
Evidence that the published 2025 baseline saw text prepared this way (tests on the published checkpoint):
  - causal loss on the TAB after CHILDES/Switchboard speaker labels ("*MOT:\\t") is 13-23 nats (never seen in
    training) while "\\n" costs ~0 nats: lines were kept, tabs were not; with tab -> space the same CHILDES snippet costs
    1.9 nats/token instead of 5.0;
  - a double space after a Gutenberg sentence costs ~7 nats on the lone-space token (never seen);
  - with this rule train_10M has 16,242,414 tokens, and the card's 9,914 steps x 16,384 tokens = 10 x 16,243,098:
    the published step count is exactly 10 epochs of this token count (raw text gives 16,971,177 tokens).
(--whitespace raw keeps lines byte-exact.)

Outputs per split in --out_dir:  <split>_tokenized.bin, <split>_docs.npy (int64 [n_docs, 3]: file index, first line
number in that file (0-based, counting blank lines), number of lines), <split>_manifest.json (tokenizer sha256, per-file
words/tokens/sha256, content hash of the token stream). For train_10M also the fixed train sample (see make_train_sample).

Usage:  python prepare_data.py --data_root /path/to/text_data [--splits train_10M,dev[,train_100M]]
"""
import argparse, glob, hashlib, json, os, random
import numpy as np
import torch
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER = os.path.join(HERE, "hf_files", "tokenizer.json")
SPLIT_GLOBS = {"train_10M": "train_10M/train_10M/*.train", "train_100M": "train_100M/train_100M/*.train", "dev": "dev/dev/*.dev"}
SAMPLE_SEED = 20251009          # constant, independent of any run seed
SAMPLE_WORDS = 1_000_000
SAMPLE_CHUNK = 127              # tokens per scored chunk (= dump context 128 incl. <s>), same chunking as the dev dump


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_documents(path, whitespace="collapse"):
    """Returns [(text, first_line_no, n_lines)] for one file (line numbers 0-based, blank lines counted)."""
    docs, cur, start = [], [], None
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if whitespace == "collapse":
                line = " ".join(line.split())
            if line.strip() == "":
                if cur:
                    docs.append(("\n".join(cur), start, len(cur)))
                    cur = []
            else:
                if not cur:
                    start = i
                cur.append(line)
    if cur:
        docs.append(("\n".join(cur), start, len(cur)))
    return docs


def newline_table(tokenizer):
    """bool[vocab]: token ends a line (its string ends with the byte-level newline 'Ċ', e.g. 'Ċ' or '."Ċ')."""
    return np.array([tokenizer.id_to_token(i).endswith("Ċ") for i in range(tokenizer.get_vocab_size())])


def word_start_table(tokenizer):
    """bool[vocab]: token starts a whitespace word (string starts with 'Ġ'). Verified: the count over train_10M and dev
    equals the number of str.split() words exactly."""
    return np.array([tokenizer.id_to_token(i).startswith("Ġ") for i in range(tokenizer.get_vocab_size())])


def tokenize_split(tokenizer, files, whitespace):
    out, table, stats, n_words, n_tok = [], [], {}, 0, 0
    h = hashlib.sha256()
    nl = newline_table(tokenizer)
    for fi, path in enumerate(files):
        docs = read_documents(path, whitespace)
        w = sum(len(d.split()) for d, _, _ in docs)
        enc = tokenizer.encode_batch([d.strip() for d, _, _ in docs], add_special_tokens=False)
        t = 0
        for (_, start, n_lines), e in zip(docs, enc):
            ids = np.asarray(e.ids, dtype=np.int64)
            assert int(nl[ids].sum()) == n_lines - 1, (path, start)  # token -> line mapping is exact
            ten = torch.tensor(e.ids, dtype=torch.int16)  # official: int16 (vocab 16,384 < 32,768)
            out.append(ten)
            table.append((fi, start, n_lines))
            h.update(ten.numpy().tobytes()); h.update(b"|")
            t += len(e.ids)
        stats[os.path.basename(path)] = {"documents": len(docs), "words": w, "tokens": t, "file_sha256": sha256_file(path)}
        n_words += w; n_tok += t
    return out, np.array(table, dtype=np.int64), stats, n_words, n_tok, h.hexdigest()


def make_train_sample(docs, tokenizer, table, out_dir, files):
    """Fixed ~1M-word sample of train_10M for per-token dumps: all docs cut into non-overlapping SAMPLE_CHUNK-token chunks
    (like the dev dump), shuffled with SAMPLE_SEED, first chunks until >= SAMPLE_WORDS whitespace words (word = token
    starting with 'G-dot'). Saves chunk table (doc, offset, length) and the covered (file, line) indices."""
    ws, nl = word_start_table(tokenizer), newline_table(tokenizer)
    chunks = [(d, o, min(SAMPLE_CHUNK, len(x) - o)) for d, x in enumerate(docs) for o in range(0, len(x), SAMPLE_CHUNK) if len(x) - o >= 2]
    random.Random(SAMPLE_SEED).shuffle(chunks)
    sel, words, lines, cache = [], 0, set(), {}
    for d, o, L in chunks:
        if words >= SAMPLE_WORDS:
            break
        ids = docs[d].numpy().astype(np.int64)
        words += int(ws[ids[o:o + L]].sum())
        if d not in cache:  # line index of every token; a newline token belongs to the line it ends
            cache[d] = np.concatenate([[0], np.cumsum(nl[ids])[:-1]])
        line_in_doc = cache[d]
        fi, start, _ = table[d]
        for li in np.unique(line_in_doc[o:o + L]):
            lines.add((int(fi), int(start + li)))
        sel.append((d, o, L))
    sel = np.array(sel, dtype=np.int64)
    lines = np.array(sorted(lines), dtype=np.int64)
    np.save(os.path.join(out_dir, "train_10M_sample1M_chunks.npy"), sel)
    np.save(os.path.join(out_dir, "train_10M_sample1M_lines.npy"), lines)
    info = {"seed": SAMPLE_SEED, "chunk_tokens": SAMPLE_CHUNK, "chunks": len(sel), "words": words, "tokens": int(sel[:, 2].sum()),
            "lines_touched": len(lines), "files": [os.path.basename(f) for f in files],
            "chunks_sha256": hashlib.sha256(sel.tobytes()).hexdigest()}
    print(f"train sample: {info}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="~/babylm/osf/text_data", help="official OSF text_data folder")
    ap.add_argument("--splits", default="train_10M,dev", help=f"comma list of {list(SPLIT_GLOBS)} (never test)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--whitespace", choices=["collapse", "raw"], default="collapse")
    args = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    for name in args.splits.split(","):
        assert name in SPLIT_GLOBS, f"unknown split {name}"
        files = sorted(glob.glob(os.path.join(args.data_root, SPLIT_GLOBS[name])))
        assert files, f"no files match {SPLIT_GLOBS[name]}"
        docs, table, stats, n_words, n_tok, content_hash = tokenize_split(tok, files, args.whitespace)
        path = os.path.join(args.out_dir, f"{name}_tokenized.bin")
        torch.save(docs, path)
        np.save(os.path.join(args.out_dir, f"{name}_docs.npy"), table)
        man = {"split": name, "path": os.path.abspath(path), "tokenizer": os.path.abspath(args.tokenizer),
               "tokenizer_sha256": sha256_file(args.tokenizer),
               "document_rule": "runs of non-blank lines joined with \\n; blank lines separate documents; files in sorted order",
               "whitespace": args.whitespace + (" (line -> ' '.join(line.split()))" if args.whitespace == "collapse" else ""),
               "encoding": "tokenizer.encode(text.strip(), add_special_tokens=False) -> int16 (corpus_tokenization/tokenize_corpus.py)",
               "file_order": [os.path.basename(f) for f in files], "files": stats, "documents": len(docs),
               "words_whitespace": n_words, "tokens": n_tok, "tokens_per_word": n_tok / n_words, "token_stream_sha256": content_hash}
        if name == "train_10M":
            man["train_sample"] = make_train_sample(docs, tok, table, args.out_dir, files)
        with open(os.path.join(args.out_dir, f"{name}_manifest.json"), "w") as f:
            json.dump(man, f, indent=2)
        print(f"{name}: {len(docs)} docs, {n_words:,} words, {n_tok:,} tokens ({n_tok / n_words:.4f} tok/word) -> {path}")


if __name__ == "__main__":
    main()
