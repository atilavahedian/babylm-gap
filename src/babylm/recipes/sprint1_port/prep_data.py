"""Tokenise BabyLM 2025 strict-small (train_10M) and the official dev split with GPT-2 BPE for the sprint-1 port.

    python babylm/recipes/sprint1_port/prep_data.py \
        --train_dir ~/babylm/osf/text_data/train_10M/train_10M \
        --dev_dir ~/babylm/osf/text_data/dev/dev --out babylm/recipes/sprint1_port/data

Join rule (deterministic, no randomness, independent of the training seed):
  * domain files are read in sorted file-name order (bnc_spoken, childes, gutenberg, open_subtitles, simple_wiki,
    switchboard); text is UTF-8, kept verbatim (no normalisation, speaker tags like "*CHI:\t" kept);
  * --doc_unit blank_line (default): a document is a maximal run of non-blank lines; blank (whitespace-only) lines
    are the separators. Each document is emitted as  <|endoftext|> + encode("\n".join(lines) + "\n").
    This is sprint-1's FineWeb convention (every document preceded by <|endoftext|>, id 50256) applied to the only
    document boundaries the BabyLM files carry (simple_wiki articles, a few hundred breaks in the other files;
    switchboard has none and is one document);
  * --doc_unit line: every non-blank line is its own document (EOT before every line). NOT the default; offered
    because the official causal evaluation prefixes every sentence with BOS = <|endoftext|>.
  * documents are NOT shuffled in the stream: the stream is the corpus in file order. Data order randomness lives
    in the trainer's epoch sampler (random window phase + random permutation of windows each epoch, seeded).
Outputs (<out>/): train.bin / dev.bin (uint16 GPT-2 ids, no header), train_domain.bin / dev_domain.bin (uint8 domain
id per token), token_bytes.npy (UTF-8 bytes per id, EOT = 1, as sprint-1), meta.json (counts, words, sha256).
Words are counted as whitespace-separated strings (str.split), the convention of `wc -w`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOKENIZER_JSON = HERE / "gpt2_tokenizer" / "tokenizer.json"
EOT = 50256


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_docs(text: str, doc_unit: str) -> list[str]:
    lines = text.split("\n")
    if doc_unit == "line":
        return [ln + "\n" for ln in lines if ln.strip()]
    docs, cur = [], []
    for ln in lines:
        if ln.strip():
            cur.append(ln)
        elif cur:
            docs.append("\n".join(cur) + "\n")
            cur = []
    if cur:
        docs.append("\n".join(cur) + "\n")
    return docs


def token_bytes_table(tok) -> np.ndarray:
    """Bytes per GPT-2 id via the byte-level alphabet (= len(tiktoken decode_single_token_bytes)); EOT = 1."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_dec = {chr(c): b for b, c in zip(bs, cs)}
    vocab = tok.get_vocab()
    t = np.zeros(50257, dtype=np.int32)
    for s, i in vocab.items():
        t[i] = 1 if i == EOT else len(bytes(byte_dec[ch] for ch in s))
    return t


def encode_split(tok, files: list[Path], doc_unit: str, batch_docs: int = 2048) -> tuple[np.ndarray, np.ndarray, dict]:
    toks, doms, info = [], [], {}
    for d, f in enumerate(files):
        text = f.read_text(encoding="utf-8")
        docs = split_docs(text, doc_unit)
        n_tok = 0
        for i in range(0, len(docs), batch_docs):
            for e in tok.encode_batch(docs[i:i + batch_docs], add_special_tokens=False):
                ids = np.asarray([EOT] + e.ids, dtype=np.uint16)
                toks.append(ids)
                n_tok += len(ids)
        doms.append(np.full(n_tok, d, dtype=np.uint8))
        info[f.stem] = {"file": f.name, "sha256": sha256_file(f), "bytes": f.stat().st_size,
                        "words": len(text.split()), "docs": len(docs), "tokens": n_tok}
        print(f"[prep] {f.name}: {len(docs):,} docs, {info[f.stem]['words']:,} words, {n_tok:,} tokens", flush=True)
    return np.concatenate(toks), np.concatenate(doms), info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_dir", default="~/babylm/osf/text_data/train_10M/train_10M")
    ap.add_argument("--dev_dir", default="~/babylm/osf/text_data/dev/dev")
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--doc_unit", default="blank_line", choices=["blank_line", "line"])
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": "openai-community/gpt2 tokenizer.json (HF hub; byte-level BPE identical to tiktoken gpt2)",
            "tokenizer_sha256": sha256_file(TOKENIZER_JSON), "doc_unit": a.doc_unit, "eot": EOT,
            "join": "sorted files; per document: [EOT] + encode(lines joined by '\\n' + '\\n'); no shuffling",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for split, d, pat in (("train", a.train_dir, "*.train"), ("dev", a.dev_dir, "*.dev")):
        files = sorted(Path(d).glob(pat))
        assert files, f"no {pat} in {d}"
        t0 = time.time()
        ids, dom, info = encode_split(tok, files, a.doc_unit)
        ids.tofile(out / f"{split}.bin")
        dom.tofile(out / f"{split}_domain.bin")
        words = sum(v["words"] for v in info.values())
        meta[split] = {"dir": str(d), "domains": [f.stem for f in files], "files": info, "tokens": int(len(ids)),
                       "words": int(words), "words_per_token": words / len(ids), "eot_count": int((ids == EOT).sum()),
                       "bin_sha256": sha256_file(out / f"{split}.bin"),
                       "domain_bin_sha256": sha256_file(out / f"{split}_domain.bin"), "encode_s": time.time() - t0}
        print(f"[prep] {split}: {len(ids):,} tokens, {words:,} words, {words / len(ids):.4f} words/token", flush=True)
    np.save(out / "token_bytes.npy", token_bytes_table(tok))
    meta["token_bytes_sha256"] = sha256_file(out / "token_bytes.npy")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[prep] wrote {out}/meta.json", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    main()
