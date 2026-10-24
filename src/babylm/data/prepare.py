"""Prepare the BabyLM 2025 text data: verify, extract, tokenize, build data-ruler splits and eval subsets.

    python -m babylm.data.prepare [--download] [--data_dir data/babylm] [--train_tokenizer]

Steps (each idempotent):
 1. zips in <data_dir>/zips are md5-verified against OSF's published md5s (--download fetches missing ones),
    then extracted to <data_dir>/raw/{train_10M,train_100M,dev,test}/<source>.<ext>.
 2. raw statistics per file: sha256, bytes, lines, documents, words (str.split) and wc_posix (what `wc -w` prints
    in the POSIX locale, where a run of non-ASCII bytes alone is not a word: the numbers BabyLM's docs quote).
 3. every line is normalised (corpus.normalize_line: " ".join(line.split()), the GPT-BERT 2025 reference's rule;
    documents and words unchanged). tokenizer: byte-level BPE (HF tokenizers), vocab 16384 including <|endoftext|> (id 0), trained on the
    train_10M documents only. The committed babylm/data/tok16k_10m/tokenizer.json is used unless
    --train_tokenizer; its sha256 is checked against manifest.json.
 4. tokenize train_10M, train_100M, dev and test (test = FINAL: tokenized, never evaluated here) into
    <data_dir>/bin/{split}.{bin,words.bin,src.bin,meta.json} (format in babylm/data/corpus.py).
 5. data-ruler splits train_20M, train_30M, train_50M (str.split words): all train_10M documents plus whole
    train_100M documents that share no line POSITION with train_10M. train_10M is a set of contiguous 1000-line
    aligned chunks of train_100M (verified here line by line); a 100M document touching a covered line is never
    added. Per source, extra words = N * (source share of train_100M words) - (source words in train_10M); extra
    documents are taken as the prefix of one seeded permutation per source (seed 1234) whose word total is
    closest to that quota, so 10M c 20M c 30M c 50M c 100M by construction.
 6. eval subsets: dev_sub (from dev) and train_sub (from train_10M), ~1M words each: per source, the prefix of a
    seeded permutation (seed 1234) of whole documents whose word total is closest to 1M x the source's share of
    the split. Concatenated in split order, each document preceded by BOS. Document lists are written to
    babylm/data/subsets/*.json (committed).
 7. babylm/data/manifest.json (committed): all of the above with hashes. With --splits (e.g. train_10M,dev for a
    laptop that never needs train_100M) only those splits and what derives from them are built; every output's
    sha256 is checked against the committed manifest and the manifest is left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from babylm.data.corpus import (BOS_ID, BOS_TOKEN, DEFAULT_DATA_DIR, MANIFEST_PATH, OFFICIAL_SPLITS, RULER_SPLITS,
                                SOURCES, SUBSET_DIR, TOKENIZER_PATH, VOCAB_SIZE, bin_dir, doc_ranges, raw_path,
                                doc_text, read_lines, sha256_array, sha256_file, token_word_ids)

ZIPS = {
    "train_10M": ("https://osf.io/download/y7djq/", "dc67e5d2db96ed4f76cb072e2504a18c"),
    "train_100M": ("https://osf.io/download/ywea7/", "f63cde477afd5673f3367fd009ea70cb"),
    "dev": ("https://osf.io/download/uyd7v/", "ad712e2f943ddbbbc94036178144e810"),
    "test": ("https://osf.io/download/ftwu3/", "dc7f838fe026dfd9781c6f41cba1ebd9"),
}
EXPECTED_WC_POSIX = {"train_10M": 9_946_560, "train_100M": 98_941_494, "dev": 10_402_202}
SUBSET_SEED = 1234
RULER_SEED = 1234
SUBSET_WORDS = 1_000_000
ALIGN_BLOCK = 1000


def log(msg: str) -> None:
    print(f"[prepare {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------
# 1. download / verify / extract
# ------------------------------------------------------------------------------------------------
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch_and_extract(data_dir: Path, download: bool, splits=tuple(ZIPS)) -> dict[str, str]:
    zdir = data_dir / "zips"
    zdir.mkdir(parents=True, exist_ok=True)
    md5s = {}
    for split in splits:
        url, md5 = ZIPS[split]
        z = zdir / f"{split}.zip"
        if not z.exists():
            if not download:
                raise SystemExit(f"{z} missing: pass --download (fetches {url})")
            log(f"downloading {url} -> {z}")
            tmp = z.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            tmp.replace(z)
        got = md5_file(z)
        if got != md5:
            raise SystemExit(f"{z}: md5 {got} != published {md5}")
        md5s[split] = got
        out = data_dir / "raw"
        want = [raw_path(data_dir, split, s) for s in SOURCES]
        if not all(p.exists() for p in want):
            with zipfile.ZipFile(z) as zf:
                for info in zf.infolist():
                    name = info.filename
                    if info.is_dir():
                        continue
                    parts = Path(name).parts
                    if len(parts) != 2 or parts[0] != split or ".." in parts:
                        raise SystemExit(f"{z}: unexpected member {name!r}")
                    target = out / split / parts[1]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        log(f"{split}: md5 ok, extracted")
    return md5s


def wc_posix_words(data: bytes) -> int:
    """`wc -w` in the POSIX locale (GNU coreutils 9.4): a maximal run of non-ASCII-whitespace bytes counts as a word
    only if it holds at least one printable ASCII byte."""
    a = np.frombuffer(data, dtype=np.uint8)
    space = np.isin(a, np.array([9, 10, 11, 12, 13, 32], dtype=np.uint8))
    ns = np.flatnonzero(~space)
    if len(ns) == 0:
        return 0
    start = np.ones(len(ns), dtype=bool)
    start[1:] = ns[1:] != ns[:-1] + 1
    seg = np.cumsum(start) - 1
    printable = (a[ns] >= 0x21) & (a[ns] <= 0x7E)
    return int(len(np.unique(seg[printable])))


# ------------------------------------------------------------------------------------------------
# documents
# ------------------------------------------------------------------------------------------------
class SourceDocs:
    """Documents of one file: line ranges, texts and word counts."""

    def __init__(self, split: str, source: str, lines: list[str]):
        self.split, self.source, self.lines = split, source, lines
        self.ranges = doc_ranges(lines, source)
        self.texts = [doc_text(lines, s, e) for s, e in self.ranges]
        self.words = np.array([len(t.split()) for t in self.texts], dtype=np.int64)
        raw_words = sum(len(lines[i].split()) for s, e in self.ranges for i in range(s, e))
        if raw_words != int(self.words.sum()):
            raise SystemExit(f"{split}/{source}: normalisation changed the word count")


def load_docs(data_dir: Path, split: str) -> dict[str, SourceDocs]:
    out = {}
    for s in SOURCES:
        p = raw_path(data_dir, split, s)
        text = p.read_text(encoding="utf-8")
        if BOS_TOKEN in text:
            raise SystemExit(f"{p} contains the literal special token {BOS_TOKEN!r}; refusing to tokenize")
        out[s] = SourceDocs(split, s, text.split("\n"))
    return out


def raw_stats(data_dir: Path, split: str, docs: dict[str, SourceDocs]) -> dict[str, Any]:
    files = {}
    for s in SOURCES:
        p = raw_path(data_dir, split, s)
        b = p.read_bytes()
        text = b.decode("utf-8")
        files[p.name] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "lines": text.count("\n"),
                         "words": len(text.split()), "wc_posix": wc_posix_words(b), "docs": len(docs[s].ranges)}
        assert files[p.name]["words"] == int(docs[s].words.sum()), (split, s)
    tot = {k: sum(f[k] for f in files.values()) for k in ("bytes", "lines", "words", "wc_posix", "docs")}
    exp = EXPECTED_WC_POSIX.get(split)
    if exp is not None and tot["wc_posix"] != exp:
        raise SystemExit(f"{split}: wc_posix total {tot['wc_posix']:,} != expected {exp:,}")
    return {"files": files, "total": tot}


# ------------------------------------------------------------------------------------------------
# tokenizer
# ------------------------------------------------------------------------------------------------
def train_tokenizer(docs10: dict[str, SourceDocs], path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[BOS_TOKEN],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    texts = [t for s in SOURCES for t in docs10[s].texts]
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    if tok.get_vocab_size() != VOCAB_SIZE or tok.token_to_id(BOS_TOKEN) != BOS_ID:
        raise SystemExit(f"tokenizer: vocab {tok.get_vocab_size()} bos id {tok.token_to_id(BOS_TOKEN)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path), pretty=True)


def load_tokenizer(path: Path = TOKENIZER_PATH):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    assert tok.token_to_id(BOS_TOKEN) == BOS_ID and tok.get_vocab_size() == VOCAB_SIZE
    assert tok.post_processor is None, "the data tokenizer must not add special tokens itself"
    return tok


# ------------------------------------------------------------------------------------------------
# tokenization
# ------------------------------------------------------------------------------------------------
class DocTokens:
    """Per-document token ids (uint16) and local word ids (uint32) for one file."""

    def __init__(self, ids: list[np.ndarray], wids: list[np.ndarray], nwords: np.ndarray):
        self.ids, self.wids, self.nwords = ids, wids, nwords


def tokenize_docs(tok, texts: list[str], chunk_chars: int = 8_000_000) -> DocTokens:
    ids_out, wid_out, nw = [], [], []
    i = 0
    while i < len(texts):
        j, c = i, 0
        while j < len(texts) and (c == 0 or c + len(texts[j]) <= chunk_chars):
            c += len(texts[j])
            j += 1
        encs = tok.encode_batch(texts[i:j], add_special_tokens=False)
        for t, e in zip(texts[i:j], encs):
            ids = np.asarray(e.ids, dtype=np.int64)
            if ids.size and (ids.max() >= VOCAB_SIZE or (ids == BOS_ID).any()):
                raise SystemExit("tokenizer produced an out-of-range id or BOS inside a document")
            starts = np.fromiter((o[0] for o in e.offsets), dtype=np.int64, count=len(e.offsets))
            w, n = token_word_ids(t, starts)
            if n != len(t.split()):
                raise SystemExit(f"word count mismatch {n} vs {len(t.split())}")
            if len(w) and (np.diff(w) < 0).any():
                raise SystemExit("word ids not monotone within a document")
            ids_out.append(ids.astype(np.uint16))
            wid_out.append(w.astype(np.uint32))
            nw.append(n)
        i = j
    return DocTokens(ids_out, wid_out, np.array(nw, dtype=np.int64))


def assemble(parts: list[tuple[int, DocTokens, list[int]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """parts: (source id, DocTokens, doc indices) in stream order -> (tokens, words, src, per-source counts)."""
    n_tok = sum(len(dt.ids[k]) + 1 for _, dt, idx in parts for k in idx)
    tokens = np.empty(n_tok, dtype=np.uint16)
    words = np.empty(n_tok, dtype=np.uint32)
    src = np.empty(n_tok, dtype=np.uint8)
    p, woff = 0, 0
    per: dict[str, dict[str, int]] = {}
    for sid, dt, idx in parts:
        st = per.setdefault(SOURCES[sid], {"tokens": 0, "words": 0, "docs": 0})
        for k in idx:
            ids, wid, n = dt.ids[k], dt.wids[k], int(dt.nwords[k])
            L = len(ids) + 1
            tokens[p] = BOS_ID
            words[p] = woff
            tokens[p + 1:p + L] = ids
            words[p + 1:p + L] = wid.astype(np.int64) + woff
            src[p:p + L] = sid
            p += L
            woff += n
            st["tokens"] += L
            st["words"] += n
            st["docs"] += 1
    assert p == n_tok
    return tokens, words, src, per


def write_split(name: str, tokens: np.ndarray, words: np.ndarray, src: np.ndarray, per: dict, extra: dict,
                data_dir: Path, tok_sha: str) -> dict:
    d = bin_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    tokens.astype("<u2").tofile(d / f"{name}.bin")
    words.astype("<u4").tofile(d / f"{name}.words.bin")
    src.astype("u1").tofile(d / f"{name}.src.bin")
    n_words = sum(v["words"] for v in per.values())
    meta = {"split": name, "tokens": int(len(tokens)), "words": int(n_words),
            "docs": int(sum(v["docs"] for v in per.values())), "tokens_per_word": len(tokens) / max(n_words, 1),
            "per_source": per, "tokenizer_sha256": tok_sha, "sha256": sha256_file(d / f"{name}.bin"),
            "sha256_words": sha256_file(d / f"{name}.words.bin"), "sha256_src": sha256_file(d / f"{name}.src.bin"),
            "format": "raw little-endian uint16 tokens / uint32 word ids / uint8 source ids; BOS (0) before every doc",
            **extra}
    (d / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    log(f"{name}: {meta['tokens']:,} tokens, {meta['words']:,} words, {meta['docs']:,} docs, "
        f"{meta['tokens_per_word']:.4f} tokens/word")
    return meta


def tokenize_split(tok, docs: dict[str, SourceDocs]) -> dict[str, DocTokens]:
    return {s: tokenize_docs(tok, docs[s].texts) for s in SOURCES}


# ------------------------------------------------------------------------------------------------
# 10M -> 100M alignment and data-ruler splits
# ------------------------------------------------------------------------------------------------
def align_lines(a: list[str], b: list[str], block: int = ALIGN_BLOCK) -> np.ndarray:
    """Position in b of every line of a (a = train_10M file, b = train_100M file; a trailing "" sentinel of a final
    newline is ignored). a is made of `block`-aligned contiguous chunks of b: every block of `block` lines of a gets
    one offset from lines unique in both files, and every line of the block is verified against b."""
    na = len(a) - 1 if a and a[-1] == "" else len(a)
    ca, cb = Counter(a[:na]), Counter(b)
    uniq_b = {l: i for i, l in enumerate(b) if cb[l] == 1}
    pos = np.full(na, -1, dtype=np.int64)
    prev_off = None
    n_blocks = (na + block - 1) // block
    offs: list[int | None] = []
    for k in range(n_blocks):
        lo, hi = k * block, min(na, (k + 1) * block)
        cand = {uniq_b[a[i]] - i for i in range(lo, hi) if ca[a[i]] == 1 and a[i] in uniq_b}
        if len(cand) > 1:
            raise SystemExit(f"alignment: block {k} has {len(cand)} offsets {sorted(cand)[:4]}")
        offs.append(cand.pop() if cand else None)
    for k in range(n_blocks):
        lo, hi = k * block, min(na, (k + 1) * block)
        tries = [o for o in (offs[k], prev_off, next((x for x in offs[k + 1:] if x is not None), None))
                 if o is not None]
        for o in tries:
            if 0 <= lo + o and hi + o <= len(b) and all(a[i] == b[i + o] for i in range(lo, hi)):
                pos[lo:hi] = np.arange(lo, hi) + o
                prev_off = o
                break
        else:
            raise SystemExit(f"alignment: block {k} (lines {lo}-{hi}) matches no candidate offset")
    if len(np.unique(pos)) != na:
        raise SystemExit("alignment: two train_10M lines map to the same train_100M line")
    return pos


def closest_prefix(cum: np.ndarray, quota: float) -> int:
    """k minimising |cum[k] - quota| (cum[0] = 0); ties -> smaller k. Monotone in quota (nesting)."""
    return int(np.argmin(np.abs(cum - quota)))


def build_ruler(docs10, docs100, tok10, tok100, raw10, raw100, data_dir: Path, tok_sha: str) -> dict:
    W100 = {s: int(docs100[s].words.sum()) for s in SOURCES}
    W10 = {s: int(docs10[s].words.sum()) for s in SOURCES}
    tot100 = sum(W100.values())
    rng = np.random.Generator(np.random.PCG64(RULER_SEED))
    align_info, pools = {}, {}
    hash10 = {hashlib.sha1(t.encode()).hexdigest() for s in SOURCES for t in docs10[s].texts}
    for s in SOURCES:
        a, b = docs10[s].lines, docs100[s].lines
        pos = align_lines(a, b)
        covered = np.zeros(len(b), dtype=bool)
        covered[pos] = True
        chunk_starts = np.flatnonzero(np.diff(pos, prepend=-10) != 1)
        free = [k for k, (lo, hi) in enumerate(docs100[s].ranges) if not covered[lo:hi].any()]
        touched = [k for k, (lo, hi) in enumerate(docs100[s].ranges) if covered[lo:hi].any()]
        partial_words = 0
        for k in touched:
            lo, hi = docs100[s].ranges[k]
            if not covered[lo:hi].all():
                partial_words += sum(len(b[i].split()) for i in range(lo, hi) if not covered[i])
        dup = sum(1 for k in free if hashlib.sha1(docs100[s].texts[k].encode()).hexdigest() in hash10)
        perm = rng.permutation(len(free))
        order = [free[i] for i in perm]
        cum = np.concatenate([[0], np.cumsum(docs100[s].words[order])])
        pools[s] = (order, cum)
        align_info[s] = {"lines_10M": int(len(pos)), "chunks": int(len(chunk_starts)),
                         "chunk_starts_100M": [int(pos[i]) for i in chunk_starts],
                         "chunk_lengths": np.diff(np.append(chunk_starts, len(pos))).tolist(),
                         "docs_100M": len(docs100[s].ranges), "docs_100M_free": len(free),
                         "docs_100M_touching_10M": len(touched),
                         "words_100M_free": int(docs100[s].words[free].sum()) if free else 0,
                         "words_unreachable_in_partially_covered_docs": int(partial_words),
                         "free_docs_identical_to_a_10M_doc": int(dup)}
        log(f"align {s}: {len(chunk_starts)} chunks, free docs {len(free)}/{len(docs100[s].ranges)}, "
            f"free words {align_info[s]['words_100M_free']:,}, unreachable {partial_words:,}, dup {dup}")
    out: dict[str, Any] = {"seed": RULER_SEED, "rule": __doc__.split(" 5. ")[1].split(" 6. ")[0].strip(),
                           "alignment": align_info, "splits": {}}
    prev_sel: dict[str, set] = {s: set() for s in SOURCES}
    rdir = data_dir / "ruler"
    rdir.mkdir(parents=True, exist_ok=True)
    for name, N in RULER_SPLITS.items():
        parts, sel_all = [], {}
        for sid, s in enumerate(SOURCES):
            order, cum = pools[s]
            quota = N * W100[s] / tot100 - W10[s]
            k = closest_prefix(cum, quota)
            sel = sorted(order[:k])
            if not prev_sel[s] <= set(sel):
                raise SystemExit(f"{name}/{s}: nesting violated")
            prev_sel[s] = set(sel)
            sel_all[s] = sel
            parts.append((sid, tok10[s], list(range(len(docs10[s].ranges)))))
            parts.append((sid, tok100[s], sel))
        tokens, words, src, per = assemble(parts)
        doclist = {"split": name, "target_words": N, "seed": RULER_SEED,
                   "extra_docs_100M": {s: sel_all[s] for s in SOURCES},
                   "extra_line_ranges_100M": {s: [list(docs100[s].ranges[k]) for k in sel_all[s]] for s in SOURCES}}
        lp = rdir / f"{name}.docs.json"
        lp.write_text(json.dumps(doclist))
        meta = write_split(name, tokens, words, src, per, {
            "composition": "per source: all train_10M docs (file order), then the extra train_100M docs (100M order)",
            "doc_list": str(lp.relative_to(data_dir)), "doc_list_sha256": sha256_file(lp),
            "extra_words": {s: int(docs100[s].words[sel_all[s]].sum()) for s in SOURCES}}, data_dir, tok_sha)
        rawd = data_dir / "raw" / name
        rawd.mkdir(parents=True, exist_ok=True)
        for s in SOURCES:
            texts = docs10[s].texts + [docs100[s].texts[k] for k in sel_all[s]]
            (rawd / f"{s}.train").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
        out["splits"][name] = {k: meta[k] for k in ("tokens", "words", "docs", "sha256", "doc_list_sha256",
                                                    "extra_words", "per_source")}
    return out


# ------------------------------------------------------------------------------------------------
# eval subsets
# ------------------------------------------------------------------------------------------------
def select_subset(doc_words: dict[str, np.ndarray], seed: int = SUBSET_SEED,
                  target: int = SUBSET_WORDS) -> dict[str, list[int]]:
    """Per source (SOURCES order, one generator): the prefix of a permutation of the documents whose word total is
    closest to target x the source's share of the split's words (at least one document). Sorted doc indices."""
    rng = np.random.Generator(np.random.PCG64(seed))
    Wtot = sum(int(doc_words[s].sum()) for s in SOURCES)
    chosen = {}
    for s in SOURCES:
        perm = rng.permutation(len(doc_words[s]))
        cum = np.concatenate([[0], np.cumsum(doc_words[s][perm])])
        k = max(1, closest_prefix(cum, target * doc_words[s].sum() / Wtot))
        chosen[s] = sorted(int(i) for i in perm[:k])
    return chosen


def build_subset(name: str, split: str, docs, toks, data_dir: Path, tok_sha: str, write_list: bool = True) -> dict:
    chosen = select_subset({s: docs[s].words for s in SOURCES})
    parts = [(sid, toks[s], chosen[s]) for sid, s in enumerate(SOURCES)]
    tokens, words, src, per = assemble(parts)
    doclist = {"subset": name, "split": split, "seed": SUBSET_SEED, "target_words": SUBSET_WORDS,
               "rule": "per source: prefix of a seeded permutation of whole docs closest to 1M x source share; "
                       "stream = docs in split order (source order, then doc index), BOS before each",
               "docs": chosen, "line_ranges": {s: [list(docs[s].ranges[k]) for k in chosen[s]] for s in SOURCES},
               "words": int(sum(v["words"] for v in per.values())), "tokens": int(len(tokens)),
               "sha256_tokens": sha256_array(tokens.astype("<u2")), "tokenizer_sha256": tok_sha}
    lp = SUBSET_DIR / f"{name}.json"
    if write_list:
        SUBSET_DIR.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(doclist, indent=1))
    elif lp.exists():
        committed = json.loads(lp.read_text())
        if committed["docs"] != chosen or committed["sha256_tokens"] != doclist["sha256_tokens"]:
            raise SystemExit(f"{name}: document list or tokens differ from the committed {lp}")
    meta = write_split(name, tokens, words, src, per, {"subset_of": split, "doc_list": f"babylm/data/subsets/{name}.json",
                                                        "fingerprint": doclist["sha256_tokens"]}, data_dir, tok_sha)
    return {k: meta[k] for k in ("tokens", "words", "docs", "sha256", "per_source")} | {"subset_of": split}


# ------------------------------------------------------------------------------------------------
def check_manifest(old: dict, section: str, name: str, meta: dict, strict: bool) -> None:
    """Compare a freshly built split's sha256 with the committed manifest (any mismatch is fatal when strict)."""
    want = (old.get(section, {}).get("splits", {}) if section == "ruler" else old.get(section, {})).get(name)
    if not want:
        return
    if want["sha256"] != meta["sha256"]:
        msg = f"{name}: sha256 {meta['sha256']} != committed manifest {want['sha256']}"
        if strict:
            raise SystemExit(msg + " (tokenization is not reproducible here; do not train on this data)")
        log("WARNING " + msg)
    else:
        log(f"{name}: sha256 matches the committed manifest")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--download", action="store_true", help="fetch missing zips from OSF")
    ap.add_argument("--splits", default=",".join(OFFICIAL_SPLITS),
                    help="official splits this host needs (comma list of train_10M,train_100M,dev,test). Derived: "
                         "train_sub needs train_10M, dev_sub needs dev, train_20M/30M/50M need train_10M+train_100M. "
                         "With a subset, babylm/data/manifest.json is not rewritten; outputs are checked against it")
    ap.add_argument("--train_tokenizer", action="store_true", help="(re)train the tokenizer instead of using the "
                    "committed one, and rewrite it")
    a = ap.parse_args(argv)
    splits = [x.strip() for x in a.splits.split(",") if x.strip()]
    bad = [x for x in splits if x not in OFFICIAL_SPLITS]
    if bad:
        raise SystemExit(f"unknown splits {bad}; choose from {list(OFFICIAL_SPLITS)}")
    full = set(splits) == set(OFFICIAL_SPLITS)
    data_dir = Path(a.data_dir)
    t0 = time.time()
    md5s = fetch_and_extract(data_dir, a.download, splits)
    docs = {sp: load_docs(data_dir, sp) for sp in splits}
    raw = {sp: raw_stats(data_dir, sp, docs[sp]) for sp in splits}
    for sp in splits:
        log(f"{sp}: words {raw[sp]['total']['words']:,} (wc_posix {raw[sp]['total']['wc_posix']:,}) "
            f"docs {raw[sp]['total']['docs']:,}")
    old = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    if a.train_tokenizer or not TOKENIZER_PATH.exists():
        if "train_10M" not in docs:
            raise SystemExit("training the tokenizer needs train_10M")
        log("training tokenizer on train_10M")
        train_tokenizer(docs["train_10M"], TOKENIZER_PATH)
    tok_sha = sha256_file(TOKENIZER_PATH)
    want = old.get("tokenizer", {}).get("sha256")
    if want and want != tok_sha and not a.train_tokenizer:
        raise SystemExit(f"tokenizer sha256 {tok_sha} != manifest {want}")
    strict = bool(old) and not a.train_tokenizer
    tok = load_tokenizer()
    toks, metas = {}, {}
    for sp in splits:
        log(f"tokenizing {sp}")
        toks[sp] = tokenize_split(tok, docs[sp])
        parts = [(sid, toks[sp][s], list(range(len(docs[sp][s].ranges)))) for sid, s in enumerate(SOURCES)]
        tokens, words, src, per = assemble(parts)
        extra = {"protected_final": True} if sp == "test" else {}
        metas[sp] = write_split(sp, tokens, words, src, per, extra, data_dir, tok_sha)
        check_manifest(old, "splits", sp, metas[sp], strict)
        del tokens, words, src
    ruler = None
    if "train_10M" in splits and "train_100M" in splits:
        ruler = build_ruler(docs["train_10M"], docs["train_100M"], toks["train_10M"], toks["train_100M"],
                            raw["train_10M"], raw["train_100M"], data_dir, tok_sha)
        for name, m in ruler["splits"].items():
            check_manifest(old, "ruler", name, m, strict)
    subsets = {}
    for name, split in (("dev_sub", "dev"), ("train_sub", "train_10M")):
        if split in splits:
            subsets[name] = build_subset(name, split, docs[split], toks[split], data_dir, tok_sha, write_list=full)
            check_manifest(old, "subsets", name, subsets[name], strict)
    if not full:
        log(f"partial prepare ({','.join(splits)}): manifest not rewritten ({time.time() - t0:.0f}s)")
        return
    manifest = {
        "created_by": "python -m babylm.data.prepare",
        "source": "BabyLM 2025 text data (OSF node ryjfm text_data, byte-identical to 2024)",
        "zips_md5": md5s,
        "document_rule": "blank-line-separated blocks (line.strip()==''); switchboard: 2000-line pieces (no blank lines)",
        "word_rule": "items of str.split(); wc_posix = `wc -w` in the POSIX locale (BabyLM's quoted counts)",
        "normalisation": "every line -> ' '.join(line.split()) (babylm.data.corpus.normalize_line) before tokenizer "
                         "training and tokenization; newlines, blank lines and document boundaries kept; word counts "
                         "unchanged (asserted)",
        "raw": raw,
        "tokenizer": {"path": "babylm/data/tok16k_10m/tokenizer.json", "sha256": tok_sha, "type": "byte-level BPE",
                      "vocab_size": VOCAB_SIZE, "special": {BOS_TOKEN: BOS_ID}, "trained_on": "train_10M documents",
                      "min_frequency": 2, "library": "tokenizers " + __import__("tokenizers").__version__},
        "splits": {sp: {k: m[k] for k in ("tokens", "words", "docs", "tokens_per_word", "sha256", "sha256_words",
                                          "sha256_src", "per_source")} for sp, m in metas.items()},
        "ruler": ruler,
        "subsets": subsets,
        "protected": {"test": "FINAL split: tokenized only; evaluation requires train.py --final"},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    log(f"manifest -> {MANIFEST_PATH} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    sys.exit(main())
