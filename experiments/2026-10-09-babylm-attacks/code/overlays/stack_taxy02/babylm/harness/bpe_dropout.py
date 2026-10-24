"""Attack A3: per-epoch BPE-dropout re-tokenisation of the training stream (Provilkov et al. 2020; subword
regularisation, Kudo 2018). Known idea, not new. Evaluation always uses the canonical (deterministic) stream.

    train.py --bpe_dropout 0.1x8,0x2 [--bpe_dropout_scope all|rare|frequent --bpe_dropout_rare_max 63]

Segmentation (Provilkov Alg. 1, = HF tokenizers' BPE dropout): a pre-token starts as its bytes; at every step the
candidate merges (adjacent pairs in the merge table) are visited in (rank, position) order and each is dropped with
probability p; the first survivor is applied; if every candidate is dropped, or none exists, segmentation stops.
p = 0 is ordinary BPE and reproduces the committed tokenizer exactly (checked on every pre-token type at build time).

Stream layout. The canonical stream (prepare.py) is cut into occurrences: one BOS (fixed, never re-segmented) or one
ByteLevel pre-token of a document (pre_tokenize_str of the decoded document; canonical tokens never cross pre-token
boundaries, asserted). Each occurrence has an anchor = the stored word id of its first canonical token. A token
starting at byte b of a pre-token gets the word of the first non-whitespace char at or after b (prepare's rule): that
is anchor + rel[b] inside the pre-token, or the next occurrence's anchor for trailing whitespace. So every epoch
stream has exactly the canonical stream's whitespace words, in order, with the same word-initial count: R0 exposure
(whitespace words) is unchanged; only the number of tokens per word grows.

Scope (per whitespace word, by its count in the training split): all | rare (count <= rare_max) | frequent
(count > rare_max). Out-of-scope occurrences keep their canonical tokens. Bottleneck bands: rare = 1-63.

Randomness: occurrences are processed in fixed chunks of CHUNK; chunk c of epoch e draws from
PCG64([seed, e, c, 0xB9E]), so a stream depends only on (data, p, scope, rare_max, seed, epoch), not on the worker
count. Streams are cached in <data_dir>/bin/bpedrop/ with a meta.json (sha256s, token count, inflation).

MultiStreamSampler: the port's EpochSampler (same PCG64([seed, e]) offset + permutation, batches straddle epochs,
the final partial batch spills into the next epoch) with stream e for epoch e (canonical beyond the plan). With every
p = 0 it yields exactly EpochSampler's windows.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from babylm.data import corpus as C

CHUNK = 1 << 17
SEED_TAG = 0xB9E
BOS_TYPE = 0
FORMAT_VERSION = 1


# ------------------------------------------------------------------------------------------------
# plan
# ------------------------------------------------------------------------------------------------
def parse_plan(spec: str, epochs: int) -> list[float]:
    """'0.1x8,0x2' -> [0.1]*8 + [0.0]*2; a single value without xN fills every epoch."""
    items = [s.strip() for s in spec.split(",") if s.strip()]
    out: list[float] = []
    for it in items:
        m = re.fullmatch(r"([0-9.eE+-]+)(?:x(\d+))?", it)
        if not m:
            raise ValueError(f"bad --bpe_dropout item {it!r} (want p or pxN)")
        p = float(m.group(1))
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout p must be in [0, 1): {p}")
        n = int(m.group(2)) if m.group(2) else (epochs if len(items) == 1 else 1)
        out += [p] * n
    if len(out) != epochs:
        raise ValueError(f"--bpe_dropout covers {len(out)} epochs, --epochs is {epochs}")
    return out


# ------------------------------------------------------------------------------------------------
# BPE with dropout
# ------------------------------------------------------------------------------------------------
def byte_alphabet() -> dict[int, str]:
    """GPT-2 / HF ByteLevel byte -> unicode char map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPE:
    """The committed ByteLevel BPE (vocab + ranked merges) with seeded dropout."""

    def __init__(self, tokenizer_path: str | Path = C.TOKENIZER_PATH):
        tj = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
        vocab: dict[str, int] = tj["model"]["vocab"]
        self.vocab = vocab
        self.id_to_str = {i: s for s, i in vocab.items()}
        self.merges: dict[tuple[int, int], tuple[int, int]] = {}
        for r, m in enumerate(tj["model"]["merges"]):
            a, b = m if isinstance(m, list) else m.split(" ")
            self.merges[(vocab[a], vocab[b])] = (r, vocab[a + b])
        ba = byte_alphabet()
        self.byte_id = [vocab[ba[b]] for b in range(256)]
        self.char_to_byte = {c: b for b, c in ba.items()}
        self.tok_bytes = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
        for s, i in vocab.items():
            self.tok_bytes[i] = len(s)
        self._cands: dict[tuple, tuple] = {}

    def init_ids(self, piece: str) -> tuple[int, ...]:
        """A ByteLevel pre-token (byte-alphabet string) -> its initial one-byte symbol ids."""
        return tuple(self.vocab[ch] for ch in piece)

    def candidates(self, syms: tuple) -> tuple:
        c = self._cands.get(syms)
        if c is None:
            mg = self.merges
            lst = []
            for i in range(len(syms) - 1):
                m = mg.get((syms[i], syms[i + 1]))
                if m is not None:
                    lst.append((m[0], i, m[1]))
            lst.sort()
            c = tuple(lst)
            if len(self._cands) > 4_000_000:
                self._cands.clear()
            self._cands[syms] = c
        return c

    def segment(self, init: tuple, p: float, u: list[float] | None = None, k: int = 0) -> tuple[tuple, int]:
        """Segment one pre-token. u: uniform draws consumed from index k (a draw < p drops the candidate).
        Returns (token ids, next k)."""
        syms = init
        while len(syms) > 1:
            cands = self.candidates(syms)
            chosen = None
            for c in cands:
                if p > 0.0:
                    drop = u[k] < p
                    k += 1
                    if drop:
                        continue
                chosen = c
                break
            if chosen is None:
                break
            _, i, nid = chosen
            syms = syms[:i] + (nid,) + syms[i + 2:]
        return syms, k


# ------------------------------------------------------------------------------------------------
# occurrence table
# ------------------------------------------------------------------------------------------------
@dataclass
class Table:
    occ_type: np.ndarray    # int32 [n_occ]
    anchor: np.ndarray      # int64 [n_occ] word id of the occurrence's first canonical token
    word_count: np.ndarray  # int64 [n_occ] training-split count of the anchor's whitespace word (BOS: -1)
    types: list[tuple]      # type id -> initial symbol ids (BOS: (0,))
    canon: list[tuple]      # type id -> canonical token ids
    rel: list[np.ndarray]   # type id -> int16 [n_bytes] word offset from the anchor, -1 = next anchor
    n_tokens: int
    n_words: int

    def __len__(self) -> int:
        return len(self.occ_type)


def _rel_words(text: str) -> np.ndarray:
    """Per byte of `text` (one pre-token): offset (from the pre-token's first non-whitespace char's word) of the word
    holding the first non-whitespace char at or after that byte's char; -1 if there is none inside the pre-token."""
    rel_char = np.full(len(text), -1, dtype=np.int64)
    w, seen, prev_space = 0, False, True
    for i, ch in enumerate(text):
        if not ch.isspace():
            if seen and prev_space:
                w += 1
            seen = True
            rel_char[i] = w
        prev_space = ch.isspace()
    nxt = -1
    for i in range(len(text) - 1, -1, -1):  # first non-space char at or after i
        if rel_char[i] >= 0:
            nxt = rel_char[i]
        rel_char[i] = nxt
    nb = [len(ch.encode("utf-8")) for ch in text]
    return np.repeat(rel_char, nb).astype(np.int16)


def build_table(split: C.SplitData, bpe: BPE, log=print) -> Table:
    """Cut the canonical stream into occurrences and check that p = 0 rebuilds it bit for bit."""
    from tokenizers import Tokenizer
    pre = Tokenizer.from_file(str(C.TOKENIZER_PATH)).pre_tokenizer
    tokens = np.asarray(split.tokens).astype(np.int64)
    words = np.asarray(split.words).astype(np.int64)
    N = len(tokens)
    bos = np.flatnonzero(tokens == C.BOS_ID)
    if len(bos) == 0 or bos[0] != 0:
        raise ValueError("stream must start with BOS")
    ends = np.append(bos[1:], N)
    tb = bpe.tok_bytes
    type_id: dict[str, int] = {}
    types: list[tuple] = [(C.BOS_ID,)]
    canon: list[tuple] = [(C.BOS_ID,)]
    rel: list[np.ndarray] = [np.zeros(1, dtype=np.int16)]
    occ_type: list[int] = []
    anchor: list[int] = []
    word_strings: list[str] = []
    id_to_bytes = [bytes(bpe.char_to_byte[c] for c in bpe.id_to_str[i]) if i in bpe.id_to_str else b""
                   for i in range(len(tb))]
    t0 = time.time()
    for d, (s, e) in enumerate(zip(bos, ends)):
        occ_type.append(BOS_TYPE)
        anchor.append(int(words[s]))
        ids = tokens[s + 1:e]
        if len(ids) == 0:
            continue
        ids_l = ids.tolist()
        wds_l = words[s + 1:e].tolist()
        text = b"".join(id_to_bytes[i] for i in ids_l).decode("utf-8")
        word_strings.extend(text.split())
        pieces = pre.pre_tokenize_str(text)
        cum = np.cumsum(tb[ids]).tolist()
        n_ids = len(ids_l)
        j = 0  # next canonical token index within the doc
        pos = 0
        for piece, _ in pieces:
            pos_end = pos + len(piece)
            j0 = j
            while j < n_ids and cum[j] < pos_end:
                j += 1
            if j >= n_ids or cum[j] != pos_end:
                raise ValueError(f"doc {d}: canonical token crosses a pre-token boundary at byte {pos_end}")
            j += 1
            seg = tuple(ids_l[j0:j])
            t = type_id.get(piece)
            if t is None:
                t = len(types)
                type_id[piece] = t
                init = bpe.init_ids(piece)
                mine, _ = bpe.segment(init, 0.0)
                if mine != seg:
                    raise ValueError(f"p=0 BPE differs from the tokenizer on {piece!r}: {mine} vs {seg}")
                types.append(init)
                canon.append(seg)
                rel.append(_rel_words(bytes(bpe.char_to_byte[c] for c in piece).decode("utf-8")))
            elif canon[t] != seg:
                raise ValueError(f"pre-token {piece!r} has two canonical segmentations")
            occ_type.append(t)
            anchor.append(wds_l[j0])
            pos = pos_end
        if j != len(ids):
            raise ValueError(f"doc {d}: {len(ids) - j} canonical tokens not covered by pre-tokens")
        if d % 20000 == 0 and d:
            log(f"[bpe_dropout] table: {d:,}/{len(bos):,} docs, {len(types):,} types, {time.time() - t0:.0f}s")
    occ_type_a = np.asarray(occ_type, dtype=np.int32)
    anchor_a = np.asarray(anchor, dtype=np.int64)
    if len(word_strings) != split.n_words:
        raise ValueError(f"decoded {len(word_strings)} words, split has {split.n_words}")
    cnt = Counter(word_strings)
    wc = np.asarray([cnt[w] for w in word_strings] + [-1], dtype=np.int64)  # index n_words: past the end
    word_count = np.where(occ_type_a == BOS_TYPE, -1, wc[np.minimum(anchor_a, split.n_words)])
    tab = Table(occ_type_a, anchor_a, word_count, types, canon, rel, N, split.n_words)
    tk, wd = render(tab, 0, len(tab), [0.0], None, bpe)
    if not (np.array_equal(tk, tokens) and np.array_equal(wd, words)):
        raise ValueError("p=0 rebuild of the stream differs from the stored tokens/words")
    log(f"[bpe_dropout] table: {len(tab):,} occurrences, {len(types):,} types, rebuild exact, {time.time() - t0:.0f}s")
    return tab


# ------------------------------------------------------------------------------------------------
# rendering one epoch
# ------------------------------------------------------------------------------------------------
_BPE: BPE | None = None
_TAB: Table | None = None


def eligible(tab: Table, scope: str, rare_max: int) -> np.ndarray:
    wc = tab.word_count
    if scope == "all":
        return wc >= 0
    if scope == "rare":
        return (wc >= 0) & (wc <= rare_max)
    if scope == "frequent":
        return wc > rare_max
    raise ValueError(f"scope {scope!r}")


def render(tab: Table, lo: int, hi: int, p_and_seed: list, elig: np.ndarray | None,
           bpe: BPE | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tokens and word ids of occurrences [lo, hi). p_and_seed = [p] (canonical) or [p, seed, epoch]."""
    p = float(p_and_seed[0])
    bpe = bpe or _BPE
    ot = tab.occ_type[lo:hi].tolist()
    an = tab.anchor[lo:hi + 1].tolist()
    if hi >= len(tab):
        an.append(tab.n_words)  # past the end: whitespace after the last word
    el = None if (p == 0.0 or elig is None) else elig[lo:hi].tolist()
    if el is not None:
        seed, epoch = int(p_and_seed[1]), int(p_and_seed[2])
        rng = np.random.Generator(np.random.PCG64([seed, epoch, lo // CHUNK, SEED_TAG]))
        u = rng.random(64 * (hi - lo) + 1024).tolist()
    toks: list[int] = []
    wds: list[int] = []
    k = 0
    tb = bpe.tok_bytes
    for i, t in enumerate(ot):
        if el is not None and el[i]:
            need = len(tab.types[t]) ** 2 + 16  # a pre-token of L bytes draws at most (L-1)^2 times
            if k + need > len(u):
                u = u[k:] + rng.random(64 * (hi - lo - i) + need + 1024).tolist()
                k = 0
            seg, k = bpe.segment(tab.types[t], p, u, k)
        else:
            seg = tab.canon[t]
        toks.extend(seg)
        if t == BOS_TYPE:
            wds.append(an[i])
            continue
        r = tab.rel[t]
        a, nx = an[i], an[i + 1]
        b = 0
        for tid in seg:
            rr = int(r[b])
            wds.append(a + rr if rr >= 0 else nx)
            b += int(tb[tid])
    return np.asarray(toks, dtype=np.uint16), np.asarray(wds, dtype=np.uint32)


def _init_worker(tab_path: str, tok_path: str) -> None:
    global _BPE, _TAB
    _BPE = BPE(tok_path)
    _TAB = load_table(Path(tab_path))


def _work(job):
    lo, hi, pse, scope, rare_max = job
    el = eligible(_TAB, scope, rare_max)
    return lo, render(_TAB, lo, hi, pse, el)


# ------------------------------------------------------------------------------------------------
# cache
# ------------------------------------------------------------------------------------------------
def cache_dir(data_dir) -> Path:
    return C.bin_dir(data_dir) / "bpedrop"


def table_path(data_dir, split: C.SplitData) -> Path:
    return cache_dir(data_dir) / f"table_{split.name}_{split.meta['sha256'][:16]}.npz"


def save_table(tab: Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lens = np.asarray([len(t) for t in tab.types], dtype=np.int64)
    clens = np.asarray([len(t) for t in tab.canon], dtype=np.int64)
    rlens = np.asarray([len(r) for r in tab.rel], dtype=np.int64)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, occ_type=tab.occ_type, anchor=tab.anchor, word_count=tab.word_count,
             types=np.concatenate([np.asarray(t, dtype=np.int32) for t in tab.types]), types_len=lens,
             canon=np.concatenate([np.asarray(t, dtype=np.int32) for t in tab.canon]), canon_len=clens,
             rel=np.concatenate(tab.rel).astype(np.int16), rel_len=rlens,
             n=np.asarray([tab.n_tokens, tab.n_words, FORMAT_VERSION], dtype=np.int64))
    os.replace(tmp, path)


def load_table(path: Path) -> Table:
    z = np.load(path)
    if int(z["n"][2]) != FORMAT_VERSION:
        raise ValueError(f"{path}: table format {int(z['n'][2])} != {FORMAT_VERSION}")

    def split(a, lens, conv):
        o = np.concatenate([[0], np.cumsum(lens)])
        return [conv(a[o[i]:o[i + 1]]) for i in range(len(lens))]
    to_t = lambda x: tuple(int(v) for v in x)  # noqa: E731
    return Table(z["occ_type"], z["anchor"], z["word_count"], split(z["types"], z["types_len"], to_t),
                 split(z["canon"], z["canon_len"], to_t), split(z["rel"], z["rel_len"], np.asarray),
                 int(z["n"][0]), int(z["n"][1]))


def get_table(split: C.SplitData, data_dir, log=print) -> tuple[Table, Path]:
    path = table_path(data_dir, split)
    if path.exists():
        return load_table(path), path
    tab = build_table(split, BPE(), log)
    save_table(tab, path)
    return tab, path


def stream_name(split: C.SplitData, p: float, scope: str, rare_max: int, seed: int, epoch: int) -> str:
    sc = scope if scope == "all" else f"{scope}{rare_max}"
    return f"{split.name}_{split.meta['sha256'][:12]}_{sc}_p{p:g}_s{seed}_e{epoch}"


def build_streams(split: C.SplitData, data_dir, plan: list[float], scope: str, rare_max: int, seed: int,
                  workers: int | None = None, log=print) -> list[dict]:
    """Make (or reuse) the cached stream of every epoch with p > 0. Returns per-epoch meta (None for p = 0)."""
    out: list[dict | None] = []
    todo = []
    d = cache_dir(data_dir)
    for e, p in enumerate(plan):
        if p == 0.0:
            out.append(None)
            continue
        name = stream_name(split, p, scope, rare_max, seed, e)
        mp_ = d / f"{name}.meta.json"
        if mp_.exists():
            out.append(json.loads(mp_.read_text()))
        else:
            out.append(None)
            todo.append((e, p, name))
    if not todo:
        return out
    tab, tpath = get_table(split, data_dir, log)
    n = len(tab)
    workers = workers or max(1, min(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                                    else os.cpu_count() or 1, 32))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(tpath), str(C.TOKENIZER_PATH))) as pool:
        for e, p, name in todo:
            t0 = time.time()
            jobs = [(lo, min(n, lo + CHUNK), [p, seed, e], scope, rare_max) for lo in range(0, n, CHUNK)]
            parts = dict(pool.imap_unordered(_work, jobs))
            tk = np.concatenate([parts[lo][0] for lo, *_ in jobs])
            wd = np.concatenate([parts[lo][1] for lo, *_ in jobs])
            meta = write_stream(d, name, tk, wd, split, dict(p=p, scope=scope, rare_max=rare_max, seed=seed, epoch=e,
                                                             build_s=time.time() - t0, workers=workers))
            out[e] = meta
            log(f"[bpe_dropout] epoch {e} p={p:g} {scope}: {meta['tokens']:,} tokens "
                f"(x{meta['inflation']:.4f} canonical), {time.time() - t0:.0f}s with {workers} workers")
    return out


def write_stream(d: Path, name: str, tk: np.ndarray, wd: np.ndarray, split: C.SplitData, info: dict) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    winit = int(C.word_initial(wd)[1:].sum())
    canon_winit = int(C.word_initial(np.asarray(split.words))[1:].sum())
    if winit != canon_winit or int(wd[-1]) != int(split.words[-1]) or tk[0] != C.BOS_ID:
        raise ValueError(f"{name}: word accounting differs from the canonical stream ({winit} vs {canon_winit})")
    for suf, arr in ((".bin", tk.astype("<u2")), (".words.bin", wd.astype("<u4"))):
        tmp = d / f"{name}{suf}.tmp"
        arr.tofile(tmp)
        os.replace(tmp, d / f"{name}{suf}")
    meta = {"name": name, "tokens": int(len(tk)), "words": split.n_words, "word_initial_targets": winit,
            "canonical_tokens": split.n_tokens, "inflation": len(tk) / split.n_tokens,
            "changed_frac": None, "sha256": C.sha256_file(d / f"{name}.bin"),
            "sha256_words": C.sha256_file(d / f"{name}.words.bin"), "canonical_sha256": split.meta["sha256"],
            "tokenizer_sha256": split.meta["tokenizer_sha256"], "format_version": FORMAT_VERSION, **info}
    (d / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_stream(data_dir, name: str) -> tuple[np.ndarray, np.ndarray]:
    d = cache_dir(data_dir)
    return (np.memmap(d / f"{name}.bin", dtype="<u2", mode="r"),
            np.memmap(d / f"{name}.words.bin", dtype="<u4", mode="r"))


# ------------------------------------------------------------------------------------------------
# sampler
# ------------------------------------------------------------------------------------------------
class MultiStreamSampler:
    """EpochSampler over a different token stream per epoch (see the module docstring)."""

    def __init__(self, streams: list[tuple[np.ndarray, np.ndarray]], canonical: tuple[np.ndarray, np.ndarray],
                 seq_len: int, data_seed: int):
        self.streams, self.canonical = streams, canonical
        self.T, self.seed = int(seq_len), int(data_seed)
        self.epoch, self.idx = 0, 0
        self._plan_epoch = -1
        self._starts = np.zeros(0, dtype=np.int64)
        self._winit: dict[int, np.ndarray] = {}

    def stream(self, e: int) -> tuple[np.ndarray, np.ndarray]:
        return self.streams[e] if e < len(self.streams) else self.canonical

    def word_init(self, e: int) -> np.ndarray:
        key = e if e < len(self.streams) else -1
        if key not in self._winit:
            if len(self._winit) > 2:
                self._winit.clear()
            self._winit[key] = C.word_initial(np.asarray(self.stream(e)[1]))
        return self._winit[key]

    def plan(self, epoch: int) -> np.ndarray:
        N = len(self.stream(epoch)[0])
        rng = np.random.Generator(np.random.PCG64([self.seed, epoch]))
        off = int(rng.integers(0, self.T))
        n = (N - 1 - off) // self.T
        return off + rng.permutation(n).astype(np.int64) * self.T

    def _ensure(self) -> None:
        if self._plan_epoch != self.epoch:
            self._starts = self.plan(self.epoch)
            self._plan_epoch = self.epoch

    def next_batch(self, n: int):
        """(x, y, word-initial targets) of the next n windows."""
        from babylm.harness.stream import gather_windows, words_in_targets
        xs, ys, nw = [], [], 0
        self.last_rows = []  # [(epoch, starts)] of the rows just returned (babylm.harness.stack: A1 cache keys)
        while n > 0:
            self._ensure()
            take = min(n, len(self._starts) - self.idx)
            st = self._starts[self.idx:self.idx + take]
            self.last_rows.append((self.epoch, st))
            tok, _ = self.stream(self.epoch)
            x, y = gather_windows(tok, st, self.T)
            xs.append(x)
            ys.append(y)
            nw += words_in_targets(self.word_init(self.epoch), st, self.T)
            self.idx += take
            n -= take
            if self.idx >= len(self._starts):
                self.epoch += 1
                self.idx = 0
        import torch
        return (xs[0] if len(xs) == 1 else torch.cat(xs)), (ys[0] if len(ys) == 1 else torch.cat(ys)), nw

    def epoch_float(self) -> float:
        self._ensure()
        return self.epoch + self.idx / max(1, len(self._starts))

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "idx": self.idx, "seed": self.seed, "T": self.T,
                "N": [len(s[0]) for s in self.streams]}

    def load_state_dict(self, sd: dict) -> None:
        if (sd["seed"], sd["T"], sd["N"]) != (self.seed, self.T, [len(s[0]) for s in self.streams]):
            raise ValueError("sampler state is for a different dropout plan / seed / seq_len")
        self.epoch, self.idx = int(sd["epoch"]), int(sd["idx"])
        self._plan_epoch = -1


def total_tokens(lengths: list[int]) -> int:
    return int(sum(lengths))


def steps_for(lengths: list[int], batch_tokens: int) -> int:
    return max(1, math.ceil(total_tokens(lengths) / batch_tokens))


# ------------------------------------------------------------------------------------------------
# calibration (token-inflation-matched null twin) and CLI
# ------------------------------------------------------------------------------------------------
def inflation(tab: Table, p: float, scope: str, rare_max: int, seed: int = 0, sample: int = 400_000) -> float:
    """Token inflation of one epoch, estimated on `sample` occurrences spread over the stream."""
    bpe = BPE()
    n = len(tab)
    el = eligible(tab, scope, rare_max)
    step = max(CHUNK, (n // max(1, sample // CHUNK)) // CHUNK * CHUNK)
    canon = drop = 0
    for lo in range(0, n, step):
        hi = min(n, lo + CHUNK)
        tk, _ = render(tab, lo, hi, [p, seed, 0], el, bpe)
        drop += len(tk)
        canon += sum(len(tab.canon[t]) for t in tab.occ_type[lo:hi].tolist())
    return drop / canon


def match_p(tab: Table, target: float, scope: str, rare_max: int, lo: float = 0.0, hi: float = 0.6,
            iters: int = 14) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        if inflation(tab, mid, scope, rare_max) < target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["table", "build", "inflation", "match"])
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split", default="train_10M")
    ap.add_argument("--plan", default="0.1x8,0x2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--scope", default="all", choices=["all", "rare", "frequent"])
    ap.add_argument("--rare_max", type=int, default=63)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--target_scope", default="all", help="match: scope whose inflation at --p is the target")
    a = ap.parse_args(argv)
    split = C.load_split(a.split, a.data_dir)
    if a.cmd == "table":
        tab, path = get_table(split, a.data_dir)
        print(json.dumps({"path": str(path), "occurrences": len(tab), "types": len(tab.types)}))
    elif a.cmd == "build":
        metas = build_streams(split, a.data_dir, parse_plan(a.plan, a.epochs), a.scope, a.rare_max, a.seed, a.workers)
        print(json.dumps([m and {k: m[k] for k in ("name", "tokens", "inflation", "sha256")} for m in metas], indent=1))
    elif a.cmd == "inflation":
        tab, _ = get_table(split, a.data_dir)
        print(json.dumps({"p": a.p, "scope": a.scope, "inflation": inflation(tab, a.p, a.scope, a.rare_max)}))
    else:
        tab, _ = get_table(split, a.data_dir)
        target = inflation(tab, a.p, a.target_scope, a.rare_max)
        p = match_p(tab, target, a.scope, a.rare_max)
        print(json.dumps({"target_scope": a.target_scope, "target_p": a.p, "target_inflation": target,
                          "scope": a.scope, "matched_p": p, "inflation": inflation(tab, p, a.scope, a.rare_max)}))


if __name__ == "__main__":
    sys.exit(main())
