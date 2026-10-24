"""Sprint-1 data: pre-tokenized GPT-2 FineWeb shards (the speedrun's) or a deterministic synthetic corpus.

Shard format (modded-nanogpt / kjj0/fineweb10B-gpt2, produced by data/fineweb.py upstream):
  bytes [0, 1024): header, 256 little-endian int32:  header[0] = 20240520 (magic), header[1] = 1 (version),
                   header[2] = number of tokens N; the rest is zero.
  bytes [1024, 1024 + 2N): N little-endian uint16 GPT-2 token ids. Documents are concatenated, each one
                   preceded by <|endoftext|> (id 50256).
Files: fineweb_val_000000.bin (100M tokens, the speedrun's val shard) and fineweb_train_%06d.bin (i >= 1,
100M tokens each).

Splits (deterministic, no randomness): the val shard is cut at N // 2:
  DEV   = tokens [0, N // 2)       used for every number seen during the sprint (curves, selection)
  FINAL = tokens [N // 2, N)       touched only with train.py --final (prints a loud warning)
TRAIN = the train shards (never overlap the val shard).

Bytes for bits-per-byte: GPT-2 BPE is byte-level, so the UTF-8 byte length of a token stream is the sum of
per-token byte lengths. The table len(enc.decode_single_token_bytes(i)) for every id is computed once with
tiktoken and cached beside the shards as gpt2_token_bytes.npy. Conventions: <|endoftext|> counts as 1 byte
(document separator), padded ids >= 50257 count 0 (never targets).

Usage on the VM:  python -m sprint1.harness.data --data_dir data/fineweb10B --train_shards 4
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MAGIC, VERSION, HEADER_INTS = 20240520, 1, 256
EOT = 50256
HF_REPO = "kjj0/fineweb10B-gpt2"
EDU_REPO = "kjj0/finewebedu10B-gpt2"  # FineWeb-Edu, same GPT-2 tokenizer (modded-nanogpt cached_finewebedu10B.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("HINTON_DATA_DIR", REPO_ROOT / "data" / "fineweb10B"))
VAL_NAME = "fineweb_val_000000.bin"


def train_name(i: int) -> str:
    return f"fineweb_train_{i:06d}.bin"


def load_shard(path: str | Path) -> np.ndarray:
    """Memory-map a shard's tokens (uint16) after validating the header."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or header[0] != MAGIC or header[1] != VERSION:
        raise ValueError(f"{path}: bad shard header {header[:3].tolist()} (want magic {MAGIC}, version {VERSION})")
    n = int(header[2])
    toks = np.memmap(path, dtype="<u2", mode="r", offset=HEADER_INTS * 4, shape=(n,))
    return toks


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0], header[1], header[2] = MAGIC, VERSION, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype="<u2").tobytes())


def download(data_dir: str | Path = DEFAULT_DATA_DIR, train_shards: int = 1) -> None:
    """Fetch the val shard and the first `train_shards` train shards from the HF hub (VM only)."""
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    names = [VAL_NAME] + [train_name(i) for i in range(1, train_shards + 1)]
    for name in names:
        if (data_dir / name).exists():
            continue
        print(f"[data] downloading {name}", flush=True)
        hf_hub_download(repo_id=HF_REPO, filename=name, repo_type="dataset", local_dir=str(data_dir))
    token_bytes_table(data_dir)


def download_edu(edu_dir: str | Path, train_shards: int) -> None:
    """FineWeb-Edu train shards only (finewebedu_train_%06d.bin, i >= 1); DEV always stays FineWeb's val shard."""
    from huggingface_hub import hf_hub_download

    edu_dir = Path(edu_dir)
    edu_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, train_shards + 1):
        name = f"finewebedu_train_{i:06d}.bin"
        if not (edu_dir / name).exists():
            print(f"[data] downloading {name}", flush=True)
            hf_hub_download(repo_id=EDU_REPO, filename=name, repo_type="dataset", local_dir=str(edu_dir))


def token_bytes_table(data_dir: str | Path, vocab_size: int = 50304) -> np.ndarray:
    """Per-token UTF-8 byte length (int32), cached as <data_dir>/gpt2_token_bytes.npy."""
    path = Path(data_dir) / "gpt2_token_bytes.npy"
    if path.exists():
        t = np.load(path)
    else:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        t = np.zeros(enc.n_vocab, dtype=np.int32)
        for i in range(enc.n_vocab):
            t[i] = 1 if i == EOT else len(enc.decode_single_token_bytes(i))
        np.save(path, t)
    if len(t) < vocab_size:
        t = np.concatenate([t, np.zeros(vocab_size - len(t), dtype=np.int32)])
    return t


# ------------------------------------------------------------------------------------------------
# Synthetic corpus (no network): deterministic, learnable, with a real new/seen mix
# ------------------------------------------------------------------------------------------------
SYN_VOCAB = 1000  # ids [1, 1000); id 0 is the document separator


def synthetic_tokens(n_tokens: int, seed: int) -> np.ndarray:
    """Documents of 64..512 tokens. Each token is (p=0.35) a copy of a token from the doc's own 12-token
    topic set, else the next state of a sparse first-order Markov chain (4 successors per token)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    succ = np.random.Generator(np.random.PCG64(7)).integers(1, SYN_VOCAB, size=(SYN_VOCAB, 4))  # fixed grammar
    out = []
    total = 0
    while total < n_tokens:
        n_docs = 256
        lens = rng.integers(64, 513, size=n_docs)
        L = int(lens.max())
        topics = rng.integers(1, SYN_VOCAB, size=(n_docs, 12))
        cur = rng.integers(1, SYN_VOCAB, size=n_docs)
        docs = np.zeros((n_docs, L), dtype=np.int64)
        u = rng.random((n_docs, L))
        pick_succ = rng.integers(0, 4, size=(n_docs, L))
        pick_topic = rng.integers(0, 12, size=(n_docs, L))
        for t in range(L):
            nxt = succ[cur, pick_succ[:, t]]
            top = topics[np.arange(n_docs), pick_topic[:, t]]
            cur = np.where(u[:, t] < 0.35, top, nxt)
            docs[:, t] = cur
        for i in range(n_docs):
            out.append(np.concatenate([[0], docs[i, : lens[i]]]))
            total += lens[i] + 1
    return np.concatenate(out)[:n_tokens].astype(np.uint16)


def synthetic_bytes_table(vocab_size: int) -> np.ndarray:
    ids = np.arange(vocab_size, dtype=np.int64)
    t = (1 + (ids * 7919) % 6).astype(np.int32)  # 1..6 bytes, fixed; bpb on synthetic data is a plumbing check only
    t[0] = 1
    return t


# ------------------------------------------------------------------------------------------------
# Splits and loaders
# ------------------------------------------------------------------------------------------------
@dataclass
class Corpus:
    train: list[np.ndarray]
    dev: np.ndarray
    final: np.ndarray
    token_bytes: np.ndarray  # int32 [vocab]
    source: str
    train_names: list[str] = field(default_factory=list)          # shard file names, in TrainSampler order
    train_file_bytes: list[int | None] = field(default_factory=list)  # file sizes (None: synthetic)

    def split(self, name: str) -> np.ndarray:
        return {"dev": self.dev, "final": self.final}[name]

    def train_shards_used(self) -> list[dict]:
        """The exact train pool (B11): [{"name", "bytes" (file size), "tokens"}] in sampler order."""
        names = self.train_names or [f"train_{i}" for i in range(len(self.train))]
        sizes = self.train_file_bytes or [None] * len(self.train)
        return [{"name": n, "bytes": b, "tokens": int(len(t))} for n, b, t in zip(names, sizes, self.train)]


def resolve_data_dir(data_dir: str | Path | None) -> Path:
    return Path(data_dir or DEFAULT_DATA_DIR)


def train_shard_names(data_dir: str | Path | None, n: int | None = None) -> list[str]:
    """Sorted fineweb_train_*.bin names in data_dir; with n, exactly the first n (error if fewer exist)."""
    d = resolve_data_dir(data_dir)
    names = sorted(p.name for p in d.glob("*_train_*.bin"))
    if n is not None:
        if n < 1:
            raise ValueError(f"train_shards must be >= 1, got {n}")
        if len(names) < n:
            raise FileNotFoundError(f"train_shards={n} but only {len(names)} train shards in {d}: {names}; "
                                    f"run python -m sprint1.harness.data --train_shards {n}")
        names = names[:n]
    return names


def load_corpus(data_dir: str | Path | None = None, synthetic: bool = False, vocab_size: int = 50304,
                train_shards: int | None = None, train_data_dir: str | Path | None = None) -> Corpus:
    """train_shards=N pins the train pool to the first N shards in sorted order (fewer present: error);
    None = every shard present (train.py always passes an explicit N and records it)."""
    if synthetic:
        if vocab_size < SYN_VOCAB:
            raise ValueError(f"--synthetic needs vocab_size >= {SYN_VOCAB}")
        train = synthetic_tokens(1 << 18, seed=1)
        val = synthetic_tokens(1 << 16, seed=2)
        h = len(val) // 2
        return Corpus([train], val[:h], val[h:], synthetic_bytes_table(vocab_size), "synthetic",
                      ["synthetic(seed=1)"], [None])
    data_dir = resolve_data_dir(data_dir)
    val = load_shard(data_dir / VAL_NAME)
    tdir = Path(train_data_dir) if train_data_dir else data_dir  # train pool may come from another corpus (Edu)
    names = train_shard_names(tdir, train_shards)
    if not names:
        raise FileNotFoundError(f"no train shards in {tdir}; run python -m sprint1.harness.data --train_shards N")
    h = len(val) // 2
    return Corpus([load_shard(tdir / n) for n in names], val[:h], val[h:],
                  token_bytes_table(data_dir, vocab_size), str(tdir), list(names),
                  [os.path.getsize(tdir / n) for n in names])


def prefix_pool(shards: list[np.ndarray], limit: int, seq_len: int,
                drop_short: bool = True) -> tuple[list[np.ndarray], int]:
    """train.py --train_tokens_limit: the train pool cut to the first `limit` tokens of the concatenation of the
    shards (in the given = sorted order), as views (no copy). With drop_short (TrainSampler, which never reads across
    shards) a tail piece too short to hold one window (< seq_len + 2 tokens) is dropped, so the pool can end a few
    tokens before `limit`; EpochSampler reads the concatenation and keeps every piece. Returns (shards, tokens
    kept). More than the shards hold: error. Neither sampler then reads past the cut."""
    if limit < 1:
        raise ValueError(f"train_tokens_limit must be >= 1, got {limit}")
    total = int(sum(len(s) for s in shards))
    if limit > total:
        raise ValueError(f"train_tokens_limit={limit:,} but the train shards hold only {total:,} tokens")
    out: list[np.ndarray] = []
    left = limit
    for s in shards:
        if left <= 0:
            break
        piece = s[:left] if len(s) > left else s
        left -= len(piece)
        if len(piece) >= seq_len + 2 or (not drop_short and len(piece)):
            out.append(piece)
    if not out:
        raise ValueError(f"train_tokens_limit={limit:,} is shorter than one window (seq_len {seq_len} + 2)")
    return out, int(sum(len(s) for s in out))


class TrainSampler:
    """Uniformly random fixed windows of seq_len + 1 tokens from the train shards, seeded.

    The same seed gives the same batches for every model (paired comparisons). State is checkpointable."""

    def __init__(self, shards: list[np.ndarray], seq_len: int, seed: int):
        self.shards, self.T = shards, seq_len
        self.sizes = np.array([len(s) - seq_len - 1 for s in shards], dtype=np.int64)
        assert (self.sizes > 0).all(), "shard shorter than one window"
        self.cum = np.cumsum(self.sizes)
        self.rng = np.random.Generator(np.random.PCG64(seed + 1_000_003))

    def next(self, n_seqs: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.rng.integers(0, int(self.cum[-1]), size=n_seqs)
        shard_i = np.searchsorted(self.cum, pos, side="right")
        starts = pos - np.concatenate([[0], self.cum[:-1]])[shard_i]
        buf = np.stack([np.asarray(self.shards[s][o:o + self.T + 1], dtype=np.int64) for s, o in zip(shard_i, starts)])
        t = torch.from_numpy(buf)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, sd: dict) -> None:
        self.rng.bit_generator.state = sd["rng"]


def epoch_pool_tokens(budget_tokens: int, batch_tokens: int = 65536) -> int:
    """--sampler epoch convention: the --train_tokens_limit T for a budget of D tokens. train.py runs
    ceil(D / batch_tokens) steps, i.e. D' = steps x batch_tokens tokens; T = D' + 1 holds exactly D' / seq_len
    windows (stride seq_len, one shared boundary token each), so every window is used once and rho = D' / T ~ 1."""
    steps = max(1, math.ceil(budget_tokens / batch_tokens))
    return steps * batch_tokens + 1


class EpochSampler:
    """--sampler epoch: one pass WITHOUT replacement. The pool (the shards concatenated in order, e.g. the
    --train_tokens_limit prefix) of P tokens is cut into n = (P - 1) // seq_len windows of seq_len + 1 tokens at stride
    seq_len (consecutive windows share one boundary token; a window may straddle two shards). A seeded permutation
    (PCG64(seed + 1_000_003), TrainSampler's convention) fixes their order and next() hands them out in that order,
    each exactly once; asking for more than n windows is an error (never wraps). State: seed, n and the cursor."""

    def __init__(self, shards: list[np.ndarray], seq_len: int, seed: int):
        self.shards, self.T, self.seed = shards, seq_len, seed
        lens = np.array([len(s) for s in shards], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(lens)])  # piece k holds pool tokens [starts[k], starts[k+1])
        self.pool_tokens = int(self.starts[-1])
        self.n = (self.pool_tokens - 1) // seq_len
        assert self.n > 0, "pool shorter than one window"
        self.perm = np.random.Generator(np.random.PCG64(seed + 1_000_003)).permutation(self.n)
        self.cursor = 0

    def window(self, w: int) -> np.ndarray:
        """Pool tokens [w * T, w * T + T + 1) as int64 (crossing pieces if needed)."""
        a, b = int(w) * self.T, int(w) * self.T + self.T + 1
        k = int(np.searchsorted(self.starts, a, side="right")) - 1
        parts = []
        while a < b:
            lo = a - int(self.starts[k])
            take = min(b - a, len(self.shards[k]) - lo)
            parts.append(np.asarray(self.shards[k][lo:lo + take], dtype=np.int64))
            a += take
            k += 1
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def next(self, n_seqs: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor + n_seqs > self.n:
            raise RuntimeError(f"--sampler epoch: {self.cursor + n_seqs:,} windows requested but the pool holds "
                               f"{self.n:,} ({self.pool_tokens:,} tokens); raise --train_tokens_limit (no wrap-around)")
        ids = self.perm[self.cursor:self.cursor + n_seqs]
        self.cursor += n_seqs
        t = torch.from_numpy(np.stack([self.window(w) for w in ids]))
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def state_dict(self) -> dict:
        return {"kind": "epoch", "seed": self.seed, "n": self.n, "cursor": self.cursor}

    def load_state_dict(self, sd: dict) -> None:
        if sd.get("kind") != "epoch" or sd["seed"] != self.seed or sd["n"] != self.n:
            raise ValueError(f"epoch sampler state {sd} does not match seed {self.seed} / n {self.n}")
        self.cursor = int(sd["cursor"])


def eval_n_windows(n_tokens: int, seq_len: int, max_tokens: int | None) -> int:
    """Number of eval windows of a slice (shared by eval_windows, eval_batches and slice_fingerprint)."""
    n_win = (n_tokens - 1) // seq_len
    if max_tokens is not None and max_tokens > 0:
        n_win = min(n_win, max(1, max_tokens // seq_len))
    return n_win


def eval_windows(tokens: np.ndarray, seq_len: int, max_tokens: int | None, batch: int):
    """Consecutive non-overlapping windows: window i predicts tokens[i*T + 1 : i*T + T + 1]; every token after
    the first is a target exactly once. Yields (idx, targets) int64 batches of consecutive windows, ragged last
    batch (the pre-interleave order; train.evaluate() uses eval_batches())."""
    n_win = eval_n_windows(len(tokens), seq_len, max_tokens)
    for b0 in range(0, n_win, batch):
        nb = min(batch, n_win - b0)
        buf = np.asarray(tokens[b0 * seq_len: (b0 + nb) * seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(buf[:-1].reshape(nb, seq_len))
        y = torch.from_numpy(buf[1:].reshape(nb, seq_len))
        yield x, y


@dataclass(frozen=True)
class EvalLayout:
    """Which DEV window each row of each eval batch holds (B4/B5). Interleaved: with n windows and batch B,
    n_batches = ceil(n / B) and row r of batch b holds window r * n_batches + b, so the rows of one batch are
    n_batches windows apart (never adjacent text when n_batches > 1). Every batch has exactly B rows; a row whose
    window index would be >= n is padding (index -1): its inputs copy row 0 of the batch, its targets are all -1.
    Padded rows: n_batches * B - n < B. interleave=False: row r of batch b = window b * B + r (padded tail)."""
    n_windows: int
    batch: int
    n_batches: int
    interleave: bool = True

    @property
    def padded_rows(self) -> int:
        return self.n_batches * self.batch - self.n_windows

    def windows(self, b: int) -> np.ndarray:
        r = np.arange(self.batch, dtype=np.int64)
        w = r * self.n_batches + b if self.interleave else b * self.batch + r
        return np.where(w < self.n_windows, w, -1)


def eval_layout(n_tokens: int, seq_len: int, max_tokens: int | None, batch: int, interleave: bool = True) -> EvalLayout:
    if batch < 1:
        raise ValueError(f"eval batch size must be >= 1, got {batch}")
    n_win = eval_n_windows(n_tokens, seq_len, max_tokens)
    return EvalLayout(n_win, batch, math.ceil(n_win / batch), interleave)


def eval_batches(tokens: np.ndarray, seq_len: int, max_tokens: int | None, batch: int, interleave: bool = True):
    """The windows of eval_windows() (same boundaries, same targets, each scored exactly once) in EvalLayout order.
    Yields (idx, targets, win): int64 [batch, seq_len] tensors and win int64 [batch] (window index, -1 = padding
    row: inputs = row 0's, targets all -1). One vectorised gather per batch."""
    lay = eval_layout(len(tokens), seq_len, max_tokens, batch, interleave)
    offs = np.arange(seq_len + 1, dtype=np.int64)
    for b in range(lay.n_batches):
        win = lay.windows(b)
        src = np.where(win >= 0, win, win[0])  # row 0 is always real (b < n_batches <= n_windows)
        buf = np.asarray(tokens[src[:, None] * seq_len + offs], dtype=np.int64)
        x = torch.from_numpy(buf[:, :-1].copy())
        y = torch.from_numpy(buf[:, 1:].copy())
        pad = torch.from_numpy(win < 0)
        if pad.any():
            y[pad] = -1
        yield x, y, win


def slice_fingerprint(tokens: np.ndarray, seq_len: int, max_tokens: int | None, split: str) -> dict:
    """Identity of an eval slice as eval_windows() reads it: sha256 over the token ids (int64) of the first and
    last window (seq_len + 1 tokens each) and the number of scored tokens. Equal fingerprints = same slice.
    Defined on DEV order (window 0 and window n-1), independent of the eval batch layout."""
    n_win = eval_n_windows(len(tokens), seq_len, max_tokens)
    first = np.asarray(tokens[0: seq_len + 1], dtype=np.int64)
    lo = (n_win - 1) * seq_len
    last = np.asarray(tokens[lo: lo + seq_len + 1], dtype=np.int64)
    n_tokens = n_win * seq_len
    h = hashlib.sha256()
    h.update(first.tobytes())
    h.update(last.tobytes())
    h.update(str(n_tokens).encode())
    return {"split": split, "seq_len": seq_len, "n_windows": n_win, "n_tokens": n_tokens, "sha256": h.hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description="download FineWeb GPT-2 shards for sprint-1")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--train_shards", type=int, default=1, help="100M tokens each")
    ap.add_argument("--edu_dir", default=None)
    ap.add_argument("--edu_shards", type=int, default=0)
    a = ap.parse_args()
    download(a.data_dir, a.train_shards)
    if a.edu_shards and a.edu_dir:
        download_edu(a.edu_dir, a.edu_shards)
    c = load_corpus(a.data_dir)
    print(f"[data] train {sum(len(s) for s in c.train):,} tokens in {len(c.train)} shards; "
          f"dev {len(c.dev):,}; final {len(c.final):,}; bytes/token(dev[:1M]) "
          f"{c.token_bytes[np.asarray(c.dev[:1_000_000], dtype=np.int64)].mean():.3f}")


if __name__ == "__main__":
    main()
