"""kNN-LM diagnostic for registered prediction P7 (critic log 2026-10-09 18:56Z): "kNN over a train_10M datastore
improves the final 10M model by >= 0.03 (DEV nats/token), mostly on words with train_10M count 11-1000". A
diagnostic label for one trained harness checkpoint, not an attack.

    python -m babylm.analysis.knn_diag --ckpt RUN/checkpoints/chck_100M.pt --out OUT [--data_dir data/babylm]
        [--train_split train_10M] [--eval_split dev_sub] [--ctx 512] [--k 1024] [--ks 16,64,256,1024]
        [--temps 1,3,10] [--lambdas 0,0.05,...,0.5] [--lm_temps 0.8,...,8] [--max_train_tokens N]
        [--max_eval_tokens N] [--device auto] [--key_dtype fp16] [--gpu_mem_gb 30]
    python -m babylm.analysis.p2_slices --runs lm=OUT:lm knn=OUT:knn lmtemp=OUT:lmtemp knntemp=OUT:knntemp \
        --pairs lm-knn lm-lmtemp lmtemp-knntemp --out OUT/slices       (summary.json "slicer_cmd" has variants)

Protocol
- Model: the checkpoint rebuilt as babylm/harness/export_hf.py does (ModelConfig.from_dict(ck["cfg"]); the module
  ck["model_name"] via babylm.harness.train.load_model_module, so baseline and baseline_reg both load), eval mode,
  bf16 autocast on CUDA and fp32 elsewhere (as the harness's evaluate_stream). Key = the input of model.lm_head
  (the final RMS-normalised hidden state), captured by a forward hook; LM log-prob = -aux["per_token_loss"] of the
  model's own forward (fp32 softcapped logits).
- Streams: babylm.data.corpus.load_split (harness stream order, BOS 0 before every document). Windows as in
  babylm/eval/dev_bpw.py: ctx input tokens, stride s = ctx // 2; window j reads inputs [j s, j s + ctx) and their
  next tokens; window 0 scores all its positions, later windows only their last s (>= ctx - s tokens of context),
  so every input position p (target p + 1) is scored exactly once. No BOS is inserted: windows cross documents
  as in harness training. Scored position p -> key h_p (--key_dtype), value = token p + 1, lm_logp.
- Datastore = every scored position of --train_split, queries = every scored position of --eval_split.
  --max_train_tokens / --max_eval_tokens keep a prefix of whole documents (eval: at least 2 documents).
- Exact kNN: squared L2 = |q|^2 - 2 q.k + |k|^2 by chunked matmul on the device with a running top-k per query
  across key chunks. The matmul multiplies fp32 copies of the 16-bit vectors (TF32 on CUDA): products of
  fp16/bf16 values are exact in TF32 and accumulation is fp32, i.e. an fp16 matmul with fp32 accumulate and an
  fp32 result. |k|^2 rides along as three extra columns (two bf16-exact parts plus a tiny remainder).
- p_knn(y) = sum of softmax(-d / T) over the k' nearest neighbours whose value is y (k' in --ks: prefixes of the
  sorted top-k list); p = lam p_knn + (1 - lam) p_lm, computed as logaddexp(log lam + log p_knn,
  log(1 - lam) + log p_lm), so lam = 0 returns the LM log-prob bit for bit.
- Recalibration control: an overconfident LM gains from any flattening, so the same held-out tokens also get
  (a) temp-only: log softmax(z / tau)[y] of the model's own logits z = model.logits(head input), tau in
  --lm_temps picked on the even documents (plus the tau oracle on the held-out tokens themselves), and
  (b) knn_over_temp: the kNN interpolation on top of the tau-scaled LM (tau fixed; T, lam, k' picked on even docs).
  gain_knn_beyond_temp = temp-only minus knn_over_temp: the part of the kNN gain recalibration cannot buy.
- Selection: every parameter is picked on the targets of the even-indexed eval documents (a BOS target belongs
  to the document it opens, as in babylm.analysis.perword); the headline is the odd-indexed documents (held
  out). Oracles (picked on the tokens they are scored on) are reported and labelled as in-sample.
Outputs in OUT: summary.json; grid.csv (nats/token per base LM, T, k', lam on even / odd / all documents);
lm_temp.csv (per tau); pt/pt_<tag>_<label>.npy (float16 per-token NLL aligned with the eval split's token
stream, NaN where unscored: token 0 and anything cut by --max_eval_tokens), label lm, knn (chosen T, lam, k'),
lmtemp (chosen tau), knntemp (kNN over lmtemp); pw/pw_<tag>_<label>.npy per-word nats via
babylm.analysis.perword.per_word (words of cut documents are NaN, so the slicer skips them), pw_<tag>_<label>_odd.npy
(held-out documents only) and pw/index.json. <tag> = dev for dev_sub, devfull for dev (slicer --subset dev).
The test split is never read (load_split refuses it without final=True).
Memory: the datastore lives on the device (16M keys x 384 x fp16 = 12 GB); chunk sizes keep the allocator's peak
under --gpu_mem_gb (default 30 GiB) on CUDA; an OOM halves the key chunk and redoes the query chunk.
MPS is untested.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from babylm.analysis.perword import per_word
from babylm.data import corpus as C
from babylm.harness import train as TR
from sprint1.harness.model_api import ModelConfig

DEFAULT_TEMPS = "1,3,10"
DEFAULT_LAMBDAS = ",".join(f"{0.05 * i:g}" for i in range(11))
DEFAULT_LM_TEMPS = "0.8,0.9,1,1.1,1.2,1.3,1.5,1.7,2,2.5,3,4,5,6,8"
KEY_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
PW_TAG = {"dev_sub": "dev", "dev": "devfull"}
P7_GAIN = 0.03
LOG_EVERY_S = 30.0
MAX_SCORE_BYTES = 4 << 30   # one [query chunk, key chunk] fp32 distance block on CUDA


def log(msg: str) -> None:
    print(f"[knn] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------
# streams and windows
# ------------------------------------------------------------------------------------------------
def window_plan(n_pos: int, ctx: int) -> np.ndarray:
    """Windows over input positions 0..n_pos-1 (the target of p is p + 1): int64 rows (lo, hi, s0) = inputs
    [lo, hi), scored [s0, hi). Stride ctx // 2; window 0 scores everything, window j >= 1 its last ctx // 2
    positions, so each position is scored exactly once (with >= ctx - ctx // 2 tokens of context after window 0)."""
    if ctx < 2:
        raise ValueError(f"ctx must be >= 2, got {ctx}")
    s = ctx // 2
    rows = []
    j = 0
    while n_pos > 0:
        lo = j * s
        rows.append((lo, min(lo + ctx, n_pos), lo if j == 0 else lo + ctx - s))
        if lo + ctx >= n_pos:
            break
        j += 1
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def doc_prefix(tokens: np.ndarray, max_tokens: int | None, min_docs: int = 1) -> int:
    """Length of the longest prefix of whole documents with <= max_tokens tokens (but at least min_docs docs)."""
    n = len(tokens)
    if max_tokens is None or max_tokens >= n:
        return n
    starts = np.flatnonzero(np.asarray(tokens) == C.BOS_ID)
    if len(starts) == 0 or starts[0] != 0:
        raise ValueError("token stream does not start with BOS")
    ends = np.append(starts[1:], n)
    k = min(len(ends), max(int((ends <= max_tokens).sum()), min_docs))
    return int(ends[k - 1])


def target_docs(tokens: np.ndarray, n_tok: int) -> np.ndarray:
    """Document index of every target 1..n_tok-1 (a BOS target opens its document)."""
    starts = np.flatnonzero(np.asarray(tokens[:n_tok]) == C.BOS_ID)
    return np.searchsorted(starts, np.arange(1, n_tok), side="right") - 1


# ------------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------------
class HeadTap:
    """Forward hook on the LM head: keeps the head's input (the final normalised hidden state)."""

    def __init__(self, head: torch.nn.Module):
        self.h: torch.Tensor | None = None
        self.calls = 0
        self.active = True
        self.handle = head.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out) -> None:
        if self.active:
            self.h = inp[0]
            self.calls += 1

    @contextlib.contextmanager
    def muted(self) -> Iterator[None]:
        self.active = False
        try:
            yield
        finally:
            self.active = True

    def take(self) -> torch.Tensor:
        if self.calls != 1 or self.h is None:
            raise RuntimeError(f"the LM head ran {self.calls} times in one forward (expected 1)")
        h, self.h, self.calls = self.h, None, 0
        return h


def load_harness_model(ckpt: Path, device: torch.device) -> tuple[torch.nn.Module, ModelConfig, dict[str, Any]]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"])
    name = ck.get("model_name", "baseline")
    model = TR.load_model_module(name).build(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    if not isinstance(getattr(model, "lm_head", None), torch.nn.Module):
        raise SystemExit(f"{ckpt}: model {name!r} has no lm_head module to hook")
    meta = {k: ck.get(k) for k in ("model_name", "label", "words_seen", "words_nominal", "tokens_seen", "step",
                                   "tokenizer_sha256", "git_commit")}
    meta["cfg"] = ck["cfg"]
    return model, cfg, meta


@torch.no_grad()
def encode_stream(model: torch.nn.Module, tap: HeadTap, tokens: np.ndarray, ctx: int, batch: int,
                  device: torch.device, amp, key_dtype: torch.dtype, key_device: torch.device, name: str,
                  want_lm: bool = True, lm_temps: list[float] | None = None
                  ) -> tuple[torch.Tensor, np.ndarray | None, np.ndarray | None, float | None]:
    """Run the window protocol over `tokens` (whole documents). Row p of the results = input position p (target
    p + 1): keys [n - 1, d] in key_dtype on key_device; LM log-probs float32 [n - 1] (want_lm); with lm_temps,
    log softmax(z / tau)[target] float32 [len(lm_temps), n - 1] from the model's own fp32 softcapped logits
    z = model.logits(head input), and the max |log softmax(z)[target] - (-per_token_loss)| as a check."""
    tok = np.asarray(tokens, dtype=np.int64)
    n_pos = len(tok) - 1
    plan = window_plan(n_pos, ctx)
    ar = np.arange(ctx, dtype=np.int64)
    keys: torch.Tensor | None = None
    lm = np.full(n_pos, np.nan, dtype=np.float32) if want_lm else None
    lmt = np.full((len(lm_temps), n_pos), np.nan, dtype=np.float32) if lm_temps else None
    check = 0.0 if lm_temps else None
    filled = 0
    t0 = last = time.time()
    for b0 in range(0, len(plan), batch):
        P = plan[b0:b0 + batch]
        ipos = P[:, :1] + ar
        valid = ipos < P[:, 1:2]
        scored = valid & (ipos >= P[:, 2:3])
        pos = ipos[scored]
        if pos[0] != filled or not np.array_equal(pos, np.arange(filled, filled + len(pos))):
            raise RuntimeError(f"{name}: scored positions are not the contiguous run after {filled}")
        x = np.where(valid, tok[np.minimum(ipos, n_pos - 1)], C.BOS_ID)   # padding after hi: causal, never read
        y = np.where(valid, tok[np.minimum(ipos + 1, n_pos)], -1)
        yt = torch.from_numpy(y).to(device)
        with amp:
            _, aux = model(torch.from_numpy(x).to(device), yt)
        h = tap.take()
        m = torch.from_numpy(scored).to(device)
        hk = h[m].to(key_dtype)
        if keys is None:
            keys = torch.empty((n_pos, hk.shape[-1]), dtype=key_dtype, device=key_device)
        sl = slice(filled, filled + len(pos))
        keys[sl].copy_(hk)
        if want_lm:
            lm[sl] = (-aux["per_token_loss"][m]).float().cpu().numpy()
        if lm_temps:
            with amp, tap.muted():
                z = model.logits(h)          # the forward's own call on the same input: same fp32 logits
            zs = z[m].float()
            del z
            zy = zs.gather(1, yt[m][:, None]).squeeze(1)
            check = max(check, float((zy - torch.logsumexp(zs, 1) + aux["per_token_loss"][m].float()).abs().max()))
            for ti, tau in enumerate(lm_temps):
                lmt[ti, sl] = (zy / tau - torch.logsumexp(zs / tau, 1)).cpu().numpy()
            del zs
        filled += len(pos)
        now = time.time()
        if now - last > LOG_EVERY_S:
            rate = filled / (now - t0)
            log(f"{name}: {filled:,}/{n_pos:,} positions, {rate:,.0f} pos/s, ETA {(n_pos - filled) / rate:,.0f}s")
            last = now
    TR.sync(device)
    if filled != n_pos or keys is None:
        raise RuntimeError(f"{name}: scored {filled} of {n_pos} positions")
    if want_lm and np.isnan(lm).any():
        raise RuntimeError(f"{name}: {int(np.isnan(lm).sum())} positions have no LM log-prob")
    return keys, lm, lmt, check


# ------------------------------------------------------------------------------------------------
# exact kNN
# ------------------------------------------------------------------------------------------------
def sqnorm_split(keys: torch.Tensor, chunk: int = 1 << 20) -> torch.Tensor:
    """[N, 3] float32 columns (hi, mid, lo) with hi + mid + lo = |k|^2 (fp32); hi, mid are bf16-representable,
    hence exact in TF32, and lo is ~2^-16 |k|^2 (its TF32 rounding is negligible)."""
    out = torch.empty((keys.shape[0], 3), dtype=torch.float32, device=keys.device)
    for i in range(0, keys.shape[0], chunk):
        kf = keys[i:i + chunk].float()
        n = (kf * kf).sum(1)
        hi = n.bfloat16().float()
        r = n - hi
        mid = r.bfloat16().float()
        out[i:i + chunk, 0], out[i:i + chunk, 1], out[i:i + chunk, 2] = hi, mid, r - mid
    return out


def is_oom(ex: BaseException) -> bool:
    return isinstance(ex, torch.cuda.OutOfMemoryError) or (isinstance(ex, RuntimeError) and "out of memory" in str(ex))


def _topk_chunk(qa: torch.Tensor, keys: torch.Tensor, ksplit: torch.Tensor, k: int, k_chunk: int,
                kbuf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Running top-k (smallest |k|^2 - 2 q.k) of one query block over all key chunks; unsorted."""
    d = keys.shape[1]
    best_v = best_i = None
    for k0 in range(0, keys.shape[0], k_chunk):
        k1 = min(keys.shape[0], k0 + k_chunk)
        kb = kbuf[:k1 - k0]
        kb[:, :d].copy_(keys[k0:k1])
        kb[:, d:d + 3].copy_(ksplit[k0:k1])
        s = torch.mm(qa, kb.t())
        v, i = torch.topk(s, min(k, k1 - k0), dim=1, largest=False, sorted=False)
        del s
        i += k0
        if best_v is None:
            best_v, best_i = v, i
        else:
            cv, ci = torch.cat((best_v, v), 1), torch.cat((best_i, i), 1)
            best_v, sel = torch.topk(cv, min(k, cv.shape[1]), dim=1, largest=False, sorted=False)
            best_i = ci.gather(1, sel)
    return best_v, best_i


def knn_search(queries: torch.Tensor, keys: torch.Tensor, k: int, q_chunk: int, k_chunk: int,
               ksplit: torch.Tensor | None = None) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor]]:
    """Exact k nearest keys (squared L2) of every query. queries [Nq, d] (any device), keys [Nk, d] on the search
    device, both 16-bit or fp32. Yields (q0, q1, dist [Q, k] float32 ascending, idx [Q, k] int64) per query block.
    On a CUDA OOM the key chunk (then the query chunk) is halved and the block redone."""
    dev = keys.device
    Nq, d = queries.shape
    k = min(k, keys.shape[0])
    d_aug = (d + 3 + 7) // 8 * 8
    if ksplit is None:
        ksplit = sqnorm_split(keys)
    kbuf: torch.Tensor | None = None
    q0 = 0
    while q0 < Nq:
        q1 = min(Nq, q0 + q_chunk)
        try:
            if kbuf is None:
                kbuf = torch.zeros((min(k_chunk, keys.shape[0]), d_aug), dtype=torch.float32, device=dev)
            qf = queries[q0:q1].to(dev).float()
            qa = torch.zeros((q1 - q0, d_aug), dtype=torch.float32, device=dev)
            qa[:, :d] = -2.0 * qf
            qa[:, d:d + 3] = 1.0
            v, i = _topk_chunk(qa, keys, ksplit, k, k_chunk, kbuf)
            dist = (v + (qf * qf).sum(1, keepdim=True)).clamp_min_(0.0)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
            if not is_oom(ex) or (k_chunk <= 4096 and q_chunk <= 256):
                raise
            kbuf = qf = qa = None
            torch.cuda.empty_cache()
            if k_chunk > 4096:
                k_chunk //= 2
            else:
                q_chunk //= 2
            log(f"OOM: retrying with q_chunk {q_chunk} k_chunk {k_chunk}")
            continue
        dist, order = dist.sort(1)
        yield q0, q1, dist, i.gather(1, order)
        q0 = q1


def knn_logp(dist: torch.Tensor, nbr_val: torch.Tensor, target: torch.Tensor, temps: list[float],
             ks: list[int]) -> torch.Tensor:
    """log p_knn(target) [len(temps), len(ks), Q]: softmax(-dist / T) over the first k' neighbours, summed over
    those whose value is the target (-inf if none is)."""
    match = nbr_val == target[:, None]
    out = torch.empty((len(temps), len(ks), dist.shape[0]), dtype=torch.float32, device=dist.device)
    for a, T in enumerate(temps):
        lg = dist / -T
        lgm = lg.masked_fill(~match, float("-inf"))
        for b, kp in enumerate(ks):
            out[a, b] = torch.logsumexp(lgm[:, :kp], 1) - torch.logsumexp(lg[:, :kp], 1)
    return out


def interp_nll(log_pknn: torch.Tensor, lm_logp: torch.Tensor, lam: float) -> torch.Tensor:
    """-log(lam p_knn + (1 - lam) p_lm) in log space; lam = 0 gives -lm_logp exactly."""
    la = math.log(lam) if lam > 0 else float("-inf")
    lb = math.log1p(-lam) if lam < 1 else float("-inf")
    return -torch.logaddexp(log_pknn + la, lm_logp + lb)


def choose_chunks(device: torch.device, n_q: int, n_k: int, d: int, k: int, budget_gib: float) -> tuple[int, int]:
    """(query chunk, key chunk). CUDA: the [Q, K] fp32 block (counted twice for topk temporaries) plus the fp32
    key-chunk copy fit in the budget left after what is already allocated (1 GiB margin), block <= 4 GiB."""
    if device.type != "cuda":
        return min(1024, n_q), min(65536, n_k)
    d_aug = (d + 3 + 7) // 8 * 8
    q = min(4096, n_q)
    avail = int(budget_gib * 2**30) - torch.cuda.memory_allocated(device) - (1 << 30) - q * (8 * d_aug + 64 * k)
    kc = min(avail // (8 * q + 4 * d_aug), MAX_SCORE_BYTES // (4 * q))
    kc = min(n_k, kc // 1024 * 1024)
    if kc < min(n_k, 4096):
        raise SystemExit(f"--gpu_mem_gb {budget_gib} leaves no room for the search next to the datastore "
                         f"({torch.cuda.memory_allocated(device) / 2**30:.1f} GiB allocated)")
    return q, kc


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="harness checkpoint (RUN/checkpoints/chck_<n>M.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=None, help=f"default {C.DEFAULT_DATA_DIR}")
    ap.add_argument("--train_split", default="train_10M", help="datastore split")
    ap.add_argument("--eval_split", default="dev_sub", help="query split (dev_sub, dev; never test)")
    ap.add_argument("--ctx", type=int, default=512, help="window length (stride ctx // 2)")
    ap.add_argument("--k", type=int, default=1024, help="neighbours retrieved per query")
    ap.add_argument("--ks", default=None, help="k' values scored from the same sorted list (default k/64,k/16,k/4,k)")
    ap.add_argument("--temps", default=DEFAULT_TEMPS, help="softmax temperatures on squared L2 distances")
    ap.add_argument("--lambdas", default=DEFAULT_LAMBDAS, help="interpolation weights of p_knn")
    ap.add_argument("--lm_temps", default=DEFAULT_LM_TEMPS,
                    help="temperatures of the LM-only recalibration baseline ('' = none)")
    ap.add_argument("--max_train_tokens", type=int, default=None, help="datastore = whole-document prefix")
    ap.add_argument("--max_eval_tokens", type=int, default=None, help="queries = whole-document prefix (>= 2 docs)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--key_dtype", default="fp16", choices=sorted(KEY_DTYPES))
    ap.add_argument("--batch", type=int, default=None, help="windows per forward (default 16 on CUDA, 8 elsewhere)")
    ap.add_argument("--gpu_mem_gb", type=float, default=30.0, help="CUDA allocator budget (GiB) for chunk sizes")
    ap.add_argument("--q_chunk", type=int, default=None)
    ap.add_argument("--k_chunk", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    return ap


def _peak(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    v = torch.cuda.max_memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    return round(v, 3)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv: list[str] | None = None) -> dict[str, Any]:
    a = build_parser().parse_args(argv)
    t_start = time.time()
    for sp in (a.train_split, a.eval_split):
        if sp in C.PROTECTED_SPLITS:
            raise SystemExit(f"{sp!r} is the protected FINAL split: this diagnostic never reads it")
    temps, lams = floats(a.temps), floats(a.lambdas)
    if (not temps or min(temps) <= 0 or not lams or min(lams) < 0 or max(lams) >= 1
            or min(floats(a.lm_temps) or [1.0]) <= 0):
        raise SystemExit("--temps and --lm_temps must be > 0 and --lambdas in [0, 1)")
    if a.train_split == a.eval_split:
        log(f"WARNING: datastore split == eval split ({a.eval_split}): every query retrieves itself")
    if a.threads:
        torch.set_num_threads(a.threads)
    device = TR.pick_device(a.device)
    amp, precision = TR.autocast_for(device, "auto")
    key_dtype = KEY_DTYPES[a.key_dtype]
    batch = a.batch or (16 if device.type == "cuda" else 8)
    out = Path(a.out)
    (out / "pt").mkdir(parents=True, exist_ok=True)
    (out / "pw").mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    mem: dict[str, float | None] = {}

    # data
    train = C.load_split(a.train_split, a.data_dir)
    ev = C.load_split(a.eval_split, a.data_dir)
    n_tr = doc_prefix(train.tokens, a.max_train_tokens, 1)
    n_ev = doc_prefix(ev.tokens, a.max_eval_tokens, 2)
    docs_ev = target_docs(ev.tokens, n_ev)
    if docs_ev.max() < 1:
        raise SystemExit(f"{a.eval_split}: need >= 2 documents for the even / odd split")

    # model
    t0 = time.time()
    model, cfg, meta = load_harness_model(Path(a.ckpt), device)
    for sp in (train, ev):
        if meta.get("tokenizer_sha256") and sp.meta.get("tokenizer_sha256") \
                and meta["tokenizer_sha256"] != sp.meta["tokenizer_sha256"]:
            raise SystemExit(f"checkpoint tokenizer {meta['tokenizer_sha256'][:12]} != {sp.name} tokenizer "
                             f"{sp.meta['tokenizer_sha256'][:12]}")
    tap = HeadTap(model.lm_head)
    with open(a.ckpt, "rb") as f:
        ck_sha = hashlib.sha256(f.read(1 << 20)).hexdigest()
    timing["load_s"] = time.time() - t0
    log(f"{a.ckpt}: {meta['model_name']} {cfg.n_layer}L d={cfg.n_embd} seq_len={cfg.seq_len} label={meta['label']} "
        f"words_seen={meta['words_seen']}; {device} {precision}; ctx {a.ctx} stride {a.ctx // 2}; "
        f"datastore {a.train_split} {n_tr - 1:,} keys; queries {a.eval_split} {n_ev - 1:,}")
    if a.ctx > cfg.seq_len:
        log(f"WARNING: ctx {a.ctx} > trained seq_len {cfg.seq_len} (RoPE extrapolation)")

    # encode (queries first: short, and it exercises the model before the long datastore pass)
    lm_temps = floats(a.lm_temps)
    if lm_temps and not callable(getattr(model, "logits", None)):
        log("WARNING: the model has no logits(h) method: no temperature-only baseline")
        lm_temps = []
    t0 = time.time()
    _peak(device)
    qkeys, lm_logp, lmt_logp, tcheck = encode_stream(model, tap, ev.tokens[:n_ev], a.ctx, batch, device, amp,
                                                     key_dtype, torch.device("cpu"), a.eval_split,
                                                     lm_temps=lm_temps or None)
    timing["eval_pass_s"] = time.time() - t0
    mem["eval_pass_peak_gib"] = _peak(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()  # return the logits blocks before the datastore allocation
    t0 = time.time()
    keys, _, _, _ = encode_stream(model, tap, train.tokens[:n_tr], a.ctx, batch, device, amp, key_dtype, device,
                                  a.train_split, want_lm=False)
    timing["datastore_pass_s"] = time.time() - t0
    mem["datastore_pass_peak_gib"] = _peak(device)
    log(f"encoded {n_ev - 1:,} queries in {timing['eval_pass_s']:.1f}s and {n_tr - 1:,} keys in "
        f"{timing['datastore_pass_s']:.1f}s ({(n_tr - 1) / max(timing['datastore_pass_s'], 1e-9):,.0f} pos/s)")
    tap.handle.remove()
    del model, tap
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # search
    t0 = time.time()
    Nq, Nk, d = qkeys.shape[0], keys.shape[0], keys.shape[1]
    k = min(a.k, Nk)
    ks = sorted({min(int(x), k) for x in (floats(a.ks) if a.ks else [k // 64, k // 16, k // 4, k]) if int(x) >= 1})
    values = torch.from_numpy(np.asarray(train.tokens[1:n_tr], dtype=np.int32)).to(device)
    targets = torch.from_numpy(np.asarray(ev.tokens[1:n_ev], dtype=np.int32))
    ksplit = sqnorm_split(keys)
    qc, kc = choose_chunks(device, Nq, Nk, d, k, a.gpu_mem_gb)
    qc, kc = a.q_chunk or qc, a.k_chunk or kc
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = key_dtype != torch.float32   # exact for 16-bit inputs
    log(f"search: {Nq:,} queries x {Nk:,} keys, d={d}, k={k}, q_chunk {qc}, k_chunk {kc}")
    lp = torch.empty((len(temps), len(ks), Nq), dtype=torch.float32)
    d1 = torch.empty(Nq, dtype=torch.float32)
    top1 = torch.empty(Nq, dtype=torch.bool)
    hit = torch.empty(Nq, dtype=torch.bool)
    last = time.time()
    try:
        for q0, q1, dist, idx in knn_search(qkeys, keys, k, qc, kc, ksplit):
            tg = targets[q0:q1].to(device)
            nv = values[idx]
            lp[:, :, q0:q1] = knn_logp(dist, nv, tg, temps, ks).cpu()
            d1[q0:q1] = dist[:, 0].cpu()
            top1[q0:q1] = (nv[:, 0] == tg).cpu()
            hit[q0:q1] = (nv == tg[:, None]).any(1).cpu()
            now = time.time()
            if now - last > LOG_EVERY_S or q1 == Nq:
                rate = q1 / (now - t0)
                log(f"search: {q1:,}/{Nq:,} queries, {rate:,.0f} q/s, {rate * Nk:.3g} distances/s, "
                    f"ETA {(Nq - q1) / rate:,.0f}s")
                last = now
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    TR.sync(device)
    timing["search_s"] = time.time() - t0
    mem["search_peak_gib"] = _peak(device)
    del keys, ksplit, values

    # selection on the even documents, report on the odd documents
    t0 = time.time()
    lm = torch.from_numpy(lm_logp)
    words_t = np.asarray(ev.words[1:n_ev], dtype=np.int64)
    groups = {"even": docs_ev % 2 == 0, "odd": docs_ev % 2 == 1, "all": np.ones(Nq, dtype=bool)}
    gm = {g: torch.from_numpy(m) for g, m in groups.items()}
    n_tok = {g: int(m.sum()) for g, m in groups.items()}
    n_word = {g: int(len(np.unique(words_t[m & (words_t < ev.n_words)]))) for g, m in groups.items()}

    def tot(nll: torch.Tensor) -> dict[str, float]:
        nll = nll.double()
        return {g: float(nll[gm[g]].sum()) for g in groups}

    def knn_grid(base: torch.Tensor) -> list[dict[str, Any]]:
        return [{"T": T, "k": kp, "lam": lam, "ai": ai, "bi": bi, "nll": tot(interp_nll(lp[ai, bi], base, lam))}
                for ai, T in enumerate(temps) for bi, kp in enumerate(ks) for lam in lams]

    def pick(rows: list[dict[str, Any]], g: str) -> dict[str, Any]:
        return min(rows, key=lambda r: r["nll"][g])

    def st(g: str, nll: dict[str, float]) -> dict[str, float]:
        return {"nats_per_token": nll[g] / n_tok[g], "nats_per_word": nll[g] / n_word[g],
                "gain_nats_per_token": (lm_tot[g] - nll[g]) / n_tok[g],
                "gain_nats_per_word": (lm_tot[g] - nll[g]) / n_word[g]}

    def edges(*items: tuple[str, float, list[float], bool]) -> list[str]:
        return ([f"{n}=max" for n, v, gv, _ in items if len(gv) > 1 and v == max(gv)]
                + [f"{n}=min" for n, v, gv, lo in items if lo and len(gv) > 1 and v == min(gv)])

    lm_tot = tot(-lm)
    grid = knn_grid(lm)
    best, orc_all, orc_odd = pick(grid, "even"), pick(grid, "all"), pick(grid, "odd")
    ko = st("odd", best["nll"])
    held: dict[str, Any] = {
        "tokens": n_tok["odd"], "words": n_word["odd"],
        "lm_nats_per_token": lm_tot["odd"] / n_tok["odd"], "lm_nats_per_word": lm_tot["odd"] / n_word["odd"],
        "knn_nats_per_token": ko["nats_per_token"], "knn_nats_per_word": ko["nats_per_word"],
        "gain_nats_per_token": ko["gain_nats_per_token"], "gain_nats_per_word": ko["gain_nats_per_word"],
        "gain_ge_0.03_nats_per_token": ko["gain_nats_per_token"] >= P7_GAIN,
        "knn_oracle_on_these_tokens": {"note": "T, lam, k picked on the odd documents themselves (in-sample)",
                                       "T": orc_odd["T"], "lam": orc_odd["lam"], "k": orc_odd["k"],
                                       **st("odd", orc_odd["nll"])}}
    headline = {"lm": held["lm_nats_per_token"], "knn": ko["nats_per_token"], "gain_knn": ko["gain_nats_per_token"],
                "gain_knn_oracle_same_tokens": held["knn_oracle_on_these_tokens"]["gain_nats_per_token"]}
    rows_t: list[dict[str, Any]] = []
    tb = best_t = None
    if lm_temps:
        rows_t = [{"tau": tau, "ti": ti, "nll": tot(-torch.from_numpy(lmt_logp[ti]))}
                  for ti, tau in enumerate(lm_temps)]
        tb, tb_odd = pick(rows_t, "even"), pick(rows_t, "odd")
        lm_tau = torch.from_numpy(lmt_logp[tb["ti"]])
        grid_t = knn_grid(lm_tau)
        best_t = pick(grid_t, "even")
        held["temp_only"] = {"note": "LM log softmax(z / tau), tau picked on the even documents",
                             "tau": tb["tau"], **st("odd", tb["nll"]),
                             "oracle_on_these_tokens": {"tau": tb_odd["tau"], **st("odd", tb_odd["nll"])}}
        kt = st("odd", best_t["nll"])
        held["knn_over_temp"] = {
            "note": "kNN interpolated with the temperature-scaled LM (tau fixed, T, lam, k picked on even documents)",
            "tau": tb["tau"], "T": best_t["T"], "lam": best_t["lam"], "k": best_t["k"], **kt,
            "gain_over_temp_nats_per_token": (tb["nll"]["odd"] - best_t["nll"]["odd"]) / n_tok["odd"],
            "gain_over_temp_nats_per_word": (tb["nll"]["odd"] - best_t["nll"]["odd"]) / n_word["odd"]}
        headline.update({"temp_only": held["temp_only"]["nats_per_token"],
                         "gain_temp_only": held["temp_only"]["gain_nats_per_token"],
                         "gain_temp_only_oracle_same_tokens":
                             held["temp_only"]["oracle_on_these_tokens"]["gain_nats_per_token"],
                         "knn_over_temp": kt["nats_per_token"], "gain_knn_over_temp": kt["gain_nats_per_token"],
                         "gain_knn_beyond_temp": held["knn_over_temp"]["gain_over_temp_nats_per_token"]})
    with open(out / "grid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["base", "T", "k", "lam", "even_nats_per_token", "odd_nats_per_token", "all_nats_per_token"])
        for base, rows in [("lm", grid)] + ([(f"lm_tau{tb['tau']:g}", grid_t)] if lm_temps else []):
            for r in rows:
                w.writerow([base, r["T"], r["k"], r["lam"], *(round(r["nll"][g] / n_tok[g], 6) for g in groups)])
    if lm_temps:
        with open(out / "lm_temp.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tau", "even_nats_per_token", "odd_nats_per_token", "all_nats_per_token"])
            for r in rows_t:
                w.writerow([r["tau"], *(round(r["nll"][g] / n_tok[g], 6) for g in groups)])
    timing["grid_s"] = time.time() - t0

    # per-token and per-word outputs (chosen parameters, every document; *_odd = held-out documents only)
    t0 = time.time()
    tag = PW_TAG.get(a.eval_split, a.eval_split)
    series = [("lm", -lm_logp), ("knn", interp_nll(lp[best["ai"], best["bi"]], lm, best["lam"]).numpy())]
    if lm_temps:
        series += [("lmtemp", -lmt_logp[tb["ti"]]),
                   ("knntemp", interp_nll(lp[best_t["ai"], best_t["bi"]], lm_tau, best_t["lam"]).numpy())]
    words_all = np.asarray(ev.words)
    files: list[str] = []
    pw_files: list[str] = []
    for label, v in series:
        pt = np.full(ev.n_tokens, np.nan, dtype=np.float32)
        pt[1:n_ev] = v
        np.save(out / "pt" / f"pt_{tag}_{label}.npy", pt.astype(np.float16))
        files.append(f"pt/pt_{tag}_{label}.npy")
        np.save(out / "pw" / f"pw_{tag}_{label}.npy", per_word(pt, words_all, ev.n_words).astype(np.float16))
        pt[1:n_ev][~groups["odd"]] = np.nan
        np.save(out / "pw" / f"pw_{tag}_{label}_odd.npy", per_word(pt, words_all, ev.n_words).astype(np.float16))
        pw_files += [f"pw_{tag}_{label}.npy", f"pw_{tag}_{label}_odd.npy"]
    (out / "pw" / "index.json").write_text(json.dumps(
        {"source": "babylm.analysis.knn_diag", "eval_split": a.eval_split, "eval_words": ev.n_words,
         "datastore_split": a.train_split, "eval_truncated": n_ev < ev.n_tokens,
         "labels": {"lm": "LM", "knn": f"kNN-LM T={best['T']:g} lam={best['lam']:g} k={best['k']}",
                    **({"lmtemp": f"LM at tau={tb['tau']:g}",
                        "knntemp": f"kNN-LM over lmtemp T={best_t['T']:g} lam={best_t['lam']:g} k={best_t['k']}"}
                       if lm_temps else {})},
         "files": pw_files}, indent=1))
    files += [f"pw/{x}" for x in pw_files] + ["pw/index.json", "grid.csv"] + (["lm_temp.csv"] if lm_temps else [])
    files.append("summary.json")
    timing["write_s"] = time.time() - t0
    timing["total_s"] = time.time() - t_start

    slicer: dict[str, str] = {}
    if a.eval_split in PW_TAG:
        labels = [s[0] for s in series]
        runs = " ".join(f"{x}={out}:{x}" for x in labels)
        pairs = "lm-knn" + (" lm-lmtemp lmtemp-knntemp" if lm_temps else "")
        tail = (" --subset dev" if a.eval_split == "dev" else "") + (f" --data_dir {a.data_dir}" if a.data_dir else "")
        base_cmd = f"python -m babylm.analysis.p2_slices --runs {runs} --pairs {pairs}"
        slicer = {"all_docs": f"{base_cmd} --out {out}/slices{tail}",
                  "drop_overlap": f"{base_cmd} --out {out}/slices_noovl --drop_overlap{tail}",
                  "held_out_only": (f"python -m babylm.analysis.p2_slices --runs "
                                    + " ".join(f"{x}={out}:{x}_odd" for x in labels)
                                    + f" --pairs {pairs} --out {out}/slices_odd{tail}")}
    d1n = d1.numpy()
    summary = {
        "tool": "babylm.analysis.knn_diag",
        "prediction": "P7 (critic log 2026-10-09 18:56Z): kNN over a train_10M datastore improves the final 10M model "
                      "by >= 0.03 DEV nats/token, mostly on words with train_10M count 11-1000 (diagnostic label)",
        "argv": sys.argv if argv is None else argv,
        "git_commit": TR.git_commit(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ckpt": {"path": str(a.ckpt), "sha256_first_1MB": ck_sha, **meta},
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": precision, "torch": torch.__version__, "key_dtype": a.key_dtype,
        "matmul": "fp32 copies of the keys; TF32 on CUDA for 16-bit keys (exact products, fp32 accumulate)",
        "protocol": {"ctx": a.ctx, "stride": a.ctx // 2, "trained_seq_len": cfg.seq_len,
                     "ctx_exceeds_trained_seq_len": a.ctx > cfg.seq_len, "batch_windows": batch,
                     "key": "input of model.lm_head (final RMS-normalised hidden state)",
                     "selection": "min mean NLL on targets of even-indexed eval documents; held out = odd-indexed",
                     "temp_check_max_abs_logp_diff": tcheck},
        "datastore": {"split": a.train_split, "split_tokens": train.n_tokens, "tokens_used": n_tr,
                      "truncated": n_tr < train.n_tokens,
                      "docs": int((np.asarray(train.tokens[:n_tr]) == C.BOS_ID).sum()),
                      "keys": Nk, "dim": d, "key_gib": round(Nk * d * keys_bytes(key_dtype) / 2**30, 3)},
        "eval": {"split": a.eval_split, "split_tokens": ev.n_tokens, "tokens_used": n_ev,
                 "truncated": n_ev < ev.n_tokens, "docs": int(docs_ev.max()) + 1, "queries": Nq,
                 "targets": n_tok, "words": n_word, "lm_nats_per_token_all": lm_tot["all"] / n_tok["all"]},
        "grid": {"temps": temps, "ks": ks, "lambdas": lams, "lm_temps": lm_temps, "q_chunk": qc, "k_chunk": kc},
        "headline_held_out_nats_per_token": headline,
        "chosen": {"T": best["T"], "lam": best["lam"], "k": best["k"],
                   "at_grid_edge": edges(("T", best["T"], temps, True), ("lam", best["lam"], lams, False),
                                         ("k", best["k"], ks, False)),
                   "even_lm_nats_per_token": lm_tot["even"] / n_tok["even"],
                   "even_knn_nats_per_token": best["nll"]["even"] / n_tok["even"],
                   **({"tau": tb["tau"], "tau_at_grid_edge": edges(("tau", tb["tau"], lm_temps, True)),
                       "knn_over_temp": {"T": best_t["T"], "lam": best_t["lam"], "k": best_t["k"],
                                         "at_grid_edge": edges(("T", best_t["T"], temps, True),
                                                               ("lam", best_t["lam"], lams, False),
                                                               ("k", best_t["k"], ks, False))}}
                      if lm_temps else {})},
        "held_out_odd_docs": held,
        "all_docs_at_chosen": {"lm_nats_per_token": lm_tot["all"] / n_tok["all"], "knn": st("all", best["nll"]),
                               **({"temp_only": st("all", tb["nll"]), "knn_over_temp": st("all", best_t["nll"])}
                                  if lm_temps else {})},
        "oracle_all_docs_IN_SAMPLE": {"note": "T, lam, k picked AND scored on all eval documents (optimistic)",
                                      "T": orc_all["T"], "lam": orc_all["lam"], "k": orc_all["k"],
                                      **st("all", orc_all["nll"])},
        "neighbours": {"nn1_sqdist_quantiles": dict(zip(("p10", "p50", "p90"),
                                                        np.quantile(d1n, [0.1, 0.5, 0.9]).tolist())),
                       "frac_nn1_sqdist_lt_1e-3": float((d1n < 1e-3).mean()),
                       "top1_value_is_target": float(top1.float().mean()),
                       "target_in_top_k": float(hit.float().mean())},
        "timing_s": {kk: round(v, 2) for kk, v in timing.items()},
        "throughput": {"datastore_pos_per_s": (n_tr - 1) / max(timing["datastore_pass_s"], 1e-9),
                       "search_queries_per_s": Nq / max(timing["search_s"], 1e-9),
                       "search_distances_per_s": Nq * Nk / max(timing["search_s"], 1e-9)},
        "memory": {**mem, "host_peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                                     * (1 if sys.platform == "darwin" else 1024) / 2**30, 3)},
        "outputs": files,
        "slicer_cmd": slicer or None,
    }
    summary = _clean(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    msg = (f"held out (odd docs, {n_tok['odd']:,} tokens), nats/token: LM {held['lm_nats_per_token']:.4f}; kNN-LM "
           f"{ko['nats_per_token']:.4f} (gain {ko['gain_nats_per_token']:+.4f}, T={best['T']:g} lam={best['lam']:g} "
           f"k={best['k']})")
    if lm_temps:
        msg += (f"; temp-only {headline['temp_only']:.4f} (gain {headline['gain_temp_only']:+.4f}, "
                f"tau={tb['tau']:g}); kNN over temp-LM {headline['knn_over_temp']:.4f} (beyond temp "
                f"{headline['gain_knn_beyond_temp']:+.4f})")
    log(msg)
    if summary["chosen"]["at_grid_edge"] or summary["chosen"].get("tau_at_grid_edge"):
        log(f"WARNING: choice at a grid edge: {summary['chosen']['at_grid_edge']} "
            f"{summary['chosen'].get('tau_at_grid_edge', '')} (extend --temps / --lm_temps / --k)")
    log(f"in-sample oracle (all docs): kNN gain {summary['oracle_all_docs_IN_SAMPLE']['gain_nats_per_token']:+.4f} "
        f"nats/token at T={orc_all['T']:g} lam={orc_all['lam']:g} k={orc_all['k']}; total {timing['total_s']:.1f}s "
        f"-> {out}")
    for name, cmd in slicer.items():
        log(f"slices ({name}): {cmd}")
    return summary


def keys_bytes(dt: torch.dtype) -> int:
    return torch.empty((), dtype=dt).element_size()


if __name__ == "__main__":
    main()
