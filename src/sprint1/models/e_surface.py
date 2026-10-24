"""E2: Surface-shared token embeddings (SSE).  Idea card: sprint1/ideas/E_surface.md

Every GPT-2 token's INPUT and OUTPUT embedding is a free row plus parts shared with other tokens through
its spelling, fixed from the tokenizer before training (no data, no counts):

    E_v = F_v + C[canon(v)] + S[shape(v)] + mean_{g in grams(v)} G[hash(g)]
      canon(v): the token's bytes with surrounding spaces stripped and ASCII-lowercased
                (" The", "The", " the", "THE"->"the"... merge into one of ~32.8k canonical forms)
      shape(v): leading-space x case pattern x has-digit x length bucket (80 shapes)
      grams(v): byte 3- and 4-grams of "<" + canon + ">", hashed into sf_hash buckets (fastText-style)

Target: context-new tokens carry ~63% of the bits, and they are disproportionately RARE tokens whose own
rows get few positive updates in a short run (Gutenberg check: 36% of context-new targets have <1000
training occurrences per 22M tokens; 16.5% have a case/space variant that is >2x more frequent). Shared
parts let a rare row borrow what its frequent relatives learned. C, S, G start at zero, so at step 0 the
model is exactly the plain model; the composed parts only grow if they lower the loss.

FLOPs: the output matrix [V, d] is composed ONCE PER STEP (V*(n_grams+3)*d adds, amortised over the
tokens of the step); the input side composes only the rows used (per token). The V x d head matmul and the
body are unchanged, so the per-token cost is the plain model's +0.3% at small. Shared tables are sf_dim=128
wide with one k->d projection per side: +17M FLOP-free parameters at small (plain: 49.3M).

Null control (cfg.extra["null"]=True): identical tables, parameter count and FLOPs, but the feature map is
applied through a fixed random permutation of the vocabulary: token v gets the canon/shape/gram indices
of token pi(v). Group sizes and sharing statistics are preserved, the spelling relation is destroyed.
This separates "spelling tells you who to share with" from "extra embedding parameters / regularisation".
sf_off=True is the plain reference on the same body.

Knobs (cfg.extra): null=False, sf_off=False, sf_side="both"|"in"|"out", sf_canon=True, sf_shape=True,
sf_grams=True, sf_hash=32768, sf_max_grams=24, sf_dim=128 (shared-table width, projected to d;
0 = full width d), sf_seed=1234 (null permutation), softcap=15.0,
log_novel=True, tokens_per_step=65536, sf_encoder_json (optional path to GPT-2 encoder.json if tiktoken's
gpt2 files are unavailable offline).
Vocabulary source: tiktoken "gpt2" (honours TIKTOKEN_CACHE_DIR; falls back to the repo's
sprint1/measure/data/tiktoken_cache), else an encoder.json. Ids >= 50257 (padding) get no shared parts.
"""
from __future__ import annotations

import functools
import json
import os
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.e_common import (EBody, body_flops_per_token, head_flops_per_token, make_head, softcap,
                                     split_ce_aux)

GPT2_N = 50257
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_FALLBACK = os.path.join(_REPO, "sprint1", "measure", "data", "tiktoken_cache")
_ENCODER_FALLBACK = os.path.join(_REPO, "sprint1", "measure", "data", "tok", "encoder.json")


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


@functools.lru_cache(maxsize=4)
def gpt2_token_bytes(encoder_json: str | None = None) -> tuple[bytes, ...]:
    """Byte string of every GPT-2 token id 0..50256 (data-free; read from the tokenizer only)."""
    errors = []
    if encoder_json is None:
        try:
            import tiktoken
            try:
                enc = tiktoken.get_encoding("gpt2")
            except Exception as e:  # offline: retry with the repo's cache (measurement agent's copy)
                errors.append(repr(e)[:200])
                old = os.environ.get("TIKTOKEN_CACHE_DIR")
                os.environ["TIKTOKEN_CACHE_DIR"] = _CACHE_FALLBACK
                try:
                    enc = tiktoken.get_encoding("gpt2")
                finally:
                    if old is None:
                        os.environ.pop("TIKTOKEN_CACHE_DIR", None)
                    else:
                        os.environ["TIKTOKEN_CACHE_DIR"] = old
            return tuple(enc.decode_single_token_bytes(i) for i in range(GPT2_N))
        except Exception as e:
            errors.append(repr(e)[:200])
            encoder_json = _ENCODER_FALLBACK
    try:
        with open(encoder_json, encoding="utf-8") as f:
            table = json.load(f)
    except OSError as e:
        raise RuntimeError(f"e_surface: no GPT-2 vocabulary available ({errors + [repr(e)]}); "
                           "set TIKTOKEN_CACHE_DIR or cfg.extra['sf_encoder_json']") from e
    inv = {c: b for b, c in _bytes_to_unicode().items()}
    out = [b""] * GPT2_N
    for tok, i in table.items():
        if 0 <= i < GPT2_N:
            out[i] = bytes(inv[ch] for ch in tok)
    return tuple(out)


def _case(core: bytes) -> int:
    letters = [c for c in core if 65 <= c <= 90 or 97 <= c <= 122]
    if not letters:
        return 0
    up = [65 <= c <= 90 for c in letters]
    if not any(up):
        return 1                      # lower
    if all(up):
        return 3 if len(letters) > 1 else 2  # UPPER (single capital counts as Title)
    if up[0] and not any(up[1:]):
        return 2                      # Title
    return 4                          # mixed


@functools.lru_cache(maxsize=8)
def surface_features(vocab_size: int, n_hash: int, max_grams: int, encoder_json: str | None = None):
    """Deterministic feature indices from the tokenizer: canon_id [V], n_canon, shape_id [V] (80 shapes),
    gram_ids [V, max_grams] in [0, n_hash] with n_hash = padding."""
    toks = gpt2_token_bytes(encoder_json)
    canon_ix: dict[bytes, int] = {}
    canon = torch.zeros(vocab_size, dtype=torch.long)
    shape = torch.zeros(vocab_size, dtype=torch.long)
    grams = torch.full((vocab_size, max_grams), n_hash, dtype=torch.long)
    for v in range(vocab_size):
        if v >= len(toks):
            canon[v] = canon_ix.setdefault(b"<pad%d>" % v, len(canon_ix))
            continue
        b = toks[v]
        core = b.strip(b" ")
        key = core.lower() if core else b
        canon[v] = canon_ix.setdefault(key, len(canon_ix))
        lead = 1 if b[:1] == b" " else 0
        digit = 1 if any(48 <= c <= 57 for c in core) else 0
        n = len(core)
        lb = 0 if n <= 1 else 1 if n <= 3 else 2 if n <= 6 else 3
        shape[v] = ((experimenter * 5 + _case(core)) * 2 + digit) * 4 + lb
        if core:
            w = b"<" + key + b">"
            gs = [w[i:i + n_] + bytes([n_]) for n_ in (3, 4) for i in range(len(w) - n_ + 1)]
            for j, g in enumerate(gs[:max_grams]):
                grams[v, j] = zlib.crc32(g) % n_hash
    return canon, len(canon_ix), shape, grams


class SurfaceLM(nn.Module):
    N_SHAPE = 80

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ex = cfg.extra
        self.cfg = cfg
        d, V = cfg.n_embd, cfg.vocab_size
        self.body = EBody(cfg)
        self.lm_head = make_head(cfg, self.body)
        self.cap = float(ex.get("softcap", 15.0))
        self.null = bool(ex.get("null", False))
        self.off = bool(ex.get("sf_off", False))
        side = str(ex.get("sf_side", "both"))
        self.use_in = (not self.off) and side in ("both", "in")
        self.use_out = (not self.off) and side in ("both", "out") and not cfg.tie_embeddings
        self.use_canon = bool(ex.get("sf_canon", True))
        self.use_shape = bool(ex.get("sf_shape", True))
        self.use_grams = bool(ex.get("sf_grams", True))
        self.H = int(ex.get("sf_hash", 32768))
        self.G = int(ex.get("sf_max_grams", 24))
        self.dim = int(ex.get("sf_dim", 128)) or d     # width of the shared tables (projected to d)
        self.log_novel = bool(ex.get("log_novel", True))
        self.tokens_per_step = float(ex.get("tokens_per_step", 65536))
        self.avg_grams = 0.0
        if self.use_in or self.use_out:
            canon, n_canon, shape, grams = surface_features(V, self.H, self.G, ex.get("sf_encoder_json"))
            if self.null:
                g = torch.Generator().manual_seed(int(ex.get("sf_seed", 1234)))
                perm = torch.randperm(V, generator=g)
                canon, shape, grams = canon[perm], shape[perm], grams[perm]
            self.register_buffer("canon_id", canon.clone())
            self.register_buffer("shape_id", shape.clone())
            self.register_buffer("gram_ids", grams.clone())
            self.avg_grams = float((grams < self.H).sum()) / V
            self.n_canon = n_canon
            # ModuleDict keys double as optimizer buckets in the harness (sprint1/harness/optim.split_adam):
            # "head" in a parameter name -> head LR, so output-side shared tables train at the head's rate.
            sides = (["in"] if self.use_in else []) + (["head"] if self.use_out else [])
            self.tabs = nn.ModuleDict({s: self._tables(n_canon, d) for s in sides})

    def _tables(self, n_canon: int, d: int) -> nn.ModuleDict:
        t = nn.ModuleDict()
        k = self.dim
        if self.use_canon:
            t["canon"] = nn.Embedding(n_canon, k)
        if self.use_shape:
            t["shape"] = nn.Embedding(self.N_SHAPE, k)
        if self.use_grams:
            t["gram"] = nn.Embedding(self.H + 1, k)  # last row = padding (excluded from the mean)
        for e in t.values():
            nn.init.zeros_(e.weight)                 # shared parts start at 0: step 0 == plain model
        if k != d:
            t["proj"] = nn.Linear(k, d, bias=False)  # shared tables are narrow; one projection per side
            nn.init.normal_(t["proj"].weight, std=k ** -0.5)
        return t

    def shared_rows(self, side: str, ids: torch.Tensor) -> torch.Tensor:
        """Shared (spelling) part of the embedding rows for token ids `ids` [...] -> [..., d]."""
        t = self.tabs[side]
        out = None
        if self.use_canon:
            out = t["canon"](self.canon_id[ids])
        if self.use_shape:
            s = t["shape"](self.shape_id[ids])
            out = s if out is None else out + s
        if self.use_grams:
            g = F.embedding_bag(self.gram_ids[ids].reshape(-1, self.G), t["gram"].weight, mode="mean",
                                padding_idx=self.H).view(*ids.shape, -1)
            out = g if out is None else out + g
        if "proj" in t:
            out = t["proj"](out)
        return out

    # ----- contract hooks -----
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = self.body.muon_params()
        ids = {id(p) for p in muon}
        return {"muon": muon, "adam": [p for p in self.parameters() if id(p) not in ids]}

    def flops_per_token(self) -> float:
        cfg = self.cfg
        f = body_flops_per_token(cfg) + head_flops_per_token(cfg)
        k, d = self.dim, cfg.n_embd
        per_row = 3.0 * (self.avg_grams + 3.0) * k            # gathers + adds, fwd + scatter-add bwd
        if k != d:
            per_row += 6.0 * k * d                             # projection to d, fwd + bwd
        if self.use_in:
            f += per_row                                       # input rows: per token
        if self.use_out:
            f += cfg.vocab_size * per_row / self.tokens_per_step   # output matrix: once per step, amortised
        return f

    # ----- model -----
    def out_weight(self) -> torch.Tensor:
        W = self.lm_head.weight
        if self.use_out:
            W = W + self.shared_rows("head", torch.arange(W.size(0), device=W.device))
        return W

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.body.wte(idx)
        if self.use_in:
            x = x + self.shared_rows("in", idx)
        return self.body.hidden_from_x(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return softcap(F.linear(h, self.out_weight().to(h.dtype)).float(), self.cap)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        B, T = idx.shape
        z = self.logits(self.hidden(idx))
        valid = targets >= 0
        ce = F.cross_entropy(z.view(B * T, -1), targets.reshape(-1).clamp_min(0), reduction="none").view(B, T)
        ce = ce * valid
        loss = ce.sum() / valid.sum().clamp_min(1)
        aux = split_ce_aux(ce, idx, targets) if self.log_novel else {}
        return loss, aux

    @torch.no_grad()
    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.logits(self.hidden(idx))


def build(cfg: ModelConfig) -> nn.Module:
    return SurfaceLM(cfg)
