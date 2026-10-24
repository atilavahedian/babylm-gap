"""Shared pieces for Agent A's sprint-1 models (sprint1/models/a_*.py).

* MiniGPT: a self-contained modded-nanogpt-style body used when sprint1/models/baseline.py is absent or
  does not expose the split API.  Recipe: parameter-free RMSNorm, RoPE, QK-norm, ReLU^2 MLP, value
  residual (first-layer values mixed into every layer), U-net skips between the two halves, zero-init
  output projections, untied zero-init LM head, logit soft-cap 30.
* build_body(cfg, extra_params): prefers the real baseline when it exposes the split API
  (hidden(idx) -> final normed hidden, logits(h) -> logits, in_embed(ids) -> input embeddings);
  otherwise MiniGPT.  `extra_params` is the number of parameters the caller's mechanism adds; the MLP
  hidden width of the body is shrunk by exactly that many parameters so the candidate is
  equal-parameter with the plain body (cfg.extra["a_equal_params"]=False disables this).
* delta_rule_chunked: exact chunk-parallel delta rule (Yang et al. 2024 WY/UT form) in plain PyTorch,
  with an optional one-step-lagged read (o_t = S_{t-1}^T q_t).
"""
from __future__ import annotations

import copy
import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig

SOFTCAP = 30.0
_SPLIT_API = ("hidden", "logits", "in_embed")


def rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def answer_code(body: nn.Module, ids: torch.Tensor, kind: str) -> torch.Tensor:
    """Code written for an answer token (detached): its LM-head row ("out", directly decodable by the head:
    adding it to the hidden state raises that token's logit) or its RMS-normed input embedding ("in")."""
    w = body.out_weight() if kind == "out" else None
    if w is not None:
        return F.embedding(ids, w.detach()).float()
    return rms(body.in_embed(ids).detach().float())


def make_null_tokens(vocab: int, seed: int = 4321, rows: int = 64, length: int = 4096) -> torch.Tensor:
    """Fixed random token block for null controls (drawn once at build time, never from the current batch)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (rows, length), generator=g)


def null_like(buf: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """[B, T] ids taken from the fixed null block: no dependence on any input, row or target."""
    r = torch.arange(B, device=buf.device) % buf.size(0)
    t = torch.arange(T, device=buf.device) % buf.size(1)
    return buf[r][:, t]




def lm_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)


def lm_loss_aux(logits: torch.Tensor, targets: torch.Tensor, training: bool):
    """(mean CE, aux) like baseline.py: in eval mode aux["per_token_loss"] is [B, T] (0 at ignored positions)."""
    if training:
        return lm_ce(logits, targets), {}
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.float().reshape(B * T, V), targets.reshape(B * T), ignore_index=-1,
                         reduction="none").view(B, T)
    return ce.sum() / (targets >= 0).sum().clamp_min(1), {"per_token_loss": ce.detach()}


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        f = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, T, H, D]
        T = x.size(1)
        cos = self.cos[:T][None, :, None, :]
        sin = self.sin[:T][None, :, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, d: int, n_head: int):
        super().__init__()
        self.h, self.hd = n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.lamb = nn.Parameter(torch.tensor(0.5))
        self.rot = Rotary(self.hd)

    def forward(self, x, v0):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).unbind(2)
        q, k = self.rot(rms(q)), self.rot(rms(k))
        if v0 is None:
            v0 = v
        v = (1 - self.lamb) * v + self.lamb * v0
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, scale=0.12)
        return self.proj(y.transpose(1, 2).reshape(B, T, d)), v0


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(d, hidden, bias=False)
        self.proj = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, hidden: int):
        super().__init__()
        self.attn = Attention(d, n_head)
        self.mlp = MLP(d, hidden)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, x0, v0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        a, v0 = self.attn(rms(x), v0)
        x = x + a
        x = x + self.mlp(rms(x))
        return x, v0


class MiniGPT(nn.Module):
    """Body with the split API: hidden(idx) -> normed final hidden; logits(h); in_embed(ids)."""

    def __init__(self, cfg: ModelConfig, mlp_hidden: int):
        super().__init__()
        d, L = cfg.n_embd, cfg.n_layer
        self.cfg = cfg
        self.mlp_hidden = mlp_hidden
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.blocks = nn.ModuleList([Block(d, cfg.n_head, mlp_hidden) for _ in range(L)])
        self.skip_w = nn.Parameter(torch.ones(L // 2))
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)

    def in_embed(self, ids):
        return self.embed(ids)

    def out_weight(self):
        return self.lm_head.weight

    def hidden(self, idx):
        x = x0 = rms(self.embed(idx))
        v0 = None
        L = len(self.blocks)
        skips = []
        for i, blk in enumerate(self.blocks):
            if i >= L - L // 2 and skips:
                x = x + self.skip_w[i - (L - L // 2)] * skips.pop()
            x, v0 = blk(x, x0, v0)
            if i < L // 2:
                skips.append(x)
        return rms(x)

    def logits(self, h):
        z = self.lm_head(h).float()
        return SOFTCAP * torch.tanh(z / SOFTCAP)

    def forward(self, idx, targets):
        return lm_loss_aux(self.logits(self.hidden(idx)), targets, self.training)

    def body_param_groups(self):
        muon = [p for blk in self.blocks for p in blk.parameters() if p.ndim == 2]
        ids = {id(p) for p in muon}
        return {"muon": muon, "adam": [p for p in self.parameters() if id(p) not in ids]}

    def param_groups(self):
        return self.body_param_groups()

    def flops_per_token(self):
        return self.body_flops_per_token()

    def forward_logits(self, idx):
        return self.logits(self.hidden(idx))

    def body_flops_per_token(self):
        cfg = self.cfg
        d, L, T = cfg.n_embd, cfg.n_layer, cfg.seq_len
        mat = sum(p.numel() for blk in self.blocks for p in blk.parameters() if p.ndim == 2)
        mat += self.lm_head.weight.numel()
        return 6.0 * mat + 6.0 * L * T * d  # causal attention: 2*T*d fwd MACs-equivalent per layer, x3


class BaselineAdapter(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def in_embed(self, ids):
        return self.inner.in_embed(ids)

    def out_weight(self):
        return self.inner.out_weight() if hasattr(self.inner, "out_weight") else None

    def hidden(self, idx):
        return self.inner.hidden(idx)

    def logits(self, h):
        return self.inner.logits(h).float()

    def body_param_groups(self):
        if hasattr(self.inner, "param_groups"):
            return self.inner.param_groups()
        muon = [p for n, p in self.inner.named_parameters()
                if p.ndim == 2 and not any(s in n for s in ("emb", "wte", "lm_head"))]
        ids = {id(p) for p in muon}
        return {"muon": muon, "adam": [p for p in self.inner.parameters() if id(p) not in ids]}

    def body_flops_per_token(self):
        return float(self.inner.flops_per_token()) if hasattr(self.inner, "flops_per_token") else float("nan")


def _try_baseline(cfg: ModelConfig):
    try:
        mod = importlib.import_module("sprint1.models.baseline")
        inner = mod.build(cfg)
    except (ImportError, AttributeError):
        return None
    if all(hasattr(inner, a) for a in _SPLIT_API):
        return inner
    return None


def plain_body_params(cfg: ModelConfig) -> tuple[int, int]:
    """(total, non-embedding) parameter count of the plain body this candidate is matched against."""
    choice = cfg.extra.get("body", "auto")
    m = _try_baseline(cfg) if choice in ("auto", "baseline") else None
    if m is None:
        m = MiniGPT(cfg, int(round(cfg.mlp_ratio * cfg.n_embd)))
    total = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters() if any(s in n for s in ("embed", "wte", "lm_head")))
    return total, total - emb


def build_body(cfg: ModelConfig, extra_params: int = 0) -> nn.Module:
    """Plain body whose MLP is shrunk by `extra_params` parameters (equal-parameter matching)."""
    d, L = cfg.n_embd, cfg.n_layer
    hidden = int(round(cfg.mlp_ratio * d))
    if cfg.extra.get("a_equal_params", True) and extra_params > 0:
        hidden = max(8, hidden - int(round(extra_params / (2 * d * L))))
    choice = cfg.extra.get("body", "auto")
    if choice in ("auto", "baseline"):
        c2 = copy.deepcopy(cfg)
        c2.mlp_ratio = hidden / d
        inner = _try_baseline(c2)
        if inner is not None:
            return BaselineAdapter(inner)
        if choice == "baseline":
            raise RuntimeError("sprint1.models.baseline missing or lacks split API " + str(_SPLIT_API))
    return MiniGPT(cfg, hidden)


def delta_rule_chunked(q, k, v, beta, chunk: int = 64, lagged: bool = False, carry: bool = True):
    """Delta rule  S_t = S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T,  o_t = S_t^T q_t  (lagged: S_{t-1}).

    q, k: [B, H, T, dk] (k should be L2-normalised), v: [B, H, T, dv], beta: [B, H, T] in (0, 1).
    Exact chunk-parallel form; computed in float32.  Returns o: [B, H, T, dv].
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _delta_rule_chunked_fp32(q.float(), k.float(), v.float(), beta.float(), chunk, lagged, carry)


def _delta_rule_chunked_fp32(q, k, v, beta, chunk, lagged, carry=True):
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    C = min(chunk, T)
    pad = (-T) % C
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
    n = (T + pad) // C
    q = q.view(B, H, n, C, dk)
    k = k.view(B, H, n, C, dk)
    v = v.view(B, H, n, C, dv)
    beta = beta.view(B, H, n, C, 1)
    kb, vb = k * beta, v * beta
    eye = torch.eye(C, device=k.device, dtype=k.dtype)
    a = torch.tril(kb @ k.transpose(-1, -2), diagonal=-1) + eye
    w = torch.linalg.solve_triangular(a, kb, upper=False, unitriangular=True)
    u = torch.linalg.solve_triangular(a, vb, upper=False, unitriangular=True)
    mask = torch.tril(torch.ones(C, C, device=k.device, dtype=k.dtype), diagonal=-1 if lagged else 0)
    S = k.new_zeros(B, H, dk, dv)
    outs = []
    for i in range(n):
        qi, ki = q[:, :, i], k[:, :, i]
        ui = u[:, :, i] - w[:, :, i] @ S
        outs.append(qi @ S + ((qi @ ki.transpose(-1, -2)) * mask) @ ui)
        S = S + ki.transpose(-1, -2) @ ui
        if not carry:                     # null control: state never crosses a chunk boundary
            S = torch.zeros_like(S)
    o = torch.stack(outs, dim=2).reshape(B, H, n * C, dv)
    return o[:, :, :T]


def delta_rule_reference(q, k, v, beta, lagged: bool = False):
    """Token-by-token recurrence, for tests."""
    B, H, T, dk = k.shape
    S = k.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for t in range(T):
        if lagged:
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
        kt = k[:, :, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + beta[:, :, t, None, None] * kt[..., None] * (v[:, :, t] - pred)[:, :, None, :]
        if not lagged:
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, dim=2)


def split_groups(model: nn.Module, body: nn.Module, adam_extra: list[nn.Parameter]) -> dict:
    """Body groups + the mechanism's params: 2-D hidden matrices -> muon, the rest (and adam_extra) -> adam."""
    g = body.body_param_groups()
    muon, adam = list(g["muon"]), list(g["adam"])
    seen = {id(p) for p in muon + adam}
    force_adam = {id(p) for p in adam_extra}
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (muon if (p.ndim == 2 and id(p) not in force_adam) else adam).append(p)
    return {"muon": muon, "adam": adam}
