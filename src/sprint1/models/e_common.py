"""Shared pieces for invention-agent-E models (sprint1/models/e_*.py): input/output representation.

Agent E changes WHERE THE BITS LAND (the LM head and the token embeddings), never the body. All E models
therefore share one plain body defined here, a minimal modded-nanogpt-style causal Transformer:
parameter-free RMSNorm, RoPE, QK-norm, ReLU^2 MLP, zero-initialised output projections, untied head
(near-zero init), logit soft-cap (cfg.extra["softcap"], default 15). It is self-contained on purpose: the
body is identical across an E candidate, its null control and its plain reference (rf_k=0 / sf_off=True),
so any difference is attributable to the head/embedding mechanism. When sprint1/models/baseline.py is
final, the experimenter can swap the body; the heads only need `hidden_from_x(x)` and the token-embedding hook.

Null controls never read another row of the batch (protocol 7.4: eval rows are consecutive windows of one DEV
stream). They read NullBank: final hidden states of one fixed TRAIN token block, refreshed only in on_step_end.

Also here: first-occurrence (context-new) masks for the per-class loss split that every E model logs
(aux["ce_new"], aux["ce_seen"], aux["frac_new"]), and the analytic FLOP helpers. FLOP convention:
6 * (matmul MACs on the trainable path) per token for fwd+bwd, causal attention counted as L*T*d MACs
(triangle-averaged QK^T + AV), plus explicit elementwise terms where a mechanism adds them.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig


def rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def first_train_shard(data_dir=None) -> str | None:
    """First harness train shard, discovered exactly as sprint1.harness.data.load_corpus does (sorted
    fineweb_train_*.bin in data_dir or DEFAULT_DATA_DIR; the harness numbers them from 000001). None if absent."""
    from pathlib import Path

    from sprint1.harness import data as D
    d = Path(data_dir or D.DEFAULT_DATA_DIR)
    names = sorted(p.name for p in d.glob("fineweb_train_*.bin")) if d.is_dir() else []
    return str(d / names[0]) if names else None


def load_null_ids(cfg: ModelConfig, n: int, seed: int = 1234) -> tuple[torch.Tensor, str]:
    """A fixed block of n TRAIN token ids for null controls (protocol 7.4: nulls draw foreign content from a fixed
    TRAIN buffer, never from another row of the current batch). Source, in order:
      cfg.extra["null_ids_path"] = a harness shard (.bin with header) -> block at a seeded offset;
      "auto" (default): the first harness train shard (first_train_shard: sorted fineweb_train_*.bin in
      cfg.extra["null_data_dir"] or DEFAULT_DATA_DIR, as load_corpus finds them), else -- only when no shard
      exists -- the harness's deterministic synthetic TRAIN stream (the stream --synthetic trains on)."""
    import numpy as np

    from sprint1.harness import data as D
    path = cfg.extra.get("null_ids_path", "auto")
    if path == "auto":
        path = first_train_shard(cfg.extra.get("null_data_dir"))
    if path:
        toks = D.load_shard(path)
        src = f"shard:{os.path.basename(path)}"
    else:
        toks = D.synthetic_tokens(1 << 16, seed=1)
        src = "synthetic-train"
    rng = np.random.default_rng(seed)
    off = int(rng.integers(0, max(1, len(toks) - n)))
    ids = torch.from_numpy(np.asarray(toks[off:off + n], dtype=np.int64)).clamp_max(cfg.vocab_size - 1)
    return ids, f"{src}@{off}"


class NullBank(nn.Module):
    """Fixed foreign context for null controls. Holds the final hidden states of ONE fixed TRAIN token block
    (load_null_ids), recomputed from the current body only in refresh() -- called at build and from
    on_step_end every `every` steps -- never inside forward and never in eval. context(B, T) gives position t
    the bank's state at position t for every row, so no row of any batch is ever read (batch isolation and
    causality hold by construction, in train and eval mode). Buffers only: no parameters."""

    def __init__(self, cfg: ModelConfig, body: "EBody", every: int = 50, seed: int = 1234):
        super().__init__()
        ids, self.source = load_null_ids(cfg, cfg.seq_len, seed)
        self.every = int(every)
        self.register_buffer("ids", ids)
        self.register_buffer("h", torch.zeros(len(ids), cfg.n_embd))
        self.refresh(body)

    @torch.no_grad()
    def refresh(self, body: "EBody") -> None:
        was = body.training
        body.eval()
        self.h.copy_(body.hidden(self.ids.unsqueeze(0)).squeeze(0).float())
        body.train(was)

    def maybe_refresh(self, body: "EBody", step: int) -> None:
        if self.every > 0 and step % self.every == 0:
            self.refresh(body)

    def context(self, B: int, T: int, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(T, device=self.h.device) % self.h.size(0)
        return self.h[pos].to(dtype).unsqueeze(0).expand(B, T, -1)

    def flops_per_token(self, body_fwd_flops_per_token: float, tokens_per_step: float) -> float:
        """Amortised refresh cost (forward only over the block every `every` steps)."""
        if self.every <= 0:
            return 0.0
        return body_fwd_flops_per_token * self.h.size(0) / (self.every * tokens_per_step)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, H, T, hd]
        t = torch.arange(x.size(-2), device=x.device, dtype=torch.float32)
        f = torch.outer(t, self.inv_freq)
        cos, sin = f.cos().to(x.dtype), f.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, mlp_ratio: float):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        hidden = int(mlp_ratio * d)
        self.fc = nn.Linear(d, hidden, bias=False)
        self.fc_out = nn.Linear(hidden, d, bias=False)
        self.rotary = Rotary(d // n_head)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.fc_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.qkv(rms(x)).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rotary(rms(q)), self.rotary(rms(k))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.fc_out(F.relu(self.fc(rms(x))).square())


class EBody(nn.Module):
    """Token embedding + blocks + final norm. `hidden(idx)` uses the plain table; models that compose
    their own input embeddings call `hidden_from_x(x)` with x [B, T, d]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        nn.init.normal_(self.wte.weight, std=0.5)
        self.blocks = nn.ModuleList(Block(cfg.n_embd, cfg.n_head, cfg.mlp_ratio) for _ in range(cfg.n_layer))

    def hidden_from_x(self, x: torch.Tensor) -> torch.Tensor:
        x = rms(x)
        for blk in self.blocks:
            x = blk(x)
        return rms(x)

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        return self.hidden_from_x(self.wte(idx))

    def muon_params(self) -> list[nn.Parameter]:
        return [p for blk in self.blocks for p in blk.parameters() if p.ndim == 2]


def make_head(cfg: ModelConfig, body: EBody) -> nn.Linear:
    head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
    nn.init.normal_(head.weight, std=0.002)  # near-zero, as modded-nanogpt's zero-init head
    if cfg.tie_embeddings:
        head.weight = body.wte.weight
    return head


def softcap(z: torch.Tensor, cap: float) -> torch.Tensor:
    return cap * torch.tanh(z / cap) if cap > 0 else z


def body_flops_per_token(cfg: ModelConfig) -> float:
    """fwd+bwd FLOPs/token of the blocks only (no embedding lookup, no head)."""
    d, L, T = cfg.n_embd, cfg.n_layer, cfg.seq_len
    hidden = int(cfg.mlp_ratio * d)
    return 6.0 * (L * (4 * d * d + 2 * d * hidden) + L * T * d)


def head_flops_per_token(cfg: ModelConfig) -> float:
    """fwd+bwd FLOPs/token of the dense V x d LM head matmul (the largest single cost at small size)."""
    return 6.0 * cfg.vocab_size * cfg.n_embd


def first_occurrence_mask(idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """new[b, i] = True iff targets[b, i] (= x_{i+1}) does not occur in idx[b, :i+1] (context-new token,
    the class carrying ~63% of the bits in the 4 Oct ledger diagnostic). Ignored targets are never new.
    O(B*T^2) bool compare, compile-friendly; call under no_grad."""
    T = idx.size(1)
    ar = torch.arange(T, device=idx.device)
    causal = ar.view(1, T) <= ar.view(T, 1)
    seen = ((targets.unsqueeze(-1) == idx.unsqueeze(-2)) & causal).any(-1)
    return (~seen) & (targets >= 0)


def split_ce_aux(ce: torch.Tensor, idx: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-class mean CE for logging (detached): context-new vs already-seen target tokens."""
    with torch.no_grad():
        valid = targets >= 0
        new = first_occurrence_mask(idx, targets)
        seen = valid & ~new
        ce = ce.detach().float()
        nn_, ns = new.sum().clamp_min(1), seen.sum().clamp_min(1)
        return {
            "ce_new": (ce * new).sum() / nn_,
            "ce_seen": (ce * seen).sum() / ns,
            "frac_new": new.sum().float() / valid.sum().clamp_min(1),
        }
