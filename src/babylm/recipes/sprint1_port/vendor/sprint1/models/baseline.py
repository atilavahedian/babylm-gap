"""Sprint-1 baseline: a single-GPU adaptation of the modded-nanogpt speedrun recipe.

Source recipe: https://github.com/KellerJordan/modded-nanogpt (train_gpt.py and track_1_short/model/gpt.py).
Kept (the parts that are architecture, not 8xH100 engineering):
  * untied token embedding and lm head (embedding N(0,1), head N(0, 0.005) as in the current record)
  * RMSNorm everywhere (no learned gain): on the embedding, before every sublayer, on q/k, at the end
  * rotary position embedding, "half-truncated" (only head_dim/4 frequencies rotate, base 1024)
  * QK-norm, attention scale 1/sqrt(head_dim) (the record's 0.12 at head_dim 64-128)
  * ReLU^2 MLP (hidden = mlp_ratio * d), zero-initialised output projections (attn and MLP)
  * value embeddings: n_ve extra embedding tables mixed into the values of the first n_ve and the last
    n_ve layers (v <- l0 * v + l1 * ve, learned l0, l1; a value-residual-style shortcut from the tokens)
  * x0 injection per block (x <- a * x + b * x0, a=1, b=0 at init) and U-net skips (encoder half pushes
    its outputs, decoder half adds them back with learned weights)
  * logit softcap  softcap * tanh(z / softcap)  (15, @KoszarskyB); computed in fp32
  * CastedLinear init: U(-sqrt(3) * 0.5 / sqrt(fan_in), +...)
Dropped (documented in sprint1/README.md): FlexAttention document masking / sliding windows (full causal
attention over fixed windows instead), FP8 heads, MTP / n-gram tables / sampled softmax, batch and
window schedules, multi-GPU sharding.

Split API (used by candidates A, B, C to reuse this body without editing it):
  hidden(idx) -> normalised final hidden [B, T, d]; logits(h) -> fp32 softcapped logits [B, T, V];
  out_weight() -> lm head weight [V, d]; in_embed(ids) -> raw token embedding [..., d].
Harness hooks: param_groups(), flops_per_token(). In eval mode aux["per_token_loss"] ([B, T], detached,
0 at ignored positions) lets the harness compute the new/seen split in one pass.

cfg.extra knobs (all optional):
  n_value_embeds (int, default max(1, round(n_layer / 4)); 0 disables)
  softcap (float, default 15.0; 0 disables)
  attn_scale (float, default 1/sqrt(head_dim))
  rope_base (float, default 1024)       rope_frac (float, default 0.25: fraction of head_dim rotating as freqs)
  unet (bool, default True)             x0_inject (bool, default True)
  head_init_std (float, default 0.005)
  embnorm (bool, default True: RMS-normalise the token embedding before the first block; False feeds the raw
           N(0, 1) embedding, for the tuner's ablation)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    """Bias-free linear with the speedrun init; dtype handled by autocast."""

    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__(in_features, out_features, bias=False)
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            bound = math.sqrt(3.0) * 0.5 * in_features ** -0.5
            nn.init.uniform_(self.weight, -bound, bound)


class Rotary(nn.Module):
    """Half-truncated RoPE: frac*head_dim frequencies (of head_dim/2 pairs) rotate, the rest have frequency 0."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 1024.0, frac: float = 0.25):
        super().__init__()
        n_rot = max(1, int(head_dim * frac))
        half = head_dim // 2
        n_rot = min(n_rot, half)
        freq = (1.0 / base) ** torch.linspace(0, 1, steps=n_rot, dtype=torch.float32)
        freq = torch.cat([freq, freq.new_zeros(half - n_rot)])
        self.register_buffer("freq", freq, persistent=False)
        self._build(max_seq_len)

    def _build(self, T: int) -> None:
        t = torch.arange(T, dtype=torch.float32, device=self.freq.device)
        theta = torch.outer(t, self.freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, H, D]
        T = x.size(-3)
        if T > self.cos.size(0):
            self._build(T)
        cos = self.cos[None, :T, None, :]
        sin = self.sin[None, :T, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x2 * cos - x1 * sin
        return torch.cat((y1, y2), dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H = cfg.n_embd, cfg.n_head
        assert d % H == 0, "n_embd must be divisible by n_head"
        self.n_head, self.head_dim = H, d // H
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary"
        ex = cfg.extra
        self.c_q = CastedLinear(d, d)
        self.c_k = CastedLinear(d, d)
        self.c_v = CastedLinear(d, d)
        self.c_proj = CastedLinear(d, d, zero_init=True)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))  # value mix: own v, value embedding
        self.rotary = Rotary(self.head_dim, cfg.seq_len, float(ex.get("rope_base", 1024.0)),
                             float(ex.get("rope_frac", 0.25)))
        self.scale = float(ex.get("attn_scale", self.head_dim ** -0.5))

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None) -> torch.Tensor:
        B, T, d = x.shape
        H, D = self.n_head, self.head_dim
        q = self.c_q(x).view(B, T, H, D)
        k = self.c_k(x).view(B, T, H, D)
        v = self.c_v(x).view(B, T, H, D)
        dt = q.dtype  # the autocast compute dtype (bf16 on CUDA); norm/rotary may upcast
        q, k = self.rotary(norm(q)).to(dt), self.rotary(norm(k)).to(dt)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.to(dt).transpose(1, 2),
                                           is_causal=True, scale=self.scale)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, d))


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.c_fc = CastedLinear(d, hidden)
        self.c_proj = CastedLinear(hidden, d, zero_init=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden: int):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg.n_embd, hidden)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))  # x <- l0 * x + l1 * x0
        self.x0_inject = bool(cfg.extra.get("x0_inject", True))

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, x0: torch.Tensor) -> torch.Tensor:
        if self.x0_inject:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), ve)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        ex = cfg.extra
        d, V, L = cfg.n_embd, cfg.vocab_size, cfg.n_layer
        hidden = int(round(cfg.mlp_ratio * d))
        self.embed = nn.Embedding(V, d)
        n_ve = int(ex.get("n_value_embeds", max(1, round(L / 4))))
        self.n_ve = max(0, min(n_ve, L // 2 if L > 1 else 1))
        self.value_embeds = nn.ModuleList(nn.Embedding(V, d) for _ in range(self.n_ve))
        self.blocks = nn.ModuleList(Block(cfg, hidden) for _ in range(L))
        self.lm_head = CastedLinear(d, V)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=float(ex.get("head_init_std", 0.005)))
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.unet = bool(ex.get("unet", True))
        self.n_enc = L // 2
        self.skip_weights = nn.Parameter(torch.ones(max(self.n_enc, 1)))
        self.softcap = float(ex.get("softcap", 15.0))
        self.embnorm = bool(ex.get("embnorm", True))

    # ---- split API ------------------------------------------------------------------------------
    def in_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids)

    def out_weight(self) -> torch.Tensor:
        return self.lm_head.weight

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        L = self.cfg.n_layer
        e = self.embed(idx)
        x = x0 = norm(e) if self.embnorm else e
        ves = [emb(idx) for emb in self.value_embeds]
        ve_at: list[torch.Tensor | None] = [None] * L
        for i, ve in enumerate(ves):  # first n_ve layers and last n_ve layers share the tables (012...012)
            ve_at[i] = ve
            ve_at[L - self.n_ve + i] = ve
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            if self.unet and i >= L - self.n_enc and skips:
                x = x + self.skip_weights[i - (L - self.n_enc)] * skips.pop()
            x = block(x, ve_at[i], x0)
            if self.unet and i < self.n_enc:
                skips.append(x)
        return norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        z = self.lm_head(h).float()
        if self.softcap > 0:
            z = self.softcap * torch.tanh(z / self.softcap)
        return z

    # ---- contract -------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        z = self.logits(self.hidden(idx))
        B, T, V = z.shape
        if self.training:
            loss = F.cross_entropy(z.view(B * T, V), targets.reshape(B * T), ignore_index=-1)
            return loss, {}
        ce = F.cross_entropy(z.view(B * T, V), targets.reshape(B * T), ignore_index=-1, reduction="none").view(B, T)
        n = (targets >= 0).sum().clamp_min(1)
        return ce.sum() / n, {"per_token_loss": ce.detach()}

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for p in self.blocks.parameters() if p.ndim == 2]
        ids = {id(p) for p in muon}
        adam = [p for p in self.parameters() if id(p) not in ids]
        return {"muon": muon, "adam": adam}

    def flops_per_token(self) -> float:
        c = self.cfg
        n_mat = sum(p.numel() for p in self.blocks.parameters() if p.ndim == 2) + self.lm_head.weight.numel()
        attn = 6 * c.n_layer * c.seq_len * c.n_embd  # causal QK^T + AV, forward+backward, per token
        return 6.0 * n_mat + attn


def build(cfg: ModelConfig) -> nn.Module:
    return GPT(cfg)
