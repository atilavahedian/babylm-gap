"""A3 - Delta hybrid (KNOWN architecture family) aimed at the never-attempted "4x less inference memory" bar.

Replaces A2 (answer delta, withdrawn as copy family; see sprint1/ideas/A_answer_delta.md).  The body is
the a_common MiniGPT (modded-nanogpt-style) in which every layer except `dh_attn_layers` swaps causal
softmax attention for a DeltaNet mixer: short causal depthwise conv + SiLU on q/k/v, L2-normalised q/k,
beta = sigmoid(W_b x), exact chunk-parallel delta rule (a_common.delta_rule_chunked), per-head RMS norm,
zero-init output projection.  A delta layer keeps a constant H x dk x dk state instead of a KV cache that
grows with context, so at small (1 attention layer of 6) the decode-time cache is ~5x smaller at 1,024
tokens and the gap grows linearly with context.  Equal parameters with a_plain within 0.1%.

Compare against sprint1/models/a_plain.py (same body, all attention); baseline.py is a different body.

Knobs (cfg.extra):
  null: bool=False           control: same model, same params/FLOPs/state, but the delta state is reset at
                             every chunk boundary (dh_chunk tokens), so delta layers see only local context;
                             tests whether the constant-size long-range state carries information
  dh_attn_layers: str="mid"  comma list of layer indices that keep softmax attention, "mid" = [n_layer//2],
                             "none" = attention-free
  dh_chunk: int=64           chunk length of the parallel delta rule
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.a_common import MiniGPT, delta_rule_chunked, l2n, rms


class DeltaMixer(nn.Module):
    def __init__(self, d: int, n_head: int, chunk: int, carry: bool):
        super().__init__()
        self.h, self.hd, self.chunk, self.carry = n_head, d // n_head, chunk, carry
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        # depthwise causal conv taps, stored FLAT: the harness puts >=2-D Adam params in the embedding bucket
        # (LR 0.78); a 1-D tensor gets the scalar bucket (LR 0.052), the sane rate for 4-tap filters
        self.conv_taps = nn.Parameter(torch.empty(3 * d * 4).uniform_(-0.5, 0.5))
        self.wb = nn.Linear(d, n_head, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x, v0):
        B, T, d = x.shape
        z = self.qkv(x).transpose(1, 2)                                     # [B, 3d, T]
        w = self.conv_taps.view(3 * d, 1, 4).to(z.dtype)
        z = F.silu(F.conv1d(F.pad(z, (3, 0)), w, groups=3 * d)).transpose(1, 2)   # causal: left pad only
        q, k, v = z.reshape(B, T, 3, self.h, self.hd).unbind(2)
        beta = torch.sigmoid(self.wb(x).float())                            # [B, T, H]
        o = delta_rule_chunked(l2n(q.float()).transpose(1, 2), l2n(k.float()).transpose(1, 2),
                               v.float().transpose(1, 2), beta.transpose(1, 2), chunk=self.chunk,
                               carry=self.carry)                            # [B, H, T, hd]
        y = rms(o.transpose(1, 2)).reshape(B, T, d)
        return self.proj(y.to(x.dtype)), v0


class DeltaHybridGPT(MiniGPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg, int(round(cfg.mlp_ratio * cfg.n_embd)))
        e = cfg.extra
        L = cfg.n_layer
        spec = str(e.get("dh_attn_layers", "mid"))
        if spec == "mid":
            keep = {L // 2}
        elif spec == "none":
            keep = set()
        else:
            keep = {int(i) for i in spec.split(",") if i.strip() != ""}
        self.attn_layers = sorted(keep)
        self.chunk = int(e.get("dh_chunk", 64))
        carry = not bool(e.get("null", False))
        for i, blk in enumerate(self.blocks):
            if i not in keep:
                blk.attn = DeltaMixer(cfg.n_embd, cfg.n_head, self.chunk, carry)

    def body_flops_per_token(self):
        cfg = self.cfg
        d, T, H = cfg.n_embd, cfg.seq_len, cfg.n_head
        hd, C = d // H, self.chunk
        mat = sum(p.numel() for blk in self.blocks for p in blk.parameters() if p.ndim == 2)
        mat += self.lm_head.weight.numel()
        n_attn = len(self.attn_layers)
        n_delta = cfg.n_layer - n_attn
        delta = 3.0 * 2.0 * H * (4 * C * hd + 3 * hd * hd) + 6.0 * 3 * d * 4      # chunk form + conv
        return 6.0 * mat + 6.0 * n_attn * T * d + n_delta * delta

    def inference_cache_numbers(self, ctx_len: int) -> dict:
        """Numbers held per sequence at decode time: KV cache of attention layers + delta states (+conv taps)."""
        cfg = self.cfg
        d, H = cfg.n_embd, cfg.n_head
        hd = d // H
        n_attn = len(self.attn_layers)
        kv = n_attn * 2 * d * ctx_len
        state = (cfg.n_layer - n_attn) * (H * hd * hd + 3 * d * 3)
        full = cfg.n_layer * 2 * d * ctx_len
        return {"hybrid": kv + state, "all_attention": full, "ratio": full / (kv + state)}


def build(cfg: ModelConfig) -> nn.Module:
    return DeltaHybridGPT(cfg)
