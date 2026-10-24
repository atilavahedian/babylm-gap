"""A2 - Answer delta layer (AD).  Idea card: sprint1/ideas/A_answer_delta.md

An in-window, label-supervised fast-weight layer placed before the LM head.  For every past position s
it writes  key k_s = f(h_s)  ->  value = g(embedding of x_{s+1}), i.e. "in a context like s, the answer
was x_{s+1}", into a per-head delta-rule matrix (error-correcting overwrite, exact chunk-parallel form).
Position t reads with a one-step lag, o_t = S_{t-1}^T q_t, so it only uses answers x_{s+1} with
s+1 <= t, which are inputs it already has (no target is ever read; causal by construction).
This is closed-form in-context dynamic evaluation as a layer: a linear regressor from context state to
next-token embedding, fitted online on the window by the delta rule, its readout trained end-to-end.

It is an IN-WINDOW mechanism: everything it reads was in the context. Its distinctive claim is soft
generalisation (similar contexts -> answers; answer embeddings -> semantically near unseen words); the
critic should kill it if its gain sits only on repeated tokens or fades at 2x/4x budget.

Knobs (cfg.extra):
  null: bool=False        control: the written "answers" come from a fixed random token block drawn once at
                          build time (same params/FLOPs; no information about any input or answer; it never
                          reads another batch row, which at eval can be the previous window of the same text)
  ad_mode: "answer"|"self"  "self" = known control: plain DeltaNet layer at the same place
                          (value = W_v h_s, read includes the current write)
  ad_heads: int=n_head    heads; key/value width = n_embd / heads
  ad_chunk: int=64        chunk length of the parallel form
  ad_code: "out"|"in"     answer code: the answer's LM-head row (default, decodable by the head) or its
                          RMS-normed input embedding; W_v and W_o start as identity, the output gain at 0
  body: "auto"|"baseline"|"mini";  a_equal_params: bool=True
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sprint1.harness.model_api import ModelConfig
from sprint1.models.a_common import (answer_code, build_body, delta_rule_chunked, l2n, lm_loss_aux, make_null_tokens,
                                     null_like, split_groups)


class AnswerDeltaLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        e = cfg.extra
        d = cfg.n_embd
        self.null = bool(e.get("null", False))
        self.mode = str(e.get("ad_mode", "answer"))
        self.H = int(e.get("ad_heads", cfg.n_head))
        self.chunk = int(e.get("ad_chunk", 64))
        self.d, self.dk = d, d // self.H
        extra = 4 * d * d + d * self.H + d + 1
        self.body = build_body(cfg, extra_params=extra)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wb = nn.Linear(d, self.H, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.wv.weight)                 # value = answer code (LM-head row), per-head slices
        nn.init.eye_(self.wo.weight)
        self.gain = nn.Parameter(torch.zeros(()))    # read enters the hidden state through a zero-init gain
        self.gate = nn.Linear(d, 1, bias=False)      # plus a zero-init context-dependent part
        nn.init.zeros_(self.gate.weight)
        self.code = str(e.get("ad_code", "out"))
        if self.null:
            self.register_buffer("null_ids", make_null_tokens(cfg.vocab_size), persistent=False)
        self.cfg = cfg

    def mix(self, h, idx):
        B, T, d = h.shape
        H, dk = self.H, self.dk
        q = l2n(self.wq(h).view(B, T, H, dk).float())
        k = l2n(self.wk(h).view(B, T, H, dk).float())
        beta = torch.sigmoid(self.wb(h).float())                            # [B,T,H]
        if self.mode == "self":
            v = self.wv(h)
            lagged = False
        else:
            src = null_like(self.null_ids, B, T) if self.null else idx
            ans = torch.cat([src[:, 1:], src[:, :1]], dim=1)                # x_{s+1}; last write never read
            v = self.wv(answer_code(self.body, ans, self.code).to(h.dtype))
            lagged = True
        v = v.view(B, T, H, dk).float()
        o = delta_rule_chunked(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), beta.transpose(1, 2),
                               chunk=self.chunk, lagged=lagged)
        return (self.gain + self.gate(h)) * self.wo(o.transpose(1, 2).reshape(B, T, d).to(h.dtype))

    def forward_logits(self, idx):
        """Logits [B, T, V] (for per-token / novelty-split scoring)."""
        h = self.body.hidden(idx)
        return self.body.logits(h + self.mix(h, idx))

    def forward(self, idx, targets):
        return lm_loss_aux(self.forward_logits(idx), targets, self.training)

    def param_groups(self):
        return split_groups(self, self.body, adam_extra=[])          # all added matrices (incl. wb, gate) -> muon

    def flops_per_token(self):
        d, T, C = self.d, self.cfg.seq_len, self.chunk
        mech = 6.0 * (4 * d * d + d * self.H + d)
        # chunk form per token: K K^T and Q K^T (2*C*dk each per head), two triangular solves (~C*dk),
        # inter-chunk W S, Q S, K^T U (3*dk*dk per head): x2 flops per MAC, x3 for fwd+bwd
        mech += 3.0 * 2.0 * self.H * (4 * C * self.dk + 3 * self.dk * self.dk)
        return float(self.body.body_flops_per_token()) + mech


def build(cfg: ModelConfig) -> nn.Module:
    return AnswerDeltaLM(cfg)
