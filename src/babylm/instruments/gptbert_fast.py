"""Exact, faster relative-position attention for the vendored GPT-BERT.

The published code materialises a [T, T, 2D] position tensor per layer and
contracts it with einsum, which dominates CPU time. Each (q, k) pair only
uses one of 2*bucket-1 position rows, so the same scores are obtained by
projecting queries/keys onto the position rows ([B, H, T, NB]) and gathering
by bucket index. patch() swaps the method in; check() asserts equality with
the original on random input.
"""
from __future__ import annotations

import torch

from babylm.instruments.vendor_gptbert import modeling_gpt_bert as M

_original = M.Attention.forward


def _fast_forward(self, hidden_states, attention_mask, relative_embedding):
    key_len, batch_size, _ = hidden_states.size()
    query_len = key_len
    if self.position_indices.size(0) < query_len:
        return _original(self, hidden_states, attention_mask, relative_embedding)
    H, Dh = self.num_heads, self.head_size
    x = self.pre_layer_norm(hidden_states)
    query, key = self.in_proj_qk(x).chunk(2, dim=2)  # [T, B, D]
    value, gate = self.in_proj_vg(x).chunk(2, dim=2)
    gate = torch.nn.functional.gelu(gate)
    pos = self.in_proj_qk(self.dropout(relative_embedding))  # [NB, 2D]
    qpos, kpos = pos.chunk(2, dim=-1)
    NB = pos.size(0)
    qpos = qpos.reshape(NB, H, Dh)
    kpos = kpos.reshape(NB, H, Dh)
    idx = self.position_indices[:query_len, :key_len]  # [Q, K] bucket ids

    q = query.reshape(query_len, batch_size * H, Dh).transpose(0, 1)  # [BH, Q, Dh]
    k = key.reshape(key_len, batch_size * H, Dh).transpose(0, 1)
    v = value.reshape(key_len, batch_size * H, Dh).transpose(0, 1)
    scores = torch.bmm(q, k.transpose(1, 2) * self.scale).view(batch_size, H, query_len, key_len)
    q4 = q.view(batch_size, H, query_len, Dh)
    k4 = k.view(batch_size, H, key_len, Dh)
    a1 = torch.einsum("bhqd,jhd->bhqj", q4, kpos * self.scale)  # query . key_pos
    a2 = torch.einsum("bhkd,jhd->bhkj", k4 * self.scale, qpos)  # key . query_pos
    scores = scores + torch.gather(a1, 3, idx.expand(batch_size, H, query_len, key_len))
    scores = scores + torch.gather(a2, 3, idx.t().expand(batch_size, H, key_len, query_len)).transpose(2, 3)
    probs = M.MaskedSoftmax.apply(scores, attention_mask, -1)
    probs = self.dropout(probs)
    out = torch.bmm(probs.flatten(0, 1), v)  # [BH, Q, Dh]
    out = out.transpose(0, 1).reshape(query_len, batch_size, self.hidden_size)
    out = out * gate
    out = self.post_layer_norm(out)
    out = self.out_proj(out)
    out = self.dropout(out)
    return out, probs


def patch() -> None:
    M.Attention.forward = _fast_forward


def unpatch() -> None:
    M.Attention.forward = _original


@torch.inference_mode()
def check(model, seq_len: int = 300, batch: int = 3, tol: float = 1e-4) -> float:
    x = torch.randint(5, 16000, (batch, seq_len))
    unpatch()
    ref = model(input_ids=x)[0].float()
    patch()
    new = model(input_ids=x)[0].float()
    err = float((ref - new).abs().max())
    if err > tol * max(1.0, float(ref.abs().max())):
        raise AssertionError(f"fast GPT-BERT attention differs by {err}")
    return err
