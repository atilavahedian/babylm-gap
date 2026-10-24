"""KV-cache decoding and sampling for the sprint-1 GPT family (attack A4's generators, sizes S and M).

forward_cached(model, idx, cache, pos) runs sprint1.models.baseline.GPT exactly as GPT.hidden + GPT.logits do, for
new tokens idx [B, t] at absolute positions pos..pos+t-1, reading and appending the keys and values of every layer
in a preallocated cache: same embedding (RMS-normed unless embnorm=False), value embeddings at the first and last
n_ve layers, x0 injection, U-net skips, QK-norm, half-truncated RoPE at the absolute positions, the softcapped fp32
logits. Works for every subclass whose eval-mode forward is the baseline's: babylm.models.baseline_reg (dropout is
off in eval) and babylm.models.switchable (causal unless flipped). Prefill (pos = 0, t > 1) uses the causal mask;
each later call adds one token (t = 1) attending to the whole cache. The model must be in eval mode.

sample(...) prefills a batch of equal-length prompts and draws n_new tokens per row at temperature T from the full
softmax (no top-k / top-p), with ids >= allowed_vocab masked, an optional BlockIndex (babylm.a4.ngram) removing, from
step block_from on, every token that would complete an indexed n-gram, an optional ban on the stop id for the first
ban_stop_steps steps, and a stop id (BOS) that ends a row. Rows that stopped keep decoding (cheaper than reshaping the cache); their tokens
after the stop are ignored by the caller via the returned lengths.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from sprint1.models.baseline import GPT, norm


class KVCache:
    def __init__(self, n_layer: int, batch: int, n_head: int, max_len: int, head_dim: int,
                 device: torch.device | str, dtype: torch.dtype):
        shape = (batch, n_head, max_len, head_dim)
        self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.max_len = max_len


def new_cache(model: GPT, batch: int, max_len: int, device, amp=None) -> KVCache:
    """A cache in the compute dtype of the attention inputs (bf16 under CUDA autocast, else fp32)."""
    a0 = model.blocks[0].attn
    with amp or contextlib.nullcontext():
        dt = a0.c_q(torch.zeros(1, 1, model.cfg.n_embd, device=device)).dtype
    return KVCache(model.cfg.n_layer, batch, a0.n_head, max_len, a0.head_dim, device, dt)


def _rope_at(rot, x: torch.Tensor, pos: int) -> torch.Tensor:
    """sprint1 Rotary.forward for x [B, t, H, D] at absolute positions pos..pos+t-1."""
    end = pos + x.size(1)
    if end > rot.cos.size(0):
        rot._build(end)
    cos = rot.cos[None, pos:end, None, :]
    sin = rot.sin[None, pos:end, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1).type_as(x)


def _attn_cached(a, x: torch.Tensor, ve: torch.Tensor | None, cache: KVCache, layer: int, pos: int) -> torch.Tensor:
    B, t, d = x.shape
    H, D = a.n_head, a.head_dim
    q = a.c_q(x).view(B, t, H, D)
    k = a.c_k(x).view(B, t, H, D)
    v = a.c_v(x).view(B, t, H, D)
    dt = q.dtype
    q, k = _rope_at(a.rotary, norm(q), pos).to(dt), _rope_at(a.rotary, norm(k), pos).to(dt)
    v = a.lambdas[0] * v + a.lambdas[1] * ve.view_as(v) if ve is not None else a.lambdas[0] * v
    K, Vc = cache.k[layer], cache.v[layer]
    K[:, :, pos:pos + t] = k.transpose(1, 2).to(K.dtype)
    Vc[:, :, pos:pos + t] = v.to(dt).transpose(1, 2).to(Vc.dtype)
    kk, vv = K[:, :, :pos + t], Vc[:, :, :pos + t]
    if t > 1 and pos != 0:
        raise ValueError("multi-token calls are prefill only (pos = 0)")
    y = F.scaled_dot_product_attention(q.transpose(1, 2), kk.to(dt), vv.to(dt), is_causal=t > 1, scale=a.scale)
    return a.c_proj(y.transpose(1, 2).reshape(B, t, d))


@torch.no_grad()
def forward_cached(model: GPT, idx: torch.Tensor, cache: KVCache, pos: int) -> torch.Tensor:
    """fp32 softcapped logits [B, t, V] for tokens idx [B, t] at positions pos..pos+t-1; fills the cache there."""
    if model.training:
        raise ValueError("forward_cached needs model.eval() (dropout off)")
    if pos + idx.size(1) > cache.max_len:
        raise ValueError("KV cache too short")
    L = model.cfg.n_layer
    e = model.embed(idx)
    x = x0 = norm(e) if model.embnorm else e
    ves = [emb(idx) for emb in model.value_embeds]
    ve_at: list[torch.Tensor | None] = [None] * L
    for i, ve in enumerate(ves):
        ve_at[i] = ve
        ve_at[L - model.n_ve + i] = ve
    skips: list[torch.Tensor] = []
    for i, block in enumerate(model.blocks):
        if model.unet and i >= L - model.n_enc and skips:
            x = x + model.skip_weights[i - (L - model.n_enc)] * skips.pop()
        if block.x0_inject:
            x = block.lambdas[0] * x + block.lambdas[1] * x0
        x = x + _attn_cached(block.attn, norm(x), ve_at[i], cache, i, pos)
        x = x + block.mlp(norm(x))
        if model.unet and i < model.n_enc:
            skips.append(x)
    return model.logits(norm(x))


def generation_flops(model: GPT, prompt_len: int, n_new: int) -> float:
    """Forward FLOPs of one sampled row: 2 x matrix params per processed token plus 4 x n_layer x d x context per token
    (QK^T and AV), over the prompt and every new token but the last (which is sampled, never processed)."""
    c = model.cfg
    emb_ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    n_mat = sum(p.numel() for p in model.parameters() if p.ndim >= 2 and id(p) not in emb_ids)
    if c.tie_embeddings:
        n_mat += model.lm_head.weight.numel()
    n = prompt_len + n_new - 1
    return 2.0 * n_mat * n + 4.0 * c.n_layer * c.n_embd * n * (n + 1) / 2


@dataclass
class SampleOut:
    tokens: torch.Tensor          # [B, n_new] sampled ids (garbage after a row's stop)
    lengths: torch.Tensor         # [B] tokens before the stop id (n_new if the row never stopped)
    stopped: torch.Tensor         # [B] bool, the row sampled the stop id
    blocked_steps: torch.Tensor   # [B] steps (before the stop) at which the block removed at least one token
    blocked_mass: torch.Tensor    # [B] summed probability mass the block removed (before the stop)
    stats: dict = field(default_factory=dict)


@torch.no_grad()
def self_check(model: GPT, prompts: torch.Tensor, steps: int = 8, amp=None) -> float:
    """Max |logit difference| between cached decoding and the plain forward (GPT.hidden + GPT.logits) over the same
    tokens: the prompts' last position, then `steps` greedy tokens."""
    amp = amp or contextlib.nullcontext()
    B, P = prompts.shape
    cache = new_cache(model, B, P + steps, prompts.device, amp)
    worst = 0.0
    with amp:
        lc = forward_cached(model, prompts, cache, 0)[:, -1].float()
        seq = prompts
        for j in range(steps + 1):
            full = model.logits(model.hidden(seq))[:, -1].float()
            worst = max(worst, float((full - lc).abs().max()))
            if j == steps:
                break
            tok = lc.argmax(-1)
            seq = torch.cat([seq, tok[:, None]], 1)
            lc = forward_cached(model, tok[:, None], cache, P + j)[:, -1].float()
    return worst


@torch.no_grad()
def sample(model: GPT, prompts: torch.Tensor, n_new: int, *, temperature: float = 1.0, allowed_vocab: int = 16384,
           block=None, block_from: int = 0, stop_id: int | None = 0, ban_stop_steps: int = 0,
           generator: torch.Generator | None = None, amp=None) -> SampleOut:
    """Draw n_new tokens after each prompt row (prompts [B, P], all rows the same length P >= 1)."""
    if n_new < 1:
        raise ValueError("n_new must be >= 1")
    amp = amp or contextlib.nullcontext()
    B, P = prompts.shape
    dev = prompts.device
    cache = new_cache(model, B, P + n_new, dev, amp)
    with amp:
        logits = forward_cached(model, prompts, cache, 0)[:, -1].float()
    out = torch.zeros(B, n_new, dtype=torch.long, device=dev)
    lengths = torch.full((B,), n_new, dtype=torch.long, device=dev)
    stopped = torch.zeros(B, dtype=torch.bool, device=dev)
    blocked_steps = torch.zeros(B, dtype=torch.long, device=dev)
    blocked_mass = torch.zeros(B, dtype=torch.float64, device=dev)
    ctx = prompts[:, -(block.n - 1):].clone() if block is not None else None
    V = logits.shape[-1]
    for j in range(n_new):
        lg = logits / temperature if temperature != 1.0 else logits.clone()
        if allowed_vocab < V:
            lg[:, allowed_vocab:] = float("-inf")
        if stop_id is not None and j < ban_stop_steps:
            lg[:, stop_id] = float("-inf")
        if block is not None and j >= block_from and ctx.shape[1] >= block.n - 1:
            p_before = torch.softmax(lg, -1)
            nb = block.mask(lg, ctx)
            removed = (p_before * ~torch.isfinite(lg)).sum(1)
            live = ~stopped
            blocked_steps += ((nb > 0) & live).long()
            blocked_mass += torch.where(live, removed.double(), torch.zeros_like(blocked_mass))
        probs = torch.softmax(lg, -1)
        tok = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        out[:, j] = tok
        if stop_id is not None:
            new = (tok == stop_id) & ~stopped
            lengths[new] = j
            stopped |= new
        if block is not None:
            ctx = torch.cat([ctx[:, 1:], tok[:, None]], 1) if ctx.shape[1] >= block.n - 1 else \
                torch.cat([ctx, tok[:, None]], 1)
        if bool(stopped.all()) or j + 1 == n_new:
            break
        with amp:
            logits = forward_cached(model, tok[:, None], cache, P + j)[:, -1].float()
    return SampleOut(out, lengths, stopped, blocked_steps, blocked_mass,
                     {"steps": j + 1, "flops_per_row": generation_flops(model, P, j + 1)})

