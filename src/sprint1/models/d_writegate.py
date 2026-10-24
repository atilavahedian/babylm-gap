"""D2 WriteGate: learned write-time KV admission. Every token is visible to the next W queries
(sliding window); beyond the window a token is visible only if its per-layer write gate admitted it
when it was written. Admission rate rho is fixed by a quantile threshold, so the inference KV cache per
layer is W + rho * (ctx - W) entries instead of ctx.

Idea card: sprint1/ideas/D_writegate.md. Contract: sprint1/harness/model_api.py.

Training (mode "learned"): far keys get an additive attention-logit bias log sigmoid((a_j - tau)/temp),
a_j = gate score of key token j at this layer (computed from the token's own normalised layer input,
so admission is decided causally at write time), tau = batch (1-rho)-quantile of a. temp anneals
temp0 -> temp1 through training (on_step_end); during the final `hard_frac` of training the forward
uses the exact inference mask (admitted 0 / rejected -inf) with soft straight-through gradients.
Eval / inference: hard mask with tau = EMA of training thresholds (causal, per token).

Controls (cfg.extra["wg_mode"]):
  "learned" (default)  the candidate
  "random"  NULL CONTROL (also cfg.extra["null"]=True): admission by a parameter-free hash of
            (token id, position, layer) at the same rate rho, hard in training and eval
  "stride"  admit every round(1/rho)-th position (positional summary tokens, same rate)
  "local"   window only (rho = 0)
  "dense"   ordinary causal attention (same code path; the matched dense reference)
Knobs: window (default max(16, seq_len // 16)), rho (0.125), temp0 (1.0), temp1 (0.05), hard_frac (0.2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.models.d_common import (Block, RaggedKV, attend, attend_parts, block_matmul_params, hash_uniform, lm_loss, lm_loss_eval, rms,
                                     softcap, split_param_groups, sum_bytes)


class RingKV:
    """Sliding-window cache of the last W positions plus a compact store of admitted older positions."""

    def __init__(self, B, H, hd, W, device, dtype):
        self.W = W
        self.k = torch.zeros(B, H, W, hd, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.adm = torch.zeros(B, W, dtype=torch.bool, device=device)
        self.valid = torch.zeros(W, dtype=torch.bool, device=device)
        self.glob = RaggedKV(B, H, hd, 64, device, dtype)

    def push(self, k, v, adm, pos):
        """k, v [B, H, 1, hd] at absolute position pos; adm [B] bool."""
        slot = pos % self.W
        if pos >= self.W:  # evict position pos - W; admitted rows move to the global store
            rows = self.adm[:, slot].nonzero(as_tuple=True)[0]
            if rows.numel():
                self.glob.append(self.k[rows, :, slot:slot + 1], self.v[rows, :, slot:slot + 1], rows)
        self.k[:, :, slot] = k[:, :, 0].to(self.k.dtype)
        self.v[:, :, slot] = v[:, :, 0].to(self.v.dtype)
        self.adm[:, slot] = adm
        self.valid[slot] = True

    def attend(self, q):
        B = q.shape[0]
        parts = [(self.k, self.v, self.valid[None].expand(B, -1))]
        if int(self.glob.n.max().item()) > 0:
            parts.append(self.glob.view())
        return attend_parts(q, parts)

    def bytes(self):
        el = self.k.element_size()
        H, hd = self.k.shape[1], self.k.shape[3]
        g = self.glob.bytes()
        ring_alloc = self.k.numel() * 2 * el
        ring_used = int(self.valid.sum().item()) * self.k.shape[0] * H * hd * 2 * el
        return dict(allocated=ring_alloc + g["allocated"], used=ring_used + g["used"])


class WriteGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ex = dict(cfg.extra or {})
        self.cfg = cfg
        d, V, L = cfg.n_embd, cfg.vocab_size, cfg.n_layer
        self.d, self.V, self.T, self.L = d, V, cfg.seq_len, L
        mode = "random" if ex.get("null", False) else ex.get("wg_mode", "learned")
        assert mode in ("learned", "random", "stride", "local", "dense"), mode
        self.mode = mode
        self.W = int(ex.get("window", max(16, cfg.seq_len // 16)))
        self.rho = 0.0 if mode == "local" else float(ex.get("rho", 0.125))
        self.temp0, self.temp1 = float(ex.get("temp0", 1.0)), float(ex.get("temp1", 0.05))
        self.hard_frac = float(ex.get("hard_frac", 0.2))
        # schedule state lives in buffers (not python attrs) so torch.compile does not recompile per step
        self.register_buffer("temp_t", torch.tensor(self.temp0), persistent=True)
        self.register_buffer("hard_t", torch.tensor(0.0), persistent=True)
        self.embed = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, cfg.n_head, cfg.mlp_ratio) for _ in range(L)])
        self.lm_head = nn.Linear(d, V, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        self.gate_w = nn.Parameter(torch.randn(L, d) * 0.02)
        self.gate_b = nn.Parameter(torch.zeros(L))
        self.register_buffer("tau", torch.zeros(L), persistent=True)
        self.register_buffer("tau_init", torch.zeros(L, dtype=torch.bool), persistent=True)

    # ---------------------------------------------------------------- admission
    def _score(self, x_norm, l, idx, pos):
        if self.mode == "random":
            return hash_uniform(idx, pos, salt=101 + l)
        return x_norm.float() @ self.gate_w[l].float() + self.gate_b[l].float()

    def _admit_fixed(self, a, l, pos):  # eval / decode path (not compiled in training)
        """Hard admission for the non-learned modes and for learned mode at eval. a [B, T]."""
        if self.mode == "random":
            return a >= 1.0 - self.rho
        if self.mode == "stride":
            s = max(1, int(round(1.0 / self.rho)))
            return ((pos % s) == (s - 1))[None, :].expand_as(a)
        if self.mode == "local":
            return torch.zeros_like(a, dtype=torch.bool)
        if not bool(self.tau_init[l]):  # untrained model: calibrate on this batch (documented)
            k = max(1, int(round(self.rho * a.numel())))
            self.tau[l] = torch.topk(a.detach().flatten(), k).values[-1]
            self.tau_init[l] = True
        return a >= self.tau[l]

    def _far_bias(self, a, l, pos):
        """Per-key additive bias for keys outside the window, [B, T] in float32."""
        if self.mode == "learned" and self.training:
            k = max(1, int(round(self.rho * a.numel())))
            thr = torch.topk(a.detach().flatten(), k).values[-1]
            with torch.no_grad():
                new = torch.where(self.tau_init[l], 0.9 * self.tau[l] + 0.1 * thr, thr)
                self.tau[l] = new
                self.tau_init[l] = True
            soft = F.logsigmoid((a - thr) / self.temp_t)
            # hard phase: forward = exact inference mask (finite -1e4 so 0*value stays finite), backward = soft
            hard = torch.where(a >= thr, 0.0, -1e4)
            return soft + self.hard_t * (hard - soft.detach())
        adm = self._admit_fixed(a, l, pos)
        return torch.where(adm, 0.0, float("-inf")).to(torch.float32)

    def _mask(self, far, T, dtype, device):
        t = torch.arange(T, device=device)
        dist = t[:, None] - t[None, :]
        local = (dist >= 0) & (dist < self.W)
        farm = dist >= self.W
        neg = torch.tensor(float("-inf"), device=device)
        bias = torch.where(local[None], torch.zeros((), device=device),
                           torch.where(farm[None], far[:, None, :], neg))
        return bias[:, None].to(dtype)

    # ---------------------------------------------------------------- trunk
    def _trunk(self, idx, collect=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = rms(self.embed(idx))
        kvs, adms = [], []
        for l, blk in enumerate(self.blocks):
            x = rms(h)
            if self.mode == "dense" or T <= self.W:
                mask, adm = None, None
            else:
                a = self._score(x, l, idx, pos)
                if self.training and self.mode == "learned":
                    far = self._far_bias(a, l, pos)
                    mask = self._mask(far, T, torch.float32, idx.device)
                    adm = None
                else:  # hard admission: exact boolean mask (4x smaller than a float bias)
                    adm = self._admit_fixed(a, l, pos)
                    t = torch.arange(T, device=idx.device)
                    dist = t[:, None] - t[None, :]
                    mask = (((dist >= 0) & (dist < self.W))[None] | ((dist >= self.W)[None] & adm[:, None, :]))[:, None]
            a_out, kv = blk.attn(x, pos, mask)
            h = h + a_out
            h = h + blk.mlp(rms(h))
            if collect:
                if adm is None and self.mode != "dense":
                    adm = self._admit_fixed(self._score(x, l, idx, pos), l, pos)
                kvs.append(kv)
                adms.append(adm)
        return h, kvs, adms

    def _head(self, h):
        return softcap(self.lm_head(rms(h)))

    def forward(self, idx, targets):
        h, _, _ = self._trunk(idx)
        aux = {"temp": self.temp_t.detach().clone(), "hard": self.hard_t.detach().clone()}
        if self.training:
            loss = lm_loss(self._head(h), targets)
        else:
            loss, aux["per_token_loss"] = lm_loss_eval(self._head(h), targets)
        return loss, aux

    def on_step_end(self, step, total_steps):
        frac = min(1.0, (step + 1) / max(1, total_steps))
        temp = self.temp0 * (self.temp1 / self.temp0) ** min(1.0, frac / max(1e-6, 1.0 - self.hard_frac))
        self.temp_t.fill_(temp)
        self.hard_t.fill_(1.0 if frac >= 1.0 - self.hard_frac else 0.0)

    @property
    def temp(self):
        return float(self.temp_t)

    @property
    def hard(self):
        return bool(self.hard_t > 0)

    def param_groups(self):
        return split_param_groups(self)

    # ---------------------------------------------------------------- cost accounting
    def _keys_eff_train(self):
        return self.T  # training attention is dense-masked (soft gates see every key)

    def keys_eff(self, ctx):
        if self.mode == "dense":
            return float(ctx)
        return float(min(ctx, self.W) + self.rho * max(0, ctx - self.W))

    def flops_per_token(self):
        Pb = block_matmul_params(self.d, self.cfg.mlp_ratio)
        return float(6 * (self.L * Pb + self.d * self.V + self.L * self.d) + 6 * self.d * self.T * self.L)

    def inference_flops_per_token(self, ctx_len):
        Pb = block_matmul_params(self.d, self.cfg.mlp_ratio)
        return float(2 * (self.L * Pb + self.d * self.V) + 4 * self.d * self.L * self.keys_eff(ctx_len))

    # ---------------------------------------------------------------- decoding
    def forward_logits(self, idx):
        h, _, _ = self._trunk(idx)
        return self._head(h)

    @torch.no_grad()
    def prefill(self, idx):
        B, T = idx.shape
        H, hd = self.cfg.n_head, self.d // self.cfg.n_head
        dev, dt = idx.device, self.embed.weight.dtype
        h, kvs, adms = self._trunk(idx, collect=True)
        caches = []
        for (k, v), adm in zip(kvs, adms):
            if self.mode == "dense":
                c = RaggedKV(B, H, hd, T + 256, dev, dt)
                c.append(k, v)
            else:
                c = RingKV(B, H, hd, self.W, dev, dt)
                cut = max(0, T - self.W)
                if cut > 0:
                    c.glob.append_masked(k[:, :, :cut], v[:, :, :cut], adm[:, :cut])
                for p in range(cut, T):
                    s = p % self.W
                    c.k[:, :, s] = k[:, :, p].to(dt)
                    c.v[:, :, s] = v[:, :, p].to(dt)
                    c.adm[:, s] = adm[:, p]
                    c.valid[s] = True
            caches.append(c)
        return self._head(h[:, -1]), dict(pos=T, layers=caches)

    @torch.no_grad()
    def decode_step(self, tok, cache):
        p = cache["pos"]
        pt = torch.tensor([p], device=tok.device)
        h = rms(self.embed(tok))
        for l, (blk, c) in enumerate(zip(self.blocks, cache["layers"])):
            x = rms(h)
            q, k, v = blk.attn.heads(x, pt)
            if self.mode == "dense":
                c.append(k, v)
                K, Vv, valid = c.view()
                y = attend(q, K, Vv, valid)
            else:
                adm = self._admit_fixed(self._score(x, l, tok, pt), l, pt)[:, 0]
                c.push(k, v, adm, p)
                y = c.attend(q)
            h = h + blk.attn.out(y)
            h = h + blk.mlp(rms(h))
        cache["pos"] = p + 1
        return self._head(h[:, 0]), cache

    def cache_bytes(self, cache):
        return sum_bytes(cache["layers"])


def build(cfg):
    return WriteGate(cfg)
