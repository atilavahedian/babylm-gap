"""F2 RG-Fovea: renormalisation-group foveated memory. Owned by agent F.

Card: sprint1/ideas/F_rgmem.md.

Each attention layer sees
  (a) exact keys/values of the last W tokens (sliding window), and
  (b) a few SUMMARY keys/values of older context, produced by a causal block-spin coarse-graining of the layer's
      own normalised input: leaves = mean of 2^leaf consecutive token states; every coarser level merges two
      adjacent blocks with ONE learned map shared across all scales (the RG step = coarse-grain + rescale):
          Phi(a, b) = rms( (a + b)/2 + W_o relu^2(W_in [a; b]) )          (W_o zero-init: starts as block averaging)
      Token t reads, at every level n, the latest r COMPLETE blocks (end <= t+1) that lie entirely before its
      window (default; rg_strict_far): a foveated memory whose resolution coarsens with distance (block size ~
      distance, scale-invariant).
Summaries are projected by the layer's own k/v weights, rotated by RoPE at their block's first position (default
rg_pos="start"; the relative distance to a level-n summary is then <= (r+1) 2^n: bounded, length-extrapolation-
friendly) and enter ONE softmax together with the window keys.

Decode state per layer: W exact KV + r KV per level + one pending block per level: O(W + r log T), versus O(T).

Arms (cfg.extra):
    rg_mode  "tree" (candidate, default) | "mean" (linear RG: block averaging, no learned map)
             | "window" (window only) | "dense" (full causal attention, same code: matched reference)
    null     True -> tree whose pyramid input is content-free unit-RMS Gaussian noise (rg_null_kind="noise"), so
             summaries carry no context; identical params, kernels and FLOPs.  rg_null_kind="roll" uses another
             batch row's content instead (only safe if batch rows are independent documents).
    rg_window W (default seq_len // 16)   rg_leaf log2 leaf block (default 4 -> 16 tokens)   rg_per_level r (2)
    rg_tie   True: one merge map for all scales (the fixed-point prior); False: one per level
    rg_hidden_ratio merge MLP width / d (2.0)       rg_full_layers  list of layer indices with dense attention
    rg_pos   "start" (default: summary RoPE position = block's first token) | "end" (block's last token)
    rg_strict_far  True (default: read only blocks entirely older than the window) | False (blocks not entirely
             inside it). Defaults changed 6 Oct after the CPU pilot (card): "end" + partial overlap made summaries
             look like recent tokens and cost 0.17 nats vs the null; "start" + strict tied the null.
    rope_base 10000.0
Training uses a dense boolean-masked SDPA over T + N_nodes keys (no custom kernels); the model therefore costs
MORE training FLOPs than causal dense attention. Its claim is decode memory/compute: inference_memory and
inference_flops_per_token are analytic; decode_all() is an exact eager incremental decoder (verified equal to the
parallel forward in tests) whose measured peak state bytes are <= the analytic figure.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.f_common import FGPT, attn_core_flops, rms, rope_freqs, rotate

MODES = ("tree", "mean", "window", "dense")


def _settings(cfg: ModelConfig) -> dict:
    ex = dict(cfg.extra or {})
    mode = ex.get("rg_mode", "tree")
    assert mode in MODES, mode
    null = bool(ex.get("null", False))
    if null:
        assert mode in ("tree", "mean"), "the null control is defined for the summary arms"
    W = int(ex.get("rg_window", max(8, cfg.seq_len // 16)))
    leaf = int(ex.get("rg_leaf", 4))
    assert cfg.seq_len >= (2 << leaf), "seq_len too short for rg_leaf"
    return dict(mode=mode, null=null, null_kind=ex.get("rg_null_kind", "noise"), W=W, leaf=leaf,
                r=int(ex.get("rg_per_level", 2)), tie=bool(ex.get("rg_tie", True)),
                hidden=int(float(ex.get("rg_hidden_ratio", 2.0)) * cfg.n_embd),
                full_layers=tuple(int(i) for i in ex.get("rg_full_layers", ())),
                exclude_in_window=bool(ex.get("rg_exclude_in_window", True)),
                pos=str(ex.get("rg_pos", "start")),
                strict_far=bool(ex.get("rg_strict_far", True)),
                base=float(ex.get("rope_base", 10000.0)))


def n_levels(T: int, leaf: int) -> int:
    """Number of summary levels built for length T: levels leaf..top with at least 2 complete blocks."""
    n = 0
    while (T >> (leaf + n)) >= 2:
        n += 1
    return n


def node_table(T: int, leaf: int, device=None):
    """Concatenated node metadata for all levels: level n, block index k, start, end (python-int derived)."""
    lv, ks = [], []
    for i in range(n_levels(T, leaf)):
        n = leaf + i
        c = T >> n
        lv.append(torch.full((c,), n, dtype=torch.long, device=device))
        ks.append(torch.arange(c, dtype=torch.long, device=device))
    lv, ks = torch.cat(lv), torch.cat(ks)
    size = torch.ones_like(lv) << lv
    return lv, ks, ks * size, (ks + 1) * size


def build_mask(T: int, W: int, leaf: int, r: int, exclude_in_window: bool, with_nodes: bool, device=None,
               strict_far: bool = False):
    """Boolean [T, T + N] (or [T, T]) mask: True = may attend."""
    t = torch.arange(T, device=device)
    tok = (t[None, :] <= t[:, None]) & (t[None, :] > t[:, None] - W)
    if not with_nodes:
        return tok
    lv, ks, start, end = node_table(T, leaf, device)
    P = (t + 1)[:, None]
    latest = (P >> lv[None, :]) - 1                                   # latest complete block index per level
    ok = (ks[None, :] <= latest) & (ks[None, :] > latest - r)
    if strict_far:                                                     # block entirely older than the window
        ok = ok & (end[None, :] <= t[:, None] - W + 1)
    elif exclude_in_window:                                            # block not entirely inside the window
        ok = ok & (start[None, :] <= t[:, None] - W)
    return torch.cat([tok, ok], dim=1)


class Merge(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.inp = nn.Linear(2 * d, hidden, bias=False)
        self.out = nn.Linear(hidden, d, bias=False)
        nn.init.normal_(self.inp.weight, std=0.5 / math.sqrt(2 * d))
        nn.init.zeros_(self.out.weight)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return rms(0.5 * (a + b) + self.out(F.relu(self.inp(torch.cat([a, b], dim=-1))).square()))


class RGAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, s: dict, layer: int):
        super().__init__()
        d, H = cfg.n_embd, cfg.n_head
        self.d, self.H, self.hd, self.s = d, H, d // H, s
        self.full = s["mode"] == "dense" or layer in s["full_layers"]
        self.summ = (not self.full) and s["mode"] in ("tree", "mean")
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.register_buffer("freqs", rope_freqs(self.hd, s["base"]), persistent=False)
        self.learned = self.summ and s["mode"] == "tree"
        if self.learned:
            nl = max(1, n_levels(cfg.seq_len, s["leaf"]) - 1)        # learned merges: levels leaf+1 .. top
            self.merges = nn.ModuleList([Merge(d, s["hidden"]) for _ in range(1 if s["tie"] else nl)])

    def _merge(self, a, b, i):
        if not self.learned:
            return rms(0.5 * (a + b))
        return self.merges[min(i, len(self.merges) - 1)](a, b)

    def pyramid(self, src: torch.Tensor) -> torch.Tensor:
        B, T, d = src.shape
        L0 = self.s["leaf"]
        c = T >> L0
        s = rms(src[:, : c << L0].reshape(B, c, 1 << L0, d).mean(2))
        levels = [s]
        for i in range(n_levels(T, L0) - 1):                           # shape-determined loop count
            n2 = s.shape[1] // 2
            s = self._merge(s[:, 0:2 * n2:2], s[:, 1:2 * n2:2], i)
            levels.append(s)
        return torch.cat(levels, dim=1)                                  # [B, N, d]

    def _rope(self, x, pos):
        f = pos.float()[:, None] * self.freqs[None, :]
        return rotate(x, f.cos(), f.sin())

    def forward(self, h: torch.Tensor, x: torch.Tensor, ctx: dict):
        B, T, _ = h.shape
        H, hd, d = self.H, self.hd, self.d
        q, k, v = self.qkv(h).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T, device=h.device)
        q, k = self._rope(rms(q), pos), self._rope(rms(k), pos)
        if self.full:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif not self.summ:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx["mask_win"])
        else:
            src = h
            if self.s["null"]:
                if self.s["null_kind"] == "roll" and B > 1:
                    src = h.roll(1, 0)
                else:
                    src = rms(torch.randn_like(h))
            S = self.pyramid(src)                                          # [B, N, d]
            N = S.shape[1]
            kvs = F.linear(S, self.qkv.weight[d:]).view(B, N, 2, H, hd).permute(2, 0, 3, 1, 4)
            ks = self._rope(rms(kvs[0]), ctx["node_pos"])
            y = F.scaled_dot_product_attention(q, torch.cat([k, ks], 2), torch.cat([v, kvs[1]], 2),
                                               attn_mask=ctx["mask_tree"])
        return self.proj(y.transpose(1, 2).reshape(B, T, d)), ctx


class RGFoveaGPT(FGPT):
    def __init__(self, cfg: ModelConfig):
        s = _settings(cfg)
        self.s = s
        super().__init__(cfg, lambda i: RGAttention(cfg, s, i))

    def init_ctx(self, idx: torch.Tensor, x0: torch.Tensor) -> dict:
        T, s, dev = idx.shape[1], self.s, idx.device
        ctx = {"idx": idx}
        if s["mode"] == "window":
            ctx["mask_win"] = build_mask(T, s["W"], s["leaf"], s["r"], s["exclude_in_window"], False, dev)
        elif s["mode"] in ("tree", "mean"):
            ctx["mask_tree"] = build_mask(T, s["W"], s["leaf"], s["r"], s["exclude_in_window"], True, dev,
                                          strict_far=s["strict_far"])
            _, _, start, end = node_table(T, s["leaf"], dev)
            ctx["node_pos"] = start if s["pos"] == "start" else end - 1
        return ctx

    # ---- accounting ------------------------------------------------------------------------------------
    def _per_token_matmul_params(self) -> int:
        c, d = self.cfg, self.cfg.n_embd
        return c.n_layer * (4 * d * d + 2 * d * int(c.mlp_ratio * d)) + self.lm_head.weight.numel()

    def flops_per_token(self) -> float:
        """Training forward+backward FLOPs per token AS EXECUTED at cfg.seq_len (dense masked kernel: every key
        is multiplied, including masked ones; dense arm uses the causal kernel at T/2 keys)."""
        c, s, d = self.cfg, self.s, self.cfg.n_embd
        T = c.seq_len
        N = sum(T >> (s["leaf"] + i) for i in range(n_levels(T, s["leaf"])))
        total = 6.0 * self._per_token_matmul_params()
        for a in self.attns:
            if a.full:
                total += attn_core_flops(T, a.H, a.hd, a.hd, T / 2)
            elif not a.summ:
                total += attn_core_flops(T, a.H, a.hd, a.hd, T)
            else:
                total += attn_core_flops(T, a.H, a.hd, a.hd, T + N)
                total += 6.0 * (N / T) * 2 * d * d                              # node k/v projections
                if a.learned:
                    per_merge = sum(p.numel() for p in a.merges[0].parameters())
                    total += 6.0 * ((N - (T >> s["leaf"])) / T) * per_merge     # learned merges (levels > leaf)
        return total

    def _levels_read(self, ctx_len: int) -> int:
        """Levels whose latest r blocks can be read for some position: r 2^n > W (strict: entirely older than the
        window) or (r+1) 2^n > W (partial overlap allowed)."""
        s = self.s
        top = int(math.floor(math.log2(max(ctx_len, 2))))
        k = s["r"] if s["strict_far"] else s["r"] + 1
        return sum(1 for n in range(s["leaf"], top + 1) if k * (1 << n) > s["W"])

    def inference_memory(self, ctx_len: int, bytes_per_el: int = 2) -> dict:
        """Analytic decode-state bytes for one sequence at context ctx_len (all layers): KV entries (k and v,
        d each) plus pyramid state (one pending block per level + the leaf accumulator, d each)."""
        s, d = self.s, self.cfg.n_embd
        top = int(math.floor(math.log2(max(ctx_len, 2))))
        kv_entries, state_vecs = 0, 0
        for a in self.attns:
            if a.full:
                kv_entries += ctx_len
            elif not a.summ:
                kv_entries += min(ctx_len, s["W"])
            else:
                kv_entries += min(ctx_len, s["W"]) + s["r"] * self._levels_read(ctx_len)
                state_vecs += (top - s["leaf"] + 1) + 1
        dense = self.cfg.n_layer * ctx_len * 2 * d * bytes_per_el
        b = (kv_entries * 2 * d + state_vecs * d) * bytes_per_el
        return dict(bytes=b, dense_bytes=dense, ratio_vs_dense=dense / b, kv_entries=kv_entries,
                    state_vectors=state_vecs)

    def inference_flops_per_token(self, ctx_len: int) -> float:
        """Analytic forward FLOPs to decode one token at context ctx_len (matmuls + attention reads + amortised
        merges and node projections)."""
        s, d = self.s, self.cfg.n_embd
        total = 2.0 * self._per_token_matmul_params()
        for a in self.attns:
            if a.full:
                keys = ctx_len
            elif not a.summ:
                keys = min(ctx_len, s["W"])
            else:
                keys = min(ctx_len, s["W"]) + s["r"] * self._levels_read(ctx_len)
                nodes_per_tok = 2.0 / (1 << s["leaf"])
                total += 2.0 * nodes_per_tok * 2 * d * d
                if a.learned:
                    total += 2.0 * (nodes_per_tok / 2) * sum(p.numel() for p in a.merges[0].parameters())
            total += 2.0 * keys * 2 * d
        return total


    # ---- exact incremental decoder (reference implementation, eager, B rows in lockstep) ---------------------
    @torch.no_grad()
    def decode_all(self, idx: torch.Tensor, ref_T: int | None = None):
        """Feed idx [B, T] one token at a time through an O(W + r log T)-per-layer decode state and return
        (logits [B, T, V], peak_state_bytes, final_state_bytes). Level count is capped as a forward pass of
        length ref_T (default T) would build it, so logits must equal forward_logits(idx). Not for the null
        (its noise draws differ between the two paths)."""
        assert not self.s["null"], "decoder is defined for the content arms"
        B, T = idx.shape
        ref_T = ref_T or T
        s, d = self.s, self.cfg.n_embd
        top = s["leaf"] + n_levels(ref_T, s["leaf"]) - 1
        states = [dict(k=None, v=None, acc=None, pending={}, latest={}) for _ in self.attns]
        out, peak = [], 0
        for t in range(T):
            x = rms(self.embed(idx[:, t:t + 1]))
            for a, mlp, st in zip(self.attns, self.mlps, states):
                h = rms(x)
                q, k, v = a.qkv(h).view(B, 1, 3, a.H, a.hd).permute(2, 0, 3, 1, 4)
                pos = torch.tensor([t], device=idx.device)
                q, k = a._rope(rms(q), pos), a._rope(rms(k), pos)
                st["k"] = k if st["k"] is None else torch.cat([st["k"], k], 2)
                st["v"] = v if st["v"] is None else torch.cat([st["v"], v], 2)
                if not a.full:
                    st["k"], st["v"] = st["k"][:, :, -s["W"]:], st["v"][:, :, -s["W"]:]
                Ks, Vs = [st["k"]], [st["v"]]
                if a.summ:
                    st["acc"] = h[:, 0] if st["acc"] is None else st["acc"] + h[:, 0]
                    if (t + 1) % (1 << s["leaf"]) == 0:
                        leaf = rms(st["acc"] / (1 << s["leaf"]))
                        st["acc"] = None
                        self._complete(a, st, s["leaf"], (t + 1 >> s["leaf"]) - 1, leaf, top)
                    for n in sorted(st["latest"]):
                        for (start, ks, vs) in st["latest"][n]:
                            if s["strict_far"]:
                                keep = start + (1 << n) <= t - s["W"] + 1
                            else:
                                keep = (not s["exclude_in_window"]) or start <= t - s["W"]
                            if keep:
                                Ks.append(ks)
                                Vs.append(vs)
                y = F.scaled_dot_product_attention(q, torch.cat(Ks, 2), torch.cat(Vs, 2))
                x = x + a.proj(y.transpose(1, 2).reshape(B, 1, d))
                x = x + mlp(rms(x))
            logits = self.lm_head(rms(x)).float()
            out.append(30.0 * torch.tanh(logits / 30.0))
            peak = max(peak, self._state_bytes(states))
        return torch.cat(out, 1), peak, self._state_bytes(states)

    def _complete(self, a, st, n, k, S, top):
        """A level-n block with index k has completed with summary S [B, d]: store its k/v (latest r per level)
        and cascade the merge upward when it is a right child."""
        if n > top:
            return
        d = self.cfg.n_embd
        B = S.shape[0]
        kv = F.linear(S, a.qkv.weight[d:]).view(B, 1, 2, a.H, a.hd).permute(2, 0, 3, 1, 4)
        end = (k + 1) << n
        p = (k << n) if self.s["pos"] == "start" else end - 1
        ks = a._rope(rms(kv[0]), torch.tensor([p], device=S.device))
        lst = st["latest"].setdefault(n, [])
        lst.append((k << n, ks, kv[1]))
        del lst[:-self.s["r"]]
        if n + 1 > top:
            return
        if k % 2 == 1:
            left = st["pending"].pop(n)
            self._complete(a, st, n + 1, k // 2, a._merge(left, S, n - self.s["leaf"]), top)
        else:
            st["pending"][n] = S

    @staticmethod
    def _state_bytes(states) -> int:
        tot = 0
        for st in states:
            for key in ("k", "v", "acc"):
                if st[key] is not None:
                    tot += st[key].numel() * st[key].element_size()
            for S in st["pending"].values():
                tot += S.numel() * S.element_size()
            for lst in st["latest"].values():
                for (_, ks, vs) in lst:
                    tot += (ks.numel() + vs.numel()) * ks.element_size()
        return tot


def build(cfg: ModelConfig) -> nn.Module:
    return RGFoveaGPT(cfg)
