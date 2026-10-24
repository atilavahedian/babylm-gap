"""H-MEM: the frozen sprint-1 baseline plus hashed n-gram memory tables
(experiments/2026-10-17-scaling-slope/governance/attacker/SPEC_HMEM.md). The dense network is bit-for-bit the baseline
(same modules, same init order, same optimizer groups); the only addition is a sparse memory read into the residual
stream at the input.

KNOWN. Hash embeddings (Svenstrup, Hansen, Winther 2017); Hash Layers (Roller et al. 2021); N-grammer (Roy et al.
2022); Over-Tokenized Transformer (Huang et al. 2025, arXiv 2501.16975: hashed multi-gram input embeddings, the
closest prior art); SCONE f-gram embeddings (Yu et al. 2025, arXiv 2502.01637); Engram (DeepSeek 2026, arXiv
2601.07372: hashed n-gram lookups fused into the hidden state); the modded-nanogpt records' bigram hash tables
("n-gram tables", listed as dropped in baseline.py).
What is tested here is not the idea but a measurement on the one-pass FineWeb ruler at fixed dense N: does the
table's gain over the random control grow with data beyond what a bigger dense model would gain (the capacity-only
null of attacker/growth_null.py), or is it a shift? The mechanism that could make it grow is in the same folder's
DIAG_SPEC.md (a per-exposure value that falls with rarity or with data in the dense net, which per-row updates do not
share).

Mechanism. For each position t and each order n in hmem_orders, the n-gram (x_{t-n+1}, ..., x_t) is hashed by
hmem_hashes independent 64-bit hashes into a table of hmem_buckets rows of width n_embd; the rows are summed to
m_t, soft-normalised, m_t / sqrt(1 + mean(m_t^2)) (zero stays zero, a row's magnitude grows with the exposures it
has had and saturates at unit RMS, so a noisy row can never swamp the unit-RMS token embedding), scaled by one
learned scalar gate g (init hmem_gate_init, AdamW scalar group) and added to the RMS-normalised token embedding,
so the memory reaches every block through x0 injection and the U-net skips exactly as the token embedding does.
Rows are zero at init (no RNG draws, so the dense init is the control's and the first forward is the control's).
Order mapping: hmem order n keys on n tokens ending at the current one and is read when predicting the next, so
hmem order 2 is the diagnostic's order 3 (two context tokens + target).

Rows are NOT interference-free. (1) Collisions: a 2^20-row table at XS 448M tokens carries of the order of 10-100
distinct n-grams per row (the diagnostic's type counts give the exact load), rising with D; two hashes per order
make a double collision rare but every row is shared. (2) Adam's per-coordinate normalisation is per row, so the
step a given n-gram's exposure produces depends on the traffic of the n-grams it shares the row with. What the
table keeps is sparsity: a row moves only when one of its n-grams occurs.

Update size and the table LR (reviewer A6). After one isolated exposure Adam's momentum keeps moving a row while
its gradient is zero again: the total displacement is ~5 x lr per coordinate (beta1 0.8, beta2 0.95), 1.6 at the
XS lr_embed 0.3225, so the soft norm saturates after a single exposure and the row is a one-shot sign memory of
one gradient. hmem_lr_mult (default 0.1) runs the tables in their own AdamW group at lr_embed x hmem_lr_mult
(make_optimizer below): ~0.16 per coordinate per exposure, saturation after ~6 exposures, a graded memory. It is
a knob, not a tuned value; the kill test's ablation (SPEC_HMEM.md) varies it.
Why the soft norm: the plain sum, tried first, let a once-seen row inject +-lr_embed per dimension (0.25-0.78),
as large as x0 itself; on the CPU smoke the arm was then far worse than the baseline.

Accounting. The tables are nn.Embedding modules, so train.count_params excludes them from the non-embedding count N
(the ruler's size axis); the arm is equal-N and unequal-parameters (341 x N at XS; PARAMS flag in SPEC_HMEM.md:
the honest frame is capacity per dense FLOP, not per parameter). flops_per_token() adds the gather and scatter
cost (6 x n_tables x n_embd per token, below 1e-4 of the dense cost); bytes_touched_per_token() is the sparse
row traffic, but the harness's AdamW is dense: every step updates all table parameters (SPEC_HMEM.md section 3).
The scored loss is the plain next-token CE in nats as for every arm.

cfg.extra knobs (all optional):
  hmem_buckets (int, default 2**20)      rows per table (power of two)
  hmem_orders  (list[int], default [2, 3])  n-gram orders (context n-1 tokens + current token)
  hmem_hashes  (int, default 2)          independent hashes per order (rows summed)
  hmem_gate_init (float, default 0.5)   init of the learned scalar gate g on the memory read (scalar Adam group)
  hmem_norm    (str, default "soft")     "soft": m / sqrt(1 + mean m^2); "none": the raw sum
  hmem_lr_mult (float, default 0.1)     table LR = lr_embed x hmem_lr_mult (own AdamW group "hmem"; 1.0 = embed group)
  hmem_enabled (bool, default True)      False -> identical to baseline (A/A check of the file itself)

CPU smoke (no GPU): python -m sprint1.harness.train --model hmem --size tiny --steps 20 --synthetic --threads 1 \\
    --extra hmem_buckets=4096
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sprint1.harness.model_api import ModelConfig
from sprint1.models.baseline import GPT, norm

# per-hash odd 64-bit multipliers (splitmix / xxhash constants); int64 arithmetic wraps, which is what a hash wants
_MULTS = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5)
_MASK63 = (1 << 63) - 1


def _as_i64(c: int) -> int:
    return c - (1 << 64) if c >= (1 << 63) else c


def ngram_buckets(idx: torch.Tensor, n: int, j: int, buckets: int, pad: int) -> torch.Tensor:
    """Bucket id in [0, buckets) of the n-gram ending at every position of idx [B, T] under hash j (deterministic,
    vectorised). Positions before the window start use the pad id; buckets must be a power of two."""
    B, T = idx.shape
    x = idx.to(torch.int64)
    if n > 1:
        x = torch.cat([x.new_full((B, n - 1), pad), x], dim=1)
    m1 = torch.tensor(_as_i64(_MULTS[j % len(_MULTS)]), dtype=torch.int64, device=idx.device)
    m2 = torch.tensor(_as_i64(0xBF58476D1CE4E5B9), dtype=torch.int64, device=idx.device)
    h = torch.zeros((B, T), dtype=torch.int64, device=idx.device)
    for k in range(n):                      # oldest token first
        tok = x[:, k:k + T] + 1
        h = (h ^ (tok * m1)) * m2
        h = h ^ (h >> 29)                   # arithmetic shift: still a deterministic mix
    h = h ^ (h >> 32)
    return (h & _MASK63) & (buckets - 1)


class HMem(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        ex = cfg.extra
        self.hm_enabled = bool(ex.get("hmem_enabled", True))
        self.hm_buckets = int(ex.get("hmem_buckets", 1 << 20))
        if self.hm_buckets & (self.hm_buckets - 1):
            raise ValueError(f"hmem_buckets must be a power of two, got {self.hm_buckets}")
        orders = ex.get("hmem_orders", [2, 3])
        self.hm_orders = [int(o) for o in (orders if isinstance(orders, (list, tuple)) else [orders])]
        self.hm_hashes = int(ex.get("hmem_hashes", 2))
        self.hm_norm = str(ex.get("hmem_norm", "soft") or "none").lower()   # --extra hmem_norm=none parses to None
        if self.hm_norm not in ("soft", "none"):
            raise ValueError(f"hmem_norm must be 'soft' or 'none', got {self.hm_norm!r}")
        self.hm_lr_mult = float(ex.get("hmem_lr_mult", 0.1))
        self.hm_pad = int(cfg.vocab_size)   # never a real token id
        d = cfg.n_embd
        n_tables = len(self.hm_orders) * self.hm_hashes if self.hm_enabled else 0
        # zero rows, built without RNG draws so the dense init stays the control's; nn.Embedding so that
        # train.count_params excludes them from N and the optimizer treats them as embeddings
        self.hmem_embeds = nn.ModuleList(
            nn.Embedding(self.hm_buckets, d, _weight=torch.zeros(self.hm_buckets, d)) for _ in range(n_tables))
        # one scalar gate on the memory read; ndim 0 -> optim.split_adam "scalar" group (lr_scalar). Only when the
        # memory exists, so hmem_enabled=false has the baseline's exact parameter list (reviewer A8)
        self.hmem_gate = nn.Parameter(torch.tensor(float(ex.get("hmem_gate_init", 0.5)))) if n_tables else None

    # ---- memory read -----------------------------------------------------------------------------
    def hm_read(self, idx: torch.Tensor) -> torch.Tensor | None:
        if not self.hm_enabled or not len(self.hmem_embeds):
            return None
        out = None
        t = 0
        for n in self.hm_orders:
            for j in range(self.hm_hashes):
                b = ngram_buckets(idx, n, j, self.hm_buckets, self.hm_pad)
                r = self.hmem_embeds[t](b)
                out = r if out is None else out + r
                t += 1
        out = out.float()
        if self.hm_norm == "soft":
            out = out * torch.rsqrt(1.0 + out.square().mean(dim=-1, keepdim=True))
        return self.hmem_gate * out

    # ---- the baseline's hidden() with the memory added to x0 -----------------------------------------
    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        L = self.cfg.n_layer
        e = self.embed(idx)
        x0 = norm(e) if self.embnorm else e
        m = self.hm_read(idx)
        if m is not None:
            x0 = x0 + m.to(x0.dtype)
        x = x0
        ves = [emb(idx) for emb in self.value_embeds]
        ve_at: list[torch.Tensor | None] = [None] * L
        for i, ve in enumerate(ves):
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

    # ---- accounting ----------------------------------------------------------------------------------
    def n_tables(self) -> int:
        return len(self.hmem_embeds)

    def table_params(self) -> int:
        return sum(e.weight.numel() for e in self.hmem_embeds)

    def flops_per_token(self) -> float:
        # gather (read d floats) + scatter-add in backward per table, counted as 6 x d like a matmul row
        return super().flops_per_token() + 6.0 * self.n_tables() * self.cfg.n_embd

    def bytes_touched_per_token(self) -> float:
        """Sparse table bytes a training token reads and writes: fp32 row read + grad write + AdamW m/v read/write.
        The harness's dense AdamW touches every table parameter every step on top of this (SPEC_HMEM.md)."""
        return float(self.n_tables() * self.cfg.n_embd * 4 * (1 + 1 + 4))


def make_optimizer(model: nn.Module, cfg: ModelConfig, lrs: dict[str, float]):
    """The harness optimizer (Muon + AdamW, train.py's knobs forwarded in cfg.extra) with the memory tables moved to
    their own AdamW group at lr_embed x hmem_lr_mult. With no tables, or hmem_lr_mult = 1, it is build_optimizer."""
    from sprint1.harness.optim import build_optimizer
    ex = cfg.extra
    betas = ex.get("adam_betas", (0.8, 0.95))
    opt = build_optimizer(model, lrs, muon_momentum=float(ex.get("muon_momentum", 0.95)),
                          muon_wd=float(ex.get("muon_wd", 0.0)), adam_betas=(float(betas[0]), float(betas[1])),
                          adam_eps=float(ex.get("adam_eps", 1e-10)), adam_wd=float(ex.get("adam_wd", 0.0)))
    tables = [e.weight for e in getattr(model, "hmem_embeds", [])]
    mult = float(getattr(model, "hm_lr_mult", 1.0))
    if not tables or mult == 1.0:
        return opt
    ids = {id(p) for p in tables}
    lr = float(lrs["adam_embed"]) * mult
    for o in opt.optimizers:
        if not isinstance(o, torch.optim.AdamW):
            continue
        for g in o.param_groups:
            g["params"] = [p for p in g["params"] if id(p) not in ids]
        o.add_param_group(dict(params=tables, lr=lr, initial_lr=lr, kind="adam", name="hmem"))
    return opt


def build(cfg: ModelConfig) -> nn.Module:
    return HMem(cfg)
