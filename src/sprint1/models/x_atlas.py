"""X_atlas: the Atlas head. A context-selected output geometry at the Transformer's own FLOPs.

Card: sprint1/ideas/X_atlas.md. Body: sprint1/models/baseline.py, reused unchanged by subclassing.

Constraint removed. A Transformer turns everything it knows about a context into a prediction through ONE
global log-linear map, logits = W h with W in R^{V x d}. Across contexts the log-probability matrix therefore has
rank <= d + 1 (the softmax bottleneck, Yang et al. 2018), and the readout's expressivity, its FLOPs (V*d per
token) and its parameters (V*d) are all tied to the same d. Raising the rank costs FLOPs in the head and d^2 in
the body.

Principle. Selected parameters are free; multiplied parameters are not. The readout becomes a field of
log-linear maps indexed by the context (an atlas of C charts): each position multiplies by exactly ONE V x d
table, chosen by its own final state, so representable rank grows with C (memory) at constant per-token FLOPs.

Mechanism (per position t, h_t = the baseline's final normed hidden state):
    chart  c_t = argmax_c ( mu_c . h_t/|h_t| - b_c )        mu: C unit centroids (online spherical k-means,
                                                              EMA in on_step_end), b: load-balancing bias
    W_c        = W_0 + Delta_c                                W_0: the baseline lm_head; Delta: zero-init
    p(. | x_<=t) = softmax( softcap( W_{c_t} h_t ) )          exact, normalised over all V for every chart
W_c is materialised once per forward (C*V*d adds, amortised over the batch), then each position does one V x d
product, exactly the baseline head's cost. Delta = 0 at init, so step 0 is the baseline (no head start by
construction). Routing reads only h_t (causal, row-local). Centroid/bias statistics are accumulated in train
forwards and applied only in on_step_end, so a forward never changes its own or another row's routing, and
nothing is written in eval.

cfg.extra knobs (documented in the card):
  atlas_charts  int   C, number of charts (default 4; 1 = plain baseline head, no Delta table)
  atlas_route   str   "kmeans" (default) | "position" (the null) | "token" (ablation: hash of the current token id)
  null          bool  True -> atlas_route = "position": the chart is a fixed balanced hash of the position in the
                      window. Same parameters, FLOPs, code path and balanced load; the chart no longer depends
                      on the content, so the charts cannot specialise (the principle is destroyed).
  atlas_rows    int   K > 0: charts differ only on K output rows (the evidence-rich ones); the other V-K rows stay the
                      shared W_0 in every chart. Rows = the first K GPT-2 ids (BPE merge order ~ frequency: the first
                      8,192 cover 86.6% of Gutenberg target mass) unless atlas_rows_file names an .npy of row ids
                      (e.g. frequency-ranked on the training shards). 0 (default) = all V rows. FLOPs unchanged.
  atlas_rank    int   r < n_embd inserts a d->r projection before the head (stress test that makes the bottleneck
                      bind at CPU scale; routing then reads the same r-dim vector the head reads). 0 = d.
  atlas_ce_chunk int  > 0: compute each chart's logits + CE in row chunks under activation checkpointing (memory only;
                      identical numbers, one extra head forward in backward). Used by the CPU pilot; 0 (default) = off.
  atlas_ema     float centroid EMA rate per optimizer step (default 0.05)
  atlas_balance float gain of the load-balancing bias (default 0.01): b_c += gain * eta_t * (C * load_c - 1), with
                      eta_t = max(0.1, (1 + t/50)^-1/2) (decaying dual ascent; a fixed sign step oscillated
                      between 0.36 and 0.68 max load when states were tightly clustered)
  atlas_lr_mode str   how the head Adam LR is shared between W_0 and Delta (a chart table W_0 + Delta_c has two Adam
                      copies; with both at the head LR it is a 2x effective-LR arm: a 4-step repeated-batch probe went
                      7.46 -> 4.55 nats on that alone):
                      "split": W_0 at lr*(1-lambda), Delta at lr*lambda, lambda = atlas_delta_lr_frac (0.5)
                      "share": W_0 at lr (exactly the baseline head), each Delta_c at lr/C (its share of the data)
                      "center" (default): as "share", and each Delta_c's gradient is centred, g_c - (n_c/N) * sum_c' g_c',
                      so Delta learns only chart-specific deviations; under the null it gets zero-mean noise only. Chosen
                      by the card's pre-registered rule (the mode whose null lands closest to the baseline on the
                      pilot: split +0.101, share ~+0.07, center +0.008).
  + every baseline knob (softcap, n_value_embeds, ...).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.baseline import GPT, CastedLinear, norm


def _balanced_assignment(n: int, C: int, seed: int) -> torch.Tensor:
    """A fixed, balanced, content-free map {0..n-1} -> charts (independent of the global RNG)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(n) % C
    return base[torch.randperm(n, generator=g)]


class AtlasGPT(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)  # baseline body + head drawn first: same init as baseline at the same seed
        ex = cfg.extra
        d, V = cfg.n_embd, cfg.vocab_size
        self.C = max(1, int(ex.get("atlas_charts", 4)))
        self.null = bool(ex.get("null", False))
        route = str(ex.get("atlas_route", "kmeans"))
        self.route = "position" if self.null else route
        assert self.route in ("kmeans", "position", "token"), f"unknown atlas_route {self.route!r}"
        r = int(ex.get("atlas_rank", 0) or 0)
        self.r = d if r <= 0 or r >= d else r
        self.ema = float(ex.get("atlas_ema", 0.05))
        self.gamma = float(ex.get("atlas_balance", 0.01))
        self.ce_chunk = int(ex.get("atlas_ce_chunk", 0) or 0)
        if self.r != d:  # stress-test variant: rank-r readout (all compared arms share it)
            self.head_proj = CastedLinear(d, self.r)
            self.lm_head = CastedLinear(self.r, V)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=float(ex.get("head_init_std", 0.005)))
        K = int(ex.get("atlas_rows", 0) or 0)
        self.K = V if K <= 0 or K >= V else K
        if self.C > 1:
            if self.K < V:
                f = ex.get("atlas_rows_file")
                if f:
                    import numpy as np
                    rows = torch.as_tensor(np.load(f)[: self.K], dtype=torch.long)
                else:
                    rows = torch.arange(self.K)
                assert rows.unique().numel() == self.K and int(rows.max()) < V, "atlas_rows_file: K distinct ids < V"
                self.register_buffer("atlas_rows_idx", rows)
            # named *lm_head* so the harness counts it as an embedding-like table and gives it the head Adam LR
            self.lm_head_delta = nn.Parameter(torch.zeros(self.C, self.K, self.r))
            mu = F.normalize(torch.randn(self.C, self.r), dim=-1)
            self.register_buffer("atlas_mu", mu)                                  # persistent model state
            self.register_buffer("atlas_bias", torch.zeros(self.C))
            self.register_buffer("atlas_steps", torch.zeros((), dtype=torch.long))
            self.register_buffer("atlas_pos_chart", _balanced_assignment(max(cfg.seq_len, 1), self.C, 1234))
            self.register_buffer("atlas_tok_chart", _balanced_assignment(V, self.C, 4321))
            self.register_buffer("atlas_acc_sum", torch.zeros(self.C, self.r), persistent=False)
            self.register_buffer("atlas_acc_cnt", torch.zeros(self.C), persistent=False)
            self.register_buffer("atlas_mu_prev", mu.clone(), persistent=False)   # diagnostics only (churn)
        self._idx: torch.Tensor | None = None
        self._last_load: torch.Tensor | None = None
        self._churn: torch.Tensor | None = None
        self._center = self.C > 1 and str(ex.get("atlas_lr_mode", "center")) == "center"
        self._hooked: int | None = None
        self._ensure_center_hook()

    def _ensure_center_hook(self) -> None:
        """(Re-)attach the centring hook if the Delta parameter changed identity (e.g. after a deepcopy)."""
        if self._center and self._hooked != id(self.lm_head_delta):
            self.lm_head_delta.register_hook(self._center_grad)
            self._hooked = id(self.lm_head_delta)

    def _center_grad(self, g: torch.Tensor) -> torch.Tensor:
        """Remove from each chart's gradient its share of the global head gradient (the part W_0 already learns)."""
        ld = self._last_load
        if ld is None:
            return g
        tot = g.sum(0, keepdim=True)
        return g - ld.to(g.dtype).view(-1, 1, 1) * tot

    # ---- split API ------------------------------------------------------------------------------
    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        if self.route == "token":
            self._idx = idx  # only the "token" ablation route reads it (current token ids, causal)
        return super().hidden(idx)

    def head_input(self, h: torch.Tensor) -> torch.Tensor:
        return norm(self.head_proj(h)) if self.r != self.cfg.n_embd else h

    @torch.no_grad()
    def charts(self, z: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Chart id per position, [B*T]. Depends only on (z_t) / position / current token: causal, row-local."""
        B, T, r = z.shape
        if self.C == 1:
            return torch.zeros(B * T, dtype=torch.long, device=z.device)
        zf = F.normalize(z.detach().float().reshape(B * T, r), dim=-1)
        km = (zf @ self.atlas_mu.t() - self.atlas_bias).argmax(-1)
        if self.training:  # statistics only; applied in on_step_end, never inside a forward
            prev = (zf @ self.atlas_mu_prev.t() - self.atlas_bias).argmax(-1)
            self._churn = (prev != km).float().mean()  # share of positions the last centroid update re-routed
            self.atlas_acc_sum.index_add_(0, km, zf)
            self.atlas_acc_cnt += F.one_hot(km, self.C).sum(0).to(self.atlas_acc_cnt.dtype)  # static shape
        if self.route == "kmeans":
            c = km
        elif self.route == "position":
            pos = torch.arange(T, device=z.device) % self.atlas_pos_chart.numel()
            c = self.atlas_pos_chart[pos].repeat(B)
        else:
            ids = idx if idx is not None else self._idx
            assert ids is not None and ids.shape == (B, T), "token route needs the current token ids"
            c = self.atlas_tok_chart[ids.reshape(-1)]
        self._last_load = F.one_hot(c, self.C).sum(0).float() / max(B * T, 1)
        return c

    def chart_weight(self, c: int) -> torch.Tensor:
        W0 = self.lm_head.weight
        if self.C == 1:
            return W0
        if self.K == W0.size(0):
            return W0 + self.lm_head_delta[c]
        # rows K..V-1 stay the shared W_0; the result is put back in vocabulary order
        delta_full = torch.zeros_like(W0).index_add(0, self.atlas_rows_idx, self.lm_head_delta[c].to(W0.dtype))
        return W0 + delta_full

    def _cap(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        return self.softcap * torch.tanh(z / self.softcap) if self.softcap > 0 else z

    @torch.compiler.disable
    def _routed(self, zf: torch.Tensor, chart: torch.Tensor, targets: torch.Tensor | None):
        """Per-chart products (one V x r product per position). Returns per-position CE [N] (targets given) or
        full logits [N, V]. Data-dependent group sizes, so this region is kept out of torch.compile graphs."""
        N = zf.size(0)
        V = self.lm_head.weight.size(0)
        if self.C == 1:
            if targets is not None and self.ce_chunk > 0 and self.training:
                return self._chunked_ce(zf, self.chart_weight(0), targets)
            z = self._cap(F.linear(zf, self.chart_weight(0)))
            if targets is None:
                return z
            return F.cross_entropy(z, targets, ignore_index=-1, reduction="none")
        out = None
        for c in range(self.C):
            sel = (chart == c).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            if targets is not None and self.ce_chunk > 0 and self.training:
                ce = self._chunked_ce(zf.index_select(0, sel), self.chart_weight(c), targets.index_select(0, sel))
                out = ce.new_zeros(N) if out is None else out
                out = out.index_copy(0, sel, ce)
                continue
            zc = self._cap(F.linear(zf.index_select(0, sel), self.chart_weight(c)))
            if targets is None:
                if out is None:
                    out = zc.new_zeros(N, V)
                out = out.index_copy(0, sel, zc)
            else:
                ce = F.cross_entropy(zc, targets.index_select(0, sel), ignore_index=-1, reduction="none")
                if out is None:
                    out = ce.new_zeros(N)
                out = out.index_copy(0, sel, ce)
        if out is None:
            out = zf.new_zeros((N, V) if targets is None else (N,), dtype=torch.float32)
        return out

    def _chunked_ce(self, z: torch.Tensor, W: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        def f(zz, WW, tt):
            return F.cross_entropy(self._cap(F.linear(zz, WW)), tt, ignore_index=-1, reduction="none")
        return torch.cat([checkpoint(f, z[i:i + self.ce_chunk], W, t[i:i + self.ce_chunk], use_reentrant=False)
                          for i in range(0, z.size(0), self.ce_chunk)])

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        z = self.head_input(h)
        B, T, _ = z.shape
        return self._routed(z.reshape(B * T, -1), self.charts(z), None).view(B, T, -1)

    def out_weight(self) -> torch.Tensor:
        return self.lm_head.weight

    # ---- contract -------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        B, T = idx.shape
        if self.training:
            self._ensure_center_hook()
        z = self.head_input(self.hidden(idx))
        chart = self.charts(z, idx)
        ce = self._routed(z.reshape(B * T, -1), chart, targets.reshape(-1)).view(B, T)
        n = (targets >= 0).sum().clamp_min(1)
        loss = ce.sum() / n
        aux: dict = {}
        if self.C > 1 and self._last_load is not None:
            ld = self._last_load
            ent = -(ld.clamp_min(1e-12).log() * ld).sum() / math.log(self.C)
            aux["atlas_load_max"] = ld.max().detach()
            aux["atlas_load_entropy"] = ent.detach()
            if self.training and self._churn is not None:
                aux["atlas_churn"] = self._churn
            if self.training:
                with torch.no_grad():
                    aux["atlas_delta_rel"] = (self.lm_head_delta.norm() /
                                              (self.lm_head.weight.norm() * math.sqrt(self.C) + 1e-12))
        if not self.training:
            aux["per_token_loss"] = ce.detach()
        return loss, aux

    @torch.no_grad()
    def on_step_end(self, step: int, total_steps: int) -> None:
        if self.C == 1:
            return
        cnt = self.atlas_acc_cnt
        tot = float(cnt.sum())
        if tot > 0:
            has = cnt > 0
            mean = F.normalize(self.atlas_acc_sum / cnt.clamp_min(1)[:, None], dim=-1)
            self.atlas_mu_prev.copy_(self.atlas_mu)
            new = F.normalize((1 - self.ema) * self.atlas_mu + self.ema * mean, dim=-1)
            self.atlas_mu.copy_(torch.where(has[:, None], new, self.atlas_mu))
            load = cnt / tot
            eta = max(0.1, (1.0 + float(self.atlas_steps) / 50.0) ** -0.5)
            self.atlas_bias.add_(self.gamma * eta * (self.C * load - 1.0))
            self.atlas_steps += 1
        self.atlas_acc_sum.zero_()
        self.atlas_acc_cnt.zero_()

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for p in self.blocks.parameters() if p.ndim == 2]
        if self.r != self.cfg.n_embd:
            muon.append(self.head_proj.weight)
        ids = {id(p) for p in muon}
        adam = [p for p in self.parameters() if id(p) not in ids]
        return {"muon": muon, "adam": adam}

    def flops_per_token(self) -> float:
        c = self.cfg
        n_mat = sum(p.numel() for p in self.blocks.parameters() if p.ndim == 2) + self.lm_head.weight.numel()
        if self.r != c.n_embd:
            n_mat += self.head_proj.weight.numel()
        attn = 6 * c.n_layer * c.seq_len * c.n_embd
        router = 2.0 * self.C * self.r if self.C > 1 else 0.0  # forward-only scores, no backward
        return 6.0 * n_mat + attn + router


def make_optimizer(model: nn.Module, cfg: ModelConfig, base_lrs: dict[str, float]):
    """Harness Muon + AdamW (same groups and LRs as build_optimizer), except that the head Adam LR is shared between
    W_0 and Delta according to atlas_lr_mode (see the module docstring)."""
    from sprint1.harness.optim import CombinedOptimizer, Muon, split_adam
    ex = cfg.extra
    groups = model.param_groups()
    muon = [p for p in groups["muon"] if p.requires_grad]
    ids = {id(p) for p in muon}
    adam = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    buckets = split_adam(model, adam)
    delta = getattr(model, "lm_head_delta", None)
    head = [p for p in buckets["head"] if p is not delta]
    lr_h = base_lrs["adam_head"]
    mode = str(ex.get("atlas_lr_mode", "center"))
    if delta is None:
        lr_w0, lr_d = lr_h, 0.0
    elif mode == "split":
        lam = float(ex.get("atlas_delta_lr_frac", 0.5))
        lr_w0, lr_d = lr_h * (1.0 - lam), lr_h * lam
    elif mode in ("share", "center"):
        lr_w0, lr_d = lr_h, lr_h / model.C
    else:
        raise ValueError(f"unknown atlas_lr_mode {mode!r}")
    spec = [("embed", buckets["embed"], base_lrs["adam_embed"]), ("head", head, lr_w0),
            ("head_delta", [delta] if delta is not None else [], lr_d),
            ("scalar", buckets["scalar"], base_lrs["adam_scalar"])]
    adam_groups = [dict(params=ps, lr=lr, initial_lr=lr, kind="adam", name=n) for n, ps, lr in spec if ps]
    opts = []
    if muon:
        opts.append(Muon(muon, lr=base_lrs["muon"], momentum=float(ex.get("muon_momentum", 0.95)),
                         weight_decay=float(ex.get("muon_wd", 0.0))))
    fused = all(p.is_cuda for g in adam_groups for p in g["params"])
    betas = tuple(ex.get("adam_betas", (0.8, 0.95)))
    opts.append(torch.optim.AdamW(adam_groups, betas=betas, eps=float(ex.get("adam_eps", 1e-10)),
                                  weight_decay=float(ex.get("adam_wd", 0.0)), fused=fused if fused else None))
    return CombinedOptimizer(opts)


def build(cfg: ModelConfig) -> nn.Module:
    return AtlasGPT(cfg)
