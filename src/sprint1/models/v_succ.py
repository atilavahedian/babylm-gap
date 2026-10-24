"""V: Successor stream. The scored network is the unchanged sprint baseline (baseline.py, identical body init at the
same seed); what changes is the body's learning signal. Card: sprint1/ideas/V_succ.md.

Every tapped block l gets its OWN local target and a tiny linear predictor P_l (d -> r, training-only):

    s_l(t) = (1 - g_l) * phi(y_t) + g_l * sg( P_l norm(x_l)(t+1) )        (TD(0) bootstrap, "td", default)
    loss_l = sum_t || P_l norm(x_l)(t) - s_l(t) ||^2 / sum_t || s_l(t) ||^2

phi(v) is a FIXED random code of token v in R^r (N(0, 1/r), seeded separately: a JL sketch of the one-hot), so s_l(t)
estimates the discounted bag of the tokens AFTER position t (the successor representation of the token stream) with
the label noise of one token, because the tail is bootstrapped from the model's own next-position state. The
discount is a depth/timescale hierarchy: horizon h_l = 1/(1-g_l) grows geometrically from sr_h_min (block 0) to
sr_h_max (top block), so each block gets a different job instead of the same top error.

Shape, never replace: the predictor learns from its own loss at full strength, but the trunk feels the local gradient
through a gradient-ratio control. At each tap the raw local gradient is rescaled so that its norm is
rho_tap = sr_rho / sqrt(n_taps) times the norm of the CE gradient arriving at the top of the body (EMA over
micro-batches, measured by backward probes), capped at loss weight sr_cap. Without it Muon's normalisation would let any
fixed-weight auxiliary dominate the body early (the CE gradient into the body is O(head scale) = 0.005 at init).

The scored path is exactly baseline.GPT: idx -> logits, targets enter only the CE and aux["aux_loss"]; eval mode runs
the baseline forward and never touches the taps (nothing is written in eval).

cfg.extra knobs (all optional; every other baseline knob is honoured):
  sr_lambda (float, default 1.0): 0 = "off" arm, the exact baseline path (predictors unused)
  sr_rank r (int, 64)          sr_taps ("all" | "top" | list of block indices, "all")
  sr_boot ("td" | "mc", "td"): mc = Monte-Carlo discounted future inside the window (no bootstrap; T x T Toeplitz)
  sr_gamma (float or list, default: hierarchy)   sr_h_min (float, 2)   sr_h_max (float, seq_len / 8)
  sr_rho (float, 0.5)   sr_cap (float, 1.0)   sr_ema (float, 0.9)   sr_code_seed (int, 1234)
  null (bool): the null twin. null_mode "reverse" (default, registered): identical machinery with the arrow of time
      flipped: the predecessor representation s_l(t) = (1-g) phi(x_{t-1}) + g sg(P_l norm(x_l)(t-1)), i.e. the same
      code statistics, gradient regime and FLOPs, but the target carries no information about the future (it is a
      function of the visible context). null_mode "noise": phi of fresh uniform random tokens (uninformative).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.baseline import GPT, CastedLinear, norm


class _Probe(torch.autograd.Function):
    """Identity. Backward adds ||grad||^2 to `stat` and passes the gradient through unchanged."""

    @staticmethod
    def forward(ctx, x, stat):
        ctx.stat = stat
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        ctx.stat.add_(g.detach().float().pow(2).sum())
        return g, None


class _ScaleGrad(torch.autograd.Function):
    """Identity. Backward adds the RAW ||grad||^2 to `stat` and returns scale * grad (the trunk's exposure)."""

    @staticmethod
    def forward(ctx, x, scale, stat):
        ctx.stat = stat
        ctx.scale = scale.detach().clone()  # a copy: the buffer is updated in place before the next forward
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        ctx.stat.add_(g.detach().float().pow(2).sum())
        return g * ctx.scale.to(g.dtype), None, None


def hierarchy_gammas(n: int, h_min: float, h_max: float) -> list[float]:
    """Discounts whose horizons 1/(1-g) are geometric from h_min (first tap) to h_max (last tap)."""
    if n == 1:
        return [1.0 - 1.0 / h_max]
    return [1.0 - 1.0 / (h_min * (h_max / h_min) ** (i / (n - 1))) for i in range(n)]


class SuccessorGPT(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)  # body first: identical init to baseline.GPT at the same torch seed
        ex = cfg.extra
        L, d, V, T = cfg.n_layer, cfg.n_embd, cfg.vocab_size, cfg.seq_len
        self.null = bool(ex.get("null", False))
        self.null_mode = str(ex.get("null_mode", "reverse")) if self.null else "off"
        if self.null_mode not in ("off", "reverse", "noise"):
            raise ValueError(f"null_mode must be reverse|noise, got {self.null_mode!r}")
        self.lam_on = float(ex.get("sr_lambda", 1.0))
        self.r = int(ex.get("sr_rank", 64))
        self.boot = str(ex.get("sr_boot", "td"))
        if self.boot not in ("td", "mc"):
            raise ValueError(f"sr_boot must be td|mc, got {self.boot!r}")
        taps = ex.get("sr_taps", "all")
        if taps == "all":
            self.taps = list(range(L))
        elif taps == "top":
            self.taps = [L - 1]
        else:
            self.taps = sorted({int(i) % L for i in (taps if isinstance(taps, (list, tuple)) else [taps])})
        n = len(self.taps)
        gam = ex.get("sr_gamma", None)
        if gam is None:
            h_max = float(ex.get("sr_h_max", max(2.0, T / 8)))
            h_min = min(float(ex.get("sr_h_min", 2.0)), h_max)
            self.gammas = hierarchy_gammas(n, h_min, h_max)
        elif isinstance(gam, (list, tuple)):
            assert len(gam) == n, "sr_gamma list must have one entry per tap"
            self.gammas = [float(g) for g in gam]
        else:
            self.gammas = [float(gam)] * n
        assert all(0.0 <= g < 1.0 for g in self.gammas)
        self.rho_tap = float(ex.get("sr_rho", 0.5)) / math.sqrt(max(n, 1))
        self.cap = float(ex.get("sr_cap", 1.0))
        self.ema_decay = float(ex.get("sr_ema", 0.9))
        # training-only parameters and buffers: own generator so the body init (and the global RNG stream used by the
        # body) is untouched relative to baseline.GPT
        gen = torch.Generator().manual_seed(int(ex.get("sr_code_seed", 1234)))
        self.register_buffer("codes", torch.randn(V, self.r, generator=gen) / math.sqrt(self.r), persistent=False)
        self.pred = nn.ModuleList(CastedLinear(d, self.r) for _ in range(n))
        bound = math.sqrt(3.0) * 0.5 * d ** -0.5
        with torch.no_grad():
            for p in self.pred:
                p.weight.copy_(torch.empty_like(p.weight).uniform_(-bound, bound, generator=gen))
        self._noise_seed = int(ex.get("sr_code_seed", 1234)) + 1
        self._noise_calls = 0
        # gradient-ratio control state (persistent so a resumed run continues exactly)
        self.register_buffer("stat_ce", torch.zeros(()))
        self.register_buffer("stat_aux", torch.zeros(max(n, 1)))
        self.register_buffer("ema_ce", torch.zeros(()))
        self.register_buffer("ema_aux", torch.zeros(max(n, 1)))
        self.register_buffer("scale", torch.zeros(max(n, 1)))

    # ---- helpers ----------------------------------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.lam_on > 0 and len(self.taps) > 0

    def _body(self, idx: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """baseline.GPT.hidden without the final norm, also returning the stream after every tapped block."""
        L = self.cfg.n_layer
        e = self.embed(idx)
        x = x0 = norm(e) if self.embnorm else e
        ves = [emb(idx) for emb in self.value_embeds]
        ve_at: list[torch.Tensor | None] = [None] * L
        for i, ve in enumerate(ves):
            ve_at[i] = ve
            ve_at[L - self.n_ve + i] = ve
        skips: list[torch.Tensor] = []
        tapset = set(self.taps)
        taps: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            if self.unet and i >= L - self.n_enc and skips:
                x = x + self.skip_weights[i - (L - self.n_enc)] * skips.pop()
            x = block(x, ve_at[i], x0)
            if self.unet and i < self.n_enc:
                skips.append(x)
            if i in tapset:
                taps.append(x)
        return x, taps

    @torch.no_grad()
    def _update_scales(self) -> None:
        """Fold the gradient statistics of the last backward(s) into the EMAs and set the trunk exposure per tap."""
        e = self.ema_decay
        has = self.stat_ce > 0
        self.ema_ce.copy_(torch.where(has, e * self.ema_ce + (1 - e) * self.stat_ce, self.ema_ce))
        has_a = self.stat_aux > 0
        self.ema_aux.copy_(torch.where(has_a, e * self.ema_aux + (1 - e) * self.stat_aux, self.ema_aux))
        ratio = self.rho_tap * torch.sqrt(self.ema_ce / self.ema_aux.clamp_min(1e-30))
        ok = (self.ema_ce > 0) & (self.ema_aux > 0)
        self.scale.copy_(torch.where(ok, ratio.clamp(max=self.cap), torch.zeros_like(ratio)))
        self.stat_ce.zero_()
        self.stat_aux.zero_()

    def _codes_for(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids.clamp_min(0), self.codes)

    def _targets(self, j: int, pred: torch.Tensor, idx: torch.Tensor, targets: torch.Tensor):
        """Local target and mask for tap j. pred: [B, T, r] fp32 (its bootstrap use is detached here)."""
        B, T, r = pred.shape
        g = self.gammas[j]
        p = pred.detach()
        if self.null_mode == "reverse":  # predecessor representation: the visible past, no future information
            code = self._codes_for(idx)  # phi(x_s)
            if self.boot == "td":
                tgt = torch.zeros_like(p)
                tgt[:, 1:] = (1 - g) * code[:, :-1] + g * p[:, :-1]
                mask = torch.ones(B, T, dtype=torch.bool, device=idx.device)
                mask[:, 0] = False
                return tgt, mask
            # mc: c(t) = sum_{k>=1} g^(k-1) phi(x_{t-k}) / sum of the weights present
            ar = torch.arange(T, device=idx.device)
            k = ar[:, None] - ar[None, :]  # t - s
            M = torch.full((T, T), g, device=idx.device).pow((k - 1).clamp_min(0).float()) * (k >= 1).float()
            den = M.sum(1, keepdim=True).clamp_min(1e-12)
            tgt = torch.einsum("ts,bsr->btr", M / den, code)
            mask = torch.ones(B, T, dtype=torch.bool, device=idx.device)
            mask[:, 0] = False
            return tgt, mask
        if self.null_mode == "noise":
            self._noise_calls += 1
            ng = torch.Generator(device=idx.device).manual_seed(self._noise_seed * 1000003 + self._noise_calls)
            nxt = torch.randint(0, self.cfg.vocab_size, targets.shape, generator=ng, device=idx.device)
            valid = targets >= 0
        else:
            nxt = targets
            valid = targets >= 0
        code = self._codes_for(nxt) * valid.unsqueeze(-1)  # phi(y_t) = phi(x_{t+1})
        if self.boot == "td":
            tgt = torch.zeros_like(p)
            tgt[:, :-1] = (1 - g) * code[:, :-1] + g * p[:, 1:]
            mask = valid.clone()
            mask[:, -1] = False
            return tgt, mask
        ar = torch.arange(T, device=idx.device)
        k = ar[None, :] - ar[:, None]  # s - t
        M = torch.where(k >= 0, torch.full((T, T), g, device=idx.device).pow(k.clamp_min(0).float()),
                        torch.zeros((), device=idx.device))
        W = M * valid.float().unsqueeze(1)  # [B, T, T]: weight of target s for position t, only valid targets
        den = W.sum(-1, keepdim=True).clamp_min(1e-12)
        tgt = torch.einsum("bts,bsr->btr", W / den, code)
        return tgt, valid

    def _successor_loss(self, idx, targets, taps):
        total = torch.zeros((), device=idx.device, dtype=torch.float32)
        diag: dict[str, torch.Tensor] = {}
        for j, xl in enumerate(taps):
            a = _ScaleGrad.apply(xl, self.scale[j], self.stat_aux[j:j + 1])
            pred = self.pred[j](norm(a)).float()
            tgt, mask = self._targets(j, pred, idx, targets)
            m = mask.unsqueeze(-1).float()
            err = ((pred - tgt).pow(2) * m).sum()
            pw = (tgt.pow(2) * m).sum().clamp_min(1e-12).detach()
            lj = err / pw
            total = total + lj
            diag[f"sr_nmse_{self.taps[j]}"] = lj.detach()
            diag[f"sr_scale_{self.taps[j]}"] = self.scale[j].detach().clone()
        return total, diag

    # ---- contract ---------------------------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        if not self.training or not self.active:
            return super().forward(idx, targets)
        self._update_scales()
        x, taps = self._body(idx)
        x = _Probe.apply(x, self.stat_ce)  # pure CE gradient at the top of the body (no tap sits above this node)
        z = self.logits(norm(x))
        B, T, V = z.shape
        loss = F.cross_entropy(z.view(B * T, V), targets.reshape(B * T), ignore_index=-1)
        aux_loss, diag = self._successor_loss(idx, targets, taps)
        return loss, {"aux_loss": aux_loss, **diag}

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        g = super().param_groups()
        pred = [p.weight for p in self.pred]
        ids = {id(p) for p in pred}
        return {"muon": g["muon"] + pred, "adam": [p for p in g["adam"] if id(p) not in ids]}

    def flops_per_token(self) -> float:
        base = super().flops_per_token()
        if not self.active:
            return base
        d, T = self.cfg.n_embd, self.cfg.seq_len
        n = len(self.taps)
        extra = n * 6.0 * self.r * d  # predictor forward (2rd) + backward (4rd) per tap
        if self.boot == "mc":
            extra += n * 2.0 * T * self.r  # Toeplitz target (no gradient)
        return base + extra


def build(cfg: ModelConfig) -> nn.Module:
    return SuccessorGPT(cfg)
