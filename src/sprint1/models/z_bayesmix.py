"""Z_bayesmix: the posterior-predictive head. A Bayes mixture over exponential tilts of the base softmax,
weighted by an exact in-context posterior. Card: sprint1/ideas/Z_bayesmix.md.

The Bayes predictive for an unknown source is a mixture in probability space (an m-mixture),
    p(y | x_<=t) = sum_k w_t(k) p_k(y | x_<=t),   w_t(k) ∝ prior_t(k) * prod_{s<t} p_k(x_{s+1} | x_<=s)^{discount},
whereas a softmax head can only emit members of the log-linear (e-flat) family softmax(W h). This file puts
the mixture on top of the unchanged baseline body (sprint1/models/baseline.py):

  base      z = softcap(W h) (the baseline head), p0 = softmax(z), lse0 = log sum_v e^{z_v}
  tilts     component k = exponential tilt of p0 on the frequent sub-vocabulary S = {ids < bm_s}:
              z_k(v) = z(v) + b_k(v) + U_v . c_k(h_t)   (v in S),   z_k(v) = z(v)   (v not in S)
            b: [K, S] static register tilt; U: [S, r] token features; c = A h: [K, r] context coefficients.
  normaliser (exact, cheap): log Z_k = logaddexp( LSE_{v in S} z_k(v),  LSE_{v not in S} z(v) ).
            rho_k = log Z_k - lse0 = log E_{p0}[exp(tilt_k)] is the cumulant generating function of the tilt
            under the base, i.e. each component's exact log-partition relative to the base.
  evidence  the realised next token r = idx[s+1] of every earlier position gives the exact log-likelihood
            ratio l_k(s) = log p_k(r | s) - log p0(r | s) = tilt_k(s, r) - rho_k(s)  (z(r) cancels).
            E_t(k) = sum_{s <= t-1} gamma_k^{t-1-s} l_k(s), restricted to the current document (an <|endoftext|>
            starts a new source). Only idx[<= t] is read: causal by construction; targets are never read.
  posterior log w_t = log_softmax( prior_bias + G h_t + eta * E_t )   (eta: generalised-Bayes learning rate)
  predictive log p(y) = z(y) - lse0 + LSE_k( log w_t(k) + tilt_k(t, y) - rho_k(t) )   (normalised exactly).

Null twin (cfg.extra["null"] = True; v2, 6 Oct, changed after the CPU pilot and before any GPU run): same
parameters, same init, same tensors, FLOPs within 0.1%. It destroys both claimed structures: (i) geometric
(log-linear, e-family) pooling of the same tilts, log p(v) = z(v) + sum_k w_k tilt_k(v) - log Z_geo, with
(ii) no in-context evidence (w = softmax(prior_bias + G h_t)). What is left is an ordinary log-linear head with a
bilinear context tilt on S, which should tie the baseline.
v1 null (bm_pool=geo bm_evidence=scramble, evidence evaluated at perm(realised token)) is kept as a label arm:
in the pilot its scrambled evidence drifted (rms ~245 nats), the weights went one-hot, and it trailed the
baseline, so it was handicapped rather than a fair twin.

cfg.extra knobs (all optional):
  null (bool, False)        bm_k (int, 4)  components      bm_r (int, 16) context-tilt rank
  bm_s (int, 4096)          tilted sub-vocabulary = ids < bm_s (GPT-2 ids < 4096 cover ~80% of targets)
  bm_pool ("arith"|"geo")   default "arith" ("geo" when null)
  bm_evidence ("real"|"scramble"|"off")  default "real" ("off" when null)
  bm_prior ("ctx"|"const")  context-dependent prior logits G h_t (default) or a learned constant only
  bm_doc_reset (bool, True) evidence restarts after <|endoftext|>;  bm_eot (int) its id (50256 if V > 50256)
  bm_tilt_init (float, 0.2) initial std of the context tilts (nats)
  bm_hl_min, bm_hl_max (float, 32, 1024) initial evidence half-lives, log-spaced over the K components
  bm_eta_init (float, 1.0)  bm_perm_seed (int, 1234)
Baseline knobs (embnorm, softcap, ...) pass through to the body unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline as base_mod

EOT_GPT2 = 50256


class _BaseStats(torch.autograd.Function):
    """Everything the head needs from the base logits z [B, T, V] (fp32) in ONE exp pass, with ONE gradient
    write in backward (CPU-cheap: no full-vocab zero-fills from slice / gather backwards):
        lse0 = log sum_v e^z_v,  lseC = log sum_{v >= S} e^z_v,  zS = z[..., :S] (copy),  zy = z[b, t, y[b, t]]."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, y: torch.Tensor, S: int):
        m = z.amax(dim=-1, keepdim=True)
        e = torch.exp(z - m)
        sC = e[..., S:].sum(-1)
        sA = e[..., :S].sum(-1) + sC
        ctx.save_for_backward(e, sA, sC, y)
        ctx.S = S
        m = m.squeeze(-1)
        return m + torch.log(sA), m + torch.log(sC), z[..., :S].contiguous(), z.gather(-1, y.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def backward(ctx, gA: torch.Tensor, gC: torch.Tensor, gS: torch.Tensor, gy: torch.Tensor):
        e, sA, sC, y = ctx.saved_tensors
        S = ctx.S
        grad = e * (gA / sA).unsqueeze(-1)
        grad[..., S:] += e[..., S:] * (gC / sC).unsqueeze(-1)
        grad[..., :S] += gS
        grad.scatter_add_(-1, y.unsqueeze(-1), gy.unsqueeze(-1))
        return grad, None, None


def base_stats(z: torch.Tensor, y: torch.Tensor, S: int):
    """(lse0, lseC or None, zS, zy); y must be in [0, V)."""
    if S >= z.size(-1):
        return torch.logsumexp(z, dim=-1), None, z, z.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return _BaseStats.apply(z, y, S)


class BayesMixHead(nn.Module):
    """Exact posterior mixture over K exponential tilts of a base softmax (see the module docstring)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ex = cfg.extra
        d, V = cfg.n_embd, cfg.vocab_size
        self.V = V
        self.K = K = int(ex.get("bm_k", 4))
        self.r = r = int(ex.get("bm_r", 16))
        self.S = S = max(1, min(int(ex.get("bm_s", 4096)), V))
        self.null = null = bool(ex.get("null", False))
        self.pool = str(ex.get("bm_pool", "geo" if null else "arith"))
        self.evidence = str(ex.get("bm_evidence", "off" if null else "real"))
        self.prior = str(ex.get("bm_prior", "ctx"))
        if self.pool not in ("arith", "geo") or self.evidence not in ("real", "scramble", "off") \
                or self.prior not in ("ctx", "const"):
            raise ValueError(f"bad bm knobs: pool={self.pool} evidence={self.evidence} prior={self.prior}")
        eot = ex.get("bm_eot", EOT_GPT2 if V > EOT_GPT2 else -1)
        self.eot = -1 if eot is None else int(eot)
        self.doc_reset = bool(ex.get("bm_doc_reset", True)) and 0 <= self.eot < V
        tilt0 = float(ex.get("bm_tilt_init", 0.2))
        bound = math.sqrt(3.0) * 0.5 * d ** -0.5  # CastedLinear init: c = A h has std ~0.5 per coordinate
        self.A = nn.Parameter(torch.empty(K * r, d).uniform_(-bound, bound))
        self.U = nn.Parameter(torch.randn(S, r) * (tilt0 / (0.5 * math.sqrt(r))))
        self.b = nn.Parameter(torch.zeros(K, S))
        self.G = nn.Parameter(torch.zeros(K, d))
        self.prior_bias = nn.Parameter(torch.zeros(K))
        hl = torch.logspace(math.log10(float(ex.get("bm_hl_min", 32.0))), math.log10(float(ex.get("bm_hl_max", 1024.0))),
                            K) if K > 1 else torch.tensor([float(ex.get("bm_hl_max", 1024.0))])
        gamma = torch.pow(2.0, -1.0 / hl)
        self.gamma_logit = nn.Parameter(torch.log(gamma) - torch.log1p(-gamma))
        self.eta = nn.Parameter(torch.tensor(float(ex.get("bm_eta_init", 1.0))))
        g = torch.Generator().manual_seed(int(ex.get("bm_perm_seed", 1234)))
        self.register_buffer("perm", torch.randperm(V, generator=g), persistent=False)

    # ---- helpers ----------------------------------------------------------------------------------
    def _tilt_at(self, c: torch.Tensor, y: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
        """tilt_k(t, y[b, t]) for 0 <= y < S, else 0.  c: [B, T, K, r], y: [B, T], tilt: [B, T, K, S] -> [B, T, K].
        CUDA: gathered from the very tensor the normalisers use (bit-consistent under TF32). CPU: recomputed as
        c_k . U[y] + b_k(y) (fp32, no TF32), which avoids the full-size zero-fill of a gather backward."""
        ok = ((y >= 0) & (y < self.S)).to(c.dtype)
        ys = y.clamp(0, self.S - 1)
        if tilt.is_cuda:
            ty = tilt.gather(-1, ys[:, :, None, None].expand(-1, -1, tilt.size(2), 1)).squeeze(-1)
        else:
            ty = torch.matmul(c, self.U.float()[ys].unsqueeze(-1)).squeeze(-1)      # [B, T, K]
            ty = ty + self.b.float().t()[ys]                                        # b[:, y] -> [B, T, K]
        return ty * ok.unsqueeze(-1)

    def _discount(self, ellp: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """E[b, t, k] = sum_{j <= t} gamma_k^{t-j} [idx[b, j] in the current document at t] * ellp[b, j, k]."""
        T = ellp.size(1)
        ar = torch.arange(T, device=ellp.device)
        diff = (ar[:, None] - ar[None, :]).to(ellp.dtype)                          # t - j
        logg = F.logsigmoid(self.gamma_logit.float())                              # [K]
        logD = (diff[None] * logg[:, None, None]).masked_fill((diff < 0)[None], float("-inf"))
        D = torch.exp(logD)                                                        # [K, T, T], lower-triangular
        if not self.doc_reset:
            return torch.einsum("ktj,bjk->btk", D, ellp)
        is_eot = (idx == self.eot).long()
        cur = torch.cumsum(is_eot, 1)                     # document of the token predicted at t
        docid = cur - is_eot                              # document the realised token idx[j] belongs to
        M = (cur[:, :, None] == docid[:, None, :]).to(ellp.dtype)                  # [B, T, T]
        W = M[:, None] * D[None]                                                    # [B, K, T, T]
        return torch.matmul(W, ellp.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

    # ---- main -------------------------------------------------------------------------------------
    def stats(self, idx: torch.Tensor, h: torch.Tensor, z: torch.Tensor, targets: torch.Tensor | None) -> dict:
        """Tilts, exact component normalisers, in-context evidence and the posterior (fp32, autocast off).
        Call inside torch.autocast(enabled=False) (forward does)."""
        B_, T, V = z.shape
        K, S, r = self.K, self.S, self.r
        hf, z = h.float(), z.float()
        c = F.linear(hf, self.A.float()).view(B_, T, K, r)
        tilt = torch.matmul(c, self.U.float().t()) + self.b.float()                # [B, T, K, S]
        ysafe = targets.clamp_min(0) if targets is not None else torch.zeros_like(idx)
        lse0, lseC, zS, zy = base_stats(z, ysafe, S)                                # [B,T], [B,T], [B,T,S], [B,T]
        zk = zS.unsqueeze(2) + tilt                                                 # component logits on S
        lseSk = torch.logsumexp(zk, dim=-1)                                         # [B, T, K]
        logZk = torch.logaddexp(lseSk, lseC.unsqueeze(-1)) if lseC is not None else lseSk
        rho = logZk - lse0.unsqueeze(-1)                                            # log E_p0[e^tilt_k]
        # ---- in-context evidence from realised tokens (reads idx[1..t] only) ----
        if self.evidence == "off" or T < 2:
            E = torch.zeros_like(rho)
        else:
            nxt = idx[:, 1:]
            if self.evidence == "scramble":
                nxt = self.perm[nxt]
            ell = self._tilt_at(c[:, :-1], nxt, tilt[:, :-1]) - rho[:, :-1]                       # [B, T-1, K]
            ellp = F.pad(ell, (0, 0, 1, 0))                                          # ellp[:, j]: realised idx[:, j]
            E = self._discount(ellp, idx)
        g = self.prior_bias.float()
        if self.prior == "ctx":
            g = g + F.linear(hf, self.G.float())
        logw = torch.log_softmax(g + self.eta.float() * E, dim=-1)                  # [B, T, K]
        return dict(z=z, c=c, tilt=tilt, lse0=lse0, lseC=lseC, zS=zS, zy=zy, zk=zk, logZk=logZk, rho=rho, E=E,
                    logw=logw)

    def forward(self, idx: torch.Tensor, h: torch.Tensor, z: torch.Tensor, targets: torch.Tensor | None,
                full: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (log p(targets) [B, T], diagnostics) or, with full=True, (log p(v) for all v [B, T, V], diag)."""
        S = self.S
        with torch.autocast(device_type=z.device.type, enabled=False):
            st = self.stats(idx, h, z, targets)
            z, c, tilt, lse0, lseC, zS, zy = st["z"], st["c"], st["tilt"], st["lse0"], st["lseC"], st["zS"], st["zy"]
            zk, logZk, rho, E, logw = st["zk"], st["logZk"], st["rho"], st["E"], st["logw"]
            if self.pool == "arith":
                if full:
                    pS = torch.logsumexp(logw.unsqueeze(-1) + zk - logZk.unsqueeze(-1), dim=2)      # [B, T, S]
                    if lseC is not None:
                        pC = z[..., S:] + torch.logsumexp(logw - logZk, dim=-1, keepdim=True)
                        out = torch.cat([pS, pC], dim=-1)
                    else:
                        out = pS
                else:
                    out = zy - lse0 + torch.logsumexp(logw + self._tilt_at(c, targets, tilt) - rho, dim=-1)
            else:  # geometric pooling (null): one log-linear softmax with the w-averaged tilt
                w = logw.exp()
                wt = (w.unsqueeze(-1) * tilt).sum(2)                                # [B, T, S] (elementwise fp32)
                lseSg = torch.logsumexp(zS + wt, dim=-1)
                logZg = torch.logaddexp(lseSg, lseC) if lseC is not None else lseSg
                if full:
                    out = torch.cat([zS + wt, z[..., S:]], dim=-1) - logZg.unsqueeze(-1)
                else:
                    out = zy + (w * self._tilt_at(c, targets, tilt)).sum(-1) - logZg
            with torch.no_grad():
                wd = logw.exp()
                diag = {
                    "bm_post_entropy": -(wd * logw).sum(-1).mean(),
                    "bm_post_max": wd.max(-1).values.mean(),
                    "bm_eta": self.eta.detach().float(),
                    "bm_gamma_mean": torch.sigmoid(self.gamma_logit.detach().float()).mean(),
                    "bm_tilt_rms": tilt.pow(2).mean().sqrt(),
                    "bm_evidence_rms": E.pow(2).mean().sqrt(),
                    "bm_rho_mean": rho.mean(),
                }
                if not full:  # nats/token the mixture adds over the base softmax p0 (same h), valid targets only
                    valid = (targets >= 0).to(z.dtype)
                    diag["bm_gain_vs_base"] = ((out - (zy - lse0)) * valid).sum() / valid.sum().clamp_min(1.0)
        return out, diag

    def extra_flops_per_token(self, d: int, T: int) -> float:
        K, S, r = self.K, self.S, self.r
        mac = K * r * d + K * r * S + K * d          # c = A h, tilt = c U^T, prior G h
        if self.pool == "geo":
            mac += K * S                             # w-averaged tilt
        return 6.0 * mac + 3.0 * T * K               # + causal discount matmul (fwd + bwd)


class ZBayesMix(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.body = base_mod.GPT(cfg)          # unchanged baseline body and base head (built first: same init)
        self.bm_head = BayesMixHead(cfg)       # 'head' in the name -> Adam head group; 1-D -> scalar group

    def _base(self, idx: torch.Tensor):
        h = self.body.hidden(idx)
        return h, self.body.logits(h)          # fp32, soft-capped

    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Exact predictive log-probabilities [B, T, V] (log-softmax of them is the identity)."""
        h, z = self._base(idx)
        out, _ = self.bm_head(idx, h, z, None, full=True)
        return out

    @torch.no_grad()
    def posterior(self, idx: torch.Tensor) -> torch.Tensor:
        """log w_t(k) [B, T, K]: the in-context posterior over components (diagnostics and tests)."""
        h, z = self._base(idx)
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.bm_head.stats(idx, h, z, None)["logw"]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if targets is None:
            return self.forward_logits(idx)
        h, z = self._base(idx)
        logp, diag = self.bm_head(idx, h, z, targets)
        valid = targets >= 0
        nll = (-logp) * valid.to(logp.dtype)
        n = valid.sum().clamp_min(1)
        loss = nll.sum() / n
        aux: dict = dict(diag)
        if not self.training:
            aux["per_token_loss"] = nll.detach()
        return loss, aux

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for p in self.body.blocks.parameters() if p.ndim == 2]
        ids = {id(p) for p in muon}
        return {"muon": muon, "adam": [p for p in self.parameters() if id(p) not in ids]}

    def flops_per_token(self) -> float:
        return self.body.flops_per_token() + self.bm_head.extra_flops_per_token(self.cfg.n_embd, self.cfg.seq_len)


def build(cfg: ModelConfig) -> nn.Module:
    return ZBayesMix(cfg)
