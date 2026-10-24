"""W LiveHead: an output layer that keeps learning from the stream it predicts (card sprint1/ideas/W_livehead.md).

The baseline body and head (sprint1/models/baseline.py, unchanged) plus FAST output parameters that are updated
INSIDE the forward pass from the stream itself and persist across windows and documents. They are the sufficient
statistics of an online estimate of the stream's density ratio r(v | context) = p_stream(v | ctx) / p_slow(v | ctx):

    F_obs[v]  +=  w_s * q~_s             when token v actually came after position s    (Hebbian: what happened)
    F_exp     +=  w_s * p_slow,s q~_s^T  (dense over v)                                 (anti-Hebbian: what the slow
                                                                                          model expected there)
    z_t(v)  =  softcap(W h_t)(v)  +  g_t * [ log(alpha + C_obs(v, t)) - log(alpha + C_exp(v, t)) ]
    C_x(v, t) = sum_k beta_k * a_{t,k} * F_x,k[v] . q~_t            (kernel-weighted, decayed counts)

  * q~_s = [sqrt(kappa / r) * norm(P h_s), 1]: a learned r-dim context key plus a constant, so the kernel
    q~_s . q~_t = 1 + kappa q_s.q_t (unit q) lies in [1 - kappa, 1 + kappa], kappa in (0, 1): every count is >= 0.
    The constant part alone is an "explained-away" unigram cache in ratio form (Kneser-Klakow dynamic marginals,
    with the slow model's own expected counts as the denominator); the key part conditions it on the context.
  * The read is the log of a kernel-smoothed ratio of observed to expected counts (one generalized-iterative-
    scaling step toward the stream's marginals in the feature space q~): tokens the slow model already predicts
    at the rate they occur get ~0 (explaining away), tokens that occur more than expected in similar contexts are
    raised, tokens predicted but absent are lowered. It has no step size and no feedback (the expected counts use
    the slow logits only), so it is bounded and stable on a stream of any length.
  * w_s = omega_s * decay: omega_s = 2 sigmoid(u.h_s / d + c) (learned: what is worth keeping);
    decay = lambda_k^{dt} * gamma_k^{#<|endoftext|> passed} with LEARNED half-life and document-boundary factor;
    g_t = c_g + u_g.h_t / d (learned read gate, ZERO at init: the model starts as the baseline exactly);
    alpha (learned pseudo-count), beta_k (learned timescale mix).
  * Consolidation is chunk-wise: chunk j is read with the state after chunk j-1 (in-window recency is attention's
    job). The model never reads `targets` except for the scored CE: the observed token of position s is idx[s+1];
    the last position's observed write is kept PENDING and completed with the next window's first input
    (stream mode). The scored loss is an exact softmax CE over the full vocabulary.

Where the learning happens (contract requirement):
  * Inference / eval: the fast statistics are updated in forward, closed form, no backprop.
  * Training: the same forward is differentiated THROUGH the fast updates (meta-learning): the key projection
    P, kappa, the gates, alpha, beta, the half-lives, the boundary factor and the whole body learn to make the
    fast estimate useful for later chunks and, with carried state (stream/pair training,
    sprint1/moonshot/w/pilot.py), for later WINDOWS. p_slow (the expected-count weights) is detached.

Modes:
  model.carry = False (default; the standard harness): every forward starts from empty statistics (in-window
      chunks only). model.carry = True (streaming eval / stream training): the statistics, the pending write
      and the stream position persist across forward calls per batch row; reset_state() clears them.
      model.carry_grad = True keeps the graph through the carried state (pair training); detach_state() cuts it.

cfg.extra knobs (all optional):
  w_rank        7      r, context-key dimension (each statistic has r+1 columns per timescale)
  w_chunk       seq_len // 8 (min 16)   consolidation interval
  w_half_lives  [4096] initial half-lives in tokens, one per timescale K (learned, log-parameterised)
  w_learn_decay True   False freezes the half-lives
  w_alpha       0.2    initial pseudo-count
  w_eot         50256 if vocab > 50256 else None   document-separator id (None: no boundary decay)
  w_gain        1.0    multiplies the read gate; 0.0 = "off" arm (identical FLOPs, logits == baseline)
  w_ckpt_head   False  activation checkpointing of each chunk's head (same numbers, less memory, +1 head fwd)
  null          False  True: the observed-token writes use FOREIGN tokens (a fixed block of training text, or a
                       seeded Zipf sample if no shard is found) instead of the stream's next tokens; the expected
                       counts stay real. Same parameters, FLOPs and code path; destroys exactly what the mechanism
                       claims to use (what actually came next in this stream).
  w_foreign     path to a harness shard (.bin) for the null's foreign block (default: first train shard in
                HINTON_DATA_DIR / data/fineweb10B, else Zipf sample); w_foreign_offset (int), w_foreign_len 65536
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from sprint1.harness.model_api import ModelConfig
from sprint1.models.baseline import GPT, CastedLinear, norm


def _foreign_block(cfg: ModelConfig, n: int) -> torch.Tensor:
    """A fixed block of TRAIN token ids for the null (never another batch row, never DEV)."""
    ex = cfg.extra
    V = cfg.vocab_size
    path = ex.get("w_foreign")
    if path is None:
        root = Path(os.environ.get("HINTON_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "fineweb10B"))
        cands = sorted(root.glob("fineweb_train_*.bin")) if root.exists() else []
        path = str(cands[0]) if cands else None
    if path is not None and Path(path).exists():
        try:
            from sprint1.harness.data import load_shard
            toks = load_shard(path)
            off = int(ex.get("w_foreign_offset", 1_000_003)) % max(1, len(toks) - n)
            blk = np.asarray(toks[off:off + n], dtype=np.int64)
            if len(blk) == n and blk.max() < V:
                return torch.from_numpy(blk.copy())
        except Exception:  # noqa: BLE001 - fall back to the synthetic block below
            pass
    g = np.random.Generator(np.random.PCG64(12345))
    pz = 1.0 / np.arange(1, V + 1, dtype=np.float64)
    pz /= pz.sum()
    perm = g.permutation(V)
    return torch.from_numpy(perm[g.choice(V, size=n, p=pz)].astype(np.int64))


class LiveHead(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        ex = cfg.extra
        d = cfg.n_embd
        self.r = int(ex.get("w_rank", 7))
        self.R = self.r + 1
        self.C = int(ex.get("w_chunk", max(16, cfg.seq_len // 8)))
        hl = ex.get("w_half_lives", [4096.0])
        hl = [float(x) for x in (hl if isinstance(hl, (list, tuple)) else [hl])]
        self.K = len(hl)
        self.null = bool(ex.get("null", False))
        self.gain = float(ex.get("w_gain", 1.0))
        eot = ex.get("w_eot", 50256 if cfg.vocab_size > 50256 else None)
        self.eot = None if eot is None else int(eot)
        # ---- slow parameters of the fast learner (all meta-trained through the fast updates) ----
        self.w_key = CastedLinear(d, self.r)                                    # P: context key projection
        self.w_kappa = nn.Parameter(torch.zeros(1))                             # kappa = sigmoid(.) = 0.5
        self.w_read = nn.Parameter(torch.zeros(d))                              # read gate g = c_g + u_g.h/d
        self.w_read_bias = nn.Parameter(torch.zeros(1))                         # zero: starts as the baseline
        self.w_write = nn.Parameter(torch.zeros(d))                             # omega = 2 sigmoid(u.h/d + c)
        self.w_write_bias = nn.Parameter(torch.zeros(1))
        self.w_alpha = nn.Parameter(torch.log(torch.tensor([float(ex.get("w_alpha", 0.2))])))
        self.w_mix = nn.Parameter(torch.zeros(self.K))                          # beta_k = exp(.)
        self.w_logh = nn.Parameter(torch.log(torch.tensor(hl)), requires_grad=bool(ex.get("w_learn_decay", True)))
        self.w_eot = nn.Parameter(torch.zeros(self.K))                          # gamma_k = sigmoid(.)
        if self.null:
            self.register_buffer("w_foreign", _foreign_block(cfg, int(ex.get("w_foreign_len", 65536))),
                                 persistent=False)
        # ---- stream state (plain attributes: never buffers, never in state_dict) ----
        self.ckpt_head = bool(ex.get("w_ckpt_head", False))  # recompute each chunk's head in backward (memory)
        self.carry = False
        self.carry_grad = False
        self._state: dict | None = None
        self.last_stats: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------------------------------ state API
    def reset_state(self) -> None:
        self._state = None

    def detach_state(self) -> None:
        if self._state is not None:
            self._state = {k: (tuple(x.detach() for x in v) if isinstance(v, tuple)
                               else (v.detach() if torch.is_tensor(v) else v)) for k, v in self._state.items()}

    # ------------------------------------------------------------------------------------------ core
    def _run(self, idx: torch.Tensor, targets: torch.Tensor | None = None, want_logits: bool = False):
        """Chunk-wise head + fast statistics. Returns (logits [B,T,V] or None, per-token CE [B,T] or None).
        The fast path uses only idx; targets enter only the CE."""
        h = self.hidden(idx)
        B, T = idx.shape
        V, d = self.cfg.vocab_size, self.cfg.n_embd
        dev = idx.device
        K, R, r = self.K, self.R, self.r
        with torch.autocast(device_type=dev.type, enabled=False):
            hf = h.float()
            kap = torch.sigmoid(self.w_kappa.float())
            q = norm(F.linear(hf, self.w_key.weight.float())) * (kap / r).sqrt()               # [B,T,r]
            qt = torch.cat([q, torch.ones(B, T, 1, device=dev)], -1)                          # [B,T,R]
            g = (self.w_read_bias.float() + hf @ self.w_read.float() / d) * self.gain         # [B,T]
            om = 2.0 * torch.sigmoid(self.w_write_bias.float() + hf @ self.w_write.float() / d)  # [B,T]
            alpha = torch.exp(self.w_alpha.float())
            logl = -math.log(2.0) * torch.exp(-self.w_logh.float())                            # [K]
            logg = F.logsigmoid(self.w_eot.float())                                            # [K]
            beta = torch.exp(self.w_mix.float())                                               # [K]
            eot = (idx == self.eot).float() if self.eot is not None else torch.zeros(B, T, device=dev)

            st = self._state if self.carry else None
            if st is not None and st["Fo"].shape[0] != B:
                st = None
            pos0 = st["pos"] if st is not None else torch.zeros(B, dtype=torch.long, device=dev)
            fbase = None
            if self.null:
                N = self.w_foreign.numel()
                fbase = st["fbase"] if (st is not None and "fbase" in st) else (idx[:, 0] * 7919) % N
                ar = torch.arange(T, device=dev)[None, :]
                wt = self.w_foreign[(fbase[:, None] + pos0[:, None] + ar + 1) % N]
            else:
                wt = torch.cat([idx[:, 1:], idx[:, -1:]], 1)
            wvalid = torch.ones(B, T, device=dev)
            wvalid[:, -1] = 0.0  # last position: the observed token is not visible yet (pending)

            Fo = st["Fo"] if st is not None else torch.zeros(B, V, K * R, device=dev)
            Fe = st["Fe"] if st is not None else torch.zeros(B, V, K * R, device=dev)
            if st is not None and st.get("pend") is not None:
                pw = st["pend"]                                                                # [B,K*R]
                tgt0 = self.w_foreign[(fbase + pos0) % self.w_foreign.numel()] if self.null else idx[:, 0]
                Fo = Fo + torch.zeros_like(Fo).scatter_add(1, tgt0[:, None, None].expand(B, 1, K * R), pw[:, None, :])

            outs, ces = [], []
            C = max(1, min(self.C, T))
            pend = None
            for s0 in range(0, T, C):
                s1 = min(T, s0 + C)
                c = s1 - s0
                ncum = torch.cumsum(eot[:, s0:s1], 1)                                          # [B,c]
                ar = torch.arange(1, c + 1, device=dev, dtype=torch.float32)
                logD = ar[None, :, None] * logl[None, None, :] + ncum[:, :, None] * logg[None, None, :]  # [B,c,K]
                # ---- read the consolidated statistics (state after the previous chunk, decayed to time t)
                Q = ((beta[None, None, :] * torch.exp(logD))[..., None] * qt[:, s0:s1, None, :]).reshape(B, c, K * R)
                tc = targets[:, s0:s1] if targets is not None else None
                if self.ckpt_head and torch.is_grad_enabled() and not want_logits:
                    ce_c, P, _ = checkpoint(self._chunk_head, h[:, s0:s1], Q, Fo, Fe, g[:, s0:s1], alpha, tc,
                                            False, use_reentrant=False)
                else:
                    ce_c, P, z = self._chunk_head(h[:, s0:s1], Q, Fo, Fe, g[:, s0:s1], alpha, tc, want_logits)
                    if want_logits:
                        outs.append(z)
                    del z
                if ce_c is not None:
                    ces.append(ce_c)
                # ---- consolidate this chunk (normalised to time s1 - 1)
                endD = logD[:, -1, :]                                                          # [B,K]
                wk = om[:, s0:s1, None] * torch.exp(endD[:, None, :] - logD)                   # [B,c,K]
                Wq = (wk[..., None] * qt[:, s0:s1, None, :]).reshape(B, c, K * R)              # [B,c,K*R]
                dec = torch.exp(endD)[:, :, None].expand(B, K, R).reshape(B, 1, K * R)
                Fe = Fe * dec + torch.bmm(P.transpose(1, 2), Wq)                               # expected counts
                del P
                Wo = Wq * wvalid[:, s0:s1, None]
                Fo = Fo * dec + torch.zeros_like(Fo).scatter_add(
                    1, wt[:, s0:s1, None].expand(B, c, K * R), Wo)                             # observed counts
                if s1 == T:
                    pend = Wq[:, -1]                    # weight exp(endD - logD) = 1 at the last position
            z = (torch.cat(outs, 1) if len(outs) > 1 else outs[0]) if want_logits else None
            ce = (torch.cat(ces, 1) if len(ces) > 1 else ces[0]) if targets is not None else None
            if self.carry:
                keep = {"Fo": Fo, "Fe": Fe, "pend": pend, "pos": pos0 + T}
                if self.null:
                    keep["fbase"] = fbase
                self._state = keep
                if not self.carry_grad:
                    self.detach_state()
            self.last_stats = {"w_g": g.detach().mean(), "w_omega": om.detach().mean(),
                               "w_alpha": alpha.detach().reshape(()), "w_kappa": kap.detach().reshape(()),
                               "w_halflife": torch.exp(self.w_logh.detach()).mean(),
                               "w_gamma": torch.sigmoid(self.w_eot.detach()).mean(),
                               "w_Fo": Fo.detach().sum(1).mean(), "w_Fe": Fe.detach().sum(1).mean()}
        return z, ce

    def _chunk_head(self, hc, Q, Fo, Fe, gc, alpha, tc, want_logits: bool):
        """Slow logits + fast log-ratio read + CE for one chunk; P = slow softmax (detached) for the write."""
        B, c = hc.shape[:2]
        V = self.cfg.vocab_size
        with torch.autocast(device_type=hc.device.type, enabled=hc.device.type == "cuda", dtype=torch.bfloat16):
            zb = self.logits(hc)                                                               # [B,c,V] fp32
        with torch.autocast(device_type=hc.device.type, enabled=False):
            co = torch.bmm(Q, Fo.transpose(1, 2))                                              # [B,c,V]
            cx = torch.bmm(Q, Fe.transpose(1, 2))
            z = zb + gc[:, :, None] * (torch.log(alpha + co) - torch.log(alpha + cx))
            ce = None if tc is None else F.cross_entropy(z.reshape(B * c, V), tc.reshape(B * c), ignore_index=-1,
                                                         reduction="none").view(B, c)
            with torch.no_grad():
                P = torch.softmax(zb.detach(), -1)                                             # slow predictions
        return ce, P, (z if want_logits else None)

    # ------------------------------------------------------------------------------------------ contract
    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self._run(idx, None, want_logits=True)[0]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        _, ce = self._run(idx, targets)
        n = (targets >= 0).sum().clamp_min(1)
        loss = ce.sum() / n
        if self.training:
            return loss, dict(self.last_stats)
        return loss, {"per_token_loss": ce.detach()}

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for p in self.blocks.parameters() if p.ndim == 2] + [self.w_key.weight]
        ids = {id(p) for p in muon}
        adam = [p for p in self.parameters() if id(p) not in ids and p.requires_grad]
        return {"muon": muon, "adam": adam}

    def fast_flops_per_token(self) -> float:
        """Forward+backward FLOPs/token of the fast learner (see the card's accounting)."""
        c = self.cfg
        d, V, KR, C = c.n_embd, c.vocab_size, self.K * self.R, max(1, min(self.C, c.seq_len))
        key = 6.0 * d * self.r + 12.0 * d                 # key projection + two gates (fwd + bwd)
        read = 2 * 6.0 * V * KR                           # two bmm reads of the statistics (fwd + 2 bwd)
        write = 4.0 * V * KR + 6.0 * KR                   # dense expected-count bmm (fwd + grad wrt weights), scatter
        state = 2 * 4.0 * V * KR / C                      # per-chunk decay of both statistics (fwd + bwd)
        elem = 3.0 * V + 2 * 3.0 * V + 3.0 * V            # softmax of the slow logits, two logs, the gated sum
        return key + read + write + state + elem

    def flops_per_token(self) -> float:
        return super().flops_per_token() + self.fast_flops_per_token()


def build(cfg: ModelConfig) -> nn.Module:
    return LiveHead(cfg)
