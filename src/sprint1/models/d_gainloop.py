"""D1 GainLoop: depth routed by a learned value-of-computation (marginal loss gain) predictor.

Idea card: sprint1/ideas/D_gainloop.md. Contract: sprint1/harness/model_api.py.

Layout:  embed -> prelude blocks (dense) -> R routed steps -> coda blocks (dense) -> head.
  share="stack" (default): each step is a distinct block (same unique parameters as a dense
      n_layer model; tokens exit early, Mixture-of-Depths/CALM family).
  share="loop": one shared unit of `core_layers` blocks is applied R times with per-step norm-gain
      modulation (recurrent depth / Mixture-of-Recursions family).
Routing is monotone (a token that stops never resumes) and per token causal at inference: at step i
token t continues iff its router score s_i(t) exceeds a fixed threshold tau_i (a quantile of recent
training scores). During training the threshold is the batch quantile that activates exactly
frac[i] of all tokens (static top-k shape: compile friendly, no data-dependent control flow).
Stopped tokens are absent from that step's attention keys (kv_mode "active"), so KV memory and
attention reads shrink with depth too.

Router training modes (cfg.extra["router"]):
  "gain"    (default) every `probe_every` steps a no-grad dense side-pass on a few probe rows measures
            each token's loss after every exit depth; the router regresses the realised gain-to-go
            G_i(t) = loss(exit before step i) - loss(exit after the last step).
  "grad"    Mixture-of-Depths style straight-through gate: forward unchanged, router gets the
            first-order LM-loss signal d loss / d gate.
  "random"  NULL CONTROL (also cfg.extra["null"]=True): parameter-free hash of (token id, position,
            step) with thresholds giving the same expected depth histogram.
  "uniform" frac must be 0/1: every token runs the same number of steps (fixed-depth control).
  "dense"   all tokens run all R steps (max-compute reference, same parameters).
Implementation (cfg.extra["impl"]): "gather" (default) runs the matmuls of a routed step on the
k = frac*B*T routed tokens only (static shapes; the T x T attention stays dense-masked); "mask" runs
everything densely and discards (reference; identical outputs).

Knobs (cfg.extra): share, n_prelude (1), n_coda (1), core_layers, r_max, frac (list, monotone
non-increasing, frac[i] = fraction of all tokens that run step i), router, gain_target
("realized"|"kl"), gain_coef (1.0), probe_frac (0.125 of rows), probe_pos_frac (0.25), probe_every (1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sprint1.models.d_common import (Block, RaggedKV, apply_rotary, attend, block_matmul_params, hash_uniform,
                                     lm_loss, lm_loss_eval, rms, softcap, split_param_groups, sum_bytes)

BANK = 8192
BANK_WRITE = 512


def _block_routed(blk, xa, flat_idx, B, T, pos, mask, g_attn=None, g_mlp=None):
    """One pre-norm block applied to the k routed tokens xa [k, d] (flat indices into B*T).
    q/k/v projections, output projection and MLP run on k rows; only the masked T x T attention is
    dense. Non-routed key slots are zero and masked out; non-routed query outputs are never read."""
    att = blk.attn
    d = xa.shape[-1]
    a_in = rms(xa) if g_attn is None else rms(xa) * (1.0 + g_attn).to(xa.dtype)
    qkv_a = att.qkv(a_in)
    qkv = torch.zeros(B * T, qkv_a.shape[-1], device=xa.device, dtype=qkv_a.dtype).index_copy(0, flat_idx, qkv_a)
    q, k, v = qkv.view(B, T, 3, att.h, att.hd).permute(2, 0, 3, 1, 4)
    q, k = rms(q), rms(k)
    cos, sin = att.rotary(pos)
    q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    y = y.transpose(1, 2).reshape(B * T, d).index_select(0, flat_idx)
    xa = xa + att.proj(y)
    m_in = rms(xa) if g_mlp is None else rms(xa) * (1.0 + g_mlp).to(xa.dtype)
    return xa + blk.mlp(m_in)


def _default_frac(R: int) -> list[float]:
    if R == 1:
        return [1.0]
    rest = [0.5 - 0.3 * (i / max(1, R - 2)) for i in range(R - 1)] if R > 2 else [0.5]
    return [1.0] + [round(x, 4) for x in rest]


class GainLoop(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ex = dict(cfg.extra or {})
        self.cfg = cfg
        d, V = cfg.n_embd, cfg.vocab_size
        self.d, self.V, self.T = d, V, cfg.seq_len
        self.share = ex.get("share", "stack")
        assert self.share in ("stack", "loop"), self.share
        self.n_prelude = int(ex.get("n_prelude", 1))
        self.n_coda = int(ex.get("n_coda", 1))
        body = max(1, cfg.n_layer - self.n_prelude - self.n_coda)
        if self.share == "stack":
            self.core_layers = int(ex.get("core_layers", 1))
            self.R = int(ex.get("r_max", max(1, body // self.core_layers)))
            frac = ex.get("frac", _default_frac(self.R))
        else:
            self.core_layers = int(ex.get("core_layers", max(1, body // 2)))
            self.R = int(ex.get("r_max", 3))
            frac = ex.get("frac", [1.0, 0.6, 0.4] if self.R == 3 else _default_frac(self.R))
        mode = "random" if ex.get("null", False) else ex.get("router", "gain")
        assert mode in ("gain", "grad", "random", "uniform", "dense"), mode
        if mode == "dense":
            frac = [1.0] * self.R
        frac = [float(f) for f in frac]
        assert len(frac) == self.R, (frac, self.R)
        assert all(0.0 <= f <= 1.0 for f in frac) and all(a >= b for a, b in zip(frac, frac[1:])), frac
        if mode == "uniform":
            assert all(f in (0.0, 1.0) for f in frac), "uniform needs 0/1 fractions"
        self.mode, self.frac = mode, frac
        self.eval_frac = list(frac)  # inference budget; set_budget() can change it after training
        self.gain_target = ex.get("gain_target", "realized")
        self.gain_coef = float(ex.get("gain_coef", 1.0))
        self.probe_frac = float(ex.get("probe_frac", 0.125))
        self.probe_pos_frac = float(ex.get("probe_pos_frac", 0.25))
        self.probe_every = int(ex.get("probe_every", 1))
        self.impl = ex.get("impl", "gather")  # "gather" (training matmuls on routed tokens only) | "mask"
        assert self.impl in ("gather", "mask")
        # kv_mode "active": exited tokens leave deeper steps' keys (saves KV); "all": exited tokens stay
        # visible as keys from their frozen state (no KV saving; hedge against losing retrieval keys)
        self.kv_mode = ex.get("kv_mode", "active")
        assert self.kv_mode in ("active", "all")
        if self.kv_mode == "all":
            assert self.core_layers == 1, "kv_mode=all is defined for one block per step"
        self._step = 0

        mk = lambda: Block(d, cfg.n_head, cfg.mlp_ratio)  # noqa: E731
        self.embed = nn.Embedding(V, d)
        self.prelude = nn.ModuleList([mk() for _ in range(self.n_prelude)])
        n_units = 1 if self.share == "loop" else self.R
        self.units = nn.ModuleList([nn.ModuleList([mk() for _ in range(self.core_layers)]) for _ in range(n_units)])
        self.coda = nn.ModuleList([mk() for _ in range(self.n_coda)])
        self.lm_head = nn.Linear(d, V, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        self.router_w = nn.Parameter(torch.randn(self.R, d) * 0.02)
        self.router_b = nn.Parameter(torch.zeros(self.R))
        if self.share == "loop":
            self.mod_attn = nn.Parameter(torch.zeros(self.R, self.core_layers, d))
            self.mod_mlp = nn.Parameter(torch.zeros(self.R, self.core_layers, d))
        self.register_buffer("bank", torch.zeros(self.R, BANK), persistent=True)
        self.register_buffer("bank_valid", torch.zeros(self.R, BANK, dtype=torch.bool), persistent=True)
        self.register_buffer("bank_ptr", torch.zeros((), dtype=torch.long), persistent=True)

    # ---------------------------------------------------------------- routing
    def _cond(self, frac, i):
        prev = 1.0 if i == 0 else frac[i - 1]
        return 0.0 if prev <= 0 else min(1.0, frac[i] / prev)

    def _scores(self, h, i, idx, pos):
        if self.mode == "random":
            return hash_uniform(idx, pos, salt=i + 1)
        x = rms(h.detach() if self.mode == "gain" else h).float()
        return x @ self.router_w[i].float() + self.router_b[i].float()

    def _bank_write(self, i, s, prev):
        """Store BANK_WRITE scores sampled uniformly among the tokens eligible for step i (static shape)."""
        flat, ok = s.detach().flatten(), prev.flatten()
        keys = torch.rand(flat.shape, device=flat.device).masked_fill(~ok, -1.0)
        pick = torch.topk(keys, min(BANK_WRITE, flat.numel())).indices
        n = pick.numel()
        slots = (self.bank_ptr + torch.arange(n, device=flat.device)) % BANK
        self.bank[i, slots] = flat[pick].to(self.bank.dtype)
        self.bank_valid[i, slots] = ok[pick]

    def _tau_eval(self, i, s, prev, frac):
        """Fixed per-token threshold for inference: quantile of banked training scores; if the bank is
        empty (untrained model) fall back to this batch's quantile (documented, non-causal)."""
        c = self._cond(frac, i)
        if self.mode == "random":
            return torch.tensor(1.0 - c, device=s.device)
        vals = self.bank[i][self.bank_valid[i]]
        if vals.numel() < 64:
            vals = s.detach()[prev].float()
            if vals.numel() == 0:
                return torch.tensor(float("inf"), device=s.device)
        return torch.quantile(vals.float(), 1.0 - c)

    def _route(self, s, prev, i, frac, use_topk, taus=None):
        """Returns (active [B, T] bool, flat_idx or None). In training (use_topk) exactly
        k = round(frac[i] * B * T) tokens are chosen by top-k over eligible scores (static shape), and
        their flat indices are returned for the gather implementation."""
        c = self._cond(frac, i)
        if c >= 1.0:
            return prev, None
        if c <= 0.0:
            return torch.zeros_like(prev), None
        if use_topk:
            k = max(1, int(round(frac[i] * s.numel())))
            top = torch.topk(s.detach().masked_fill(~prev, float("-inf")).flatten(), k)
            if self.mode != "random":
                self._bank_write(i, s, prev)
            active = torch.zeros(prev.numel(), dtype=torch.bool, device=prev.device)
            active = active.index_fill(0, top.indices, True).view_as(prev) & prev
            return active, top.indices
        if self.mode == "random":
            return prev & (s >= 1.0 - c), None
        tau = taus[i] if taus is not None and taus[i] is not None else self._tau_eval(i, s, prev, frac)
        if taus is not None:
            taus[i] = tau
        return prev & (s >= tau), None

    # ---------------------------------------------------------------- trunk
    def _mods(self, i, j):
        if self.share != "loop":
            return None, None
        return self.mod_attn[i, j], self.mod_mlp[i, j]

    def _unit(self, i):
        return self.units[0 if self.share == "loop" else i]

    def _trunk(self, idx, collect=False, use_topk=None, taus=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        use_topk = self.training if use_topk is None else use_topk
        frac = self.frac if use_topk else self.eval_frac
        h = rms(self.embed(idx))
        kv_pre, kv_steps, kv_coda, actives = [], [], [], []
        for blk in self.prelude:
            h, kv = blk(h, pos)
            if collect:
                kv_pre.append(kv)
        h_pre = h
        active = torch.ones(B, T, dtype=torch.bool, device=idx.device)
        depth = torch.zeros(B, T, device=idx.device)
        causal = None
        for i in range(self.R):
            if frac[i] <= 0.0:
                actives.append(torch.zeros_like(active))
                kv_steps.append(None)
                continue
            s = self._scores(h, i, idx, pos)
            active, flat_idx = self._route(s, active, i, frac, use_topk, taus)
            mask = None
            if frac[i] < 1.0 and self.kv_mode == "active":
                if causal is None:
                    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
                eye = torch.eye(T, dtype=torch.bool, device=idx.device)
                mask = (causal[None] & (active[:, None, :] | eye[None]))[:, None]
            if flat_idx is not None and self.impl == "gather" and not collect and self.kv_mode == "active":
                # gather implementation: every matmul runs on the k routed tokens only
                d = h.shape[-1]
                hf = h.reshape(B * T, d)
                xa = hf.index_select(0, flat_idx)
                x0 = xa
                for j, blk in enumerate(self._unit(i)):
                    ga, gm = self._mods(i, j)
                    xa = _block_routed(blk, xa, flat_idx, B, T, pos, mask, ga, gm)
                upd = xa - x0
                if self.mode == "grad":
                    g = torch.sigmoid(s).reshape(B * T).index_select(0, flat_idx)
                    upd = upd * (1.0 + g - g.detach()).unsqueeze(-1).to(upd.dtype)
                h = hf.index_add(0, flat_idx, upd.to(hf.dtype)).view(B, T, d)
            else:
                hn, step_kv = h, []
                for j, blk in enumerate(self._unit(i)):
                    ga, gm = self._mods(i, j)
                    hn, kv = blk(hn, pos, mask, ga, gm)
                    step_kv.append(kv)
                upd = hn - h
                if self.mode == "grad" and frac[i] < 1.0 and self.training:
                    g = torch.sigmoid(s)
                    upd = upd * (1.0 + g - g.detach()).unsqueeze(-1).to(upd.dtype)
                h = h + upd * active.unsqueeze(-1).to(upd.dtype)
            depth = depth + active.to(depth.dtype)
            actives.append(active)
            kv_steps.append(step_kv if collect else None)
            step_kv = None
        if use_topk and self.mode not in ("random", "uniform", "dense"):
            self.bank_ptr.copy_((self.bank_ptr + BANK_WRITE) % BANK)
        for blk in self.coda:
            h, kv = blk(h, pos)
            if collect:
                kv_coda.append(kv)
        extras = dict(h_pre=h_pre, depth=depth, actives=actives, kv_pre=kv_pre, kv_steps=kv_steps, kv_coda=kv_coda)
        return h, extras

    def _head(self, h):
        return softcap(self.lm_head(rms(h)))

    # ---------------------------------------------------------------- probe (gain labels)
    def _probe(self, idx, targets, h_pre):
        """No-grad dense side-pass on probe rows: per-token loss at every exit depth -> gain-to-go labels.
        Returns router regression loss (grad only into router params) and diagnostics."""
        B, T = idx.shape
        P = max(1, int(round(self.probe_frac * B)))
        idx_p, tgt_p = idx[:P], targets[:P]
        pos = torch.arange(T, device=idx.device)
        S = max(1, int(round(self.probe_pos_frac * T)))
        sel = torch.randperm(T, device=idx.device)[:S]
        with torch.no_grad():
            hp = h_pre[:P].detach()
            states = [hp]
            for i in range(self.R):
                for j, blk in enumerate(self._unit(i)):
                    ga, gm = self._mods(i, j)
                    hp, _ = blk(hp, pos, None, ga, gm)
                states.append(hp)
            logps = []
            for k in range(self.R + 1):
                hc = states[k]
                for blk in self.coda:
                    hc, _ = blk(hc, pos)
                logps.append(torch.log_softmax(self._head(hc[:, sel]), -1))  # [P, S, V] float
            tsel = tgt_p[:, sel]
            ok = tsel >= 0
            tcl = tsel.clamp(min=0)
            nll = torch.stack([-lp.gather(-1, tcl.unsqueeze(-1)).squeeze(-1) for lp in logps])  # [R+1, P, S]
            if self.gain_target == "kl":
                pR = logps[-1].exp()
                G = torch.stack([(pR * (logps[-1] - lp)).sum(-1) for lp in logps[:-1]])
            else:
                G = nll[:-1] - nll[-1:]  # [R, P, S] realised gain-to-go
            # oracle novelty of the target within the window (diagnostic only)
            seen = (idx_p[:, None, :] == tsel[:, :, None]) & (torch.arange(T, device=idx.device)[None, None, :]
                                                               <= sel[None, :, None])
            novel = ~seen.any(-1)
        loss = torch.zeros((), device=idx.device)
        diag = {}
        n_routed = 0
        okf = ok.float()
        n_ok = okf.sum().clamp(min=1.0)
        for i in range(self.R):
            c = self._cond(self.frac, i)
            if c >= 1.0 or c <= 0.0:
                continue
            s = rms(states[i][:, sel].detach()).float() @ self.router_w[i].float() + self.router_b[i].float()
            g = G[i]
            loss = loss + (((s - g) ** 2) * okf).sum() / n_ok
            n_routed += 1
            if "cap_router" not in diag:  # first routed step: how much of the oracle gain does top-c capture?
                # static shapes (compile friendly): invalid positions rank last and contribute zero gain
                neg = torch.tensor(float("-inf"), device=g.device)
                gz = torch.where(ok, g, torch.zeros_like(g)).flatten()
                rank = lambda key: torch.where(ok, key, neg).flatten()  # noqa: E731
                k = max(1, int(round(c * g.numel())))
                best = gz[torch.topk(rank(g), k).indices].sum().clamp(min=1e-6)
                diag["cap_router"] = gz[torch.topk(rank(s.detach()), k).indices].sum() / best
                nv = novel.float() + 1e-3 * torch.rand_like(g)
                diag["cap_novel"] = gz[torch.topk(rank(nv), k).indices].sum() / best
                diag["cap_random"] = gz.sum() / n_ok * k / best
                sm, gm = (s.detach() * okf).sum() / n_ok, gz.sum() / n_ok
                cov = (((s.detach() - sm) * (g - gm)) * okf).sum()
                var = (((s.detach() - sm) ** 2) * okf).sum() * (((g - gm) ** 2) * okf).sum()
                diag["corr_router"] = cov / var.clamp(min=1e-12).sqrt()
                diag["gain_mean"] = gm
        diag["probe_gain_total"] = ((nll[0] - nll[-1]) * okf).sum() / n_ok
        return loss / max(1, n_routed), diag

    # ---------------------------------------------------------------- contract
    def forward(self, idx, targets):
        h, ex = self._trunk(idx)
        aux = {}
        if self.training:
            loss = lm_loss(self._head(h), targets)
        else:
            loss, aux["per_token_loss"] = lm_loss_eval(self._head(h), targets)
        aux.update({"avg_steps": ex["depth"].mean().detach(),
               "avg_layers": (self.n_prelude + self.n_coda + self.core_layers * ex["depth"].mean()).detach()})
        if self.training and self.mode == "gain" and (self.probe_every == 1 or self._step % self.probe_every == 0) and \
                any(0.0 < self._cond(self.frac, i) < 1.0 for i in range(self.R)):
            rl, diag = self._probe(idx, targets, ex["h_pre"])
            aux["aux_loss"] = self.gain_coef * rl
            aux["router_mse"] = rl.detach()
            aux.update(diag)
        return loss, aux

    def on_step_end(self, step, total_steps):
        self._step = step + 1

    def param_groups(self):
        return split_param_groups(self)

    def set_budget(self, frac):
        """Change the inference compute budget (fractions per step) without retraining."""
        assert len(frac) == self.R
        self.eval_frac = [float(f) for f in frac]

    @torch.no_grad()
    def calibrate(self, idx):
        """Fill the threshold bank from data (training-style routing), e.g. for an untrained model."""
        was = self.training
        self.train()
        self._trunk(idx, use_topk=True)
        self.train(was)

    # ---------------------------------------------------------------- cost accounting
    def _per_layer(self):
        return block_matmul_params(self.d, self.cfg.mlp_ratio)

    def _layers_eff(self, frac, attn=False):
        dense = self.n_prelude + self.n_coda
        return dense + self.core_layers * sum((f * f if attn else f) for f in frac)

    def flops_per_token(self):
        """Training FLOPs/token (fwd+bwd) for a gather implementation of the routing, plus the probe
        side-pass. NOTE: the current mask implementation *executes* dense compute for every step; see
        executed_flops_per_token()."""
        Pb, d, V, T = self._per_layer(), self.d, self.V, self.T
        f = self.frac
        main = 6 * (self._layers_eff(f) * Pb + d * V) + 6 * d * T * self._layers_eff(f, attn=True)
        probe = 0.0
        if self.mode == "gain":
            p = self.probe_frac / self.probe_every
            extra_core = self.core_layers * (self.R - sum(f)) * (2 * Pb + 2 * d * T)
            coda = (self.R + 1) * self.n_coda * (2 * Pb + 2 * d * T)
            head = (self.R + 1) * self.probe_pos_frac * 2 * d * V
            probe = p * (extra_core + coda + head)
        return float(main + probe)

    def executed_flops_per_token(self):
        """What the current training code actually executes: with impl="gather" the matmuls of routed
        steps run on routed tokens only but their T x T attention is dense-masked; impl="mask" is dense."""
        Pb, d, V, T = self._per_layer(), self.d, self.V, self.T
        steps = sum(1 for f in self.frac if f > 0)
        L_att = self.n_prelude + self.n_coda + self.core_layers * steps
        L_mm = self._layers_eff(self.frac) if self.impl == "gather" else L_att
        main = 6 * (L_mm * Pb + d * V) + 6 * d * T * L_att
        return float(main + (self.flops_per_token() - self._main_flops()))

    def _main_flops(self):
        Pb, d, V, T, f = self._per_layer(), self.d, self.V, self.T, self.frac
        return 6 * (self._layers_eff(f) * Pb + d * V) + 6 * d * T * self._layers_eff(f, attn=True)

    def inference_flops_per_token(self, ctx_len):
        Pb, d, V = self._per_layer(), self.d, self.V
        f = self.eval_frac
        if self.kv_mode == "all":  # active queries read all keys; exited tokens still project k, v
            steps = sum(1 for x in f if x > 0)
            kv_proj = 2 * d * d * self.core_layers * (steps - sum(f))
            att = self.n_prelude + self.n_coda + self.core_layers * sum(f)
            return float(2 * (self._layers_eff(f) * Pb + d * V + kv_proj) + 4 * d * ctx_len * att)
        return float(2 * (self._layers_eff(f) * Pb + d * V) + 4 * d * ctx_len * self._layers_eff(f, attn=True))

    # ---------------------------------------------------------------- decoding
    def forward_logits(self, idx):
        h, _ = self._trunk(idx)
        return self._head(h)

    @torch.no_grad()
    def prefill(self, idx):
        B, T = idx.shape
        H, hd = self.cfg.n_head, self.d // self.cfg.n_head
        dev, dt = idx.device, self.embed.weight.dtype
        taus = [None] * self.R
        h, ex = self._trunk(idx, collect=True, use_topk=False, taus=taus)
        mk = lambda: RaggedKV(B, H, hd, T + 64, dev, dt)  # noqa: E731
        pre = []
        for k, v in ex["kv_pre"]:
            c = mk(); c.append(k, v); pre.append(c)
        steps = []
        for i in range(self.R):
            row = []
            if ex["kv_steps"][i] is not None:
                for k, v in ex["kv_steps"][i]:
                    c = RaggedKV(B, H, hd, T // 2 + 64, dev, dt); c.append_masked(k, v, ex["actives"][i] if self.kv_mode == "active" else torch.ones_like(ex["actives"][i])); row.append(c)
            else:
                row = [RaggedKV(B, H, hd, 64, dev, dt) for _ in range(self.core_layers)]
            steps.append(row)
        coda = []
        for k, v in ex["kv_coda"]:
            c = mk(); c.append(k, v); coda.append(c)
        cache = dict(pos=T, pre=pre, steps=steps, coda=coda, taus=taus, steps_run=0, tokens=0)
        return self._head(h[:, -1]), cache

    @staticmethod
    def _kvfn(cache, rows=None):
        def fn(k, v, q):
            cache.append(k, v, rows)
            K, Vv, valid = cache.view(rows)
            return attend(q, K, Vv, valid)
        return fn

    @torch.no_grad()
    def decode_step(self, tok, cache):
        B = tok.shape[0]
        p = torch.tensor([cache["pos"]], device=tok.device)
        frac = self.eval_frac
        h = rms(self.embed(tok))
        for blk, c in zip(self.prelude, cache["pre"]):
            h = blk.step(h, p, self._kvfn(c))
        allrows = torch.arange(B, device=tok.device)
        rows = allrows
        for i in range(self.R):
            if frac[i] <= 0.0:
                break
            c_i = self._cond(frac, i)
            if c_i < 1.0 and rows.numel() > 0:
                s = self._scores(h[rows], i, tok[rows], p)[:, 0]
                if self.mode == "random":
                    tau = 1.0 - c_i
                else:
                    tau = cache["taus"][i]
                    if tau is None:
                        tau = self._tau_eval(i, s, torch.ones_like(s, dtype=torch.bool), frac)
                        cache["taus"][i] = tau
                rows = rows[s >= tau]
            if self.kv_mode == "all" and rows.numel() < B:
                # exited rows still write keys/values for this step from their frozen state
                out = torch.ones(B, dtype=torch.bool, device=tok.device)
                out[rows] = False
                ex_rows = allrows[out]
                blk = self._unit(i)[0]
                ga, _ = self._mods(i, 0)
                xin = rms(h[ex_rows]) if ga is None else rms(h[ex_rows]) * (1.0 + ga).to(h.dtype)
                _, k_ex, v_ex = blk.attn.heads(xin, p)
                cache["steps"][i][0].append(k_ex, v_ex, ex_rows)
            if rows.numel() == 0:
                if self.kv_mode == "active":
                    break
                continue
            hr = h[rows]
            for j, blk in enumerate(self._unit(i)):
                ga, gm = self._mods(i, j)
                hr = blk.step(hr, p, self._kvfn(cache["steps"][i][j], rows), ga, gm)
            h = h.index_copy(0, rows, hr)
            cache["steps_run"] += rows.numel()
        for blk, c in zip(self.coda, cache["coda"]):
            h = blk.step(h, p, self._kvfn(c))
        cache["pos"] += 1
        cache["tokens"] += B
        return self._head(h[:, 0]), cache

    def cache_bytes(self, cache):
        cs = list(cache["pre"]) + [c for row in cache["steps"] for c in row] + list(cache["coda"])
        return sum_bytes(cs)


def build(cfg):
    return GainLoop(cfg)
