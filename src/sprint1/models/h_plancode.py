"""H2 PlanCode: plan-then-realise with an exact two-part code length at the next-token CE interface.

Idea card: sprint1/ideas/H_plancode.md. Contract: sprint1/harness/model_api.py.

Every K tokens the model commits to a discrete plan c_b in {0..M-1} for the next block, then realises the tokens
conditioned on it. The plan is a deterministic function of the data (an encoder reads the code-free lower stack
at the END of the block, so it "sees the future block"), and it is paid for in full: the scored loss is the
length of a lossless two-part code,

    loss = ( sum_t -log p(x_{t+1} | x_<=t, c_<=b(t))  +  sum_b -log p_prior(c_b | x_<=bK, c_<b) ) / N_tokens,

which upper-bounds the NLL of the marginal model sum_c p(c) p(x | c) (a receiver decodes c_b from the prior,
then the block's tokens). So BPB from this loss is an honest compression rate; aux reports the split.

Layout (L layers, plan layer lp = L//2, block size K):
  * layers [0, lp): plain causal stack, no plan input  -> h_low
  * encoder:  c_b = argmax W_enc rms(h_low[min(bK+K, T-1)])     (straight-through softmax gradient)
  * position t >= 1 receives code embedding E[c_{(t-1)//K}] added to its residual at layer lp
    (block b's code reaches positions bK+1..bK+K, whose targets are x[bK+2..bK+K+1])
  * layers [lp, L): causal stack -> h_final -> token logits
  * prior:    p(c_b | .) = softmax W_prior rms(h_final[bK])     (position bK holds codes < b only)
Generation: at each plan point the prior's argmax (or a sample) picks the block's code; prefill uses encoder
codes for blocks the prompt completes.

extra knobs (cfg.extra):
    plan_block   K (default 8)          plan_codes  M (default 64)       plan_layer  lp (default n_layer // 2)
    plan_mode    "learned" (default) | "off" (same backbone, no codes, no prior: the matched control)
                 | "shuffle" (each row gets another row's codes: no information about its own block, still
                   charged; the attribution null)
    plan_prior_grad  "st" (default: prior CE also trains the encoder, pure MDL) | "stop" (encoder trained by the
                 decoder only, the VQ-VAE + prior recipe; the scored loss still charges full code bits)
    plan_score   eval-mode scored loss: "two_part" (default; one pass, the code length above, an upper bound) |
                 "mixture" (EXACT next-token likelihood of the plan-mixture model, block code marginalised under the
                 prior; M+1 upper-half passes, eval only; learned mode only). Training is unaffected.
    plan_beta_warmup  fraction of training over which the prior term's weight in the TRAINING objective ramps
                 0 -> 1 (default 0.0 = honest MDL from step 0). The scored loss always charges full code bits;
                 the ramp is applied through a negative aux_loss and logged.
Encoder weights start at zero, so at init every block gets code 0 (zero bits once the prior learns it) and the
model equals the plain causal stack; codes diversify only if paying for them shortens the total code length.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.h_common import (Block, KVCache, Rotary, block_matmul_params, ce, rms, softcap,
                                     split_param_groups)


class PlanCodeLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ex = cfg.extra or {}
        self.cfg = cfg
        d, L = cfg.n_embd, cfg.n_layer
        self.d, self.L, self.H, self.hd, self.V = d, L, cfg.n_head, d // cfg.n_head, cfg.vocab_size
        self.K = int(ex.get("plan_block", 8))
        self.M = int(ex.get("plan_codes", 64))
        self.lp = int(ex.get("plan_layer", L // 2))
        self.mode = str(ex.get("plan_mode", "learned"))
        self.beta_warmup = float(ex.get("plan_beta_warmup", 0.0))
        # "st": the prior term's gradient also reaches the encoder (pure MDL, collapses early in CPU smokes);
        # "stop": the encoder learns only from the decoder (VQ-VAE + prior recipe). Scoring is identical.
        self.prior_grad = str(ex.get("plan_prior_grad", "st"))
        assert self.prior_grad in ("st", "stop")
        # eval-mode scored loss: "two_part" (1 pass, upper bound) | "mixture" (exact AR likelihood, M+1 upper passes)
        self.score = str(ex.get("plan_score", "two_part"))
        assert self.score in ("two_part", "mixture")
        assert not (self.score == "mixture" and self.mode == "shuffle"), "mixture needs own-row past codes"
        self.mix_chunk = int(ex.get("plan_mix_chunk", 1))
        assert self.mode in ("learned", "off", "shuffle")
        assert 1 <= self.lp < L and self.K >= 1 and self.M >= 2
        # training-only weight of the prior term (a buffer, so torch.compile does not recompile as it ramps)
        self.register_buffer("beta", torch.tensor(1.0 if self.beta_warmup <= 0 else 0.0), persistent=False)
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.blocks = nn.ModuleList([Block(d, cfg.n_head, cfg.mlp_ratio) for _ in range(L)])
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        if self.mode != "off":
            self.plan_enc_head = nn.Linear(d, self.M, bias=False)
            nn.init.zeros_(self.plan_enc_head.weight)
            self.plan_prior_head = nn.Linear(d, self.M, bias=False)
            nn.init.zeros_(self.plan_prior_head.weight)
            self.plan_emb = nn.Parameter(torch.randn(self.M, d) * 0.5)
        self.rotary = Rotary(self.hd)

    def param_groups(self):
        return split_param_groups(self)

    def on_step_end(self, step: int, total_steps: int) -> None:
        if self.beta_warmup > 0:
            self.beta.fill_(min(1.0, (step + 1) / max(1.0, self.beta_warmup * total_steps)))

    def head(self, x):
        return softcap(self.lm_head(rms(x)))

    # ------------------------------------------------------------------ geometry
    def n_blocks(self, T: int) -> int:
        return (T - 2) // self.K + 1 if T >= 2 else 0

    def block_of_pos(self, T: int, dev) -> torch.Tensor:
        return ((torch.arange(T, device=dev) - 1).clamp_min(0)) // self.K

    # ------------------------------------------------------------------ core
    def _run(self, idx: torch.Tensor, codes: torch.Tensor | None = None):
        """Full-sequence pass. codes (optional LongTensor [B, NB]) forces the plan codes (used for validity tests
        and for re-scoring generated text); otherwise codes come from the encoder (straight-through)."""
        B, T = idx.shape
        dev = idx.device
        cos, sin = self.rotary(torch.arange(T, device=dev))
        x = self.embed(idx)
        for blk in self.blocks[:self.lp]:
            x, _, _ = blk.causal(x, cos, sin)
        h_low = x
        out = dict(h_low=h_low)
        NB = self.n_blocks(T)
        if self.mode != "off" and NB > 0:
            plan_pos = torch.arange(NB, device=dev) * self.K
            enc_pos = (plan_pos + self.K).clamp_max(T - 1)
            if codes is None:
                e = self.plan_enc_head(rms(h_low[:, enc_pos])).float()  # [B, NB, M]
                soft = e.softmax(-1)
                hard_idx = e.argmax(-1)
                st = F.one_hot(hard_idx, self.M).float() + soft - soft.detach()
                if self.mode == "shuffle":
                    st, hard_idx = st.roll(1, 0), hard_idx.roll(1, 0)
                out["enc_soft"] = soft
            else:
                hard_idx = codes
                st = F.one_hot(codes, self.M).float()
            z = st.to(x.dtype) @ self.plan_emb.to(x.dtype)  # [B, NB, d]
            bpos = self.block_of_pos(T, dev)
            zt = z[:, bpos] * (torch.arange(T, device=dev) >= 1).to(x.dtype)[None, :, None]
            x = x + zt
            out.update(st=st, codes=hard_idx, plan_pos=plan_pos)
        for blk in self.blocks[self.lp:]:
            x, _, _ = blk.causal(x, cos, sin)
        out["h_final"] = x
        out["logits"] = self.head(x)
        if self.mode != "off" and NB > 0:
            out["prior_logp"] = F.log_softmax(self.plan_prior_head(rms(x[:, out["plan_pos"]])).float(), -1)
        return out

    def forward(self, idx: torch.Tensor, targets: torch.Tensor, codes: torch.Tensor | None = None):
        B, T = idx.shape
        if not self.training and self.score == "mixture" and self.mode == "learned" and codes is None:
            with torch.no_grad():
                nll, n, two = self._mixture(idx, targets, self.mix_chunk)
            return nll / n.clamp_min(1), {"two_part_nats_per_token": (two / n.clamp_min(1)).detach(),
                                          "mixture_gain_nats_per_token": ((two - nll) / n.clamp_min(1)).detach()}
        o = self._run(idx, codes)
        valid = targets != -1
        n_valid = valid.sum().clamp_min(1).float()
        tok_sum = ce(o["logits"], targets, reduction="sum")
        if "prior_logp" not in o:
            loss = tok_sum / n_valid
            return loss, {"token_ce": loss.detach()}
        NB = o["prior_logp"].shape[1]
        st = o["st"].detach() if self.prior_grad == "stop" else o["st"]
        code_nll = -(st * o["prior_logp"]).sum(-1)  # [B, NB]; forward value = -log p(c_hard)
        # charge a block if its plan point or any position it conditions has a scored target
        bpos = self.block_of_pos(T, idx.device)
        blk_valid = torch.zeros(B, NB, device=idx.device).scatter_reduce(
            1, bpos[1:].expand(B, -1), valid[:, 1:].float(), reduce="amax", include_self=True)
        blk_valid = torch.maximum(blk_valid, valid[:, o["plan_pos"]].float())
        code_sum = (code_nll * blk_valid).sum()
        loss = (tok_sum + code_sum) / n_valid
        with torch.no_grad():
            usage = F.one_hot(o["codes"], self.M).float().mean((0, 1))
            ent = -(usage * usage.clamp_min(1e-12).log()).sum()
        aux = {"token_ce": (tok_sum / n_valid).detach(), "plan_nats_per_token": (code_sum / n_valid).detach(),
               "plan_nats_per_block": (code_sum / blk_valid.sum().clamp_min(1)).detach(),
               "code_usage_perplexity": ent.exp(), "plan_beta": self.beta.clone()}
        if self.training and self.beta_warmup > 0:
            aux["aux_loss"] = (self.beta - 1.0) * code_sum / n_valid  # zero once the ramp is done
        return loss, aux

    @torch.no_grad()
    def forward_logits(self, idx: torch.Tensor, codes: torch.Tensor | None = None) -> torch.Tensor:
        return self._run(idx, codes)["logits"]

    @torch.no_grad()
    def mixture_nll(self, idx: torch.Tensor, targets: torch.Tensor, code_chunk: int = 8) -> dict:
        nll, n, two = self._mixture(idx, targets, code_chunk)
        return dict(nll_sum=float(nll), n_tokens=int(n), two_part_sum=float(two))

    def _mixture(self, idx: torch.Tensor, targets: torch.Tensor, code_chunk: int = 8):
        """EXACT next-token likelihood of the latent-plan mixture model (evaluation only; M+1 upper-stack passes).

        Model: earlier blocks' codes are the encoder's (deterministic functions of x<=bK, so known to a
        left-to-right reader); block b's code is a latent with prior p(c_b | x<=bK, c<b), marginalised:
            p(x_span(b) | past) = sum_c p(c | past) prod_{t in span(b)} p(x_{t+1} | x<=t, c<b, c).
        Its chain-rule per-token conditionals are p(x_{t+1}|x<=t) = sum_c post_t(c) p_c(x_{t+1}), a proper
        AR model, so this is an exact NLL (<= the two-part code length the training loss reports).
        Returns (nll_sum, n_tokens, two_part_sum) tensors; mixture_nll wraps them as floats."""
        assert self.mode == "learned"
        B, T = idx.shape
        dev = idx.device
        cos, sin = self.rotary(torch.arange(T, device=dev))
        x = self.embed(idx)
        for blk in self.blocks[:self.lp]:
            x, _, _ = blk.causal(x, cos, sin)
        h_low = x
        NB = self.n_blocks(T)
        plan_pos = torch.arange(NB, device=dev) * self.K
        enc_pos = (plan_pos + self.K).clamp_max(T - 1)
        codes = self.plan_enc_head(rms(h_low[:, enc_pos])).float().argmax(-1)  # [B, NB]
        span = self.block_of_pos(T, dev)
        span = torch.where(torch.arange(T, device=dev) >= 1, span, torch.full_like(span, -1))
        has_code = (span >= 0).to(x.dtype)[None, :, None]
        # true pass (encoder codes everywhere): KV per upper layer, prior, token log-probs
        xt = h_low + self.plan_emb[codes[:, span.clamp_min(0)]].to(x.dtype) * has_code
        kv = []
        for blk in self.blocks[self.lp:]:
            xt_new, k, v = blk.causal(xt, cos, sin)
            kv.append((k, v))
            xt = xt_new
        valid = targets != -1
        tg = targets.clamp_min(0)
        lp_true = torch.log_softmax(self.head(xt), -1).gather(-1, tg[..., None])[..., 0] * valid
        prior = F.log_softmax(self.plan_prior_head(rms(xt[:, plan_pos])).float(), -1)  # [B, NB, M]
        # mask over keys [true KV ; code-m KV]: earlier spans from the true pass, same span from pass m
        s_idx = torch.arange(T, device=dev)
        causal = s_idx[None, :] <= s_idx[:, None]
        m_true = causal & (span[None, :] < span[:, None])
        m_same = causal & (span[None, :] == span[:, None])
        mask = torch.cat([m_true, m_same], 1)[None, None]
        # per-code span log-likelihoods: [B, M, NB]
        span_ll = torch.zeros(B, self.M, NB, device=dev)
        onehot_span = F.one_hot(span.clamp_min(0), NB).float() * (span >= 0).float()[:, None]  # [T, NB]
        for c0 in range(0, self.M, code_chunk):
            cs = list(range(c0, min(self.M, c0 + code_chunk)))
            n = len(cs)
            xm = (h_low[:, None] + self.plan_emb[cs].to(x.dtype)[None, :, None, :] * has_code[:, None]).reshape(
                B * n, T, self.d)
            for i, blk in enumerate(self.blocks[self.lp:]):
                kt, vt = kv[i]
                kt = kt[:, None].expand(B, n, *kt.shape[1:]).reshape(B * n, *kt.shape[1:])
                vt = vt[:, None].expand(B, n, *vt.shape[1:]).reshape(B * n, *vt.shape[1:])
                xm, _, _ = blk.masked(xm, cos, sin, kt, vt, mask)
            lpm = torch.log_softmax(self.head(xm), -1).view(B, n, T, -1)
            lpm = lpm.gather(-1, tg[:, None, :, None].expand(B, n, T, 1))[..., 0] * valid[:, None]  # [B, n, T]
            span_ll[:, c0:c0 + n] = lpm @ onehot_span
        mix = torch.logsumexp(prior.transpose(1, 2) + span_ll, dim=1)  # [B, NB]
        nll = -(lp_true * (span < 0).float()).sum() - mix.sum()
        two_part = -lp_true.sum() - prior.gather(-1, codes[..., None]).sum()
        return nll, valid.sum().float(), two_part

    # ------------------------------------------------------------------ cost accounting
    def _per_pos_flops(self, ctx: float) -> float:
        return 2 * self.L * block_matmul_params(self.d, self.cfg.mlp_ratio) + self.L * 4 * ctx * self.d \
            + 2 * self.d * self.V

    def _plan_flops_per_token(self) -> float:
        if self.mode == "off":
            return 0.0
        # encoder + prior (M x d each) once per block, code embedding add once per token (gather: ~0)
        return 2 * 2 * self.M * self.d / self.K

    def flops_per_token(self) -> float:
        return float(3 * (self._per_pos_flops(self.cfg.seq_len / 2) + self._plan_flops_per_token()))

    def inference_flops_per_token(self, ctx_len: float) -> float:
        # generation needs only the prior (M x d per block); the encoder runs on prompt blocks only
        return float(self._per_pos_flops(ctx_len) + (0 if self.mode == "off" else 2 * self.M * self.d / self.K))

    # ------------------------------------------------------------------ cached decoding
    def _upper_from(self, x, cos, sin, cache_up: KVCache, start: int, layer_off: int):
        """Run upper layers on queries x [B,Q,d] at positions start..start+Q-1 attending to cached upper KV
        [0, start) plus causal own keys; writes KV."""
        Q = x.shape[1]
        dev = x.device
        mask = torch.cat([torch.ones(Q, start, dtype=torch.bool, device=dev),
                          torch.ones(Q, Q, dtype=torch.bool, device=dev).tril()], 1)[None, None]
        cache_up.ensure(start + Q)
        for i, blk in enumerate(self.blocks[layer_off:]):
            kc = cache_up.k[i][:, :, :start]
            vc = cache_up.v[i][:, :, :start]
            x, k, v = blk.masked(x, cos, sin, kc, vc, mask)
            cache_up.write(i, k, v, start)
        return x

    @torch.no_grad()
    def prefill(self, idx: torch.Tensor):
        B, P = idx.shape
        dev, dt = idx.device, self.embed.weight.dtype
        cos, sin = self.rotary(torch.arange(P, device=dev))
        lo = KVCache(self.lp, B, self.H, self.hd, P + 64, dev, dt)
        up = KVCache(self.L - self.lp, B, self.H, self.hd, P + 64, dev, dt)
        x = self.embed(idx)
        for li, blk in enumerate(self.blocks[:self.lp]):
            x, k, v = blk.causal(x, cos, sin)
            lo.write(li, k, v, 0)
        lo.n = P
        h_low = x
        cache = dict(lo=lo, up=up, z=None, n=P, codes=[])
        if self.mode == "off":
            for i, blk in enumerate(self.blocks[self.lp:]):
                x, k, v = blk.causal(x, cos, sin)
                up.write(i, k, v, 0)
            up.n = P
            return self.head(x[:, -1]), cache
        bstar = (P - 1) // self.K
        zt = torch.zeros(B, P, self.d, device=dev, dtype=x.dtype)
        if bstar > 0:  # complete blocks: encoder codes from the observed prompt
            enc_pos = torch.arange(bstar, device=dev) * self.K + self.K
            c = self.plan_enc_head(rms(h_low[:, enc_pos])).float().argmax(-1)  # [B, bstar]
            cache["codes"] = [c[:, i] for i in range(bstar)]
            bpos = self.block_of_pos(P, dev)
            sel = (torch.arange(P, device=dev) >= 1) & (bpos < bstar)
            zt[:, sel] = self.plan_emb[c[:, bpos[sel]]].to(x.dtype)
        x1 = h_low + zt
        for i, blk in enumerate(self.blocks[self.lp:]):
            x1, k, v = blk.causal(x1, cos, sin)
            up.write(i, k, v, 0)
        up.n = P
        p0 = bstar * self.K
        cb = self.plan_prior_head(rms(x1[:, p0])).float().argmax(-1)  # prior picks the open block's code
        cache["codes"].append(cb)
        z = self.plan_emb[cb].to(x.dtype)[:, None]
        cache["z"] = z
        if P - 1 > p0:  # recompute the open block's prompt positions with its code
            s = p0 + 1
            xq = h_low[:, s:] + z
            xf = self._upper_from(xq, cos[s:], sin[s:], up, s, self.lp)
            last = xf[:, -1]
        else:
            last = x1[:, -1]
        return self.head(last), cache

    @torch.no_grad()
    def decode_step(self, tok: torch.Tensor, cache: dict):
        t = cache["n"]
        lo, up = cache["lo"], cache["up"]
        cos, sin = self.rotary(torch.tensor([t], device=tok.device))
        x = self.embed(tok)
        lo.ensure(t + 1)
        for li, blk in enumerate(self.blocks[:self.lp]):
            kc, vc = lo.get(li)
            x, k, v = blk.masked(x, cos, sin, kc, vc, None)
            lo.write(li, k, v, t)
        lo.n = t + 1
        if cache["z"] is not None and t >= 1:
            x = x + cache["z"]
        up.ensure(t + 1)
        for i, blk in enumerate(self.blocks[self.lp:]):
            kc, vc = up.get(i)
            x, k, v = blk.masked(x, cos, sin, kc, vc, None)
            up.write(i, k, v, t)
        up.n = t + 1
        cache["n"] = t + 1
        if self.mode != "off" and t % self.K == 0:
            cb = self.plan_prior_head(rms(x[:, -1])).float().argmax(-1)
            cache["codes"].append(cb)
            cache["z"] = self.plan_emb[cb].to(x.dtype)[:, None]
        return self.head(x[:, -1]), cache

    def cache_bytes(self, cache: dict) -> dict:
        a, b = cache["lo"].bytes(), cache["up"].bytes()
        return dict(allocated=a["allocated"] + b["allocated"], used=a["used"] + b["used"])

    @torch.no_grad()
    def generate_plain(self, prompt: torch.Tensor, n_new: int):
        lg, cache = self.prefill(prompt)
        out = []
        for _ in range(n_new):
            t = lg.argmax(-1, keepdim=True)
            out.append(t)
            lg, cache = self.decode_step(t, cache)
        codes = torch.stack(cache["codes"], 1) if cache["codes"] else None
        return torch.cat(out, 1), codes


def build(cfg: ModelConfig) -> nn.Module:
    return PlanCodeLM(cfg)
