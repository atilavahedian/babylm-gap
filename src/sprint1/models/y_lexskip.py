"""Y_lexskip: the token's own successor prior gets its own table, mixed in by the final norm.
Idea card: sprint1/ideas/Y_lexskip.md (pilot, controls and GPU commands there).

THE CANDIDATE IS lex_place="pre" (pass --extra lex_place=pre). The default "post" is the arm registered before
the CPU pilot; it died there (tied its null). Status and history are in the card.

Measurement (sprint1/moonshot/y/, Gutenberg GPT-2 tokens, bench recipe, d=128, 410k tokens):
  * at equal data the 4-layer baseline is WORSE than its own 0-layer special case (token embedding -> norm ->
    head, a factorised neural bigram on the same batches) on word-continuation targets whose (input, target)
    pair was seen 1-31 times (6.20 vs 5.69 nats), while 0.34 nats better overall: the token's successor prior
    reaches the readout only through the trunk (MLP0 computes it; the input table E, which also has to encode
    meaning, carries it; the final RMSNorm divides it by everything the trunk adds);
  * an ungated after-norm prior ("post") recovers that cell (-0.36 vs its null) but loses as much on unseen
    pairs and rare targets: the prior must be gated by the context's evidence.

Mechanism ("pre", zero extra head FLOPs, no gate parameters):

    h_t      = norm(x_L,t + g * norm(U[k_t]))        k_t = idx_t (the current token)
    logits_t = softcap(W h_t)

U is a dedicated V x d token table (FLOP-free lookup, embedding Adam group), g a learned d-vector (scalar group,
init 1), x_L the baseline's residual stream after the last block. With a_t the trunk's rms, h_t rotates between
the trunk direction and the prior direction by tan(theta_t) = g / a_t: the trunk opts out of the prior where it
raises its own magnitude, and the readout scale stays fixed. "post" instead adds g * norm(U[k_t]) after the
final norm (an exact product of experts with a fixed weight). Body and head are baseline.py's (subclass).

Null twin (cfg.extra["null"]=True): identical table, gain, parameters, FLOPs and code path, but the key is
a token from a fixed pseudo-random earlier position of the same row, k_t = idx[b, pi(t)] with
pi(t) = floor(frac(t * phi) * (t - 7)) in [0, t-8] for t >= 8 (phi = golden ratio), and a reserved row
(V-1) for t < 8. The table rows are used with the same unigram statistics, the readout skip is the same,
but the key no longer identifies the token whose successor is being predicted (it carries only a weak
bag-of-words/topic signal, which can only help the null). Causal (pi(t) <= t) and row-local.

cfg.extra knobs:
  null (bool, False)            see above
  lex_place ("post"|"pre"|"gated")
                                post: after the final norm (the registered claim). pre: g*norm(U[k]) is added to
                                the residual stream before the final norm, so the trunk's own magnitude decides the
                                mix. gated: post placement times an explicit per-position gate read from the
                                trunk, gamma_t = 2*sigmoid(v . norm(x_L,t) + b) (v, b zero-init: gamma = 1 at init)
  lex_gain_init (float, 1.0)   initial value of every coordinate of g
  lex_init_std (float, 1.0)    init std of U (the baseline embedding uses N(0, 1))
  lex_table ("own"|"input")    own: the dedicated table U (the claim). input: reuse the trunk's input table E
                                (diagnostic: same placement and gain, no role separation, V*d fewer parameters)
  vocab_ids (str, ignored)     accepted for pilot scripts that pass a vocabulary map
Baseline knobs (embnorm, softcap, n_value_embeds, ...) pass through to the body unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.baseline import GPT, norm

PHI = (math.sqrt(5.0) - 1.0) / 2.0
NULL_MIN_LAG = 8


def null_positions(T: int) -> torch.Tensor:
    """pi(t) for the null key: a fixed pseudo-random earlier position at least 8 back; -1 for t < 8."""
    t = torch.arange(T, dtype=torch.float64)
    frac = torch.remainder(t * PHI, 1.0)
    pi = torch.floor(frac * (t - (NULL_MIN_LAG - 1))).long()
    pi = torch.where(t >= NULL_MIN_LAG, pi.clamp(min=0), torch.full_like(pi, -1))
    return pi


class LexSkipGPT(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        ex = cfg.extra
        d, V = cfg.n_embd, cfg.vocab_size
        self.null = bool(ex.get("null", False))
        self.place = str(ex.get("lex_place", "post"))
        assert self.place in ("post", "pre", "gated"), self.place
        if self.place == "gated":
            self.lex_gate_w = nn.Parameter(torch.zeros(d))
            self.lex_gate_b = nn.Parameter(torch.zeros(()))
        self.lex_table = str(ex.get("lex_table", "own"))
        assert self.lex_table in ("own", "input"), self.lex_table
        if self.lex_table == "own":
            self.lex_embed = nn.Embedding(V, d)
            nn.init.normal_(self.lex_embed.weight, mean=0.0, std=float(ex.get("lex_init_std", 1.0)))
        self.lex_gain = nn.Parameter(torch.full((d,), float(ex.get("lex_gain_init", 1.0))))
        self.register_buffer("null_pi", null_positions(cfg.seq_len), persistent=False)

    # ---- key ------------------------------------------------------------------------------------
    def lex_keys(self, idx: torch.Tensor) -> torch.Tensor:
        if not self.null:
            return idx
        T = idx.size(1)
        if T > self.null_pi.numel():
            self.null_pi = null_positions(T).to(idx.device)
        pi = self.null_pi[:T]
        gathered = idx.gather(1, pi.clamp(min=0).expand(idx.size(0), T))
        return torch.where(pi >= 0, gathered, torch.full_like(gathered, self.cfg.vocab_size - 1))

    def lex_vector(self, idx: torch.Tensor) -> torch.Tensor:
        table = self.lex_embed if self.lex_table == "own" else self.embed
        u = norm(table(self.lex_keys(idx)))
        return self.lex_gain.to(u.dtype) * u

    # ---- split API (overrides baseline.hidden; logits/out_weight/in_embed are inherited) ---------
    def trunk(self, idx: torch.Tensor) -> torch.Tensor:
        """The baseline's residual stream before the final norm (same code path as GPT.hidden)."""
        L = self.cfg.n_layer
        e = self.embed(idx)
        x = x0 = norm(e) if self.embnorm else e
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
        return x

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.trunk(idx)
        lex = self.lex_vector(idx)
        if self.place == "pre":
            return norm(x + lex)
        hx = norm(x)
        if self.place == "gated":
            gate = 2.0 * torch.sigmoid(hx @ self.lex_gate_w.to(hx.dtype) + self.lex_gate_b.to(hx.dtype))
            return hx + gate.unsqueeze(-1) * lex
        return hx + lex

    # ---- contract -------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        loss, aux = super().forward(idx, targets)
        if not self.training:
            aux["lex_gain_rms"] = self.lex_gain.detach().float().pow(2).mean().sqrt()
        return loss, aux

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for p in self.blocks.parameters() if p.ndim == 2]
        ids = {id(p) for p in muon}
        adam = [p for p in self.parameters() if id(p) not in ids]
        return {"muon": muon, "adam": adam}

    def flops_per_token(self) -> float:
        # baseline + RMSNorm of the looked-up row (~4d), gain multiply (d), add (d); x3 for fwd+bwd
        # (gated: + the gate's dot product and scale, 3d more)
        extra = 6.0 + (3.0 if self.place == "gated" else 0.0)
        return super().flops_per_token() + 3.0 * extra * self.cfg.n_embd


def build(cfg: ModelConfig) -> nn.Module:
    return LexSkipGPT(cfg)
