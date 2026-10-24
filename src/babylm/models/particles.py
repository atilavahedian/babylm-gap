"""Shared-trunk particle ensemble (moonshot card PE: docs/babylm/moonshot/cards/
W1_one-forward-shared-trunk-particle-ensemble-submitted-as-one-.md). Known idea (Bootstrapped DQN, TreeNet, MoS), untested
regime.

One connected network, one forward per window. Shared: the tied embedding / unembedding, x0, value-embedding tables (if
any) and blocks 0..Ls-1. Private: K particles each own blocks Ls..L-1 and their U-net skip weights. Particle 0 IS the
sprint-1 GPT (same modules, names and init draws); particles 1..K-1 are built after it, so K = 1 is the port model bit
for bit. U-net (n_enc = L // 2): a particle's skip stack holds the shared encoder outputs, then its own private ones;
decoder blocks pop it with the particle's own weights. Final RMSNorm (parameter-free), unembedding and softcap shared.

Output: p_bar = (1/K) sum_k p_k (log-mean-exp of the K fp32 softcapped log-softmaxes). Eval loss and per_token_loss are
-log p_bar(y), so the harness's evaluate_stream scores the ensemble. Training loss (cfg.extra pe_objective):
  particles (PE arms): (1/K) sum_k CE_k   each particle fits its own CE (Jeffares 2023); the trunk gets the mean
  mixture   (NJ null): CE(log p_bar)      same net, forward, FLOPs and output; the collusion objective
Both need only each particle's per-token CE at the target, so no [K, B, T, V] mixture tensor is built.

cfg.extra knobs (on top of the sprint-1 GPT's):
  pe_k (int, default 1)              particles K
  pe_shared (int, default 4)         Ls. K > 1 needs 4 <= Ls <= L - L // 2: the critic (00:40Z Q1, ref-0110) counts a
                                     grouped design once only if every particle depends on shared blocks, and Ls < 4
                                     needs a new critic entry; decoder blocks must be private so each particle owns its
                                     skip weights.
  pe_objective (particles | mixture, default particles)
  pe_tied (bool, default False)      tied twin: every particle re-runs particle 0's private modules, one function at
                                     K x the FLOPs (CPU code-path and FLOP-ledger check only, never a GPU arm)
  pe_private_wd_scale (float, default 1.0)  Muon weight decay on all private block matrices = scale x --muon_wd (PE3w)
flops_per_token counts executed matmuls (Ls + K (L - Ls) blocks and K head matmuls), never parameters, so the tied twin
logs K x as well. hidden() (AutoModel contract) is the mean of the K final normalised particle states; the particles
read out through one shared unembedding, so they share one readout space. mix_logprobs() is log p_bar.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.baseline import GPT, Block, norm

OBJECTIVES = ("particles", "mixture")
MIN_SHARED = 4


class ParticleGPT(GPT):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        ex = cfg.extra
        L = cfg.n_layer
        self.n_particles = K = int(ex.get("pe_k", 1))
        self.n_shared = Ls = int(ex.get("pe_shared", MIN_SHARED))
        self.objective = str(ex.get("pe_objective", "particles"))
        self.tied = bool(ex.get("pe_tied", False))
        self.private_wd_scale = float(ex.get("pe_private_wd_scale", 1.0))
        if K < 1:
            raise ValueError(f"pe_k must be >= 1, got {K}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"pe_objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if K > 1 and not MIN_SHARED <= Ls <= L - self.n_enc:
            raise ValueError(f"pe_shared must be in [{MIN_SHARED}, {L - self.n_enc}] for K > 1 (n_layer {L}), got {Ls}")
        hidden = int(round(cfg.mlp_ratio * cfg.n_embd))
        n_priv = 0 if K == 1 or self.tied else K - 1
        self.pblocks = nn.ModuleList(nn.ModuleList(Block(cfg, hidden) for _ in range(L - Ls)) for _ in range(n_priv))
        self.pskip = nn.ParameterList(nn.Parameter(torch.ones(max(self.n_enc, 1))) for _ in range(n_priv))
        if self.private_wd_scale != 1.0:
            self.make_optimizer = make_optimizer_private_wd  # read by the harness (train.py: model.make_optimizer)

    # ---- particles ------------------------------------------------------------------------------
    def stacks(self) -> list[tuple[list[nn.Module], torch.Tensor]]:
        """(private blocks, skip weights) per particle; particle 0 = the GPT's own blocks[Ls:] and skip_weights."""
        own = (list(self.blocks[self.n_shared:]), self.skip_weights)
        if self.tied:
            return [own] * self.n_particles
        return [own] + [(list(b), w) for b, w in zip(self.pblocks, self.pskip)]

    def private_matrices(self) -> list[nn.Parameter]:
        seen: set[int] = set()
        out = []
        for blocks, _ in self.stacks():
            for b in blocks:
                for p in b.parameters():
                    if p.ndim == 2 and id(p) not in seen:
                        seen.add(id(p))
                        out.append(p)
        return out

    def particle_hiddens(self, idx: torch.Tensor) -> torch.Tensor:
        """[K, B, T, d] final normalised hidden state of every particle."""
        if self.n_particles == 1:
            return super().hidden(idx)[None]
        L, Ls = self.cfg.n_layer, self.n_shared
        e = self.embed(idx)
        x = x0 = norm(e) if self.embnorm else e
        ve_at: list[torch.Tensor | None] = [None] * L
        for i, emb in enumerate(self.value_embeds):
            ve_at[i] = ve_at[L - self.n_ve + i] = emb(idx)
        skips: list[torch.Tensor] = []
        for i in range(Ls):  # Ls <= L - n_enc: shared blocks never pop a skip
            x = self.blocks[i](x, ve_at[i], x0)
            if self.unet and i < self.n_enc:
                skips.append(x)
        outs = []
        for blocks, sw in self.stacks():
            xk, sk = x, list(skips)
            for j, block in enumerate(blocks):
                i = Ls + j
                if self.unet and i >= L - self.n_enc and sk:
                    xk = xk + sw[i - (L - self.n_enc)] * sk.pop()
                xk = block(xk, ve_at[i], x0)
                if self.unet and i < self.n_enc:
                    sk.append(xk)
            outs.append(norm(xk))
        return torch.stack(outs)

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        if self.n_particles == 1:
            return super().hidden(idx)
        return self.particle_hiddens(idx).mean(0)

    def mix_logprobs(self, idx: torch.Tensor) -> torch.Tensor:
        """log p_bar [B, T, V] (fp32)."""
        lp = torch.stack([F.log_softmax(self.logits(h), -1) for h in self.particle_hiddens(idx)])
        return torch.logsumexp(lp, 0) - math.log(self.n_particles)

    # ---- contract -------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        if self.n_particles == 1:
            return super().forward(idx, targets)
        B, T = idx.shape
        y = targets.reshape(B * T)
        ces = torch.stack([F.cross_entropy(self.logits(h).view(B * T, -1), y, ignore_index=-1, reduction="none")
                           for h in self.particle_hiddens(idx)])  # [K, B*T], 0 at ignored targets
        n = (y >= 0).sum().clamp_min(1)
        mix = -(torch.logsumexp(-ces, 0) - math.log(self.n_particles))  # -log p_bar(y); 0 at ignored targets
        member = ces.sum(1) / n
        aux = {"pe_member_ce": member.mean().detach(), "pe_mix_ce": (mix.sum() / n).detach()}
        if self.training:
            return (member.mean() if self.objective == "particles" else mix.sum() / n), aux
        aux["per_token_loss"] = mix.view(B, T).detach()
        return mix.sum() / n, aux

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        muon = [p for m in (self.blocks, self.pblocks) for p in m.parameters() if p.ndim == 2]
        ids = {id(p) for p in muon}
        return {"muon": muon, "adam": [p for p in self.parameters() if id(p) not in ids]}

    def flops_per_token(self) -> float:
        if self.n_particles == 1:
            return super().flops_per_token()
        c = self.cfg
        per_block = sum(p.numel() for p in self.blocks[0].parameters() if p.ndim == 2)
        n_exec = self.n_shared + self.n_particles * (c.n_layer - self.n_shared)
        n_mat = n_exec * per_block + self.n_particles * self.lm_head.weight.numel()
        return 6.0 * n_mat + 6.0 * n_exec * c.seq_len * c.n_embd


def make_optimizer_private_wd(model: ParticleGPT, cfg: ModelConfig, lrs: dict[str, float]):
    """The harness optimizer with the private block matrices in their own Muon group at pe_private_wd_scale x WD."""
    from sprint1.harness.optim import Muon, build_optimizer
    ex = cfg.extra
    opt = build_optimizer(model, lrs, muon_momentum=ex["muon_momentum"], muon_wd=ex["muon_wd"],
                          adam_betas=tuple(ex["adam_betas"]), adam_eps=ex["adam_eps"], adam_wd=ex["adam_wd"])
    mu = [o for o in opt.optimizers if isinstance(o, Muon)]
    assert len(mu) == 1 and len(mu[0].param_groups) == 1, "unexpected Muon layout"
    priv = model.private_matrices()
    ids = {id(p) for p in priv}
    g = mu[0].param_groups[0]
    g["params"] = [p for p in g["params"] if id(p) not in ids]
    mu[0].add_param_group(dict(params=priv, weight_decay=ex["muon_wd"] * model.private_wd_scale, name="private"))
    return opt


def build(cfg: ModelConfig) -> nn.Module:
    return ParticleGPT(cfg)
