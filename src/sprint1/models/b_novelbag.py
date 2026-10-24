"""B2 — Novel-Future Bag prediction (NFB).  Idea card: sprint1/ideas/B_novelbag.md

Scored model: a plain causal LM (shared body from b_common). `loss` is plain next-token CE.

Training-only auxiliary: from the student's state h_t, predict WHICH TOKENS NEW TO THE CONTEXT will
appear in the next N positions (x_{t+2} .. x_{t+N+1}), scored in the student's OWN unembedding space:
    score(t, v) = <W_q h_t, W_out[v]> - log Q(v)
with a sampled softmax (in-batch frequency negatives, logQ correction). "New to the context" means the
token's first occurrence in the training window lies after t, i.e. exactly the token class that carries
~63% of the bits in the ledger diagnostic. Each novel content token thereby sends a positive gradient to
its output embedding from up to N preceding positions instead of 1 (the next-token CE), and every
position is trained to encode where the text is going (topic/plan), not only the next token.

Known family: future-summary / bag-of-future-words prediction (FSP, Mahajan et al. 2025), token-order
prediction (TOP, Zuhri et al. 2025), MTP. Twist here: novelty-filtered targets, shared unembedding,
sampled softmax so the aux costs ~T*d + S*d MACs/token instead of an extra full-vocab head.

Null control (cfg.extra["null"]=True): identical compute; the future bag (tokens and novelty mask) is
taken from a different sequence of the batch, so it is uninformative about this context.

Knobs (cfg.extra):
  null: bool=False
  nb_window: int=32       N, future window length (positions t+2 .. t+N+1 in x)
  nb_negatives: int=512   S, shared in-batch negatives per step
  nb_beta: float=0.3      aux weight
  nb_all_tokens: bool=False   ablation: use every future token, not only context-novel ones
  nb_detach_w: bool=False     ablation: aux trains only the body/query, not the shared unembedding
  body: "auto" | "baseline" | "mini"
  b_plain: bool=False      same body, no auxiliary at all (same-body reference arm if the shared
                            baseline does not expose the split API and "mini" is used)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.b_common import build_body, next_token_ce, rms_norm, roll_batch


def first_occurrence_mask(idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """novel[b, i] = True iff targets[b, i] (= x_{i+1}) does not occur in x_0..x_i = idx[b, :i+1].
    Ignored targets (-1) are never novel. O(T^2) bool compare, compile-friendly."""
    T = idx.size(1)
    eq = targets.unsqueeze(-1) == idx.unsqueeze(-2)  # [B, T(i), T(s)]
    ar = torch.arange(T, device=idx.device)
    seen = (eq & (ar.view(1, T) <= ar.view(T, 1))).any(-1)  # s <= i
    return (~seen) & (targets >= 0)


class NovelBagLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ex = cfg.extra
        self.cfg = cfg
        self.body = build_body(cfg)
        self.null = bool(ex.get("null", False))
        self.plain = bool(ex.get("b_plain", False))
        self.N = int(ex.get("nb_window", 32))
        self.S = int(ex.get("nb_negatives", 512))
        self.beta = float(ex.get("nb_beta", 0.3))
        self.all_tokens = bool(ex.get("nb_all_tokens", False))
        self.detach_w = bool(ex.get("nb_detach_w", False))
        d = cfg.n_embd
        self.q_proj = nn.Linear(d, d, bias=False)

    def param_groups(self):
        g = self.body.body_param_groups()
        g["muon"] = list(g["muon"]) + [self.q_proj.weight]
        return g

    def flops_per_token(self) -> float:
        if self.plain:
            return self.body.base_flops_per_token(self.cfg.seq_len)
        cfg = self.cfg
        d, T = cfg.n_embd, cfg.seq_len
        macs = d * d + T * d + self.S * d  # q proj, banded score matrix (dense), negatives
        # all three sit on the trainable path (fwd + 2x bwd)
        return self.body.base_flops_per_token(T) + 6.0 * macs

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        h = self.body.hidden(idx)
        z = self.body.logits(h)
        loss, ce_tok, valid = next_token_ce(z, targets)
        if not self.training:
            return loss, {"per_token_loss": ce_tok.detach()}
        if self.plain:
            return loss, {}
        nb, stats = self.novel_bag_loss(idx, h, targets, valid)
        aux = {"aux_loss": self.beta * nb, "nb_loss": nb.detach()}
        aux.update(stats)
        return loss, aux

    def positives(self, idx, targets, valid):
        """Returns (tok [B,T], pair [B,T(t),T(i)]): pair[b,t,i] marks tok[b,i] as a positive for query t.
        Idea: tok = own targets, novel-filtered, i in [t+1, t+N] (never i == t, the next token itself).
        Null: tok and novelty mask come from a different sequence of the batch."""
        T = targets.size(1)
        novel = valid if self.all_tokens else first_occurrence_mask(idx, targets)
        tok = targets.clamp_min(0)
        if self.null:
            tok, novel = roll_batch(tok), roll_batch(novel)
        ar = torch.arange(T, device=idx.device)
        lag = ar.view(1, T) - ar.view(T, 1)  # i - t
        band = (lag >= 1) & (lag <= self.N)
        pair = band.unsqueeze(0) & novel.unsqueeze(1) & valid.unsqueeze(2)
        return tok, pair, novel

    def novel_bag_loss(self, idx, h, targets, valid):
        tok, pair, novel = self.positives(idx, targets, valid)
        W = self.body.out_weight()
        if self.detach_w:
            W = W.detach()
        q = self.q_proj(rms_norm(h))  # [B,T,d]

        # sampled-softmax log Q from in-batch unigram counts of the valid targets
        flat_tok = targets.clamp_min(0).reshape(-1)
        flat_w = valid.reshape(-1).float()
        counts = torch.zeros(W.size(0), device=idx.device, dtype=torch.float32).index_add_(0, flat_tok, flat_w)
        logQ = torch.log(counts.clamp_min(1.0) / flat_w.sum().clamp_min(1.0))

        # shared negatives: S positions drawn uniformly from the batch (=> unigram-frequency sampling)
        neg_pos = torch.randint(0, flat_tok.numel(), (self.S,), device=idx.device)
        neg_tok = flat_tok[neg_pos]
        neg_ok = flat_w[neg_pos] > 0  # ignored-target positions are dropped from the negatives
        neg_bias = torch.where(neg_ok, -logQ[neg_tok], torch.full_like(logQ[neg_tok], -1e4))
        neg_logit = q @ W[neg_tok].to(q.dtype).t() + neg_bias.to(q.dtype)  # [B,T,S]
        lse_neg = torch.logsumexp(neg_logit.float(), dim=-1)  # [B,T]

        # positives: banded score matrix, query t vs future target index i in [t+1, t+N]
        E = W[tok].to(q.dtype)  # [B,T,d]  output embeddings of x_{i+1}
        score = torch.matmul(q, E.transpose(1, 2)).float() - logQ[tok].unsqueeze(1)  # [B, T(t), T(i)]
        pair_loss = F.softplus(lse_neg.unsqueeze(-1) - score)  # = -log softmax(pos | {pos} U negs)
        pairf = pair.float()
        n_pairs = pairf.sum()
        nb = (pair_loss * pairf).sum() / n_pairs.clamp_min(1.0)
        stats = {
            "nb_pairs_per_tok": (n_pairs / valid.float().sum().clamp_min(1.0)).detach(),
            "nb_novel_frac": (novel.float().sum() / valid.float().sum().clamp_min(1.0)).detach(),
        }
        return nb, stats


def build(cfg: ModelConfig) -> nn.Module:
    return NovelBagLM(cfg)
