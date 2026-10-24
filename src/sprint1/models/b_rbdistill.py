"""B1 — Rao-Blackwellised future-teacher distillation (RBD).  Idea card: sprint1/ideas/B_rbdistill.md
(Renamed from b_hindsight to avoid the unrelated tournament arm `hindsight`.)

Scored model: a plain causal LM (shared body from b_common). `loss` is plain next-token CE.

Training-only addition: a small FUTURE TEACHER that, for position t, sees
  * the student's causal state h_t (stop-gradient by default), and
  * the FUTURE tokens x_{t+2}, x_{t+3}, ... (local K-tap window + multi-scale backward EMAs),
  * but NEVER the target x_{t+1} itself,
and outputs a posterior over a coarse partition of the vocabulary: the student's top-k candidates
(detached, a function of the past only) plus one "tail" bucket. Its logits are the student's own
partition log-probs (stop-grad) plus a learned correction, so at init teacher == student.

  aux_loss = alpha * (H(sg(T), S_partition) - CE)  +  beta * CE_teacher(partition label)
  => the harness trains on  (1 - alpha) * CE  +  alpha * H(T, S)  +  beta * teacher CE

Why this can be unbiased: E_future[ p(x_{t+1} | past, future) | past ] = p(x_{t+1} | past) (tower rule),
so the future-conditioned posterior is a Rao-Blackwellisation of the one-hot target: same expectation, lower
variance (by E[Var(onehot | past, future)]). A calibrated teacher therefore leaves the optimum unchanged
and only reduces gradient noise and adds "which alternatives were plausible" information per token.

Null control (cfg.extra["null"]=True): identical compute, but the teacher's future tokens come from a
different sequence in the batch, so it carries no privileged information about x_{t+1}.

Knobs (cfg.extra):
  null: bool=False          teacher future from another sequence (uninformative control)
  rb_k: int=32              student top-k candidates in the partition
  rb_K: int=4               local future window (x_{t+2} .. x_{t+1+K})
  rb_gammas: list=[0.9,0.98] backward-EMA decays over x_{t+2..} (0 entries -> none)
  rb_rank: int=64           rank of the teacher's candidate-scoring space (W_out_sg @ P)
  rb_mlp: float=2.0         teacher MLP hidden = rb_mlp * d
  rb_alpha: float=0.5       distillation weight
  rb_mode: "convex"|"add"="convex"   convex: (1-a)CE + a*H(T,S_part)  (CE on the within-tail ranking is
                            down-weighted by (1-a));  add: CE + a*KL(T||S_part) (full CE kept, KD added)
  rb_beta: float=1.0        teacher CE weight
  rb_body_grad: bool=False  let the teacher CE back-propagate into the student body/embeddings
  rb_alpha_warmup: float=0.1  alpha ramps linearly 0 -> rb_alpha over this fraction of total steps
                            (driven by on_step_end; logged as aux["rb_alpha_eff"]. With 0, alpha is
                            constant from step 0. If the harness never calls on_step_end and warmup>0,
                            alpha stays 0 -- the log makes that visible.)
  rb_alpha_warmup_steps: int=200  warm-up length when the harness passes total_steps=-1 (time budget)
  rb_tokens_per_step: int=65536  only for amortising the per-step W_out @ P projection in FLOPs
  body: "auto" | "baseline" | "mini"
  b_plain: bool=False      same body, no auxiliary at all (same-body reference arm if the shared
                            baseline does not expose the split API and "mini" is used)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprint1.harness.model_api import ModelConfig
from sprint1.models.b_common import build_body, masked_mean, rms_norm, roll_batch, shift_future


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x <= 0, numerically stable."""
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.6931471805599453, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


class FutureTeacher(nn.Module):
    def __init__(self, d: int, K: int, gammas: list[float], rank: int, mlp_mult: float):
        super().__init__()
        self.K = K
        self.gammas = tuple(float(g) for g in gammas if g > 0)  # python constants: compile-friendly
        n_comp = (1 if K > 0 else 0) + len(self.gammas)
        assert n_comp > 0, "teacher needs some future input"
        # per-offset diagonal gates for the local future window
        # stored 1-D so the harness puts it in the AdamW *scalar* bucket (2-D adam params get the embed LR)
        self.local_gate = nn.Parameter(torch.ones(max(K, 1) * d) / max(K, 1))
        self.d = d
        self.fut_proj = nn.Linear(n_comp * d, d, bias=False)
        m = int(mlp_mult * d)
        self.fc1 = nn.Linear(2 * d, m, bias=False)
        self.fc2 = nn.Linear(m, rank + 1, bias=False)  # rank-dim correction + tail correction
        nn.init.zeros_(self.fc2.weight)  # teacher == student at init
        self.P = nn.Linear(d, rank, bias=False)  # candidate-scoring projection of (detached) W_out
        self.rank = rank

    def future_features(self, e: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """e[:, i] = embedding of targets[:, i] (= x_{i+1}); position t may use only i >= t+1."""
        e = e * valid.unsqueeze(-1).to(e.dtype)
        comps = []
        if self.K > 0:
            loc = torch.zeros_like(e)
            gate = self.local_gate.view(-1, self.d)
            for j in range(1, self.K + 1):
                loc = loc + gate[j - 1].to(e.dtype) * shift_future(e, j)
            comps.append(rms_norm(loc))
        if self.gammas:
            T = e.size(1)
            ar = torch.arange(T, device=e.device)
            lag = (ar.view(1, T) - ar.view(T, 1) - 1).to(torch.float32)  # i - t - 1
            strict = lag >= 0  # i >= t+1 : excludes the target x_{t+1} (i == t)
            for g in self.gammas:
                D = torch.where(strict, (1.0 - g) * torch.exp(lag.clamp_min(0) * math.log(g)),
                                torch.zeros_like(lag))
                comps.append(rms_norm(torch.matmul(D.to(e.dtype), e)))
        return rms_norm(self.fut_proj(torch.cat(comps, dim=-1)))


class RBDistillLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ex = cfg.extra
        self.cfg = cfg
        self.body = build_body(cfg)
        self.null = bool(ex.get("null", False))
        self.plain = bool(ex.get("b_plain", False))
        self.k = int(ex.get("rb_k", 32))
        self.alpha = float(ex.get("rb_alpha", 0.5))
        self.mode = str(ex.get("rb_mode", "convex"))
        assert self.mode in ("convex", "add"), self.mode
        self.beta = float(ex.get("rb_beta", 1.0))
        self.body_grad = bool(ex.get("rb_body_grad", False))
        self.tokens_per_step = int(ex.get("rb_tokens_per_step", 65536))
        self.alpha_warmup = float(ex.get("rb_alpha_warmup", 0.1))
        self.alpha_warmup_steps = int(ex.get("rb_alpha_warmup_steps", 200))  # used if total_steps is unknown
        # tensor (not python float) so a changing value never triggers torch.compile recompiles
        self.register_buffer("alpha_scale", torch.tensor(0.0 if self.alpha_warmup > 0 else 1.0),
                             persistent=True)
        self.teacher = FutureTeacher(cfg.n_embd, int(ex.get("rb_K", 4)), list(ex.get("rb_gammas", [0.9, 0.98])),
                                        int(ex.get("rb_rank", 64)), float(ex.get("rb_mlp", 2.0)))

    # ---- harness hooks -------------------------------------------------------------------------
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        g = self.body.body_param_groups()
        t = self.teacher
        g["muon"] = list(g["muon"]) + [t.fut_proj.weight, t.fc1.weight, t.fc2.weight, t.P.weight]
        g["adam"] = list(g["adam"]) + [t.local_gate]
        return g

    def on_step_end(self, step: int, total_steps: int) -> None:
        if self.alpha_warmup > 0:
            warm = self.alpha_warmup * total_steps if total_steps > 0 else self.alpha_warmup_steps
            self.alpha_scale.fill_(min(1.0, (step + 1) / max(1.0, warm)))

    def flops_per_token(self) -> float:
        if self.plain:
            return self.body.base_flops_per_token(self.cfg.seq_len)
        cfg, t = self.cfg, self.teacher
        d, T, V, r, k = cfg.n_embd, cfg.seq_len, cfg.vocab_size, t.rank, self.k
        m = t.fc1.out_features
        n_g = len(t.gammas)
        n_comp = (1 if t.K > 0 else 0) + n_g
        macs = (t.K * d                       # local future window
                + n_g * T * d                 # backward EMA as a dense triangular matmul
                + n_comp * d * d              # fut_proj
                + 2 * d * m + m * (r + 1)     # teacher MLP
                + k * r                       # candidate scoring
                + V * d * r / max(self.tokens_per_step, 1))  # W_out_sg @ P, once per step
        # teacher params get fwd+bwd (x3); the student only receives an extra logit gradient.
        return self.body.base_flops_per_token(T) + 6.0 * macs

    # ---- forward -------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        h = self.body.hidden(idx)
        z = self.body.logits(h)  # float32
        valid = targets >= 0
        # scored loss = plain next-token CE, written as lse - z_target so the same logsumexp pass
        # also serves the partition (one fewer full-vocab read than F.cross_entropy + logsumexp)
        lse = torch.logsumexp(z, dim=-1, keepdim=True)
        z_tgt = torch.gather(z, -1, targets.clamp_min(0).unsqueeze(-1))
        ce_tok = (lse - z_tgt).squeeze(-1)
        loss = masked_mean(ce_tok, valid)
        if not self.training:
            return loss, {"per_token_loss": ce_tok.masked_fill(~valid, 0.0).detach()}
        if self.plain:
            return loss, {}
        out = self.distill_terms(h, z, targets, valid, lse)
        a, b = self.alpha * self.alpha_scale, self.beta
        if self.mode == "convex":
            aux_loss = a * (out["kd_h"] - loss) + b * out["teacher_ce"]
        else:  # H(T,S) - H(T) = KL(T||S); H(T) is constant w.r.t. the student
            aux_loss = a * out["kd_h"] + b * out["teacher_ce"]
        aux = {"aux_loss": aux_loss, "rb_alpha_eff": a.detach()}
        aux.update({key: v.detach() for key, v in out.items()})
        return loss, aux

    def teacher_partition(self, h, z, targets, valid, lse=None):
        """Returns (cand [B,T,k], s_part [B,T,k+1] student partition log-probs,
        t_logp [B,T,k+1] teacher partition log-probs, label [B,T])."""
        k = self.k
        zf = z.float()
        if lse is None:
            lse = torch.logsumexp(zf, dim=-1, keepdim=True)
        cand = torch.topk(zf.detach(), k, dim=-1).indices  # function of the past only
        s_c = torch.gather(zf, -1, cand) - lse
        s_tail = log1mexp(torch.logsumexp(s_c, dim=-1, keepdim=True))
        s_part = torch.cat([s_c, s_tail], dim=-1)

        fut_src = targets.clamp_min(0)
        fut_valid = valid
        if self.null:
            fut_src, fut_valid = roll_batch(fut_src), roll_batch(fut_valid)
        e = self.body.in_embed(fut_src)
        if not self.body_grad:
            e = e.detach()
        f = self.teacher.future_features(e, fut_valid)
        hh = h if self.body_grad else h.detach()
        u = self.teacher.fc1(torch.cat([rms_norm(hh), f.to(hh.dtype)], dim=-1))
        u = self.teacher.fc2(F.relu(u).square())
        delta, tail_corr = u[..., :-1], u[..., -1:]
        W = self.body.out_weight()
        Wr = self.teacher.P(W.detach().to(delta.dtype))  # [V, r]
        corr = (Wr[cand] * delta.unsqueeze(-2)).sum(-1)  # [B,T,k]
        t_logits = s_part.detach() + torch.cat([corr, tail_corr], dim=-1).float()
        t_logp = F.log_softmax(t_logits, dim=-1)

        hit = cand == targets.unsqueeze(-1)
        label = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(targets, k))
        return cand, s_part, t_logp, label

    def distill_terms(self, h, z, targets, valid, lse=None):
        _, s_part, t_logp, label = self.teacher_partition(h, z, targets, valid, lse)
        tp = t_logp.detach().exp()
        kd_h_tok = -(tp * s_part).sum(-1)
        teacher_ce_tok = -torch.gather(t_logp, -1, label.unsqueeze(-1)).squeeze(-1)
        student_pce_tok = -torch.gather(s_part.detach(), -1, label.unsqueeze(-1)).squeeze(-1)
        ent_t = -(tp * t_logp.detach()).sum(-1)
        return {
            "kd_h": masked_mean(kd_h_tok, valid),
            "teacher_ce": masked_mean(teacher_ce_tok, valid),
            "kd_kl": masked_mean(kd_h_tok.detach() - ent_t, valid),
            # nats of privileged information the teacher extracts on the partition
            "teacher_gain": masked_mean(student_pce_tok - teacher_ce_tok.detach(), valid),
            "topk_cover": masked_mean((label < self.k).float(), valid),
        }


def build(cfg: ModelConfig) -> nn.Module:
    return RBDistillLM(cfg)
