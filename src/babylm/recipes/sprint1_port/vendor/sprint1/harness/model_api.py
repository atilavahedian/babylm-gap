"""Sprint-1 model contract. Every candidate and the baseline implement exactly this.

A model file lives at sprint1/models/<name>.py and exposes:

    def build(cfg: ModelConfig) -> torch.nn.Module

The returned module must satisfy:

    loss, aux = model(idx, targets)
        idx:     LongTensor [B, T]  input token ids (GPT-2 tokenizer, vocab padded to 50304)
        targets: LongTensor [B, T]  next-token ids (idx shifted by one); -1 marks ignored positions
        loss:    scalar Tensor, mean cross-entropy in nats over non-ignored target positions.
                 This is THE scored quantity. Any auxiliary objective must be added separately and
                 reported in aux (e.g. aux["aux_loss"]) so the harness can log it and back-propagate
                 loss + aux["aux_loss"] while scoring only `loss`.
        aux:     dict of python floats / 0-d tensors for logging (may be empty).

Optional hooks (the harness checks with hasattr):

    model.param_groups() -> dict[str, list[nn.Parameter]]
        keys: "muon" (2-D hidden weight matrices, orthogonalised updates), "adam" (everything else:
        embeddings, lm head, norms, scalars). Default: all 2-D params not in embedding/lm_head -> muon.
    model.flops_per_token() -> float   forward+backward FLOPs per token, for cost accounting.
    model.on_step_end(step: int, total_steps: int) -> None   schedules living inside the model.
    model.reset_state() -> None   for models carrying state across a document (called at doc boundaries
        only if cfg.reset_state_on_doc is True; default training is standard fixed-window LM).

Scoring is tokenizer-agnostic: the harness reports val loss (nats/token on the shared GPT-2 tokenization)
AND bits per byte on the same held-out byte stream, so a candidate that changes the unit of prediction
must still accept GPT-2 token ids at this interface and return a per-token loss whose sum over the
sequence equals the total negative log-likelihood of the target bytes. Candidates that cannot express
their likelihood this way must say so in their idea card and propose an evaluation for the critic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    seq_len: int = 1024
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    # mlp hidden = mlp_ratio * n_embd
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    tie_embeddings: bool = False
    # generic knob bag for candidates; keep every candidate's knobs here, documented in its idea card
    extra: dict[str, Any] = field(default_factory=dict)
    reset_state_on_doc: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ModelConfig":
        d = dict(d)
        extra = d.pop("extra", {}) or {}
        known = {k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__}
        unknown = {k: v for k, v in d.items() if k not in ModelConfig.__dataclass_fields__}
        extra.update(unknown)
        return ModelConfig(extra=extra, **known)


# Size ladder (non-embedding parameter counts are approximate; the harness prints exact counts).
# Chosen for one spot L4 (24 GB): "small" trains in ~5 min, "medium" in ~15 min, at 1024 context.
SIZE_LADDER: dict[str, dict[str, Any]] = {
    "tiny":   dict(n_layer=2, n_head=2, n_embd=64,  seq_len=256),   # CPU smoke tests only
    "small":  dict(n_layer=6, n_head=6, n_embd=384, seq_len=1024),  # ~10.6M non-embedding
    "medium": dict(n_layer=12, n_head=8, n_embd=512, seq_len=1024), # ~37.7M non-embedding
}
