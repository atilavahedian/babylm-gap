"""Pooling-only adapter (critic ref-0058): the official GLUE finetune pools position 0 (classifier_model.py, no --causal /
--take_final), which for a causal model with a BOS token sees only BOS. This AutoModel runs the unchanged GPT-2 body and
writes each row's LAST non-padding hidden state into position 0. Nothing else changes: same weights, same other positions,
and AutoModelForCausalLM (zero-shot) still resolves to the stock GPT2LMHeadModel. Declared before scoring; GLUE scored
with it is reported next to the official BOS-pooled number, never instead of it."""
import torch
from transformers import GPT2Model


class GPT2LastTokenPoolModel(GPT2Model):
    def forward(self, input_ids=None, attention_mask=None, **kw):
        if attention_mask is not None and attention_mask.dim() == 3:   # (B,S,S) tril mask from --causal: last row = pad mask
            attention_mask = attention_mask[:, -1, :]
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kw)
        h = out.last_hidden_state
        if attention_mask is None:
            last = torch.full((h.size(0),), h.size(1) - 1, device=h.device, dtype=torch.long)
        else:   # right padding (pipeline default): last real token = number of real tokens - 1
            m = attention_mask.long()
            assert bool((m[:, 1:] <= m[:, :-1]).all()), 'lastpool adapter needs right padding (critic ref-0059)'
            last = m.sum(-1) - 1
        pooled = h[torch.arange(h.size(0), device=h.device), last]
        out.last_hidden_state = torch.cat([pooled[:, None], h[:, 1:]], 1)
        return out
