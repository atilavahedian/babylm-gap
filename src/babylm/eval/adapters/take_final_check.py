"""python take_final_check.py EVAL2025_DIR : which token does the official --take_final flag pool? (critic ref-0345)

Runs the unchanged ModelForSequenceClassification.forward (bf55c1131e53654c2a87418f5629c26959acd710 classifier_model.py) with a stub trunk whose
hidden state at (batch b, position s) is the number 100*b + s and an identity head, on a right-padded batch (the
pipeline default), and prints the (b, s) each pooling mode selects next to the last NON-PAD token.
"""
import sys
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, sys.argv[1])
from evaluation_pipeline.finetune.classifier_model import ModelForSequenceClassification  # noqa: E402


class Stub(nn.Module):
    def forward(self, input_ids=None, attention_mask=None):
        B, S = input_ids.shape
        tag = (100 * torch.arange(B)[:, None] + torch.arange(S)[None, :]).float()
        return BaseModelOutput(last_hidden_state=tag[..., None])


am2 = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])   # lengths 6, 3, 5; right padding
am3 = am2.unsqueeze(1).repeat(1, am2.size(-1), 1).tril(diagonal=0)              # what the collate builds with --causal
ids = torch.ones_like(am2)
last = (am2.sum(-1) - 1).tolist()
print("last non-pad token (b, s):", [(b, s) for b, s in enumerate(last)])
for take_final, causal, mask in ((False, False, am2), (True, False, am2), (True, True, am3)):
    m = ModelForSequenceClassification.__new__(ModelForSequenceClassification)
    nn.Module.__init__(m)
    m.transformer, m.classifier, m.enc_dec, m.causal, m.take_final = Stub(), nn.Identity(), False, causal, take_final
    try:
        v = m(ids, mask).reshape(-1).long().tolist()
        print(f"take_final={take_final} causal={causal}: pools (b, s) =", [(x // 100, x % 100) for x in v])
    except Exception as e:   # the causal branch indexes the BATCH dimension with token positions
        print(f"take_final={take_final} causal={causal}: crashes: {type(e).__name__}: {e}")
