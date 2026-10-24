"""Per-token mixture of K independent Hinton models as ONE causal HF model (critic 44552922: official K=2 zero-shot =
the per-token mixture, mean of member softmaxes, exported as one causal model, official pipeline unchanged).

HintonMixForCausalLM.forward() returns logits = log p_bar = logsumexp_k(log_softmax(z_k)) - log K (fp32), where z_k is
member k's own softcapped fp32 logits (modeling_hinton.HintonForCausalLM, unchanged). These logits are already
normalised, so the pipeline's log_softmax returns them unchanged. Members keep their own weights under members.<k>.
HintonMixModel (AutoModel) refuses to run: GLUE for a mixture is each member fine-tuned separately with the member
probabilities averaged (babylm/eval/glue_ensemble.py), never a fine-tune of this export.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_hinton import HintonConfig
from .configuration_hinton_mix import HintonMixConfig
from .modeling_hinton import HintonForCausalLM


class HintonMixPreTrainedModel(PreTrainedModel):
    config_class = HintonMixConfig
    base_model_prefix = "mix"

    def _init_weights(self, module):  # weights always come from model.safetensors
        pass


class HintonMixModel(HintonMixPreTrainedModel):
    def __init__(self, config: HintonMixConfig):
        super().__init__(config)
        self.post_init()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("a mixture has no single finetuning trunk: GLUE = members fine-tuned separately, "
                                  "member probabilities averaged (critic 44552922; babylm/eval/glue_ensemble.py)")


class HintonMixForCausalLM(HintonMixPreTrainedModel):
    def __init__(self, config: HintonMixConfig):
        super().__init__(config)
        if len(config.member_configs) < 2:
            raise ValueError("a mixture needs >= 2 members")
        self.members = nn.ModuleList(HintonForCausalLM(HintonConfig(**{k: v for k, v in mc.items()
                                                                        if k not in ("auto_map", "architectures")}))
                                     for mc in config.member_configs)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None,
                labels: torch.LongTensor | None = None, **kwargs) -> CausalLMOutput:
        lps = [F.log_softmax(m(input_ids=input_ids, attention_mask=attention_mask).logits.float(), -1)
               for m in self.members]
        z = torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(z[:, :-1].reshape(-1, z.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=z)
