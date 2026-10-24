"""Config for the vendored GPT-BERT model (BabyLM 2025 baselines).

The published configuration_gpt_bert.py overwrites config.json values with
hard-coded small-model defaults, so the 100M (768-wide) checkpoints would not
load through it. This replacement takes every value from config.json and maps
num_hidden_layers to the num_layers name the modeling code reads.
"""
from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "gpt_bert_vendored"

    def __init__(self, **kwargs):
        num_layers = kwargs.pop("num_layers", None)
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.num_layers = num_layers or kwargs.get("num_hidden_layers", 12)
