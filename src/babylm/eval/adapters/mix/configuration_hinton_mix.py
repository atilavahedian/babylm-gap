"""Config of a K-member per-token mixture of independent Hinton exports (babylm/eval/adapters/make_hinton_mix.py)."""
from transformers import PretrainedConfig


class HintonMixConfig(PretrainedConfig):
    model_type = "hinton_mix"

    def __init__(self, member_configs=None, vocab_size=16384, hidden_size=384, bos_token_id=0, eos_token_id=0,
                 pad_token_id=0, **kwargs):
        self.member_configs = list(member_configs or [])   # each member's own config.json (hinton_gpt) as a dict
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
