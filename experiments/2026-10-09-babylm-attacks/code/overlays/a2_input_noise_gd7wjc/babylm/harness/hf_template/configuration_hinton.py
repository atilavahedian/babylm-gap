"""Hugging Face config for the Hinton sprint-1 baseline GPT (exported by babylm/harness/export_hf.py)."""
from transformers import PretrainedConfig


class HintonConfig(PretrainedConfig):
    model_type = "hinton_gpt"

    def __init__(self, vocab_size=16384, n_layer=6, n_head=6, n_embd=384, mlp_ratio=4.0, seq_len=1024,
                 n_value_embeds=None, softcap=15.0, attn_scale=None, rope_base=1024.0, rope_frac=0.25, unet=True,
                 x0_inject=True, embnorm=True, hidden_size=None, automodel_bidirectional=False, automodel_pool="first", bos_token_id=0,
                 eos_token_id=0, pad_token_id=0, **kwargs):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.mlp_ratio = mlp_ratio
        self.seq_len = seq_len
        self.n_value_embeds = n_value_embeds
        self.softcap = softcap
        self.attn_scale = attn_scale
        self.rope_base = rope_base
        self.rope_frac = rope_frac
        self.unet = unet
        self.x0_inject = x0_inject
        self.embnorm = embnorm
        self.hidden_size = n_embd if hidden_size is None else hidden_size
        self.automodel_bidirectional = bool(automodel_bidirectional)  # AutoModel attention (MNTP-trained models)
        if automodel_pool not in ("first", "last"):
            raise ValueError(f"automodel_pool must be 'first' or 'last', got {automodel_pool!r}")
        if automodel_pool == "last" and self.automodel_bidirectional:
            raise ValueError("automodel_pool='last' is for causal AutoModels only")
        self.automodel_pool = automodel_pool  # 'last': position 0 of AutoModel output = last real token (critic ref-0059)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
