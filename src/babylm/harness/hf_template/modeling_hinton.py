"""Standalone Hugging Face copy of sprint1/models/baseline.py (exported by babylm/harness/export_hf.py).

Same parameters and maths as the training model: untied embedding / head, RMSNorm without gain, half-truncated
RoPE, QK-norm, ReLU^2 MLP, value embeddings, x0 injection, U-net skips, fp32 logit softcap. HintonForCausalLM.forward() returns
CausalLMOutput with fp32 softcapped logits; HintonModel (AutoModel) returns BaseModelOutput with the final normalised
hidden states. attention_mask (2-D key padding or 3-D query x key, see build_mask): with right padding (what the
BabyLM pipeline uses) real positions are unaffected. Positions are absolute indices, so left padding is unsupported.
config.automodel_bidirectional (set by the exporter for models trained with the MNTP hybrid): HintonModel (AutoModel,
the finetuning trunk) attends bidirectionally, as GPT-BERT's AutoModel does, so the classifier's encoding[:, 0] sees
the whole input; key padding is still masked. HintonForCausalLM is always causal.
config.automodel_pool = "last" (causal AutoModels only; critic ref-0058 / ref-0059): HintonModel writes each row's
last real token state (attention_mask.sum(-1) - 1; a 3-D mask uses its last row; right padding asserted) into
position 0 of last_hidden_state, so the classifier's encoding[:, 0] pools the last token. Other positions unchanged.
"""
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from .configuration_hinton import HintonConfig


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float, frac: float):
        super().__init__()
        n_rot = max(1, int(head_dim * frac))
        half = head_dim // 2
        n_rot = min(n_rot, half)
        freq = (1.0 / base) ** torch.linspace(0, 1, steps=n_rot, dtype=torch.float32)
        freq = torch.cat([freq, freq.new_zeros(half - n_rot)])
        self.register_buffer("freq", freq, persistent=False)
        self._build(max_seq_len)

    def _build(self, T: int) -> None:
        t = torch.arange(T, dtype=torch.float32, device=self.freq.device)
        theta = torch.outer(t, self.freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-3)
        if T > self.cos.size(0) or self.cos.device != x.device:
            self.freq = self.freq.to(x.device)
            self._build(max(T, self.cos.size(0)))
        cos = self.cos[None, :T, None, :]
        sin = self.sin[None, :T, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x2 * cos - x1 * sin
        return torch.cat((y1, y2), dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, c: HintonConfig):
        super().__init__()
        d, H = c.n_embd, c.n_head
        self.n_head, self.head_dim = H, d // H
        self.c_q = CastedLinear(d, d)
        self.c_k = CastedLinear(d, d)
        self.c_v = CastedLinear(d, d)
        self.c_proj = CastedLinear(d, d)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim, c.seq_len, float(c.rope_base), float(c.rope_frac))
        self.scale = float(c.attn_scale if c.attn_scale is not None else self.head_dim ** -0.5)

    def forward(self, x, ve, mask, causal=True):
        B, T, d = x.shape
        H, D = self.n_head, self.head_dim
        q = self.c_q(x).view(B, T, H, D)
        k = self.c_k(x).view(B, T, H, D)
        v = self.c_v(x).view(B, T, H, D)
        dt = q.dtype
        q, k = self.rotary(norm(q)).to(dt), self.rotary(norm(k)).to(dt)
        v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) if ve is not None else self.lambdas[0] * v
        if mask is None:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.to(dt).transpose(1, 2),
                                               is_causal=causal, scale=self.scale)
        else:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.to(dt).transpose(1, 2),
                                               attn_mask=mask, scale=self.scale)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, d))


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.c_fc = CastedLinear(d, hidden)
        self.c_proj = CastedLinear(hidden, d)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, c: HintonConfig, hidden: int):
        super().__init__()
        self.attn = CausalSelfAttention(c)
        self.mlp = MLP(c.n_embd, hidden)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.x0_inject = bool(c.x0_inject)

    def forward(self, x, ve, x0, mask, causal=True):
        if self.x0_inject:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), ve, mask, causal)
        x = x + self.mlp(norm(x))
        return x


def build_mask(attention_mask: torch.Tensor | None, T: int, device, causal: bool = True) -> torch.Tensor | None:
    """None when every token is real (pure causal, or full bidirectional, attention). Otherwise a boolean [B, 1, T, T]
    mask = the given mask (AND causal when causal): 2-D [B, T] (1 = real token; key padding) or 3-D [B, T, T] (query x
    key, as the finetuning collate builds with --causal). The diagonal is always kept so no row is empty (padded rows'
    outputs are unused)."""
    if attention_mask is None:
        return None
    am = attention_mask.to(device)
    if am.dim() == 2:
        if bool((am != 0).all()):
            return None
        keep = (am != 0)[:, None, None, :]
    elif am.dim() == 3:
        keep = (am != 0)[:, None, :, :]
    else:
        raise ValueError(f"attention_mask must be [B, T] or [B, T, T], got {tuple(am.shape)}")
    eye = torch.eye(T, dtype=torch.bool, device=device)
    if not causal:
        return keep | eye
    tri = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    return (keep & tri) | eye


class HintonModel(PreTrainedModel):
    """Trunk: input ids -> final RMS-normalised hidden states [B, T, d] (BaseModelOutput.last_hidden_state).
    Registered as AutoModel (used by the BabyLM finetuning classifier)."""
    config_class = HintonConfig
    base_model_prefix = "hinton"
    _no_split_modules = ["Block"]
    _keys_to_ignore_on_load_unexpected = [r"lm_head\.weight"]

    def __init__(self, config: HintonConfig):
        super().__init__(config)
        self._build_trunk(config)
        self.post_init()

    def _build_trunk(self, c: HintonConfig) -> None:
        d, V, L = c.n_embd, c.vocab_size, c.n_layer
        hidden = int(round(c.mlp_ratio * d))
        self.embed = nn.Embedding(V, d)
        n_ve = c.n_value_embeds if c.n_value_embeds is not None else max(1, round(L / 4))
        self.n_ve = max(0, min(int(n_ve), L // 2 if L > 1 else 1))
        self.value_embeds = nn.ModuleList(nn.Embedding(V, d) for _ in range(self.n_ve))
        self.blocks = nn.ModuleList(Block(c, hidden) for _ in range(L))
        self.unet = bool(c.unet)
        self.n_loops = int(getattr(c, "n_loops", 1) or 1)
        self.n_enc = L * self.n_loops // 2
        self.skip_weights = nn.Parameter(torch.ones(max(self.n_enc, 1)))
        self.embnorm = bool(c.embnorm)

    def _init_weights(self, module):  # weights always come from model.safetensors
        pass

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def hidden(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None,
               causal: bool = True) -> torch.Tensor:
        B, T = input_ids.shape
        mask = build_mask(attention_mask, T, input_ids.device, causal)
        L = self.config.n_layer
        Le = L * self.n_loops  # looped depth: effective layer i runs blocks[i % L] (babylm/models/looped.py)
        e = self.embed(input_ids)
        x = x0 = norm(e) if self.embnorm else e
        ves = [emb(input_ids) for emb in self.value_embeds]
        ve_at = [None] * Le
        for i, ve in enumerate(ves):
            ve_at[i] = ve
            ve_at[Le - self.n_ve + i] = ve
        skips = []
        for i in range(Le):
            if self.unet and i >= Le - self.n_enc and skips:
                x = x + self.skip_weights[i - (Le - self.n_enc)] * skips.pop()
            x = self.blocks[i % L](x, ve_at[i], x0, mask, causal)
            if self.unet and i < self.n_enc:
                skips.append(x)
        return norm(x)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None,
                **kwargs) -> BaseModelOutput:
        causal = not bool(getattr(self.config, "automodel_bidirectional", False))
        h = self.hidden(input_ids, attention_mask, causal)
        if getattr(self.config, "automodel_pool", "first") == "last":
            assert causal, "automodel_pool='last' needs a causal AutoModel"
            h = last_pool(h, attention_mask)
        return BaseModelOutput(last_hidden_state=h)


def last_pool(h: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Copy each row's last real token state into position 0 (as babylm/eval/adapters/modeling_lastpool.py)."""
    if attention_mask is not None and attention_mask.dim() == 3:  # (B, T, T) query x key: last row = key padding
        attention_mask = attention_mask[:, -1, :]
    if attention_mask is None:
        last = torch.full((h.size(0),), h.size(1) - 1, device=h.device, dtype=torch.long)
    else:
        m = attention_mask.long()
        assert bool((m[:, 1:] <= m[:, :-1]).all()), "automodel_pool='last' needs right padding (critic ref-0059)"
        last = m.sum(-1) - 1
    pooled = h[torch.arange(h.size(0), device=h.device), last]
    return torch.cat([pooled[:, None], h[:, 1:]], 1)


class HintonForCausalLM(HintonModel):
    """Trunk + untied lm head; logits are fp32 and softcapped (softcap * tanh(z / softcap))."""
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, config: HintonConfig):
        PreTrainedModel.__init__(self, config)
        self._build_trunk(config)
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.softcap = float(config.softcap or 0.0)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None,
                labels: torch.LongTensor | None = None, **kwargs) -> CausalLMOutput:
        h = self.hidden(input_ids, attention_mask, causal=True)
        z = self.lm_head(h).float()
        if self.softcap > 0:
            z = self.softcap * torch.tanh(z / self.softcap)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(z[:, :-1].reshape(-1, z.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=z)
