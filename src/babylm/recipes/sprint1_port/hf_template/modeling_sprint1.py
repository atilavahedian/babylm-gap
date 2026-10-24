"""HF wrapper (trust_remote_code) for the sprint-1 baseline GPT trained by babylm/recipes/sprint1_port/train.py.

The network below is sprint1/models/baseline.py (origin/sprint-1, blob ref-0494) with exactly two edits, neither of
which changes the function computed:
  1. ModelConfig is replaced by the HF config (same field names: vocab_size, seq_len, n_layer, n_head, n_embd,
     mlp_ratio, tie_embeddings, extra);
  2. Rotary keeps no registered buffers (HF may build modules on the meta device); cos/sin are computed with the
     same float32 ops on the CPU and moved to the input's device, which is what the training module does
     (buffers built on the CPU at init, then .to(device)).
Forward: logits = softcap * tanh(lm_head(norm(h)) / softcap) in float32, shape [B, T, vocab_size=50304].
attention_mask: the network is plain causal attention with no padding mask, so only RIGHT padding (or none) gives
correct logits on the real tokens; any other mask raises. Position 0 is the first input token (the official causal
evaluation puts BOS = <|endoftext|> there, the training document separator).
config.autocast_bf16 = True runs the CUDA forward under bf16 autocast, the training numerics; default False =
float32, the exact function the trained weights define.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput


class Sprint1Config(PretrainedConfig):
    model_type = "sprint1_gpt"

    def __init__(self, vocab_size=50304, seq_len=1024, n_layer=6, n_head=6, n_embd=384, mlp_ratio=4.0,
                 tie_embeddings=False, extra=None, autocast_bf16=False, bos_token_id=50256, eos_token_id=50256,
                 pad_token_id=50256, **kwargs):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.mlp_ratio = mlp_ratio
        self.tie_embeddings = tie_embeddings
        self.extra = dict(extra or {})
        self.autocast_bf16 = autocast_bf16
        self.hidden_size = n_embd  # read by the official finetuning head
        # HF ties lm_head to the input embedding unless told otherwise; sprint-1 is untied
        kwargs["tie_word_embeddings"] = bool(tie_embeddings)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# sprint1/models/baseline.py (verbatim apart from the two edits named above)
# ---------------------------------------------------------------------------------------------------------------
def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__(in_features, out_features, bias=False)
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            bound = math.sqrt(3.0) * 0.5 * in_features ** -0.5
            nn.init.uniform_(self.weight, -bound, bound)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 1024.0, frac: float = 0.25):
        super().__init__()
        self.head_dim, self.base, self.frac, self.max_seq_len = head_dim, base, frac, max_seq_len
        self._cache: dict = {}

    def _tables(self, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        L = max(T, self.max_seq_len)
        key = (L, str(device))
        if key not in self._cache:
            n_rot = max(1, int(self.head_dim * self.frac))
            half = self.head_dim // 2
            n_rot = min(n_rot, half)
            freq = (1.0 / self.base) ** torch.linspace(0, 1, steps=n_rot, dtype=torch.float32, device="cpu")
            freq = torch.cat([freq, freq.new_zeros(half - n_rot)])
            t = torch.arange(L, dtype=torch.float32, device="cpu")
            theta = torch.outer(t, freq)
            self._cache[key] = (theta.cos().to(device), theta.sin().to(device))
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, H, D]
        T = x.size(-3)
        cos_t, sin_t = self._tables(T, x.device)
        cos = cos_t[None, :T, None, :]
        sin = sin_t[None, :T, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x2 * cos - x1 * sin
        return torch.cat((y1, y2), dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, H = cfg.n_embd, cfg.n_head
        self.n_head, self.head_dim = H, d // H
        ex = cfg.extra
        self.c_q = CastedLinear(d, d)
        self.c_k = CastedLinear(d, d)
        self.c_v = CastedLinear(d, d)
        self.c_proj = CastedLinear(d, d, zero_init=True)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim, cfg.seq_len, float(ex.get("rope_base", 1024.0)),
                             float(ex.get("rope_frac", 0.25)))
        self.scale = float(ex.get("attn_scale", self.head_dim ** -0.5))

    def forward(self, x: torch.Tensor, ve):
        B, T, d = x.shape
        H, D = self.n_head, self.head_dim
        q = self.c_q(x).view(B, T, H, D)
        k = self.c_k(x).view(B, T, H, D)
        v = self.c_v(x).view(B, T, H, D)
        dt = q.dtype
        q, k = self.rotary(norm(q)).to(dt), self.rotary(norm(k)).to(dt)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.to(dt).transpose(1, 2),
                                           is_causal=True, scale=self.scale)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, d))


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.c_fc = CastedLinear(d, hidden)
        self.c_proj = CastedLinear(hidden, d, zero_init=True)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, cfg, hidden: int):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg.n_embd, hidden)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.x0_inject = bool(cfg.extra.get("x0_inject", True))

    def forward(self, x, ve, x0):
        if self.x0_inject:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), ve)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ex = cfg.extra
        d, V, L = cfg.n_embd, cfg.vocab_size, cfg.n_layer
        hidden = int(round(cfg.mlp_ratio * d))
        self.embed = nn.Embedding(V, d)
        n_ve = int(ex.get("n_value_embeds", max(1, round(L / 4))))
        self.n_ve = max(0, min(n_ve, L // 2 if L > 1 else 1))
        self.value_embeds = nn.ModuleList(nn.Embedding(V, d) for _ in range(self.n_ve))
        self.blocks = nn.ModuleList(Block(cfg, hidden) for _ in range(L))
        self.lm_head = CastedLinear(d, V)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=float(ex.get("head_init_std", 0.005)))
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.unet = bool(ex.get("unet", True))
        self.n_enc = L // 2
        self.skip_weights = nn.Parameter(torch.ones(max(self.n_enc, 1)))
        self.softcap = float(ex.get("softcap", 15.0))
        self.embnorm = bool(ex.get("embnorm", True))

    def hidden(self, idx):
        L = self.cfg.n_layer
        e = self.embed(idx)
        x = x0 = norm(e) if self.embnorm else e
        ves = [emb(idx) for emb in self.value_embeds]
        ve_at = [None] * L
        for i, ve in enumerate(ves):
            ve_at[i] = ve
            ve_at[L - self.n_ve + i] = ve
        skips = []
        for i, block in enumerate(self.blocks):
            if self.unet and i >= L - self.n_enc and skips:
                x = x + self.skip_weights[i - (L - self.n_enc)] * skips.pop()
            x = block(x, ve_at[i], x0)
            if self.unet and i < self.n_enc:
                skips.append(x)
        return norm(x)

    def logits(self, h):
        z = self.lm_head(h).float()
        if self.softcap > 0:
            z = self.softcap * torch.tanh(z / self.softcap)
        return z


# ---------------------------------------------------------------------------------------------------------------
def _check_right_padding(attention_mask):
    if attention_mask is not None and attention_mask.ndim == 2:
        am = attention_mask.to(torch.long)
        # right padding only: every row is 1...1 0...0 (a 0 followed by a 1 would change real positions)
        if bool(((am[:, 1:] - am[:, :-1]) > 0).any()):
            raise ValueError("Sprint1 models support right padding only (no left/inner padding)")


def _ctx(config, input_ids):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if config.autocast_bf16 and input_ids.is_cuda else torch.autocast("cpu", enabled=False))


class Sprint1Model(PreTrainedModel):
    """AutoModel (official finetuning): last_hidden_state = the final RMS-normed hidden state [B, T, n_embd]."""
    config_class = Sprint1Config
    base_model_prefix = "model"
    _no_split_modules = ["Block"]
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]

    def __init__(self, config: Sprint1Config):
        super().__init__(config)
        self.model = GPT(config)
        self.post_init()

    def _init_weights(self, module):
        pass

    def forward(self, input_ids=None, attention_mask=None, return_dict=None, **kwargs):
        _check_right_padding(attention_mask)
        with _ctx(self.config, input_ids):
            h = self.model.hidden(input_ids)
        if return_dict is False:
            return (h,)
        return BaseModelOutput(last_hidden_state=h)


class Sprint1ForCausalLM(PreTrainedModel):
    config_class = Sprint1Config
    base_model_prefix = "model"
    _no_split_modules = ["Block"]

    def __init__(self, config: Sprint1Config):
        super().__init__(config)
        self.model = GPT(config)
        self.post_init()

    def _init_weights(self, module):  # weights come from the checkpoint; GPT.__init__ holds the training init
        pass

    def get_input_embeddings(self):
        return self.model.embed

    def get_output_embeddings(self):
        return self.model.lm_head

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=None, **kwargs):
        _check_right_padding(attention_mask)
        with _ctx(self.config, input_ids):
            logits = self.model.logits(self.model.hidden(input_ids))
        loss = None
        if labels is not None:
            shift = logits[:, :-1].reshape(-1, logits.size(-1))
            tgt = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift, tgt, ignore_index=-100)
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return CausalLMOutput(loss=loss, logits=logits)
