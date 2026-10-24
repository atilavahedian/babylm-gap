"""Standalone Hugging Face copy of babylm/models/particles.py, the shared-trunk particle ensemble (exported by
babylm/harness/export_hf.py for checkpoints with pe_k > 1). Blocks, masks and pooling come from modeling_hinton.py.

config: pe_k (K particles), pe_shared (Ls shared blocks), pe_tied (every particle re-runs particle 0's private blocks)
on top of the baseline keys. Particle 0 uses blocks[Ls:] and skip_weights; particle k >= 1 uses pblocks[k-1] and
pskip[k-1] (harness parameter names). HintonPEForCausalLM.forward() returns logits = log p_bar (fp32): the log-mean-exp
over particles of their softcapped log-softmaxes, already normalised (its log_softmax is itself). HintonPEModel
(AutoModel) returns the mean of the K final normalised particle states (one shared unembedding reads them all);
automodel_pool works as in modeling_hinton.py. Always causal.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .modeling_hinton import Block, CastedLinear, HintonModel, build_mask, norm


class HintonPEModel(HintonModel):
    def _build_trunk(self, c) -> None:
        super()._build_trunk(c)
        L, K, Ls = c.n_layer, int(c.pe_k), int(c.pe_shared)
        if K < 2 or not 1 <= Ls <= L - self.n_enc:
            raise ValueError(f"pe_k {K} / pe_shared {Ls} unsupported (n_layer {L})")
        hidden = int(round(c.mlp_ratio * c.n_embd))
        n_priv = 0 if getattr(c, "pe_tied", False) else K - 1
        self.pblocks = nn.ModuleList(nn.ModuleList(Block(c, hidden) for _ in range(L - Ls)) for _ in range(n_priv))
        self.pskip = nn.ParameterList(nn.Parameter(torch.ones(max(self.n_enc, 1))) for _ in range(n_priv))

    def stacks(self):
        own = (list(self.blocks[int(self.config.pe_shared):]), self.skip_weights)
        if getattr(self.config, "pe_tied", False):
            return [own] * int(self.config.pe_k)
        return [own] + [(list(b), w) for b, w in zip(self.pblocks, self.pskip)]

    def particle_hiddens(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None,
                         causal: bool = True) -> torch.Tensor:
        B, T = input_ids.shape
        mask = build_mask(attention_mask, T, input_ids.device, causal)
        L, Ls = self.config.n_layer, int(self.config.pe_shared)
        e = self.embed(input_ids)
        x = x0 = norm(e) if self.embnorm else e
        ve_at = [None] * L
        for i, emb in enumerate(self.value_embeds):
            ve_at[i] = ve_at[L - self.n_ve + i] = emb(input_ids)
        skips = []
        for i in range(Ls):
            x = self.blocks[i](x, ve_at[i], x0, mask, causal)
            if self.unet and i < self.n_enc:
                skips.append(x)
        outs = []
        for blocks, sw in self.stacks():
            xk, sk = x, list(skips)
            for j, block in enumerate(blocks):
                i = Ls + j
                if self.unet and i >= L - self.n_enc and sk:
                    xk = xk + sw[i - (L - self.n_enc)] * sk.pop()
                xk = block(xk, ve_at[i], x0, mask, causal)
                if self.unet and i < self.n_enc:
                    sk.append(xk)
            outs.append(norm(xk))
        return torch.stack(outs)

    def hidden(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None,
               causal: bool = True) -> torch.Tensor:
        return self.particle_hiddens(input_ids, attention_mask, causal).mean(0)


class HintonPEForCausalLM(HintonPEModel):
    """Trunk + K particles + one lm head; logits = log p_bar (fp32)."""
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self._build_trunk(config)
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.softcap = float(config.softcap or 0.0)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None,
                labels: torch.LongTensor | None = None, **kwargs) -> CausalLMOutput:
        lps = []
        for h in self.particle_hiddens(input_ids, attention_mask, causal=True):
            z = self.lm_head(h).float()
            if self.softcap > 0:
                z = self.softcap * torch.tanh(z / self.softcap)
            lps.append(F.log_softmax(z, -1))
        z = torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(z[:, :-1].reshape(-1, z.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=z)
