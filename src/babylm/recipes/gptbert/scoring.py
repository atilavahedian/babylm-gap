"""Per-token causal (autoregressive) scoring shared by train.py (dumps) and eval_bpw.py (any HF causal LM).

Protocol (identical in both, so numbers agree exactly for the same model):
  each document's token ids are cut into non-overlapping chunks of (ctx-1) tokens (tail chunks kept if >= 1 token);
  every chunk is scored as  [bos] + chunk  with the causal LM: token j of the chunk is predicted from bos + chunk[:j].
  Every token of every document is scored exactly once; context resets at chunk boundaries.
  bits per word = sum(NLL nats) / ln 2 / (#whitespace words of the scored text), words = tokens that start a word.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def make_chunks(docs, ctx, limit_tokens=0):
    """docs: list of 1-D int tensors. Returns list of (doc_idx, offset, length) in document order."""
    L = ctx - 1
    out, n = [], 0
    for d, x in enumerate(docs):
        for o in range(0, len(x), L):
            k = min(L, len(x) - o)
            out.append((d, o, k)); n += k
            if limit_tokens and n >= limit_tokens:
                return out
    return out


@torch.no_grad()
def score_chunks(logits_fn, docs, chunks, bos_id, device, batch_size=128):
    """Returns float32 numpy array of per-token NLL (nats), concatenated in `chunks` order.
    logits_fn(input_ids LongTensor[B, T] on device) -> logits [B, T, V]. Chunks are grouped by length (no padding)."""
    lens = np.array([c[2] for c in chunks])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out = np.zeros(int(lens.sum()), dtype=np.float32)
    order = np.argsort(-lens, kind="stable")
    i = 0
    while i < len(order):
        L = lens[order[i]]
        j = i
        while j < len(order) and j - i < batch_size and lens[order[j]] == L:
            j += 1
        idx = order[i:j]
        x = torch.stack([torch.cat([torch.tensor([bos_id]), docs[chunks[t][0]][chunks[t][1]:chunks[t][1] + L].long()]) for t in idx])
        x = x.to(device)
        logits = logits_fn(x[:, :-1]).float()                       # [B, L, V]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none").view(len(idx), L)
        nll = nll.cpu().numpy()
        for r, t in enumerate(idx):
            out[starts[t]:starts[t] + L] = nll[r]
        i = j
    return out


def chunk_token_arrays(docs, chunks, doc_table=None, newline_tab=None):
    """Static per-token arrays for a chunk list: token ids, doc ids, position in chunk, and (optional) file id/line no."""
    tok = np.concatenate([docs[d][o:o + L].numpy().astype(np.int32) for d, o, L in chunks]) if chunks else np.zeros(0, np.int32)
    doc = np.concatenate([np.full(L, d, np.int32) for d, o, L in chunks])
    pos = np.concatenate([np.arange(L, dtype=np.int16) for d, o, L in chunks])
    res = {"token_ids": tok, "doc_ids": doc, "pos_in_chunk": pos}
    if doc_table is not None and newline_tab is not None:
        line = np.zeros(len(tok), np.int32); fid = np.zeros(len(tok), np.int16)
        cache, k = {}, 0
        for d, o, L in chunks:
            if d not in cache:
                ids = docs[d].numpy().astype(np.int64)
                cache[d] = np.concatenate([[0], np.cumsum(newline_tab[ids])[:-1]])
            line[k:k + L] = doc_table[d][1] + cache[d][o:o + L]
            fid[k:k + L] = doc_table[d][0]
            k += L
        res["line_ids"], res["file_ids"] = line, fid
    return res


def bits_per_word(nll_nats, n_words):
    return float(np.sum(nll_nats, dtype=np.float64) / math.log(2) / n_words)


def train_side_logits_fn(model):
    """Causal logits from the train-side official Bert (identical to the published HF GPTBERTForCausalLM; see export_hf --verify)."""
    def fn(x):
        B, T = x.shape
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1).unsqueeze(0).expand(B, T, T)
        h = model.get_contextualized(x.t(), mask)
        return model.classifier.nonlinearity(h).transpose(0, 1)
    return fn
