"""Per-token MLM pseudo-losses for an AMLM (DebertaV2ForMaskedLM) checkpoint, for Phase-2 slicing.

AMLM cannot score causally, so this is NOT causal loss and is not comparable to it (critic ruling ref-0261).
Method: lines are tokenized as in training ([CLS] ... [SEP] per line), concatenated per file and cut into
windows of SEQ tokens. In each window every position is assigned to one of K=7 partitions by a fixed seeded
permutation; K forward passes mask one partition each with [MASK] (no 80/10/10), so every token is scored exactly
once with ~1/7 = 14% of its window masked, close to AMLM's final 15% mask rate.
Outputs OUT.{loss.f16.npy, tok.i32.npy, doc.i32.npy, file.u8.npy, json}: doc = global non-blank line index.
Speed: the LM head runs only on the masked positions (it is position-wise, so losses equal the full forward's), and
--cache stores the tokenized windows so later checkpoints skip tokenization.
usage: python per_token.py CKPT_DIR TOKENIZER_SPM OUT_PREFIX FILE [FILE ...] [--seq 256] [--max_words N] [--bs 256] [--cache F]"""
import argparse, json, hashlib, numpy as np, torch
from transformers import AutoModelForMaskedLM, DebertaV2Tokenizer
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("tok"); ap.add_argument("out"); ap.add_argument("files", nargs="+")
ap.add_argument("--seq", type=int, default=256); ap.add_argument("--k", type=int, default=7); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--max_words", type=int, default=0); ap.add_argument("--revision", default=None); ap.add_argument("--cache", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = DebertaV2Tokenizer.from_pretrained(a.tok)
model = AutoModelForMaskedLM.from_pretrained(a.ckpt, revision=a.revision).to(dev).eval()
import os
if a.cache and os.path.exists(a.cache):
    c = np.load(a.cache); ids, docs, fidx, words = c["ids"], c["docs"], c["fidx"], int(c["words"])
else:
    ids, docs, fidx, words, d = [], [], [], 0, 0
    for fi, f in enumerate(a.files):
        if a.max_words and words >= a.max_words: break
        for line in open(f, encoding="utf-8"):
            if not line.strip(): continue
            t = tok.encode(line.rstrip("\n")); ids += t; docs += [d] * len(t); fidx += [fi] * len(t); d += 1
            words += len(line.split())
            if a.max_words and words >= a.max_words: break
    if a.cache: np.savez(a.cache, ids=np.array(ids, np.int32), docs=np.array(docs, np.int32), fidx=np.array(fidx, np.uint8), words=words)
n = len(ids) // a.seq * a.seq  # drop the remainder, as group_texts does in training
X = torch.as_tensor(np.asarray(ids[:n], dtype=np.int64)).view(-1, a.seq)
g = torch.Generator().manual_seed(0); part = torch.randperm(a.seq, generator=g) % a.k  # fixed partition of positions
loss = torch.empty_like(X, dtype=torch.float32)
with torch.no_grad():
    for i in range(0, len(X), a.bs):
        x = X[i:i + a.bs].to(dev)
        for p in range(a.k):
            m = (part.to(dev) == p).expand_as(x)
            xin = x.masked_fill(m, tok.mask_token_id)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                h = model.deberta(input_ids=xin, attention_mask=torch.ones_like(x))[0][m]  # masked positions only
                logits = (model.cls(h) if model.legacy else model.lm_predictions(h, model.deberta.embeddings.word_embeddings)).float()
            loss[i:i + a.bs][m.cpu()] = torch.nn.functional.cross_entropy(logits, x[m], reduction="none").cpu()
L = loss.flatten().numpy()
np.save(a.out + ".loss.f16.npy", L.astype(np.float16)); np.save(a.out + ".tok.i32.npy", X.flatten().numpy().astype(np.int32))
np.save(a.out + ".doc.i32.npy", np.array(docs[:n], np.int32)); np.save(a.out + ".file.u8.npy", np.array(fidx[:n], np.uint8))
meta = dict(ckpt=a.ckpt, revision=a.revision, files=a.files, files_sha256=[hashlib.sha256(open(f, "rb").read()).hexdigest() for f in a.files],
            tokens=int(n), words_read=words, k=a.k, seq=a.seq, mean_loss_nats_per_token=float(L.mean()),
            nats_per_word=float(L.sum() / words) if not a.max_words else None, kind="MLM pseudo-loss, not causal")
json.dump(meta, open(a.out + ".json", "w"), indent=1); print(json.dumps(meta))
