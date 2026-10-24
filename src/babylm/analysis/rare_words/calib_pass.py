"""Second scoring pass: per target token, the model's probability mass per token-frequency band (train_10M counts
of the model's own tokens) and on word-initial tokens. Same chunks/windows as score_rare.py, so arrays align with
<model>.npz tok_* arrays. Writes <model>.calib.npz: tok_nll, tok_id, tok_band, tok_init, band_mass [N,K], init_mass."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "instr"))
sys.path.insert(0, str(HERE))
import rare_words as RW  # noqa: E402
from babylm.instruments import score as SC  # noqa: E402

if "--trust-model-code" in sys.argv:
    sys.argv.remove("--trust-model-code")
    SC.TRUST_MODEL_CODE = True

EDGES = [1, 8, 64, 512, 4096, 32768]  # token count bands: 0, 1-7, 8-63, 64-511, 512-4095, 4096-32767, 32768+
K = len(EDGES) + 1


def token_counts(tok, raw_root: Path, V: int, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    counts = np.zeros(V, dtype=np.int64)
    for s in RW.SOURCES:
        lines = []
        for ln in (raw_root / "train_10M" / f"{s}.train").read_text(encoding="utf-8").split("\n"):
            if RW.is_header(ln):
                continue
            c = " ".join(ln.split())
            if c:
                lines.append(c)
        for i in range(0, len(lines), 2000):
            enc = tok(lines[i:i + 2000], add_special_tokens=False)["input_ids"]
            counts += np.bincount(np.concatenate([np.asarray(e, dtype=np.int64) for e in enc]), minlength=V)
    np.save(cache, counts)
    return counts


def main():
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--models", default="babylm-baseline-10m-gpt-bert-causal-focus,babylm-baseline-100m-gpt-bert-causal-focus")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = [json.loads(l) for l in (args.out / "sample.jsonl").read_text().splitlines()]
    for name in args.models.split(","):
        t0 = time.time()
        model, tok, family, n_params = SC.load_model(args.ckpt_dir / name, torch.float32)
        V = model.config.vocab_size if hasattr(model, "config") and hasattr(model.config, "vocab_size") else len(tok)
        V = int(V)
        counts = token_counts(tok, args.raw_root, V, args.out / f"tokcounts_{name}.npy")
        band = np.searchsorted(np.array(EDGES), counts, side="right")  # 0..K-1
        vocab = tok.get_vocab()
        init = np.zeros(V, dtype=bool)
        for s, i in vocab.items():
            if i < V and s.startswith("Ġ"):
                init[i] = True
        Bm = torch.zeros(V, K)
        Bm[torch.arange(V), torch.tensor(band)] = 1.0
        Iv = torch.tensor(init, dtype=torch.float32)
        print(f"{name}: loaded in {time.time() - t0:.0f}s; token bands sizes {np.bincount(band, minlength=K).tolist()}; init tokens {init.sum()}", flush=True)
        chunks = SC.tokenize_chunks(tok, rows, "newline", (), True)
        jobs = []
        for ci, (ids, _, _) in enumerate(chunks):
            for s, e, f in SC.windows_for(ids, 512, 256):
                jobs.append((ci, s, e, f))
        jobs.sort(key=lambda j: j[2] - j[1])
        n_tok = [len(ids) - 1 for ids, _, _ in chunks]
        r_nll = [np.zeros(n, np.float32) for n in n_tok]
        r_bm = [np.zeros((n, K), np.float32) for n in n_tok]
        r_im = [np.zeros(n, np.float32) for n in n_tok]
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        t0 = time.time()
        i = 0
        with torch.inference_mode():
            while i < len(jobs):
                j = i
                while j < len(jobs) and (j - i + 1) * (jobs[j][2] - jobs[j][1]) <= 8192:
                    j += 1
                j = max(j, i + 1)
                batch = jobs[i:j]
                L = max(e - s for _, s, e, _ in batch)
                x = torch.full((len(batch), L), pad, dtype=torch.long)
                for bi, (ci, s, e, _) in enumerate(batch):
                    x[bi, : e - s] = torch.tensor(chunks[ci][0][s:e])
                out = model(input_ids=x)
                logits = out.logits if hasattr(out, "logits") else out[0]
                for bi, (ci, s, e, f) in enumerate(batch):
                    ids = chunks[ci][0]
                    p0, p1 = f - s - 1, e - s - 1
                    lp = F.log_softmax(logits[bi, p0:p1].float(), dim=-1)
                    tgt = torch.tensor(ids[f:e])
                    sl = slice(f - 1, e - 1)
                    r_nll[ci][sl] = (-lp.gather(1, tgt[:, None])[:, 0]).numpy()
                    p = lp.exp()
                    r_bm[ci][sl] = (p @ Bm).numpy()
                    r_im[ci][sl] = (p @ Iv).numpy()
                i = j
        tok_ids = np.concatenate([np.asarray(ids[1:], dtype=np.int64) for ids, _, _ in chunks])
        word_offsets = np.cumsum([0] + [nw for _, _, nw in chunks])
        tok_word = np.concatenate([wd + word_offsets[ci] for ci, (_, wd, _) in enumerate(chunks)]).astype(np.int32)
        # word-initial by position (first token of its whitespace word): exact for any tokenizer; the id-based flag
        # (space-prefixed token) misses line-initial words for tokenizers without a prepended space
        tok_init_pos = np.ones(len(tok_word), bool)
        tok_init_pos[1:] = tok_word[1:] != tok_word[:-1]
        np.savez_compressed(args.out / f"{name}.calib.npz", tok_nll=np.concatenate(r_nll), tok_id=tok_ids, tok_band=band[tok_ids].astype(np.int8),
                            tok_init=tok_init_pos, tok_init_by_id=init[tok_ids], band_mass=np.concatenate(r_bm), init_mass=np.concatenate(r_im),
                            tok_word=tok_word, edges=np.array(EDGES), token_counts=counts)
        print(f"{name}: calib pass {len(tok_ids):,} tokens in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
