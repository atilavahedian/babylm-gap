"""Per-token losses under THE registered DEV protocol (babylm/eval/dev_bpw.py, critic ref-0053), for readouts that
need more than a total: K-model ensembles (ceiling map C3) and overlap-removed nats/word.

    python -m babylm.moonshot.dev_bpw_dump MODEL_DIR SPLIT_DIR --out dump.npz [--check dev_bpw.json] [--twin]
                                            [--temps 1.05,1.1,...] [--ws collapse] [--line_join newline] [--tab_space ...]

Text, tokenisation, windows (ctx 512 / stride 256, BOS slot never scored), fp32 and no autocast are dev_bpw.py's own
(its load_text and windowing are reused, not copied). The dump holds, per file in sorted name order: `nats` (float32,
one per text token, in text order), `word_step` (uint8, 0 or 1: whether the token starts a new whitespace word, where a
token belongs to the word containing its last character, or to the word before it if it is whitespace only; the
cumulative sum within a file is the word index), `files`, `file_words` and `file_tokens`.
--check compares sum(nats) / words with a registered dev_bpw.json (same model, same flags) and fails above 1e-5
relative, so the dump can never drift from the scorer.
--twin: metric v2, the blank-line twin (babylm/eval/dev_bpw_twin.py; critic ref-0418): each file's blank-line-separated
documents are tokenized on their own and joined by one scored BOS, which belongs to the word before it; --check then
takes the twin's dev_bpw_twin.json.
--temps T1,T2,...: also `nats_temps` (float16, [tokens, len(temps)]) and `temps`: the same tokens' -log p at each
temperature, logsumexp(z / T) - z_y / T on the exported logits z (softcapped as exported; a PE export's log p_bar), from
the same forward passes; `nats` (T = 1) is computed exactly as without --temps (experimenter 07:45Z, temperature control).
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re

import numpy as np


def token_nats(model, ids: list[int], start_id: int, device, ctx: int, stride: int, bs: int = 16,
               temps: tuple[float, ...] = ()) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """-log p for every token of ids, scored exactly as dev_bpw.nll_of (same windows, same scored tails); with temps,
    also [n, len(temps)] -log p at each temperature."""
    import torch
    import torch.nn.functional as F
    n = len(ids)
    jobs = []  # (window start in ids, window tokens, number of scored tail tokens)
    k = 0
    while True:
        lo = k * stride
        if lo >= n and k > 0:
            break
        win = ids[lo: lo + ctx - 1]
        score = len(win) if k == 0 else max(0, len(win) - (ctx - 1 - stride))
        if score:
            jobs.append((lo, [start_id] + win, score))
        if lo + ctx - 1 >= n:
            break
        k += 1
    out = np.full(n, np.nan, dtype=np.float64)
    out_t = np.full((n, len(temps)), np.nan, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(jobs), bs):
            batch = jobs[i:i + bs]
            L = max(len(w) for _, w, _ in batch)
            inp = torch.full((len(batch), L), start_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (_, w, _) in enumerate(batch):
                inp[j, :len(w)] = torch.tensor(w)
                att[j, :len(w)] = 1
            inp, att = inp.to(device), att.to(device)
            o = model(input_ids=inp, attention_mask=att)
            logits = (o[0] if isinstance(o, tuple) else o["logits"]).float()
            lp = F.log_softmax(logits[:, :-1], -1).gather(-1, inp[:, 1:, None]).squeeze(-1).double().cpu().numpy()
            if temps:
                z, zy = logits[:, :-1], logits[:, :-1].gather(-1, inp[:, 1:, None]).squeeze(-1)
                nt = torch.stack([torch.logsumexp(z / T, -1) - zy / T for T in temps], -1).double().cpu().numpy()
            for j, (lo, w, s) in enumerate(batch):
                t = len(w) - 1  # predicted positions 1..len(w)-1 are text tokens lo..lo+t-1
                out[lo + t - s: lo + t] = -lp[j, t - s:t]
                if temps:
                    out_t[lo + t - s: lo + t] = nt[j, t - s:t]
    if np.isnan(out).any() or np.isnan(out_t).any():
        raise RuntimeError(f"{int(np.isnan(out).sum())} tokens were not scored")
    return (out, out_t) if temps else out


def token_words(text: str, offsets: list[tuple[int, int]]) -> np.ndarray:
    """Whitespace-word index (as in text.split()) of each token: the word containing its last character, else the
    word before it (whitespace-only tokens such as a newline attach to the end of their line)."""
    starts = [m.start() for m in re.finditer(r"\S+", text)]
    w = np.empty(len(offsets), dtype=np.int32)
    for i, (a, b) in enumerate(offsets):
        w[i] = max(0, bisect.bisect_right(starts, max(a, b - 1)) - 1)
    return w


def twin_encode(tok, text: str) -> tuple[list[int], np.ndarray]:
    """The blank-line twin's ids for one file (dev_bpw_twin.TwinTokenizer) and each token's whitespace-word index."""
    from babylm.eval.dev_bpw_twin import documents
    sep = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ids: list[int] = []
    words: list[int] = []
    base = 0
    for k, d in enumerate(documents(text)):
        if k:
            ids.append(sep)
            words.append(base - 1)
        enc = tok(d, add_special_tokens=False, return_offsets_mapping=True)
        nw = len(d.split())
        w = token_words(d, enc["offset_mapping"]) + base if nw else np.full(len(enc["input_ids"]), base - 1)
        ids += enc["input_ids"]
        words += w.tolist()
        base += nw
    if base != len(text.split()) or (words and min(words) < 0):
        raise RuntimeError("twin documents do not cover the file's words")
    return ids, np.asarray(words, dtype=np.int32)


def main(argv: list[str] | None = None) -> None:
    from babylm.eval import dev_bpw as D
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("split_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None, help="a registered dev_bpw.json of the same model and flags")
    ap.add_argument("--ws", choices=["raw", "collapse"], default="collapse")
    ap.add_argument("--line_join", choices=["newline", "newline_space"], default="newline")
    ap.add_argument("--tab_space", default="")
    ap.add_argument("--ctx", type=int, default=D.CTX)
    ap.add_argument("--twin", action="store_true", help="metric v2: the blank-line twin (needs --line_join newline)")
    ap.add_argument("--temps", default="", help="comma-separated temperatures: also save nats_temps / temps")
    a = ap.parse_args(argv)
    temps = tuple(float(x) for x in a.temps.split(",") if x)
    if any(T <= 0 for T in temps):
        raise SystemExit("--temps must be positive")
    assert "test" not in os.path.basename(os.path.normpath(a.split_dir)), "test split is off-limits"
    if a.twin and a.line_join != "newline":
        raise SystemExit("--twin matches the training stream only with --line_join newline")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True).float().to(device).eval()
    start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    files, nats, nats_t, words, fw, ft = [], [], [], [], [], []
    for f in sorted(os.listdir(a.split_dir)):
        text = D.load_text(os.path.join(a.split_dir, f), "full", a.ws, a.line_join,
                           tuple(x for x in a.tab_space.split(",") if x))
        if a.twin:
            ids, w = twin_encode(tok, text)
        else:
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, w = enc["input_ids"], token_words(text, enc["offset_mapping"])
        r = token_nats(model, ids, start_id, device, a.ctx, a.ctx // 2, temps=temps)
        nats.append(r[0] if temps else r)
        if temps:
            nats_t.append(r[1])
        words.append(w)
        files.append(f)
        fw.append(len(text.split()))
        ft.append(len(ids))
    tot = float(sum(x.sum() for x in nats)) / sum(fw)
    if a.check:
        ref = json.load(open(a.check))
        if abs(tot - ref["nats_per_word"]) > 1e-5 * ref["nats_per_word"] or ref["words"] != sum(fw):
            raise SystemExit(f"dump {tot:.6f} nats/word vs registered {ref['nats_per_word']:.6f}: mismatch")
    steps = []
    for w, n_w in zip(words, fw):
        d = np.diff(w, prepend=0)
        if d.min() < 0 or d.max() > 1 or (len(w) and (w[0] != 0 or w[-1] != n_w - 1)):
            raise RuntimeError("token-to-word map is not a 0/1 step sequence")
        steps.append(d.astype(np.uint8))
    extra = {"nats_temps": np.concatenate(nats_t).astype(np.float16), "temps": np.array(temps)} if temps else {}
    np.savez_compressed(a.out, nats=np.concatenate(nats).astype(np.float32), word_step=np.concatenate(steps), **extra,
                        files=np.array(files),
                        file_words=np.array(fw), file_tokens=np.array(ft), model_dir=a.model_dir,
                        protocol=json.dumps({"ws": a.ws, "line_join": a.line_join, "tab_space": a.tab_space,
                                             "ctx": a.ctx, "stride": a.ctx // 2, "twin": a.twin}))
    rep = {"out": a.out, "nats_per_word": tot, "words": sum(fw), "tokens": sum(ft)}
    if temps:
        rep["nats_per_word_at"] = {f"{T:g}": float(sum(x[:, i].sum() for x in nats_t)) / sum(fw)
                                   for i, T in enumerate(temps)}
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
