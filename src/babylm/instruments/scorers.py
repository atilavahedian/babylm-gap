"""Why do the registered DEV number (babylm/eval/dev_bpw.py on the HF export) and the training harness's own full-DEV
eval differ for a model, and which part of the difference moves from seed to seed? Scores whole documents drawn at
random from every dev file (--frac of its words) under six protocols, each changing one element of the registered
protocol towards the harness's:
  registered             dev_bpw: each file one document with its blank lines kept as text, 512-token windows
                         (stride 256) with a BOS at every window start, fp32
  blank lines dropped    the documents joined by single newlines (the training text has no blank lines)
  BOS between documents  the harness's token stream: every document preceded by BOS
  no BOS at window start the same windows without the added BOS (the stream's own BOS tokens stay)
  1024-token blocks      the harness's windows: non-overlapping 1024-token blocks, so context 0 to 1023 tokens
  harness (bf16)         the same in bf16 autocast (the harness evaluates in bf16 on CUDA; here CPU autocast)
Consecutive differences are the parts of registered minus harness; words are the same in every protocol. Also the
nats the registered protocol spends on blank-line newline tokens and on the token after a blank line, and the nats
the harness protocol spends on its BOS tokens (document ends) and the token after each.

  python -m babylm.instruments.scorers --tag base_r1 --data-root ~/babylm_data name=hf_dir ...
Writes results/babylm/<tag>/scorers_parts.csv (per model, source and protocol), scorers_tokens.csv (the separator
token categories) and scorers_parts.png.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from babylm.instruments import slices as S

SWB_PIECE_LINES = 2000  # babylm/data/corpus.py: switchboard documents are 2000-line pieces
CTX, STRIDE, T = 512, 256, 1024
PROTOCOLS = ["registered", "blank lines dropped", "BOS between documents", "no BOS at window start",
             "1024-token blocks", "harness (bf16)"]
PARTS = ["blank lines kept as text", "file as one document (no BOS between documents)", "BOS at every window start",
         "512/256 windows vs 1024-token blocks", "fp32 vs bf16"]


def doc_ranges(lines: list[str], source: str) -> list[tuple[int, int]]:
    """[start, end) line ranges of the blank-line-separated documents (babylm/data/corpus.py)."""
    out, start = [], None
    for i, line in enumerate(lines):
        if line.strip() == "":
            if start is not None:
                out.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        out.append((start, len(lines)))
    if source == "switchboard":
        cut = []
        for s, e in out:
            b = s
            while b < e:
                nxt = min(e, (b // SWB_PIECE_LINES + 1) * SWB_PIECE_LINES)
                cut.append((b, nxt))
                b = nxt
        out = cut
    return out


def texts(path: Path, frac: float, seed: int = 0) -> tuple[str, list[str], int, int]:
    """Whole documents of a dev file drawn at random (seed) until they hold frac of its words, kept in file order:
    (registered text: the documents with the blank lines before each as in the file, the harness's document texts,
    words drawn, words in the file)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    norm = [" ".join(x.split()) for x in lines]
    rng = doc_ranges(lines, path.stem)
    words = [sum(len(x.split()) for x in norm[a:b]) for a, b in rng]
    pick, n = [], 0
    for i in np.random.default_rng(seed).permutation(len(rng)):
        pick.append(int(i))
        n += words[i]
        if n >= frac * sum(words):
            break
    pick.sort()
    seg = ["\n".join(norm[(rng[i - 1][1] if i > 0 and k > 0 else rng[i][0]):rng[i][1]]) for k, i in enumerate(pick)]
    return "\n".join(seg), ["\n".join(norm[a:b]) for a, b in (rng[i] for i in pick)], n, sum(words)


def sliding_jobs(ids: list[int], bos: int | None) -> list[tuple[list[int], int, int]]:
    """(window, scored targets, position of the first scored target in ids). With bos: dev_bpw.nll_of (windows of CTX
    tokens including the added BOS, stride STRIDE, every token of ids scored). Without: windows of CTX tokens of ids
    itself, every token but the first scored."""
    jobs, n = [], len(ids)
    if bos is not None:
        k = 0
        while True:
            lo = k * STRIDE
            if lo >= n and k > 0:
                break
            win = ids[lo: lo + CTX - 1]
            score = len(win) if k == 0 else max(0, len(win) - (CTX - 1 - STRIDE))
            if score:
                jobs.append(([bos] + win, score, lo + len(win) - score))
            if lo + CTX - 1 >= n:
                break
            k += 1
        return jobs
    k = 0
    while True:
        lo = k * STRIDE
        win = ids[lo: lo + CTX]
        score = len(win) - 1 if k == 0 else max(0, len(win) - (CTX - STRIDE))
        if score:
            jobs.append((win, score, lo + len(win) - score))
        if lo + CTX >= n:
            break
        k += 1
    return jobs


def block_jobs(ids: list[int]) -> list[tuple[list[int], int, int]]:
    """train.evaluate_stream: non-overlapping windows of T inputs; every token but the first scored once."""
    return [(ids[lo: lo + T + 1], len(ids[lo: lo + T + 1]) - 1, lo + 1) for lo in range(0, len(ids) - 1, T)]


def run_jobs(model, jobs, n: int, autocast, bs: int = 8) -> np.ndarray:
    """Per-position nats (NaN where not scored) of ids of length n."""
    import torch
    import torch.nn.functional as Fn

    out = np.full(n, np.nan)
    jobs = sorted(jobs, key=lambda j: len(j[0]))
    with torch.inference_mode():
        i = 0
        while i < len(jobs):
            L = len(jobs[i][0])
            group = [j for j in jobs[i:i + bs] if len(j[0]) == L]
            i += len(group)
            x = torch.tensor([w for w, _, _ in group])
            with autocast():
                lg = model(input_ids=x[:, :-1]).logits
            lp = Fn.log_softmax(lg.float(), -1).gather(-1, x[:, 1:, None])[..., 0].double().numpy()
            for r, (_, s, p0) in enumerate(group):
                out[p0:p0 + s] = -lp[r, L - 1 - s:]
    return out


def main() -> None:
    import contextlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    from babylm.instruments import score as SC

    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path of a harness HF export")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.02, help="share of each dev file's words (whole documents)")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((args.data_root / "dev" / "dev").glob("*.dev"))
    data = [(f.stem, *texts(f, args.frac)) for f in files]
    fp32 = contextlib.nullcontext
    bf16 = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
    rows, cats = [], []
    for spec in args.models:
        name, path = spec.split("=", 1)
        model, tok, _, _ = SC.load_model(Path(path), torch.float32)
        model.eval()
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        assert len(nl) == 1, nl
        nl = nl[0]
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        for src, reg_text, docs, words, file_words in data:
            reg, joined = enc(reg_text), enc("\n".join(docs))
            stream = [t for d in docs for t in [bos] + enc(d)]
            ids = {"registered": reg, "blank lines dropped": joined, "BOS between documents": stream[1:],
                   "no BOS at window start": stream, "1024-token blocks": stream, "harness (bf16)": stream}
            jobs = {"registered": sliding_jobs(reg, bos), "blank lines dropped": sliding_jobs(joined, bos),
                    # the window's own BOS stands in for the first document's
                    "BOS between documents": sliding_jobs(stream[1:], bos),
                    "no BOS at window start": sliding_jobs(stream, None),
                    "1024-token blocks": block_jobs(stream), "harness (bf16)": block_jobs(stream)}
            per, secs = {}, {}
            for p in PROTOCOLS:
                t0 = time.time()
                per[p] = run_jobs(model, jobs[p], len(ids[p]), bf16 if p == "harness (bf16)" else fp32)
                secs[p] = round(time.time() - t0)
                first = 0 if p in ("registered", "blank lines dropped", "BOS between documents") else 1
                if np.isnan(per[p][first:]).any():
                    raise RuntimeError(f"{name} {src} {p}: {int(np.isnan(per[p][first:]).sum())} tokens not scored")
                rows.append({"model": name, "path": path, "source": src, "protocol": p, "words": words,
                             "file_words": file_words, "tokens": int((~np.isnan(per[p])).sum()),
                             "nats": float(np.nansum(per[p]))})
            # separator tokens: blank-line newlines (a newline after a newline) and the token after a blank line in
            # the registered text; BOS tokens (document ends) and the token after each in the harness stream
            r = np.asarray(reg)
            blank = np.r_[False, (r[1:] == nl) & (r[:-1] == nl)]
            after_blank = np.r_[False, (r[1:] != nl) & blank[:-1]]
            h = np.asarray(stream)
            bos_t = np.r_[False, h[1:] == bos]
            after_bos = np.r_[False, bos_t[:-1]]
            for p, m, what in (("registered", blank, "blank-line newline"),
                               ("registered", after_blank, "token after a blank line"),
                               ("1024-token blocks", bos_t, "BOS (document end)"),
                               ("1024-token blocks", after_bos, "token after BOS")):
                v = per[p][m]
                cats.append({"model": name, "source": src, "protocol": p, "category": what, "count": int(m.sum()),
                             "nats": float(np.nansum(v)), "words": words, "file_words": file_words})
            print(name, src, words, {p: round(rows[k - len(PROTOCOLS)]["nats"] / words, 4) for k, p in enumerate(PROTOCOLS)},
                  secs, flush=True)
        del model
    t = pd.DataFrame(rows)
    t.to_csv(out / "scorers_parts.csv", index=False)
    c = pd.DataFrame(cats)
    c.to_csv(out / "scorers_tokens.csv", index=False)
    # pooled over sources with each dev file's share of the split's words (the draw is whole documents)
    t["npw"] = t.nats / t.words
    t["share"] = t.file_words / t.groupby(["model", "protocol"]).file_words.transform("sum")
    npw = (t.npw * t.share).groupby([t.model, t.protocol], sort=False).sum().unstack()[PROTOCOLS]
    parts = pd.DataFrame({PARTS[k]: npw[PROTOCOLS[k]] - npw[PROTOCOLS[k + 1]] for k in range(len(PARTS))})
    parts["registered minus harness"] = npw["registered"] - npw["harness (bf16)"]
    c["npw"] = c.nats / c.words
    c["share"] = c.file_words / c.groupby(["model", "category"]).file_words.transform("sum")
    cw = (c.npw * c.share).groupby([c.model, c.category], sort=False).sum().unstack()
    g = c.groupby(["model", "category"], sort=False)[["nats", "count"]].sum()
    cm = (g.nats / g["count"]).unstack()
    pd.set_option("display.width", 250)
    print(npw.round(4).to_string())
    print(parts.round(4).to_string())
    print("separator tokens, nats/word (pooled to dev word shares):\n" + cw.round(4).to_string())
    print("separator tokens, mean nats per token:\n" + cm.round(2).to_string())
    if len(parts) > 1:
        print("SD across models:\n" + parts.std().round(4).to_string())

    names = list(dict.fromkeys(t.model))
    fig, ax = plt.subplots(figsize=(10, 1.8 + 1.3 * len(names)), facecolor=S.SURFACE)
    S.setup_axes(ax, f"Registered minus harness DEV score by protocol element ({args.frac:.0%} of each dev file, "
                     "whole documents)", "nats / word (above 0: the registered number is higher)", "")
    y = np.arange(len(names))[::-1]
    hgt = 0.8 / (len(PARTS) + 1)
    for k, col in enumerate(PARTS):
        ax.barh(y + 0.4 - hgt * (k + 0.5), parts.loc[names, col], height=hgt * 0.9, color=S.CAT[k % len(S.CAT)], label=col)
    ax.scatter(parts.loc[names, "registered minus harness"], y + 0.4 - hgt * (len(PARTS) + 0.5), marker="D",
               color=S.INK, s=22, zorder=3, label="total: registered minus harness")
    ax.axvline(0, color=S.INK2, lw=0.8)
    ax.set_yticks(y, names, fontsize=8)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "scorers_parts.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
