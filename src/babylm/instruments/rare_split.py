"""Bottleneck 2 (rare words) split, for choosing its attack (experimenter, 2026-10-10): the gap on words seen under 64 times
in train_10M (never seen, seen 1 to 63 times), split two ways:
  (a) first occurrence of the word in its chunk vs a repeat (the word occurred earlier in the chunk);
  (b) whether the 10M model's tokenizer cuts the word into a rare piece (a token seen under 64 times in train_10M,
      counted in the model's own training text format) or into frequent pieces only.
Same protocol as prize_bands.csv: lines verbatim in train_100M removed, sources reweighted to the dev split, document
bootstrap stratified by source. The 10M and 100M models of a family have different tokenizers, so a per-token gap is not
defined: (b) splits words, and "pieces" rows split each model's own loss on these words by the rarity of its own pieces
(their difference is an approximate gap on rare vs frequent pieces).

  python -m babylm.instruments.rare_split --data-root ~/babylm_data
Writes results/babylm/<tag>/rare_split.csv and rare_split.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from babylm.instruments import corpus as C
from babylm.instruments import prize as P
from babylm.instruments import slices as S

RARE = 64
CELLS = {"a": ["never seen, first in chunk", "never seen, repeat", "seen 1-63, first in chunk", "seen 1-63, repeat"],
         "b": ["has a rare piece", "frequent pieces only"],
         "ab": [f"{band}, {occ}, {pc}" for band in ("never seen", "seen 1-63") for occ in ("first", "repeat")
                for pc in ("rare piece", "frequent pieces")]}


def tokenizer_for(run: str, rev: str):
    from transformers import PreTrainedTokenizerFast

    m = json.loads((S.SCORES / run / rev / "meta.json").read_text())
    if m.get("local_path"):
        path = Path(m["local_path"]) / "tokenizer.json"
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(m["repo"], "tokenizer.json", revision=m["commit"]))
    fmt = (m.get("line_join", "newline"), tuple(m.get("tab_space", ())), bool(m.get("collapse", False)))
    return PreTrainedTokenizerFast(tokenizer_file=str(path)), hashlib.sha256(path.read_bytes()).hexdigest()[:16], fmt


def train_counts(tok, sha: str, fmt: tuple, data_root: Path) -> np.ndarray:
    """Token counts over train_10M in the model's training text format (cached)."""
    from babylm.instruments.score import SAMPLES, format_text

    cache = SAMPLES / f"tokcount_train_10M_{sha}_{fmt[0]}_{'-'.join(fmt[1]) or 'tabs'}_{int(fmt[2])}.npy"
    if cache.exists():
        return np.load(cache)
    cnt = np.zeros(len(tok), dtype=np.int64)
    for src, path in zip(C.SOURCES, C.split_files(data_root, "train_10M")):
        text = format_text({"text": path.read_text(encoding="utf-8"), "source": src}, *fmt)
        for i in range(0, len(text), 2_000_000):  # cuts mid-line now and then: a few tokens, irrelevant at 64
            cnt += np.bincount(tok(text[i:i + 2_000_000], add_special_tokens=False)["input_ids"], minlength=len(cnt))
    np.save(cache, cnt)
    return cnt


def pieces(run_rev: str, data_root: Path) -> dict:
    """Per dev word: whether its rarest piece is under RARE in train_10M, and the model's loss on rare / frequent pieces."""
    from babylm.instruments.score import load_rows, tokenize_chunks

    run, rev = run_rev.split("/")
    tok, sha, fmt = tokenizer_for(run, rev)
    sc = S.load_score(run, rev, "dev")
    ids = np.concatenate([np.asarray(c[0][1:]) for c in tokenize_chunks(tok, load_rows("dev"), *fmt)])
    if len(ids) != len(sc["tok_nll"]):
        raise RuntimeError(f"{run_rev}: {len(ids)} tokens retokenized vs {len(sc['tok_nll'])} scored")
    cnt = train_counts(tok, sha, fmt, data_root)[ids]
    w, n, nll = sc["tok_word"], len(sc["word_nll"]), sc["tok_nll"].astype(np.float64)
    minc = np.full(n, np.iinfo(np.int64).max)
    np.minimum.at(minc, w, cnt)
    rare = cnt < RARE
    return {"rare_piece": minc < RARE, "loss_rare": np.bincount(w, weights=np.where(rare, nll, 0.0), minlength=n),
            "loss_freq": np.bincount(w, weights=np.where(rare, 0.0, nll), minlength=n), "tokens_rare": rare.mean()}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pair", action="append", default=[], help="name=10M_run/rev:100M_run/rev (default: both headline pairs)")
    args = ap.parse_args()
    pairs = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split(":")) for x in args.pair} or P.HEADLINE
    F = S.load_feats("dev")
    share, keep = P.dev_source_shares(args.data_root), ~F["in_train_100M"]
    src, unit = F["source"].astype(int), F["unit"]
    band = P.freq_groups(F)  # 0 never seen, 1 seen 1-63, 2-3 seen 64+
    b2 = band <= 1
    rep = (S.dims(F)["context"][0] > 0).astype(int)
    out = S.REPO / "results" / "babylm" / args.tag
    rows = []
    for fam, (a, b) in pairs.items():
        na, nb = (S.load_score(*x.split("/"), "dev")["word_nll"].astype(np.float64) for x in (a, b))
        pa, pb = pieces(a, args.data_root), pieces(b, args.data_root)
        rp = pa["rare_piece"].astype(int)
        groupings = {"a": np.where(b2, band * 2 + rep, -1), "b": np.where(b2, 1 - rp, -1),
                     "ab": np.where(b2, band * 4 + rep * 2 + (1 - rp), -1)}
        gap = na - nb
        for split, grp in groupings.items():
            K = len(CELLS[split])
            cg, wd = P.group_reps(gap, grp, src, unit, share, keep, K)
            ca, _ = P.group_reps(na, grp, src, unit, share, keep, K)
            cb, _ = P.group_reps(nb, grp, src, unit, share, keep, K)
            tot = cg.sum(1, keepdims=True)
            q = {k: P.ci(v) for k, v in {"contrib": cg, "share_b2": cg / tot, "gap_per_word": cg / wd}.items()}
            for k, cell in enumerate(CELLS[split]):
                rows.append({"family": fam, "split": split, "cell": cell, "words_pct": 100 * wd[0, k],
                             **{f"{m}{s}": q[m][k, j] for m in q for j, s in enumerate(("", "_lo", "_hi"))},
                             "nll10_per_word": ca[0, k] / wd[0, k], "nll100_per_word": cb[0, k] / wd[0, k]})
        # each model's own loss on these words, by the rarity of its own pieces
        grp = np.where(b2, 0, -1)
        for what, d in (("10M loss on rare pieces", pa["loss_rare"]), ("10M loss on frequent pieces", pa["loss_freq"]),
                        ("100M loss on rare pieces", pb["loss_rare"]), ("100M loss on frequent pieces", pb["loss_freq"]),
                        ("approx. gap on rare pieces", pa["loss_rare"] - pb["loss_rare"]),
                        ("approx. gap on frequent pieces", pa["loss_freq"] - pb["loss_freq"])):
            c, wd = P.group_reps(d, grp, src, unit, share, keep, 1)
            cg, _ = P.group_reps(gap, grp, src, unit, share, keep, 1)
            q = {"contrib": P.ci(c[:, 0]), "share_b2": P.ci(c[:, 0] / cg[:, 0]), "gap_per_word": P.ci(c[:, 0] / wd[:, 0])}
            rows.append({"family": fam, "split": "pieces", "cell": what, "words_pct": 100 * wd[0, 0],
                         **{f"{m}{s}": q[m][j] for m in q for j, s in enumerate(("", "_lo", "_hi"))}})
        print(f"{fam}: {pa['tokens_rare']:.2%} of 10M-model tokens and {pb['tokens_rare']:.2%} of 100M-model tokens are "
              f"rare pieces", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "rare_split.csv", index=False)
    fams = list(pairs)
    fig, axes = plt.subplots(2, len(fams), figsize=(7.6 * len(fams), 6.4), facecolor=S.SURFACE, squeeze=False)
    for j, fam in enumerate(fams):
        for i, split in enumerate(("a", "b")):
            ax = axes[i, j]
            tp = t[(t.family == fam) & (t.split == split)]
            title = (f"{fam}: rare-word gap {tp.contrib.sum():.3f} nats/word, by first vs repeat in chunk" if split == "a"
                     else "by the 10M tokenizer's rarest piece (train_10M count under 64)")
            S.setup_axes(ax, title, "nats / word (contribution to the gap)" if i == 1 else "", "")
            ax.grid(True, axis="x", color=S.GRID, linewidth=0.6)
            ax.grid(False, axis="y")
            y = np.arange(len(tp))[::-1]
            ax.barh(y, tp.contrib, color=S.CAT[0], height=0.6)
            ax.errorbar(tp.contrib, y, xerr=[tp.contrib - tp.contrib_lo, tp.contrib_hi - tp.contrib], fmt="none", ecolor=S.INK2, lw=1)
            for yy, (_, r) in zip(y, tp.iterrows()):
                ax.text(r.contrib_hi + 0.003, yy, f"{r.contrib:.3f}  ({r.share_b2:.0%} [{r.share_b2_lo:.0%}, {r.share_b2_hi:.0%}]; "
                        f"{r.words_pct:.1f}% of words; {r.nll10_per_word:.1f} vs {r.nll100_per_word:.1f} nats/word)",
                        va="center", fontsize=7.5, color=S.INK)
            ax.set_yticks(y, tp.cell, fontsize=8)
            ax.set_xlim(0, t[t.split.isin(["a", "b"])].contrib_hi.max() * 2.4)
    fig.suptitle("Bottleneck 2 (words seen under 64 times in train_10M): share of it, word share, 10M vs 100M loss per word "
                 "(dev sample, train_100M overlap removed, reweighted)", color=S.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "rare_split.png", dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
