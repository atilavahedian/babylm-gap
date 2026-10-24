"""Full-DEV comparison of two groups of harness runs (e.g. solo vs packed seeds) per source, per CHILDES corpus and per
document, from each run's final full-DEV dump: pw/pw_devfull_final.npy (nats per whitespace word) or
pt/pt_devfull_final.npy (nats per stream token, summed into words as corpus.py attaches tokens: whitespace-only tokens
and BOS go to the following word). Words holding a character that never occurs in train_10M (CHILDES conversation-
analysis markup such as ⌈ ⌉ ⌊ ⌋ ↗ ⁎) are split out, and 'markup-free DEV' drops every document with >= --min-unseen
such words, since no 10M-only method can learn them.

  python -m babylm.instruments.devfull_docs --tag base_r1 --data-root DATA \\
      --group solo s0=RUN_DIR s1=RUN_DIR s2=RUN_DIR --group packed s3=RUN_DIR s4=RUN_DIR s5=RUN_DIR
Writes results/babylm/<tag>/devfull_docs.csv (per document: words, never-seen-character words, nats/word per run),
devfull_groups.csv (per view and source: each run, group means and SDs, the gap second minus first group),
devfull_corpora.csv (each CHILDES corpus's share of the CHILDES gap) and devfull_groups.png.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from babylm.instruments import slices as S
from babylm.instruments.scorers import doc_ranges


def dev_docs(data_root: Path) -> list[tuple[str, str]]:
    """(source, document text) of the dev split in training-stream order (corpus.py documents)."""
    out = []
    for f in sorted((data_root / "dev" / "dev").glob("*.dev")):
        lines = f.read_text(encoding="utf-8").split("\n")
        norm = [" ".join(x.split()) for x in lines]
        out += [(f.stem, "\n".join(norm[a:b])) for a, b in doc_ranges(lines, f.stem)]
    return out


def token_words(text: str, offsets: list[tuple[int, int]]) -> np.ndarray:
    """Word index of each token of a document (whitespace-only tokens attach to the following word)."""
    ends, i = [], 0
    for w in text.split():
        j = text.index(w, i)
        ends.append(j + len(w))
        i = j + len(w)
    first = []
    for a, b in offsets:
        while a < b and text[a].isspace():
            a += 1
        first.append(a)
    return np.searchsorted(np.asarray(ends), np.asarray(first), side="right")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("NAME", "RUN=DIR"),
                    help="a group name then name=run_dir items; exactly two groups")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, help="tokenizer.json (default: the first run's hf/tokenizer.json)")
    ap.add_argument("--min-unseen", type=int, default=20, help="markup-free DEV drops docs with this many such words")
    args = ap.parse_args()
    if len(args.group) != 2:
        raise SystemExit("give exactly two --group")
    groups = {g[0]: dict(x.split("=", 1) for x in g[1:]) for g in args.group}
    (ga, A), (gb, Bg) = groups.items()
    runs = {**A, **Bg}
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    tok_path = args.tokenizer or next(Path(d) / "hf" / "tokenizer.json" for d in runs.values()
                                      if (Path(d) / "hf" / "tokenizer.json").exists())
    tk = Tokenizer.from_file(str(tok_path))
    seen = set()
    for f in (args.data_root / "train_10M" / "train_10M").glob("*.train"):
        seen |= set(f.read_text(encoding="utf-8"))
    docs = dev_docs(args.data_root)
    encs = tk.encode_batch([t for _, t in docs], add_special_tokens=False)
    tw, rows, unseen, base = [], [], [], 0
    for (src, t), e in zip(docs, encs):
        words = t.split()
        tw.append(np.r_[0, token_words(t, e.offsets)] + base)  # the document's BOS goes to its first word
        flags = [any(c not in seen for c in w) for w in words]
        m = re.match(r"= = = ([^ ]+) = = =", t)
        corpus = "/".join(m.group(1).split("/")[1:3]) if (m and src == "childes") else ""
        rows.append({"source": src, "corpus": corpus, "words": len(words), "tokens": 1 + len(e.ids),
                     "unseen_words": int(sum(flags)), "head": t[:80]})
        unseen += flags
        base += len(words)
    tw = np.concatenate(tw)
    D = pd.DataFrame(rows)
    W = pd.DataFrame({"doc": np.repeat(np.arange(len(D)), D.words.values), "unseen": unseen})
    for name, d in runs.items():
        d = Path(d)
        if (d / "pw" / "pw_devfull_final.npy").exists():
            v = np.load(d / "pw" / "pw_devfull_final.npy").astype(np.float64)
            if len(v) != base:
                raise SystemExit(f"{name}: {len(v)} words in pw_devfull_final, the dev split has {base}")
        else:
            t = np.nan_to_num(np.load(d / "pt" / "pt_devfull_final.npy").astype(np.float64))
            if len(t) != len(tw):
                raise SystemExit(f"{name}: {len(t)} tokens in pt_devfull_final, the dev stream has {len(tw)}")
            v = np.bincount(tw, weights=t, minlength=base)
        W[name] = v
        D[name] = np.bincount(W.doc, weights=v, minlength=len(D)) / D.words.clip(lower=1)
    D.to_csv(out / "devfull_docs.csv", index_label="doc")

    W["source"] = D.source.values[W.doc]
    W["markup_doc"] = (D.unseen_words.values >= args.min_unseen)[W.doc]
    views = {"all words": np.ones(len(W), bool), "never-seen-character words": W.unseen.values,
             "other words": ~W.unseen.values, f"markup-free DEV (docs with >= {args.min_unseen} dropped)": ~W.markup_doc.values}
    res = []
    for view, m in views.items():
        for src in ["all"] + sorted(D.source.unique()):
            k = m & ((W.source == src).values if src != "all" else True)
            if not k.any():
                continue
            npw = {r: W.loc[k, r].mean() for r in runs}
            ra, rb = np.array([npw[r] for r in A]), np.array([npw[r] for r in Bg])
            res.append({"view": view, "source": src, "words": int(k.sum()), **npw, f"{ga}_mean": ra.mean(),
                        f"{gb}_mean": rb.mean(), f"{ga}_sd": ra.std(ddof=1) if len(ra) > 1 else np.nan,
                        f"{gb}_sd": rb.std(ddof=1) if len(rb) > 1 else np.nan, "gap": rb.mean() - ra.mean()})
    G = pd.DataFrame(res)
    G.to_csv(out / "devfull_groups.csv", index=False)
    C = D[D.source == "childes"].copy()
    C["d"] = (C[list(Bg)].mean(axis=1) - C[list(A)].mean(axis=1)) * C.words
    K = C.groupby("corpus").agg(docs=("d", "size"), words=("words", "sum"), unseen_words=("unseen_words", "sum"), d=("d", "sum"))
    K["gap_share_of_childes"] = K.d / C.words.sum()
    K["gap_per_word"] = K.d / K.words
    K = K.drop(columns="d").sort_values("gap_share_of_childes", ascending=False)
    K.to_csv(out / "devfull_corpora.csv")

    pd.set_option("display.width", 250)
    print(G[G.source == "all"].drop(columns="source").round(4).to_string(index=False))
    print(G[G.view == "all words"].drop(columns="view").round(4).to_string(index=False))
    print(K.head(8).round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=S.SURFACE)
    S.setup_axes(axes[0], "Full-DEV nats per word by run (harness dumps)", "", "nats / word")
    tot = G[G.source == "all"].set_index("view")
    for i, (gname, members) in enumerate(((ga, A), (gb, Bg))):
        for j, view in enumerate([v for v in views if v != "never-seen-character words"]):
            ys = [tot.loc[view, r] for r in members]
            axes[0].scatter([j + (i - 0.5) * 0.25] * len(ys), ys, color=S.CAT[i], s=26, zorder=3,
                            label=gname if j == 0 else None)
    labs = [v for v in views if v != "never-seen-character words"]
    axes[0].set_xticks(range(len(labs)), [v.replace(" (", "\n(") for v in labs], fontsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    S.setup_axes(axes[1], f"Share of the CHILDES gap ({gb} minus {ga}) by corpus", "nats / CHILDES word", "")
    top = K.head(10)[::-1]
    axes[1].barh(range(len(top)), top.gap_share_of_childes, color=[S.CAT[1] if u else S.CAT[0] for u in top.unseen_words > 0])
    axes[1].set_yticks(range(len(top)), top.index, fontsize=8)
    axes[1].axvline(0, color=S.INK2, lw=0.8)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=S.CAT[1], label="has never-seen-character words"),
                            Patch(color=S.CAT[0], label="none")], fontsize=8, frameon=False)
    fig.suptitle(f"{gb} vs {ga} on full DEV", color=S.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "devfull_groups.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
