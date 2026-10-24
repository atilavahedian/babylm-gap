"""Slice per-word losses into bottleneck tables and curves (CSV + PNG).

Reads data/babylm_scores/<run>/<revision>/<split>.npz (from score.py) and
data/babylm_instruments/features_<split>_v1.npz (from prepare.py), writes
results/babylm/<tag>/ with:
  curves.csv          mean nats/word per slice for every scored checkpoint
  gap_<A>__<B>.csv    per-slice gap between two checkpoints, with document-bootstrap CIs
  memorisation.csv    train vs dev loss per frequency band per checkpoint
  extraction.csv      greedy span recall (all argmax correct) on train vs dev
  summary.json        headline numbers
  *.png               one figure per table

Usage:
  python -m babylm.instruments.slices --tag preread_v1 \
      --pair gpt2_10m/chck_100M:gpt2_100m/chck_1000M --pair ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from babylm.instruments.corpus import sname

REPO = Path(__file__).resolve().parents[2]
SCORES = REPO / "data" / "babylm_scores"
FEATS = REPO / "data" / "babylm_instruments"
VERSION = "v1"
LN2 = np.log(2.0)
SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]

FREQ_EDGES = [0, 1, 2, 4, 8, 16, 64, 256, 1024, 8192]
FREQ_LABELS = ["0", "1", "2-3", "4-7", "8-15", "16-63", "64-255", "256-1023", "1k-8k", "8k+"]
POS_EDGES = [0, 1, 4, 16, 64, 128, 256]
POS_LABELS = ["0", "1-3", "4-15", "16-63", "64-127", "128-255", "256+"]
MATCH_LABELS = ["0 new word", "1", "2", "3", "4", "5", "6", "7", "8"]
CTX_LABELS = ["first in chunk", "word repeated", "bigram repeated"]
# ordinal ramp (dataviz reference palette, blue 250..700) and categorical slots
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def words_seen(rev: str) -> float:
    m = re.fullmatch(r"chck_(\d+)M", rev)
    return float(m.group(1)) * 1e6 if m else float("nan")


SPEAKER_TAG = re.compile(r"\*[A-Z]{2,4}:|[AB]:")  # CHILDES *CHI: / *MOT:, Switchboard A: / B:
# CHILDES conversation-analysis markup (overlap brackets, rising pitch, ...): absent from train_10M, present in
# train_100M and dev, so no 10M-only method can learn it (rare-word thread, PR #65); gaps are reported without it too
CHAT_MARKUP = frozenset("\u230a\u230b\u2308\u2309\u2197\u2206\u2219\u204e")
KIND_LABELS = ["speaker label", "punctuation only", "function word", "content word"]
# closed-class English words (determiners, pronouns, prepositions, conjunctions, auxiliaries and modals,
# negation, a few degree and focus adverbs, contractions), matched on the corpus normal form; fixed before use
FUNCTION_WORDS = frozenset("""
a an the this that these those my your his her its our their whose
some any no every each either neither all both few many much more most several such another other others enough
i me mine myself you yours yourself yourselves he him himself she hers herself it itself we us ours ourselves
they them theirs themselves one ones someone somebody something anyone anybody anything everyone everybody
everything nobody nothing none what which who whom whoever whatever whichever
about above across after against along amid among around as at before behind below beneath beside besides between
beyond by despite down during except for from in inside into like near of off on onto out outside over past per
since than through throughout till to toward towards under underneath unlike until up upon via with within without
and but or nor so yet if because although though while whereas unless whether when whenever where wherever why how
be am is are was were been being have has had having do does did doing
will would shall should can could may might must ought
not n't there here then now just very too also only even
i'm i've i'll i'd you're you've you'll you'd he's he'll he'd she's she'll she'd it's it'll it'd
we're we've we'll we'd they're they've they'll they'd that's that'll there's there're here's what's who's
where's how's let's isn't aren't wasn't weren't don't doesn't didn't haven't hasn't hadn't won't wouldn't
can't cannot couldn't shouldn't mustn't mightn't shan't needn't ain't s re ve ll d m
""".split())
PIECE_LABELS = ["1 piece", "2 pieces", "3 pieces", "4+ pieces"]


def load_feats(split: str) -> dict:
    """The frozen per-word features, plus speaker_tag (computed from the sample text; nearly free
    tokens that the experimenter asked to keep out of per-word averages)."""
    f = dict(np.load(FEATS / f"features_{sname(split)}_{VERSION}.npz"))
    from babylm.instruments.score import load_rows

    from babylm.instruments.corpus import norm

    words = [w for r in load_rows(split) for w in r["text"].split()]
    f["speaker_tag"] = np.array([bool(SPEAKER_TAG.fullmatch(w)) for w in words])
    f["markup"] = np.array([not CHAT_MARKUP.isdisjoint(w) for w in words])
    f["function"] = np.array([norm(w.replace("\u2019", "'")) in FUNCTION_WORDS for w in words])
    un = FEATS / f"units_{sname(split)}_{VERSION}.npy"  # units.py: document ids, the bootstrap and cross-fit unit
    f["unit"] = np.load(un) if un.exists() else f["chunk"]
    ov = FEATS / f"overlap_{sname(split)}_{VERSION}.npz"  # overlap.py: words in lines found verbatim in train_10M / train_100M
    if ov.exists():
        f.update(np.load(ov))
    return f


TOKEN_CLASS_LABELS = ["space or newline", "punctuation", "letters or digits"]


def token_kind(run: str, rev: str, split: str) -> np.ndarray | None:
    """Per scored token (the order of the run's tok_* arrays): 0 whitespace only, 1 punctuation only,
    2 has a letter or digit. Re-tokenizes the sample in the run's own text format with its tokenizer:
    the export's for a local checkpoint, else tokenizer.json at the scored commit. Cached."""
    import hashlib

    from transformers import PreTrainedTokenizerFast

    from babylm.instruments.score import format_text, load_rows

    mp, sp = SCORES / run / rev / "meta.json", SCORES / run / rev / f"{sname(split)}.npz"
    if not mp.exists() or not sp.exists() or "tok_word" not in np.load(sp).files:  # harness dumps: word_nll only
        return None
    m = json.loads(mp.read_text())
    if "line_join" not in m:
        return None
    if m.get("local_path"):
        tf = Path(m["local_path"]) / "tokenizer.json"
    else:
        from huggingface_hub import hf_hub_download

        tf = Path(hf_hub_download(m["repo"], "tokenizer.json", revision=m["commit"]))
    fmt = (m["line_join"], tuple(m.get("tab_space", ())), bool(m.get("collapse", False)))
    key = hashlib.sha256(tf.read_bytes() + repr(fmt).encode()).hexdigest()[:16]
    cache = FEATS / f"tokkind_{key}_{sname(split)}_{VERSION}.npy"
    if cache.exists():
        return np.load(cache)
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tf))
    kind = []
    for r in load_rows(split):
        text = format_text(r, *fmt)
        for a, b in tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]:
            t = text[a:b].strip()
            kind.append(0 if not t else 2 if any(c.isalnum() for c in t) else 1)
    kind = np.array(kind, dtype=np.int8)
    np.save(cache, kind)
    return kind


def word_pieces(run: str, rev: str, split: str) -> np.ndarray | None:
    """Per word: tokens with visible characters in the run's own scoring tokenization."""
    k, sc = token_kind(run, rev, split), load_score(run, rev, split)
    if k is None or sc is None or len(k) != len(sc["tok_word"]):
        return None
    return np.bincount(sc["tok_word"][k > 0], minlength=len(sc["word_nll"])).astype(np.int16)


def load_score(run: str, rev: str, split: str) -> dict | None:
    p = SCORES / run / rev / f"{sname(split)}.npz"
    return dict(np.load(p)) if p.exists() else None


def binned(x: np.ndarray, edges: list[int]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), x, side="right") - 1


COV_LABELS = ["seen in 10M", "only in 100M", "in neither", "word only in 100M", "word in neither"]


def coverage(f: dict, n: int) -> np.ndarray:
    """Is the n-gram ending at this word attested in train_10M, only in train_100M, or in neither?

    Separates what the 100M model could have memorised (coverage) from what it must
    generalise (contexts neither model saw)."""
    m10, m100 = f["match10"], f["match100"]
    cat = np.where(m10 >= n, 0, np.where(m100 >= n, 1, 2))
    cat = np.where(m10 == 0, np.where(m100 >= 1, 3, 4), cat)
    return cat


REF_EDGES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
REF_LABELS = ["<0.5", "0.5-1", "1-2", "2-4", "4-8", "8+"]


def dims(f: dict, train_like: bool = False, ref_nll: np.ndarray | None = None,
         pieces: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Slice dimensions. For train-sample words the frequency excludes the word's own occurrence.

    ref_nll: per-word loss of an independent reference model (a third model, never one of
    the pair being compared, so the slice does not select on either model's own errors).
    pieces: visible tokens per word in one model's tokenization (word_pieces)."""
    f10 = f["freq10"] - (1 if train_like else 0)
    ctx = np.where(f["ctx_bigram"], 2, np.where(f["ctx_prev"] > 0, 1, 0))
    extra = {} if ref_nll is None else {"ref_difficulty": (binned(ref_nll, REF_EDGES), REF_LABELS)}
    if pieces is not None:
        extra["pieces"] = (np.where(pieces == 0, -1, np.minimum(pieces, 4) - 1), PIECE_LABELS)
    return {**extra,
        "freq10": (binned(np.maximum(f10, 0), FREQ_EDGES), FREQ_LABELS),
        "freq100": (binned(np.maximum(f["freq100"] - (1 if train_like else 0), 0), FREQ_EDGES), FREQ_LABELS),
        "position": (binned(f["widx"], POS_EDGES), POS_LABELS),
        "ngram10": (np.clip(f["match10"], 0, 8).astype(int), MATCH_LABELS),
        "context": (ctx, CTX_LABELS),
        "source": (f["source"].astype(int), SOURCES),
        "bigram_cov": (coverage(f, 2), COV_LABELS),
        "trigram_cov": (coverage(f, 3), COV_LABELS),
        "word_kind": (np.where(f["speaker_tag"], 0, np.where(f["punct"], 1, np.where(f["function"], 2, 3))), KIND_LABELS),
        "freq10_first": (np.where(ctx == 0, binned(np.maximum(f10, 0), FREQ_EDGES), -1), FREQ_LABELS),
        "freq10_repeat": (np.where(ctx > 0, binned(np.maximum(f10, 0), FREQ_EDGES), -1), FREQ_LABELS),
    }


def slice_rows(nll: np.ndarray, b: np.ndarray, labels: list[str]) -> list[dict]:
    N = len(nll)
    rows = []
    for k, lab in enumerate(labels):
        m = b == k
        n = int(m.sum())
        rows.append({"bin": k, "label": lab, "n": n, "share": n / N,
                     "mean_nats": float(nll[m].mean()) if n else float("nan"),
                     "contrib_nats": float(nll[m].sum() / N)})
    return rows


def bootstrap_gap(d: np.ndarray, chunk: np.ndarray, b: np.ndarray, nbins: int, reps=1000, seed=0):
    """Cluster (document) Poisson bootstrap of per-bin contribution (sum d / N) and mean d; chunk = cluster ids."""
    nc = int(chunk.max()) + 1
    S = np.zeros((nc, nbins))
    Cn = np.zeros((nc, nbins))
    ok = b >= 0
    np.add.at(S, (chunk[ok], b[ok]), d[ok])
    np.add.at(Cn, (chunk[ok], b[ok]), 1)
    Nc = np.bincount(chunk, minlength=nc).astype(float)
    W = np.random.default_rng(seed).poisson(1.0, size=(reps, nc)).astype(float)
    contrib = (W @ S) / (W @ Nc)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (W @ S) / (W @ Cn)
    lo = lambda a: np.nanpercentile(a, 2.5, axis=0)
    hi = lambda a: np.nanpercentile(a, 97.5, axis=0)
    total = (W @ S.sum(1)) / (W @ Nc)
    return lo(contrib), hi(contrib), lo(mean), hi(mean), (np.percentile(total, 2.5), np.percentile(total, 97.5))


def span_recall(top1: np.ndarray, chunk: np.ndarray, k: int, min_ctx: int = 32) -> tuple[float, int]:
    hits = tot = 0
    starts = np.r_[0, np.flatnonzero(np.diff(chunk)) + 1, len(chunk)]
    for a, z in zip(starts[:-1], starts[1:]):
        t = top1[a + min_ctx : z]
        m = len(t) // k
        if m:
            hits += int(t[: m * k].reshape(m, k).all(1).sum())
            tot += m
    return hits / max(tot, 1), tot


def scored(run: str) -> list[str]:
    d = SCORES / run
    revs = [p.name for p in d.iterdir() if (p / f"{sname('dev')}.npz").exists()] if d.exists() else []
    return sorted(revs, key=words_seen)


def setup_axes(ax, title, xlabel, ylabel, logx=False):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    if logx:
        ax.set_xscale("log")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--runs", default="gpt2_10m,gpt2_100m,gptbert_cf_10m,gptbert_cf_100m")
    ap.add_argument("--pair", action="append", default=[], help="runA/revA:runB/revB (gap = A - B)")
    ap.add_argument("--ref", default="", help="run/rev[,run/rev]: independent models; each gap gets a ref_difficulty slice from the first one not in the pair")
    args = ap.parse_args()
    out = REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in args.runs.split(",") if scored(r)]
    F = {"dev": load_feats("dev"), "train": load_feats("train")}
    D = {"dev": dims(F["dev"]), "train": dims(F["train"], train_like=True)}
    summary: dict = {"runs": {}}

    # 1. curves: every checkpoint, every slice, both splits
    rows = []
    for run in runs:
        for rev in scored(run):
            for split in ("dev", "train"):
                s = load_score(run, rev, split)
                if s is None:
                    continue
                nll = s["word_nll"].astype(np.float64)
                base = {"run": run, "revision": rev, "words_seen": words_seen(rev), "split": split}
                rows.append({**base, "dim": "all", "bin": 0, "label": "all", "n": len(nll), "share": 1.0,
                             "mean_nats": float(nll.mean()), "contrib_nats": float(nll.mean())})
                for dim, (b, labels) in D[split].items():
                    for r in slice_rows(nll, b, labels):
                        rows.append({**base, "dim": dim, **r})
    curves = pd.DataFrame(rows)
    curves["mean_bits"] = curves["mean_nats"] / LN2
    curves.to_csv(out / "curves.csv", index=False)
    for run in runs:
        c = curves[(curves.run == run) & (curves.dim == "all")]
        summary["runs"][run] = {split: {r.revision: round(r.mean_bits, 4) for r in c[c.split == split].itertuples()}
                                for split in ("dev", "train")}

    # 2. total curve
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    setup_axes(ax, "Held-out (dev sample) bits per word vs words seen; dashed = train sample", "words seen", "bits / word", logx=True)
    for i, run in enumerate(runs):
        c = curves[(curves.run == run) & (curves.dim == "all")].sort_values("words_seen")
        for split, ls in (("dev", "-"), ("train", "--")):
            cc = c[c.split == split]
            ax.plot(cc.words_seen, cc.mean_bits, ls, color=CAT[i], lw=2, marker="o", ms=3,
                    label=f"{run} {split}" if split == "dev" else None)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "curve_total.png", dpi=150)
    plt.close(fig)

    # 3. per-dimension curves per run (dev)
    for dim in ("freq10", "position", "ngram10", "context", "source"):
        fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4), facecolor=SURFACE, squeeze=False)
        for ax, run in zip(axes[0], runs):
            c = curves[(curves.run == run) & (curves.dim == dim) & (curves.split == "dev")]
            labels = c.drop_duplicates("bin").sort_values("bin")
            setup_axes(ax, f"{run}: dev nats/word by {dim}", "words seen", "nats / word", logx=True)
            nb = len(labels)
            for j, lab in enumerate(labels.itertuples()):
                cc = c[c.bin == lab.bin].sort_values("words_seen")
                col = RAMP[int(round(j * (len(RAMP) - 1) / max(nb - 1, 1)))] if dim not in ("source", "context") else CAT[j % len(CAT)]
                ax.plot(cc.words_seen, cc.mean_nats, color=col, lw=2, label=lab.label)
            ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"curve_{dim}.png", dpi=150)
        plt.close(fig)
        curves[(curves.dim == dim)].to_csv(out / f"curve_{dim}.csv", index=False)

    # 4. gap decompositions, on dev and on the train sample (text both models saw equally often)
    summary["gaps"] = {}
    refs = [tuple(x.split("/")) for x in args.ref.split(",") if x]
    for spec in args.pair:
        a, b_ = spec.split(":")
        ra, va = a.split("/")
        rb, vb = b_.split("/")
        for split in ("dev", "train"):
            sa, sb = load_score(ra, va, split), load_score(rb, vb, split)
            if sa is None or sb is None:
                print(f"skip pair {spec} {split}: missing scores")
                continue
            d = (sa["word_nll"] - sb["word_nll"]).astype(np.float64)
            name = f"{ra}-{va}__{rb}-{vb}" + ("" if split == "dev" else "_train")
            base = lambda run: re.sub(r"_(collapse|tabspace|tabkeep)$", "", run)  # same model, another text format
            ref_spec = next((x for x in refs if base(x[0]) not in (base(ra), base(rb))), None)
            ref_s = load_score(*ref_spec, split) if ref_spec else None
            r_nll = ref_s["word_nll"] if ref_s is not None else None
            dd = dims(F[split], train_like=split == "train", ref_nll=r_nll, pieces=word_pieces(ra, va, split))
            grows = []
            for dim, (bins_, labels) in dd.items():
                lo_c, hi_c, lo_m, hi_m, _ = bootstrap_gap(d, F[split]["unit"], bins_, len(labels))
                for k, lab in enumerate(labels):
                    m = bins_ == k
                    n = int(m.sum())
                    grows.append({"dim": dim, "bin": k, "label": lab, "n": n, "share": n / len(d),
                                  "mean_A": float(sa["word_nll"][m].mean()) if n else np.nan,
                                  "mean_B": float(sb["word_nll"][m].mean()) if n else np.nan,
                                  "mean_gap": float(d[m].mean()) if n else np.nan, "mean_gap_lo": lo_m[k], "mean_gap_hi": hi_m[k],
                                  "contrib": float(d[m].sum() / len(d)), "contrib_lo": lo_c[k], "contrib_hi": hi_c[k],
                                  "share_of_gap": float(d[m].sum() / d.sum())})
            ka, kb = token_kind(ra, va, split), token_kind(rb, vb, split)
            if ka is not None and kb is not None and len(ka) == len(sa["tok_nll"]) and len(kb) == len(sb["tok_nll"]):
                for k, lab in enumerate(TOKEN_CLASS_LABELS):  # each model's own tokens, summed per word
                    pa = np.bincount(sa["tok_word"][ka == k], weights=sa["tok_nll"][ka == k].astype(np.float64), minlength=len(d))
                    pb = np.bincount(sb["tok_word"][kb == k], weights=sb["tok_nll"][kb == k].astype(np.float64), minlength=len(d))
                    ci = bootstrap_gap(pa - pb, F[split]["unit"], np.zeros(len(d), dtype=int), 1)[4]
                    grows.append({"dim": "token_class", "bin": k, "label": lab, "n": int((ka == k).sum()), "share": np.nan,
                                  "mean_A": float(pa.mean()), "mean_B": float(pb.mean()), "mean_gap": float((pa - pb).mean()),
                                  "mean_gap_lo": float(ci[0]), "mean_gap_hi": float(ci[1]), "contrib": float((pa - pb).mean()),
                                  "contrib_lo": float(ci[0]), "contrib_hi": float(ci[1]), "share_of_gap": float((pa - pb).sum() / d.sum())})
            g = pd.DataFrame(grows)
            g.to_csv(out / f"gap_{name}.csv", index=False)
            tot_ci = bootstrap_gap(d, F[split]["unit"], np.zeros(len(d), dtype=int), 1)[4]
            summary["gaps"][name] = {"split": split, "ref": "/".join(ref_spec) if r_nll is not None else None,
                                     "total_nats": float(d.mean()), "total_ci95": [float(tot_ci[0]), float(tot_ci[1])],
                                     "A_nats": float(sa["word_nll"].mean()), "B_nats": float(sb["word_nll"].mean())}
            panels = ["freq10", "position", "trigram_cov", "context", "source", "ref_difficulty" if r_nll is not None else "freq10_first",
                      "word_kind", "token_class" if (g.dim == "token_class").any() else "freq100"]
            fig, axes = plt.subplots(2, 4, figsize=(18.5, 7.5), facecolor=SURFACE)
            for ax, dim in zip(axes.flat, panels):
                gg = g[g.dim == dim]
                setup_axes(ax, f"{dim}: contribution to gap (nats/word of all text)", "", "nats / word")
                x = np.arange(len(gg))
                ax.bar(x, gg.contrib, color=CAT[0], width=0.6)
                ax.errorbar(x, gg.contrib, yerr=[gg.contrib - gg.contrib_lo, gg.contrib_hi - gg.contrib], fmt="none", ecolor=INK2, lw=1)
                ax.set_xticks(x, gg.label, rotation=45, ha="right", fontsize=7)
                ax.axhline(0, color=INK2, lw=0.8)
            what = "dev sample" if split == "dev" else "train sample (seen equally often by both)"
            fig.suptitle(f"{what}: {ra}/{va} minus {rb}/{vb} = {d.mean():.3f} nats/word "
                         f"(95% CI {tot_ci[0]:.3f} to {tot_ci[1]:.3f}); bars sum to the total", color=INK, fontsize=11)
            fig.tight_layout()
            fig.savefig(out / f"gap_{name}.png", dpi=150)
            plt.close(fig)

    # 5. memorisation: train vs dev by frequency band, per checkpoint; span recall
    mem, ext = [], []
    for run in runs:
        for rev in scored(run):
            sd, st = load_score(run, rev, "dev"), load_score(run, rev, "train")
            if sd is None or st is None:
                continue
            bd, labels = D["dev"]["freq10"]
            bt, _ = D["train"]["freq10"]
            for k, lab in enumerate(labels):
                md, mt = bd == k, bt == k
                mem.append({"run": run, "revision": rev, "words_seen": words_seen(rev), "bin": k, "label": lab,
                            "n_dev": int(md.sum()), "n_train": int(mt.sum()),
                            "dev_nats": float(sd["word_nll"][md].mean()) if md.any() else np.nan,
                            "train_nats": float(st["word_nll"][mt].mean()) if mt.any() else np.nan})
            for split, s in (("dev", sd), ("train", st)):
                if "tok_top1" not in s:  # harness dumps: word_nll only
                    continue
                r8, n8 = span_recall(s["tok_top1"], s["tok_chunk"], 8)
                r16, n16 = span_recall(s["tok_top1"], s["tok_chunk"], 16)
                r32, n32 = span_recall(s["tok_top1"], s["tok_chunk"], 32)
                ext.append({"run": run, "revision": rev, "words_seen": words_seen(rev), "split": split,
                            "top1": float(s["tok_top1"].mean()), "pmax": float(s["tok_pmax"].astype(np.float64).mean()),
                            "span8": r8, "span16": r16, "span32": r32, "n_span16": n16})
    mem, ext = pd.DataFrame(mem), pd.DataFrame(ext)
    mem["gen_gap"] = mem.dev_nats - mem.train_nats
    mem.to_csv(out / "memorisation.csv", index=False)
    if len(ext):
        ext.to_csv(out / "extraction.csv", index=False)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4), facecolor=SURFACE, squeeze=False)
    for ax, run in zip(axes[0], runs):
        m = mem[mem.run == run]
        setup_axes(ax, f"{run}: dev minus train nats/word by train-10M freq", "words seen", "generalisation gap (nats)", logx=True)
        for j, lab in enumerate(FREQ_LABELS):
            cc = m[m.label == lab].sort_values("words_seen")
            ax.plot(cc.words_seen, cc.gen_gap, color=RAMP[j], lw=2, label=lab)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "memorisation.png", dpi=150)
    plt.close(fig)
    if len(ext):
        fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
        setup_axes(ax, "Greedy 16-token span recall (all argmax correct); dashed = train sample", "words seen", "fraction of spans", logx=True)
        for i, run in enumerate(runs):
            e = ext[ext.run == run].sort_values("words_seen")
            for split, ls in (("dev", "-"), ("train", "--")):
                ee = e[e.split == split]
                ax.plot(ee.words_seen, ee.span16, ls, color=CAT[i], lw=2, label=run if split == "dev" else None)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "extraction.png", dpi=150)
        plt.close(fig)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["gaps"], indent=1))


if __name__ == "__main__":
    main()
