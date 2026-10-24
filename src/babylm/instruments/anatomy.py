"""Mixture probes: how much loss does averaging predictions recover (CSV + PNG)?

For scored models on the same text, a linear mixture sum_k w_k p_k is itself a language
model, so its loss is exact from the stored target losses:
  same run (one tokenizer): mixed per token;
  different tokenizers: mixed per whitespace word (each word's probability given the same
  text prefix).
Weights are fitted by EM on even documents and scored on odd documents, and vice versa, so no
weight is tuned on the text it is scored on. A large gain from mixing checkpoints of one
run says that run's predictions are noisy (variance); a large gain from mixing two 10M
recipes bounds what ensembling or ensemble distillation could buy at 10M.

Usage:
  python -m babylm.instruments.anatomy --tag preread_v1 \
      --mix gpt2_10m/chck_100M+gptbert_cf_10m/chck_100M --mix gpt2_10m/chck_70M+gpt2_10m/chck_100M
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from babylm.instruments import slices as S


def _lse(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def fit_weights(logp: np.ndarray, iters: int = 300) -> np.ndarray:
    """EM for mixture weights. logp: [K, n] log-probabilities of the observed units."""
    K = logp.shape[0]
    w = np.full(K, 1.0 / K)
    for _ in range(iters):
        a = logp + np.log(w)[:, None]
        r = np.exp(a - _lse(a, 0)[None])
        w_new = r.mean(1)
        if np.abs(w_new - w).max() < 1e-7:
            w = w_new
            break
        w = w_new
    return w


def mix_nll(logp: np.ndarray, w: np.ndarray) -> np.ndarray:
    return -_lse(logp + np.log(np.maximum(w, 1e-12))[:, None], 0)


def mixture(specs: list[tuple[str, str]], split: str = "dev") -> dict:
    sc = [S.load_score(r, v, split) for r, v in specs]
    if any(s is None for s in sc):
        raise FileNotFoundError(f"missing scores for {specs}")
    same_tok = len({r for r, _ in specs}) == 1 and all("tok_word" in s for s in sc) and all(
        len(s["tok_word"]) == len(sc[0]["tok_word"]) and np.array_equal(s["tok_word"], sc[0]["tok_word"]) for s in sc)
    n_words = len(sc[0]["word_nll"])
    if same_tok:
        logp = -np.stack([s["tok_nll"].astype(np.float64) for s in sc])
        unit_chunk = S.load_feats(split)["unit"][sc[0]["tok_word"]]
        to_word = lambda x, m: np.bincount(sc[0]["tok_word"][m], weights=x, minlength=n_words)
    else:
        logp = -np.stack([s["word_nll"].astype(np.float64) for s in sc])
        unit_chunk = S.load_feats(split)["unit"]
        to_word = lambda x, m: np.bincount(np.flatnonzero(m), weights=x, minlength=n_words)
    word_mix = np.zeros(n_words)
    weights = {}
    for fit_par in (0, 1):
        fit, ev = unit_chunk % 2 == fit_par, unit_chunk % 2 != fit_par
        w = fit_weights(logp[:, fit])
        weights["fit_on_even" if fit_par == 0 else "fit_on_odd"] = [round(float(x), 4) for x in w]
        word_mix += to_word(mix_nll(logp[:, ev], w), ev)
    singles = [float(s["word_nll"].astype(np.float64).mean()) for s in sc]
    return {"components": [f"{r}/{v}" for r, v in specs], "level": "token" if same_tok else "word",
            "weights": weights, "single_nats": singles, "mix_nats": float(word_mix.mean()),
            "gain_vs_best": float(min(singles) - word_mix.mean()), "word_mix": word_mix,
            "best_word": sc[int(np.argmin(singles))]["word_nll"].astype(np.float64)}


def cache_logp(split: str = "dev") -> np.ndarray:
    """Per word, log p of a unigram cache over the earlier words of its chunk (exact surface
    strings), -inf if the word has not occurred yet (Grave et al. 2017 style, no decay).
    The cache predicts only the word, not the whitespace before it, so mixing it in is
    slightly optimistic."""
    from babylm.instruments.score import load_rows

    out = []
    for r in load_rows(split):
        counts: dict[str, int] = {}
        words = r["text"].split()
        lp = np.full(len(words), -np.inf)
        for i, w in enumerate(words):
            c = counts.get(w, 0)
            if c:
                lp[i] = np.log(c / i)
            counts[w] = c + 1
        out.append(lp)
    return np.concatenate(out)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--mix", action="append", default=[], help="run/rev+run/rev[+...]")
    ap.add_argument("--confidence", default="", help="run/rev,run/rev: surprise minus expected surprise per frequency band")
    ap.add_argument("--temps", default="", help="run/rev,...: gain from one softmax temperature (cross-fitted)")
    ap.add_argument("--cache", default="", help="run/rev,...: gain from mixing in a copy-from-context cache (cross-fitted weight)")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    F = S.load_feats("dev")
    fb, flabels = S.dims(F)["freq10"]
    # overconfidence: a calibrated model's loss on real text equals, on average, the entropy of
    # its own predictions; loss minus entropy > 0 means it is surprised more than it expects
    crow = []
    for spec in [x for x in args.confidence.split(",") if x]:
        r, v = spec.split("/")
        s = S.load_score(r, v, "dev")
        if s is None:
            print(f"skip confidence {spec}: missing scores")
            continue
        nw = len(s["word_nll"])
        ent = np.bincount(s["tok_word"], weights=s["tok_ent"].astype(np.float64), minlength=nw)
        nll = s["word_nll"].astype(np.float64)
        crow.append({"model": spec, "label": "all", "n": nw, "nll": nll.mean(), "entropy": ent.mean(), "surprise_gap": (nll - ent).mean()})
        for k, lab in enumerate(flabels):
            sel = fb == k
            crow.append({"model": spec, "label": lab, "n": int(sel.sum()), "nll": nll[sel].mean(), "entropy": ent[sel].mean(),
                         "surprise_gap": (nll[sel] - ent[sel]).mean()})
    if crow:
        cf = pd.DataFrame(crow)
        cf.to_csv(out / "confidence.csv", index=False)
        print(cf[cf.label == "all"].to_string(index=False))
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=S.SURFACE)
        S.setup_axes(ax, "Loss minus the model's own expected loss (entropy), by train-10M frequency; above 0 = overconfident",
                     "train-10M frequency of the word", "nats / word")
        x = np.arange(len(flabels))
        for i, spec in enumerate(cf.model.unique()[:8]):
            cc = cf[(cf.model == spec) & (cf.label != "all")]
            ax.plot(x[: len(cc)], cc.surprise_gap, color=S.CAT[i], lw=2, marker="o", ms=4, label=spec)
        ax.axhline(0, color=S.INK2, lw=0.8)
        ax.set_xticks(x, flabels, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "confidence.png", dpi=150)
        plt.close(fig)

    # temperature: needs score.py --temps; one T for every prediction, picked on even documents and
    # scored on odd, and vice versa. (A T chosen by the target word's frequency would not be a
    # normalised distribution, so it is deliberately not offered.)
    trow = []
    for spec in [x for x in args.temps.split(",") if x]:
        r, v = spec.split("/")
        p_t = S.SCORES / r / v / "dev_temps.npz"
        if not p_t.exists():
            print(f"skip temps {spec}: run score.py --temps first")
            continue
        z = dict(np.load(p_t))
        Ts = np.r_[1.0, z["temps"]]
        W = np.vstack([z["word_nll"][None].astype(np.float64), z["word_nll_temps"].astype(np.float64)])  # [T, words]
        par = F["unit"] % 2
        glob = np.zeros(W.shape[1])
        picks = {}
        for fit in (0, 1):
            ev = par != fit
            tg = int(np.argmin(W[:, par == fit].mean(1)))
            glob[ev] = W[tg, ev]
            picks[f"fit_on_{'even' if fit == 0 else 'odd'}"] = float(Ts[tg])
        base = W[0]
        trow.append({"model": spec, "T1_nats": base.mean(), "global_T_nats": glob.mean(), "global_gain": base.mean() - glob.mean(),
                     "picks": json.dumps(picks), **{f"nats_T{T:g}": float(W[i].mean()) for i, T in enumerate(Ts)}})
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in trow[-1].items()}))
    if trow:
        pd.DataFrame(trow).to_csv(out / "temperature.csv", index=False)

    # copy-from-context: how much would a simple cache of the chunk's earlier words add?
    krow = []
    specs = [x for x in args.cache.split(",") if x]
    if specs:
        lc = cache_logp("dev")
        ctx = S.dims(F)["context"][0]
        par = F["unit"] % 2
        for spec in specs:
            r, v = spec.split("/")
            s = S.load_score(r, v, "dev")
            if s is None:
                print(f"skip cache {spec}: missing scores")
                continue
            lm = -s["word_nll"].astype(np.float64)
            mixed = np.zeros(len(lm))
            lams = []
            for fit in (0, 1):
                w = fit_weights(np.vstack([lm[par == fit], lc[par == fit]]))
                lams.append(round(float(w[1]), 4))
                ev = par != fit
                mixed[ev] = mix_nll(np.vstack([lm[ev], lc[ev]]), w)
            gain = -lm - mixed
            ci = S.bootstrap_gap(gain, F["unit"], np.zeros(len(gain), dtype=int), 1)[4]
            row = {"model": spec, "nats": float(-lm.mean()), "with_cache": float(mixed.mean()), "gain": float(gain.mean()),
                   "gain_lo": ci[0], "gain_hi": ci[1], "cache_weight": json.dumps(lams)}
            for k, lab in enumerate(S.CTX_LABELS):
                row[f"gain_{lab}"] = float(gain[ctx == k].mean())
            krow.append(row)
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}))
        if krow:
            pd.DataFrame(krow).to_csv(out / "cache.csv", index=False)

    rows, band_rows = [], []
    for spec in args.mix:
        specs = [tuple(x.split("/")) for x in spec.split("+")]
        try:
            m = mixture(specs)
        except FileNotFoundError as e:
            print(f"skip {spec}: {e}")
            continue
        gain = m["best_word"] - m["word_mix"]
        ci = S.bootstrap_gap(gain, F["unit"], np.zeros(len(gain), dtype=int), 1)[4]
        rows.append({"mixture": spec, "level": m["level"], "single_nats": json.dumps([round(x, 4) for x in m["single_nats"]]),
                     "mix_nats": m["mix_nats"], "gain_vs_best": m["gain_vs_best"], "gain_lo": ci[0], "gain_hi": ci[1],
                     "weights": json.dumps(m["weights"])})
        for k, lab in enumerate(flabels):
            sel = fb == k
            band_rows.append({"mixture": spec, "bin": k, "label": lab, "n": int(sel.sum()),
                              "gain": float(gain[sel].mean()) if sel.any() else np.nan,
                              "contrib": float(gain[sel].sum() / len(gain))})
        print(json.dumps({k: v for k, v in rows[-1].items()}, indent=None))
    if not rows:
        return
    pd.DataFrame(rows).to_csv(out / "mixtures.csv", index=False)
    bands = pd.DataFrame(band_rows)
    bands.to_csv(out / "mixtures_by_freq10.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=S.SURFACE)
    S.setup_axes(ax, "Mixture gain over the best single model, by train-10M frequency (cross-fitted weights)",
                 "train-10M frequency of the word", "nats / word recovered")
    x = np.arange(len(flabels))
    for i, spec in enumerate(bands.mixture.unique()[:8]):
        bb = bands[bands.mixture == spec].sort_values("bin")
        ax.plot(x, bb.gain, color=S.CAT[i], lw=2, marker="o", ms=4, label=spec)
    ax.axhline(0, color=S.INK2, lw=0.8)
    ax.set_xticks(x, flabels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "mixtures_by_freq10.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
