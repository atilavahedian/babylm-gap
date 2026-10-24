#!/usr/bin/env python3
"""Full-DEV band split for the A5 verdict (docs/babylm/attacks/A5_rare_word_focus.md).

Reads the harness's per-word dump of a finished run (RUN_DIR/pw/pw_devfull_final.npy from babylm.analysis.perword:
one float16 nats value per whitespace word of the official dev split, NaN at the unscored first word) and slices it by
the frequency band of the word in train_10M itself (count of the normal form, as babylm.data.focus_mask defines it:
0 = never seen, 1-63, 64-8k (< 8192), 8k+), by the attack's focus set (rare = count <= 63; after-rare = the 3 words
after a rare word inside the document; focus = rare or after-rare), by source, and with the CHILDES CHAT markup words
excluded (critic ref-0437: the markup-excluded figure is reported as a labelled second number only).

    python babylm/analysis/rare_words/band_split.py --base RUN_DIR [--runs RUN_DIR ...] --raw_dir DIR --out OUT_DIR
        [--metric dev_nats_per_word_twin --gain_bar 0.010 --inconclusive_from 0.0074 --gate 0.0036 --guard 0.0051
         --blimp_drop 2.0 | --no_verdict]
        [--boot 1000] [--block 2000] [--label NAME]

RAW_DIR holds raw/dev/<source>.dev and raw/train_10M/<source>.train (the data dir prepare.py extracted into; on this
thread's box the OSF copy). Deltas against the base are paired block bootstraps over --block consecutive dev words.
With one or more --runs the card's kill rule is evaluated for the first run (the arm) against the base and, when a
second run is given, against the null twin, at the registered numbers (critic ref-0019, superseding ref-0021's
thresholds): primary metric score.json dev_nats_per_word_twin (the blank-line twin scorer; v1 dev_nats_per_word is
reported for information only), interim screen floor 0.0036, screen pass >= 0.010 nats/word vs the same-seed base,
inconclusive 0.0074-0.010, kill below 0.0074, consistency gate 0.0036 (1x floor, same sign), 8k+ guard 0.0051
(1x floor x sqrt 2), BLiMP-fast drop above 2.0 points kills. Flags override the defaults; a run without the v2 score
leaves the verdict "pending".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from babylm.data import corpus as C  # noqa: E402
from babylm.data import focus_mask as FM  # noqa: E402

BANDS = ("0", "1-63", "64-8k", "8k+")
MARKUP = "⌊⌋⌈⌉↗∆∙⁎"  # ⌊ ⌋ ⌈ ⌉ ↗ ∆ ∙ ⁎
COUNT_MAX, AFTER = 63, 3
PW_NAME = "pw_devfull_final.npy"


def band_index(c: int) -> int:
    return 0 if c == 0 else 1 if c <= COUNT_MAX else 2 if c < 8192 else 3


def dev_table(raw_dir: Path) -> dict[str, np.ndarray]:
    """Per-word arrays for the official dev split in the harness's stream order: band, rare, after, markup, src, doc."""
    from babylm.data import prepare as P

    t0 = time.time()
    docs = P.load_docs(raw_dir, "dev")
    dev = [t.split() for s in C.SOURCES for t in docs[s].texts]
    n_per_src = [len(docs[s].texts) for s in C.SOURCES]
    counts = FM.word_counts(FM.split_doc_words(raw_dir, "train_10M"))
    focus_w, rare_w = FM.build_focus_words(dev, counts, COUNT_MAX, AFTER)
    words = [w for d in dev for w in d]
    n = len(words)
    band = np.fromiter((band_index(counts[FM.normal_form(w)]) for w in words), dtype=np.int8, count=n)
    markup = np.fromiter((any(ch in MARKUP for ch in w) for w in words), dtype=bool, count=n)
    # never-seen-character guard (critic ref-0001): words containing a character absent from the train_10M raw text
    chars: set[str] = set()
    for src_name in C.SOURCES:
        chars |= set(C.raw_path(raw_dir, "train_10M", src_name).read_text(encoding="utf-8"))
    unseen = np.fromiter((any(ch not in chars for ch in w) for w in words), dtype=bool, count=n)
    doc_len = np.array([len(d) for d in dev], dtype=np.int64)
    doc = np.repeat(np.arange(len(dev), dtype=np.int32), doc_len)
    src_per_doc = np.repeat(np.arange(len(C.SOURCES), dtype=np.int8), n_per_src)
    src = src_per_doc[doc]
    print(f"[bands] dev {len(dev):,} documents, {n:,} words; train_10M {len(counts):,} types; {time.time() - t0:.0f}s",
          file=sys.stderr)
    return {"band": band, "rare": rare_w.astype(bool), "after": focus_w.astype(bool) & ~rare_w.astype(bool),
            "markup": markup, "unseen": unseen, "src": src, "doc": doc}


def load_table(raw_dir: Path, cache: Path | None) -> dict[str, np.ndarray]:
    if cache is not None and cache.exists():
        z = np.load(cache)
        if "unseen" in z.files:  # older caches predate the never-seen-character flag
            return {k: z[k] for k in z.files}
    t = dev_table(raw_dir)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **t)
    return t


def slices(t: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    band, rare, after, markup, src, unseen = t["band"], t["rare"], t["after"], t["markup"], t["src"], t["unseen"]
    s: dict[str, np.ndarray] = {"all": np.ones(len(band), dtype=bool)}
    for i, b in enumerate(BANDS):
        s[f"band {b}"] = band == i
    s["rare (count <= 63)"] = rare
    s["after-rare (3 words, not rare)"] = after
    s["focus (rare + after-rare)"] = rare | after
    s["frequent, not after-rare"] = ~(rare | after)
    s["all, unseen-chars excluded"] = ~unseen
    s["band 0, unseen-chars excluded"] = (band == 0) & ~unseen
    s["unseen-char words"] = unseen
    s["all, markup-excluded"] = ~markup
    s["band 0, markup-excluded"] = (band == 0) & ~markup
    s["markup words"] = markup
    for i, name in enumerate(C.SOURCES):
        s[f"source {name}"] = src == i
    return s


def load_pw(run_dir: Path, n: int) -> np.ndarray:
    p = run_dir / "pw" / PW_NAME
    pw = np.load(p).astype(np.float64)
    if len(pw) != n:
        raise SystemExit(f"{p}: {len(pw):,} words, the dev split has {n:,}")
    return pw


def block_stats(pw: np.ndarray, valid: np.ndarray, sl: dict[str, np.ndarray], blocks: np.ndarray, nb: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per-slice per-block sums and counts (M, nb)."""
    S = np.zeros((len(sl), nb)); N = np.zeros((len(sl), nb))
    for i, m in enumerate(sl.values()):
        mm = m & valid
        S[i] = np.bincount(blocks[mm], weights=pw[mm], minlength=nb)
        N[i] = np.bincount(blocks[mm], minlength=nb)
    return S, N


def bootstrap(S: np.ndarray, N: np.ndarray, pairs: list[tuple[int, int]], n_boot: int, seed: int = 0
              ) -> np.ndarray:
    """S: (R, M, nb) block sums per run; N: (M, nb) block counts. Returns (len(pairs), M, n_boot) deltas of slice
    means, run a minus run b, under block resampling."""
    rng = np.random.default_rng(seed)
    R, M, nb = S.shape
    out = np.zeros((len(pairs), M, n_boot))
    for k in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(np.float64)
        cnt = N @ w                                   # (M,)
        means = (S @ w) / np.where(cnt > 0, cnt, np.nan)  # (R, M)
        for j, (a, b) in enumerate(pairs):
            out[j, :, k] = means[a] - means[b]
    return out


def fmt(x: float, nd: int = 4) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}" if nd == 4 else f"{x:.{nd}f}"


def read_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="run dir of the control (scored base)")
    ap.add_argument("--runs", nargs="*", default=[], help="run dirs: arm first, then the null twin, then others")
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="npz cache of the dev word table (default: OUT/dev_words_cache.npz)")
    ap.add_argument("--metric", default="dev_nats_per_word_twin", choices=["dev_nats_per_word_twin", "dev_nats_per_word"],
                    help="score.json key of the registered metric (v2 twin by default; v1 is always listed for information)")
    ap.add_argument("--gain_bar", type=float, default=0.010, help="screen pass bar vs the same-seed base (ref-0019)")
    ap.add_argument("--inconclusive_from", type=float, default=0.0074, help="screen gains in [this, gain_bar) are inconclusive")
    ap.add_argument("--gate", type=float, default=0.0036, help="consistency gate: |harness - official total delta| < 1x floor")
    ap.add_argument("--guard", type=float, default=0.0051, help="8k+ guard: arm worse than base by more than this kills")
    ap.add_argument("--blimp_drop", type=float, default=2.0, help="BLiMP-fast drop (points) above which the arm is killed")
    ap.add_argument("--no_verdict", action="store_true", help="table only, no kill-rule block")
    ap.add_argument("--same_condition", action="store_true",
                    help="assert that arm, N1 and base ran under the same execution condition (critic ref-0027)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = load_table(Path(a.raw_dir), Path(a.cache) if a.cache else out / "dev_words_cache.npz")
    n = len(t["band"])
    sl = slices(t)
    run_dirs = [Path(a.base)] + [Path(r) for r in a.runs]
    names = [p.name for p in run_dirs]
    pws = [load_pw(p, n) for p in run_dirs]
    valid = np.ones(n, dtype=bool)
    for pw in pws:
        valid &= np.isfinite(pw)
    blocks = np.arange(n) // a.block
    nb = int(blocks.max()) + 1
    S_list, N = [], None
    for pw in pws:
        S, N = block_stats(pw, valid, sl, blocks, nb)
        S_list.append(S)
    S = np.stack(S_list)                      # (R, M, nb)
    cnt = N.sum(1)                            # (M,)
    means = S.sum(2) / np.where(cnt > 0, cnt, np.nan)  # (R, M)
    share = cnt / cnt[0]
    pairs = [(r, 0) for r in range(1, len(pws))]
    if len(pws) >= 3:
        pairs.append((1, 2))                  # arm minus null twin
    boots = bootstrap(S, N, pairs, a.boot) if pairs else None
    pair_names = [f"{names[x]} - {names[y]}" for x, y in pairs]

    res = {"base": names[0], "runs": names[1:], "n_words": int(n), "n_valid": int(valid.sum()), "block": a.block,
           "n_blocks": nb, "boot": a.boot, "bands": "count of the normal form in train_10M: 0, 1-63, 64-8k (<8192), 8k+",
           "markup_chars": MARKUP, "slices": {}}
    for i, name in enumerate(sl):
        row = {"share_of_words": float(share[i]), "n_words": int(cnt[i]),
               "mean_nats_per_word": {names[r]: float(means[r, i]) for r in range(len(pws))}}
        if boots is not None:
            row["delta"] = {}
            for j, pn in enumerate(pair_names):
                x, y = pairs[j]
                d = boots[j, i]
                row["delta"][pn] = {"point": float(means[x, i] - means[y, i]), "se": float(np.nanstd(d)),
                                    "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
        res["slices"][name] = row
    # exposure counters and official scores, when the run dirs carry them
    res["result_json"] = {}
    for p in run_dirs:
        r = read_json(p / "result.json")
        sc = read_json(p / "score" / "score.json") or read_json(p / "score.json")
        res["result_json"][p.name] = {
            "tokens_seen": r.get("tokens_seen") if r else None, "words_seen": r.get("words_seen") if r else None,
            "gpu": r.get("gpu") if r else None, "tokens_per_s": r.get("tokens_per_s") if r else None,
            "harness_full_dev_nats_per_word": (r.get("full_eval") or {}).get("nats_per_word") if r else None,
            "official_dev_nats_per_word": sc.get("dev_nats_per_word") if sc else None,
            "official_dev_nats_per_word_twin": sc.get("dev_nats_per_word_twin") if sc else None,
            "dev_twin_protocol": sc.get("dev_twin_protocol") if sc else None,
            "fast": sc.get("fast") if sc else None}
    # headline on the registered instrument (experimenter 03:31Z, critic ref-0053): score.json dev_nats_per_word, the dev_bpw
    # protocol on the fp32 HF export. The harness dumps (result.json full_eval and pw_devfull_final.npy) differ from it
    # by -0.02 to +0.03 per model, so the overall verdict uses the official numbers and the band split stays on the
    # harness dumps as the diagnostic of where the change lands. No delta mixes the two instruments.
    i_all = list(sl).index("all")
    official = {nm: rj[f"official_{a.metric}"] for nm, rj in res["result_json"].items()}
    official_v1 = {nm: rj["official_dev_nats_per_word"] for nm, rj in res["result_json"].items()}
    res["headline"] = {"instrument": f"official scorer on the fp32 HF export, score.json {a.metric}"
                                     + (" (v2, blank-line twin)" if a.metric.endswith("_twin") else " (v1)"),
                       "metric": a.metric, "official": official, "official_v1_for_information": official_v1,
                       "harness_all_words": {names[r]: float(means[r, i_all]) for r in range(len(pws))},
                       "delta_official": {}}
    for x, y in pairs:
        ox, oy = official.get(names[x]), official.get(names[y])
        res["headline"]["delta_official"][f"{names[x]} - {names[y]}"] = \
            (float(ox - oy) if ox is not None and oy is not None else None)
    missing_metric = [nm for nm in names[:3] if official.get(nm) is None]
    # execution condition (critic ref-0027): arm, N1 and base must have run alone-vs-alone or packed-vs-packed on the
    # same GPU type. Checked from result.json: same GPU string and tokens/s within 25% of the base; --same_condition
    # asserts it when the operator knows the jobs ran alike (recorded as an assertion).
    tps = {nm: rj.get("tokens_per_s") for nm, rj in res["result_json"].items()}
    gpus = {nm: rj.get("gpu") for nm, rj in res["result_json"].items()}
    tb = tps.get(names[0])
    ratios = {nm: (float(tps[nm]) / float(tb) if tps.get(nm) and tb else None) for nm in names[1:3]}
    same_gpu = all(gpus.get(nm) == gpus.get(names[0]) for nm in names[1:3]
                   if gpus.get(nm) is not None and gpus.get(names[0]) is not None)
    heur = bool(ratios) and all(r is not None and 0.75 <= r <= 1.33 for r in ratios.values()) and same_gpu
    res["execution_condition"] = {
        "rule": "critic ref-0027: arm, N1 and base under the same execution condition (alone vs alone or packed vs "
                "packed, same GPU type)", "gpu": gpus, "tokens_per_s": tps, "tokens_per_s_ratio_to_base": ratios,
        "same_condition": bool(a.same_condition or heur or len(names) == 1),
        "basis": ("control only, nothing to compare" if len(names) == 1 else
                  "asserted by the operator (--same_condition)" if a.same_condition else
                  "same GPU type and tokens/s within 25% of the base" if heur else
                  "GPU type or tokens/s differ from the base (packed vs alone suspected), or result.json missing")}

    def gain(arm: int, ref: int) -> tuple[float, str]:
        """ref minus arm (positive = the arm is better) on the registered instrument; harness only when a score is
        missing, and then labelled."""
        oa, orf = official.get(names[arm]), official.get(names[ref])
        if oa is not None and orf is not None:
            return float(orf - oa), "official"
        return float(means[ref, i_all] - means[arm, i_all]), "harness (official score missing)"

    # kill rule at the registered numbers (card; critic noise floor v1, ref-0022): total-gain legs on the official
    # instrument (arm vs the same-seed base, arm vs N1): >= gain_bar passes, [inconclusive_from, gain_bar) is
    # inconclusive, below fails. The consistency gate (sign agreement and |harness - official| < gate) decides whether
    # the band legs count; the 8k+ guard (arm worse than base by more than `guard` on the harness dumps) and the
    # mechanism share (>= 60% of the harness gain on bands 0 / 1-63 / after-rare) are read only when they count. A
    # BLiMP-fast drop above blimp_drop points kills.
    if len(pws) >= 2 and not a.no_verdict and (missing_metric or not res["execution_condition"]["same_condition"]):
        why = []
        if missing_metric:
            why.append(f"{a.metric} missing in score.json for {missing_metric}")
        if not res["execution_condition"]["same_condition"]:
            why.append("execution condition differs from the base (" + res["execution_condition"]["basis"]
                       + "); a same-condition base is needed")
        res["kill_rule"] = {"verdict": "pending: " + "; ".join(why), "metric": a.metric, "thresholds": None}
    elif len(pws) >= 2 and not a.no_verdict:
        thr = {"metric": a.metric, "gain_bar": a.gain_bar, "inconclusive_from": a.inconclusive_from,
               "gate": a.gate, "guard": a.guard, "blimp_drop": a.blimp_drop, "registered_by": "critic ref-0019"}

        def leg(g: float) -> str:
            return "pass" if g >= a.gain_bar else "inconclusive" if g >= a.inconclusive_from else "fail"

        gain_base, inst_b = gain(1, 0)
        verdict = {"thresholds": thr, "gain_vs_base": gain_base, "gain_instrument": inst_b,
                   "leg_vs_base": leg(gain_base)}
        if len(pws) >= 3:
            gain_null, inst_n = gain(1, 2)
            verdict.update({"gain_vs_null": gain_null, "gain_vs_null_instrument": inst_n, "leg_vs_null": leg(gain_null)})
        gain_base_h = float(means[0, i_all] - means[1, i_all])

        def consistency(arm: int, ref: int) -> dict:
            g_off, inst = gain(arm, ref)
            g_h = float(means[ref, i_all] - means[arm, i_all])
            ok = inst == "official" and (np.sign(g_off) == np.sign(g_h)) and abs(g_off - g_h) < a.gate
            return {"official": g_off, "harness": g_h, "abs_diff": abs(g_off - g_h), "consistent": bool(ok)}

        cons = {"arm_vs_base": consistency(1, 0)}
        if len(pws) >= 3:
            cons["arm_vs_null"] = consistency(1, 2)
        bands_count = all(c["consistent"] for c in cons.values())
        verdict.update({"instrument_consistency": cons, "band_legs_count": bands_count,
                        "band_legs_label": "count" if bands_count else "instrument-inconsistent"})
        i8 = list(sl).index("band 8k+")
        worse8 = float(means[1, i8] - means[0, i8])
        verdict.update({"band_8k_plus_worse_by": worse8, "guard_8k_plus_tripped": worse8 > a.guard})
        part = ["band 0", "band 1-63", "after-rare (3 words, not rare)", "frequent, not after-rare"]
        contrib = {k: float(share[list(sl).index(k)] * (means[0, list(sl).index(k)] - means[1, list(sl).index(k)]))
                   for k in part}
        mech = sum(contrib[k] for k in part[:3])
        verdict.update({"harness_gain_vs_base": gain_base_h, "gain_contribution_by_part": contrib,
                        "mechanism_share": (mech / gain_base_h) if gain_base_h else None,
                        "mechanism_check_passed": bool(gain_base_h > 0 and mech / gain_base_h >= 0.6),
                        "band_instrument": "harness per-word dumps"})
        fast = {nm: (rj.get("fast") or {}).get("blimp/blimp_fast") for nm, rj in res["result_json"].items()}
        fb, fa = fast.get(names[0]), fast.get(names[1])
        bdrop = float(fb - fa) if fb is not None and fa is not None else None
        verdict.update({"blimp_fast": fast, "blimp_fast_drop_vs_base": bdrop,
                        "blimp_kill": bool(bdrop is not None and bdrop > a.blimp_drop)})
        if not bands_count:  # no mechanism claim and no guard reading when the instruments disagree
            verdict.update({"guard_8k_plus_tripped": None, "mechanism_share": None, "mechanism_check_passed": None})
        # never-seen-character guard (critic ref-0001): the gain must also clear the bar with the words containing a
        # character absent from train_10M removed. Read here on the harness dumps (the only instrument with per-word
        # losses for these runs); gains on untrained byte rows are not language learning.
        i_ex = list(sl).index("all, unseen-chars excluded")
        ex_base = float(means[0, i_ex] - means[1, i_ex])
        verdict["unseen_excluded"] = {"instrument": "harness per-word dumps", "gain_vs_base": ex_base,
                                      "leg_vs_base": leg(ex_base)}
        if len(pws) >= 3:
            ex_null = float(means[2, i_ex] - means[1, i_ex])
            verdict["unseen_excluded"].update({"gain_vs_null": ex_null, "leg_vs_null": leg(ex_null)})
        ex_legs = [verdict["unseen_excluded"]["leg_vs_base"]] + \
                  ([verdict["unseen_excluded"]["leg_vs_null"]] if "leg_vs_null" in verdict["unseen_excluded"] else [])
        legs = [verdict["leg_vs_base"]] + ([verdict["leg_vs_null"]] if "leg_vs_null" in verdict else [])
        verdict["unseen_guard_blocks_pass"] = bool(all(x == "pass" for x in legs) and any(x != "pass" for x in ex_legs))
        kill = ("fail" in legs or "fail" in ex_legs or bool(bands_count and verdict["guard_8k_plus_tripped"])
                or verdict["blimp_kill"])
        legs = legs + ex_legs  # an inconclusive excluded leg makes the whole reading inconclusive
        verdict["kill"] = bool(kill)
        verdict["verdict"] = "KILL" if kill else "inconclusive" if "inconclusive" in legs else "survives"
        res["kill_rule"] = verdict

    label = a.label or ("base" if not a.runs else "_vs_".join(names[1:] + [names[0]]))[:120]
    (out / f"bands_{label}.json").write_text(json.dumps(res, indent=1) + "\n")
    n_unseen = int(t["unseen"].sum())
    res["n_unseen_char_words"] = n_unseen
    lines = [f"# Full-DEV band split: {', '.join(names)}", "",
             f"{n:,} dev words ({res['n_valid']:,} scored in every run), {nb:,} blocks of {a.block:,} words, "
             f"{a.boot} paired block bootstraps. Bands = count of the word's normal form in train_10M "
             f"(0 / 1-63 / 64-8k / 8k+). Unseen-chars-excluded rows drop the {n_unseen:,} dev words containing a "
             f"character absent from the train_10M raw text (critic ref-0001; the CHAT markup words, {MARKUP}, are a "
             f"subset); markup-excluded rows are the older labelled second number. Neither is the ranking number.", ""]
    hl = res["headline"]
    lines += [f"Headline, registered instrument ({hl['instrument']}): " +
              "; ".join(f"{nm} {v:.4f}" if v is not None else f"{nm} n/a" for nm, v in hl["official"].items())
              + (". Δ: " + "; ".join(f"{k} {v:+.4f}" if v is not None else f"{k} n/a" for k, v in hl["delta_official"].items())
                 if hl["delta_official"] else "") + ". For information, v1 dev_nats_per_word: " +
              "; ".join(f"{nm} {v:.4f}" if v is not None else f"{nm} n/a" for nm, v in hl["official_v1_for_information"].items())
              + ". Harness all-words figures (the table's instrument): " +
              "; ".join(f"{nm} {v:.4f}" for nm, v in hl["harness_all_words"].items()) + ".", ""]
    head = ["slice", "share", f"{names[0]} (nats/word)"] + [f"{nm}" for nm in names[1:]] + \
           [f"Δ {pn} [95% CI]" for pn in pair_names]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, name in enumerate(sl):
        cells = [name, f"{share[i]:.4f}", f"{means[0, i]:.4f}"] + [f"{means[r, i]:.4f}" for r in range(1, len(pws))]
        if boots is not None:
            for j in range(len(pairs)):
                x, y = pairs[j]
                d = boots[j, i]
                cells.append(f"{means[x, i] - means[y, i]:+.4f} [{np.nanpercentile(d, 2.5):+.4f}, "
                             f"{np.nanpercentile(d, 97.5):+.4f}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for nm, rj in res["result_json"].items():
        lines.append(f"- {nm}: tokens_seen {rj['tokens_seen']}, words_seen {rj['words_seen']}, harness full-DEV "
                     f"{rj['harness_full_dev_nats_per_word']}, official dev_bpw {rj['official_dev_nats_per_word']}")
    ec = res["execution_condition"]
    lines.append("- Execution condition (critic ref-0027): " + "; ".join(
        f"{nm}: gpu {ec['gpu'].get(nm)}, {ec['tokens_per_s'].get(nm) or 'n/a'} tok/s" for nm in names)
        + f". Same condition: {'yes' if ec['same_condition'] else 'NO'} ({ec['basis']}).")
    if "kill_rule" in res and res["kill_rule"].get("thresholds") is None:
        lines += ["", f"Kill rule: {res['kill_rule']['verdict']}."]
    elif "kill_rule" in res:
        v = res["kill_rule"]; t = v["thresholds"]
        line = (f"Kill rule (registered thresholds, critic ref-0019; gains on the {v['gain_instrument']} instrument "
                f"{t['metric']}, bands on the harness dumps): gain vs base {v['gain_vs_base']:+.4f} ({v['leg_vs_base']}; pass "
                f">= {t['gain_bar']}, inconclusive from {t['inconclusive_from']})")
        if "gain_vs_null" in v:
            line += f"; gain vs null {v['gain_vs_null']:+.4f} ({v['leg_vs_null']})"
        line += ("; harness-vs-official total deltas "
                 + ", ".join(f"{k} {c['harness']:+.4f} vs {c['official']:+.4f}" for k, c in v["instrument_consistency"].items())
                 + f" -> band legs {v['band_legs_label']} (gate {t['gate']})")
        if v["band_legs_count"]:
            line += (f"; 8k+ worse by {v['band_8k_plus_worse_by']:+.4f} "
                     f"({'guard tripped' if v['guard_8k_plus_tripped'] else 'guard ok'}, limit {t['guard']}); mechanism share "
                     f"{round(v['mechanism_share'], 3) if v['mechanism_share'] is not None else 'n/a'} "
                     f"({'ok' if v['mechanism_check_passed'] else 'not met'})")
        else:
            line += "; guard and mechanism not evaluated (instrument-inconsistent), no mechanism claim"
        ux = v["unseen_excluded"]
        line += (f"; never-seen-character guard (harness dumps, unseen-char words removed): gain vs base "
                 f"{ux['gain_vs_base']:+.4f} ({ux['leg_vs_base']})"
                 + (f", vs null {ux['gain_vs_null']:+.4f} ({ux['leg_vs_null']})" if "gain_vs_null" in ux else ""))
        bd = v["blimp_fast_drop_vs_base"]
        line += (f"; BLiMP-fast drop vs base {bd:+.2f} points ({'kills' if v['blimp_kill'] else 'ok'}, limit {t['blimp_drop']})"
                 if bd is not None else "; BLiMP-fast not available")
        lines += ["", line + f". Verdict: {v['verdict']}."]
    (out / f"bands_{label}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
