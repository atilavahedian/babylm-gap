"""Where learning stalls: every scored checkpoint of one run, dev and train sample, by slice (CSV + PNG).

For each checkpoint (sub-epoch ones included) the dev and train-sample nats/word, overall
and per train-10M frequency group, next to a reference final (the 100M-track model):
  progress     share of the distance from epoch 1 to the reference that a checkpoint has
               closed; a group whose progress flattens well below 1 has stalled
  late gain    dev loss drop per epoch over the last scored epochs (7 to 10 when present),
               with a document-bootstrap CI, and the epochs it would take at that rate to
               reach the reference (learning-rate decay makes late epochs look better, so
               this is optimistic)
  gen. gap     dev minus train-sample loss; train text is seen once per epoch, so its
               growth is how much of each epoch's progress is specific to the 10M words
--max-tok-pos N keeps only words whose tokens all sit at window positions under N in the
run's own scoring, for runs whose early checkpoints trained on short sequences only
(GPT-BERT: 128 tokens for the first 60% of training).

Usage:
  python -m babylm.instruments.stall --tag preread_v1 --run gpt2_10m --ref gpt2_100m/chck_1000M
  python -m babylm.instruments.stall --tag preread_v1 --run gptbert_cf_10m --ref gptbert_cf_100m/chck_1000M --max-tok-pos 128
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from babylm.instruments import slices as S

GROUPS = [("never seen in 10M", ["0"]), ("seen 1-63 times", ["1", "2-3", "4-7", "8-15", "16-63"]),
          ("seen 64-8k times", ["64-255", "256-1023", "1k-8k"]), ("seen 8k+ times", ["8k+"])]


def word_mask(score: dict, n_words: int, max_tok_pos: int | None) -> np.ndarray:
    if not max_tok_pos:
        return np.ones(n_words, dtype=bool)
    top = np.zeros(n_words, dtype=np.int64)
    np.maximum.at(top, score["tok_word"], score["tok_pos"].astype(np.int64))
    return top < max_tok_pos


def group_index(f10_bins: np.ndarray) -> np.ndarray:
    g = np.full(len(f10_bins), -1)
    for gi, (_, labels) in enumerate(GROUPS):
        for lab in labels:
            g[f10_bins == S.FREQ_LABELS.index(lab)] = gi
    return g


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--run", required=True, help="10M-track run; checkpoints chck_<N>M")
    ap.add_argument("--ref", required=True, help="run/rev of the reference final (100M track)")
    ap.add_argument("--max-tok-pos", type=int, default=0)
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_pos{args.max_tok_pos}" if args.max_tok_pos else ""
    revs = sorted(S.scored(args.run), key=S.words_seen)
    ref_run, ref_rev = args.ref.split("/")
    rows, curves, info = [], {}, {"run": args.run, "ref": args.ref, "max_tok_pos": args.max_tok_pos}
    for split in ("dev", "train"):
        F = S.load_feats(split)
        n = len(F["chunk"])
        anchor = S.load_score(args.run, revs[-1], split)
        if anchor is None:
            continue
        keep = word_mask(anchor, n, args.max_tok_pos)
        f10 = S.dims(F, train_like=split == "train")["freq10"][0]
        grp = np.where(keep, group_index(f10), -1)
        info[f"{split}_words_kept"] = int(keep.sum())
        models = [(args.run, r) for r in revs] + [(ref_run, ref_rev)]
        for run, rev in models:
            s = S.load_score(run, rev, split)
            if s is None:
                continue
            nll = s["word_nll"].astype(np.float64)
            curves[(split, run, rev)] = nll
            ws = S.words_seen(rev) if run == args.run else np.nan
            rows.append({"split": split, "run": run, "revision": rev, "words_seen": ws, "group": "all",
                         "n": int(keep.sum()), "mean_nats": float(nll[keep].mean())})
            for gi, (name, _) in enumerate(GROUPS):
                m = grp == gi
                rows.append({"split": split, "run": run, "revision": rev, "words_seen": ws, "group": name,
                             "n": int(m.sum()), "mean_nats": float(nll[m].mean()) if m.any() else np.nan})
        info[f"{split}_grp"] = grp
        info[f"{split}_keep"] = keep
        info[f"{split}_chunk"] = F["unit"]
    t = pd.DataFrame(rows)
    t.to_csv(out / f"stall_{args.run}{suffix}.csv", index=False)

    # per-group summary on dev: progress at each epoch, late gain with CI, epochs to close
    dev = t[(t.split == "dev") & (t.run == args.run)]
    ref = t[(t.split == "dev") & (t.run == ref_run)].set_index("group").mean_nats
    epochs = sorted({int(w // 10e6) for w in dev.words_seen if w % 10e6 == 0 and w >= 10e6})
    late = [e for e in epochs if e >= 7] if 7 in epochs else epochs[-2:]
    e0, e1 = (late[0], late[-1]) if len(late) >= 2 else (None, None)
    summ = {"run": args.run, "ref": args.ref, "max_tok_pos": args.max_tok_pos, "epochs_scored": epochs,
            "late_window": [e0, e1], "dev_words_kept": info.get("dev_words_kept"), "groups": {}}
    grp, keep, chunk = info["dev_grp"], info["dev_keep"], info["dev_chunk"]
    for gi, name in enumerate(["all"] + [g for g, _ in GROUPS]):
        d = dev[dev.group == name].set_index("words_seen").mean_nats
        L = {e: float(d.get(e * 10e6, np.nan)) for e in epochs}
        g = {"ep1": L.get(1), f"ep{epochs[-1]}": L[epochs[-1]], "ref": float(ref.get(name, np.nan))}
        dist = g["ep1"] - g["ref"] if g["ep1"] is not None else np.nan
        g["progress"] = {e: round((L[1] - L[e]) / dist, 4) for e in epochs} if 1 in L and dist > 0 else {}
        if e0 is not None:
            a = curves[("dev", args.run, f"chck_{e0 * 10}M")]
            b = curves[("dev", args.run, f"chck_{e1 * 10}M")]
            sel = keep if name == "all" else grp == gi - 1
            lo, hi = S.bootstrap_gap(a - b, chunk, np.where(sel, 0, -1), 1)[2:4]
            per = (e1 - e0)
            g["late_gain_per_epoch"] = float((a[sel] - b[sel]).mean() / per)
            g["late_gain_per_epoch_ci95"] = [float(lo[0] / per), float(hi[0] / per)]
            rem = L[e1] - g["ref"]
            g["remaining_to_ref"] = float(rem)
            g["epochs_to_ref_at_late_rate"] = float(rem / g["late_gain_per_epoch"]) if g["late_gain_per_epoch"] > 0 else None
        trn = t[(t.split == "train") & (t.run == args.run) & (t.group == name)].set_index("words_seen").mean_nats
        # train-sample words are all in train_10M: their "never seen" group is words seen only there, not comparable
        g["gen_gap"] = {} if name == GROUPS[0][0] else {e: round(float(d[e * 10e6] - trn[e * 10e6]), 4) for e in epochs if e * 10e6 in trn.index}
        rt = t[(t.split == "train") & (t.run == ref_run) & (t.group == name)].mean_nats
        if len(rt) and g["gen_gap"]:
            g["gen_gap_ref"] = round(g["ref"] - float(rt.iloc[0]), 4)
        summ["groups"][name] = g
    (out / f"stall_{args.run}{suffix}.json").write_text(json.dumps(summ, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=S.SURFACE)
    pos = f"\nwords in the first {args.max_tok_pos} tokens of each window" if args.max_tok_pos else ""
    S.setup_axes(axes[0], f"{args.run}: loss vs words seen{pos}", "words seen (M)", "nats / word", logx=True)
    for split, ls in (("dev", "-"), ("train", "--")):
        c = t[(t.split == split) & (t.run == args.run) & (t.group == "all")]
        if len(c):
            axes[0].plot(c.words_seen / 1e6, c.mean_nats, ls, color=S.CAT[0], lw=2, marker="o", ms=3,
                         label=f"{split} ({'new text' if split == 'dev' else 'text it trains on'})")
        r = t[(t.split == split) & (t.run == ref_run) & (t.group == "all")]
        if len(r):
            axes[0].axhline(float(r.mean_nats.iloc[0]), ls=":" if split == "dev" else "-.", color=S.CAT[1], lw=1.5,
                            label=f"{args.ref} {split}")
    axes[0].legend(fontsize=7, frameon=False)
    S.setup_axes(axes[1], "Share of the epoch-1-to-reference distance closed (dev)", "epoch", "share closed")
    S.setup_axes(axes[2], "Generalisation gap: dev minus train-sample loss\n(words seen in train_10M)", "epoch", "nats / word")
    for j, (name, _) in enumerate([("all", None)] + GROUPS):
        g = summ["groups"][name]
        col = S.INK if name == "all" else S.CAT[j - 1]
        if g["progress"]:
            es = sorted(g["progress"])
            axes[1].plot(es, [g["progress"][e] for e in es], color=col, lw=2, marker="o", ms=3, label=name)
        if g["gen_gap"]:
            es = sorted(g["gen_gap"])
            axes[2].plot(es, [g["gen_gap"][e] for e in es], color=col, lw=2, marker="o", ms=3, label=name)
            if "gen_gap_ref" in g:
                axes[2].plot([max(es) + 0.4], [g["gen_gap_ref"]], "D", color=col, ms=5)
    axes[1].axhline(1, color=S.INK2, lw=0.8, ls=":")
    axes[1].legend(fontsize=7, frameon=False, title="train-10M count", title_fontsize=7)
    axes[2].legend(fontsize=7, frameon=False, title=f"diamonds = {args.ref}", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"stall_{args.run}{suffix}.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
