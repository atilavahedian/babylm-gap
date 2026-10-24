"""Is a model confidently wrong, and on which tokens? Runs harness HF exports on DEV-sub documents in the harness
format (whitespace collapsed, newline joins; the BOS and the first 512 tokens of each document) and reports, per band
of the target token's count in train_10M (tokenized the same way):
  - loss, the model's own entropy, and loss minus entropy (above 0: more confident than it is right), with a
    document-bootstrap CI;
  - the target's logit, the share of targets with logit under -14 (the softcap floor is -15), and the max logit;
  - the mean output-row norm over vocabulary rows in the band.
The port's late-epoch damage looked like this: rare targets pushed to the logit floor while the max logit stayed as
high as the 100M twin's. A regulariser that stops the overfit should pull rare targets off the floor.

  python -m babylm.instruments.headcheck --tag base_r1 --data-root ~/babylm_data \
      port_e10=data/babylm_ckpt_local/port_S_10M_e10_s0 twin_100M=shared/babylm/p2/hf/grid_S_100M_e10_s0
Writes results/babylm/<tag>/headcheck.csv and headcheck.png; rows of models already in headcheck.csv under other names
are kept (--fresh drops them), so exports can be added as they land.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from babylm.instruments import slices as S

BANDS = [(0, 0), (1, 15), (16, 255), (256, 4095), (4096, 65535), (65536, 10**12)]
LABELS = ["0", "1-15", "16-255", "256-4k", "4k-65k", "65k+"]
FMT = ("newline", (), True)  # the harness's text format
FLOOR = -14.0


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import torch.nn.functional as Fn

    from babylm.instruments import score as SC
    from babylm.instruments.rare_split import train_counts

    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path of a harness HF export")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--docs", type=int, default=120, help="DEV-sub documents, drawn with seed 0")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--fresh", action="store_true", help="drop rows of earlier runs from headcheck.csv")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in open(S.FEATS / f"devsub_sample_{S.VERSION}.jsonl", encoding="utf-8")]
    pick = sorted(np.random.default_rng(0).choice(len(rows), args.docs, replace=False).tolist())
    texts = [SC.format_text(rows[i], *FMT) for i in pick]
    W = np.random.default_rng(0).poisson(1.0, size=(args.reps, len(pick))).astype(float)
    res = []
    for spec in args.models:
        name, path = spec.split("=", 1)
        path = Path(path)
        model, tok, family, _ = SC.load_model(path, torch.float32)
        sha = hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()[:16]
        cnt = train_counts(tok, sha, FMT, args.data_root)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
        head = model.lm_head.weight.detach().float().norm(dim=1).numpy()
        per = {k: [] for k in ("doc", "nll", "ent", "tl", "mx", "cnt")}
        with torch.inference_mode():
            for j, text in enumerate(texts):
                ids = torch.tensor([[bos] + tok(text, add_special_tokens=False)["input_ids"][:512]])
                lg = model(input_ids=ids[:, :-1]).logits[0].float()
                tgt = ids[0, 1:]
                lp = Fn.log_softmax(lg, -1)
                per["doc"].append(np.full(len(tgt), j))
                per["nll"].append(-lp.gather(1, tgt[:, None])[:, 0].numpy())
                per["ent"].append(-(lp.exp() * lp).sum(-1).numpy())
                per["tl"].append(lg.gather(1, tgt[:, None])[:, 0].numpy())
                per["mx"].append(lg.max(-1).values.numpy())
                per["cnt"].append(cnt[tgt.numpy()])
        p = {k: np.concatenate(v) for k, v in per.items()}
        over = p["nll"] - p["ent"]
        for lab, (lo, hi) in zip(["all"] + LABELS, [(0, 10**12)] + BANDS):
            m = (p["cnt"] >= lo) & (p["cnt"] <= hi)
            vm = (cnt >= lo) & (cnt <= hi)
            r = {"model": name, "path": str(path), "tokenizer_sha256_16": sha, "docs": args.docs, "band": lab, "tokens": int(m.sum()),
                 "token_share": m.mean(), "vocab_rows": int(vm.sum()), "head_norm": head[vm].mean() if vm.any() else np.nan}
            if m.any():
                s = np.bincount(p["doc"][m], weights=over[m], minlength=len(pick))
                n = np.bincount(p["doc"][m], minlength=len(pick)).astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    boot = (W @ s) / (W @ n)
                r.update({"nll": p["nll"][m].mean(), "entropy": p["ent"][m].mean(), "overconf": over[m].mean(),
                          "overconf_lo": np.nanpercentile(boot, 2.5), "overconf_hi": np.nanpercentile(boot, 97.5),
                          "target_logit": p["tl"][m].mean(), "at_floor": (p["tl"][m] < FLOOR).mean(),
                          "max_logit": p["mx"][m].mean(), "nll_share_at_floor": p["nll"][m & (p["tl"] < FLOOR)].sum() / p["nll"][m].sum()})
            res.append(r)
        a = res[-len(BANDS) - 1]
        print(f"{name}: {a['tokens']} tokens, nll {a['nll']:.3f}, loss minus entropy {a['overconf']:+.3f} "
              f"[{a['overconf_lo']:+.3f}, {a['overconf_hi']:+.3f}], {a['at_floor']:.2%} of targets under {FLOOR:g}", flush=True)
        del model
    t = pd.DataFrame(res)
    if (out / "headcheck.csv").exists() and not args.fresh:
        old = pd.read_csv(out / "headcheck.csv")
        keep = ~old.model.isin(set(t.model)) & (old.docs == args.docs if "docs" in old else False)
        t = pd.concat([old[keep], t], ignore_index=True)
    t.to_csv(out / "headcheck.csv", index=False)

    names = list(dict.fromkeys(t.model))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=S.SURFACE)
    S.setup_axes(axes[0], "Loss minus own entropy (above 0: overconfident), 95% CI", "target token's count in train_10M", "nats / token")
    S.setup_axes(axes[1], f"Share of targets with logit under {FLOOR:g} (softcap floor -15)", "target token's count in train_10M", "share of targets")
    S.setup_axes(axes[2], "Mean output-row norm over the vocabulary", "token's count in train_10M", "row norm")
    x = np.arange(len(LABELS))
    for i, nm in enumerate(names):
        g = t[(t.model == nm) & (t.band != "all")].set_index("band").reindex(LABELS)
        col = S.CAT[i % len(S.CAT)]
        axes[0].plot(x, g.overconf, color=col, lw=2, marker="o", ms=4, label=nm)
        axes[0].fill_between(x, g.overconf_lo, g.overconf_hi, color=col, alpha=0.15, lw=0)
        axes[1].plot(x, g.at_floor, color=col, lw=2, marker="o", ms=4, label=nm)
        axes[2].plot(x, g.head_norm, color=col, lw=2, marker="o", ms=4, label=nm)
    axes[0].axhline(0, color=S.INK2, lw=0.8)
    for ax in axes:
        ax.set_xticks(x, LABELS, fontsize=8)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"Confident-wrong check on {args.docs} DEV-sub documents (first 512 tokens each)", color=S.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "headcheck.png", dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(t.drop(columns=["path", "tokenizer_sha256_16", "docs"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
