"""Frequency-bias calibration headroom and word-boundary mass, from <model>.calib.npz.
Usage: analyze_calib.py OUT MODEL [drop_mask.npy]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT, NAME = Path(sys.argv[1]), sys.argv[2]
drop = np.load(sys.argv[3]) if len(sys.argv) > 3 else None
cal = np.load(OUT / f"{NAME}.calib.npz")
tok_nll = cal["tok_nll"].astype(np.float64)
band_mass = cal["band_mass"].astype(np.float64)
init_mass = cal["init_mass"].astype(np.float64)
tok_init = cal["tok_init"].astype(bool)
tok_band = cal["tok_band"].astype(np.int64)
tok_word = cal["tok_word"].astype(np.int64)
K = band_mass.shape[1]
edges = cal["edges"].tolist()
band_names = ["0"] + [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
rows = [json.loads(l) for l in (OUT / "sample.jsonl").read_text().splitlines()]
piece_len = np.array([len(r["text"].split()) for r in rows])
piece_of_word = np.repeat(np.arange(len(rows)), piece_len)
lines = (OUT / "words.tsv").read_text(encoding="utf-8").split("\n")
hdr = lines[0].split("\t")
c10 = np.array([int(l.split("\t")[hdr.index("c10")]) for l in lines[1:] if l])
n_words_all = len(c10)
keep_w = np.ones(n_words_all, bool) if drop is None else ~drop
keep = keep_w[tok_word]
n_words = int(keep_w.sum())
wband = np.where(c10 == 0, 0, np.where(c10 <= 63, 1, np.where(c10 < 8192, 2, 3)))
wband_names = ["0", "1-63", "64-8k", "8k+"]
res = {"model": NAME, "n_tokens": int(keep.sum()), "n_words": n_words, "token_bands": band_names}
# 1. calibration per token band: realized share vs mean predicted mass
realized = np.bincount(tok_band[keep], minlength=K) / keep.sum()
predicted = band_mass[keep].mean(0)
res["token_band_realized_share"] = realized.tolist()
res["token_band_predicted_mass"] = predicted.tolist()
res["token_band_log_ratio_pred_over_real"] = np.log(np.maximum(predicted, 1e-12) / np.maximum(realized, 1e-12)).tolist()
# per-band mean nll per token (for the record)
res["token_band_mean_nll"] = [float(tok_nll[keep & (tok_band == k)].mean()) if (keep & (tok_band == k)).any() else None for k in range(K)]
# 2. optimal per-band logit offsets, cross-fitted on odd/even pieces
piece_of_tok = piece_of_word[tok_word]
tb = torch.tensor(tok_band)
M = torch.tensor(band_mass)
nll_t = torch.tensor(tok_nll)


def fit(mask):
    d = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([d], lr=0.5, max_iter=200, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    mt = torch.tensor(mask)

    def closure():
        opt.zero_grad()
        new = nll_t[mt] - d[tb[mt]] + torch.log((M[mt] * torch.exp(d)).sum(1))
        loss = new.mean()
        loss.backward()
        return loss

    opt.step(closure)
    return d.detach()


def apply(d, mask):
    mt = torch.tensor(mask)
    new = nll_t[mt] - d[tb[mt]] + torch.log((M[mt] * torch.exp(d)).sum(1))
    return new.numpy()


odd = keep & (piece_of_tok % 2 == 1)
even = keep & (piece_of_tok % 2 == 0)
gain_tok = np.zeros(len(tok_nll))
d_odd, d_even = fit(odd), fit(even)
gain_tok[even] = tok_nll[even] - apply(d_odd, even)
gain_tok[odd] = tok_nll[odd] - apply(d_even, odd)
res["offsets_fit_on_odd"] = d_odd.tolist()
res["offsets_fit_on_even"] = d_even.tolist()
res["offset_headroom_nats_per_word"] = float(gain_tok[keep].sum() / n_words)
res["offset_headroom_by_word_band_overall"] = {wband_names[b]: float(gain_tok[keep & (wband[tok_word] == b)].sum() / n_words) for b in range(4)}
res["offset_headroom_by_token_band_overall"] = {band_names[k]: float(gain_tok[keep & (tok_band == k)].sum() / n_words) for k in range(K)}
# 3. word-boundary mass: at continuation targets, mass on word-initial tokens (model thinks the word ended);
#    at word-initial targets, mass on continuation tokens (model thinks the word goes on)
res["boundary"] = {}
for b in range(4):
    mw = keep & (wband[tok_word] == b)
    cont = mw & ~tok_init
    init = mw & tok_init
    res["boundary"][wband_names[b]] = dict(
        n_cont=int(cont.sum()), mean_init_mass_at_cont=float(init_mass[cont].mean()) if cont.any() else None,
        median_init_mass_at_cont=float(np.median(init_mass[cont])) if cont.any() else None,
        nats_lost_to_boundary_at_cont_per_word=float(-np.log(np.maximum(1 - init_mass[cont], 1e-9)).sum() / max(1, keep_w[wband == b].sum())) if cont.any() else None,
        mean_cont_mass_at_init=float((1 - init_mass[init]).mean()) if init.any() else None,
        nats_lost_to_boundary_at_init_per_word=float(-np.log(np.maximum(init_mass[init], 1e-9)).sum() / max(1, keep_w[wband == b].sum())) if init.any() else None,
        mean_nll_cont_per_word=float(tok_nll[cont].sum() / max(1, keep_w[wband == b].sum())),
    )
(OUT / f"{NAME}.calib_summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
