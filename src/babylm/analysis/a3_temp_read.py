"""A3 temperature-calibrated gain on temp_check.py windows, with the unseen-character guard leg. Diagnostic only.

    python3 -I babylm/analysis/a3_temp_read.py tc.npz        (model 0 = base, models 1.. = arms)

Same windows, T grid and window-parity cross-fit as shared/babylm/tools/temp_read.py (pick T on even windows,
read on odd, and the reverse; picked per model). Two legs, nats/word:
  full DEV   : mean scored-token nats x DEV tokens per word (as temp_read.py)
  ex-unseen  : tokens whose word holds a character never in train_10M dropped (freq_bands.py's word list), scaled by the
               kept set's tokens per word over the full DEV.
Gains are base - arm (positive = arm better), with window-bootstrap SEs.
"""
import json
import os
import sys

import numpy as np

B = 'data/babylm/'
R = B + 'raw/'
r = np.load(sys.argv[1])
Ts, y, starts = r['Ts'], r['y'], r['starts']
K = sum(1 for k in r.files if k.startswith('zy'))
nwin = starts.size
per = y.size // nwin
P0, L = 256, 1024
win = np.repeat(np.arange(nwin), per)
odd = win % 2 == 1

dev = np.fromfile(B + 'bin/dev.bin', dtype=np.uint16)
wid = np.fromfile(B + 'bin/dev.words.bin', dtype=np.uint32)
W = json.load(open(B + 'bin/dev.meta.json'))['words']
pos = (starts[:, None] + np.arange(P0, L)[None]).ravel()
assert np.array_equal(dev[pos].astype(np.int64), y)

train_chars = set()
for f in os.listdir(R + 'train_10M'):
    train_chars |= set(open(R + 'train_10M/' + f, encoding='utf-8').read())
words = []
for f in sorted(os.listdir(R + 'dev')):
    words += open(R + 'dev/' + f, encoding='utf-8').read().split()
assert len(words) == W, (len(words), W)
bad_w = np.fromiter((any(c not in train_chars for c in w) for w in words), bool, count=W)
bad_tok = bad_w[np.minimum(wid, W - 1)]  # BOS rows carry word id 0 of each doc; a bad first word is negligible
keep_all = ~bad_tok
TPW_full = dev.size / W
TPW_ex = keep_all.sum() / (W - bad_w.sum())
keep = keep_all[pos]
print(f'{K} models, {nwin} windows, {y.size} scored tokens; unseen-char words {bad_w.sum()} '
      f'({bad_w.mean():.3%}); scored tokens dropped by the guard {(~keep).sum()}')

nll = np.stack([r[f'lse{k}'] - r[f'zy{k}'][:, None] / Ts[None, :] for k in range(K)]).astype(np.float64)
i1 = int(np.argmin(abs(Ts - 1)))


def xfit(x, m):
    out, picks = np.full(x.shape[0], np.nan), []
    for a, b in ((~odd, odd), (odd, ~odd)):
        g = int(x[a & m].mean(0).argmin())
        picks.append(round(float(Ts[g]), 2))
        out[b] = x[b, g]
    return out, picks


def boot(d, m, tpw, Bn=2000):
    num = np.bincount(win[m], weights=d[m], minlength=nwin)
    den = np.bincount(win[m], minlength=nwin).astype(float)
    rng = np.random.default_rng(1)
    bs = [num[i].sum() / den[i].sum() for i in (rng.integers(0, nwin, nwin) for _ in range(Bn))]
    return num.sum() / den.sum() * tpw, np.std(bs) * tpw


for leg, m, tpw in (('full DEV', np.ones_like(keep), TPW_full), ('ex-unseen chars', keep, TPW_ex)):
    print(f'== {leg}')
    cal = []
    for k in range(K):
        raw = nll[k, :, i1]
        c, p = xfit(nll[k], m)
        print(f'model {k}: raw {raw[m].mean() * tpw:.4f} | T-calibrated {c[m].mean() * tpw:.4f} (T {p}) | '
              f'per-T {np.round(nll[k][m].mean(0) * tpw, 4).tolist()}')
        cal.append((raw, c))
    for k in range(1, K):
        print(f'base - model {k}: raw %+.4f +- %.4f | T-calibrated both %+.4f +- %.4f'
              % (*boot(cal[0][0] - cal[k][0], m, tpw), *boot(cal[0][1] - cal[k][1], m, tpw)))
