"""Read temp_check.py output. Diagnostic only, never a verdict.
Per model: free post-hoc calibration by temperature T, by unigram mix lam (u = add-0.5 train_10M token counts, as in
calib_control.py), and by both, each cross-fitted on window parity (pick on even windows, read on odd, and the reverse).
Then: model k vs model 0 gains, raw and temperature-calibrated, with window-bootstrap SEs; and the K-model token-level
mixture gain before and after calibrating every side. Units: nats/word via DEV's tokens per word.
usage: python3 temp_read.py tc.npz"""
import sys
import numpy as np

r = np.load(sys.argv[1])
Ts, y = r['Ts'], r['y']
K = sum(1 for k in r.files if k.startswith('zy'))
nwin = r['starts'].size
per = y.size // nwin
win = np.repeat(np.arange(nwin), per)
odd = win % 2 == 1
TPW = 16542003 / 10418979  # DEV tokens per whitespace word (dev.meta.json)
tr = np.fromfile('data/babylm/bin/train_10M.bin', dtype=np.uint16)
cnt = np.bincount(tr, minlength=16384).astype(np.float64) + 0.5
u = (cnt / cnt.sum())[y]
lams = np.array([0, .0025, .005, .0075, .01, .0125, .015, .02, .03, .05])
nll = np.stack([r[f'lse{k}'] - r[f'zy{k}'][:, None] / Ts[None, :] for k in range(K)]).astype(np.float64)  # [K, n, nT]
i1 = int(np.argmin(abs(Ts - 1)))


def xfit(x):
    """x [n, G]: per-token loss after picking the setting on the other half; returns (loss [n], picks)."""
    out, picks = np.empty(x.shape[0]), []
    for a, b in ((~odd, odd), (odd, ~odd)):
        g = int(x[a].mean(0).argmin())
        picks.append(g)
        out[b] = x[b, g]
    return out, picks


def boot(d, B=2000):
    w = np.bincount(win, weights=d, minlength=nwin) / per
    rng = np.random.default_rng(1)
    return w.mean() * TPW, np.std([w[rng.integers(0, nwin, nwin)].mean() for _ in range(B)]) * TPW


print(f'{K} models, {nwin} windows, {y.size} scored tokens; T grid {[round(float(t), 2) for t in Ts]}')
cal = []
for k in range(K):
    raw = nll[k, :, i1]
    cT, pT = xfit(nll[k])
    p1 = np.exp(-raw)
    cL, pL = xfit(-np.log((1 - lams[None]) * p1[:, None] + lams[None] * u[:, None]))
    pT_all = np.exp(-nll[k])
    both = -np.log((1 - lams[None, None]) * pT_all[:, :, None] + lams[None, None] * u[:, None, None]).reshape(y.size, -1)
    cB, pB = xfit(both)
    print(f'model {k}: raw {raw.mean() * TPW:.4f} nats/word | temperature gain {(raw - cT).mean() * TPW:+.4f} '
          f'(T {[round(float(Ts[i]), 2) for i in pT]}) | unigram gain {(raw - cL).mean() * TPW:+.4f} '
          f'(lam {[float(lams[i]) for i in pL]}) | both {(raw - cB).mean() * TPW:+.4f} '
          f'(T,lam {[(round(float(Ts[i // lams.size]), 2), float(lams[i % lams.size])) for i in pB]})')
    print(f'         per-T nats/word: {np.round(nll[k].mean(0) * TPW, 4).tolist()}')
    cal.append((raw, cT, cB))
for k in range(1, K):
    print(f'model 0 - model {k}: raw %+.4f +- %.4f | temperature-calibrated both %+.4f +- %.4f | T+lam both %+.4f +- %.4f'
          % (*boot(cal[0][0] - cal[k][0]), *boot(cal[0][1] - cal[k][1]), *boot(cal[0][2] - cal[k][2])))
for k2 in range(2, K + 1):
    mix = -np.log(np.exp(-nll[:k2]).mean(0))  # [n, nT], mixture of the members at a shared T
    cm, pm = xfit(mix)
    g_raw = np.mean([cal[k][0] for k in range(k2)], 0) - mix[:, i1]
    g_cal = np.mean([cal[k][1] for k in range(k2)], 0) - cm
    print(f'K={k2} mixture of models 0..{k2 - 1}: g raw %+.4f +- %.4f | g after temperature on every side %+.4f +- %.4f '
          f'(mixture T {[round(float(Ts[i]), 2) for i in pm]})' % (*boot(g_raw), *boot(g_cal)))
