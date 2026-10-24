"""Read temp_check.py output as an arm-vs-reference comparison. Diagnostic only, never a verdict.
The arm is one model or the token-level mixture of several (members tempered at one shared T, as critic ref-0029's
mixture rule); the reference is the mean of single models (e.g. the solo 10-epoch seed pool). Raw = T 1; calibrated =
T cross-fitted on window parity for every single model and for the arm's mixture loss. Optional --base adds the arm vs
its same-batch base. Units: nats/word via DEV's tokens per word; SEs from a window bootstrap.
usage: python3 group_read.py tc.npz --ref 0,1,2,3,4 --arm 5,6 [--base 7]"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('tc')
ap.add_argument('--ref', required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--base')
a = ap.parse_args()
ids = lambda s: [int(i) for i in s.split(',')]
r = np.load(a.tc)
Ts, y = r['Ts'], r['y']
nwin = r['starts'].size
per = y.size // nwin
win = np.repeat(np.arange(nwin), per)
odd = win % 2 == 1
TPW = 16542003 / 10418979  # DEV tokens per whitespace word (dev.meta.json)
i1 = int(np.argmin(abs(Ts - 1)))
nll = lambda k: (r[f'lse{k}'] - r[f'zy{k}'][:, None] / Ts[None, :]).astype(np.float64)  # [n, nT]


def xfit(x):
    out, picks = np.empty(x.shape[0]), []
    for p, q in ((~odd, odd), (odd, ~odd)):
        g = int(x[p].mean(0).argmin())
        picks.append(round(float(Ts[g]), 2))
        out[q] = x[q, g]
    return out, picks


def boot(d, B=2000):
    w = np.bincount(win, weights=d, minlength=nwin) / per
    rng = np.random.default_rng(1)
    return w.mean() * TPW, np.std([w[rng.integers(0, nwin, nwin)].mean() for _ in range(B)]) * TPW


def group(ks):
    """(raw, calibrated, picks) per-token losses of one model or of the shared-T mixture of several."""
    x = nll(ks[0]) if len(ks) == 1 else -np.log(np.mean([np.exp(-nll(k)) for k in ks], 0))
    c, p = xfit(x)
    return x[:, i1], c, p


print(f'{a.tc}: {nwin} windows, {y.size} tokens; ref {a.ref}, arm {a.arm}, base {a.base}')
ref = [group([k]) for k in ids(a.ref)]
for k, (raw, c, p) in zip(ids(a.ref), ref):
    print(f'  ref model {k}: raw {raw.mean() * TPW:.4f} | calibrated {c.mean() * TPW:.4f} (T {p})')
ref_raw, ref_cal = np.mean([g[0] for g in ref], 0), np.mean([g[1] for g in ref], 0)
arm_raw, arm_cal, arm_p = group(ids(a.arm))
for k in ids(a.arm):
    raw, c, p = group([k])
    print(f'  arm member {k}: raw {raw.mean() * TPW:.4f} | calibrated {c.mean() * TPW:.4f} (T {p}) | '
          'behind ref mean raw %+.4f +- %.4f, calibrated %+.4f +- %.4f' % (*boot(raw - ref_raw), *boot(c - ref_cal)))
print(f'  arm: raw {arm_raw.mean() * TPW:.4f} | calibrated {arm_cal.mean() * TPW:.4f} (shared T {arm_p})')
print('ARM GAIN vs ref mean (ref - arm): raw %+.4f +- %.4f | calibrated %+.4f +- %.4f'
      % (*boot(ref_raw - arm_raw), *boot(ref_cal - arm_cal)))
if a.base:
    b_raw, b_cal, b_p = group(ids(a.base))
    print(f'  base {a.base}: raw {b_raw.mean() * TPW:.4f} | calibrated {b_cal.mean() * TPW:.4f} (T {b_p})')
    print('ARM GAIN vs same-batch base: raw %+.4f +- %.4f | calibrated %+.4f +- %.4f'
          % (*boot(b_raw - arm_raw), *boot(b_cal - arm_cal)))
