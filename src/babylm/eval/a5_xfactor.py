"""A5 final DEV row (critic ref-0185 row A5, amended 10:46Z/11:00Z): x = 10^(gain / Y), Y = S E6 pool cal_t - T100 cal_t.

    python babylm/eval/a5_xfactor.py --json OUT.json > OUT.md

Inputs are the critic-verified cal_t values (full, guarded) cited inline; per-rep gains are recomputed from them.
Gain interval = 95% t-interval of the 3 per-rep gains (df 2), propagated through 10^(g/Y) at the point Y.
Y's own uncertainty is shown separately (Welch SE, Welch-Satterthwaite df), as the rule requires.
"""
import argparse
import json
import math
import statistics as st



def _betacf(a, b, x):
    qab, qap, qam, c, d = a + b, a + 1, a - 1, 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
            h *= d * c
    return h


def t_cdf(t, df):  # Student t CDF via the regularized incomplete beta (Numerical Recipes 6.4)
    x = df / (df + t * t)
    a, b = df / 2, 0.5
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    ib = bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b
    return 1 - ib / 2 if t > 0 else ib / 2


def t_ppf(q, df):
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2

LEGS = ("full", "guarded")
E6_POOL = {"full": (4.9534, 0.0056, 5), "guarded": (4.9251, 0.0033, 5)}          # critic 21:12Z (k2_estar), torch 2.7
T100 = {"full": [4.240369743213724, 4.23797779650208, 4.236948902191036],          # byi6gg ref-0373 calib_port_s0s2.json
        "guarded": [4.236490470846743, 4.234225516075534, 4.233244326273456]}     # torch 2.7, A100
S_SPLIT = {"full": [4.8394, 4.8462, 4.8374], "guarded": [4.8128, 4.8192, 4.8133]}  # p01/p23/p45, 2.7 (critic 21:12Z)
L28_E6_MEAN = {"full": 4.9013, "guarded": 4.8752}                                  # 28L e6 s0-s2 mean, 2.9.1 (critic 06:20Z)
L28_GAIN_A3 = {"full": [0.1140, 0.1117, 0.1178], "guarded": [0.1112, 0.1099, 0.1137]}  # per rep vs L28_E6_MEAN (06:20Z)
F4_BEST = {"full": 4.8728, "guarded": 4.8461}                                      # 36L/512d e4, 2.9.1 (critic 10:40Z)

ROWS = {
    "A1": ("S split vs S E6 pool (same size)", "within 2.7", lambda l: [E6_POOL[l][0] - m for m in S_SPLIT[l]]),
    "A2": ("S split vs best F=2 menu single (28L e6, FLOPs-matched)", "ENV: arm 2.7 vs ref 2.9.1",
           lambda l: [L28_E6_MEAN[l] - m for m in S_SPLIT[l]]),
    "A3": ("28L split vs 28L e6 mean (same size; original dev reps)", "within 2.9.1; Y from 2.7",
           lambda l: list(L28_GAIN_A3[l])),
    "A4": ("28L split vs best of F=4.146 menu (FLOPs-matched)", "within 2.9.1; Y from 2.7",
           lambda l: [F4_BEST[l] - (L28_E6_MEAN[l] - g) for g in L28_GAIN_A3[l]]),
}


def y_stats(leg):
    m1, s1, n1 = E6_POOL[leg]
    t = T100[leg]
    m2, s2, n2 = st.mean(t), st.stdev(t), len(t)
    y = m1 - m2
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    h = t_ppf(0.975, df) * se
    return {"Y": y, "se": se, "df": df, "lo": y - h, "hi": y + h, "T100": m2, "T100_sd": s2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {"label": "DEV-loss data-equivalent multiplier (log-linear), UPPER BOUND (100M point not exhaustively tuned)",
           "Y": {l: y_stats(l) for l in LEGS}, "rows": {}}
    print("# A5: DEV-loss data-equivalent multiplier (log-linear), UPPER BOUND (100M point not exhaustively tuned)\n")
    for l in LEGS:
        y = out["Y"][l]
        print(f"- Y {l} = E6 pool {E6_POOL[l][0]:.4f} - T100 {y['T100']:.5f} = **{y['Y']:.4f}** "
              f"(SE {y['se']:.4f}, Welch df {y['df']:.1f}, 95% [{y['lo']:.4f}, {y['hi']:.4f}]); "
              f"2x bar g >= {y['Y'] * math.log10(2):.4f}")
    print("\n| row | comparison | env | leg | per-rep gains | mean gain (95% t, df 2) | x (95% from gain) | x at Y 95% ends |")
    print("|---|---|---|---|---|---|---|---|")
    for k, (desc, env, f) in ROWS.items():
        out["rows"][k] = {"comparison": desc, "env": env}
        for l in LEGS:
            g = f(l)
            m, s = st.mean(g), st.stdev(g)
            h = t_ppf(0.975, 2) * s / math.sqrt(3)
            y = out["Y"][l]
            x = lambda gg, yy=y["Y"]: 10 ** (gg / yy)
            r = {"gains": g, "mean": m, "sd": s, "g_lo": m - h, "g_hi": m + h,
                 "x": x(m), "x_lo": x(m - h), "x_hi": x(m + h), "x_at_Y_lo": x(m, y["hi"]), "x_at_Y_hi": x(m, y["lo"])}
            out["rows"][k][l] = r
            print(f"| {k} | {desc} | {env} | {l} | {' / '.join(f'{v:+.4f}' for v in g)} | {m:+.4f} [{r['g_lo']:+.4f}, "
                  f"{r['g_hi']:+.4f}] | **{r['x']:.2f}** [{r['x_lo']:.2f}, {r['x_hi']:.2f}] | "
                  f"{r['x_at_Y_lo']:.3f} to {r['x_at_Y_hi']:.3f} |")
    print("\n2x NOT MET: every row's upper end is below 2 on both legs." if all(
        out["rows"][k][l]["x_hi"] < 2 for k in ROWS for l in LEGS) else "\nCHECK: some row reaches 2.")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
