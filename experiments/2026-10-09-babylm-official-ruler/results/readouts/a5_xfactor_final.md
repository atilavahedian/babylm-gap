# A5 FINAL: DEV-loss data-equivalent multiplier (log-linear), UPPER BOUND (100M point not exhaustively tuned)

Source: executor 3, byi6gg ref-0176 babylm/gcp/reads/a5/yardstick/a5_out.json (sha256 3cfbd05f39e0...). Tool: a5_yardstick @ref-0377 via the ruler's fixed-T wrapper ref-0380 (the ovr view reuses the full-DEV (T, lam) picks with no refit, per critic ref-0385). Config ref-0389. T100 = port E10 s0-s2.
VALIDITY (critic ref-0385): the all-view g and Y reproduce ref-0376 within 1e-4 on every row and leg (max |dY| 2.3e-5). VALID.
ovr = overlap-removed DEV: 134,739 words from DEV lines found verbatim in train_100M are dropped (critic ref-0260).

| Y | all | ovr |
|---|---|---|
| full | 0.7149 ± 0.0068 (Welch df 5.2) | 0.7099 ± 0.0069 |
| guarded | 0.6904 ± 0.0043 (Welch df 6.0) | 0.6851 ± 0.0043 |

g interval = 95% t, df 2, with SE = sqrt(sd_arm^2/n_arm + sd_ref^2/n_ref). The reference spread is included, so this is wider than the fixed-reference interval in a5_xfactor.md (ref-0376), which this supersedes. x interval = g interval mapped through 10^(g/Y) at the point Y.

| row | comparison | env | leg | view | g (95% t, df 2) | x (95%) |
|---|---|---|---|---|---|---|
| A1 | S split vs S E6 pool (same size) | within 2.7 | full | all | +0.1124 [+0.0968, +0.1280] | 1.436 [1.366, 1.510] |
| A1 | S split vs S E6 pool (same size) | within 2.7 | full | ovr | +0.1126 [+0.0969, +0.1283] | 1.441 [1.369, 1.516] |
| A1 | S split vs S E6 pool (same size) | within 2.7 | guarded | all | +0.1100 [+0.0992, +0.1208] | 1.443 [1.392, 1.496] |
| A1 | S split vs S E6 pool (same size) | within 2.7 | guarded | ovr | +0.1102 [+0.0994, +0.1210] | 1.448 [1.397, 1.502] |
| A2 | S split vs 28L e6 seed mean (FLOPs-matched) | ENV: arm 2.7 vs ref 2.9.1 | full | all | +0.0603 [+0.0462, +0.0744] | 1.214 [1.161, 1.271] |
| A2 | S split vs 28L e6 seed mean (FLOPs-matched) | ENV: arm 2.7 vs ref 2.9.1 | full | ovr | +0.0605 [+0.0464, +0.0747] | 1.217 [1.162, 1.274] |
| A2 | S split vs 28L e6 seed mean (FLOPs-matched) | ENV: arm 2.7 vs ref 2.9.1 | guarded | all | +0.0601 [+0.0490, +0.0712] | 1.222 [1.177, 1.268] |
| A2 | S split vs 28L e6 seed mean (FLOPs-matched) | ENV: arm 2.7 vs ref 2.9.1 | guarded | ovr | +0.0603 [+0.0491, +0.0715] | 1.225 [1.180, 1.272] |
| A3 | 28L split vs 28L e6 mean (same size, original reps) | within 2.9.1; Y from 2.7 | full | all | +0.1145 [+0.1032, +0.1258] | 1.446 [1.394, 1.499] |
| A3 | 28L split vs 28L e6 mean (same size, original reps) | within 2.9.1; Y from 2.7 | full | ovr | +0.1148 [+0.1034, +0.1262] | 1.451 [1.398, 1.506] |
| A3 | 28L split vs 28L e6 mean (same size, original reps) | within 2.9.1; Y from 2.7 | guarded | all | +0.1116 [+0.1032, +0.1199] | 1.451 [1.411, 1.492] |
| A3 | 28L split vs 28L e6 mean (same size, original reps) | within 2.9.1; Y from 2.7 | guarded | ovr | +0.1118 [+0.1035, +0.1202] | 1.456 [1.416, 1.498] |
| A4 | 28L split vs best of F=4.146 menu (FLOPs-matched) | within 2.9.1; Y from 2.7 | full | all | +0.0860 [+0.0609, +0.1111] | 1.319 [1.217, 1.430] |
| A4 | 28L split vs best of F=4.146 menu (FLOPs-matched) | within 2.9.1; Y from 2.7 | full | ovr | +0.0862 [+0.0610, +0.1114] | 1.323 [1.219, 1.435] |
| A4 | 28L split vs best of F=4.146 menu (FLOPs-matched) | within 2.9.1; Y from 2.7 | guarded | all | +0.0825 [+0.0677, +0.0972] | 1.317 [1.253, 1.383] |
| A4 | 28L split vs best of F=4.146 menu (FLOPs-matched) | within 2.9.1; Y from 2.7 | guarded | ovr | +0.0827 [+0.0679, +0.0975] | 1.320 [1.256, 1.388] |


## A2 in registered form (critic ref-0181; row 189: best F=2 single = 28L e6 s0, n_r=1, sd_r = max(28L e6 SD, S pool SD))

| view | leg | arm mean (n=3) | ref 28L e6 s0 | sd_r used | g (95% t, df 2) | x (95%) |
|---|---|---|---|---|---|---|
| all | full | 4.8410 | 4.8975 | 0.0056 | +0.0565 [+0.0300, +0.0829] | 1.199 [1.102, 1.306] |
| all | guarded | 4.8151 | 4.8721 | 0.0032 | +0.0570 [+0.0405, +0.0735] | 1.210 [1.145, 1.278] |
| ovr | full | 4.8495 | 4.9062 | 0.0056 | +0.0567 [+0.0301, +0.0833] | 1.202 [1.102, 1.310] |
| ovr | guarded | 4.8233 | 4.8805 | 0.0032 | +0.0572 [+0.0407, +0.0738] | 1.212 [1.147, 1.281] |

The A2 rows in the main table above use the s0-s2 mean (n_r=3): post-registration, shown for reference only.

## Headline: smaller lower bound of the all and ovr views (registered rule)

| row | full | guarded | point (all, full) |
|---|---|---|---|
| A1 S same-size | 1.37 | 1.39 | 1.44 |
| A2 S FLOPs-matched (crosses env), REGISTERED form: ref = 28L e6 s0 (the DEV-picked best F=2 single), n_r=1, sd_r = S pool SD | 1.10 | 1.15 (1.145) | 1.20 |
| A2, post-registration 3-seed reference (28L e6 s0-s2 mean), NOT the headline | 1.16 | 1.18 | 1.21 |
| A3 28L same-size | 1.39 | 1.41 | 1.45 |
| A4 28L FLOPs-matched | 1.22 | 1.25 | 1.32 |

In every row the binding lower bound comes from the all view; ovr is 0.003 to 0.005 higher, so removing the overlap does not inflate the gain.
2x NOT MET: the highest upper end is 1.52 (A1 full ovr). The best point gain is about half the 2x bar (0.215 / 0.208 nats/word).
