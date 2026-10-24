# A5: DEV-loss data-equivalent multiplier (log-linear), UPPER BOUND (100M point not exhaustively tuned)

- Y full = E6 pool 4.9534 - T100 4.23843 = **0.7150** (SE 0.0027, Welch df 5.1, 95% [0.7081, 0.7219]); 2x bar g >= 0.2152
- Y guarded = E6 pool 4.9251 - T100 4.23465 = **0.6904** (SE 0.0018, Welch df 6.0, 95% [0.6861, 0.6948]); 2x bar g >= 0.2078

| row | comparison | env | leg | per-rep gains | mean gain (95% t, df 2) | x (95% from gain) | x at Y 95% ends |
|---|---|---|---|---|---|---|---|
| A1 | S split vs S E6 pool (same size) | within 2.7 | full | +0.1140 / +0.1072 / +0.1160 | +0.1124 [+0.1009, +0.1239] | **1.44** [1.38, 1.49] | 1.431 to 1.441 |
| A1 | S split vs S E6 pool (same size) | within 2.7 | guarded | +0.1123 / +0.1059 / +0.1118 | +0.1100 [+0.1012, +0.1188] | **1.44** [1.40, 1.49] | 1.440 to 1.447 |
| A2 | S split vs best F=2 menu single (28L e6, FLOPs-matched) | ENV: arm 2.7 vs ref 2.9.1 | full | +0.0619 / +0.0551 / +0.0639 | +0.0603 [+0.0488, +0.0718] | **1.21** [1.17, 1.26] | 1.212 to 1.217 |
| A2 | S split vs best F=2 menu single (28L e6, FLOPs-matched) | ENV: arm 2.7 vs ref 2.9.1 | guarded | +0.0624 / +0.0560 / +0.0619 | +0.0601 [+0.0513, +0.0689] | **1.22** [1.19, 1.26] | 1.220 to 1.223 |
| A3 | 28L split vs 28L e6 mean (same size; original dev reps) | within 2.9.1; Y from 2.7 | full | +0.1140 / +0.1117 / +0.1178 | +0.1145 [+0.1068, +0.1222] | **1.45** [1.41, 1.48] | 1.441 to 1.451 |
| A3 | 28L split vs 28L e6 mean (same size; original dev reps) | within 2.9.1; Y from 2.7 | guarded | +0.1112 / +0.1099 / +0.1137 | +0.1116 [+0.1068, +0.1164] | **1.45** [1.43, 1.47] | 1.448 to 1.454 |
| A4 | 28L split vs best of F=4.146 menu (FLOPs-matched) | within 2.9.1; Y from 2.7 | full | +0.0855 / +0.0832 / +0.0893 | +0.0860 [+0.0783, +0.0937] | **1.32** [1.29, 1.35] | 1.316 to 1.323 |
| A4 | 28L split vs best of F=4.146 menu (FLOPs-matched) | within 2.9.1; Y from 2.7 | guarded | +0.0821 / +0.0808 / +0.0846 | +0.0825 [+0.0777, +0.0873] | **1.32** [1.30, 1.34] | 1.314 to 1.319 |

2x NOT MET: every row's upper end is below 2 on both legs.

Notes (ruler):
- Rules: critic ref-0185 (A5), 10:46Z amendment (cal_t only; raw Y and 0.83 never quoted), 11:00Z (Welch SE/df for Y). T100 = port E10 s0-s2 (final_table_lines/a5_t100_port.json, ref-0219).
- ENV: the E6 pool and T100 are both torch 2.7.0 on A100, so Y is within-env. A3 and A4 are within-2.9.1 gains divided by a 2.7 Y. A2 is cross-env (arm 2.7 vs ref 2.9.1; P-F2d env interval applies).
- A3 uses the ORIGINAL dev reps (+0.1145 / +0.1116) per critic 05:13Z ("DEV rows A3/A4 stay on the original reps"). The retrained set (+0.1124 / +0.1112) gives 1.44 / 1.45; the critic's 10:44Z line quoted that version.
- SUPERSEDED for intervals by a5_xfactor_final.md: that interval includes the reference spread (wider). Point values are unchanged.
- Owed: the overlap-removed column (critic 10:46Z, row 44). It needs g and Y re-scored on the DEV subset with train_100M-verbatim lines dropped. Not computed here. Any "N x" sentence must use the smaller lower bound once it exists.
- 2x NOT MET on DEV at either size. The best gain is about half the 2x bar (0.215 / 0.208).
