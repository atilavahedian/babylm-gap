| model | BLiMP | Supplement | EWoK | COMPS | WUG_adj | WUG_past | Reading_rt | Reading_spr | Entity | composite | GLUE7 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixS_p01 | 68.34 | 58.98 | 49.96 | 53.89 | 45.00 | 19.00 | 0.16 | 0.00 | 25.32 | 36.92 | 67.29 |  |
| mixS_p23 | 69.61 | 58.28 | 50.99 | 53.70 | 44.00 | 25.00 | 0.21 | 0.00 | 28.19 | 37.72 | 67.97 |  |
| mixS_p45 | 67.95 | 58.14 | 50.91 | 54.08 | 49.00 | 21.00 | 0.18 | 0.01 | 25.89 | 37.66 | 68.28 |  |
| S_e6_s0 | 67.24 | 57.20 | 50.47 | 54.03 | 40.00 | 22.00 | 0.18 | 0.00 | 23.33 | 36.39 | 66.51 |  |
| S_e6_s1 | 66.82 | 55.91 | 50.22 | 53.86 | 38.00 | 10.00 | 0.14 | 0.00 | 21.70 | 34.37 | 66.23 |  |
| S_e6_s2 | 68.01 | 58.44 | 51.09 | 54.16 | 44.00 | 23.00 | 0.19 | 0.08 | 23.80 | 37.37 | 66.95 |  |

| arm | n (comp/GLUE) | composite mean (sd) | GLUE7 mean (sd) | Entity mean | claim |
|---|---|---|---|---|---|
| S_split | 3/3 | 37.43 (0.45) | 67.85 (0.51) | 26.47 | ok |
| S_E6 | 3/3 | 36.04 (1.53) | 66.56 (0.37) | 22.94 | ok |
- S_split vs S_E6, composite: diff 1.39, bar 1.84 -> does not beat
- S_split vs S_E6, GLUE7: diff 1.29, bar 0.72 -> BEATS

Amended after data (critic 81702719): Reading with add_prefix_space=True, same weights; never changes a verdict

| model | Reading_rt (amended) | Reading_spr (amended) | composite (amended) |
|---|---|---|---|
| mixS_p01 | 8.27 | 3.13 | 38.32 |
| mixS_p23 | 8.74 | 2.96 | 39.16 |
| mixS_p45 | 8.90 | 3.17 | 39.14 |
| S_e6_s0 | 7.27 | 2.95 | 37.65 |
| S_e6_s1 | 8.72 | 3.28 | 35.85 |
| S_e6_s2 | 8.14 | 2.50 | 38.67 |

| arm | n | composite amended mean (sd) |
|---|---|---|
| S_split | 3 | 38.88 (0.48) |
| S_E6 | 3 | 37.39 (1.43) |
- S_split vs S_E6, composite (amended): diff 1.49, bar 1.74 -> amended, no claim

Notes (ruler):
- Sources: byi6gg final_i_s_b1 (ref-0393 tree), final_i_s_b2, final_i_s_split_fix, final_i_s_split_fix2. All official @ bf55c1131e53654c2a87418f5629c26959acd710, transformers 4.51.3.
- Same 8-task composite as zs_28L_mix_vs_single (Entity reported only). Bar = 2*sqrt(sdA^2/nA + sdR^2/nR).
- GLUE7: lastpool, ens2 (seeds 42+43); for mixtures m0 seed 42 and m1 seed 43. S_split provenance: p01 ran sequential and alone on B1 (eval_finetuning.sh; B1 jobs strictly serial, p01 13:16:37-15:55:50Z, next job 15:55:57Z); p23 (fix2) spot IDENTICAL; p45 multirc rerun solo (critic ref-0191). Corrected 23:33Z: an earlier draft wrongly said p01 ran packed.
- S_split = two 5-epoch members averaged (~1.67x the pretraining compute of one E6 single). GLUE7 BEATS is NOT compute-matched; the compute-matched evidence (F4) is dev-loss only.
- WUG_past (n=100 items) drives most composite spread (S_e6_s1 = 10.0).
