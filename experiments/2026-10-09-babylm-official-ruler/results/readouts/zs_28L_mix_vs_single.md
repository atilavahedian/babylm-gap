# 28L: K=2 mixtures vs single models, official zero-shot (no GLUE yet)

Sources: executor 3 syncs on wip-branch: final_ii_28l_zs @ref-0357 (VM bl-final-ii-28l-zs-1011-151920; 3 twin guards PASS) and final_ii_28l_zs_singles @ref-0358 (VM bl-final-ii-28l-zs-singles-1011-170730). Code ref-0173, official_readout.py unchanged. Composite = mean of 8 zero-shot tasks (critic ref-0185), Entity reported only; beats-bar = 2*sqrt(sdA^2/nA + sdR^2/nR). 'no claim (n<3)' in the arm table refers to the missing GLUE7 column.

| model | BLiMP | Supplement | EWoK | COMPS | WUG_adj | WUG_past | Reading_rt | Reading_spr | Entity | composite | GLUE7 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mix28L_p01 | 69.72 | 57.87 | 51.22 | 53.59 | 42.00 | 12.00 | 0.17 | 0.00 | 23.41 | 35.82 |  | no GLUE7 (mix28L_p01_ens2: 0/7) |
| mix28L_p23 | 69.71 | 59.70 | 51.58 | 53.99 | 48.00 | 18.00 | 0.23 | 0.02 | 27.50 | 37.65 |  | no GLUE7 (mix28L_p23_ens2: 0/7) |
| mix28L_p45 | 70.62 | 60.54 | 50.32 | 54.01 | 55.00 | 23.00 | 0.29 | 0.00 | 30.07 | 39.22 |  | no GLUE7 (mix28L_p45_ens2: 0/7) |
| L28_e6_s0 | 69.96 | 59.47 | 50.94 | 53.90 | 45.00 | 15.00 | 0.13 | 0.00 | 23.34 | 36.80 |  | no GLUE7 (L28_e6_s0_lastpool_ens2: 0/7) |
| L28_e6_s1 | 69.23 | 58.64 | 50.31 | 53.53 | 37.00 | 19.00 | 0.15 | 0.00 | 18.84 | 35.98 |  | no GLUE7 (L28_e6_s1_lastpool_ens2: 0/7) |
| L28_e6_s2 | 68.22 | 60.36 | 51.46 | 53.99 | 38.00 | 22.00 | 0.19 | 0.01 | 21.51 | 36.78 |  | no GLUE7 (L28_e6_s2_lastpool_ens2: 0/7) |

| arm | n (comp/GLUE) | composite mean (sd) | GLUE7 mean (sd) | Entity mean | claim |
|---|---|---|---|---|---|
| mix28L | 3/0 | 37.57 (1.70) |  () | 26.99 | no claim (n<3) |
| single28L | 3/0 | 36.52 (0.47) |  () | 21.23 | no claim (n<3) |
- mix28L vs single28L, composite: diff 1.05, bar 2.04 -> does not beat
- mix28L vs single28L, GLUE7: diff , bar  -> no claim (n<3)

Amended after data (critic 81702719): Reading with add_prefix_space=True, same weights; never changes a verdict

| model | Reading_rt (amended) | Reading_spr (amended) | composite (amended) |
|---|---|---|---|
| mix28L_p01 | 8.33 | 2.82 | 37.19 |
| mix28L_p23 | 8.70 | 3.22 | 39.11 |
| mix28L_p45 | 8.11 | 3.17 | 40.60 |
| L28_e6_s0 | 8.02 | 3.19 | 38.18 |
| L28_e6_s1 | 7.82 | 3.08 | 37.33 |
| L28_e6_s2 | 8.11 | 3.28 | 38.18 |

| arm | n | composite amended mean (sd) |
|---|---|---|
| mix28L | 3 | 38.97 (1.71) |
| single28L | 3 | 37.90 (0.49) |
- mix28L vs single28L, composite (amended): diff 1.07, bar 2.05 -> amended, no claim

Amended section added 23:55Z from final_ii_28l_z1r (byi6gg ref-0392, VM bl-final-ii-28l-z1r-1011-190119; alignment check per critic 81702719). Registered rows above re-derived byte-identical.
