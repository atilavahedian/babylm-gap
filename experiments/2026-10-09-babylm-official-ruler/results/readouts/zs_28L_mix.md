# 28L K=2 mixtures: official zero-shot readout (provisional, no GLUE yet)

Source: executor 3 sync babylm/results/final_ii_28l_zs @ref-0357 (branch wip-branch), VM bl-final-ii-28l-zs-1011-151920, code ref-0173. Guards: all 3 PASS (|diff| <= 1.63e-09). Readout: babylm/eval/official_readout.py unchanged. Composite = mean of 8 zero-shot tasks (critic ref-0185), Entity reported only.

| model | BLiMP | Supplement | EWoK | COMPS | WUG_adj | WUG_past | Reading_rt | Reading_spr | Entity | composite | GLUE7 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mix28L_p01 | 69.72 | 57.87 | 51.22 | 53.59 | 42.00 | 12.00 | 0.17 | 0.00 | 23.41 | 35.82 |  | no GLUE7 (mix28L_p01_ens2: 0/7) |
| mix28L_p23 | 69.71 | 59.70 | 51.58 | 53.99 | 48.00 | 18.00 | 0.23 | 0.02 | 27.50 | 37.65 |  | no GLUE7 (mix28L_p23_ens2: 0/7) |
| mix28L_p45 | 70.62 | 60.54 | 50.32 | 54.01 | 55.00 | 23.00 | 0.29 | 0.00 | 30.07 | 39.22 |  | no GLUE7 (mix28L_p45_ens2: 0/7) |

| arm | n (comp/GLUE) | composite mean (sd) | GLUE7 mean (sd) | Entity mean | claim |
|---|---|---|---|---|---|
| mix28L | 3/0 | 37.57 (1.70) |  () | 26.99 | no claim (n<3) |
