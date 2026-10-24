# v2f (E10 s1, torch 2.7, solo) vs the torch 2.9.1 solo E10 pool (executor 3, 2026-10-10 21:25Z)
calib_set.py (reads/k2_estar) on dev_v2.npz from bl-exp-p2v2f-1010-193702. Pool cal_t from calib_v2e_vs_pool.txt.
- v2f: full raw 5.0005 cal_t 4.9668 (T 1.10); guarded raw 4.9676 cal_t 4.9363 (T 1.10)
- 2.9 s1 minus v2f (positive = 2.7 better): full raw -0.0009 cal_t -0.0006; guarded raw +0.0010 cal_t +0.0015
- pool mean (4.9670 / 4.9386) minus v2f: cal_t +0.0002 / +0.0023
- with v2e (s0): same-seed 2.7 minus 2.9 cal_t shifts are +0.0050/+0.0034 (s0) and -0.0006/+0.0015 (s1): mean +0.0022/+0.0025.
