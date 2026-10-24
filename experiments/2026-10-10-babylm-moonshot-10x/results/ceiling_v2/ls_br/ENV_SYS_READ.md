# Branch split (trunk e4 + 2 children; torch 2.9.1) under ENV_SYS (executor 3, 2026-10-10 21:35Z)
calib_set.py (reads/k2_estar) on ms_ls_br4k2_c1/c2 dev_v2.npz (bl-ms-legal-split-1010-113631). One shared T.
- BR mixture: full raw 4.8934 cal_t 4.8783; guarded raw 4.8665 cal_t 4.8524 (T 1.05)
- members cal_t full 4.9539 / 4.9538; guarded 4.9277 / 4.9270 (T 1.10)
- vs E6 pool (2.7, cal_t 4.9534 / 4.9251): +0.0751 / +0.0727; minus ENV_SYS 0.014 = +0.061 / +0.059 -> positive split
- vs plain K=2 3-rep mean (2.7, cal_t 4.8410 / 4.8151): -0.0373 / -0.0373 (branch worse; |d| > ENV_SYS 0.014)
- measured env shift (v2e, v2f) mean +0.002 on cal_t, so the cross-env sign cannot flip.
Verdict: branch is dominated by plain K=2.
