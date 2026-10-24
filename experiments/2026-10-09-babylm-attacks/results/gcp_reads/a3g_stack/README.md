# A3g stack, 3 whole-mixture replicates (executor 3, 2026-10-11 01:10Z; registered path critic ref-0076)
calib_set.py (reads/k2_estar; calib_control @ref-0033; one shared T per mixture). All members solo, p2venv torch 2.7.0.
Rep 1: a3a E5 s0+s1 vs p01 (reads/k2_estar). Reps 2,3: VM bl-a3g-stack-r23-1010-220345 (a3a E5 s2-s5, @ref-0078), vs p23 / p45.
Re-read e5 s2-s5 singles and p23/p45 match k2_estar to 1e-5.

| leg / metric | rep1 (p01-A3K2) | rep2 (p23-A3K2b) | rep3 (p45-A3K2c) | mean (SD) |
|---|---|---|---|---|
| full raw | +0.0251 | +0.0205 | +0.0112 | +0.0189 (0.0071) |
| full cal_t | +0.0203 | +0.0156 | +0.0054 | +0.0138 (0.0076) |
| guarded raw | +0.0224 | +0.0178 | +0.0121 | +0.0174 (0.0052) |
| guarded cal_t | +0.0184 | +0.0134 | +0.0070 | +0.0129 (0.0057) |

A3K2 mixtures cal_t full/guarded: s0+s1 4.8192/4.7944, s2+s3 4.8306/4.8058, s4+s5 4.8319/4.8063 (all pick T 1.00).
Same-seed singles a3a E5 minus plain e5, cal_t full: s0 .0290 s1 .0349 s2 .0293 s3 .0244 s4 .0094 s5 .0262 (mean .0255); guarded mean .0239.
BLiMP-fast a3a s2-s5: 68.0 / 68.6 / 66.2 / 69.5.
