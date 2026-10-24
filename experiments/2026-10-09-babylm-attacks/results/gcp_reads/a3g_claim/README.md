# A3g claim read (executor 3, 2026-10-10 22:20Z)
VM bl-a3g-claim-1010-201551 (project-b, a2-highgpu-1g, solo serial; instrument.json: solo, torch 2.7.0+cu126, 1 GPU).
Arm: a3a E10 s1-s3 (jobs_a3g_claim_S_a3a_e10_s123.json @ref-0075). calib_set.py (reads/k2_estar) = calib_control @ref-0033.

| run | full raw | full cal_t | guarded raw | guarded cal_t | T | BLiMP-fast |
|---|---|---|---|---|---|---|
| a3a E10 s1 | 4.9332 | 4.9060 | 4.8975 | 4.8731 | 1.10 | 70.74 |
| a3a E10 s2 | 4.9321 | 4.9079 | 4.9006 | 4.8786 | 1.10 | 69.43 |
| a3a E10 s3 | 4.9450 | 4.9169 | 4.9120 | 4.8863 | 1.10 | 69.67 |
| arm mean (SD) | 4.9368 | 4.9103 (0.0058) | 4.9034 | 4.8793 (0.0067) | | |

vs E6 solo pool s0-s4 (2.7; cal_t 4.9534 SD 0.0056 / 4.9251 SD 0.0032; raw 4.9693 / 4.9398):
- gain cal_t +0.0431 / +0.0458; raw +0.0325 / +0.0364
- claim floor 2*SD_pool cal_t 0.0112 / 0.0064; arm-variance 2*sqrt(sdA^2/3+sdP^2/5) 0.0084 / 0.0082 -> CLEARS
- BLiMP-fast 69.4-70.7 vs base 68.4: no drop

vs R4.4 F~2 entry, 28L/384d s0 (torch 2.9.1, menu F2 VM bl-ms-menu-f2-1010-140927, _dumps_v2), cal_t full / guarded:
- e4 4.9076 / 4.8833, e5 4.9008 / 4.8761, e6 4.8975 / 4.8721 (best)
- arm minus best 28L: -0.0128 / -0.0072 (A3g WORSE). Cross-env (measured 2.7-2.9 shift ~+0.002 cal_t); 28L is a single seed (28L e6 s1 lands in rep2).
