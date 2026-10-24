# 28L claim rep3 read, 3-rep table (executor 3, 2026-10-11 06:20Z)
calib_set.py (reads/k2_estar; calib_control @ref-0033; one shared T per mixture). Every member: torch 2.9.1, solo (instrument.json).
Members: rep1 e5 s0 (menu F2 @ref-0073), e5 s1 (@ref-0074), e6 s0 (menu F2); rep2 e5 s2/s3 + e6 s1 (bl-ms-28l-claim-rep2-1010-214308, @ref-0169); rep3 e5 s4/s5 + e6 s2 (bl-ms-28l-claim-rep3-1011-034552, relaunch after a 03:43Z preemption, @ref-0170).
Spec (moonshot): each rep mixture vs the mean of ALL 28L e6 singles (s0, s1, s2); raw and cal_t, both legs; 3-rep mean. Gain = reference - mixture (nats/word, positive = mixture better).

| rep (mix) | full raw | full cal_t | guarded raw | guarded cal_t |
|---|---|---|---|---|
| rep1 (e5 s0+s1) | +0.1245 | +0.1140 | +0.1213 | +0.1112 |
| rep2 (e5 s2+s3) | +0.1234 | +0.1117 | +0.1213 | +0.1099 |
| rep3 (e5 s4+s5) | +0.1292 | +0.1178 | +0.1245 | +0.1137 |
| **3-rep mean** | +0.1257 | +0.1145 | +0.1224 | +0.1116 |

Reference = mean of 28L e6 s0, s1, s2 singles: full raw 4.9195 cal_t 4.9013; guarded raw 4.8924 cal_t 4.8752.
Mixtures cal_t full/guarded: M28a 4.7873/4.7640; M28b 4.7896/4.7653; M28c 4.7835/4.7615
Singles cal_t full/guarded: e5s0 4.9008/4.8761; e5s1 4.9127/4.8863; e5s2 4.9124/4.8858; e5s3 4.9069/4.8802; e6s0 4.8975/4.8721; e6s1 4.9029/4.8761; e5s4 4.9047/4.8829; e5s5 4.9016/4.8741; e6s2 4.9035/4.8773

Gate (critic ref-0387): cal_t >= +0.0872 full / +0.0881 guarded per rep -> all three reps clear. S claim for comparison (critic ref-0149): +0.112 / +0.110 cal_t vs the S E6 pool. Rep2 differs from its 00:45Z line (+0.1105/+0.1088) only because the reference now includes e6 s2.
