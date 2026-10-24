# 28L claim rep2 read (executor 3, 2026-10-11 00:45Z)
calib_set.py (reads/k2_estar; calib_control @ref-0033; one shared T per mixture), torch 2.9.1 solo for every member.
Members: 28L e5 s0 (menu F2 @ref-0073), e5 s1 (@ref-0074), e5 s2/s3 + e6 s1 (rep2 VM bl-ms-28l-claim-rep2-1010-214308, @ref-0169), e6 s0 (menu F2).
Singles cal_t full/guarded: e5 s0 4.9008/4.8761, s1 4.9127/4.8863, s2 4.9124/4.8858, s3 4.9069/4.8802; e6 s0 4.8975/4.8721, s1 4.9029/4.8761.
Mixtures cal_t: M28a (e5 s0+s1) 4.7873/4.7640; M28b (e5 s2+s3) 4.7896/4.7653.

| M28 | full raw | full cal_t | guarded raw | guarded cal_t |
|---|---|---|---|---|
| rep2 = mean(e6 s0,s1) - M28b | 0.1221 | 0.1105 | 0.1200 | 0.1088 |
| rep2 vs e6 s0 only | 0.1196 | 0.1078 | 0.1183 | 0.1068 |
| rep2 vs e6 s1 only | 0.1245 | 0.1133 | 0.1217 | 0.1108 |
| rep1 = e6 s0 - M28a | 0.1208 | 0.1101 | 0.1183 | 0.1081 |
| rep1 vs mean(e6 s0,s1) | 0.1232 | 0.1128 | 0.1200 | 0.1101 |

Gate (critic ref-0387, scheduling only): cal_t >= +0.0872 / +0.0881 -> HOLDS (+0.1105 / +0.1088).
