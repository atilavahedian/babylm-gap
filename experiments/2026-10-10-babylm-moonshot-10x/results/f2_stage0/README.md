# F2 Stage 0 lane A (executor 3, 2026-10-10 18:50Z)
All members torch 2.9.1, E10, temps dumps. Tools: hetero_pairs.sh @ref-0326, ceiling_report @ref-0153, seed ref script seedref/c3_pairref.py.

Hetero pairs (laneA/summary.txt), g(2) full/guarded:
- c3s1_a2a: raw 0.1564/0.1523, cal_t 0.1390/0.1359, mix DEV 4.8376/4.8136
- c3s2_a2a: raw 0.1566/0.1529, cal_t 0.1399/0.1372, mix DEV 4.8370/4.8126

Seed reference (not a trial):
- raw g(2), 28 C3 pairs (f1 _dumps_v2): full 0.1535 sd 0.0027; guarded 0.1497 sd 0.0024
  - solo-solo (s0-s2, n 3) 0.1517/0.1488; packed-packed (ms_c3 s3-s7, n 10) 0.1549/0.1508; mixed (n 15) 0.1529/0.1492. Solo vs packed gap 0.0032/0.0020 (< 0.005).
- cal_t g(2), 10 pairs of solo base_r1 s0-s4 (temps back-fill; s5-s7 have no temps dumps): full 0.1331 sd 0.0024; guarded 0.1313 sd 0.0022. Seed mix DEV raw 4.8472/4.8195.

Registered kill rule: d = g_cal(hetero) - g_cal(seed) >= 0.03 on both legs AND hetero mix DEV beats seed mix by >= 0.016.
- d (cal_t): c3s1_a2a +0.0059/+0.0046; c3s2_a2a +0.0068/+0.0059. Mix DEV gain: 0.0096/0.0059 and 0.0102/0.0069.
- Both conditions fail on both pairs.
