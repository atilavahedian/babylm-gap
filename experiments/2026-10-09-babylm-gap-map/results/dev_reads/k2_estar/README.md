# K=2 legal split vs E6 pool, A3g stacking, F2 lane B, M_S (executor 3, 2026-10-10 21:10Z)

Tool: calib_set.py (this dir) wrapping calib_control @ref-0033 (one shared T per mixture), run from the
git-archive ref-0153 tree on dev_bpw_dump --twin --temps @ref-0034 dumps. Raw output: calib_set_out.json.
All members solo, torch 2.7 (e5 s0-s5, e6 s0-s4, a3a E5 s0-s1). Within one env.

## K=2 (2 x e5) replicates vs the E6 solo pool (s0-s4)
| leg / metric | E6 pool mean (SD) | p01 / p23 / p45 | rep mean (SD) | gain vs E6 pool |
|---|---|---|---|---|
| full raw | 4.9693 (0.0057) | 4.8443 / 4.8511 / 4.8430 | 4.8461 (0.0044) | +0.1232 |
| full cal_t | 4.9534 (0.0056) | 4.8394 / 4.8462 / 4.8374 | 4.8410 (0.0046) | +0.1124 |
| guarded raw | 4.9398 (0.0031) | 4.8168 / 4.8235 / 4.8184 | 4.8196 (0.0035) | +0.1202 |
| guarded cal_t | 4.9251 (0.0032) | 4.8128 / 4.8192 / 4.8133 | 4.8151 (0.0036) | +0.1100 |
All 15 e5 pairs: cal_t full 4.8418 (SD 0.0052), guarded 4.8157 (0.0041).

## M_S (critic ref-0154) = DEV(E6 s0) - DEV(p01 = e5 s0+s1 mix)
raw +0.1193 / +0.1199; cal_t +0.1072 / +0.1081 (full / guarded).

## A3g stacking (a3a E5 vs plain e5, same seeds; kill <0.0074, pass >=0.020 on cal_t both legs)
- singles cal_t: s0 +0.0290 / +0.0272, s1 +0.0349 / +0.0331
- A3K2 (a3a s0+s1) vs p01: raw +0.0251 / +0.0224, cal_t +0.0203 / +0.0184 (A3K2 picks T 1.00)
- A3K2 vs 15-pair pool mean: cal_t +0.0226 / +0.0213

## F2 lane B (hetero pairs hB1 = e5s0+a3as1, hB2 = e5s1+a3as0)
- mix cal_t full/guarded: hB1 4.8143/4.7894, hB2 4.8159/4.7910, A3K2 4.8192/4.7944
- hetero vs homogeneous A3K2: +0.0048/+0.0050 (hB1), +0.0032/+0.0034 (hB2)
- g vs member mean (cal_t full): hB1 0.1260, hB2 0.1273, A3K2 0.1066, seed pairs 0.1181 (SD 0.0020)
- mix vs BETTER member (cal_t full): hB1 0.1135, p01 0.1134: hetero adds no diversity beyond member quality.
