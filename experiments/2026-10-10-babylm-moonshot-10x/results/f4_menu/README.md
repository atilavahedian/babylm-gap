# F4 menu reads (executor 3)
calib_set.py (reads/k2_estar; calib_control @ref-0033), twin dumps ref-0034; torch 2.9.1 solo (instrument.json); job ms_menu_f4_a.json @ref-0170 (VM bl-ms-menu-f4-a-1011-061030), keep_checkpoints true (weights + score/hf on GCS).

## F4 a: deep 60L/384d (2026-10-11 10:25Z)
| cell | twin full / guarded (raw) | cal_t full / guarded | T |
|---|---|---|---|
| 60L e4 | 4.9128 / 4.8886 | 4.9028 / 4.8795 | 1.05 |
| 60L e5 | 4.9195 / 4.8944 | 4.8966 / 4.8732 | 1.10 |
| 60L e6 | 4.9043 / 4.8789 | 4.8852 / 4.8607 | 1.05 |

Best-of-3 (counted trials) = e6: cal_t 4.8852 / 4.8607. For reference, 28L K=2 mixtures cal_t M28a 4.7873/4.7640, M28b 4.7896/4.7653, M28c 4.7835/4.7615 (reads/ls_28l/rep3).

## F4 b: balanced 36L/512d e4-e6 + wide 12L/880d e4 (2026-10-11 10:55Z)
Job ms_menu_f4_b.json @ref-0170, 2.9.1 solo, keep_checkpoints.
| cell | twin full / guarded (raw) | cal_t full / guarded | T |
|---|---|---|---|
| 36L e4 | 4.8862 / 4.8584 | 4.8728 / 4.8461 | 1.05 |
| 36L e5 | 4.9058 / 4.8777 | 4.8749 / 4.8488 | 1.10 |
| 36L e6 | 4.9070 / 4.8765 | 4.8761 / 4.8478 | 1.10 |
| 12L/880d e4 | 4.8938 / 4.8692 | 4.8758 / 4.8524 | 1.05 |

## Full F4 menu (all 7 registered cells landed; wide last per critic ref-0346)
Best cell = 36L/512d e4: raw 4.8862 / 4.8584, cal_t 4.8728 / 4.8461 (best of 7 counted trials).
28L K=2 mixture cal_t mean (3 reps) 4.7868 / 4.7636 -> split beats the best F4 cell by ~0.086 full / ~0.083 guarded cal_t (exec arithmetic; the registered comparison and FLOPs factor are moonshot's/critic's).
