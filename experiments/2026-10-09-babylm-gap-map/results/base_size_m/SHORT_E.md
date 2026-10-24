# Size M short-E cells (executor 3, 2026-10-10 21:45Z; VM bl-base-size-m-shorte-1010-191937, project-c, solo, system torch 2.9.1, job @ref-0167)
Twin dev_bpw_twin (ref-0026) full / guarded (ex_unseen_chars):
- M wd1.6 e5: 5.1747 / 5.1492 (BLiMP-fast in score.json)
- M wd3.2 e5: 5.3577 / 5.3334
- M wd1.6 e3: 5.1745 / 5.1502
Reference (earlier, E10 lr0.5): M wd0.4 5.1323 / 5.0912, M wd0.8 5.1059 / 5.0728. S E6 (2.7) 4.9635 / 4.9367.
Higher WD hurts M; shortening E at wd1.6 does not recover it. Best M cell is still E10 wd0.8, 0.14 worse than S E6.
