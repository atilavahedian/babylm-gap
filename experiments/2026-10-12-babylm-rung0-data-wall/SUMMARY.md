# Data-wall rung 0: heavy regularisation and looped depth at 16 epochs

**Status:** closed · **Dates:** 2026-10-12

## Question
Labs at the data wall have a fixed corpus and spare compute. Two known ideas were tested on our 10M-word recipe past
BabyLM's 10-epoch cap: (Q1) is the recipe under-regularised once it trains 16 epochs, so that heavier weight decay or a
re-tuned learning rate wins; (Q2) does reusing depth (12 layers looped twice, shared weights) beat a plain model of
equal FLOPs? Out of track (16 epochs > 10); a wall question only.

## Method
- Size S, train_10M, 16 epochs, seed 0, solo A100, torch 2.7.0, locked recipe (tied output, no value embeddings,
  lr_scale 0.5) unless varied; per run: score_trial, v2 twin, temperature dump, CSV + PNG curves.
- Q1 arms: control C (Muon WD 0.4) vs W in {0.2, 0.8, 1.6, 3.2}; then one LR cell (lr_scale 1.0) at W* = argmin over W.
  The pick set excludes C (critic ref-0371).
- Q2 arms: 12L looped x2 (n_loops=2: shared blocks over effective depth 24, x0/value-embedding/U-net layout over the
  effective depth) vs plain 24L; both 349.2M FLOPs/token, identical words (10,418,979) and tokens (16,542,003) to C.
- Pre-registered (critic ref-0368, before any run): alive if the Q1 pick beats C, or looped beats its plain FLOPs twin,
  by the 3-seed noise bar on both cal_t legs; screen kill 0.0074 on the s0 screen before using compute on extra seeds
  (critic ref-0371); otherwise dead and added to the paper's failed-attacks table.
- Metric: cal_t on full / guarded DEV from the v2 twin temperature dumps (calib_control via calib_set.py).

## Results
| Cell (S, 16 epochs, s0) | cal_t full | cal_t guarded | vs control C (full / guarded; + = worse; W0.2, W1.6, W3.2, LR differences computed here from the 4-decimal values) |
|---|---|---|---|
| C, WD 0.4 | 5.0112 | 4.9806 | - |
| W 0.2 | 5.0820 | 5.0431 | +0.0708 / +0.0625 |
| W 0.8 (W*) | 5.0677 | 5.0381 | +0.0565 / +0.0575 |
| W 1.6 | 5.2031 | 5.1763 | +0.1919 / +0.1957 |
| W 3.2 | 5.3669 | 5.3408 | +0.3557 / +0.3602 |
| LR x2 at W 0.8 | 5.2137 | 5.1876 | +0.2025 / +0.2070 |
| 12L looped x2 | 5.0256 | 4.9924 | +0.0144 / +0.0118 (and +0.0381 / +0.0382 vs plain 24L) |
| plain 24L (FLOPs twin) | 4.9875 | 4.9542 | -0.0237 / -0.0264 (1.8x FLOPs and params) |
| reference: 6 epochs, E6 pool (5 seeds) | 4.9534 | 4.9251 | C at 16 epochs is +0.058 / +0.055 worse |

| Question | Measured | Kill line | Verdict |
|---|---|---|---|
| Q1: W* beats C | W* worse by 0.0565 / 0.0575 | 0.0074 | DEAD (critic ref-0208) |
| Q2: looped beats plain 24L | looped worse by 0.0381 / 0.0382 | 0.0074 | DEAD (critic ref-0211) |

## What it rules out
- That our 10M recipe is under-regularised at 16 epochs: WD 0.4 is the best of five values (U-shaped grid), and
  doubling the learning rate at the best alternative makes it worse still.
- That weight-shared looped depth buys data efficiency at equal FLOPs on this data: it loses to the plain FLOPs twin
  and even to the 12-layer control with the same parameters.
- That more passes over 10M words help this recipe: 16 epochs lose 0.058 to 6 epochs.

## What it does not rule out / caveats
- Single-seed screens, as registered; the confirming seeds were skipped once both screens killed (margins are 5-8x
  the kill line).
- One size (S), one corpus, one schedule length (16 epochs); other regularisers (dropout, data augmentation) and
  looped variants with per-loop embeddings or adaptive depth were not tested.
- Plain 24L does beat C (ordinary scale at 1.8x FLOPs and params) but is still 0.034 / 0.029 worse than the 6-epoch
  12L pool; it is not a data-efficiency lever.
- `results/summary/rung0_final_scores.csv` leaves the LR cell's cal_t blank ("pending"); the value above (5.2137 /
  5.1876) is from `results/calib_lr.json`, the file critic ref-0208 verified. The paper's table prints only its raw 5.244.
- The registration's design source ("design agent 3 memo_data_wall.md, section 5") is not in any git branch; the binding
  text is critic entry ref-0368 (`governance/critic_log_rung0.md`).

## References
- [governance/critic_log_rung0.md](governance/critic_log_rung0.md): registration, interim reads and the final ruling (verbatim).
- [results/summary/rung0_final_scores.csv](results/summary/rung0_final_scores.csv), [results/summary/rung0_curves.png](results/summary/rung0_curves.png), [results/summary/rung0_curves.csv](results/summary/rung0_curves.csv).
- [results/calib_s1a.json](results/calib_s1a.json), [results/calib_s1b.json](results/calib_s1b.json), [results/calib_lr.json](results/calib_lr.json), [results/calib_lx.json](results/calib_lx.json); per-run `results/dw0_*/`.
- Experimenter report section 4 and paper section "Beyond the BabyLM cap": [../2026-10-09-babylm-gap-map/docs/FINAL_REPORT_DRAFT.md](../2026-10-09-babylm-gap-map/docs/FINAL_REPORT_DRAFT.md), [../../papers/babylm-gap-map/main.tex](../../papers/babylm-gap-map/main.tex).
- Code: `common/babylm/models/looped.py`, `common/babylm/tests/test_looped.py`, `common/babylm/gcp/dw/make_dw_r0_jobs.py`.
