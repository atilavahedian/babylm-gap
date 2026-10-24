# BabyLM data efficiency: final report (2026-10-13)

Status: all runs closed and every result read (critic ref-0179: nothing pending). Numbers carry
their git/critic references. Do not quote this draft outside the project.

## 1. Answer

- **Best legal result: train two models on the same 10M words for 5 epochs each and average their predictions (the
  K=2 "split").** At our base size (12 layers, S) it beats our best single model by **+0.112 / +0.110 nats/word**
  (calibrated, full / guarded DEV, 3 replicates; critic ref-0149). That is **x ≥ 1.37 (point 1.44)** data-equivalent (A5 final, below).
- **Against a single model given the same compute it is worth ~0.06 nats/word, x ≈ 1.1 (point 1.21)** (P-F2d, critic ref-0186; crosses torch envs).
- **Second size (28 layers):** the split's gain is **+0.1145 / +0.1116** (3 reps: +0.114, +0.112, +0.118 full; executor
  ref-0203) vs +0.112 / +0.110 at S: it holds, flat within noise (critic ref-0187), x ≥ 1.39 (point 1.45) same-size.
  Against the best of 7 equal-compute singles at 28L (F4 menu, best = 36L/512d e4): **+0.086 / +0.083**, x ≥ 1.22 (point 1.32), vs
  +0.058 / +0.059 (x ≈ 1.1) at S (executor ref-0204; critic ref-0182 PASS, bar 0.012). The compute-matched edge grows with
  size because a bigger single stops paying: the best F4 cell beats the 28L single by only 0.03.
- **Official BabyLM scores:** zero-shot composite is a tie at both sizes (S +1.39 vs bar 1.84; 28L +1.05 vs bar 2.04).
  GLUE7 at S beats the S single (+1.29 vs bar 0.72) but TIES the compute-matched 28L E6 single (67.85 vs 67.22,
  +0.63 vs bar 0.75; ruler ref-0205, critic ref-0188 FINAL): a compute effect, not data efficiency. 28L GLUE7 split vs single 67.10 vs 67.22, -0.11 vs bar 1.09:
  tie (ruler ref-0206, critic ref-0179 FINAL). The 28L split ties its single on every official column.
- **Best 10M number:** 28L split mixture 4.795 nats/word raw, 0.17 below our base, at ~4x its compute.
- **This is a replication of a known idea** (data-split / seed ensembles under a fixed exposure budget), not a
  discovery.
- **10x is not reachable on this data under the rules; 2x is unlikely.** Legal reach estimate 1.5-1.8x (design agent memo,
  PR #68); a perfect learner of everything 10M can teach still tops out near ~4x (D1 bias map: 0.45 of a measured 0.743 gap;
  critic ref-0180; earlier drafts said ~3x on the retired 0.83 yardstick).

Yardstick (A5 FINAL, ruler ref-0207 from executor ref-0176; critic ref-0181 / ref-0180): x = 10^(gain / Y), a
"DEV-loss data-equivalent multiplier (log-linear)". Y = S E6 pool (5 seeds) minus T100 (port recipe E10, 3 seeds) on
cal_t = **0.715 / 0.690** (Welch 95% [0.708, 0.722] / [0.686, 0.695]); 2x needs g ≥ 0.215 / 0.208. Every x is an UPPER
BOUND (100M point not exhaustively tuned) and quoted as the lower end of its 95% interval (df 2, reference spread
included), the smaller of all-DEV and overlap-removed DEV (134,739 words of train_100M-verbatim lines dropped). The 0.83
working value (raw, older metric) is retired.

## 2. Rules we judged by

- Track: BabyLM 2025 strict-small (10M words, ≤10 epochs; R0 counts every model's exposure, teachers and generators
  included, ≤100M; no outside pretrained weights). EWoK eval-only.
- DEV metric: v2 blank-line twin scorer, full and guarded legs (guarded drops 10,277 words with characters never seen
  in training). Claims gate on the calibrated score (T x lambda, doc-parity cross-fit). Both legs must pass.
- Reference: the E6 pool (base trained 6 epochs, the best epoch count; 5 seeds, solo, torch 2.7): 4.9693 raw
  (SD 0.0057). Claim floor = 2 SD on cal_t = 0.0111 / 0.0065.
- Base: 12L/384d, tied output, no value embeddings, weight decay 0.4. 10 epochs overtrain it; E6 beats E10 by 0.039.
- Systematics: torch 2.7 vs 2.9 ENV_SYS 0.012 / 0.011 (direction consistent with zero); packed vs solo PACK_SYS
  0.012 / 0.010. Claims are solo.
- The critic registered every bar before the result it judged and can veto.

## 3. DEV results (cal_t, full / guarded)

| Row | Comparison | Gain | x | Status |
|---|---|---|---|---|
| A1 | S split (3 reps) vs E6 pool | +0.112 / +0.110 | ≥1.37 (1.44) | PASS (ref-0149) |
| A2 | S split vs equal-compute single (28L e6, F=2 menu) | +0.058 / +0.059 | ≈1.1 (1.21); 3-seed ref, post-registration: ≥1.16 | HIT (ref-0186) |
| A3 | 28L split (3 reps) vs mean of 28L e6 s0-s2 | +0.1145 / +0.1116 (reps +0.114, +0.112, +0.118 full) | ≥1.39 (1.45); Y from S only | HOLDS, flat within noise (critic ref-0187) |
| A4 | 28L split vs equal-compute single (best of 7 F4 cells: 36L/512d e4, cal_t 4.8728 / 4.8461) | +0.086 / +0.083 (split mean 4.7868 / 4.7636) | ≥1.22 (1.32); Y from S only | PASS (critic ref-0182, bar 0.012) |
| A5 | Gain holds across sizes? (A3 ≥ M_S − 0.02, A4 clears its bar, A4 ≥ A2 − 0.02) | A3 0.1145 / 0.1116 ≥ M_S − 0.02; A4 clears 0.012; A4 0.086 ≥ A2 0.058 − 0.02 | | DEV rule MET (critic ref-0182). 2x NOT MET: highest interval upper end 1.52 |

## 4. What did not work (all closed)

| Lever | Result |
|---|---|
| A1 temporal self-distillation | Dropped before the screen (prior ~3%) |
| A2 input noise | Calibration-equivalent: the raw gain was overconfidence |
| A3/A3g BPE-dropout (segmentation noise) | +0.043 single at equal words, but loses to a deeper model at equal compute; +0.014 inside the split (bar 0.020) |
| A4 model writes its own contexts | 0.038 / 0.045 worse than its no-synthetic twin |
| F1 distillation from an 8-model ensemble | Plain self-distillation, no ensemble-diversity gain |
| Shared-trunk / branch ensembles | Lose diversity; branch is 0.037 worse than the plain split |
| Mixed-recipe split members (F2) | Add nothing beyond a seed change |
| K=3 split | Members lag (3.33-epoch schedule) |
| Wider models (12L/768d, early screen) | 0.14-0.2 worse than S at that screen's settings. Not general: at matched compute F4's 12L/880d e4 (cal_t 4.876) is within 0.003 of the best cell |
| Data-wall rung 0, re-tune at 16 epochs (outside the cap) | WD 0.4 already best of {0.2, 0.4, 0.8, 1.6, 3.2}; 2x LR at 0.8 worse still. Best alternative -0.057 / -0.058 cal_t vs control (kill 0.0074). Control at E16 is 0.058 / 0.055 worse than E6 (critic ref-0208; executor ref-0209) |
| Data-wall rung 0, looped depth (12L x 2 vs plain 24L, equal FLOPs) | -0.038 / -0.038 vs P24, -0.014 / -0.012 vs the 12L control; plain depth +0.024 / +0.026 (executor ref-0210). Rung 0 DEAD no rung 1 |

## 5. Official BabyLM scores

Protocol (critic 44552922 + ref-0212): zero-shot tasks score the per-token mixture exported as one causal model; GLUE7
uses last-non-pad-token pooling (BOS-pooled GLUE is reported only, it is near chance for causal models by
construction); each split member is fine-tuned separately and probabilities averaged, and every single-model comparator
gets the same 2-fine-tune-seed ensemble.

Pipeline certification (ruler, babylm/results/certify_v3/glue_certification_1011.md @ref-0213): the pinned pipeline
reproduces published GPT-BERT GLUE in 12 of 14 cells exactly to 2 decimals; the other two are MultiRC, off by 0.24
(68.52 vs 68.28) and 0.04 (67.08 vs 67.04). The published mixed "QQP F1" 76.56 matches our QQP accuracy (76.557), which
looks like a README inconsistency. Published GPT-2 under BOS pooling is majority-class on 7/7, as expected for a causal
model. Published GPT-2 last-token GLUE was cut at 2/7 tasks for budget; last-token pooling for our causal models
was certified by the critic instead (ref-0214), with the untrained model as a reference (62.6 on 6/7 tasks).

| Model (3 each) | BLiMP | Supplement | EWoK | COMPS | WUG_adj | WUG_past | Composite (8 tasks), mean (sd) | Amended composite | Entity | GLUE7 lastpool, mean (sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| S E6 single | 67.4 | 57.2 | 50.6 | 54.0 | 40.7 | 18.3 | 36.04 (1.53) | 37.39 (1.43) | 22.9 | 66.56 (0.37) |
| S split | 68.6 | 58.5 | 50.6 | 53.9 | 46.0 | 21.7 | 37.43 (0.45) | 38.88 (0.48) | 26.5 | 67.85 (0.51)* |
| 28L e6 single | 69.1 | 59.5 | 50.9 | 53.8 | 40.0 | 18.7 | 36.52 (0.47) | 37.90 (0.49) | 21.2 | 67.22 (0.41) |
| 28L split | 70.0 | 59.4 | 51.0 | 53.9 | 48.3 | 17.7 | 37.57 (1.70) | 38.97 (1.71) | 27.0 | 67.10 (0.86) |
| GPT-BERT causal-focus, published 10M (reference; ruler ref-0177) | 71.66 | 63.21 | 49.49 | 52.83 | 59.0 | 13.0 | 40.32† | — | 34.55 | 65.77‡ |
| GPT-BERT mixed, published 10M (reference; ruler ref-0177) | 69.62 | 61.56 | 50.23 | 52.99 | 49.0 | -3.0 | 36.66† | — | 25.41 | 66.10‡ |

† GPT-BERT's tokenizer adds the prefix space, so its composite compares with our AMENDED column: our best (38.88 / 38.97)
sits between the two published baselines, 1.4 below the stronger. ‡ Not comparable: official bidirectional whole-input
pooling, one fine-tune seed (ours: last-token, ens2). Our own 3-seed GPT-BERT reproduction was cut for budget.

\* S split GLUE cells carry the concurrent-run footnote (one member pair ran packed; identity mismatch on MultiRC only).
Amended composite = Reading tasks rescored with add_prefix_space=True, same weights, amended after data, no claim (critic
81702719). The 100M anchor is DEV-only (Section 9).

S official (ruler ref-0215): zero-shot composite +1.39, bar 1.84 → **does not beat** (amended +1.49, bar 1.74; WUG_past,
100 items, drives the spread: S_e6_s1 = 10.0). GLUE7 +1.29, bar 0.72 → **beats, but not compute-matched**: the split
used ~1.67x the pretraining of one E6 single. The compute-matched GLUE check (S split vs A2's 28L e6 single, registered before any 28L GLUE
cell was read) is a **tie**: 67.85 (0.51) vs 67.22 (0.41), n=3 each, +0.63 vs bar 0.75 (ruler ref-0205, critic ref-0188 FINAL,
readouts/glue_S_split_vs_28L_E6.md). The S GLUE win is a compute effect, not data efficiency.

28L official zero-shot (ruler ref-0216): split vs single composite +1.05, bar 2.04 → **does not beat**. BLiMP +0.9 (flat);
the composite's noise is dominated by the small WUG tasks; Reading_rt / Reading_spr sit near 0 for every model: a tokenization interaction (the official reading code scores the
target word without a leading space; our BPE export has add_prefix_space=False), not model signal (ruler ref-0217;
GPT-BERT 9.5-9.9 / 3.3-3.5). Arm-vs-arm stays unbiased (both arms ~0); absolute composites are ~1.6 low and are not
comparable to GPT-BERT or the leaderboard. Amended (Reading rescored with add_prefix_space=True; ruler ref-0218): +1.07,
bar 2.05, no claim; amended Reading lands at 7.8-8.7 / 2.8-3.3, near GPT-BERT's 9.5-9.9 / 3.3-3.5. The +0.11 nats/word DEV gain does not clear the official 3-seed bar at 28L.

Decision rule (critic ref-0185): composite = unweighted mean of every zero-shot task the pinned pipeline (bf55c1131e53654c2a87418f5629c26959acd710)
emits, entity tracking reported only; GLUE7 a separate column. Each arm ≥ 3 reps; "beats" = 2·sqrt(sd_a²/n_a +
sd_r²/n_r). Per-task rows are descriptive. No claim from < 3 reps, GPT-BERT, published numbers or T100 (anchor only).
The mission's 2x needs the x lower bound ≥ 2 at both sizes with the official scores not contradicting it.

## 6. Why 10x is out of reach here

- Illegal ceiling: 8 seed models averaged close ~0.29 nats/word (C3); perfect 100M-trained teachers close ~70% of the
  gap (~5x). Both break R0.
- Where the 10M→100M gap lives (D1 bias map): ~0.29 single-model variance (what ensembles recover), ≤0.23 learnable
  bias (upper bound; frequent words ≤0.12), ~0.22 information simply absent from 10M (rare words and facts).
- So any legal learner tops out near ~4x even if perfect (0.45 of a measured 0.743 gap, log-linear; a ceiling, not a
  result), and our levers each move 0.01-0.06 at equal compute.

## 7. Cost


 ruler ref-0178, ledger seq 175).
Actuals are executor runtime x spot rate, not reconciled against GCP accounting. Three GPUs
(quota-bound). Every loss curve saved as CSV + PNG.

## 8. Caveats

- One second size (28L). "Holds across sizes" means two sizes.
- 28L runs are torch 2.9, S runs torch 2.7: cross-size comparisons carry ENV_SYS.
- GLUE for our causal exports needed a pooling fix (last-token pooling); BOS-pooled side numbers were cut for budget.
- GLUE7 barely sees pretraining at this scale: our lastpool pipeline gains only ~4.7 points over a randomly initialised
  model (critic ref-0214), so GLUE ties between arms are expected and are reported as ties.
- All levers are known ideas; none is claimed as new.
- Our 10M-tuned recipe (WD 0.4, tied, no VE) does not transfer to 100M: ~0.31 nats/word worse (cal_t, ruler ref-0219) than the untuned port
  recipe at E10 (executor 3, 10-11 23:11Z): a loss at 100M from confirmed underfit (train 4.32 vs 3.52); which knob
  (WD 0.4, lr_scale 0.5, tied/no-VE) is untested.

## 9. Pending before this report is final

- A5 DONE (experimenter ruling 10-11 23:15Z amending critic ref-0189; final 10-12, ruler ref-0207, critic ref-0181): the
  registered locked-recipe 100M grid (WD 0.4, tied, no VE) reads 4.547 cal_t at E10, ~0.31 WORSE than the port-recipe
  T100 (WD 0; 3 seeds, cal_t 4.2384 / 4.2347, SD 0.0018), so the E-grid was cut (a weaker 100M point would inflate x).
  The locked recipe is underfit at 100M (train 4.32 vs 3.52 nats/word); no further 100M grid was run (budget). Untuned at
  100M: WD > 0, lr_scale, E < 10, model size. Final Y and x are in section 1 and the section 3 table.
- Official rows: S zero-shot/GLUE7 (running), 28L GLUE7 (serial on one GPU), amended Reading column (critic 81702719).
- GPT-BERT: our own 3-seed reproduction of the winning recipe was cut for budget (10-11 12:50Z); the comparator is the
  published checkpoints, certified on GLUE (12/14 cells exact) and zero-shot. Reference only under ref-0185 in any case.
- BOS-pooled GLUE side numbers cut for budget.
