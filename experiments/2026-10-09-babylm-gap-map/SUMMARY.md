# BabyLM data efficiency: the 10M-to-100M gap map



## Question
How far does a model trained on 10M words fall short of the same recipe trained on 100M words, where does that gap
live, and how much of it can a recipe legal under BabyLM 2025 strict-small (10M unique words, at most 10 epochs, at
most 100M words of total exposure counting every model, no outside weights) close? The mission asked for 10x; 2x was
the stated realistic target.

## Method
- **Base.** 12L/384d causal Transformer (S) ported unchanged from the sprint-1 modded-nanogpt recipe, tied output, no
  value embeddings, Muon weight decay 0.4, chosen by a logged base sweep (`docs/base_sweep_log.md`); 6 epochs beat 10
  by 0.039, so the base trains E6. Second size 28L/384d (torch 2.9.1; S runs torch 2.7.0).
- **Metric.** Nats per whitespace word on full DEV, v2 blank-line twin scorer; claims gate on cal_t (temperature x
  lambda, cross-fitted on DEV document parity) on two legs, full and guarded (10,277 never-seen-character words
  dropped). Reference = E6 pool, 5 seeds, 4.9693 raw (SD 0.0057); claim floor 2 SD on cal_t = 0.0111 / 0.0065.
  Systematics: torch 2.7 vs 2.9 ~0.012, packed vs solo ~0.012; claims are solo within one env.
- **Gap map (D1).** On the same DEV words: mean of 8 seed members (10M x 10 epochs), their 8-model mixture, a
  C1-oracle student (10M contexts, targets from a 100M teacher) and the 100M model T100. Variance = member - ens8;
  learnable-bias upper bound = ens8 - c1oracle; absent information = c1oracle - t100; per train_10M frequency band.
- **Arms.** K=2 split: two models x 5 epochs on the same 10M words (100M summed exposure), token probabilities
  averaged. Controls: the E6 pool (same size); compute-matched menus of single models at the split's FLOPs, F=2 at S
  (28L/384d, 12L/592d at E4-6) and F=4.146 at 28L (60L/384d, 36L/512d, 12L/880d; 7 cells).
- **Yardstick (A5).** x = 10^(gain / Y), Y = E6 pool minus port-recipe T100 (E10, 3 seeds) on cal_t; every x is an
  upper bound (100M point not exhaustively tuned) quoted as the lower end of its 95% interval (df 2, reference spread
  included), the smaller of all-DEV and overlap-removed DEV (134,739 words dropped).
- **Official.** Pinned upstream pipeline bf55c1131e53654c2a87418f5629c26959acd710; zero-shot composite = unweighted mean of 8 tasks (entity tracking reported
  only); GLUE7 with last-token pooling, 2-fine-tune-seed ensembles for every arm; "beats" = difference >=
  2 sqrt(sd_a^2/n_a + sd_r^2/n_r), 3 runs per arm.
- **Pre-registration.** Every bar below was registered by the critic before the result it judged (decision rule
  ref-0185; final readout table logged 2026-10-11T05:09Z); the critic could veto.
- **Compute.** GCP spot A100-40GB (a2-highgpu-1g), three GPUs at most, three projects (project-b, project-c, gcp-project-a).

## Results
| Row | Comparison (cal_t, full / guarded) | Result | Bar | Verdict |
|---|---|---|---|---|
| D1 | Gap decomposition (raw, torch 2.9.1; guarded leg 0.2815 / 0.2310 / 0.2021) | 0.743 = 0.289 variance + 0.233 learnable (upper bound) + 0.221 absent | diagnostic | perfect legal learner <= 0.45 of 0.743, ~4x (critic ref-0180) |
| D1b | Learnable slice in the 64-8k + 8k+ bands | 0.086 + 0.036 = 0.12 (<= ~0.16 with half of 1-63) | diagnostic | no single mechanism has a target above ~0.09 |
| A1 | S split (3 reps) vs E6 pool | +0.112 / +0.110; x >= 1.37 (point 1.44) | floor 0.0111 / 0.0065 | PASS (critic ref-0149) |
| A2 | S split vs best F=2 single (28L e6 s0; crosses torch envs) | +0.058 / +0.059; x >= 1.10 as registered (3-seed ref, post-registration: >= 1.16) | P-F2d band [0.05, 0.08] | HIT (critic ref-0186; claim ref-0149) |
| A3 | 28L split (reps +0.114, +0.112, +0.118 full) vs 28L e6 mean | +0.1145 / +0.1116; x >= 1.39 (point 1.45) | >= M_S - 0.02 = 0.0872 / 0.0881 | HOLDS, flat within noise (critic ref-0187) |
| A4 | 28L split vs best of 7 F=4.146 cells (36L/512d e4: 4.8728 / 4.8461) | +0.086 / +0.083; x >= 1.22 (point 1.32) | 0.012 | PASS (critic ref-0182) |
| 2x | Any row with g >= 0.215 / 0.208 (x lower bound >= 2 at both sizes) | highest interval upper end 1.52 | 2x | NOT MET |
| Y | E6 pool minus T100 | 0.715 / 0.690 (Welch 95% [0.708, 0.722] / [0.686, 0.695]) | - | - |
| O1 | Zero-shot composite, S split vs S E6 single | 37.43 (0.45) vs 36.04 (1.53): +1.39 | 1.84 | tie |
| O2 | Zero-shot composite, 28L split vs 28L e6 single | 37.57 (1.70) vs 36.52 (0.47): +1.05 | 2.04 | tie |
| O3 | GLUE7, S split vs S E6 single | 67.85 (0.51) vs 66.56 (0.37): +1.29 | 0.72 | beats, but not compute-matched |
| O4 | GLUE7, S split vs 28L e6 single (compute-matched, registered first) | 67.85 vs 67.22 (0.41): +0.63 | 0.75 | tie (critic ref-0188) |
| O5 | GLUE7, 28L split vs 28L e6 single | 67.10 (0.86) vs 67.22: -0.11 | 1.09 | tie (critic ref-0179) |
| O6 | Amended composite (Reading rescored with the prefix space; after data, no claim) | S 38.88 vs 37.39 (+1.49, bar 1.74); 28L 38.97 vs 37.90 (+1.07, bar 2.05) | - | ties; published GPT-BERT 10M: 40.32 / 36.66 |

Best 10M number: the 28L split mixture at 4.795 nats/word raw, 0.17 below the base at ~4x its compute.

## What it rules out
- 10x under the BabyLM rules on this data: 0.221 nats/word (~30% of the gap) is information absent from 10M words,
  and a perfect posterior plus a perfect prior tops out near ~4x on the map's own metric.
- The K=2 split reaching 2x: its gain is about half the 2x bar at both sizes (no interval upper end above 1.52).
- "The split is just more parameters": it beats every compute-matched single (F=2 at S; 7 F=4.146 cells at 28L), and
  the compute-matched edge grows from 0.058 to 0.086 because bigger singles stop paying at 10M words.
- A data-efficiency claim on the official BabyLM scores at 3 seeds: every official comparison ties except the
  non-compute-matched S GLUE7 win, which vanishes against a 28L single of equal compute.

## What it does not rule out / caveats
- x is a model: it assumes log-linear returns between 10M and 100M, and every x is an upper bound because the 100M
  point is the port recipe, not one tuned at 100M. The 10M-tuned recipe is ~0.31 nats/word worse at 100M from
  confirmed underfitting (train 4.32 vs 3.52); which knob causes it is untested.
- The learnable slice is an upper bound set by a teacher that saw 100M words; most of the 1-63 band in it is really
  100M information. A learner with a better prior than a Transformer could still attack <= ~0.12-0.16 nats/word.
- Two sizes only (12L, 28L) in two torch environments (cross-size comparisons carry ~0.012 ENV_SYS); A2 crosses envs.
- A2 point estimate: the ruler's registered-form readout gives point 1.20 (g +0.0565 / +0.0570 vs 28L e6 s0); the
  draft report and paper print 1.21, which is the post-registration 3-seed point. The registered lower bound (1.10)
  agrees in every source.
- GLUE7 barely sees pretraining at this scale (~4.7 points over an untrained model), so ties are expected; composite
  spread is dominated by the 100-item WUG tasks; absolute composites are ~1.6 low from a Reading-task tokenization
  interaction (amended column, no claim).
- Our own 3-seed GPT-BERT reproduction was cut for budget; GPT-BERT rows are published checkpoints scored on our
  certified pipeline (12/14 GLUE cells exact), reference only.
- Every lever is a known idea (seed / data-split ensembles under a fixed exposure budget); nothing is claimed as new.
- Failed levers live in their own folders: `../2026-10-09-babylm-attacks/` (A1-A4), `../2026-10-10-babylm-moonshot-10x/`
  (F1, F2, PE, trunk/branch, K=3), `../2026-10-12-babylm-rung0-data-wall/`, `../2026-10-10-babylm-rare-words/`.

## References
- [docs/FINAL_REPORT_DRAFT.md](docs/FINAL_REPORT_DRAFT.md): the experimenter's final report (primary record, verbatim).
- [docs/RULER_REPORT.md](docs/RULER_REPORT.md): the ruler's plain-English final report and compute.
- [docs/base_sweep_log.md](docs/base_sweep_log.md), [docs/facts_2025.md](docs/facts_2025.md), [docs/priors_data_efficiency.md](docs/priors_data_efficiency.md).
- [governance/critic_rules.md](governance/critic_rules.md), [governance/critic_log.md](governance/critic_log.md), [governance/prior_art.md](governance/prior_art.md).
- [results/d1_bias/bias.csv](results/d1_bias/bias.csv): the gap map; [results/official/readouts/a5_xfactor_final.md](results/official/readouts/a5_xfactor_final.md): Y and every x.
- [results/official/readouts/](results/official/readouts/): zero-shot and GLUE7 verdicts; [results/dev_reads/](results/dev_reads/): K=2 and 28L reads.
- [../2026-10-10-babylm-moonshot-10x/results/f4_menu/README.md](../2026-10-10-babylm-moonshot-10x/results/f4_menu/README.md): the F=4.146 menu (A4 comparator).
- Compute: `common/babylm/gcp/runs.jsonl` and `common/babylm/ledger/runs.jsonl` (hash-chained ruler ledger, seq 0-174, same total).
- Paper draft: [../../papers/babylm-gap-map/](../../papers/babylm-gap-map/).
