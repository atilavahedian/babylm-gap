# BabyLM ruler: final report (13 Oct 2026)

**Bottom line: the ruler works, and nothing beat the bar on official scores.** The split trick (two half-length models averaged) helps on held-out loss: 1.1-1.4x data-equivalent as a lower bound, and that figure is itself an upper bound. It does not move the official BabyLM scores once compute is matched. 2x is not met.

## What was built
- An official BabyLM 2025 pipeline (evaluation-pipeline-2025 @ bf55c1131e53654c2a87418f5629c26959acd710, transformers 4.51.3, torch 2.7.0) with every input hashed.
- Certified against published checkpoints:
  - GPT-BERT zero-shot reproduces exactly on BLiMP, Supplement and EWoK.
  - GPT-BERT GLUE reproduces in 12 of 14 cells to 2 decimals. The only miss is MultiRC, by 0.24 or less.
- Last-token GLUE pooling for our causal models was certified by the critic (ref-0214). The untrained model is kept as a reference row (62.6 on 6/7 tasks).
- Every run has an append-only, hash-chained cost ledger (babylm/ledger). Loss curves for the GPU runs are saved as CSV and PNG on the executor branch.

## What was cut (and why)
- **Training our own GPT-BERT for 3 seeds. The published GPT-BERT and GPT-2 checkpoints serve as the reference rows instead. So the brief's "reproduce baseline and winner, 3 seeds each" was met as certification of published checkpoints, not as retraining.
- **Published GPT-2 last-token GLUE.** It was cut at 2/7 tasks, and the BOS-pooling GLUE rows were cut too.
- **Amended Reading.** add_prefix_space=True is reported as "amended, no claim". The pipeline's get_p2 quirk hits 8.3% of words for us vs 9.5% for GPT-BERT.

## Results (official, n=3 vs 3, bar = 2 x combined SE)
| comparison | zero-shot | GLUE7 | compute |
|---|---|---|---|
| S split vs S single | +1.39 vs bar 1.84: tie | +1.29 vs bar 0.72: BEATS | ~1.67x, not matched |
| S split vs 28L single | n/a | +0.63 vs bar 0.75: tie | ~matched (critic ref-0220) |
| 28L split vs 28L single | +1.05 vs bar 2.04: tie | -0.11 vs bar 1.09: tie | ~1.67x, not matched |

The only official "win" (S GLUE7) disappears once compute is matched. That is consistent with a known ensembling/extra-compute effect.

## Data-equivalent multiplier (A5, DEV loss)
x = 10^(gain / Y), where Y = S 10M E6 pool minus the 100M reference (T100, port E10, 3 seeds). These are lower 95% bounds, worse of the full and overlap-removed views:

| | full | guarded |
|---|---|---|
| A1 S same size | 1.37 | 1.39 |
| A2 S FLOPs-matched (registered ref) | 1.10 | 1.15 |
| A3 28L same size | 1.39 | 1.41 |
| A4 28L FLOPs-matched | 1.22 | 1.25 |

- These are UPPER BOUNDS. The 100M point was not exhaustively tuned, and the locked 10M recipe is 0.31 nats/word worse at 100M than the port (4.547 vs 4.238). The 10M recipe does not transfer to 100M.
- **2x NOT MET.**

## Compute




Sources: readouts/zs_S_split_vs_E6.md, glue_S_split_vs_28L_E6.md, zs_28L_mix_vs_single.md, glue_28L_mix_vs_single.md, a5_xfactor_final.md; final_table_lines/a5_t100_port.json, glue_lastpool_certification.md; certify_v3/glue_certification_1011.md; ledger/runs.jsonl.
