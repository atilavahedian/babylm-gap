# Last-token-pooled GLUE7 for causal Hinton models: certification (critic ref-0214, 11 Oct 13:18Z)

Data: base_S_e6, sequential official GLUE, fine-tune seed 42.
- Trained half: byi6gg @ref-0194.
- Untrained half (PyTorch default init, seed 0): byi6gg @ref-0356. multirc was lost to the maxdur cut.

| task (official metric) | untrained | trained |
|---|---|---|
| boolq acc | 66.7 | 67.8 |
| mnli acc | 45.1 | 58.2 |
| mrpc F1 | 82.2 | 81.6 |
| multirc acc | - | 66.9 |
| qqp F1 | 62.1 | 70.1 |
| rte acc | 56.1 | 62.6 |
| wsc acc | 63.5 | 63.5 |
| mean, 6 shared tasks | 62.6 | 67.3 |
| GLUE7 | - | 67.24 |

## Ruling (critic ref-0214)
- **Certified for claim-grade GLUE7.**
  - The registered trained-half criterion is met: 7/7 tasks off majority, mcc > 0.18 on every task, all 7 complete.
  - MNLI, 58.2 against 45.1 untrained, shows the pretraining signal reaches the classifier through last-token pooling.
- **The "untrained ≈ majority" half failed.** The fault was in the criterion, not the pipeline: full fine-tuning from random init learns surface features.
  - No replacement threshold was adopted. The "≥3 tasks above the floor" rule was proposed after seeing the data.
- **The untrained floor is a reference row, labelled "amended after data".** It covers 6/7 tasks. It never yields a GLUE7 and never gates anything.
- **Registered before any arm's GLUE7:** pretraining moves GLUE only about 4.7 points over random init. A between-arm GLUE7 tie is therefore the expected result, and it is reported as a tie.

## Concurrency
- The S identity check came back MISMATCH on multirc only; the other 6 tasks were bit-identical. The concurrent run took 3533 s against 3942 s sequential.
- All GLUE after this point runs sequentially.
- The S split p23/p45 cells keep their 6 concurrent tasks, with these conditions (critic ref-0191):
  - multirc is rerun sequentially;
  - a spot rte identity check is run;
  - the footnote reads "concurrent, identity-verified on 2 S models".
