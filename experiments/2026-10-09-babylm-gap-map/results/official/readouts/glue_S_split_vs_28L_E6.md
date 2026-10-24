# GLUE7: S_split vs 28L E6 single (compute-matched, registered by critic ref-0220)

Registered on 11 Oct at 23:31:20Z, before any L28_e6 GLUE cell existed. The first one (x0) started at 05:57Z on 12 Oct. Claim-grade.
Labels: **compute-matched ~2x**, **cross-env pretraining**, and the S_split provenance footnote below.

| arm | n | GLUE7 mean (sd) | per replicate |
|---|---|---|---|
| S_split (K=2 S mixture, 2 x 5 epochs) | 3 | 67.85 (0.51) | p01 67.29, p23 67.97, p45 68.28 |
| L28_E6 (28L single, 6 epochs) | 3 | 67.22 (0.41) | s0 67.63, s1 66.82, s2 67.20 |

**S_split vs L28_E6, GLUE7: diff +0.63, bar 0.75 -> DOES NOT BEAT (a tie at matched compute).**

Reading: the S_split GLUE7 "beats" against S_E6 (+1.29 vs bar 0.72, zs_S_split_vs_E6.md) was not compute-matched. With about the same pretraining compute used on one bigger model, the split's GLUE advantage falls inside noise. That is consistent with a known ensembling or extra-compute effect, not a data-efficiency effect.

Method:
- GLUE7 = mean x100 of the 7 official metrics (accuracy, except F1 for MRPC and QQP).
- Pooling and ensembling:
  - Single models use lastpool ens2 (fine-tune seeds 42 and 43, probabilities averaged; critic 44552922).
  - Mixtures use <model>_ens2 (member 0 seed 42, member 1 seed 43).
- Beats = diff >= 2 sqrt(sdA²/nA + sdR²/nR).
- Run as babylm/eval/official_readout.py --arm S_split=mixS_p01,mixS_p23,mixS_p45 --arm L28_E6=L28_e6_s0,L28_e6_s1,L28_e6_s2 --beats S_split:L28_E6.
- The OUT_DIRs are byi6gg @ref-0390: final_i_s_b1, final_i_s_b2, final_i_s_split_fix, final_i_s_split_fix2, final_ii_28l_x0, final_ii_28l_x1 and final_ii_28l_x2.
- The composite rows in the .json are empty for L28_E6 because its zero-shot sits in other dirs (zs_28L_mix_vs_single.md). They are not part of this registered check.

28L provenance:
- x0, x1 and x2 each ran on gcp-project-c, one job at a time, all at the same code commit, ref-0174.
- Every job used the official sequential GLUE script, eval_finetuning.sh, with nothing running in parallel.
- Software: torch 2.7.0+cu126, transformers 4.51.3, A100-SXM4-40GB, eval @ bf55c1131e53654c2a87418f5629c26959acd710.
- Wall times: x0 16108 s, x1 16236 s, x2 16308 s.

S_split provenance:
- p01 ran sequentially and alone on B1. B1 jobs were strictly serial: p01 ran 13:16:37-15:55:50Z and the next job started at 15:55:57Z.
- p23 (fix2) spot check was IDENTICAL. p45's multirc was rerun solo (critic ref-0191).

Cross-env pretraining: the S and 28L arms were not pretrained in the same environment. The label is the critic's (ref-0220), and it is carried as registered.
