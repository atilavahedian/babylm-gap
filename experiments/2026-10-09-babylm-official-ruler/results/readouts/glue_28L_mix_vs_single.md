# GLUE7: 28L K=2 mixture vs 28L E6 single

| arm | n | GLUE7 mean (sd) | per replicate |
|---|---|---|---|
| mix28L (2 x 5 epochs, averaged) | 3 | 67.10 (0.86) | p01 66.97, p23 66.32, p45 68.02 |
| single28L (E6) | 3 | 67.22 (0.41) | s0 67.63, s1 66.82, s2 67.20 |

**mix28L vs single28L, GLUE7: diff -0.11, bar 1.09 -> DOES NOT BEAT (tie).** The comparison is NOT compute-matched: the mixture has about 1.67x the pretraining compute of one E6 single.

With the 28L zero-shot result (zs_28L_mix_vs_single.md: +1.05 against a bar of 2.04, DOES NOT BEAT), the 28L split ties the 28L single on every official column. Its only measured gain is DEV loss (A5 A3/A4).

Method:
- Same as glue_S_split_vs_28L_E6.md: official_readout.py, GLUE7 = mean x100 of the 7 official metrics.
- Single models use lastpool ens2 (fine-tune seeds 42 and 43). Mixtures use <model>_ens2 (member 0 seed 42, member 1 seed 43).
- Beats = diff >= 2 sqrt(sdA²/nA + sdR²/nR).
- The reference is the critic's L28_e6 singles (ref-0188).
- OUT_DIRs, byi6gg @ref-0009:
  - final_ii_28l_x0/glue_28L_e6_s0_lastpool, final_ii_28l_x1/glue_28L_e6_s1_lastpool, final_ii_28l_x2/glue_28L_e6_s2_lastpool
  - final_ii_28l_z1r/glue_resume_mix28L_p01, final_ii_28l_z2/glue_mix_28L_p23, final_ii_28l_z3/glue_mix_28L_p45
- The composite is empty here because the zero-shot readout is in its own file.
- Mixture guards: every member passed its export-vs-read DEV check, with |diff| below 2e-9.

Provenance:
- All six jobs ran one at a time on gcp-project-c, at commit ref-0174, with the official sequential script eval_finetuning.sh.
- Software: torch 2.7.0, transformers 4.51.3, eval @ bf55c1131e53654c2a87418f5629c26959acd710.
- p01 (z1r) is a resume of z1. z1 was preempted at 18:54Z on 11 Oct, after m0 boolq, 1 of 14 tasks. That one cell was restored through the per-task resume shim. z1 used the same official sequential script (job cmd FT_PARALLEL=0, final_table_lines/final_ii_28l_z1.json) but ran at commit ref-0173; the other 13 tasks ran in z1r at ref-0174. Critic ref-0179: the verdict holds even if that cell moved by 7 points (about 0.33 on the diff).
- p23 ran in z2 and p45 in z3, each in a single pass.
