# BabyLM attacks: status index (experimenter, 2026-10-11)

Every attack below is closed. Gains are DEV nats/word on the v2 blank-line twin, calibrated (cal_t), full / guarded legs;
2x ≈ 0.26.

| Attack | Card | Result | Code on this branch |
|---|---|---|---|
| A1 temporal self-distillation | A1_temporal_self_distill.md | Dropped before the screen (prior ~3%) | No; branch wip-branch (PR #55) |
| A2 input noise | A2_input_noise.md | Killed: calibration-equivalent | No; branch wip-branch (PR #60) |
| A3 BPE-dropout / A3g annealed segmentation noise | A3_bpe_dropout.md, A3g_segmentation_noise.md | Single model +0.043 / +0.046 at equal words, but fails the compute-matched test (28L depth wins); inside the K=2 split +0.014 / +0.013 over 3 reps, below the 0.020 bar | Yes (`--bpe_dropout`, PR #61) |
| A4 model writes its own contexts | A4_reexpression.md, A4_reads/ | A4a 0.038 / 0.045 worse than its no-synthetic twin; A4b and T2 dropped | Yes (`--init_from`, `--sources`, PR #63) |

A3 and A4 are exclusive in train.py (`--bpe_dropout` with `--sources` exits). The legal K=2 data split (moonshot,
PR #62) is the standing result; see the moonshot branch for its cards and reads.
