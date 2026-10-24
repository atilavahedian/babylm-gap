# BabyLM attacks A1-A4: cheap training-time levers for data efficiency at 10M words A1 and the stacker used no GPU.

## Question
Four cheap training-time changes were tested on the tuned round-1 base. Does any of them buy real data efficiency for
a 10M-word causal LM, with the gain surviving temperature calibration, a matched null, and a compute-matched
reference? The 2x bar is about 0.26 DEV nats/word.

## Method
- **Base.** `base_S_wd0p4_lr0p5_r1_ve0_tied`: 12-layer S model, WD 0.4, LR x0.5, tied embeddings, no value embeds,
  trained on train_10M. A3 and A4 read against the E10 seed-0 base and the epoch-optimal E6 ("E*").
- **Metric.** DEV nats/word on the v2 blank-line twin, temperature-calibrated with its own DEV-fit T for each arm
  (`cal_t`). Two legs: full, and guarded (drops words with never-seen characters). BLiMP-fast is a guard.
- **Kill lines.** 0.0074 for solo runs and 0.0102 packed (critic). Claims need ≥ 3 seeds, and the official eval comes
  only after promotion.
- **Arms and controls:**
  - **A1**, temporal self-distillation (`--sd_mode temporal|smooth`). Judged against the same-seed early-stopped,
    calibrated base E*_cal.
  - **A2**, input noise. Arms A2a-A2e (replace, mask, gauss, unigram distractors, R2L). Nulls at p 0.05.
  - **A3**, BPE-dropout on rare words. Arms a3a (all, annealed), a3b, a3c, plus a token-matched frequent-only null
    a3n.
  - **A3g**, annealed segmentation noise `--bpe_dropout 0.1x(E-r),0xr` as a generic regulariser. Read three ways:
    single-model seeds 1-3 at E10; the R4.4 compute-matched 28L/384d entry; inside the legal K=2 split, over 3 reps.
  - **A4**, closed-system re-expression. A generator G is trained on train_10M and writes synthetic text, with a
    13-gram DEV block. Student A4a gets a second LR cycle at a 50% synthetic share. The no-synthetic twin is N0'_a.
    The text was audited for leaks.
  - **Stacker**, one harness in which every attack flag composes. CPU tests only.
- **Compute.** A100 40GB spot VMs, torch 2.7.0, pinned commits. The jobs are in `configs/gcp_jobs/` and `configs/gcp/`.
- **Pre-registration.** Every card and amendment was accepted by the critic before its GPU run. The relevant rows are
  in `governance/critic_rulings.md`.

## Results
All gains are cal_t (A3 raw twin), shown as full / guarded legs. A positive value means the arm is better.

| Attack | Comparison | Gain | Bar | Verdict |
|---|---|---|---|---|
| A1 | none run | none | ≥ 0.0074 vs E*_cal (revive only if a snapshot mix beats E*_cal by ≥ 0.015) | Dropped before the screen (10-10 10:53Z), prior ~3% |
| A2 | A2a − base (raw +0.0220 / +0.0128) | +0.0047 / −0.0025 | 0.0102 (packed) | **Killed**, calibration-equivalent (final 10-10 15:35Z, critic ref-0011) |
| A2 | A2b / A2c / A2d / A2e, screen v1 | −0.077 / −0.258 / −0.081 / −0.094 | 0.0102 | losses |
| A3 | a3a − a3n(p 0.1185), v2 twin raw | +0.0039 | 0.0074 | **Killed** (critic ref-0002) |
| A3 | a3a / a3n vs same-batch base 5.0077, raw | +0.0632 / +0.0593 | none | generic, not rare-word |
| A3g | single E10, s1-s3 vs plain E6 pool | +0.043 / +0.046 | claim floor + arm variance | clears |
| A3g | same, vs R4.4 28L/384d e6 (~1.9x FLOPs) | −0.013 / −0.007 | must beat it | **FAIL** (ref-0012) |
| A3g | A3K2 vs plain K=2, 3 reps (.020/.016/.005) | +0.014 / +0.013 | ≥ 0.020 both legs | **FAIL** (ref-0013) |
| A3g | second size, 28L e5 (Amendment 1) | not run | G = max(0.024, 2·seed gap) | dropped by the experimenter |
| A4 | A4a − N0'_a, v2 twin | −0.0380 / −0.0450 | 0.0074 | **Killed** (experimenter closed 10-11 03:18Z) |
| A4 | A4a vs base E10 / vs E6 seed 0 | −0.0101 / −0.0553 | none | worse |
| A4 | A4b, T2 | not run | none | dropped (~1% / ~4% odds) |
| Stacker | off = base, every pair, resume | bit-exact on CPU | none | built and never launched (no attack promoted) |

The A4 audits were clean:
- The 13-gram share of DEV in the synthetic text was about 1e-6.
- No BLiMP or EWoK item was hit.
- BLiMP-fast was +1.15 against N0'_a, so there was no BLiMP kill.

## What it rules out
- **Input noise (A2)** at this base only buys smoothing that a post-hoc temperature also buys.
- **BPE-dropout as rare-word learning (A3).** The token-matched frequent-only null did as well.
- **Annealed segmentation noise (A3g)**:
  - as a single-model lever at equal compute, because depth spends the FLOPs better;
  - as a registered add-on to the K=2 split, where its gain is below the bar and shrinks across reps.
- **Self-generated text at 10M words (A4).** It carries no information the student lacks. At a 50% share it displaces
  real epochs and costs 0.038 nats/word.
- **Taken together**, none of these levers comes near the 2x bar (~0.26). A3g's best single gain is ~17% of the bar,
  and inside the split it is ~5%.

## What it does not rule out / caveats
- **A1 was never trained.** The drop rests on a prior and on A2 and A3 dying as calibration- or epoch-equivalent. Its
  revival condition (snapshot mix ≥ 0.015 over E*_cal) was not tested here.
- **A3g inside the split** is probably a small positive add-on of about +0.013. It is unestablished, and Amendment 1
  (28L) was not run.
- **A4 batch 1 was cut** by the 3 h VM limit during A4a's per-token dump, after training and scoring. N1 and N2 were
  not run, but they could only add kills.
- **The bases matter.** Kills against a "provisional base" would need a re-screen if WD, LR, EMA or tokenisation
  changed.
- **Noise.** In the A3 family, run-to-run noise is about 3x the base seed SD: the two nulls differ by 0.024.
- **Known families.** The attacks come from known families: temporal ensembling/PS-KD, Xie et al. 2017 noising,
  BPE-dropout, and BERTtime Stories/Kang+ 2025 for synthetic text. These are clean negatives, not new methods.
- **Raw files are not in git.** Per-token dumps and checkpoints are listed in `ARTIFACTS.md`, and the GCS paths there
  were not re-verified from this container.

## References
- Index: [docs/README.md](docs/README.md).
- Cards:
  - [A1](docs/A1_temporal_self_distill.md)
  - [A2](docs/A2_input_noise.md)
  - [A3](docs/A3_bpe_dropout.md)
  - [A3g](docs/A3g_segmentation_noise.md)
  - [A4](docs/A4_reexpression.md), with reads in [A4_reads/](docs/A4_reads/)
  - [Stacker](docs/STACKING.md)
- Critic entries: [governance/critic_rulings.md](governance/critic_rulings.md). The full log is in
  [../2026-10-09-babylm-official-ruler/governance/critic_log.md](../2026-10-09-babylm-official-ruler/governance/critic_log.md).
- Calibration reads:
  - [results/gcp_reads/calib_a2_check.txt](results/gcp_reads/calib_a2_check.txt)
  - [calib_a3.txt](results/gcp_reads/calib_a3.txt)
  - [calib_a3_estar.txt](results/gcp_reads/calib_a3_estar.txt)
  - [a3g_claim/](results/gcp_reads/a3g_claim/), [a3g_stack/](results/gcp_reads/a3g_stack/), [a4_b1/](results/gcp_reads/a4_b1/)
