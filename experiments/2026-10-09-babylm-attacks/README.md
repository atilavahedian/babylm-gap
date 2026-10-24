# BabyLM attacks A1-A4 and the stacker (2026-10-09 to 2026-10-11)

## Question
Can a cheap training-time change buy data efficiency for a 10M-word causal LM, on top of the tuned round-1 base?
Each attack had to beat its matched base and registered null at equal words (2x ≈ 0.26 DEV nats/word). A1: temporal
self-distillation. A2: input noise. A3: BPE-dropout for rare words; A3g: annealed segmentation noise. A4: the model
writes its own contexts.

## Setup
- Base: 12-layer S model, `base_S_wd0p4_lr0p5_r1_ve0_tied`, trained on train_10M.
- Metric: DEV nats/word on the v2 blank-line twin, temperature-calibrated (`cal_t`), on full / guarded legs. The
  guarded leg drops words with never-seen characters.
- Kill line: 0.0074 for solo runs, 0.0102 packed.
- Every card was critic-accepted before its GPU run (`governance/critic_rulings.md`).
- Shared trainer: `common/babylm` (has `self_distill.py`, `input_noise.py`, `bpe_dropout.py`, `cycle.py`, `stack.py`).
- Experiment code: `code/babylm/a4/` (A4 generator, data, audit, verdict), and `code/overlays/`, which holds the
  trainer files of the unmerged A1, A2 and stacker branches that differ from common (`code/overlays/MANIFEST.json`). A1 and the stacker used no GPU.

## Result
| Attack | Outcome (cal_t, full / guarded) |
|---|---|
| A1 | Dropped before the screen (prior ~3%). No GPU run. |
| A2 | Killed as calibration-equivalent. A2a − base: +0.0047 / −0.0025, against a packed kill line of 0.0102. |
| A3 | Killed. a3a − null a3n = +0.0039, against a 0.0074 line. The gain is generic, not rare-word learning. |
| A3g | Closed, no claim. The single model gains +0.043 / +0.046 but loses to depth at ~1.9x FLOPs. Inside the K=2 split it gains +0.014 / +0.013, against a 0.020 bar. |
| A4 | A4a killed: −0.0380 / −0.0450 against its no-synthetic twin. A4b and T2 were dropped. |
| Stacker | Built and CPU-tested (off is bit-identical to the base). Never launched, because nothing was promoted. |

## Verdict
All attacks are closed and none buys data efficiency. The best result is A3g: +0.043 for a single model, which loses
to a compute-matched deeper model, and +0.014 inside the K=2 split, which is below the 0.020 bar.

## Reproduce
    cd common && export PYTHONPATH=$PWD                    # one CUDA GPU, torch 2.7.0, transformers 4.51.3
    X=$PWD/../experiments/2026-10-09-babylm-attacks
    PYTHONPATH=$PWD:$X/code python -m babylm.harness.run_jobs $X/configs/jobs_a3g_claim_S_a3a_e10_s123.json --device cuda
    PYTHONPATH=$X/code/overlays/a2_input_noise_gd7wjc:$PWD python -m babylm.harness.run_jobs \
        $X/configs/jobs_a2_screen.json --only a2_base a2_A2a --device cuda   # A1: overlays/a1_self_distill_2szb4k
    PYTHONPATH=$PWD:$X/code python -m pytest -q $X/code/babylm/tests/test_a4.py   # CPU; needs torch
- A4 is a pipeline (generation, build, audits between trainer jobs): replay `configs/gcp/a4_screen.json` via `babylm/gcp/launch.sh`.
- The `configs/gcp_jobs/*.json` files pin the original commits and repo layout. Replay them against the archive tags.
- Read results with `babylm.a4.verdict` and the twin/calibration tools in `common/babylm/eval` and `tools/`.
- The stacker's `test_make_stack_jobs.py` and `test_a4.py`'s `git show HEAD:` check read the old path
  `babylm/configs/jobs_a*.json`. Run them in a scratch copy of `common` with `configs/jobs_a*.json` and
  `configs/a1_bases.json` copied into its `babylm/configs/`; the git check also needs the archive tag.

## Provenance
| Item | Source branch | Commit |
|---|---|---|
| A1 | wip-branch (PR #55) | ref-0003 |
| A2 | wip-branch (PR #60) | ref-0004 |
| A3/A3g | wip-branch (PR #61) | ref-0005 |
| A4 | wip-branch | ref-0006 |
| Stacker | wip-branch | ref-0007 |
| Cards and index | wip-branch | ref-0008 |
| Reads, ledger | wip-branch | ref-0009 |
| Critic | wip-branch | ref-0010 |

A3 and A4 are merged into ouf5ji. Archive tags: `archive/<branch>` for each branch above.
