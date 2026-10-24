# Data-wall rung 0: heavy regularisation and looped depth at 16 epochs (2026-10-12)

## Question
With a fixed corpus and spare compute (the "data wall", outside BabyLM's 10-epoch cap), is our 10M-word recipe
under-regularised at 16 epochs (Q1), and does looping a 12-layer model twice with shared weights beat a plain 24-layer
model of equal FLOPs (Q2)? Out of track; feeds no BabyLM claim.

## Setup
- Size S (12L/384d, tied, no value embeddings, lr_scale 0.5) on train_10M for 16 epochs, seed 0, solo A100, torch 2.7.0.
- Q1: control C at Muon weight decay 0.4 vs W in {0.2, 0.8, 1.6, 3.2} and one LR cell (lr_scale 1.0) at the best W.
- Q2: 12L looped x2 (`n_loops=2`, `common/babylm/models/looped.py`) vs plain 24L, both 349.2M FLOPs/token, WD 0.4.
- Registered before any run (critic ref-0368, 2026-10-12T05:37Z; Atila's go 05:35Z;: alive if the pick
  beats C, or looped beats its plain twin, by the noise bar on both cal_t legs; screen kill 0.0074 before extra seeds.
- Metric: cal_t on full / guarded DEV (v2 twin dumps with temperatures), as in the gap map.

## Result
| Cell (16 epochs, s0) | cal_t full / guarded |
|---|---|
| C, WD 0.4 | 5.0112 / 4.9806 |
| W 0.2 / 0.8 / 1.6 / 3.2 | 5.0820 / 5.0431, 5.0677 / 5.0381, 5.2031 / 5.1763, 5.3669 / 5.3408 |
| LR x2 at W 0.8 | 5.2137 / 5.1876 |
| 12L looped x2 | 5.0256 / 4.9924 |
| plain 24L (FLOPs twin) | 4.9875 / 4.9542 |
| reference: E6 pool, 5 seeds | 4.9534 / 4.9251 |
- Q1: best alternative W* = W0.8 is worse than C by 0.0565 / 0.0575 (kill 0.0074). Q1 DEAD (critic ref-0208).
- Q2: looped is worse than its FLOPs twin by 0.0381 / 0.0382 and worse than the 12L control by 0.0144 / 0.0118.
  Plain 24L beats C by 0.0237 / 0.0264 at 1.8x FLOPs and params (ordinary scale). Q2 DEAD (critic ref-0211).
- C at 16 epochs is 0.058 / 0.055 worse than the same recipe at 6 epochs.

## Verdict
DEAD on both levers (critic ref-0211): retuning at 16 epochs loses 0.057 to WD 0.4, looped depth loses 0.038 to a plain
24L FLOPs twin; no rung 1.

## Reproduce
    export PYTHONPATH=common:experiments/2026-10-10-babylm-moonshot-10x/code    # from the repo root
    python common/babylm/gcp/dw/make_dw_r0_jobs.py     # run from a tree holding babylm/gcp/jobs/a5_100m_grid_a.json
        # (copy in experiments/2026-10-09-babylm-gap-map/configs/gcp_jobs/); regenerates dw0_s1a/s1b/L.json byte-identical
        # to configs/gcp_jobs/ (checked); dw0_Lx and dw0_s3lr are the executor's hand-trimmed variants
    POOL=rung0 bash common/babylm/gcp/launch.sh <ticket.json> gcp-project-b us-central1-a 2h30m   # one spot VM per ticket
    python experiments/2026-10-09-babylm-gap-map/results/dev_reads/k2_estar/calib_set.py OUT.json C04=<dump> L12x2=<dump> P24=<dump> --
    python -m pytest -q common/babylm/tests/test_looped.py            # needs torch
All code is in `common/babylm` (no experiment-only code). Per-run curves and scores are in `results/`.

## Provenance
- Experimenter: `wip-branch` (looped.py and its test ref-0369; rung-0 docs, curves, failed-attacks rows
  ref-0438). Executor: `wip-branch` (job builder ref-0439; s1a ref-0370, s1b ref-0372, LR ref-0209,
  L screen ref-0210). Critic: ref-0368 (registration), ref-0371, ref-0208 (Q1), ref-0211 (final); log at
  `wip-branch` @ ref-0010.
