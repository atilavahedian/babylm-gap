# BabyLM official ruler (2026-10-09 to 2026-10-13)

## Question
Can we run the official BabyLM 2025 evaluation and reproduce published numbers exactly, so that every BabyLM attack
is scored with one certified ruler? A second question was which bottleneck instruments to use for reading attacks.

## Setup
- Pipeline: evaluation-pipeline-2025 @ `bf55c1131e53654c2a87418f5629c26959acd710`, transformers 4.51.3, torch 2.7.0, A100 40GB, every input hashed.
- Data: OSF `ryjfm`.
- Certified against the published checkpoints: GPT-BERT causal-focus and mixed (10M), and GPT-2.
- Controls: an untrained model and a NaN model.
- Gates come from the critic's R1 rule: zero-shot within 0.5 points of published.
- GLUE runs the official sequential script, all 7 tasks, before any number counts.
- Causal models are scored on GLUE with last-token pooling, certified separately.
- Instruments use frozen DEV/train samples and slices (`docs/instruments.md`).
- Code is in `common/babylm` (`eval/`, `recipes/`, `jobs/vm_job.sh`, `ledger/`, `instruments/`, `gcp/`).

## Result
- GPT-BERT causal-focus zero-shot is exact: BLiMP 71.66, Supplement 63.21, full EWoK 49.49.
- Entity tracking scored 34.55 vs 33.96 published. Re-scored on the sha-pinned OSF v1 entity files, it gives 33.96
  exactly (critic, 10-10 22:50Z).
- GPT-BERT GLUE: 12 of 14 cells (causal-focus and mixed) are exact to 2 decimals. The only miss is MultiRC, by +0.24 / +0.04.
- Official BOS pooling is degenerate for causal models (mcc 0, majority class).
- Last-token pooling is certified for claim-grade GLUE7 (critic ref-0214). The trained base_S_e6 scores GLUE7 67.24.
- Our Reading scores are about 0 because of the target-word token form. An amended `add_prefix_space=True` column is
  reported, with no claim.
- Final-table outcome (`docs/RULER_REPORT.md`): nothing beat the bar on official scores. The K=2 split ties at
  matched compute; its only official "win" (S GLUE7 +1.29 vs bar 0.72) is at ~1.67x compute. Data-equivalent x is at
  most 1.37-1.41 (lower 95% bounds). 2x is not met.

## Verdict
Ruler certified. GPT-BERT reproduces exactly on BLiMP/Supplement/EWoK, 12/14 GLUE cells match (MultiRC off by ≤0.24),
and last-token GLUE is certified for causal models. Nothing measured with it beat the bar.

## Reproduce
    cd common && export PYTHONPATH=$PWD                       # one CUDA GPU; vm_job.sh sets up /opt/babylm itself
    EWOK_ZIP=${BUCKET}/ewok/ewok-core-1.0_ref-0155.zip FT_PARALLEL=0 \
      bash babylm/jobs/vm_job.sh score_pub_gptbert_causal out/score_pub_gptbert_causal   # zero-shot + sequential GLUE
    bash babylm/jobs/vm_job.sh ftverify_pub_gptbert_causal out/ftverify_pub_gptbert_causal
    python babylm/eval/official_readout.py --arm NAME=model1,model2 out/...    # readouts as in results/readouts/
The `configs/gcp_jobs/*.json` files pin original commits and the original repo layout. Replay them with
`babylm/gcp/launch.sh <job.json> <project> <zone> <maxdur>` against the archive tags, not this clean tree. EWoK is
read from the private bucket at run time. Never copy it into the repo.

## Provenance
- Ruler: wip-branch @ ref-0177 (`babylm/results/RULER_REPORT.md` ref-0178; readouts, certify).
- Executor: wip-branch @ ref-0009 (certify_v3, ftverify, zs_cpu, gcp jobs, ledger).
- Instruments: wip-branch @ ref-0314.
- Critic: wip-branch @ ref-0010 (`governance/`).
- Archive tags: `wip-branch-{ij746t,byi6gg,o3stxz,3k2sjz}`.
