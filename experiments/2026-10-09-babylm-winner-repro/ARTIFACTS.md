# Artifacts: BabyLM winner reproduction

GCS paths come from the job files and the executor ledger. They were **not re-verified** on 2026-10-13: there are
no GCS credentials in this container.

| artifact | where | size |
|---|---|---|
| AMLM hard_decay_off seed-0 final checkpoint (copied for the eval-only job) | `${BUCKET}/babylm/restore/amlm_off_s0/checkpoint-10155` | - |
| Seed-0 VM outputs (all epoch checkpoints, ~1.3 GB per run; per-token float16 losses) | `${BUCKET}/babylm/vms/bl-amlm-hard-decay-off-seed0-1009-215759/out/` | - |
| Aborted first launch (12 min, no result) | `${BUCKET}/babylm/vms/bl-amlm-hard-decay-off-seed0-1009-192700/out/` | - |
| Raw job log and training log (per-step) | `wip-branch:results/babylm/winner/train_hard_decay_off_seed0/job.log`, `.../run/train.log` | 1.03 MB, 0.72 MB |
| Authors' code, data (`bb24.train`), tokenizer | github.com/Leukas/babylm25 @ d1438b077761a891ec2b47c6854ea34be6e6e6aa (MIT; fetched by `fetch.sh`, sha256s in PIN.json) | - |
| Released checkpoints | HF `leukas/amlm_hard_decay` @ref-0233, `leukas/amlm_hard` @ref-0404 | - |
| Full EWoK (private; never vendor) | `${BUCKET}/ewok/` | - |
