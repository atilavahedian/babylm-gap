# Artifacts not in this folder: 2026-10-12-babylm-rung0-data-wall

| artifact | where | size if known |
|---|---|---|
| Raw per-step training logs (`dw0_*/train_log.csv`) | wip-branch:babylm/results/dw0/ | ~0.42 MB each, ~3.3 MB total |
| Checkpoints (keep_checkpoints true), HF exports and v2 temperature dumps (`dw0_*.npz`) | VM output prefixes below, `out/babylm/runs/dw0_*/` | not recorded |
| Design memo `memo_data_wall.md` (design agent 3, section 5) | not in any git branch | - |

## VM output prefixes (GCS; executor ledger `common/babylm/gcp/runs.jsonl`, pool rung0)
| artifact | where | contents |
|---|---|---|
| VM outputs, job `dw0-s1a` (commit ref-0369; | ${BUCKET}/babylm/vms/bl-dw0-s1a-1012-055728/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `dw0-s1a` (commit ref-0369; | ${BUCKET}/babylm/vms/bl-dw0-s1a-1012-060558/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `dw0-s1a` (commit ref-0369; | ${BUCKET}/babylm/vms/bl-dw0-s1a-1012-063156/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `dw0-s1b` (commit ref-0369; | ${BUCKET}/babylm/vms/bl-dw0-s1b-1012-083654/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `dw0-s3lr` (commit ref-0369; | ${BUCKET}/babylm/vms/bl-dw0-s3lr-1012-100818/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `dw0-lx` (commit ref-0369; | ${BUCKET}/babylm/vms/bl-dw0-lx-1012-110616/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
