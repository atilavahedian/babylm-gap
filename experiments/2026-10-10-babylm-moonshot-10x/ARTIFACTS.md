# Artifacts not in this folder: 2026-10-10-babylm-moonshot-10x

| artifact | where | size if known |
|---|---|---|
| Per-token DEV dumps (`score/dev_pt.npz`, `dev_v2.npz` with temps) of C1/C3/PE/F1/legal-split/menu runs | VM output prefixes below, `out/babylm/runs/<run>/score/` | not recorded (npz, > 1 MB each) |
| Oracle teacher T100 (`grid_S_100M_e10_s0/chck_1000M.pt`) | ${BUCKET}/babylm/vms/bl-exp-p2a-1009-204558/out/babylm/runs/grid_S_100M_e10_s0/ | not recorded |
| F1 stage-1 twin jsons (`babylm/moonshot/f1/`, cited by critic entry 116) | not found in any branch; expected under the ms_f1_stage1 VM prefix below | - |
| Raw design-round outputs (`W{1,2,3}_result.json`) and the first moonshot handoff (`handoff_v0.md`) | shared/babylm/moonshot/design/ in the original session (not in git, not found now) | - |
| C3 seeds s3-s7 and legal-split members (ms_c3, ms_ls_*) | VM output prefixes below | - |
| Earlier versions of divergent files (`ceiling_report.py`, `make_menu_jobs.py`, `dev_bpw_dump.py`, `kd.py`, cards) | wip-branch, wip-branch, wip-branch | - |

## VM output prefixes (GCS)
| artifact | where | contents |
|---|---|---|
| VM outputs, job `ms_ceiling_a_p2` (commit ref-0413; | ${BUCKET}/babylm/vms/bl-ms-ceiling-a-p2-1010-034139/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_ceiling_b_p2` (commit ref-0276; | ${BUCKET}/babylm/vms/bl-ms-ceiling-b-p2-1010-050243/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_f1_stage1` (commit ref-0282; | ${BUCKET}/babylm/vms/bl-ms-f1-stage1-1010-074742/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_pe_gate2` (commit ref-0282; | ${BUCKET}/babylm/vms/bl-ms-pe-gate2-1010-130954/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_menu_f2` (commit ref-0073; | ${BUCKET}/babylm/vms/bl-ms-menu-f2-1010-140927/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_menu_f4_b` (commit ref-0170; | ${BUCKET}/babylm/vms/bl-ms-menu-f4-b-1011-060534/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_menu_f4_a` (commit ref-0170; | ${BUCKET}/babylm/vms/bl-ms-menu-f4-a-1011-061030/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_legal_split` (commit ref-0163; K=2 s0/s1, K=3, trunk+branches) | ${BUCKET}/babylm/vms/bl-ms-legal-split-1010-113631/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
