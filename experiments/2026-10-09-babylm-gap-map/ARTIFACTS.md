# Artifacts not in this folder: 2026-10-09-babylm-gap-map

Checkpoints, HF exports and per-token DEV dumps were never committed to git; every GPU job wrote them under its VM's
output prefix on GCS (bucket `${BUCKET}/babylm/vms/<vm id>/out/`, ids and costs from the executor
ledger `common/babylm/gcp/runs.jsonl`). Files dropped from `results/` for size or licence are listed with their
archive path (`archive/<branch>:<path>`).

| artifact | where | size if known |
|---|---|---|
| Oracle teacher T100 (S, port recipe, 100M x 10; `chck_1000M.pt`) and its HF export | ${BUCKET}/babylm/vms/bl-exp-p2a-1009-204558/out/babylm/runs/grid_S_100M_e10_s0/ | not recorded |
| HF exports used for official scoring (one URI per model) | `results/official/runs/**/hf_uri_*.txt` list each GCS path | - |
| Raw per-step training logs of the A5 100M runs (`train_log.csv`, 1.3-2.7 MB each) | wip-branch:babylm/gcp/reads/a5/ and wip-branch:babylm/results/a5_curves/ | ~10 MB total |
| Raw per-step logs of the size-M cells (`train_log.csv`) | wip-branch:babylm/gcp/reads/base_size_m/ | ~0.85 MB |
| Official-eval per-item predictions (`predictions.json`, `valid_probs.json`, `prediction.jsonl`, `ties_*.jsonl`) and eval logs of the final runs | wip-branch:babylm/results/{final_i_s_*,final_ii_28l_*,ftverify,zs_cpu}/ | ~223 MB total; dropped (bulk; some embed eval-item text) |
| Reading-task alignment checks (`reading_align_*.json`, quote eval words) | same archive paths | small; dropped (eval data text) |
| Pipeline certification runs (ruler) | wip-branch:babylm/results/certify_v3/ (owned by the official-ruler experiment) | ~19 MB |
| EWoK evaluation data (never redistributed) | ${BUCKET}/ewok/ewok-core-1.0_ref-0155.zip (private bucket) | - |
| BabyLM 2025 corpus (train_10M / train_100M / dev; never committed) | OSF, fetched by `python -m babylm.data.prepare --download` | 2.6 GB of bins |
| Experimenter's Claude Doc mirror of the paper | linked from `papers/babylm-gap-map/main.tex` line 2 (private) | - |

## VM output prefixes (GCS)
| artifact | where | contents |
|---|---|---|
| VM outputs, job `smoke` (commit ref-0156; | ${BUCKET}/babylm/vms/bl-smoke-1009-185524/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_s1` (commit ref-0157; | ${BUCKET}/babylm/vms/bl-base-s1-1009-200254/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_s1` (commit ref-0157; | ${BUCKET}/babylm/vms/bl-base-s1-1009-200500/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `lead_p2a` (commit ref-0158; | ${BUCKET}/babylm/vms/bl-exp-p2a-1009-204558/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_s1` (commit ref-0157; | ${BUCKET}/babylm/vms/bl-base-s1-1009-215511/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_s1` (commit ref-0157; | ${BUCKET}/babylm/vms/bl-base-s1-1009-215715/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_s1_lr0p5` (commit ref-0157; | ${BUCKET}/babylm/vms/bl-base-s1-lr0p5-1009-225246/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_r1_a` (commit ref-0159; | ${BUCKET}/babylm/vms/bl-base-r1-a-1010-002045/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_r1_b` (commit ref-0159; | ${BUCKET}/babylm/vms/bl-base-r1-b-1010-002136/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_r1_a` (commit ref-0064; | ${BUCKET}/babylm/vms/bl-base-r1-a-1010-003047/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_r1_b` (commit ref-0064; | ${BUCKET}/babylm/vms/bl-base-r1-b-1010-003932/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_cells_x` (commit ref-0065; | ${BUCKET}/babylm/vms/bl-base-cells-x-1010-014111/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_cells_y` (commit ref-0065; | ${BUCKET}/babylm/vms/bl-base-cells-y-1010-014657/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_cells_y2` (commit ref-0065; | ${BUCKET}/babylm/vms/bl-base-cells-y2-1010-024204/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_seeds_r1win` (commit ref-0160; | ${BUCKET}/babylm/vms/bl-base-seeds-r1win-1010-030918/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_seeds_s34` (commit ref-0160; | ${BUCKET}/babylm/vms/bl-base-seeds-s34-1010-055210/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_seeds_s34` (commit ref-0160; | ${BUCKET}/babylm/vms/bl-base-seeds-s34-1010-060632/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `lead_p2v2b_10m` (commit ref-0161; | ${BUCKET}/babylm/vms/bl-exp-p2v2b-10m-1010-070824/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `lead_p2v2c` (commit ref-0162; | ${BUCKET}/babylm/vms/bl-exp-p2v2c-1010-093536/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_legal_split` (commit ref-0163; | ${BUCKET}/babylm/vms/bl-ms-legal-split-1010-113631/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_size_m` (commit ref-0164; | ${BUCKET}/babylm/vms/bl-base-size-m-1010-134040/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `lead_p2v2d_estar` (commit ref-0165; | ${BUCKET}/babylm/vms/bl-exp-p2v2d-estar-1010-154711/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `lead_p2v2d_ls` (commit ref-0166; | ${BUCKET}/babylm/vms/bl-exp-p2v2d-ls-1010-160122/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `lead_p2v2d_ls_b` (commit ref-0166; | ${BUCKET}/babylm/vms/bl-exp-p2v2d-ls-b-1010-170941/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_28l_e5_s1` (commit ref-0074; | ${BUCKET}/babylm/vms/bl-ms-28l-e5-s1-1010-185942/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `base_size_m_shorte` (commit ref-0167; | ${BUCKET}/babylm/vms/bl-base-size-m-shorte-1010-191937/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `lead_p2v2f` (commit ref-0168; | ${BUCKET}/babylm/vms/bl-exp-p2v2f-1010-193702/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_28l_claim_rep2` (commit ref-0169; | ${BUCKET}/babylm/vms/bl-ms-28l-claim-rep2-1010-214308/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_28l_claim_rep3` (commit ref-0170; | ${BUCKET}/babylm/vms/bl-ms-28l-claim-rep3-1011-030026/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_28l_claim_rep3` (commit ref-0170; | ${BUCKET}/babylm/vms/bl-ms-28l-claim-rep3-1011-034552/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ms_28l_retrain_zs` (commit ref-0170; | ${BUCKET}/babylm/vms/bl-ms-28l-retrain-zs-1011-093615/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `ftverify-base-s-e6-lastpool` (commit ref-0171; | ${BUCKET}/babylm/vms/bl-ftverify-base-s-e6-lastpool-1011-101001/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-i-s-split-p23-p45` (commit ref-0171; | ${BUCKET}/babylm/vms/bl-final-i-s-split-p23-p45-1011-104317/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-i-s-b1` (commit ref-0171; | ${BUCKET}/babylm/vms/bl-final-i-s-b1-1011-131433/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-i-s-split-fix` (commit ref-0172; | ${BUCKET}/babylm/vms/bl-final-i-s-split-fix-1011-144959/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-i-s-b2` (commit ref-0171; | ${BUCKET}/babylm/vms/bl-final-i-s-b2-1011-150504/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-i-s-split-fix` (commit ref-0173; | ${BUCKET}/babylm/vms/bl-final-i-s-split-fix-1011-150745/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-x0` (commit ref-0173; | ${BUCKET}/babylm/vms/bl-final-ii-28l-x0-1011-150826/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-zs` (commit ref-0173; | ${BUCKET}/babylm/vms/bl-final-ii-28l-zs-1011-151344/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-zs` (commit ref-0173; | ${BUCKET}/babylm/vms/bl-final-ii-28l-zs-1011-151920/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-zs-singles` (commit ref-0173; | ${BUCKET}/babylm/vms/bl-final-ii-28l-zs-singles-1011-170730/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-z1` (commit ref-0173; | ${BUCKET}/babylm/vms/bl-final-ii-28l-z1-1011-174734/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-i-s-split-fix2` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-i-s-split-fix2-1011-181023/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-i-s-b2` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-i-s-b2-1011-183322/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-i-s-b2` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-i-s-b2-1011-183852/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-z1r` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-ii-28l-z1r-1011-185917/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-z1r` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-ii-28l-z1r-1011-190119/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `a5-100m-grid-a` (commit ref-0165; | ${BUCKET}/babylm/vms/bl-a5-100m-grid-a-1011-195520/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `a5-100m-port-s1s2` (commit ref-0165; | ${BUCKET}/babylm/vms/bl-a5-100m-port-s1s2-1011-232508/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-z2` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-ii-28l-z2-1011-234646/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-x0` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-ii-28l-x0-1012-053724/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-x1` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-ii-28l-x1-1012-103035/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-x2` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-ii-28l-x2-1012-151209/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
| VM outputs, job `final-ii-28l-z3` (commit ref-0174; | ${BUCKET}/babylm/vms/bl-final-ii-28l-z3-1012-195758/out/ | run dirs incl. checkpoints, score/hf exports, dumps |
