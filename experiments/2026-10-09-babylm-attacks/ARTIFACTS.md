# Artifacts: BabyLM attacks

Nothing below is in git. The GCS paths come from the executor ledger and the reads
(`wip-branch:babylm/gcp/runs.jsonl` and `.../babylm/gcp/reads/delete_manifest_v1.tsv`).
They were not re-listed from this container, which has no GCS credentials.

## VM outputs (checkpoints, per-token dumps, HF exports, logs)

Each VM wrote `out/babylm/runs/<job>/` under its own prefix.

| artifact | where | size / cost |
|---|---|---|
| a2_screen_p2 VM `bl-a2-screen-p2-1010-040235` outputs | `${BUCKET}/babylm/vms/bl-a2-screen-p2-1010-040235/out/` | A2 screen done rc=0 (packed) |
| a3_screen_p2 VM `bl-a3-screen-p2-1010-054831` outputs | `${BUCKET}/babylm/vms/bl-a3-screen-p2-1010-054831/out/` | deleted in setup: A3 switched to solo (critic ref-0001) |
| a3_screen VM `bl-a3-screen-1010-060157` outputs | `${BUCKET}/babylm/vms/bl-a3-screen-1010-060157/out/` | A3 solo: hung in eval setup.sh after prepare (no log growth 06:13-07:40Z); deleted, relaun |
| a3_screen VM `bl-a3-screen-1010-074159` outputs | `${BUCKET}/babylm/vms/bl-a3-screen-1010-074159/out/` | A3 solo relaunch: 5 arms rc=0; A2 twin back-fill failed (hf dir bug, A2 killed) |
| a3n_fix VM `bl-a3n-fix-1010-095247` outputs | `${BUCKET}/babylm/vms/bl-a3n-fix-1010-095247/out/` | a3n p0.1185 rc=0; front A3 temps back-fill failed (twin module missing), re-run on a3_prom |
| a3_promote VM `bl-a3-promote-1010-104808` outputs | `${BUCKET}/babylm/vms/bl-a3-promote-1010-104808/out/` | A3 temps back-fill 5 arms ok; VM cut before E5 trial (A3 killed, critic ref-0002) |
| a3g_stack VM `bl-a3g-stack-1010-181156` outputs | `${BUCKET}/babylm/vms/bl-a3g-stack-1010-181156/out/` | a3g stack: a3a E5 s0 4.9308/4.9036, s1 4.9385/4.9099 (p2venv 2.7); temps dumps ok |
| a3g_claim VM `bl-a3g-claim-1010-195910` outputs | `${BUCKET}/babylm/vms/bl-a3g-claim-1010-195910/out/` | spot preempted 20:13Z during setup; no jobs ran; relaunched |
| a3g_claim VM `bl-a3g-claim-1010-201551` outputs | `${BUCKET}/babylm/vms/bl-a3g-claim-1010-201551/out/` | a3g claim a3a E10 s1-s3 rc=0; self-deleted 22:01Z |
| a3g_stack_r23 VM `bl-a3g-stack-r23-1010-220345` outputs | `${BUCKET}/babylm/vms/bl-a3g-stack-r23-1010-220345/out/` | A3g stack reps 2+3: a3a E5 s2-s5 rc=0 |
| a4_screen VM `bl-a4-screen-1010-235629` outputs | `${BUCKET}/babylm/vms/bl-a4-screen-1010-235629/out/` | CUT by max-run-duration 3h at 02:56Z: a4_G, halfgens, generate (4612 s), build, audit, n0a | A1 and the stacker made no GPU run.

## A2 run files moved off the project drive (GCS twin is the only copy; never delete)

| artifact | where | size |
|---|---|---|
| a2_base: hf/model.safetensors, pt/pt_dev_*.npy, dev_v2.npz (42 files) | GCS twins listed in `wip-branch:babylm/gcp/reads/delete_manifest_v1.tsv` (rows `shared/babylm/attacks/A2/a2_base/...`) | 661 MB |
| a2_A2a (41 files) | same manifest, `.../A2/a2_A2a/...`; dev_v2.npz under `${BUCKET}/babylm/vms/bl-exp-p2v2c-1010-093536/out/babylm/runs/_ptv2_bf/a2_A2a/` | 526 MB |
| a2_A2b, a2_A2c, a2_A2d, a2_A2e (39 files each) | same manifest; twins under `${BUCKET}/babylm/vms/bl-a2-screen-p2-1010-040235/out/babylm/runs/` | 154 MB each |

## Project-drive files (old drive `shared/babylm/attacks/`, absent from this container)

| artifact | where | size |
|---|---|---|
| A3 band and run files, `reads/calib_a3.txt` (copy kept in `results/gcp_reads/`) | `shared/babylm/attacks/A3/` (excluded from the delete manifest); VM twins above | unknown |
| A4 runs (curves, per-token dumps) | `shared/babylm/attacks/A4/<run>/`; VM `bl-a4-screen-1010-235629` out/ | unknown |
| A4 synthetic corpora (syn_a4a/n1a/n2a/a4b token files; generated text) | VM `bl-a4-screen-1010-235629` out/ `babylm/runs/a4_data/` (meta and sha256 in `results/gcp_reads/a4_b1/a4_data/build/`) | a4a 29.86M words; a4b plan 19.91M words |
| A4 generators G, G_A, G_B (checkpoints) | same VM out/ `babylm/runs/a4_G*`, result.json in `results/gcp_reads/a4_b1/` | unknown |

## Source trees

| artifact | where | size |
|---|---|---|
| A1 branch (self_distill, tests, PR #55) | `wip-branch` @ ref-0003 | none |
| A2 branch (input_noise, PR #60) | `wip-branch` @ ref-0004 | none |
| A3 branch (bpe_dropout, PR #61) | `wip-branch` @ ref-0005 | none |
| A4 branch | `wip-branch` @ ref-0006 | none |
| Stacker branch | `wip-branch` @ ref-0007 | none |
| Executor reads (all) | `wip-branch:babylm/gcp/reads/` | none |
