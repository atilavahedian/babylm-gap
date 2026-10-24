# Artifacts: BabyLM official ruler

Large or excluded files live in GCS or in the archive tags. The GCS paths come from the executor ledger and job files.
Their existence was **not re-verified** on 2026-10-13: this container has no GCS credentials. Never copy EWoK data out
of its private bucket.

| artifact | where | size |
|---|---|---|
| Full EWoK zip (private; never vendor) | `${BUCKET}/ewok/ewok-core-1.0_ref-0155.zip` | - |
| Published checkpoints certified | HF `BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus` @ref-0234, `...-gpt-bert-mixed` @ref-0235, `...-gpt2` (revisions in `results/**/published_revision.txt`) | - |
| base_S_e6 HF export (lastpool GLUE subject) | `${BUCKET}/babylm/hf/{base_S_e6,base_S_e6_lastpool}/` (per `common/babylm/gcp/HANDOFF.md`) | - |
| Pipeline predictions.json / prediction.jsonl (1-7.6 MB each; may include eval items) | `wip-branch:babylm/results/certify_v3/**` and `wip-branch:babylm/results/certify/**`; also each VM's out/ below | 45 MB total across dropped files |
| Raw eval logs (eval_finetune_*.log up to 6.7 MB, ties_*.jsonl, startup logs) | same archive paths (`certify`, `certify_v3`, `ftverify`, `zs_cpu`) | incl. above |
| Instruments per-document tables with DEV text excerpts (`devfull_docs.csv`; BabyLM corpus text, LDC/TalkBank-licensed sources: kept out) | `wip-branch:results/babylm/base_r1/devfull_docs.csv`, `.../packing_pairs/devfull_docs.csv` | 2.7 MB, 3.1 MB |
| Instruments gap_*.png figures (CSV twins kept in results/instruments) | `wip-branch:results/babylm/{harness_port,preread_v1}/gap_*.png` | ~7.4 MB |
| Instrument per-word score dumps | `data/babylm_scores/` on the instruments machine (not in git; GCS location not recorded) | - |

## Certification VM outputs (out/ prefix synced by the executor)

| artifact | where | cost |
|---|---|---|
| smoke VM `bl-smoke-1009-185524` outputs | `${BUCKET}/babylm/vms/bl-smoke-1009-185524/out/` | smoke test; VM ran 17 s after boot then self-deleted; log unreadable (storage blocked) |
| certify_a VM `bl-certify-a-1009-192735` outputs | `${BUCKET}/babylm/vms/bl-certify-a-1009-192735/out/` |  |
| certify_b VM `bl-certify-b-1009-193122` outputs | `${BUCKET}/babylm/vms/bl-certify-b-1009-193122/out/` |  |
| gptbert_causal_seed0 VM `bl-gptbert-causal-seed0-1009-195204` outputs | `${BUCKET}/babylm/vms/bl-gptbert-causal-seed0-1009-195204/out/` | run_job.py not found (ruler bug) |
| certify_a_ewok VM `bl-certify-a-ewok-1009-195310` outputs | `${BUCKET}/babylm/vms/bl-certify-a-ewok-1009-195310/out/` | deleted 21:52Z at ruler request (ref-0394 eval bug); DEV bpw valid |
| certify_b_v2 VM `bl-certify-b-v2-1009-195340` outputs | `${BUCKET}/babylm/vms/bl-certify-b-v2-1009-195340/out/` | deleted 21:52Z at ruler request (ref-0394 eval bug); DEV bpw valid |
| certify_a_v3 VM `bl-certify-a-v3-1010-024900` outputs | `${BUCKET}/babylm/vms/bl-certify-a-v3-1010-024900/out/` | certify_a_v3 CUT by max-run-duration 3h during score_pub_gptbert_causal GLUE finetune (fas |
| certify_a_v3 VM `bl-certify-a-v3-1010-203235` outputs | `${BUCKET}/babylm/vms/bl-certify-a-v3-1010-203235/out/` | PREEMPTED 22:37Z after score_pub_gptbert_causal + entity_v1_pub_gptbert_causal (rc=0); sco |
| certify_a_v3_rest VM `bl-certify-a-v3-rest-1010-224052` outputs | `${BUCKET}/babylm/vms/bl-certify-a-v3-rest-1010-224052/out/` | stopped by executor at ruler's request 22:49Z: concurrent GLUE OOMs on 40GB A100 for GPT-B |
| certify_a_v3_rest VM `bl-certify-a-v3-rest-1010-225030` outputs | `${BUCKET}/babylm/vms/bl-certify-a-v3-rest-1010-225030/out/` | certify_a rest done (pub_gptbert_mixed GLUE seq + ftverify_pub_gptbert_causal sequential) |
| certify_b_v3_ewok VM `bl-certify-b-v3-ewok-1011-001059` outputs | `${BUCKET}/babylm/vms/bl-certify-b-v3-ewok-1011-001059/out/` | CUT by experimenter 09:34Z: score_pub_gpt2 hit 6h timeout (lastpool concurrent OOM, 2/7 off-majori |
| ftverify-base-s-e6-lastpool VM `bl-ftverify-base-s-e6-lastpool-1011-101001` outputs | `${BUCKET}/babylm/vms/bl-ftverify-base-s-e6-lastpool-1011-101001/out/` | ftverify done (MISMATCH multirc only); untrained half cut at maxdur |
