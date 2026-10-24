# certify_a_v3, preempted VM (executor 3, 2026-10-10 22:50Z)
Source: ${BUCKET}/babylm/vms/bl-certify-a-v3-1010-203235/ (spot-preempted 22:37Z).
Copied: every text output under out/ (eval/ = eval2025 results, entity_v1/, job logs and meta) plus VM logs/, startup.log, job.json.
Left out: predictions.json files (1-7.6 MB each; stay on GCS) and any weights.
Complete jobs: score_pub_gptbert_causal (rc=0, 5177 s), entity_v1_pub_gptbert_causal (rc=0, EWoK avg 33.96).
Partial (lost on preemption): score_pub_gptbert_mixed. Rerun with ftverify_pub_gptbert_causal on bl-certify-a-v3-rest-1010-224052.
