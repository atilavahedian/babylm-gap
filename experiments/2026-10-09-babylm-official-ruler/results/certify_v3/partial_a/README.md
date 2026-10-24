Partial certify_a_v3 output (bl-certify-a-v3-1010-024900, commit ref-0402). Cut by GCE max-run-duration 3h at 05:49Z during the sequential GLUE finetune of score_pub_gptbert_causal; fast + full zero-shot complete; score_pub_gptbert_mixed never started. Copied by GPU executor 3.

## Zero-shot read-out (ruler runner, from eval_zero_shot_pub_gptbert_causal_causal.log; causal backend, full EWoK)

| task | ours | published (causal-focus, causal) | diff |
|---|---|---|---|
| BLiMP | 71.66 | 71.66 | 0.00 |
| Supplement | 63.21 | 63.21 | 0.00 |
| EWoK (full, sha-checked) | 49.49 | 49.49 | 0.00 |
| Entity tracking | 34.55 | 33.96 | +0.59 |
| COMPS | 52.83 | - | |
| WUG adj / past (rho) | 0.59 / 0.13 | - | |
| Reading (eye tracking / SPR) | 9.89 / 3.45 | - | |

GLUE and mntp were not reached; they are re-run in the certify_a relaunch.
