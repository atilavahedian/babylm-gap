# GLUE audit, 10 Oct 2026 22:55Z (critic ask ref-0343: every earlier GLUE number made in concurrent mode must be 7/7 before anyone cites it)

Scope: every `eval_finetune_*.log` and `finetune/<task>/results.txt` on every origin branch (git fetch 22:52Z).
GCS-only outputs that were never pushed to git are not covered.

| run | branch / path | mode | tasks finished | verdict |
|---|---|---|---|---|
| certify v1: pub gpt2, gptbert causal/mixed/masked | ij746t babylm/results/certify/score_pub_*/ | sequential | 0/7 (all 7 tasks: `ModuleNotFoundError: wandb`) | no GLUE number |
| AMLM winner hard_decay_off s0 | o6j641 results/babylm/winner/train_hard_decay_off_seed0/ | sequential | 0/7 (same wandb error) | no GLUE number |
| certify_a_v3 partial_a (bl-certify-a-v3-1010-024900) | babylm/results/certify_v3/partial_a/ | sequential | 0 results.txt (cut by 3h max-run-duration mid-GLUE) | no GLUE number |
| certify_a_v3 preempted (bl-certify-a-v3-1010-203235) | babylm/results/certify_v3/certify_a_v3_preempted/ | concurrent | 1/7 (MRPC only; 5x OutOfMemoryError, 1x CUBLAS alloc) | GLUE_INCOMPLETE, never averaged |

Result: no complete (7/7) GLUE number from our pipeline exists yet in git, concurrent or sequential. Any GLUE figure in a
memo for our own runs has no backing file and must not be cited. From ref-0339 on, vm_job.sh enforces 7/7 (else
sequential fallback, else GLUE_INCOMPLETE_<name>).

## GCS sweep (executor 3, 22:53Z: all 3 sprint1 buckets, every eval_finetune_*.log and finetune/**/results.txt)
GLUE ever ran only on bl-certify-b-v2-1009-195340, bl-certify-a-ewok-1009-195310, bl-amlm-hard-decay-off-seed0-1009-215759
(all 56-line logs, no results.txt), bl-certify-a-v3-1010-024900 (cut mid-GLUE, no results.txt) and bl-certify-a-v3-1010-203235
(OOM, 1/7). No GLUE run in our buckets ever reached 7/7. score_hf_port_s0 and score_pub_gpt2_v2 left no finetune logs there;
runs in other buckets (winner runner, experimenter) are not covered.
