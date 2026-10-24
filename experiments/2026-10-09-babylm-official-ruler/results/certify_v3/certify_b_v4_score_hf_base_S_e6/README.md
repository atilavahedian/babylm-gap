# certify_b_v4 job 1: score_hf base_S_e6 (executor 3, 2026-10-11 01:55Z)
VM bl-certify-b-v3-ewok-1011-001059 (project-c, job certify_b_v4 @ref-0395, FT_PARALLEL=0 sequential official GLUE). HF = v2_S_10M_e6_s0 export.
GLUE 7/7 tasks finished (00:36-01:40Z). results.txt:
boolq acc .6404 f1 .7808 mcc 0 | mnli acc .3570 | mrpc acc .6814 f1 .8105 mcc 0 | multirc acc .5755 f1 0 mcc 0 | qqp acc .6278 f1 0 | rte acc .5396 f1 0 | wsc acc .6154 f1 0
Every binary task has mcc 0 (constant prediction; accuracies sit at the majority-class rate). Likely cause: the official classifier pools encoding[:,0] = BOS, which a causal model computes without context (export_hf docstring; the lastpool adapter exists for this). Mechanically complete, not a meaningful score.
Copied: all text outputs except predictions.json; plus the VM job log.
