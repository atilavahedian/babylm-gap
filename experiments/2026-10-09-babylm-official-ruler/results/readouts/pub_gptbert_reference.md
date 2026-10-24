# Published GPT-BERT reference rows (already-run certification jobs; nothing new run)

Pipeline: official @ bf55c1131e53654c2a87418f5629c26959acd710, transformers 4.51.3, torch 2.7.0, A100 40GB. Scorer: babylm/eval/official_readout.py model_scores (same task list and lines as every readout).

| row | BLiMP | Supp | EWoK | COMPS | WUG_adj | WUG_past | Read_rt | Read_spr | Entity | composite (8) | GLUE7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-BERT causal-focus (BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus @ref-0234) | 71.66 | 63.21 | 49.49 | 52.83 | 59.0 | 13.0 | 9.89 | 3.45 | 34.55 | 40.32 | 65.77* |
| GPT-BERT mixed (BabyLM-community/babylm-baseline-10m-gpt-bert-mixed @ref-0235) | 69.62 | 61.56 | 50.23 | 52.99 | 49.0 | -3.0 | 9.50 | 3.37 | 25.41 | 36.66 | 66.10* |
| untrained GPT-2 (ctl_untrained_gpt2, zero-shot) | not run** | not run | not run | not run | not run | not run | 9.5 | 3.4 | not run | not run | not run |
| untrained base_S (lastpool GLUE, 6/7 tasks) | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | 6-task mean 62.6 (no GLUE7) |

(*) Not comparable to our GLUE7. GPT-BERT GLUE is the official BOS (whole-input, bidirectional) pooling with a single fine-tune seed (42). Our GLUE7 uses last-token pooling and a 2-seed ensemble (ens2).
- Causal-focus per task: boolq 68.07, mnli 46.86, mrpc F1 83.44, multirc 68.52, qqp F1 72.03, rte 56.12, wsc 65.38.
- Mixed per task: boolq 66.67, mnli 53.30, mrpc F1 84.56, multirc 67.08, qqp F1 72.24, rte 55.40, wsc 63.46.
- 12 of 14 cells match the published values to 2 decimals (certify_v3/glue_certification_1011.md).

Reading: GPT-BERT's tokenizer adds the prefix space by default. So its Reading cells, and the composite built from them, are comparable to our *amended* composite: S_split 38.88 and 28L mix 38.97. They are not comparable to our registered composite, where Reading is about 0.2 / 0.0 (reading_tokenizer_check.md).

(**) The untrained GPT-2 control crashed after Reading. The tie-log path bug stopped it at comps. Only the Reading cells exist.

Sources (byi6gg @ref-0009, babylm/results/):
- certify_v3/certify_a_v3_preempted/score_pub_gptbert_causal: zero-shot, causal-focus
- certify_v3/certify_a_v3_rest/score_pub_gptbert_mixed: zero-shot and GLUE, mixed
- certify_v3/certify_a_v3_rest/ftverify_pub_gptbert_causal/eval/finetune_sequential: GLUE, causal-focus
- This branch, babylm/results/certify/score_controls: untrained GPT-2
- final_table_lines/glue_lastpool_certification.md: untrained base_S
