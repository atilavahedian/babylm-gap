# GLUE certification: published GPT-BERT, official BOS pooling (11 Oct 06:15Z)

Source: certify_a_v3_rest (bl-certify-a-v3-rest-1010-225030, ruler ref-0339, FT_PARALLEL=0 so the official sequential script ran, A100 40GB). The outputs are on byi6gg @ref-0398, babylm/results/certify_v3/certify_a_v3_rest/. All values below are x100.

| task (metric) | causal-focus ours | published | mixed ours | published |
|---|---|---|---|---|
| BoolQ (acc) | 68.07 | 68.07 | 66.67 | 66.67 |
| MNLI (acc) | 46.86 | 46.86 | 53.30 | 53.30 |
| MRPC (F1) | 83.44 | 83.44 | 84.56 | 84.56 |
| QQP (F1) | 72.03 | 72.03 | 72.24 (acc 76.56) | 76.56 * |
| MultiRC (acc) | 68.52 | 68.28 | 67.08 | 67.04 |
| RTE (acc) | 56.12 | 56.12 | 55.40 | 55.40 |
| WSC (acc) | 65.38 | 65.38 | 63.46 | 63.46 |

Result: 12 of 14 cells are exact to 2 decimals. MultiRC is the only miss, by +0.24 (causal) and +0.04 (mixed).

(*) The published mixed QQP cell (76.56) equals our QQP accuracy (76.557) to the rounding, not F1. The README uses F1 for QQP everywhere else, including the causal row, so the published mixed row appears to report QQP accuracy. Our numbers match the published ones under either reading.

Verdict: the official GLUE pipeline reproduces the published GPT-BERT GLUE within tolerance. The only deviation is MultiRC at 0.24 or less. This is the experimenter's release gate for the GPT-BERT reference training runs.

Caveat: GPT-BERT's AutoModel is bidirectional, so BOS pooling sees the whole input ("whole-input pooling"). This certifies the fine-tune path, not causal pooling. The GPT-2 last-token run in certify_b is the causal-pooling proof. Published GPT-2 under official BOS pooling is majority-class on 7/7 (certify_b), because position 0 of a causal model sees one token.
