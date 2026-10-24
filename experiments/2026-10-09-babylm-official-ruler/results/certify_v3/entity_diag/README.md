# Entity tracking +0.59 vs published (published GPT-BERT causal-focus, 10M)

Published (pipeline README, added at ref-0284, 8 May 2025): 33.96. Ours on A100 (certify_a_v3 partial): 34.55.

| run | pipeline | data | device | entity |
| --- | --- | --- | --- | --- |
| certify_a_v3 | bf55c1131e53654c2a87418f5629c26959acd710 | OSF v2 (9,483 items) | A100 | 34.55 |
| local | bf55c1131e53654c2a87418f5629c26959acd710 | OSF v2 | CPU fp32 | 34.55 |
| local | ref-0284 (baseline-era code) | OSF v2 | CPU fp32 | 34.55 |

Per-subtask reports are identical (`cpu_*_report.txt`). Neither code version nor GPU/dtype explains the gap
(critic ref-0297 case iii). The model weights have been unchanged since 6 May 2025; the 7 May modeling patch is
non-numeric (`+=` to `x = x + ...`, a default arg, base_model_prefix).

Cause candidate: OSF replaced all three entity files on 15 May 2025 (version 1, 7 May: 53,904 items; version 2,
15 May: 9,483 items, the current filtered set). The published number predates v2. Job
`entity_v1_pub_gptbert_causal` scores the published model on v1 (sha256-pinned):
ambiref ref-0399..., move_contents ref-0400..., regular ref-0401....
