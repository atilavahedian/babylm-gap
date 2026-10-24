# Full-DEV band split: base_S_wd0p4_lr0p5_r1_ve0_tied

10,418,979 dev words (10,418,978 scored in every run), 5,210 blocks of 2,000 words, 1000 paired block bootstraps. Bands = count of the word's normal form in train_10M (0 / 1-63 / 64-8k / 8k+). Unseen-chars-excluded rows drop the 10,277 dev words containing a character absent from the train_10M raw text (critic ref-0001; the CHAT markup words, ⌊⌋⌈⌉↗∆∙⁎, are a subset); markup-excluded rows are the older labelled second number. Neither is the ranking number.

Headline, registered instrument (official scorer on the fp32 HF export, score.json dev_nats_per_word_twin (v2, blank-line twin)): base_S_wd0p4_lr0p5_r1_ve0_tied 5.0023. For information, v1 dev_nats_per_word: base_S_wd0p4_lr0p5_r1_ve0_tied 5.0367. Harness all-words figures (the table's instrument): base_S_wd0p4_lr0p5_r1_ve0_tied 5.0406.

| slice | share | base_S_wd0p4_lr0p5_r1_ve0_tied (nats/word) |
|---|---|---|
| all | 1.0000 | 5.0406 |
| band 0 | 0.0176 | 21.2835 |
| band 1-63 | 0.0638 | 11.4798 |
| band 64-8k | 0.3458 | 6.3345 |
| band 8k+ | 0.5729 | 3.0445 |
| rare (count <= 63) | 0.0813 | 13.5975 |
| after-rare (3 words, not rare) | 0.1848 | 4.6817 |
| focus (rare + after-rare) | 0.2662 | 7.4062 |
| frequent, not after-rare | 0.7338 | 4.1825 |
| all, unseen-chars excluded | 0.9990 | 5.0096 |
| band 0, unseen-chars excluded | 0.0166 | 20.3809 |
| unseen-char words | 0.0010 | 36.4689 |
| all, markup-excluded | 0.9991 | 5.0108 |
| band 0, markup-excluded | 0.0166 | 20.4189 |
| markup words | 0.0009 | 36.4355 |
| source bnc_spoken | 0.1202 | 5.0642 |
| source childes | 0.2607 | 4.0101 |
| source gutenberg | 0.2706 | 5.4716 |
| source open_subtitles | 0.1993 | 5.5815 |
| source simple_wiki | 0.1349 | 5.4444 |
| source switchboard | 0.0142 | 4.1268 |

- base_S_wd0p4_lr0p5_r1_ve0_tied: tokens_seen 159580160, words_seen 99541960, harness full-DEV 5.040629947673423, official dev_bpw 5.036669336147283
- Execution condition (critic ref-0027): base_S_wd0p4_lr0p5_r1_ve0_tied: gpu NVIDIA A100-SXM4-40GB, 366223.25566368445 tok/s. Same condition: yes (control only, nothing to compare).
