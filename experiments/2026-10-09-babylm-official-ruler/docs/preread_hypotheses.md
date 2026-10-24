# Pre-read predictions (written before any correct-format score)

Registered 2026-10-09 ~20:40Z, while the correct-format rescoring was running. A first pass in the wrong text format (quarantined, never reported) suggested directions, so these are not blind; they are fixed now so the corrected numbers can contradict them. Gap = 10M-track final minus 100M-track final, dev sample, nats/word.

| id | hypothesis (cause) | instrument | prediction that would confirm it |
|---|---|---|---|
| H1 | lexical exposure: mid-frequency words are under-learned at 10M | gap by train_10M frequency | words seen 16 to 1024 times carry at least 40% of the gap |
| H2 | weaker abstraction: the 100M model generalises better to new contexts, it does not just store more n-grams | gap by trigram coverage | trigrams in neither corpus carry at least 40% of the gap; "only in 100M" carries less than half |
| H3 | repetition penalty (known: Muennighoff et al. 2023) | 10M run vs 100M run at equal words seen | fresh text beats repeated text and the penalty grows with epochs |
| H4 | late epochs memorise | epoch curve, train vs dev | epochs 7 to 10 buy under 10% of the epoch 1 to 10 gain; train minus dev gap grows each epoch |
| H5 | in-context fast mapping fails at 10M | gap on repeated vs first occurrences of rare words | lag larger on repeats of rare words, but under 5% of the total gap |
| H6 | overconfidence | best single temperature | best T above 1 for the 10M final, worth under 0.05 nats/word |
| H7 | few contexts per type (registered 2026-10-09 ~21:45Z, before computing dispersion) | gap on first occurrences in the chunk, by dispersion of the word over 1000-word blocks of train_10M (observed range / range expected at its count), terciles within each frequency band 8 to 1023 | the most bursty tercile carries a per-word gap at least 20% larger than the most spread tercile, in both families |

Recipe confounds to report with every number: the GPT-2 baselines use AdamW at lr 5e-5 for 21.3k (10M) vs 211.7k (100M) steps, so part of their gap is optimisation; GPT-BERT 100M has 120M parameters vs 31M at 10M and an 8x larger batch. A bottleneck counts only if it shows in both families.
