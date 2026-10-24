# Pre-read: provisional top 3 bottlenecks (published 2025 baselines, plus our port)

2026-10-10, instruments thread. Replaces the 2026-10-09 version: document-cluster CIs, the critic's format rulings (ref-0259), train_100M overlap removal (ref-0260), band shares, the new-text split by frequency, and our own port pair. This is a pre-read, not a verdict; the research experimenter owns the final ranking.

Published baselines: dev sample of 138,745 words in 300 documents of the official dev split (no eval items), each model in its own training text format; GPT-BERT 10M gives 3.109 nats/token here vs 3.10 on its model card. Our port: the experimenter's DEV-sub (1,199 documents) through `babylm/instruments/harness.py`. Gap = 10M final minus 100M final, nats per whitespace word. 95% CIs resample documents (stratified by source where sources are reweighted). Tables and figures: `results/babylm/preread_v1/` (start with `prize_anatomy.png`) and `results/babylm/harness_port/`.

## The prize

- **GPT-BERT causal 0.846 [0.786, 0.914]** (5.020 vs 4.174), whitespace collapsed as its reference code does; as in the files 0.844 [0.785, 0.911].
- **GPT-2 1.071 [0.994, 1.157]** (5.505 vs 4.433), all speaker-label tabs to spaces; per-source tabs 1.076 [0.999, 1.162]; all tabs kept 0.947 [0.871, 1.040].
- Reweighted to the dev split's source shares: 0.877 [0.831, 0.941] and 1.124 [1.059, 1.205]. Removing the 0.17% of words in lines found verbatim in train_100M moves neither by more than 0.001. Speaker labels (`*CHI:`, `A:`; 5.4% of words) carry 1.3% and 2.2%.
- GPT-BERT 10M is 0.485 better than GPT-2 10M in these formats (0.413 [0.382, 0.445] in the first-scored formats the slice tables use).

Where it sits, by how often the word occurs in train_10M (overlap removed, reweighted; `prize_bands.csv`):

| word seen in train_10M | dev words | GPT-BERT: share of gap, nats/word | GPT-2: share of gap, nats/word |
|---|---|---|---|
| never | 1.8% | 12% [9, 16], 6.11 | 12% [9, 16], 7.85 |
| 1 to 63 times | 6.5% | 20% [19, 21], 2.71 | 26% [24, 27], 4.41 |
| 64 to 8k times | 34.7% | 43% [40, 45], 1.08 | 41% [38, 43], 1.32 |
| 8k+ times | 57.0% | 25% [23, 26], 0.38 | 21% [20, 23], 0.42 |

Seen vs new text: on train text both models saw 10 times, GPT-BERT's gap is 0.240 [0.212, 0.266]. The other 0.606 appears only on new text: 0.384 [0.354, 0.414] (45% of the prize) on words seen 64+ times and 0.216 [0.169, 0.272] (26%) on words seen under 64 times (per band, dev word share x (gap per dev word minus gap per train word)). GPT-2's gap is mostly on seen text (0.683 [0.637, 0.728], 64%; its lr 5e-5 under-trains); its new-text parts are 0.203 [0.166, 0.242] and 0.150 [0.101, 0.209].

## Provisional top 3

| rank | bottleneck | size: GPT-BERT (GPT-2; port at best epoch) | attack families (all known) | prize x chance |
|---|---|---|---|---|
| 1 | Frequent words in new contexts | 0.384 [0.354, 0.414], 45% (0.203, 19%; 0.209, 21%) | tuned regularisation (Kim 2025 weight decay, the critic's comparator; dropout), noised, recombined or paraphrased repeats, ensemble then distil (Baby Llama 2023, Kim 2025), objectives that force abstraction | about 0.38 x 0.3 |
| 2 | Rare words | 0.265 [0.219, 0.324], 31% (0.391, 37%; 0.319, 31%); never-seen words 0.102 of it | share strength across word forms (subword regularisation such as BPE-dropout, character or morphology-aware input), frequency-aware regularisation | about 0.16 x 0.2 (never-seen words mostly out of reach) |
| 3 | Headroom left in the 10M data | three cheap add-ons together +0.125 [0.112, 0.142] (0.146; 0.160) | kNN-LM, cache and n-gram interpolation (Grave 2017, Khandelwal 2020), checkpoint averaging or EMA, ensembling or distilling them into one model | about 0.125 x 0.7; overlaps 1 and 2 |

**1. Frequent words in new contexts.**
- Per frequent word, GPT-BERT 10M is 3x further behind on new text than on train text: 0.63 vs 0.21 nats (1.06 vs 0.34 for words seen 64 to 8k times, 0.38 vs 0.14 at 8k+).
- The gap as a whole sits in meaning-bearing text: content words are 44% of words but carry 73% of the gap (1.41 vs 0.41 nats/word for function words; GPT-2 75%; port 75%). Trigrams in neither corpus carry 42% (GPT-2 43%, port 45%); trigrams only in the 100M corpus 21% and 20%. Punctuation tokens carry 11%, almost all on new text.
- Most of it is not local statistics or recall: a trigram count model of the 100M words supplies 0.162 [0.143, 0.184] (19%; GPT-2 0.246), and 16-token greedy span recall is under 0.4% for every model.
- It looks like overfitting: dev minus train loss is 2.0 to 2.6x the 100M model's in every band seen 8+ times (GPT-2 1.7 to 2.4x). GPT-BERT 10M is overconfident (loss minus its own entropy 0.57 vs 0.39; GPT-2 0.36 for both).
- More epochs do not buy it (H4). Epochs 7 to 10 buy 6% (GPT-BERT, words in the first 128 tokens) and 2.4% (GPT-2) of the epoch 1 to 10 gain: 0.016 [0.013, 0.019] and 0.011 [0.009, 0.012] nats/word per epoch, about 50 and 100 more epochs to reach the 100M final at that rate. GPT-BERT words seen 8k+ times are flat after epoch 7 (0.000 [-0.002, 0.003] per epoch) while their train loss keeps falling. Dev minus train grows every epoch, 0.51 to 1.11 (GPT-BERT) and 0.38 to 0.77 (GPT-2), against 0.51 and 0.38 for the 100M finals. What late epochs do buy is tilted to contexts already in train_10M (novelty tilt 0.43 [0.30, 0.60] and 0.66 [0.58, 0.75]); 10x data is not (1.04 and 1.05). Fresh text beats 10 repeats at equal steps by at least 0.502 [0.439, 0.581] (GPT-2).
- Overfitting destroys this first (port, below: novelty tilt 3.40 from epoch 2 to 5).
- Cause hypothesis: repeats sharpen predictions for the contexts in train_10M instead of building representations that transfer (our reading: missing abstraction).

**2. Rare words.**
- 0.216 of the 0.265 shows only on new text: on train text GPT-BERT 10M trails by 0.62 nats per rare word, on new text by 3.46 (2.69 for words seen 1 to 63 times, 6.45 for never-seen words).
- These words are still learning late: words seen 1 to 63 times gain 0.084 [0.062, 0.107] nats/word per epoch in epochs 7 to 10 (GPT-2 0.058). 10x data buys them first (rare tilt 1.83 [1.69, 2.02]; GPT-2 2.31).
- Bunching matters a little (H7): at equal frequency, words whose train_10M uses sit in few 1000-word blocks carry a 1.29x [1.20, 1.38] larger gap per word than evenly spread ones (GPT-2 1.15x [1.07, 1.23]), about 0.02 nats/word in total. At equal frequency (16 to 255 times), plain lowercase words split into several tokens carry 1.1 to 1.6x the gap of single-token words (GPT-2 1.4 to 1.6x; small counts, no CI).
- Split for the attack choice (`rare_split.csv`, `rare_split.png`; same protocol as the band table, bottleneck 2 = 0.284 [0.245, 0.339]). Repeats of a word earlier in its chunk carry 19% [17, 22] of it (GPT-2 32% [29, 36]). Words that the 10M tokenizer cuts into a rare piece (a token seen under 64 times in train_10M) carry 17% [10, 24] (GPT-2 16%). First occurrences of never-seen words carry 33% [26, 40] (GPT-2 23%). The largest cell is first occurrences of words seen 1 to 63 times made only of frequent pieces: 44% [37, 51] (GPT-2 41%). GPT-BERT 10M's gap per word on repeats is no larger than on first occurrences (2.60 vs 2.74 for words seen 1 to 63 times), so in-context word learning is not its deficit; GPT-2's is (6.18 vs 3.85). The two models of a family use different tokenizers, so this is a split of words, not of tokens.
- Cause hypothesis: too few contexts per type, so the model stores the occurrences instead of learning the word. It has the pieces; it lacks the word.

**3. Headroom left in the 10M data** (all mixtures cross-fitted on document halves).
- GPT-BERT 10M: count model of the same 10M words +0.051 [0.041, 0.065]; copy cache +0.037 [0.029, 0.047]; ensemble with GPT-2 10M +0.077 [0.068, 0.086]; one temperature +0.013. Together +0.125 [0.112, 0.142], against 0.165 if they simply added.
- GPT-2 10M: +0.076, +0.084, no ensemble row, 0.000; together 0.146.
- Port at its best epoch: count model +0.073 [0.066, 0.081]; cache over the whole document so far +0.110 [0.085, 0.141] (not comparable with the 360-word chunk cache above); together +0.160 [0.136, 0.193]. Its 100M twin gets only +0.020 and +0.038 from the same add-ons.
- Single models carry variance: GPT-BERT's own epoch-7 and epoch-10 checkpoints mixed give +0.040 [0.035, 0.046] (epochs 5 and 10: +0.048). GPT-2: +0.001. Port epochs 2 and 10: +0.015.
- Cause hypothesis: the network under-uses exact local statistics and earlier words. This recovers part of the loss behind 1 and 2 at test time, so it is not additive with them.

**Ruled out as main bottlenecks (both families):**
- Long context: the gap per word is flat from the 4th word to the end of 360-word chunks (0.82 to 0.87 GPT-BERT, 1.05 to 1.09 GPT-2).
- Global calibration: the best single temperature is worth 0.013 and 0.000.
- Verbatim memorisation (see above).
- Text format: whitespace tokens carry 1.3% and 2.1% of the gap, speaker labels 1.3% and 2.2%.
- In-context copying is a GPT-2-only weakness: at equal frequency, the gap on repeated words exceeds first occurrences by 0.065 nats/word in total for GPT-2 but by -0.064 for GPT-BERT.

## Our recipe: the port pair (DEV-sub)

`port_S_10M_e10_s0` (the sprint-1 port, 10 epochs of train_10M) against its twin `grid_S_100M_e10_s0` (same recipe, 10 epochs of train_100M) and a one-epoch 100M run `grid_S_100M_e1_s0`. Battery: `harness_battery.sh`; outputs in `results/babylm/harness_port/`.

- DEV-sub nats/word at epochs 1, 2, 4, 5, 10: port 5.652, **5.434**, 5.955, 6.427, 9.831; twin 4.634, 4.514, 4.464, 4.466, 4.420. One epoch of 100M words on its own full schedule (`grid_S_100M_e1_s0`): 4.481.
- **Best epoch (2, 20M words) vs the twin's final: 1.014 [0.902, 1.120]**; reweighted 1.012 [0.946, 1.079] (DEV-sub has one Switchboard document, so that source adds no between-document variance and reweighted CIs on DEV-sub are slightly narrow). Final vs final: 5.411 [4.818, 6.218].
- At epoch 2 the anatomy matches published GPT-BERT: bands 10 / 21 / 44 / 25% of the gap; content words 75%; trigrams in neither corpus 45%. 10x-data signature: rare tilt 1.66 [1.55, 1.80], novelty 1.00 [0.94, 1.05] (GPT-BERT 1.83, 1.04). On train text it is 0.642 [0.588, 0.699] behind after 2 passes vs the twin's 10; the extra on new text is 0.209 [0.135, 0.279] on frequent and 0.123 [0.088, 0.156] on rare words.
- **It overfits from epoch 2, and the damage lands on new contexts first.** Epoch 1 to 2 gains 0.218 (novelty tilt 0.55: already-seen contexts, like the published late epochs). Epochs 2 to 5 lose 0.992, with novelty tilt 3.40 [3.02, 3.83] and rare tilt 1.77 (damage relative to the loss in each slice). Epochs 5 to 10 lose 3.404 everywhere (novelty 1.35, rare 1.18). DEV-sub minus train-sub loss: 0.91 at epoch 1 (twin final 0.90), 1.27, 2.54, 3.30, 7.98. Every source overfits, simple_wiki most.
- At epoch 10 it is confidently wrong: its top logit is as high as the twin's, while the target's logit falls with the token's train_10M count; only 2.1% of targets sit at the softcap floor (7.7% of the loss).
- Fresh vs repeated text at 100M words seen: the one-epoch 100M run beats the port's final by 5.349 (rare 1.23, novelty 1.32); this measures the overfit more than the repetition penalty.
- Likely cause (sent to the experimenter and the base builder, untested): in all 8 base_s1 cells only `muon_wd` varies; `adam_wd` = 0 and dropout = 0, so the 31.5M lookup-table parameters (embedding, untied head, 3 value-embedding tables; 60% of 52.7M) get no regularisation at an embedding learning rate of 0.78. All 8 cells bottom out at epoch 2 to 3, and more Muon weight decay raises the minimum. Both published baselines run dropout 0.1 and have no value embeddings.

## Lever signatures

`signature.png`, `signature_tilts.csv`. Share of the weaker model's loss each lever removes, as two tilts (95% CI): rare = words seen 1 to 15 times vs 64+ times; novelty = trigram in neither corpus vs in train_10M.
- 10x data: rare 1.83 [1.69, 2.02] and 2.31 [2.14, 2.52]; novelty 1.04 [1.00, 1.09] and 1.05 [1.00, 1.10]. Port at its best epoch: 1.66, 1.00.
- Extra epochs: novelty 0.43 to 0.76, always under 1 (port epoch 1 to 2: 0.55).
- Fresh vs repeated text (GPT-2, equal steps): rare 3.37 [2.88, 3.95], novelty 1.64 [1.49, 1.83].
- The GPT-2 to GPT-BERT recipe change: rare 3.48 [3.12, 3.87], novelty 0.96 [0.88, 1.05], so on novelty it looks like data, not like more passes.
- Overfitting (port epochs 2 to 5, loss rising): novelty 3.40, rare 1.77.

An attack that sits near the 10x-data point is buying what data buys; one with novelty under 1 is buying what epochs buy. For a regulariser, the novelty tilt of its late epochs is a quick read on whether it stopped the overfit.

## Pre-registered predictions

`preread_hypotheses.md`. H1 not confirmed (words seen 16 to 1023 times carry 33 to 34%, predicted at least 40%). H2 confirmed (trigrams in neither corpus 42 to 43%, only in 100M 20 to 21%). H3 confirmed at 10 epochs. H4 confirmed in both families (6% and 2.4% of the gain after epoch 7, predicted under 10%; dev minus train grows every epoch). H5 mixed (GPT-2 copy lag 6% of the gap, above the predicted 5%; GPT-BERT has none). H6 mixed (GPT-BERT best T 1.1 worth 0.013; GPT-2 best T 1.0). H7 mixed (bursty words 1.29x [1.20, 1.38] for GPT-BERT, 1.15x [1.07, 1.23] for GPT-2, predicted at least 1.2x in both).

## Confounds

- The GPT-2 baselines train at lr 5e-5 (64% of their gap already shows on seen text, so part is under-training).
- GPT-BERT 100M has 120M parameters vs 31M at 10M and an 8x larger batch, so part of its seen-text gap is size.
- The count model and cache predict the word but not the whitespace before it, so their gains are slightly optimistic.
- Slice and epoch tables use the first-scored formats (critic ref-0259): they differ from the headline formats only on words after leading or repeated spaces (GPT-BERT, 0.4% of dev words) or on Switchboard (GPT-2, 1.5%).
- The harness dumps count the BOS loss in each document's first word and give a document's early words the previous document as context, so position slices on DEV-sub read high at document starts. The port numbers above use no position slice.
