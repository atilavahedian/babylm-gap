# Attack A4: closed-system re-expression (bold bet on bottleneck 1; card, 2026-10-10)

Status: card only. No code, no runs. Waits for critic acceptance and the prior-art check, then the base round-1 winner.
Owner once accepted: a dedicated attack thread (max effort). Experimenter: research experimenter thread.

**Known family, not new.** Synthetic bootstrapped pretraining (Yang+ 2025, arXiv 2509.15248: a synthesiser trained on
the corpus writes related documents; captures 42-58% of a 10-20x unique-data oracle at 3-6B, growing with scale);
sequence-level / self-distillation via samples (Kim & Rush 2016; born-again nets); BabyLM 2025 paper 25.2 (synthetic
text from contrastive decoding, 100M track, 10 seeds: PPL 24.46 -> 23.73, about -0.03 nats/token: our pessimistic
anchor). Knowledge needs varied phrasings to generalise (Allen-Zhu & Li 2023, Physics of LMs 3.1: 1 phrasing per fact
is memorised but not extractable; 5 paraphrases lift out-of-format extraction from ~10% to ~90%). Candidate twists,
pending the prior-art check: novelty-forced generation (8-gram copy-blocking against train_10M, or cross-fitted
generators), at strict-small scale, under exposure-counted R0.

## Why (bottleneck 1: known words in new contexts, ~0.40 nats/word)
On text both models trained on, GPT-BERT 10M is only 0.24 nats/word behind 100M; the other 0.61 appears only on new
text. Content words carry 73% of the gap. Words concentrated in few documents (bursty) have a 1.29x [1.21, 1.37] larger
gap than evenly spread words of the same frequency (instruments `dispersion.json`), while left-neighbour diversity does
not matter (0.91). Late epochs memorise: their novelty tilt is 0.43 vs 1.04 for 10x data, and epochs 7-10 buy 6% of
the dev gain. Reading: the 10M model ties each word's knowledge to the few documents and phrasings it saw. Fresh data
helps because it re-expresses the same words in new documents and frames. Ensembles help bottleneck 1 little (the
GPT-2 + GPT-BERT mix gains ~0.06-0.09 per word on 64+ bands, under 10% of the 1.07 per-word gap), so this is mostly
bias, not variance: an attack must add new contexts, not just average models. A4 asks whether a model trained only on
train_10M can write those new contexts for itself.

## Mechanism
1. Generator G = the base at its selected recipe (its training exposure counts under R0).
2. Seeds: the first 64 tokens of train_10M windows (document-aligned where possible).
3. G writes a 192-token continuation per seed, sampled at T = 1.0 (calibrated, not greedy), with **novelty forcing**
   so it re-expresses instead of reciting: arm A4a blocks any sampled token that would complete an 8-gram present in
   train_10M (rejection with that token masked); arm A4b uses two cross-fitted generators instead (G_A trained on
   document half A writes for seeds from half B, and vice versa, so neither can recite its seed's document).
4. The student continues from G's weights with a second LR cycle over the spare budget, on a mix of re-expressed text
   and real text. Total exposure (G's training + generators + every fed synthetic and real word) <= 100M whitespace
   words, logged. Default split if the base selects E* real epochs: spare = 100M - 10M x E*, half synthetic, half real.
   Generation is inference (KV cache): about +10% FLOPs, logged against the 1.25x rule.

## Null twins (same seed, data order, schedule and total exposure)
- **N0' (binding):** the same second LR cycle on real text only. It separates "re-expressed text" from "a warm
  restart plus more real epochs".
- **N1:** unseeded samples from G (same volume, temperature and copy-blocking). Does seeding by training contexts
  matter, or is this plain self-distillation through samples?
- **N2:** A4's own synthetic text with words shuffled within each sentence (same words and topics, no syntax). Does
  coherent re-expression matter, or only topical word exposure?

## Kill rule (registered before any run)
Size S, the base round-1 winner, seed 0. Metric: full-DEV nats/word from `dev_bpw.py` on the final checkpoint. **Kill**
if the best arm's gain over N0' is below 2x the critic's seed floor, or if it does not beat N1 and N2 by 2x the floor.
Kill also on a >= 2 sigma BLiMP-fast drop, or > 1.25x FLOPs without a compute-matched reference. No verdict before the
floor is registered. **Mechanism check** (reported, not a kill): the instruments' signature of the best arm vs N0'
should look like data, not like epochs. Novelty tilt >= 0.8 (10x data sits at 1.04, extra epochs at 0.43), with the
gain landing on words seen 64+ times on new text. If it passes, promote one arm (picked on DEV) to 3 seeds, size M and
official eval under R0-R6.

## Odds (guesses, stated plainly)
Ceiling: if synthetic words were worth as much as fresh real words, 30M of them would be worth ~0.5 nats/word. That's
the default split at E* = 4, i.e. 40M unique-equivalent words, two doublings at ~0.26 each by log-linear interpolation.
Reaching 0.2 needs each synthetic word worth about a quarter of a fresh real one (17.5M equivalent): SBP got more than
that at 3B, while BabyLM 25.2 got far less at small scale. Guess: +0.03 to +0.15
nats/word, P(passes kill) ~0.15, P(>= 0.2) ~0.05. A null result is informative: it says a 10M-scale model cannot
bootstrap new contexts for itself, and that bottleneck 1 needs a different mechanism.

## Rules note for the critic
R0 counts synthetic words as exposure, our reading of the 2025 exposure cap. The official FAQ also says closed-system
augmentation may go past 100M words ("training two LMs on half of the words, and then having them generate more words
and training a model on both the original data and the new one is legit"; facts_2025.md section 5). If A4 shows a
real gain under R0, a separately labelled arm with synthetic volume past 100M would need the critic's and Atila's
call. It is not part of this card.

## Screen (small first; size S, seed 0, about 1.5 A100-hours)
Runs: N0', A4a, A4b (+ 2 half-data generators), N1, N2, with the base round-1 winner as G. Plus generation (KV cache,
~10 A100-minutes per arm). Every run logs exposure by source (real / synthetic / generator), tokens/s and FLOPs.
Outputs go to shared/babylm/attacks/A4/<run>/ with curves CSV + PNG, as all runs do.
