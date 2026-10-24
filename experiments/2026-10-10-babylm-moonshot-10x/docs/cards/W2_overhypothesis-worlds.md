<!-- moonshot design round W2; status alive; ceiling 0.15 nats/word; expected 0.003; P(pass screen) 0.02; screen GPU 105 A100-min. Draft card: not yet submitted to the critic. -->

# Attack: overhypothesis-worlds (card, 2026-10-10; folds the prior-art, rules and mechanism skeptics)

Status: card only. No code and no runs. It needs a critic R0.1 ruling (Gate R) before any GPU run. All three skeptics returned "wounded", none "dead", and every rule problem below has a workable fix, so the card is alive. It is a low-odds bet that is cheap to falsify, and it runs only through the CPU and mini-ruler gates.

**Changes from the candidate.**
- Arm b (in-corpus HSP placeholders) is dropped. It is a known idea (Hermann 2015 anonymisation, Huang 2023 lexinvariance App. G, Minnow), its dose is about 0.45% of words with 95% in the 64-8k band, and it overlaps A2. If anyone wants it, it becomes an A2 sub-arm under an A2 amendment and is never summed with A2.
- The ceiling drops from 1.0 to 0.15.
- The base's E* is 10 epochs, not 4, so the ledger and FLOPs are recomputed (see below).
- Doses are now {0.4M, 1.5M, 4M}; the 12M dose is gone.
- Two binding nulls are added (N-a2, N-dyck), and N-fix is made binding.
- Gate 0 now uses an ideal-observer reference.
- A CPU transfer test can fail and kill the card.
- The worlds-phase and English-phase softmaxes are each restricted to their own rows, and the head is trimmed at export.
- The fade rule is in the kill rule.
- Open-class words get multi-piece spellings.

## Known ideas (label: known ideas, new combination; known family)
- **Synthetic / formal pre-pretraining.**
  - Papadimitriou & Jurafsky 2020 and 2023. Correction to the candidate: in 2023 the best bias was non-context-free crossing dependencies, and a Zipfian vocabulary helped independently. The candidate's "Zipf plus nesting transfer best" was wrong.
  - Ri & Tsuruoka 2022.
  - Hu, Petty, Shi, Merrill & Linzen, ACL 2025. k-Shuffle Dyck, about 33% token saving at 1B. At 160M on C4: 3.741 vs 3.754 for the C4 pre-pretraining control vs 3.780 with none. The optimum is about 500 steps, and longer pre-pretraining transfers less.
  - Jiang et al. 2026, Procedural Pretraining (arXiv 2601.21725). Front-loading 0.1-0.3% procedural data reaches the same loss with 55-86% of the data, up to 1.3B.
  - Lee et al. 2026, NCA (2603.10055). 164M NCA tokens give up to 6% LM gain and up to 1.6x faster convergence.
  - Riazhskykh, Luu & Bojar 2026 (2608.08800). Dyck gains "depend heavily on setup and random seed"; only one stable trend.
  - Yamaguchi et al. 2026 (2609.39827). Gains survive at 500M-7B but "not as a grammatical prior"; they are attributed to long-range retrieval.
  - BabyLM. ChapGTP 2023: emergent-language pre-pretraining +0.7 BLiMP, bracket text -0.6. Kumar, Lango & Dusek 2025: NEST/CROSS init, near chance. Yao et al. 2022: emergent-communication pre-pretraining, measured with only 2M downstream tokens.
- **Prior distillation.** McCoy & Griffiths 2025, MAML on synthetic languages. At full data PPL is 19.67 vs 19.75 (about 0.004 nats). Plain pretraining on the same languages was "substantially worse" than MAML.
- **Episode-resampled meanings and in-context learning from data properties.**
  - Lake & Baroni 2023 (MLC) and Kirsch et al. 2022 (GPICL).
  - Chan et al. 2022: burstiness, Zipf and dynamic meanings drive ICL, which trades off against in-weights learning.
  - Singh et al. 2023 (ICL is transient) and strategy coopetition 2025 (2503.05631).
  - Huang et al. 2023 (lexinvariant LMs), Akyürek et al. 2024 (ICLL) and Xie et al. 2022 (GINC).
  - Carstens-Behrens & Fröhlich 2026 (2610.05879). A fresh synthetic language is drawn per sequence; evaluation is in-context on Wikipedia with frozen weights; there is no continued-training transfer test.
- **Overhypotheses and word learning.** Kemp, Perfors & Tenenbaum 2007; Smith et al. 2002; Gillette et al. 1999; Feinman & Lake 2018; Gandhi & Lake 2020; Hill et al. 2021; Minnow (Wang, Jiang, Linzen & Lake 2025).
- **What is new here.** The exact generator was not found as LM pre-pretraining: an episode-resampled lexicon whose word properties come from a low-rank feature overhypothesis that only selectional frames reveal. It sits inside a known family, and the label moves to "known" if the prior-art check finds a match.
- **Citation status.**
  - Abstracts verified by direct fetch this turn: 2608.08800, 2609.39827, 2603.10055, 2601.21725, 2610.05879.
  - In-paper numbers come from the prior-art skeptic's reading and were not re-verified: the 2608.08800 seed spread (+1.0 / +19.0 / -11.0%), the Hu losses and the ChapGTP table.

## Mechanism
**Phase W (worlds).** The base model at size S is pre-pretrained on D tokens of procedurally generated world episodes. Then the unchanged round-1 English recipe runs with a fresh optimizer and schedule, and the global RNG and sampler are re-seeded to the base seed, so the English batch order is identical to the base's.

**Generator: `babylm/moonshot/worlds.py`.** It uses numpy only: no LM, no corpus text, no statistic fitted on train_10M.
- **Ids.** 2,048 reserved ids, 16512-18559. The model vocab is 18560 with a tied table; rows 16384-16511 are unused pad.
- **Closed class.** About 40 closed-class ids (det, prep, comp, plural and agreement markers, 'and', period) are fixed across episodes. They are never mapped to English tokens; doing so would make the stream English text and need a new card.
- **Lexicon per episode.** Each 1,024-token episode draws a fresh open-class lexicon: 48 nouns, 24 verbs and 16 adjectives, without replacement from 1,900 ids.
- **Multi-piece spellings** (mechanism skeptic fix 1). Each open-class word is spelled with 1/2/3 reserved pieces, with P = 0.5/0.3/0.2 (round numbers, declared). The last piece comes from a per-episode set of 8 'suffix' ids. Which suffix marks which feature is redrawn every episode, with 70% agreement. So binding is learned over spans, the way English rare words arrive (1-15 band: 3.0 tokens/word; 16-63 band: 1.95).
- **Features and frames.** Nouns get hidden features in {0,1}^3. A verb has a subject requirement and, 70% of the time, an object requirement on one feature. An adjective applies to one feature value.
- **Syntax and statistics.** A PCFG with depth at most 3 and number agreement (S -> NP V (NP) (PP); NP -> Det (Adj) N (Pl) (RC); RC -> Comp V NP). Zipf(1) frequencies within each category, paragraph burstiness, and 2-4 noun introductions mid-episode (fast mapping).
- **What can transfer.** Ids are redrawn every episode. Only the inference procedure can transfer: a word's properties are consistent across its contexts and are set by a few latent features that frames reveal.

**Phase W training.**
- Objective: causal only (the base is causal-only).
- Logits are computed over the reserved rows 16512-18559 only (mechanism skeptic fix 4). English rows get no gradient and stay at init. With tied embeddings, this stops the 16,384 English rows from being trained into a shared "never predicted" direction.
- Batch 8,192 tokens. Constant LR at the base's peak (lr_scale 0.5), the base's Muon/AdamW hyperparameters, no cooldown. Any dose checkpoint is then a valid endpoint: 0.4M = 49 steps, 1.5M = 183, 4M = 488 (about Hu's 500-step optimum).

**English phase.**
- Logits are computed over rows 0-16383 only, which is exactly the base's softmax.
- Export trims the tied table to 16384 rows. The HF model then has the base's shape, and no probability mass can sit on reserved rows in dev_bpw or in the official pipeline. This is declared in the export meta before scoring and applied identically to every arm and the base-18560 re-run.

**Dose D.** Gate 0 sets D as the smallest of {0.4M, 1.5M, 4M} whose synthetic probe passes. D is frozen before any English run.
- 0.4M = 0.25% of the English 159.6M tokens, inside the procedural-pretraining 0.1-0.3% band.
- 4M = 2.5%, inside Hu's band.
- 12M is dropped: it is past the family's measured optimum.

**W-mix (conditional arm, registered now).** The same D synthetic tokens go in as interleaved windows spread evenly through the English phase (2.5% of windows at 4M), each window with its own row-restricted softmax. Its twin is N-a2-mix. W-mix runs at most once per gate, and only if W's survival probe falls below 25% of its pre-English value (the Singh-style transience fix). Only one of {W, W-mix} can be promoted.

Not in this card: a MAML or meta-gradient variant (McCoy & Griffiths' stronger form). It needs its own card if W shows an in-weights signal at 1M.

## Why (which measured slice)
**Targeted slice: in-weights generalisation of familiar words to new contexts.** That is the 64-8k band (0.36 nats/word) and the 8k+ band (0.22). Content words carry 73% of the gap. The multi-piece spellings add a secondary hope for the 1-63 band (0.16-0.18). Never-seen words (0.10) are dropped as a target.

**Hypothesis.** The body acquires a low-dimensional feature-space prior plus a cross-situational binding circuit. That biases how SGD shapes English embeddings and MLP memories, so a familiar word's existing contexts determine its behaviour in new ones with fewer examples.

**Why a bigger, tuned base would not already have it.** WD, dropout, EMA and size change capacity and smoothness, not the hypothesis space. English ids are fixed, so English can always be memorised. Episode-resampled ids make memorising impossible.

**Counter-evidence, stated plainly.**
- An initialisation is not a Bayesian prior. Here it is followed by 160M English tokens (10M x 10 epochs x 1.6) at full Muon LR.
- Transient ICL (Chan 2022, Singh 2023) predicts the in-context circuit is competed away by in-weights solutions.
- McCoy & Griffiths' prior faded to 0.004 nats at full data, even with MAML.

**Step 0 premise check (run while writing this card, CPU, zero cost).**
- Script: scratchpad `ohw/step0.py`, over the full-DEV per-word dumps of the round-1 base (5.0406 harness) and T100 (4.298). T100 is the port recipe at 100M words x 10 and is an oracle (out of track). Numbers are raw, not overlap-removed.
- Gap per word, for words repeated within the previous 320 words of the same document vs words not recently seen:

| Band | Recent repeat | Not recent |
|---|---|---|
| Never seen | 3.07 | 4.70 |
| 1-63 | 1.57 | 1.82 |
| 64-8k | 0.638 | 0.720 |
| 8k+ | 0.237 | 0.322 |

- Recent repeats are 52% of words and contribute 0.21 of the 0.74 gap. Paired on the same words, the base gains at least as much from a recent occurrence as T100 does.
- This agrees with the GPT-BERT pre-read (-0.064). The in-context channel is not lagging and is capped near the copy-cache headroom (+0.037).
- So the card rests entirely on in-weights transfer, which the CPU transfer test and Gate 1 test directly.

## R0 exposure ledger (round-1 base @ref-0065: train_10M x 10 epochs, causal, tied, vocab 16384; 9,953,773 harness words/epoch = 99.54M; headroom 0.46M)
**The candidate's ledger is void.** It assumed E* = 4. At the real E* = 10, 12M + 99.54M = 111.5M, which is over R0.2.

**Gate R (before any GPU run).** One registered critic ruling on reserved-id procedural streams, applied to R0.1 and R0.2 alike (rules skeptic fix 1).

| Item | Ruling A: not words (no surface form, no LM, no corpus statistics; precedent: the prior-art map's "Legal: Yes (synthetic, no LM)" row for formal-language pre-pretraining) | Ruling B: corpus words |
|---|---|---|
| (1) Generator | 0 | 0 |
| (2) Phase W, D ids | 0, reported as a ledger row 'synthetic: D tokens' | D words |
| (3) English | 99.54M, 10 epochs unchanged | — |
| (4) Probes | inference, 0 | inference, 0 |
| R0.1 unique words | 9.95M | 9.95M + D > 10M for every useful D (headroom about 50k words) |
| Total exposure | 99.54M | — |
| Status | in track | out of strict-small track, pipeline-tier label only; arm a is killed as a 10x candidate before any GPU compute |

**Charge-to-exposure-only ruling.** If the critic instead charges ids to R0.2 only: at D = 0.4M the English phase stays at 10 epochs (99.94M total). At D >= 1.5M, English = (99.54M - D) / 9.954M epochs (9.6 at 4M), with the base schedule scaled to that token budget. Nulls get the same.

**Ledger mechanics and contamination.**
- The world bin carries a words array with 1 id = 1 word, so words_seen, milestones and the 99% cap check stay correct.
- Checkpoints are labelled by English words and by total exposure.
- No double counting: one forward per window.
- Generator parameters are round numbers or literature values, frozen by hash, and never fitted on train_10M (fitting would cost 10M under the count-model precedent).
- No BLiMP, COMPS or DEV feedback goes into generator design. The generator deliberately encodes agreement and selectional phenomena that overlap BLiMP and COMPS, which is why it is frozen before any fast-suite score.
- R5: a test asserts every synthetic id is >= 16512, so the 13-gram overlap is zero by construction.

## FLOPs (R4.4)
- Body and head cost about 1 + D / 159.6M: 1.003x at 0.4M, 1.009x at 1.5M, 1.025x at 4M.
- The phase-W head covers only 2,048 rows, which is cheaper.
- W-mix is the same.
- The model is <= 1.03x at every dose, so no FLOP-matched reference is needed. The rules skeptic's 1.19-1.37x figures came from the void E* = 4 ledger.
- On the mini-ruler (1M x 10 = 16M tokens) the multiple is up to 1.25x, so only the nulls (same dose) control compute there.

## Null twins (same generator code path, ids, dose, phase-W steps/LR/batch, English phase, seed)
- **N-a2 (binding mechanism null; mechanism skeptic fix 2).** Lexicon resampling, syntax, Zipf, burstiness and pieces are kept. Compatibility is a consistent per-episode random noun x predicate table with the same per-episode density and no feature factorisation. This removes the low-rank overhypothesis but keeps within-episode consistency and the retrieval demand.
- **N-fix (binding structural null; rules skeptic fix 5).** A fixed lexicon and fixed features across all episodes, with the same syntax and statistics. This removes the learn-to-learn information (Papadimitriou/Hu-style structural plus lexical pre-pretraining).
- **N-dyck (binding family calibration; prior-art skeptic fix 3).** Hu's k-Shuffle Dyck on the same reserved ids, dose and steps. If generic pre-pretraining matches W, the overhypothesis is not the mechanism.
- **N-a1 (diagnostic only).** Features and requirements are redrawn every sentence. It is an anti-prior, so it is never binding.
- **Tests across modes.** Per-category unigram, sentence length and depth distributions match within 2%. N-a2's density matches. N-a1 breaks cross-sentence consistency. N-fix ids are fixed.
- **Base comparators.** base-18560: the base re-run at --vocab_pad 18560 with the same row slicing and trim (shared init stream, ruling ref-0062). Frozen base @16384: per the A4 amendment (1). The binding base is the better of the two.

## Kill rules (registered before any run; full-DEV nats/word from dev_bpw.py, ref-0054 collapse protocol, final checkpoint, no checkpoint picking; every gate and proxy run counts as a trial and the base gets equal DEV trials; no verdict before the floors are registered)
**Gate 0 (implementation only, never evidence about English).**
- Setup: 200 held-out worlds. The ideal observer is the exact Bayesian posterior over each noun's 3-bit features and each predicate's requirement, given the episode prefix (mechanism skeptic fix 3).
- Pass if, on inferable never-co-occurred (noun, verb) pairs, (i) W reaches >= 50% of the ideal observer's compatible-minus-incompatible log-odds advantage, and (ii) W beats N-a2 by >= 25% of that advantage.
- On failure, fix the generator or dose using synthetic probes only. No verdict.
- Gate 0 also picks D.

**CPU transfer test (can kill; the favourable regime for a prior).**
- 4L/128d model. Pre-pretrain 1.5M tokens on W, N-a2 and none. Then the first 250k words of train_1M x 8 epochs, 2 seeds each. Score on a fixed 200k-word dev_sub slice under the dev_bpw protocol, labelled "CPU proxy".
- Kill if W minus the better of N-a2 and none is < max(0.03 nats/word, 2 x the seed SD).
- Kill if the post-English probe advantage is < 25% of its pre-English value, unless W-mix (run once, registered) clears both bars.

**Gate 1 (GPU mini-ruler; label, falsification only, ruling ref-0110).**
- Size S, train_1M x 10, base recipe unchanged, D frozen.
- W, N-a2, N-fix, N-dyck and base-18560 run with 3 seeds each (seed SD from the base triple). N-a1 runs 1 seed.
- Kill if any of:
  - (a) W is not ahead of the best of {N-a2, N-fix, N-dyck, base-18560} by max(0.03 nats/word, 2 x the 1M seed SD), on 3-seed means;
  - (b) the novelty tilt of W vs the best null is < 0.8 (epoch-like head start);
  - (c) >= 70% of the gain falls on recent repeats (in-context, capped at about 0.04);
  - (d) the survival probe is < 25% and W-mix fails;
  - (e) the fade check fails: at train_5M x 10 (W and the best null, 2 seeds each), gain_5M < 0.5 x gain_1M (the McCoy-Griffiths fade).

**S screen.**
- Size S, train_10M, round-1 base recipe, seed 0, one batch: W (or W-mix), N-a2, N-fix, N-dyck and the base-18560 re-run. The frozen base is already on file.
- Kill if:
  - (1) W's gain over the binding base (the better of base-18560 and the frozen base) is < 2 x the seed floor;
  - (2) W does not beat the best of {N-a2, N-fix, N-dyck} by 2 x the floor;
  - (3) fade rule (18:37Z): the DEV-sub gain at the end is more than 0.5 sigma below the gain 3 English epochs earlier, with arms aligned at English-epoch boundaries and phase-W tokens excluded;
  - (4) BLiMP-fast drops >= 2 sigma;
  - (5) FLOPs exceed 1.25x without a FLOP-matched reference (not expected).
- **Promotion** (one arm, picked on DEV before seeds 1-2) also needs:
  - novelty tilt >= 0.8 vs the best null;
  - >= 50% of the gain on not-recent occurrences of words seen 64+ times;
  - Gate 1 (e) passed;
  - the Gate R ruling recorded.
- A null that beats the base by 2 x the floor (for example N-dyck) is reported as a known-idea finding, not promoted.

## Odds (guesses)
- **Ceiling: 0.15 nats/word.** This is Hu-scale 1.5x token efficiency, about 0.155 by the log-linear rate. That is the best measured single-pass transfer, and it was at 1B, not 10M x 10. The in-context part is capped near 0.04 by Step 0 and the copy cache.
- **Median: about 0.00.**
- **Expected: about 0.003.**
- **P(pass S screen): about 0.02.** P(CPU test passes) is about 0.25. P(Gate 1 | CPU) is about 0.35. P(S screen including the fade rule and three binding nulls | Gate 1) is about 0.2. A small noise term is added on top.
- **Most likely failure.**
  1. Gate 0 passes.
  2. English training at full Muon LR re-forms generic heads.
  3. The selectional circuit decays.
  4. The worlds-minus-null difference comes out at 0.00-0.02 at 1M, and the card is killed at Gate 1 (or earlier on CPU).
- **What a null result buys.** A cheap, clean negative for the "priors via synthetic pre-pretraining" channel at strict-small scale, with the family calibrated by N-dyck.

## Fastest falsification
- **Step 0: done.** See Why.
- **Step 1: CPU Gate 0, about 25 min.** 4L/128d (about 4k tok/s at vocab 18560) on 2M tokens of W, N-a1 and N-a2, plus the ideal-observer probe. A 2L/64d failure would only show a lack of capacity, so 2L/64d is not used.
- **Step 2: CPU transfer test, about 2.** As in the kill rules.
- **Step 3: GPU Gate 1, about 60 A100-min5).**
  - 16 runs at train_1M x 10, about 2.5 min each including phase W (under 1 min), compile and full-DEV eval.
  - 4 runs at train_5M x 10, about 5 min each.
  - MPS packing 2-3 runs per A100.
- **Step 4 (only if Gate 1 passes): S screen.** 5 runs x about 9 min = about 45 A100-min
- Total to a verdict: about 105 A100-min. Expected compute is about 60, because P(Gate 1) is low.

## Implementation plan (harness; minimal pieces first)
1. **`babylm/moonshot/worlds.py` (~250 LOC).** WorldSpec, sample_episode(rng, mode in {worlds, null_table, null_sentence, fixed_lexicon}), PCFG with agreement, selectional sampler, Zipf and bursts, fast-mapping introductions, multi-piece spellings with per-episode suffix mapping. Writes a corpus-format bin plus a words array plus meta (generator sha, mode, D). Also `dyck.py` (~50 LOC), k-Shuffle Dyck on reserved ids.
2. **`babylm/moonshot/world_probe.py` (~120 LOC).** Never-co-occurred compatible vs incompatible log-odds on held-out worlds, and the exact ideal observer by enumeration.
3. **`babylm/harness/train.py` (+~110 LOC).**
   - Flags: --prepretrain_bin, --prepretrain_tokens, --prepretrain_batch_tokens 8192, --prepretrain_lr const, --vocab_pad 18560, --logit_rows (slice the tied table in logits(): 16512:18560 in phase W, 0:16384 in English), --interleave_bin/--interleave_frac (W-mix).
   - At English start: fresh optimizer, base schedule, re-seed of the global RNG and sampler (and EMA if on).
   - Ledger rows {english, synthetic}; milestones labelled by English words and total.
   - `export_hf.py --export_trim 16384`, written into hinton_export.json.
4. **Tests `babylm/tests/test_worlds.py` (~220 LOC).**
   - Every synthetic id is in [16512, 18559].
   - Cross-mode distribution matching within 2%, N-a2 density matching, N-a1 consistency break, N-fix fixed ids.
   - Ideal observer checked on hand-built episodes.
   - No gradient to rows < 16384 in phase W, and none to rows >= 16384 in English.
   - English loss under 18560 slicing equals the 16384-row model's loss on the same weights.
   - The hash of the first 50 English batches after phase W equals the base's.
   - Flags off are bit-identical to the base commit.
   - The trimmed export gives the same dev_bpw on a tiny model.
   - Ledger counts are correct.
5. **`babylm/configs/make_worlds_jobs.py` (~60 LOC).** Gate 1 and S-screen job lists, pinned to the round-1 base args (@ref-0065) plus the world flags.

Total about 800 LOC, about 1.5 days. Steps 1-2 plus the train.py hook give the CPU verdict after about 1 day, before any GPU compute.

## Stacks with
It is an initialisation and leaves the English recipe unchanged, so in principle it stacks with A1-A4. W-mix is a multitask regulariser and overlaps A1/A2 regularisation, so it is never summed with them. If both promote, a combination needs its own card.
