<!-- moonshot design round W2; status alive; ceiling 0.1 nats/word; expected 0.008; P(pass screen) 0.03; screen GPU 64 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card: frame-mate substitution (cogsci lens; final card after three skeptics, 2026-10-10)

Status: card only. No code exists and no GPU run has been made. All three skeptics (prior art, rules, mechanism) returned "wounded". No rule kills the card without a workable fix, so it stays alive. **It is parked at its own CPU gate.** The card writer ran the registered CPU gate on the near-final, syntax-safe table and it currently reads FAIL: the margin is 0.0065-0.0074 nats/token against a bar of 0.008. These are the lowest odds in the A-series. The skeptics' fixes are adopted below, and every fix that was modified or rejected is listed with its reason.

**Novelty label: known idea, not new.** The candidate's "known ideas, new combination" was too generous.
- **In-regime precedent: WhatIf** (Lyman & Hepner, BabyLM 2024, aclanthology 2024.conll-babylm.20).
  - Setup: Word2Vec trained on the ~9.3M-word BabyLM corpus itself; content words swapped for in-corpus nearest neighbours, with a POS check and a distance threshold; GPT-2 26M; one seed.
  - Results: BLiMP 63.7 -> 63.9-66.3, Supp 54.6 -> 56.1-59.0. A category-free re-chunking control got 66.9 / 56.4. Combining the two added nothing. No perplexity was reported, and text quality fell.
- **Class signal tied a class-free signal in controlled tests.**
  - Syntactic Smoothing (Diehl Martinez et al., EMNLP 2024; BabyLM 10M): BLiMP 71.4 -> 73.2/72.1. Plain label smoothing got 73.2/73.0.
  - Gao et al. 2019 (ACL, soft contextual data augmentation): hard LM-sampled replacement was about equal to unigram replacement and to blanks (De-En 35.40 / 35.45 / 35.37). Only the soft mixture won, by +0.3-0.7 BLEU.
- **Xie et al. 2017** (ICLR), with corrected numbers.
  - PTB medium: val 84.3 -> 79.9, test 80.4 -> 76.9. PTB large: val 81.6 -> 76.2, test 77.5 -> 73.4. That is 0.045-0.069 nats/token, but over a dropout-only LSTM.
  - The KN proposal q(x) ~ N1+(.,x) does not depend on the replaced token, and it noises targets too. It is therefore evidence for neither class-conditional nor clean-target replacement.
  - On PTB medium, unigram noising (test 80.1) lost to blank noising (78.8).
- **GECA** (Andreas 2020): fragments licensed by a shared environment. On a 5-gram MKN over 2M English tokens it moved perplexity 369 -> 365, about 1% or 0.011 nats/token.
- **Others in the family:**
  - Brown et al. 1992 (class-based n-gram LMs); Dagan, Lee & Pereira 1999; Mintz 2003 (frequent frames).
  - Wang & Yang 2015; Fadaee et al. 2017; Kobayashi 2018; SwitchOut (Wang et al. 2018); Wei & Zou 2019 (EDA); SSMBA (Ng et al. 2020).
  - RecombiText (BabyLM 2025): mixed results. Guven, Rogers & van der Goot (BabyLM 2025): category curricula mostly do not help.
  - Chen et al. 2606.16246: -0.17 nats from uniform replacement, but only at epoch 28 of a 100-epoch run.
  - In-house A2, which uses the same code path.

**What this card tests:** only whether the online, clean-target, single-forward version with a Mintz-frame/PPMI content-word table beats class-free replacement, matched for rate and for perturbation energy, on a tuned 10M base.

## Facts corrected (measured on train_10M with tok16k)
- **The original table** (token-id eligibility, measured by the mechanism skeptic):
  - 9,925 types, 9,026 with mates; 54.2% of tokens eligible.
  - About 83 swaps per 1,024 tokens at p = 0.15, not 150.
  - 59% of swaps fall on 8k+ types, and the top 100 types alone take 58%. In practice it is a function-word scheme: he->they, a->very, is->has, and->which, I->you.
- **With whole-word gating** (rules skeptic): 8,384 types, 7,287 with mates, about 57 swaps per window.
- 25.8% of the occurrences of the eligible token ids are first pieces of multi-token words (' don', ' bark' in 'barked'). Id-level eligibility therefore creates non-words.
- The leading-space filter does nothing: every eligible type carries the space.
- The proposed audit (share of neighbours sharing the word's top Mintz frame >= 0.6) is circular, because the neighbour filter already requires shared frames. It measured 0.59 and 0.76.
- **The base already has the classes.** Frame-mates appear among the provisional base's own top-50 tied-embedding neighbours for 41% of 64-8k types and 70% of 8k+ types. That is 100-200x chance.

## Mechanism (after the fixes)
**Builder** (CPU, about 1 minute; babylm/moonshot/frames.py):
- **Eligibility.** A position is eligible only if it holds a complete word: an alphabetic token not glued to an alphabetic piece on either side, judged from words.bin. Eligibility is stored as a bitmap aligned with the token stream.
- **Types.** Words with a complete-word count >= 20.
- **Neighbour search.**
  - Contexts: the top 2,000 Mintz frames plus +/-1 and +/-2 contexts.
  - Embedding: PPMI with alpha = 0.75, then SVD to 128 dimensions.
  - Mates: cosine >= 0.5 and at least 3 of the top-20 frames shared; up to 16 per type.
- **Content-only.** Applied to sources and to mates. It excludes a closed-class list of about 200 words (pronouns, determiners, auxiliaries, copulas and modals, prepositions, conjunctions, wh-words, negation and contraction pieces). It also excludes uppercase-initial types: names and sentence-initial forms.
- **Inflection and agreement filter.** Drops x paired with x+s/es/ed/ing/er/est/ly/'s, the y->ies and e->ing forms, and listed agreement pairs.
- **Gender-signature filter (new).** A mate pair is dropped if the share of female pronouns within the next 10 tokens differs by more than 0.4 between the two types, when both have at least 20 gendered mentions. Measured before this filter: 1.6% of swap mass was gender flips (man->woman/girl/lady, boy->girl, mommy->daddy, mother->father).
- **Mate weights.** q(r|x) is proportional to cos^4. The count^0.25 term is dropped.
- **Inflow cap.** A substituted-in occurrence of r is accepted with probability min(1, count_r / (p * inflow_r)). Before the cap, 142 types would receive more fake occurrences than real ones at p = 0.15.
- **Left licensing.** r is drawn only if the bigram (x[j-1], r) is attested in train_10M.

**The near-final table** (gender filter not yet applied):
- 6,476 content types with mates; 18.8% of tokens eligible.
- About 28 swaps per 1,024 tokens at p = 0.15 (25.7 with licensing) and about 56 at p = 0.30. A token rate of 5.5% is about a third of A2b's.
- Swap sources by band: 0.9% in the 20-63 band, 82.8% in 64-8k, 16.2% in 8k+.
- Example mates: dog->cat/bird/horse/pig; dogs->babies/pigs/cows (number kept); went->came/ran/walked; red->blue/green/brown; said->asked/answered/replied; happy->sad/hungry.

**Training:**
- From epoch 2 (start epoch 1.0), on causal rows only, each eligible position is chosen with probability p.
- r is drawn from the licensed, capped q(.|x).
- Targets stay clean. MNTP rows and eval are untouched.
- Draws come from A2's private CPU generator, and its seed is logged so the R5 audit can regenerate samples.
- **Arms:**
  - **F1h (hard):** the input token becomes r.
  - **F1s (soft; prior-art fix 5, after Gao 2019):** the raw input embedding becomes sum_r q(r|x) e_r over the licensed mates, with no weight on x itself. It is a forward hook on model.embed, like A2's gauss mode.
- Registered p = 0.30 of eligible positions. This was picked over 0.15 on the proxy and is logged as a proxy trial.
- Arm b (self-embedding neighbours) is out of the default screen. It is only a fallback under R0 (see the R0 section).

## Why (bottleneck 1: the 64-8k band in new contexts)
- **Target slice.** Familiar words in new contexts: the 64-8k band (0.36 nats/word, the biggest band) and the 8k+ band (0.22). Content words carry 73% of the gap, and 83% of F1's swaps fall on 64-8k sources.
- **The hypothesis is not that the base lacks classes.** Its embeddings have them (see above). The hypothesis is that its continuation predictions stay tied to specific contexts:
  - the provisional base's dev-minus-train gap grows from 0.72 to 1.29 nats/word over 10 epochs;
  - extra epochs buy a novelty tilt of only 0.43-0.76.
- **What the mechanism does.** It shows each real continuation behind contexts that hold class-mates, which pushes predictions toward the class level. The mechanism check below tests exactly that.
- **Honest counter-evidence:**
  - The class information comes from the same 10M words.
  - No new target events are added.
  - The provisional base (wd 0.4, tied) is still improving on DEV at epoch 10: DEV-sub 5.315 / 5.262 / 5.121 at 80M / 90M / 100M words. It is not in the overfit regime where replacement noise paid off for Chen et al. (gains only past the baseline optimum).

## R0 exposure ledger
| item | E* = 4 | E* = 10 (every base round-1 cell ran 10) |
|---|---|---|
| Builder: one pass over train_10M (frames, PPMI-SVD, bigram set, signatures, bitmap) | 9.95M if charged, 0 if ruled free | same |
| Training: one forward per window, 1:1 substitution, clean targets | 39.8M | 99.54M |
| Arm total | 49.8M, legal either way | 109.5M if charged (OUT OF TRACK); 99.54M if ruled free |

- **Default reading.** The builder is a PPMI-SVD word-embedding model (Levy & Goldberg 2014: SGNS factorises shifted PMI). Under R0.3 it is charged 10M, as the count model was.
- **Ruling requested before any run** (logged). Are corpus statistic tables that only transform training inputs free? The ruling should be consistent across the frame table, the band table, A2d's unigram table and the BPE tokenizer. It should also cover tables counted from the run's own epoch-1 batches.
- **If the builder is charged and E* = 10:**
  - (i) Arm a could train 9 epochs: 89.58M + 9.95M = 99.54M, exactly the base's exposure. It would be judged against the 10-epoch base and would probably lose about 0.03-0.05 nats/word to the missing epoch (inferred from the DEV-sub curve; not measured). That is more than the median gain, so this option is not run.
  - (ii) Arm b-fixed alone. Neighbours come from the model's own tied embedding at each epoch boundary. The embedding is centred with the mean and the top 2 PCs removed (all-but-the-top, Mu & Viswanath 2018), against the FRAGE frequency and anisotropy confound. Neighbours are band-matched by tokenizer merge rank (free), or by A2d's table if that is ruled free. The same string filters apply, but there is no frame filter. Zero helper exposure; it gets its own N-perm null. P(pass) is about 0.02, so the recommendation is to park the card rather than run arm b alone.
- **Parity.** Nulls are charged the same as their arm.
- 9.95M unique words; no outside data or models.

## FLOPs and R4.4
- F1h: the CPU gather costs nothing on the GPU, so 1.00x.
- F1s: one [n, 16, d] gather-sum on about 5.5% of positions, under 0.1% of step FLOPs.
- The builder is CPU only.
- Measured tokens/s against the base is reported.
- R4.4 is not triggered (under 1.25x), so no FLOP-matched reference is needed.
- The candidate's claim that "a FLOP-matched wider model gains none of this" is withdrawn. It was asserted, not shown.

## Null twins
All nulls share with the arm: the eligibility bitmap, filters, licensing, cap, start epoch, seed and data order, and the base's vocab padding (ref-0062).
- **N-perm** (binding candidate; mechanism-skeptic fix 3).
  - Each type gets the mate list and q of another type in the same log2 count band and leading-space class: a seeded permutation, self excluded, filters applied first.
  - The realised swap rate is matched to the arm's.
  - It keeps consistency, entropy and the frequency profile, and removes only class identity.
  - For F1s, the soft version uses the centroid of the permuted mates.
- **N-perm-E** (strength confound; rules-skeptic fix 4a).
  - The same as N-perm, but p is rescaled at the epoch-1 boundary so that p * E||e_r - e_x||^2 on the live tied table matches the arm's.
  - Arm and null are bit-identical through epoch 1, so this calibration is well defined. It is logged.
  - For F1s this null is N-gauss-E: A2's gauss mode with sigma^2 matched to F1s's displacement.
- **N-unigram-pos** (generic plausible noise).
  - A2d unigram replacement, restricted to the arm's eligible positions, at the arm's realised rate.
  - For F1s it is the unigram-mean embedding (Gao's "Smooth").
- **A2-best.** If A2's screen promoted a replace arm, it is re-run in this batch. Its gain is never credited to this card.

The arm must beat the best of these nulls by 2x the floor (A1 null-calibration ruling, critic ref-0083 and the 22:45Z row).

Band-random is dropped. In the proxy it never beat N-unigram-pos at a matched rate (KN: +0.002/-0.002 vs +0.006/+0.005), and N-perm(-E) is the cleaner class-free control.

## Kill rule (registered before any run)
**Setup.**
- Size S, train_10M x E*, using the frozen or provisional base recipe and its epoch count.
- Seed 0; every run in one batch on one GPU type.
- Metric: **full-DEV nats/word (dev_bpw.py, whitespace-collapse protocol) at the final checkpoint.** No checkpoint picking. DEV-sub is diagnostics only.

**Phases:**
- **Phase 1:** base, F1h and F1s. Kill if the best arm is not ahead of the base by max(2x floor, 0.02).
- **Phase 2:** the picked arm's three nulls, plus A2-best if it exists. Kill if the arm is not ahead of the best null by max(2x floor, 0.02).
- **Phase 3, only on a pass:** seed 1 of the arm, the best null and the base. Kill if either paired difference flips sign, or if the 2-seed means miss either margin.

**Also kill on any of these:**
- The fade rule (18:37Z): the gain at the end on DEV-sub is more than 0.5 sigma below the gain 3 epochs earlier.
- A drop of 2 sigma or more on BLiMP-fast.
- A drop of 1 point or more on the BLiMP-fast agreement paradigms (anaphor, determiner-noun and subject-verb agreement). This sub-score is reported separately.
- A drop of 2 sigma or more on Entity tracking (fast).
- More than 1.25x the base's FLOPs without a matched reference.

**Procedure.**
- No kill or promote until the floors are registered. Until then, numbers only.
- **Promote exactly one arm**, picked on DEV before seed 1. Ties within 0.005 go to F1h.
- This is a regulariser-type attack (critic ref-0086). Any change to the base's WD, EMA, MNTP or dropout forces one re-screen. Until then a kill reads "killed (provisional base)", and the firewall applies.
- The trial count includes every CPU proxy variant and every mini-ruler run (critic ref-0110).

**Mechanism check** (registered; a fail labels the result "regulariser, not class information" and blocks promotion):
- Positions: full-DEV positions whose preceding word w is content-eligible and whose (w, y) bigram is unseen in train_10M. There are 523k.
- Split: 282k are covered (some licensed mate r of w has (r, y) in train_10M) and 241k are uncovered.
- Test: [gain(arm) - gain(best null)] on covered positions minus the same on uncovered must be > 0, with a document-bootstrap 95% CI that excludes 0.
- The absolute tests "novelty tilt >= 0.8" and ">= 50% of the gain on 64+ words" are dropped, because every smoothing regulariser passes them. The arm-minus-null tilt and the band shares are reported.

## Odds
- **Ceiling about 0.10 nats/word.** The candidate said 0.30; the skeptics said 0.12, 0.12 and 0.10. Reasons:
  - the class information comes from the same 10M words, and the base already encodes it;
  - exact 100M trigrams supply only 0.16;
  - in a bigram proxy with no sharing, the safe table's margin over generic noise is only about 0.01 nats/word.
- **Median about +0.008 nats/word over the base**, of which about 0.004 is over the best null. 80% interval: -0.02 to +0.03.
- **P(passes the GPU screen if it were run regardless of the gates): about 0.04.** The gates mostly save money.
- **P(the card reaches the GPU): about 0.1-0.15.**
- **P(pass) unconditional: about 0.03.**

## Fastest falsification
**CPU (about 10 minutes).** The card writer has already run part of it. Scripts (scratchpad, not in the repo):
- scratch/fm_fixed_proxy.py
- scratch/fm_signature_audit.py
- scratch/fm_cover.py

**How the proxy works.** It is a bigram count model. It takes the expected (input, next) counts under each scheme and applies Dirichlet smoothing toward a backoff. Beta is tuned on dev_sub half A over 37 geometric values and the model is scored on half B. Every null runs at the arm's realised rate.

Proxy results, nats/token (gain over the proxy base; "margin" = F1 minus the best null):

| backoff | variant | p | F1 | N-perm | band-random | N-unigram-pos | margin |
|---|---|---|---|---|---|---|---|
| KN (base 5.101) | unlicensed | 0.15 | +0.0107 | -0.0025 | +0.0024 | +0.0064 | +0.0043 |
| KN | unlicensed | 0.30 | +0.0120 | -0.0069 | -0.0021 | +0.0047 | +0.0073 |
| KN | licensed | 0.15 / 0.30 | | | | | +0.0035 / +0.0065 |
| KN | licensed, prepositions allowed | 0.15 / 0.30 | | | | | +0.0046 / +0.0074 |
| unigram (base 5.163) | unlicensed | 0.15 | +0.0152 | +0.0020 | +0.0102 | +0.0227 | -0.0075 |
| unigram | unlicensed | 0.30 | +0.0197 | -0.0042 | +0.0059 | +0.0254 | -0.0057 |

**What it shows:**
- Under the unigram backoff the generic null wins outright, as in Gao 2019.
- Against N-perm, F1 experimenters by 0.013-0.019 nats/token. The class signal is real, but at a matched rate generic replacement captures most of it.
- The mechanism skeptic's unsafe all-word table had a margin of 0.012-0.016. Most of that margin came from function-word swaps, which the base already handles and which break agreement.

**Registered CPU gate.** No GPU compute unless all of these hold on the FINAL table (gender filter on, licensed, p = 0.30):
1. The KN-proxy margin over the best rate-matched null is **>= 0.008 nats/token**. This is the mechanism skeptic's bar, and it was not moved after seeing the near-final 0.0065, which is a FAIL.
2. Gender-flip mass is <= 0.3%, and the mass on listed inflection and agreement pairs is 0.
3. The share of F1 swaps whose trigram (x[j-1], r, x[j+1]) is attested in train_10M is at least 2x N-perm's.
4. Surprisal gap s(r) - s(x) under the provisional base, at 2,000 substituted DEV-sub positions: F1's median is below half of N-perm's.
5. A tiny run with the flag off is bit-identical to the base, and the measured rate is within 5% of p * accept.

**A2 gate** (prior-art fix 7). No GPU unless A2's screen has a replace arm at least 2x the floor above the base (>= 0.02 nats/word before the floors exist). Otherwise the card is parked as "dead by gate (A2 null)" with no GPU compute.

**GPU mini-ruler** only after both gates).
- train_5M x 10 epochs at size S, with the **table built from train_5M only**.
- The 1M version is barred: a 10M table leaks into it (it inflated the proxy margin 4.6x), and a 1M table handicaps it (margin 0.005).
- Runs: F1h, N-perm, N-unigram-pos, base.
- Early kill (a label, counted as trials): F1h minus the best null < 0.01 nats/word on full DEV.

## Screen (size S, about 8 A100-minutes per run)
| phase | runs | A100-minutes |
|---|---|---|
| 1 | base, F1h, F1s | 24 |
| 2 | 3 nulls of the picked arm (+ A2-best) | 24 (32) |
| 3 (pass only) | arm, best null and base at seed 1 | 24 |
- Diagnostics:
  - swap rate per epoch;
  - an inflow audit: drift of rare types' output logits under tied embeddings;
  - dev-minus-train gap;
  - the instruments' lever signature;
  - the BLiMP agreement sub-score.
- FINAL is untouched.
- Outputs go to shared/babylm/attacks/FM/<run>/ with a curves CSV and PNG.

## Implementation plan (about 1.5 days after A2 merges)
1. **babylm/moonshot/frames.py** (~260 LOC, numpy and scipy). It builds:
   - the complete-word bitmap and the frame and context counts;
   - the PPMI-SVD embedding and the shared-frame filter;
   - the content, inflection and gender filters;
   - q proportional to cos^4 and the inflow-cap accept table;
   - sorted keys of the attested bigrams;
   - the N-perm and band tables;
   - the audits and an exposure-ledger entry.
   It writes frames_<split>.npz plus meta (input sha and parameters).
2. **Extend babylm/harness/input_noise.py** (branch wip-branch, ~170 LOC):
   - new flags: --noise_dist frames|frames_perm|unigram_pos|selfemb, --noise_table, --noise_start_epoch, --noise_soft and --noise_match energy;
   - per-position eligibility gathered by stream offset (as A1 keys its cache);
   - inverse-CDF sampling over the [V,16] rows on the private generator;
   - licensing via searchsorted, and thinning by the accept table;
   - the soft centroid via an embed hook;
   - the energy calibration, logged at the epoch-1 boundary.
3. **Plumbing:** train.py flags (~30 LOC) and babylm/configs/make_fm_jobs.py.
4. **Tests** (~220 LOC):
   - flag off, and epochs before the start epoch, are bit-identical to the base;
   - the measured rate equals p * accept;
   - substitutes are always licensed mates, and the bitmap is respected;
   - N-perm is band-matched and never maps a type to itself;
   - soft equals hard when q is one-hot;
   - the energy match is correct on a toy table;
   - targets, words seen and tokens seen are unchanged;
   - resume is bit-exact;
   - the R5 sample regenerates from the logged seed.

## Skeptics' fixes
**Adopted:**
- the "known idea" label and the corrected Xie numbers;
- whole-word gating, content-only and the inflection filter;
- the inflow cap;
- the soft arm;
- non-circular audits;
- the A2 gate;
- arm b dropped from the default screen;
- the N-perm and energy-matched nulls, and the A2-best comparator;
- the R0 ledger at both E* values;
- the fade rule, the floor wait, one-arm promotion, vocab parity, the re-screen clause, the trial count and the logged RNG seed;
- the 1M proxy barred;
- the covered/uncovered mechanism check;
- the lowered odds.

**Modified:**
- licensing is left-side only;
- the 2-seed decision run became the phase-3 confirmation;
- band-random was replaced by N-perm, N-perm-E and N-unigram-pos;
- a gender filter was added.

**Rejected:**
- Right-neighbour licensing. It restricts swaps to already-attested (r, x[j+1]) pairs, which deletes the new-bigram information the mechanism exists to supply. Agreement is handled by the content and inflection filters instead.

## Stacking
- Mutually exclusive with A2 on the same positions. They are the same family and their gains are never summed; a combined arm needs its own card.
- It could stack with A1, A3 and A4 (A4's synthetic text could also be substituted) under a new card.
