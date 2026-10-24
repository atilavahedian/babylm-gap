# Attack A5: rare and never-seen words (bottleneck 2) — anatomy, ceiling, and a late-epoch focus lever (card, 2026-10-10)

Status: accepted by the critic as written (02:20Z, ref-0427: label known family; beta, start epoch and focus set are
fixed; no verdict until the seed floors are registered). Lever implemented in this branch (`--focus_plan`, below) with
CPU tests; no run yet. Waits for the base round-1 winner, then a floor ticket through executor 3. Owner: this thread
(rare-word attack thread). Experimenter: research experimenter thread. Added 02:45Z: the premise check on the new base and the
re-derived prior (section "New base"), which lowers the odds. **Recommendation to the experimenter (02:55Z): on this base A5
is a trade, not an attack; run arm vs N1 only as a two-run diagnostic of the late-epoch reallocation elasticity, or
drop it. The lever stays in the harness, off by default.** Tests at ref-0431: all 7 focus tests pass (mask builder, keep rule, harness wiring incl. resume, MNTP hybrid and line windows) and the existing lever/harness/data suite passes unchanged (43 passed, 3 CUDA-only skipped). Critic 02:53Z (ref-0267): the new prior is registered as the binding prediction, the old one stays on record; a two-run diagnostic is labelled "diagnostic" and cannot be promoted from those runs; the DEV-sub ceiling figures below are diagnostics, not ranking sizes. **Decision (experimenter 03:01Z): run arm + N1 as a labelled diagnostic, skip N2; ticketed to executor 3 at the commit that adds `babylm/configs/jobs_a5_S.json` (pool floor, last in the floor order after v2b). Verdict: band split on full DEV (never-seen / 1-63 / 64-8k / 8k+ / after-rare) against the scored base, with the markup-excluded number as the labelled second figure only (critic ref-0437); the kill rule still reads the result. If the all-on wd0.4 cell re-names the base (~03:25Z) the jobs are regenerated before launch. Executor 3 accepted the ticket 03:08Z as job a5_diag pinned at ref-0428 (focus mask built in setup, score_trial and full perword per run, max 1h45m spot A100, outputs to shared/babylm/attacks/A5/<name>/); launch is many hours out, behind the floor queue. Critic cleared A5 to run at 03:36Z (ref-0429). Noise floor v2 registered 05:33Z (ref-0021), replacing v1 (ref-0022): the primary metric is score.json `dev_nats_per_word_twin` (blank-line twin scorer: each run of blank lines is one scored BOS, documents tokenized on their own; the executor runs dev_bpw_twin.py after score_trial per cell and merges the number into score.json at copy-back); base seeds 0/1/2 = 5.0023 / 4.9996 / 4.9987, floor 0.0019 (the harness floor is 0.0018, so the v1 scatter was the scorer's). Critic correction ref-0019 (05:34Z, before any read; the n=3 floor of 0.0019 has a 95% CI of 0.001-0.012): interim screen floor 0.0036. A5's numbers: screen pass >= 0.010 vs the same-seed base (seed 0, twin 5.0023), 0.0074-0.010 inconclusive (2 more seeds, once per attack), kill below 0.0074; consistency gate 0.0036 with sign agreement; 8k+ guard 0.0051; BLiMP-fast drop above 2.0 points kills; the "pass (small)" label is retired; the claim floor will be the pooled SD of >= 5 plain base seeds, registered before any claim is read. v1 `dev_nats_per_word` is reported for information only. These are the script's defaults. Critic ref-0027: arm, N1 and base must share the execution condition (alone vs alone or packed vs packed, same GPU type); the a5_diag job runs both cells alone and sequentially on an A100 like the seed-0 base (366k tok/s), executor 3 asked to keep it so or to add a packed base re-run if it packs the VM; the script checks GPU type and tokens/s (within 25% of the base) and leaves the verdict pending otherwise. Executor 3 confirmed 05:43Z: a5_diag runs unpacked at ref-0428, two cells alone and sequential on an A100, no MPS, with the blank-line twin scorer after score_trial on each cell; outputs to shared/babylm/attacks/A5/. Critic ref-0001 (05:59Z): never-seen-character guard, a pass must also clear the bar with the dev words containing a character absent from train_10M removed (the critic's tool counts 10,275; this thread's raw-text reconstruction counts 10,277, the CHAT markup words being 9,882 of them); the script adds the excluded rows (base: all 5.0096, band 0 20.38 on the harness dumps) and the leg, read on the harness dumps since no v2 per-word dump exists for these runs. A5 is a solo screen, so the 0.010 bar stands; packed screens use 0.0144. Critic 06:04Z: the excluded leg is read on the harness dumps for all three runs (interim rule ref-0015, no v2 dumps added); the pinned tool's set of 10,275 words (unseen_chars.py, sha256 ref-0017...) binds and its indices will replace the reconstruction once committed; A5 is clear to launch.** The oracle
numbers below are from the published GPT-BERT 10M; the same passes on our round-1 base (ve0, tied) are running and will
be appended to `results/babylm/rare_words_v1/` when done.

**Known idea, not new.** Arm 1 is loss focusing / example reweighting in late epochs (hard-example mining, RHO-LOSS
family: Mindermann+ 2022 train on points whose held-out loss is still reducible; focal loss, Lin+ 2017; importance
sampling of training examples, Katharopoulos & Fleuret 2018). Its new part is only the application: BabyLM strict-small,
the last 40M words of the 100M budget, targets chosen by word frequency in train_10M, no reference model. Nearest published rows: Rho-1 selective token loss
(Lin+ 2024, needs a reference model, continued pretraining only), frequency-informed masking (BabyLM 2025.38) and
undersampling frequent sentences (Zevallos+ 2023). Prior-art check: the prior-art map (PR #46, `docs/babylm/prior_art.md`
on its branch: RHO-LOSS "never measured in LM pretraining", label known-family; no rare-word-focused causal loss; vocab
size and n-hot sub-token embeddings are the nearest rare-token rows), plus a general search (AGG, FRAGE, syntactic
smoothing, a la carte / BERTRAM embeddings, Minnow; see the prior-art section). A second arm, tokenizer character
coverage, was checked and dropped (below).

## Headline for the experimenter (plain)

The >= 0.15 nats/word target for bottleneck 2 on its own is out of reach, and I can show why with numbers rather than
guess. Bottleneck 2 is 0.34 nats/word on my sample (0.28 on the instruments' protocol), and 0.045 of that is a corpus
artifact: CHILDES transcription markup that train_100M has and train_10M lacks entirely (below), so the real number is
about 0.29. It is not a representation bug that a clever output layer fixes: four cheap oracles come back at zero or near zero (below). About 40% of it is
general competence that shows up on rare words (the 100M model is still 1.8 nats better on words both models saw 1-3
times), i.e. bottleneck 1 wearing a rare-word costume. The occurrence-driven remainder is about 0.10-0.12, and the only
R0-legal ways at it are more occurrences (A4's job) or using compute the training budget where the dev loss is still
falling. That second one is cheap, stackable and measurable, so it is the card: expected +0.02 to +0.05 nats/word,
P(passes the kill rule) about 0.45. A strong null here is also worth having: it would say the rare-word gap is pure
information shortage, and that no 10M-only trick recovers it.

## New base (ve0, tied, Muon WD 0.4): premise check, ceiling and why A5 adds to it (added 02:45Z)

The base changed at 01:18Z: dropping the value-embedding tables and tying the head fixed the port's collapse, and the
experimenter asked whether the card's premise ("frequent words are flat in epochs 7-10, rare words still learn") survives on
that base. Measured from the per-word dumps of `base_S_wd0p4_lr0p5_r1_ve0_tied` (DEV-sub, 993k words, train_10M
bands as above, the same scripts in `results/babylm/rare_words_v1/`):

| band | word share | DEV-sub loss e7 -> e10 | gain e7 -> e10 per word (e7-8, e8-9, e9-10) | share of the e7 -> e10 gain |
|---|---|---|---|---|
| never seen | 1.8% | 21.31 -> 20.94 | +0.37 (-0.15, +0.01, +0.51) | 2.7% |
| 1-63 | 6.6% | 12.51 -> 11.76 | +0.75 (+0.07, +0.21, +0.47) | 20.1% |
| 64-8k | 35.8% | 6.52 -> 6.28 | +0.24 (-0.01, +0.05, +0.20) | 34.4% |
| 8k+ | 55.8% | 3.27 -> 3.08 | +0.19 (+0.09, +0.04, +0.06) | 42.7% |
| 3 words after a rare word (64+) | 19.0% | 5.00 -> 4.79 | +0.21 | 16.9% of the e9 -> e10 gain |

Total 5.366 -> 5.121 (0.245 nats/word, a third of it in the last epoch's LR cooldown). **The premise does not hold as
stated: frequent words are not flat on the new base** (0.06 nats per word per epoch, 43% of the late gain by mass).
What does hold: rare words still have the steepest slope, 0.25 per word per epoch, 4x the frequent words' and 3x the
mid band's, so per unit of target share the late gradient returns about 0.24 (rare), 0.08 (mid) and 0.06 (frequent)
nats per epoch. A5 now trades frequent-word gradient that is still buying something for rare-word gradient that buys
3-4x more per target, instead of trading memorisation for learning. That is a smaller, riskier bet, and N1 (random
drop, same kept share) is the control that prices the lost frequent-word gradient.

**Ceiling against our own base.** DEV-sub, the same cell at epoch 10 against the 100M-trained size-S twin
(`grid_S_100M_e10_s0` at 1B tokens): gap 0.701 nats/word. Never seen 0.083 (11.8%; CHAT markup only 0.013 here, the
DEV subset holds few of those lines), 1-63 0.152 (21.7%), 64-8k 0.307 (43.8%), 8k+ 0.159 (22.7%). Bottleneck 2 on our
base = 0.235 (33.5% of the gap). The focus set (rare words + 3 after, 27.4% of words) carries 0.351, half the gap; the
aftermath words alone 0.116 (16.5%).

**Why A5 adds to ve0+tied.** Tying and dropping the value tables changed how rare words are stored and read out (the
lookup-table memoriser), which is where their gain on never-seen (+0.87) and once-seen (+0.51) words came from. A5 does
not touch the parameterisation; it changes which targets drive the last 40M words of gradient. On the new base the
words it targets still have 4x the per-word slope and carry half of the remaining gap, so the lever has something to
act on; whether it beats its random twin is the question the screen answers.

**Anneal caveat (instruments, 02:31Z; their band numbers for epochs 7-10 and for the gap to the twin match the table
above to the third decimal).** Epochs 5-10 are this recipe's LR cooldown (factor 1.0 to 0.1), and epoch 9 to 10 alone
is 0.141 of the 0.245. At constant LR the rare bands had stalled and begun to rise (epoch 4 to 5: 1-63 +0.20, never
seen +0.41 worse), so the instruments read the late rare-word gain as mostly the anneal, not the extra passes. For A5
this cuts both ways: if rare words were over-fitting at full LR, a larger gradient share on them during the cooldown
buys refinement at low noise, but a larger share at a still-high LR (epoch 7 is at factor 0.6) can buy more
over-fitting first. It is a reason to expect the twin comparison to be close and to read the per-band curves of all
three runs, not only the final numbers.

**Re-derived prior (new base).** If late gains were linear in target share, moving 32 points of share from a 0.06
return to a 0.12 return (the focus set's blend of rare 0.24 and aftermath 0.07) would be worth about +0.08 over four
epochs; real returns are not linear (frequent-word gains ride on shared features that every target trains, rare-word
gains saturate on the 1-63 contexts they have), and the frequent words lose gradient they were still using. Guess:
**-0.01 to +0.04 nats/word, centre +0.01**, P(passes the kill rule) about 0.3 (was 0.45), P(>= 0.05) under 0.10.
beta = 0.25 and the start epoch 7 stay as registered: a weaker beta would hide a one-seed signal under the floor, and
the twin absorbs the regularisation-and-noise part either way. If arm 1 and N1 both lose against the base, late-epoch
target dropping hurts a base that is still learning everywhere, and the finding is "compute the late epochs on all
targets": useful for A1 and the stacker.

## Implementation (this branch)

- `python -m babylm.data.focus_mask --split train_10M` writes `<data_dir>/bin/train_10M.focus.bin` (uint8 per token,
  1 = word seen <= 63 times in train_10M or one of the 3 words after one; the count table is over train_10M's own
  words and nothing else; per-document word counts are checked against the tokenized split; sha256 of the mask and
  of the split in `train_10M.focus.json`). `--after 0 --tag focus_a0` writes the rare-only mask for N2.
- `babylm.harness.train --focus_plan 7:0.25`: from the step at which the tokens consumed reach 6 epochs, causal
  targets with mask 0 are set to -1 (ignore_index) with probability 0.75 from a dedicated generator (seeded from the
  data seed, saved in resume.pt), mask-1 targets always stay; the loss is the mean over the kept targets. Inputs,
  windows, data order, `tokens_seen`, `words_seen`, FLOPs, the model and the HF export are unchanged; MNTP rows keep
  their own targets. `--focus_null` ignores the mask and keeps every target with the plan's expected share (N1).
  `--focus_mask focus_a0` selects the rare-only mask (N2). Per-step `focus_kept` is logged in `log.jsonl`;
  `config.json` records the plan, the mask's sha256 and shares.
- Tests: `babylm/tests/test_focus.py` (mask rule, keep rule, mask build and load, harness wiring: epoch 1 identical to
  the base path, kept share, exposure counters equal, null twin, MNTP, line windows, resume).
- Jobs: `python -m babylm.configs.make_a5_jobs --base babylm/configs/jobs_base_r1_S.json <winner>` writes the three
  runs (arm, N1, N2) with the winner's exact flags.

## Stack interactions (from the attack stacker, 02:57Z; A5 is held out of the stack until the experimenter decides)

If A5 is promoted, four interactions have to be defined before it joins PR #64:

1. A1 (self-distillation): its KL runs only where `y >= 0`, so a target A5 drops receives no A1 signal (the drop is
   complete, not partial), and A1's keyed-position check requires dropped targets to be unkeyed as well (one-line fix
   on the stacker's side).
2. A2 (input noise, R2L rows): `reverse_rows` assumes each row's real targets form a prefix; A5's mid-row `-1`s break
   that. Cheapest fix: apply the keep mask after the reversal, reversing the boolean mask with the row.
3. A3 (BPE-dropout): the mask is defined on canonical words, so the per-stream mask for re-segmented epoch streams
   goes through the stacker's canonical word map; the N1 twin needs no map.
4. A4 (model-written contexts): synthetic rows have no train_10M counts; default is keep all their targets.

## Pre-registered control numbers (full DEV, base_S_wd0p4_lr0p5_r1_ve0_tied)

Written before the diagnostic runs start (03:15Z), from the base's own per-word dump (`pw/pw_devfull_final.npy`,
10,418,979 dev words = the dump's length; per-source means reproduce the harness's `result.json` per-source nats/word to
four decimals, so the word alignment is exact). Script: `babylm/analysis/rare_words/band_split.py`; table and JSON in
`results/babylm/rare_words_v1/a5_bands_fulldev/`. The arm and the twin are read against these rows with paired block
bootstraps (2,000-word blocks). Bands = count of the word's normal form in train_10M (0 / 1-63 / 64-8k / 8k+). The
markup words (0.09% of dev words at 36.4 nats each) carry 0.030 nats/word of the base's total; the markup-excluded
total is the labelled second number only.

| slice | share of dev words | base nats/word |
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
| all, markup-excluded | 0.9991 | 5.0108 |
| band 0, markup-excluded | 0.0166 | 20.4189 |
| markup words | 0.0009 | 36.4355 |

Instruments and settings, fixed before the run (critic ref-0125, experimenter 03:31Z scorer rule, critic ref-0053): the **headline verdict number is the registered scorer**, since 05:33Z `score.json` dev_nats_per_word_twin (v2, the blank-line twin of the dev_bpw protocol on the fp32 HF export): base 5.0023; v1 dev_nats_per_word (base 5.0367) is listed for information. Arm vs base and arm vs N1 are read on that instrument against the 2x-floor rule. The band split (every row of the table, base included) comes from the harness's per-word dumps (`pw_devfull_final.npy`, harness total 5.0406) and is the diagnostic of where the change lands; the 8k+ guard and the mechanism share are evaluated on it. The two instruments differ by -0.020 to +0.034 per model (instruments, ref-0084), so no delta mixes them: official against official, harness against harness. Critic ref-0270 (supersedes ref-0125): the band legs count only if the harness and official total deltas (arm vs base, arm vs N1) agree in sign and differ by less than 1x floor; otherwise the band split is labelled "instrument-inconsistent", no mechanism claim is made, the guard is not read, and only the official gain legs decide. The script prints both deltas side by side and the label. Bootstrap: 1000 paired replicates over 2,000-word blocks, RNG seed 0 (the script's defaults; the committed control table carries the same settings and the headline line).

Reading the diagnostic (registered v2 numbers above): the arm's twin-scorer total must beat the same-seed base and the
null twin by at least 0.010 nats/word to count as a gain, 0.0074-0.010 is inconclusive, below 0.0074 kills; the 8k+
row must not be worse than the base by more than 0.0051 on the harness dumps; the mechanism check asks for >= 60% of
the harness gain on bands 0 / 1-63 / after-rare; both band legs are read only if the consistency gate (0.0036, same
sign) is open; a BLiMP-fast drop above 2.0 kills. Without a twin score the verdict stays pending. The elasticity number the experimenter asked for is
the pair (band 0 + 1-63 + after-rare change, 8k+ change) of the arm against the twin, both in nats/word of the whole
set, i.e. share x row delta.

## Measurements (this thread, published GPT-BERT causal-focus 10M vs 100M; code and tables in `results/babylm/rare_words_v1/`)

Protocol: 215,312 whitespace words of the official dev split (287 pieces of <= 2,500 words from whole documents,
at most 2 pieces per document, all 6 sources, seeded), scored with the instruments' scorer (512-token windows, stride
256, BOS, collapsed whitespace, each model's own tokenizer). Words in a line found verbatim in train_100M (13.0% of the
sample) are dropped as in the instruments' protocol; sources are reweighted to the dev split's word shares; CIs are a
piece bootstrap. GPT-BERT 10M scores 5.019 nats/word on the raw sample, the instruments' sample gives 5.020. Gap =
10M minus 100M, nats per word.

| band (train_10M count of the normal form) | word share | gap per word | gap share | gap, nats/word overall |
|---|---|---|---|---|
| never seen | 1.85% | 7.95 | 14.6% | 0.147 |
| 1-63 | 6.70% | 2.90 | 19.4% | 0.194 |
| 64-8k | 35.5% | 1.21 | 42.7% | 0.429 |
| 8k+ | 55.9% | 0.42 | 23.3% | 0.234 |

Total gap 1.003 [0.895, 1.154]; bottleneck 2 (never seen + 1-63) = 0.341.

**Where inside bottleneck 2 (filtered sample):**
- First use vs repeat within 300 words: 1-63 first uses carry 75.6% of the band's gap at 2.94 [2.82, 3.06] per word,
  repeats 2.77 [2.24, 3.38]. Never-seen first uses 9.18 vs repeats 4.30. No copy lag, as the instruments found.
- First piece vs the rest of the spelling (each model with its own tokenizer): 1-63 words split 1.40 (first piece) vs
  1.50 (continuation), 48/52. Never-seen words split 1.30 vs 6.65: 84% of their gap is spelling the rest of the word.
  Words of 64+ count are 86% first piece. Longer rare words are worse: 1-63 first uses of 2 / 3 / 4+ pieces (43 / 33 /
  21% of them) carry 2.26 / 3.17 / 4.09 per word, almost all of the growth in the continuation.
- By count: words seen once carry 5.28 per word, 2-3 times 4.18, 4-7 3.26, 8-15 2.65, 16-31 2.45, 32-63 2.18. Words
  undersampled in 10M (train_100M count >= 15x the 10M count) are 32% of the band but 45% of its gap (4.10 per word).
- Rare words as context: frequent (64+) targets 1 / 2 / 3 words after a rare word carry 1.01 / 0.80 / 0.77 per word
  against 0.60 with no rare word in the last 50; the excess within 3 words is 0.044 [0.031, 0.061] nats/word overall.
  Part of this is hard-region confounding, so I count about half of it as a rare-word input-side cost.
- Local context novelty, 1-63 first uses: the exact (previous word, word) bigram occurs in train_10M for 36.8% of them,
  and those still carry 2.67 per word (10M loss 10.63); bigram only in train_100M 27.2% (3.67); in neither 36.0%
  (2.61). 76% have a trigram in neither corpus (2.74). The 100M edge is not memorised collocations.
- Never-seen words by type (share of band words / share of band gap / gap per word): contains a non-ascii character
  20.6% / 46.6% / 18.0 (10M 30.5 nats per word vs 12.5; 5.32 vs 4.17 pieces per word; 5.56 vs 3.12 nats per piece), of
  which 77% are CHILDES conversation-analysis markup (overlap brackets, rising pitch, quiet-speech marks) carrying 87% of
  the category's gap, and 23% accented Latin words carrying 13% (0.0075 nats/word overall); capitalised
  name-like 25.8 / 17.2 / 5.3; lowercase novel 11.7 / 8.5 / 5.8; morphologically decomposable from seen words
  (inflection, derivation, compounds) 34.6 / 20.7 / 4-6; numbers 3.1 / 1.4; symbols 1.8 / 2.8. Words never seen by
  either model (41% of the band) still show a 6.63 per-word gap: better spelling and context, not exposure.
- **Corpus artifact (new, for the ruler and the critic).** The CHAT markup symbols ⌊ ⌋ ⌈ ⌉ ↗ ∆ ∙ ⁎ occur zero times in
  train_10M, about 7,000 times each (brackets) and 1,202 (↗) in train_100M childes, and about 2,500 each in the dev
  childes split (0.94% of its lines, 4x the train_100M rate). Words carrying them cost the 10M model 28.6 nats each
  (8.4 pieces at 5.4 nats a piece, i.e. near-uniform over the vocabulary) against 11.8 for the 100M model, and sum to
  about 0.045 nats/word of the dev gap, 91% of it in childes. No 10M-only method can learn a convention with zero
  training occurrences except in-context at test time, so this slice should be reported separately by the ruler and
  excluded from the reachable prize. It also inflates the instruments' never-seen band by about a third. Critic
  (02:20Z): zero occurrences in the train_10M count table confirmed; attack and win scoring stay on full DEV; for
  bottleneck sizing the prize is reported with and without these words, labelled "markup-excluded (reachable) prize".

**Four cheap fixes measured as oracles on the published 10M model, all small:**
1. Token frequency bias (recalibration): tokens seen 1-7 times in train_10M get 4.4x too little mass and tokens seen
   8-63 times 1.35x too little, but optimal per-band logit offsets (7 parameters, cross-fitted on odd/even pieces)
   buy **0.001 nats/word**. Rare *pieces* are not the problem; 83% of the rare-word gap sits on words made only of
   frequent pieces (instruments' split). So output-row fixes (FRAGE, Yu+ 2022 gradient gating, prototype or imprinted
   rows, frequency-aware decay) are capped by the 17% rare-piece share and have no measured miscalibration to fix.
2. Lexicon prior for spelling: a trie over the 365,885 train_10M word types, mixed into the within-word continuation
   distribution (exact mixture nll from the model's target probability and its word-boundary mass), gains **+0.0095
   nats/word** at the best single weight (0.05) and +0.012 with a weight per band; on 1-63 words 0.13 of their 1.50
   continuation gap. Spelling a rare word is deciding which word it is, piece by piece; a unigram lexicon cannot help.
3. Word boundary: at continuation positions of 1-63 words the model puts 13.6% of its mass on "the word ends here",
   costing 0.44 of the 3.69 continuation nats per rare word. Real but not the bulk.
4. In-context word learning: repeats are 25% of the band and carry no larger gap than first uses (above). Out.

**Matched own-count check (first uses; loss minus the model's own unigram surprise, i.e. what context and knowledge
buy):** 10M model at own count 1-3 / 4-15 / 16-63 / 64-255: +1.60 / -0.60 / -1.66 / -2.22; 100M model at the same own
counts: -0.82 / -2.94 / -3.34 / -3.67. The 10M model is worse than its own unigram on words it saw 1-3 times: those
representations are noise after 10 epochs on 1-3 contexts. The populations differ (the 100M model's rare words are
rarer in the world), so the cleaner test is the same words: for 1-63 words the 100M model also saw at most 63 times
(n = 1,972, median counts 3 vs 31) the gap is 3.86; where it saw them 1-3 times (n = 89, median 1 vs 1) the gap is
still 1.78. Reading: roughly 40-45% of the rare-word gap is general competence (size, batch, better context use) and
55-60% is the extra occurrences. Only the second part is bottleneck 2's own.

**Ceiling.** Occurrence-driven part about 0.10-0.12 (1-63 first uses 0.15 x 0.6, plus a third of the never-seen
decomposable and name channels, plus half the context-side excess). Reachable under R0 without synthetic text: the
share that better use of the existing 10 passes can buy. The instruments' curves give the one hard number: 1-63 words
still gain 0.084 [0.062, 0.107] nats per rare word per epoch in epochs 7-10 while 8k+ words are flat (0.000 [-0.002,
0.003]) and late epochs tilt to memorisation (novelty tilt 0.43 vs 1.04 for 10x data). That is the lever below.

## Mechanism, arm 1: late-epoch rare-word loss focusing (primary)

1. Offline, once, at prepare time: for every training token, flag whether its whitespace word has train_10M normal-form
   count <= 63 (`rare`), and whether it is rare or within the 3 words after a rare word (`focus`; 7.1% and 23.4% of
   train_10M words). Stored as a uint8 array parallel to the token array; the sampler gathers it with each window.
2. Epochs 1-6: unchanged.
3. Epochs 7-10: targets outside the focus set are dropped from the loss (`ignore_index`) with probability 1 - beta,
   beta = 0.25, fixed per step by the data seed; focus-set targets are always kept; the loss is the mean over kept
   targets. Kept share 0.234 + 0.25 x 0.766 = 42.5% of targets. Rare-word targets go from 7% to 17% of the gradient,
   the focus set from 23% to 55%. Every word is still fed as input (exposure unchanged, counter unchanged); FLOPs
   unchanged (forward and backward run on the full window). No model change; the HF export is the plain model.
4. Harness: a `--focus_plan 7:0.25` lever in `babylm/harness/train.py` (epoch start, beta) next to `--window_plan`; the
   mask file comes from `babylm/data/prepare.py` (token-to-word offsets are already computed for the collapse format).
   Works with tied or untied embeddings, with or without value embeddings, with MNTP (masked rows keep their own
   targets; the focus mask applies to the causal rows only).

Why it should close its bottleneck: the last four epochs are 40M of the 100M words. Today they buy 6% of the dev gain,
almost all of it on words that are still learning: rare words at 0.084 per rare word per epoch, 8k+ words nothing
(the instruments did not report the 64-8k band's late-epoch slope separately). Dropping three quarters of the frequent-word targets in those epochs takes gradient away
from positions that are being memorised (their dev loss is flat while train loss falls) and gives it to the positions
whose dev loss is still falling, including the 3 words after a rare word that carry the input-side cost.

## Null twins (same seed, data order, schedule, exposure, FLOPs, code path)

- **N1 (binding):** in epochs 7-10 drop the same number of targets, chosen uniformly at random (kept share 42.5%),
  focus mask replaced by a random mask with the same density. Separates "rare-word information" from "fewer late-epoch
  targets is a regulariser".
- **N2 (mechanism, not in the kill rule):** focus set = rare-word tokens only, no following words. Says whether the
  aftermath channel contributes.

## Kill rule (registered before any run)

Size S, the base round-1 winner, seed 0. Metric: full-DEV nats/word from `babylm/eval/dev_bpw.py` on the final
checkpoint, plus the per-word dumps (`babylm/analysis/perword.py`) by train_10M band. **Kill** if the gain over the
base is below 2x the critic's seed floor, or if the gain over N1 is below 2x the floor. **Kill** if the 8k+ band's
DEV nats/word is worse than the base by more than 1x the floor (frequent words must not lose), or on a >= 2 sigma
BLiMP-fast drop. No verdict before the floor is registered. **Mechanism check (reported, not a kill):** at least 60%
of the DEV gain must land on words seen 0-63 times and the 3 words after them; the instruments' signature should show a
rare tilt above 1.5 (10x data: 1.83). If it passes: 3 seeds, size M and official eval under R0-R6; beta and the start
epoch are fixed by this card, the critic may tighten them before the run, not after.

## Odds (guesses, stated plainly; superseded by the new-base prior above, kept for the record)

Rare words get 2.4x their gradient share for 4 epochs, worth perhaps 1.5-2 extra rare-word epochs at the measured
0.084 per epoch: +0.13 to +0.17 per rare word, +0.01 to +0.015 nats/word overall, with diminishing returns unknown.
The aftermath channel adds +0.005 to +0.015. Taking gradient away from memorising positions may help frequent words
on new text (A1's logic) or cost them calibration: -0.01 to +0.01. **Guess: +0.02 to +0.05 nats/word**, P(passes
the kill rule) about 0.45, P(>= 0.05) about 0.15, P(>= 0.15) under 0.02. The null is informative: if N1 matches
arm 1, late-epoch target dropout is a regulariser worth keeping and the rare-word story is dead; if neither moves,
the rare-word gap is an information shortage that only more occurrences (A4) can touch.

## Arm 2, checked and dropped: character coverage in the tokenizer

The idea was to add every character with train_10M count >= 2 to the byte-level BPE alphabet so accented words are not
spelled as raw bytes. The check killed it: the 10M tokenizer already has single tokens for the Latin accented letters
(à á â ã ä å æ ç è é ê í ñ ó ô ö ú ü ā č ł ō š ū; 183 single non-ascii-character tokens in all, the 100M tokenizer 197),
accented words carry 0.0075 nats/word of the gap, and 93% of them are spelled with single-token letters (the ones with a
two-byte letter carry 0.0005). Greek, Cyrillic and CJK strings are spelled byte by byte by both tokenizers alike. The
rest of the non-ascii gap is the CHAT markup artifact above. Nothing for a tokenizer change to buy; no run.

## Not proposed, and why (for the experimenter's registered rule)

- Prototype / imprinted output rows, FRAGE, gradient gating (Yu+ 2022 cut rare-token perplexity 439 -> 75 on
  WikiText-103 with total perplexity flat): token-level fixes; here rare tokens carry 17% of the band and show 0.001
  of recalibration headroom. Ceiling under 0.02. Not worth a GPU slot.
- In-context word learning (Minnow-style nonce augmentation): no copy lag to fix; repeats already carry no larger
  gap than first uses.
- Lexicon or count priors at inference: +0.01 (trie); the word-level KN mixture the instruments measured (+0.051) is
  bottleneck 3 and overlaps everything.
- Window resampling toward rare-word windows (the data-reweighting variant): a window of 640 words holds 45 rare words
  on average with CV 0.73, so sampling windows proportional to rare content buys only 1.3-1.5x rare exposure even at
  80-word windows. Loss focusing moves the gradient share 2.4x at zero exposure cost, so it is the version on the card.
- Vocabulary 32k (rows for rare words; BabyLM 2023 found larger Unigram vocabularies better on BLiMP): a base-level
  change with 1.2x head FLOPs; flagged to the base builder as a test, not an attack.

## Screen (small first; size S, seed 0, about 0.5 A100-hours)

Runs: base control (the round-1 winner, already exists), arm 1, N1, N2, each about 8 minutes at S plus scoring and
per-word dumps, three new runs. Every run saves curves as CSV plus PNG; outputs go to
shared/babylm/attacks/A5/<run>/ as all runs do; the exposure counter is logged and must read the same as
the base.

## Prior art and labels

- Loss focusing / reducible-loss selection: RHO-LOSS (Mindermann+ 2022, arXiv 2206.07137), Rho-1 selective token
  loss (Lin+ 2024, arXiv 2404.07965: reference-model scores pick the tokens, continued pretraining only), focal loss
  (Lin+ 2017), importance sampling for deep learning (Katharopoulos & Fleuret 2018). The prior-art map lists RHO-LOSS
  as "never measured in LM pretraining" and says to label it known-family. **Label: known idea, new application
  (frequency-chosen targets, no reference model, late epochs only).**
- Frequency-aware exposure: frequency-informed masking (BabyLM 2025.38: +1 BLiMP-supplement, +7.5 adjective
  nominalisation, NLP worse than baseline), undersampling frequent sentences (Zevallos+ 2023, "up to 3x"), the
  instruments' late-epoch band slopes (PR #52). **Label: known family.**
- Character coverage: SentencePiece (Kudo & Richardson 2018). Checked and dropped, see arm 2.
- Measured and rejected here: FRAGE (Gong+ 2018, WT2 perplexity 67.1 -> 64.8), AGG (Yu+ 2022, ACL), syntactic
  smoothing (arXiv 2410.11462, BabyLM-scale MLM), weight decay hurting rare tokens (arXiv 2410.11985), a la carte
  embeddings (Khodak+ 2018), BERTRAM (Schick & Schütze 2020), Minnow (Wang+ 2025), n-hot sub-token embeddings
  (BabyLM 2025.31: adjective nominalisation up, BLiMP down).
