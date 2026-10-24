<!-- design agent escalation card, 2026-10-10 ~17:30Z; status KILLED at Stage 0 (lanes A and B, 2026-10-10 21:11Z; accepted by the critic ref-0327 with amendments); ceiling ~0.20 total
(K=2 split 0.12 + heterogeneity 0.03-0.08); expected +0.01 over K=2 net of odds; P(Stage 0 pass) 0.35; Stage 0 0 GPU,
Stage 1 ~8 A100-min. Label: known idea. -->

# Card F2: heterogeneous legal split (K=2, members chosen to disagree)

**Label: known idea.** Ensembles of members that differ in architecture or training method are more diverse than
seed ensembles (Gontijo-Lopes, Dauphin & Cubuk, "No one representation to rule them all", ICLR 2022); neural + count
mixtures (Mikolov 2011); the instruments' GPT-2 10M + GPT-BERT 10M word-level mix measured here (+0.077 at 2x
exposure). The legal split itself is the critic's V8 / ruling ref-0135 (two 5-epoch models, 100M summed). Nothing
here is new; the untested point is whether heterogeneity raises g(2) at strict-small scale within one torch env.

## Mechanism

The legal K=2 split as run (two models x 5 epochs, 100M summed exposure, probability average at inference, 2x
inference FLOPs, calibrated_t read), with member B of a different *kind* from member A (the round-1 base). Candidate
kinds, each already a trained run or a registered menu entry, so Stage 0 needs no training:

| kind of B | why it may disagree with A | dumps that exist |
|---|---|---|
| noise-regularised base (A3g BPE-dropout arm; A2a replace arm) | different memorisation pattern on the same text | attacks/A3, attacks/A2 (torch 2.9, e10) |
| deep-narrow 28L/384d at its E* | different depth/width prior | menu F2 (torch 2.7) when it lands |
| wide 12L/592d at its E* | | menu F2 (torch 2.7) |
| size M 12L/768d | | p2 / base builder M runs if a 2.7 e5 dump exists |
| GPT-BERT reproduction (hybrid objective, own tokenizer) | different objective; word-level mix only | instruments' word-level mixer, if available |

## Why (which slice)

The variance part of the gap (C3 g_inf ~0.29 cal). Seed members share the architecture, objective, tokenizer and
regulariser, so their errors are correlated beyond trajectory noise; g_seed(2) ~0.136 cal. If a heterogeneous pair
has g(2) higher by d, the legal split gains d at zero extra exposure and zero extra FLOPs (member B replaces a seed,
it is not added). Ceiling: if hetero g_inf were ~0.40, a K=2 hetero mixture would read ~0.20 total, ~1.7x.

## Stage 0: zero-GPU diagnostic on existing dumps (executor 3 runs it; CPU minutes)

For every pair below, `ceiling_report c3 A B --guard --calib` and read g(2) raw and calibrated_t, full and guarded
legs. The seed reference is the same read on two base seeds of the same env and epoch count.

- torch 2.9, e10: (base s0, base s1) reference; (base s_i, A3g) for i in 0..2; (base s_i, A2a); (A2a, A3g).
- torch 2.7, E*: (e5 s0, e5 s1) reference [= the legal-split read, +0.118 cal]; (e5 s0, 28L/384d e5); (e5 s0,
  12L/592d e5); (28L/384d, 12L/592d); (e5 s0, M e5) if present.
- optional, word level: (base, GPT-BERT repro) through the instruments' mixer, labelled cross-tokenizer.

Readout: d = g_cal(hetero pair) - g_cal(seed pair, same env). Also the mixture's absolute DEV against the better
member (a weak B can raise g and still lose the mixture). `babylm/moonshot/hetero_pairs.sh` runs the list.

**Kill rule (Stage 0, registered now; the critic may tighten):** dead with no GPU compute unless some pair has
d >= 0.03 on both legs AND its mixture DEV beats the seed mixture of the same env by >= 0.016 (the one-seed screen
threshold). Mini-ruler proxies are not used; these are 10M runs.

## Stage 1 (only if Stage 0 passes): one run, ~4-8 A100-min, torch 2.7

Train the passing kind at e5, seed 0, base WD (or its registered WD if it is a menu entry), as member B; A = e5 s0.
Read the mixture vs (i) the e5 singles, (ii) the E* pool, (iii) the K=2 seed mixture of the same env, calibrated_t,
both legs. Promote only if it beats the K=2 seed mixture by >= 2x the K=2 replicate SD once the e5 s2-s5 replicates
land (until then, >= 0.016). The F=2 menu reference (R4.4) applies exactly as for the seed split; a mixture whose
member B is the 28L or 12L/592d model executes ~3x base FLOPs and moves to the F=3 menu row.

**Null twin.** The homogeneous K=2 seed split at identical exposure, FLOPs and read: the mechanism's information
(heterogeneity) removed, everything else equal. A shuffled-signal twin does not exist for an ensemble; the critic rules
whether this comparator stands as the twin.

**Exposure ledger.** Stage 0: no new training. Stage 1: 2 x 5 x 9.954M = 99.5M summed; unique text 9.95M; no
teacher, nothing generated. Inference 2x (reported). FLOPs: K=2 of equal shape = 2.000x; with a 2x-shape member B,
~3.0x (F=3 menu row).

**Odds (guesses).** P(Stage 0 finds d >= 0.03) 0.35. P(Stage 1 promotes | Stage 0) 0.5. Gain if it works: +0.02-0.05
over K=2 (total 0.14-0.17 cal, 1.45-1.55x). Expected net over K=2: ~+0.01. P(this card alone reaches 0.26): ~0.02.

**Honesty lines.** Known idea; the split is 2x inference; the 1.4x is a same-size number and the compute-matched
multiplier is read against the F=2/F=3 menu; report raw and calibrated, both legs, every pair including the losers.

## Critic ruling ref-0327 (2026-10-10 ~17:15Z): accepted with amendments (binding)

1. Stage 0 pairs must be same-env (torch version and epoch count of both members match the seed reference pair).
2. Every Stage 0 pair counts as a trial. Stage 1 uses fresh seeds for both members; the best-pair Stage 0 number is
   never quoted as a result.
3. A member executing 2x the base's FLOPs is read against the matching R4.4 menu row (F=3 for the mixture).
4. GPT-BERT (cross-tokenizer) pairs are a label only until a leakage check is registered.
5. Noised tokens (A2/A3 arms) count as exposure.
6. The final-step and torch-env rules apply to each member.

Null twin = the homogeneous seed split, same env, e5: accepted. Predictions P-F2a-d registered as written, scored once
on calibrated_t, both legs, at the first torch 2.7 read; P-F2d is void until v2e passes or the menu entry is rerun
under 2.7. A pass is a replication (known idea), never a discovery.

## Experimenter amendments (17:28Z; binding alongside the critic's)

1. **Same seed = same data order** (train.py:460: data_seed defaults to --seed), so no pair may share a seed. Menu F2
   runs and size-M cells are seed 0: pair them with e5 s1, never e5 s0. Dropped: (28L, 12L/592d) and (base s0, A3g s0).
2. **Menu F2 ran torch 2.9.1** (ticket ms_menu_f2), not 2.7: it is lane C, read only after v2e passes.
3. **Decision comparator = the better homogeneous split.** a3a alone is +0.058 cal_t better than its base within 2.9,
   so mix(base, a3a) clears 0.016 on member quality alone; for a3a the comparator is the A3g stack's own noisy K=2.
   Where B has one seed: d = DEV(B) - mean seed g(2). A2a is the clean case (cal_t within +/-0.005 of its base).
4. **Reference = all same-env seed pairs** (28 at E10/2.9; 15 at e5/2.7 once the LS replicates land), d in SD units.

Run plan: lane A now = A2a with C3 s1 and s2 (2.9, e10); lane B from dumps already ticketed = a3a vs the A3g stack's
K=2, size M vs e5 s1 (short-E); lane C = menu F2 held for v2e. Stage 1 GPU only on a pass.

## Stage 0 lane A read (executor 3, 18:49Z; reads at wip-branch ref-0329, babylm/gcp/reads/f2_stage0/)

All torch 2.9.1, E10. Seed reference (28 C3 pairs, not a trial): raw g(2) 0.1535 (SD 0.0027) full / 0.1497 (SD 0.0024)
guarded; cal_t g(2) over the 10 solo s0-s4 pairs (the only temps dumps) 0.1331 (SD 0.0024) / 0.1313 (SD 0.0022);
seed mixture DEV 4.8472 / 4.8195.

| pair (2 trials) | cal_t g(2) full / guarded | d vs seed ref | mixture gain vs seed mixture |
|---|---|---|---|
| C3 s1 + A2a | 0.1390 / 0.1359 | +0.0059 / +0.0046 | +0.0096 / +0.0059 |
| C3 s2 + A2a | 0.1399 / 0.1372 | +0.0068 / +0.0059 | +0.0102 / +0.0069 |

**Verdict: the A2a (input-noise) kind is dead under the registered rule** (d >= 0.03 on both legs and mixture gain >=
0.016; it reaches ~0.006 and ~0.010). d is about 2.5 SD above a seed pair, so a noise-regularised member is very
slightly more different from a base seed than another seed is, but the effect is a fifth of the bar, and part of the
mixture gain is A2a being the better single (4.9884 vs ~4.999). Lane B (a3a vs the A3g stack's K=2; size M vs e5 s1)
stays open until its dumps land; lane C waits for v2e. No GPU used.

## Stage 0 lane B read (executor 3, 21:11Z; reads at wip-branch ref-0099, babylm/gcp/reads/k2_estar/)

Torch 2.7, e5, solo, cal_t at one shared T, full / guarded. Comparator per the experimenter's amendment 3 = the homogeneous
A3g K=2 (a3a s0 + a3a s1): 4.8192 / 4.7944. Plain seed pair p01: 4.8394 / 4.8128.

| pair (2 trials) | mixture DEV | vs A3K2 | g vs member mean | mixture vs its better member |
|---|---|---|---|---|
| e5 s0 + a3a s1 | 4.8143 / 4.7894 | +0.0048 / +0.0050 | 0.1260 | 0.1135 |
| e5 s1 + a3a s0 | 4.8159 / 4.7910 | +0.0032 / +0.0034 | 0.1273 | 0.1079 |
| seed pairs (15) | | | 0.1181 (SD 0.0020) | p01 0.1134 |

**Verdict: the a3a (BPE-dropout) kind is dead** (gain over the homogeneous noisy split 0.003-0.005 against the 0.016
bar; the g excess over seed pairs is the unequal-member arithmetic, and against the better member the mixture is
seed-level). Mixing the plain and noise recipes gives seed-level diversity, nothing more.

## Card verdict (2026-10-10 21:15Z): killed at Stage 0

Two member kinds (input noise, BPE-dropout), four pairs, two torch envs: every heterogeneous pair mixes like a seed
pair. Architecture-shape members (28L/384d, 12L/592d, size M) were never read because their
runs are torch 2.9.1 (lane C, void until v2e); they stay a label, not a card, if the experimenter wants them after v2e. Stage 1
does not run. The legal split's diversity is trajectory diversity; a different regulariser does not add to it. Known
result, consistent with Gontijo-Lopes et al. finding that *architecture and objective* changes, not regularisers, move
error correlation; those were not testable here within one env.
