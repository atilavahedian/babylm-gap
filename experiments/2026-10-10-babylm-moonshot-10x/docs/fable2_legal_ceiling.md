# design agent escalation 2 (2026-10-10 ~17:30Z): where the ensemble gain comes from, what is legal, and the honest ceiling

Owner: the design agent escalation thread "legal route past 1.4x". Every number below is read from the moonshot's own
readouts (ceiling_v2, F1, PE Gate 2, the legal-split K=2 read) or is a stated guess; nothing here is a verdict.
Units: nats per whitespace word on full DEV, calibrated where the source was.

## 1. The gap, in three parts (10M x 10 -> 100M, size S, ~0.83; ~0.79 if the torch-2.9 penalty holds)

| part | size | evidence | legal lever |
|---|---|---|---|
| information absent from train_10M | ~0.25 +/- 0.08 | bits lens (F1 Part A), never-seen words 0.10, unrecoverable half of 1-63 band, 100M-only facts | none (R0.4) |
| posterior variance of one 10M learner | 0.29-0.33 | C3 g_inf 0.333 raw / ~0.29 cal; per band it follows (1 - 1/K) | K=2 split +0.118 (the only one found) |
| residual bias: in the data, not learned by any ensemble of our transformers | <= 0.25-0.30 (upper bound) | C1-oracle f1 ~0.70 (a smooth 100M teacher closes 70% on the same 10M contexts) minus the ensemble ceiling; T100's targets carry 100M information (10M-only targets moved the F1 student ~0.05, T100's moved C1 by 0.51), so this is an upper bound, and count-0 and most of the 1-63 band inside it are 100M information, not learnable bias | nobody has a learner for it |

So the 5x "ceiling" (perfect 100M teachers) is a statement about the data, not a route. Under R0 the whole legal
prize is the variance part, ~0.3 at best, which is ~2.2x by the log-linear rate. 10x is not in the data.

## 2. Why no single-model mechanism captures the variance (what the moonshot's runs actually say)

1. **Sharing kills diversity.** PE (4 shared blocks, shared tied head, same batch order) kept 36% (rho 0.364);
   trunk e4 + 2 branches x e3 kept ~2/3 of K=2's gain. Kim/Kotha App. C.1: data order is the larger diversity
   source, and every one-forward design shares it by construction.
2. **Distillation cannot carry it.** F1: an 8-teacher student beat a 1-teacher student by 0.05 (inside noise)
   although the raw 8-ensemble is 0.29 better than a member. The student is itself a 10M x 10 learner with the same
   ~0.29 of variance; soft targets from an ensemble are only ~0.29 smoother than one-hot, so after the student adds
   its own noise almost nothing is left. Contrast C1-oracle (f1 ~0.70): a teacher that is *much* smoother than the
   student's own noise does transfer. Target-side cards are therefore capped by the student's variance, not by the
   teacher's quality, unless the student sees far more contexts (which is the T2 synthetic route, odds ~5-10%).
3. **The split family sits on a flat ridge.** Mixture DEV ~= mean member DEV L(e) - g(K) with L minimised at
   E* ~ 6 and g(K) ~ g_inf (1 - 1/K). Any legal allocation (100M summed) trades member quality against K and the
   independent fraction of each member's trajectory. K=2 at e5 is already near the top of that ridge.

**Registered predictions (score once, from the first within-env read, calibrated_t, torch 2.7):**
- P-F2a: the 3 x 3.33 split lands within [-0.03, +0.02] of the K=2 e5 mixture.
- P-F2b: every trunk-plus-branches variant with 100M summed exposure lands within [-0.06, -0.01] of K=2.
- P-F2c: no split, branch, snapshot or particle design with our base recipe beats the K=2 e5 mixture by more than
  0.02. If one does, section 3 is wrong and that design is the new main line.
- P-F2d: the F=2 menu's best single (28L/384d or 12L/592d at its E*) lands within 0.05-0.08 of the K=2 mixture, so
  the split's margin over the R4.4 reference is ~0.04-0.07, not 0.12. The 1.4x is a same-size, same-inference-FLOPs
  number; the compute-matched data multiplier is ~1.15-1.25x. (Critic ref-0420: void until v2e or a 2.7 rerun. Scored HIT by the critic ref-0149: 0.058/0.059 cal_t vs the F=2 28L e6 menu single, env-robust.)
- P-F2e (registered 17:35Z, before the cells run; experimenter's size-M short-E read, torch 2.7): size M 12L/768d (~4x S
  FLOPs) at the best of {E5 wd1.6, E5 wd3.2, E3 wd1.6} on DEV, versus the K=2 e5 mixture, cal_t, both legs, same env:
  mixture DEV minus M DEV in [-0.06, +0.02], point guess -0.02 (the mixture still ahead by ~0.02). Below -0.06 (M
  clearly better): the split has no compute-matched case at F=4 and the legal lever is "more parameters with tuned
  WD", a known idea. Above +0.02 (M worse than S's mixture by more): M is under-tuned at 10M, not evidence for the split.
  The three M cells count as three trials for M's side.

## 3. What is still legal and untested: diversity per exposure, not exposure per model

The only term in the ridge equation nobody has pushed is g itself. Seeds give g(2) ~0.136 cal. Members that differ
in architecture, regulariser or objective have less correlated errors (Gontijo-Lopes, Dauphin & Cubuk, ICLR 2022;
the instruments' GPT-2 + GPT-BERT word-level mix here). If g_hetero(2) is 0.03-0.06 above g_seed(2), the legal K=2
split inherits it at zero extra exposure. That is card F2 (cards/F2_heterogeneous_legal_split.md). Stage 0 needs no
GPU: pairwise mixtures of dumps that already exist.

Stackers on top of the split, each small, each legal, owned by the stacker thread: EMA or final-epoch snapshot mix
per member (+0.01-0.03 each, 1x inference each), test-time context-offset averaging (+0.01-0.02), temperature
(already applied). The count-model mix (+0.05) costs one epoch, which the split cannot spare; it competes, not stacks.

**Honest reach of this program:** 0.15-0.22 nats/word, i.e. 1.5-1.8x by the log-linear rate, with 1.4x in hand.
P(>= 0.26, a legal 2x, under strict R0 with our base family) ~0.15. P(5x or 10x) ~0.

## 4. Considered and rejected (so nobody re-proposes them)

- Minimal-sharing particles (one shared block, private everything else, per-particle data-order subsets) and
  per-particle loss masks on one forward: they count exposure once only by the letter of ruling ref-0110, which the
  critic already flagged as the loophole it closes; and the shared embedding/unembedding still carries the rare-word
  variance. Not proposed.
- Bagged members (each on 1/K of the words x 10 epochs): members lose ~0.25 each (one halving on the ruler);
  q0 and Branch-Train-Merge agree. Dead on arrival.
- Weight-space averaging of forked children (soups): same basin by construction, so it captures the within-basin
  part only (the snapshot mix's +0.05 is its ceiling); the W3 cycle-and-average card already prices it at 0.01.
- Dynamic evaluation / test-time training on DEV or TEST text: conditioning on the evaluation stream (critic: no
  DEV/TEST conditioning). Not a legal lever and useless for the official suite.
- KD from a count model as soft targets: the A1 unigram null twin, stepped up; cannot beat mixing the same count
  model at inference (+0.05), which already costs an epoch.

## 5. The one question worth a zero-GPU readout before any new card: the bias map (D1)

From dumps that exist (C3 members, the 8-ensemble, C1-self, C1-oracle, T100): per band (count 0, 1-63, 64-8k, 8k+;
seen vs new text; with and without the never-seen-char guard) report DEV(mean member) - DEV(ens8) beside
DEV(ens8) - DEV(C1-oracle) and DEV(C1-oracle) - DEV(T100). The middle column is the residual bias of section 1 by
slice. It says which slice a "better prior" would have to attack, and whether it is concentrated (one mechanism
could matter) or spread (no single mechanism can). The instruments' p2_slices code already cuts these bands.
Request to the experimenter; no card needed. Built as `ceiling_report bias` (experimenter 17:28Z): columns member / ens8 / c1self / c1oracle / t100 / f1stu, bands x seen-new x raw-ovr-guarded. Reading rules the experimenter registered with the critic: the c1oracle - t100 column is cross-env (C1 2.9, T100 2.7) and held until v2e; ens8 - c1oracle is an upper bound on learnable bias; only the 64-8k and 8k+ bands bound a better-prior idea, and the f1stu column per band separates 10M-learnable from 100M information.

## 6. What 2x-plus would actually need (for the mission, not for this week)

A learner whose inductive bias extracts the residual bias of section 1 from 10M words: better per-occurrence
estimates for familiar words in new frames. Everything measured in that family is small (procedural pre-pretraining
<= 0.013 at 160M, overhypothesis worlds ceiling 0.15, lexicon-first 0.05-0.10, memory readers ~0.05). The D1 map is
the cheapest way to find out whether the residual is concentrated enough for one idea to matter. If it is spread
across every band, the honest answer is that 10M words of English hold about a 2x, and the field-changing result
Atila wants is a different kind of learner, which is a research programme, not a card.

## 7. D1 bias map: read (executor 3, 19:07Z; `ceiling_report bias` @ ref-0421, --guard, twin checks passed; CSV at
wip-branch ref-0175, babylm/gcp/reads/d1_bias/)

All columns torch 2.9.1; member = mean of C3 s0-s7, ens8 = their mixture, c1self / c1oracle / t100 as in the ceiling
map; no F1 student dump exists. Raw leg, text = all (seen text is 0.09% of words; new-text rows match within 0.002).
Nats/word per band, then each band's contribution to the whole (share x per-word difference):

| band | share | member | ens8 | c1oracle | t100 | variance (member - ens8) | learnable-bias UPPER bound (ens8 - c1oracle) | c1oracle - t100 (cross-env, held) |
|---|---|---|---|---|---|---|---|---|
| all | 1.000 | 5.004 | 4.715 | 4.482 | 4.261 | 0.289 | 0.233 | 0.221 |
| c10 = 0 | 0.037 | 17.24 | 15.43 | 14.25 | 12.83 | 0.067 | 0.043 (100M information) | 0.052 |
| c10 = 1-63 | 0.115 | 9.37 | 8.68 | 8.09 | 7.59 | 0.080 | 0.068 (mostly 100M information) | 0.058 |
| c10 = 64-8k | 0.366 | 5.48 | 5.20 | 4.97 | 4.79 | 0.100 | 0.086 | 0.067 |
| c10 = 8k+ | 0.482 | 2.67 | 2.58 | 2.51 | 2.41 | 0.043 | 0.036 | 0.044 |

Guarded leg: all 0.2815 / 0.2310 / 0.2021; only the count-0 band moves. Self-distillation (c1self) is +0.014 over
the member mean, in line with F1.

**Reading (per the experimenter's registered rules).** The learnable-bias upper bound that a better-prior learner could
attack is the 64-8k and 8k+ bands: 0.086 + 0.036 = **0.12 nats/word**, at most ~0.16 if half of the 1-63 band is
counted. It is spread over the two biggest bands in proportion to their share, not concentrated anywhere. So:
- No single better-prior mechanism has a target bigger than ~0.09 nats/word (the 64-8k band), and that is an upper
  bound set by a teacher that saw 100M words.
- The whole legal ceiling, with a perfect posterior (0.29) and a perfect prior (<= 0.16), is <= 0.45 nats/word,
  about 3x by the log-linear rate, and no learner for the prior part exists. 10x is not in the data; 5x is not in it
  either; 2x needs most of the posterior part, of which the legal split captures 0.12.
- The count-0 and 1-63 bands (0.11 of the middle column) are where the 100M model's edge is largest per word, and they
  are information, not learning: no card touches them.

## 8. P-F2e scored: MISS (experimenter 21:48Z; executor 3 @ ref-0228, reads/base_size_m/SHORT_E.md)

Best size-M short-E cell = wd1.6 e3: 5.1745 (twin 5.1502). K=2 e5 mixture 4.8443 minus that = **-0.330**, outside the
registered [-0.06, +0.02]. Scored once, a miss. Size M (12L/768d, ~4x S FLOPs) is 0.14-0.20 *worse* than the S E6
single at every WD and epoch tried, so the wide ~4x single does not come near the split.

**Correction to my own labels in section 2, P-F2e.** The numeric band was registered as "mixture DEV minus M DEV";
negative means the mixture is ahead. My two interpretive labels had the sign reversed ("below -0.06 means M clearly
better" should read "the mixture clearly better"; "above +0.02 means M worse" should read "M better"). The band and
the score are unaffected; the labels are withdrawn. The correct reading of -0.33 is the one the experimenter gave: at 10M
words, width hurts (M loses to S at every WD, even heavy), while depth helps (the 28L/384d menu entry is the live F=2
competitor). That is the opposite of the "bigger with tuned WD keeps winning" rule from Kim/Kotha at 200M tokens, at
least along the width axis at this scale; P-F2d itself already scored a HIT (critic ref-0149: K=2 mixture 0.058/0.059 cal_t ahead of the F=2 28L e6 menu
single, env-robust), so the split beats the best deep single at matched compute by the ~1.2x the experimenter reports. The one
compute-matched check still open is the 28L K=2 claim against the F=4.146 menu. Either way the 1.4x split cannot be
explained away as parameter count.
