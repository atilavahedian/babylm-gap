<!-- moonshot design round W2; status alive; ceiling 0.2 nats/word; expected 0.005; P(pass screen) 0.03; screen GPU 56 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card (final, folded): out-of-fold contextual re-expression of content words
Candidate key `xfit-gibbs-reexpression`, renamed because the update is not a Gibbs sweep. It is parallel conditional resampling.

**Status: alive, deprioritised.** This is a card only: no code, no runs.
- **Skeptic verdicts:** three (prior art, rules, mechanism). All three say "wounded"; none says "dead".
- **Rules:** once redesigned, no rule kills it. As written, though, the candidate is not legal on the current base, and its 1-nat ceiling is withdrawn.
- **Order of work:** the free CPU check first, then a ~33 A100-minute proxy gate. The 10M screen runs only if the gate passes.
- **Not a 10x lever.** It is a variant in the A2/A4b family, with a ceiling of about 0.2 nats/word.

**Facts that changed the design (checked 2026-10-10).** The provisional base is the round-1 winner `base_S_wd0p4_lr0p5_r1_ve0_tied` @ ref-0065.
- **Recipe:** causal only: muon WD 0.4, lr_scale 0.5, tied embeddings, no value embeddings, no MNTP, no EMA, no dropout.
- **Windows and epochs:** 1024-token stream windows with a fresh random offset and permutation every epoch; **10 epochs**.
- **Score:** full DEV 5.0367 nats/word (dev_bpw). DEV-sub is still falling at epoch 10 (5.262 at 90M words, 5.121 at 100M).

Consequences:
- E* = 10, not 4. The candidate's strict ledger (E_t + 2E* - 1 <= 10) cannot be met with E_s = E*.
- The base already re-chunks every epoch.
- It is not overfitting at 10 epochs, so there is less regularisation headroom than the candidate assumed.

## Label and known ideas

**Label: known idea.** This corrects the candidate's "known ideas, new combination". The search budget ran out, and under R6 an unknown defaults to known.

**Core prior art.**
- **Xie et al., ICLR 2017** ("Data noising as smoothing"; arXiv 1703.02573). Contextual substitution during LM training, with the target following the swap. The Kneser-Ney proposal noises both context and target. Gains over unigram noising: PTB -0.036 nats/token, Text8 -0.006.
- **WhatIf (Lyman & Hepner, BabyLM 2024).** Already tried in BabyLM: POS-matched word2vec substitution of 50-100% of content words over augmented passes plus 1 gold pass.
  - It gained +0.6 to +2.0 on the BabyLM average, with 1 seed and no LM loss reported.
  - It gave no extra gain on top of Contextualizer-style re-chunking.
- **Masked-LM-sampled replacements:** Kobayashi, NAACL 2018; Wu et al. 2019 (conditional BERT); SSMBA (Ng et al., EMNLP 2020); ELECTRA (Clark et al., ICLR 2020), where generator samples replace 15% of pretraining tokens.

**Counter-evidence.**
- **Gao/Zhu et al., ACL 2019.** Hard samples from an LM trained on the same corpus did worse than unigram replacement on 3 of 4 MT tasks (BLEU 35.40 vs 35.45, 42.09 vs 41.69, 34.31 vs 34.61, 28.73 vs 28.97).
- **The sampler is known to be invalid.** Drawing every slot in parallel from one masked forward is the "degenerate block Gibbs" sampler of Goyal, Dyer & Berg-Kirkpatrick (ICLR 2022). It is invalid even in principle, because masked-LM conditionals are inconsistent (Wang & Cho 2019).
- **Re-expression costs likelihood.** Same-data re-expression raised perplexity in Data Diversification (Nguyen et al., NeurIPS 2020).

**Other related work.** Chen et al., ACL 2020 (bidirectional-teacher soft targets).

**In-house evidence.**
- Sprint-1 b_rbdistill was NULL: -0.018 [-0.072, +0.035] vs the baseline.
- Label smoothing hurt by 0.45-0.67 nats.
- A2 (uniform replacement; Chen et al. 2606.16246) gains only past ~60 epochs.
- A4b (cross-fitted half-data generators) is the official FAQ's own example.

**Possible twist, untested.** Out-of-fold masked-LM proposals, re-expressed in place each epoch, under exposure-counted R0 at strict-small. This is pending a completed search and defaults to known.

**Follow-ups:**
- Add Xie 2017 (KN noising), WhatIf 2024, Gao/Zhu 2019 and Goyal 2022 to prior_art.md.
- When the search budget resets, check the CS224N BabyLM resampling/GECA report and 2025-2026 work on MLM- or diffusion-generated augmentation for AR pretraining.

## Mechanism (registered form)

1. **Halves.** Split train_10M's 13,713 BOS-delimited documents into halves A and B.
   - The split uses a seeded hash, stratified by source and balanced to 4.98M +/- 1% words per half.
   - Stratification is needed because Gutenberg is 51 documents of about 50k words each, so an unstratified split comes out lumpy.
   - A4b uses the same split file.
2. **Teachers.** T_A is trained on A and T_B on B.
   - Size S, base recipe, `--mntp_frac 1.0 --vocab_pad 16512` (MASK = 16384).
   - Mask p 0.3 -> 0.15, **E_t = 2** epochs.
3. **Slots (X-content, the default arm).** A slot is a single-token word whose token is an alphabetic word with a leading space.
   - The original type must have a train_10M word count in [64, 8192), i.e. the 64-8k band.
   - The slot must be at least 8 tokens from the edge of the teacher's 1024-token window.
   - Chosen slots must be at least 9 tokens apart (greedy, every eligible slot, rho = 1).
   - Measured on train_10M: 949,657 slots, 5.95% of tokens, none of them 8k+ function words.
   - Originals seen fewer than 64 times are never swapped out, so rare words keep every context they have.
   - **Pre-registered fallback:** if gate g1 measures under 3% realised change, the spacing drops to >= 5 tokens (7.93% of tokens), and g3 must still pass.
   - **X-all** (any alphabetic single-token word: 8.18% of tokens at spacing 9, 63% of them 8k+ function words) is used only if X-content fails g1 at both spacings.
4. **One inference pass.** T_B masks every chosen slot of half A in a single forward, and T_A does the same for half B.
   - The posterior for slot j is read at logit j-1 (the GPT-BERT shift in `levers.mntp_corrupt`).
   - It is restricted to alphabetic leading-space single-token words seen standalone in the teacher's own half.
   - The top-32 is stored (int16 ids, fp16 mass) plus the tail mass, keyed by train_10M position.
5. **Re-expression.** Student epoch 1 reads the real stream.
   - In each epoch e >= 2, every slot is redrawn independently from its stored posterior at T = 1, with the original allowed. The seed is PCG64([seed, e, 2]).
   - Swaps are whole word for whole word, so token count, word ids, line starts and BOS positions stay the same (asserted).
   - Targets follow the swapped text.
   - The student's windowing is untouched. Each window reads the stream of its own epoch, because batches can straddle epochs.
6. **Student.** Base recipe, **E_s = 7**, vocab 16384.

**What it is not.**
- **Not Gibbs.** It does not leave p_data invariant, even with exact conditionals.
- **Spacing helps but is not a full fix.** At least 9 tokens apart removes most local dependencies (the rules skeptic measured 13-22% of slots with another chosen slot within 3 tokens, and 35-54% within 8, at the candidate's rates). But a subject and verb 9 or more tokens apart can still be drawn incoherently; gate g3 measures this.
- **The skeleton is fixed.** Multi-token words, function words, punctuation and positions never move. Even an ideal chain samples p(fillers | skeleton), never a fresh draw from p_data.
- **No new information.** Nothing beyond train_10M enters (data processing inequality). The only new signal is the inductive bias of two masked models trained on 5M words each, plus an input/target perturbation.

**RB arm (the posterior as target): deferred, not in this screen.** As registered, its unbiasedness claim fails three ways:
- It needs a calibrated teacher.
- The stored posterior conditions on MASK at the other slots, not on the drawn values the student actually sees.
- Each left context occurs with exactly one right context, so the student fits the seer posterior rather than p(y|left).

It also overlaps A1/C1, and b_rbdistill was null. It gets its own card only if X promotes and C1 reports f1 >= 0.6. That card needs an N-KD twin (real inputs, the same out-of-fold soft targets) and a calibration gate.

## Why (which measured slice)

**Target: bottleneck 1, familiar words in new contexts.**
- Extra loss on new text only: 0.61 of 0.875.
- Words seen 64-8k times: 0.36.
- Content words: 73% of the gap.

X-content puts all its swaps on 64-8k content types. Each swap shows the drawn type in a frame it has not been seen in, and shows the frame with a new filler.

**Out of reach.**
- Never-seen words.
- The 1-63 band: protected, not helped.
- 8k+ words.
- Bursty words. Cross-fitting barely touches them: only 1.7% of 64-511-band slot types sit at least 90% in one half (mechanism skeptic).

**What the base cannot do.** It sees the same (frame, filler) pairs every epoch. Per-epoch re-chunking and weight decay create no counterfactual pairs.

**Counterweights.**
- WhatIf found that substitution and re-chunking act on the same axis and do not stack, and our base already re-chunks.
- The base is still improving at epoch 10, so the arm gives up 3 useful student epochs to pay for its teachers.

**Ceiling: about 0.2 nats/word for the registered form** (about 1.7x data; range 0.1-0.3).
- Changed words per epoch: 0.95M slots x (1 - p(orig)). With p(orig) unmeasured, 1 - p(orig) is taken as 0.55-0.75, giving 0.52-0.71M.
- Six re-expressed epochs give at most 3.1-4.3M new (frame, filler) pairs.
- At par with fresh words, that takes 10M to at most 14.3M words: 0.88 x log10(1.43), about 0.13.
- Crediting each swap with refreshing the next ~3 positions gives about 0.3-0.38.
- Minus the 3 epochs given up: an estimated 0.03-0.06, unmeasured (B7 measures it).

**Family ceiling.** Even a valid sampler with many sweeps is bounded by the fixed skeleton and the closed system, at about A4b's ~0.5. That ceiling is shared with A4b, not additive.

**The candidate's 1 nat is withdrawn.** It assumed exact conditionals, which is an oracle (the C2-oracle channel, out of track). It also multiplied a corpus-wide data factor by a Rao-Blackwell factor that only applies at the swapped slots.

## R0 exposure ledger (strict reading registered as binding)

| item | words |
|---|---|
| T_A: 4.98M x 2 epochs | 9.95M |
| T_B: 4.98M x 2 epochs | 9.95M |
| one inference pass (T_B over A, T_A over B; frozen, no grad; counted under the strict reading) | 9.95M |
| student: epoch 1 real + epochs 2-7 re-expressed, 7 x 9.95M | 69.7M |
| **total** | **99.5M <= 100M**; unique words 9.95M; no outside model or text |

Words per epoch are identical by construction (asserted).

**Every arm carries its full ledger, shared teachers included.**
- N-type, I-only and L-only share T_A, T_B and the inference pass: 99.5M each.
- N-insample: one teacher on all 10M for 2 epochs (19.9M) + 9.95M pass + 69.7M student = 99.5M.
- B7: 69.7M.
- base@10: 99.5M.

**Pipeline counter.** The R0.5 milestones and the word audit use a pipeline counter, not words_seen: student words plus a 29.9M offset, logged as `pipeline_words`.

**Ruling requested.** Does a frozen teacher's forward pass over the whole corpus count as training input? The A4 ruling ref-0102 (2) covered 64-token seeds only.
- If the critic rules that it does not count, a separately registered arm X-L (E_t = 2, E_s = 8) may follow. It is not part of this screen.

**E* contingency.** E_s = min(E*, 7) with E_t fixed at 2.

## FLOPs (R4.4)

(19.9M x 6N + 9.95M x 2N + 69.7M x 6N) / (99.5M x 6N) = **0.93x** the 10-epoch base.
- R4.4 does not bind. The candidate's 2.0x came from assuming E* = 4.
- Teacher vocab padding and top-32 storage are negligible.
- At size M the teachers are size M, and the ratio is the same.

## Null twins and controls

All students use vocab 16384, so ref-0062's shared-init concern does not arise. If any student arm needs the padded vocab, they all get it.

- **N-type: the binding plausibility null.** The mechanism's information is context fit.
  - Same teachers, inference pass, slots and realised change mask as X.
  - Each slot that X changed gets a filler drawn from the type-averaged out-of-fold posterior of the original type (the mean of the stored posteriors over that type's slots in that half), with the original excluded. Targets follow.
  - It keeps the paradigmatic class and the change rate, and removes the context fit, at zero extra forwards.
  - This is Kobayashi's comparison. Gao/Zhu 2019 predict a tie.
- **N-insample: the binding cross-fit null.** One teacher on all of train_10M for 2 epochs rewrites its own training text: same slot rule, one pass, E_s = 7, same ledger. Its realised change rate is reported.
- **B7: the binding control for the epoch handicap.** The base recipe at 7 epochs on real text (the analogue of A4's N0').
- **base@10:** the frozen base, re-run in the batch.
- **Decomposition twins** on X's exact draws (diagnostics, with relabel rules):
  - **I-only:** swapped inputs, original targets (plausible distractors, A2 family).
  - **L-only:** real inputs, swapped targets (sparse out-of-fold label noise / KD).
- **N-unigram:** a sanity row in the proxy only, with the same change mask (original excluded). As a binding null it would be a strawman.

## Kill rule (registered before any run)

**Setup.**
- Size S, seed 0, the provisional base above (recipe hash recorded).
- All 7 student runs and 3 teachers run in one batch on one GPU type.
- Metric: full-DEV nats/word from `dev_bpw.py` at the final checkpoint, with no checkpoint picking.
- sigma = the critic's registered seed floor for this base, still pending. The planning value is 0.01-0.015 (unregistered), so 2 sigma is about 0.02-0.03.
- No kill or promote before the floor is registered.

**Gates on the size-S teachers, before any student runs.** If any gate fails, the students do not run, and the verdict is "mechanism not exercised (untested)".
- **g1:** realised change >= 3.0% of tokens per re-expressed epoch (mean over epochs 2-7), after the registered spacing fallback.
- **g2:** teacher calibration slope (empirical frequency of the original vs deciles of p(orig), on held-out out-of-fold slots) in [0.85, 1.15].
- **g3:** parallel-vs-sequential incoherence ratio <= 1.5.
  - Take slot pairs 9-16 tokens apart, re-run the teacher with the first draw filled in, and count second-slot draws with probability < 1% under that re-run.
  - The ratio is that fraction for parallel draws divided by the same fraction for sequential draws.
- **g4 (R5):** 13-gram check of every re-expressed stream against every eval set. Only 13-grams that cover a changed token are new. Any non-trivial overlap voids the screen until explained.

These gates replace the candidate's teacher-NLL gate, which any bidirectional model passes automatically. The in-sample vs out-of-fold p(orig) is reported as a cross-fit diagnostic.

**Kill X if any of the following:**
- DEV(X) > min DEV{base@10, B7, N-type, N-insample} - 2 sigma.
- BLiMP-fast drops >= 2 sigma vs base@10 (agreement subtasks reported separately).
- **Fade:** X's DEV-sub gain over B7 at the end is more than 0.5 sigma below its gain at epoch 4. The two runs share a schedule; this is the 18:37Z rule applied against the matched control.

**Relabel rules** (reported; X stays the only arm that can be promoted):
- I-only within 1 sigma of X: "input noise (A2 family)". The result goes to A2, and a combined arm would need its own registration there.
- L-only within 1 sigma of X: "sparse cross-fitted KD / label noise".
- X raises TRAIN-sub loss vs B7 while lowering DEV: "regulariser, not data".

**Mechanism check** (reported, not decisive):
- Novelty tilt >= 0.8. Not decisive, because regularisers also score about 0.96.
- At least 50% of the gain on 64-8k words in new text.
- The seen-text vs new-text split.

**Promotion.** Only X can be promoted; X-content or X-all is fixed at the proxy gate, before any 10M run.
- Promotion means 3 seeds each of X, N-type, B7 and base@10; size M with size-M teachers (R4.3); the official eval; and FINAL scored once, on the critic's call.
- As a regulariser/augmentation-type attack it falls under ref-0086(b): any later WD/EMA/MNTP/dropout change to the base forces a re-screen, and a kill reads "killed (provisional base)".
- Every CPU, proxy and screen run goes in the trial log.

**Rulings requested before any GPU run:**
1. Whether frozen-teacher inference counts as exposure.
2. B7 and base@10 jointly as the binding comparator when the arm runs fewer student epochs than the base.
3. The 2M proxy gate as a falsification under ref-0110.

## Odds (guesses, stated plainly)

- **Ceiling:** about 0.2 nats/word for the registered form; about 0.5 for the family (shared with A4b).
- **Median:** +0.005 nats/word vs base@10, 80% interval [-0.04, +0.03]; +0.01 to +0.02 vs B7.
- **Probabilities:**
  - CPU gates pass: about 0.6.
  - Advance from the 2M proxy gate: about 0.12.
  - **Pass the 10M screen: about 0.03.**
  - Gain >= 0.1: under 0.01.
- **Most likely failure:**
  - About 3-4% of tokens change, as content swaps from a 2-epoch masked teacher trained on 5M words.
  - X, N-type and N-insample tie (the Gao/Zhu prediction), all close to B7.
  - All of them land 0.02-0.06 behind base@10, because of the 3 student epochs given up.

## Fastest falsification

**CPU, about 1.5 h, free.** Train two 4L/128d masked teachers (`--mntp_frac 1.0`) on 0.5M-word halves of train_1M, 2 epochs each. Apply the registered slot rule and one pass, then measure:
- g1 for X-content at spacings 9 and 5, and for X-all.
- g2 and g3.
- The frequency-band and POS mix of the changed slots.
- The unigram KL between the swapped and real streams.
- Read 200 swapped sentences by eye.

**Stop** if both X-content and X-all fall under 3% realised change, if g3 > 1.5 at spacing 9, or if the read shows systematic agreement or grammar damage. Tiny-teacher numbers test mechanics only; g1-g3 are re-measured on the size-S teachers before any student runs.

**GPU proxy gate:** mini-ruler on train_2M with 1M-word halves; about 33 A100-minutes, or about 23 if the 5M cells already exist at this recipe.
- **Runs:**
  - B2 (base, 2M x 10) and X (E_s = 7; teachers 2 x 1M x 2 epochs), seeds 0 and 1.
  - N-type, N-insample, I-only and L-only, seed 0.
  - B5 (5M x 10), seeds 0 and 1.
- 2M rather than 1M, because 0.5M-word teachers cannot test the mechanism (rules skeptic).
- s = DEV(B2) - DEV(B5), a 2.5x data step. sigma = the pooled seed SD at 2M.

**ADVANCE** to the 10M screen only if all of these hold:
- mean(X) - mean(B2) >= max(0.3 s, 2.5 sigma), which is about 1.3x data.
- Both seeds show a gain.
- X beats both N-type and N-insample by >= 2 sigma.

Otherwise **KILL** the card; the proxy may be used for falsification under ref-0110.

**Relabels at the proxy:**
- I-only or L-only within max(0.1 s, 2 sigma) of X: relabel as in the kill rule.
- X raises TRAIN-sub loss while B5 lowers it: "regulariser, not data".

Proxy numbers are labels, never verdicts. Every proxy run counts in the trial log. The tokenizer and the type bands carry the label "saw 10M".

## 10M screen4

| step | time |
|---|---|
| Teachers T_A and T_B | about 44 s each |
| Teacher T_all | about 88 s |
| Inference passes | about 1 min |
| base@10 | about 7.3 min + about 2 min evals |
| B7, X, N-type, N-insample, I-only, L-only | about 5.1 + 2 min each |

MPS packing is allowed. Every run logs exposure by source (teacher, inference, real, re-expressed), the pipeline counter, tokens/s and FLOPs, and saves its curves as CSV and PNG.

## Implementation plan (harness; about 800 LOC, about 2 agent-days)

Branch `wip-branch`, after the card is accepted.

- **`babylm/data/halves.py` (~80 LOC).** Source-stratified, word-balanced document split into train_10M_A/B, plus the 1M/2M proxy halves.
  - Carries word, line and BOS metadata and a position map back to train_10M.
  - Shared with A4b.
- **Teachers.** The existing `train.py --split train_10M_A --mntp_frac 1.0 --vocab_pad 16512 --epochs 2`.
  - The all-MNTP path runs (`mntp_rows(mic, 1.0)` returns mic), but the logged causal loss is 0.
  - Add a teacher-mode masked-NLL diagnostic on DEV-sub (~20 LOC).
- **`babylm/moonshot/xfit.py` (~250 LOC).** Load a teacher and switch it to bidirectional attention.
  - Tile its half into 1024-token windows and select slots (registered rule, deterministic).
  - One masked forward per window; read logits at j-1; restrict to the proposal class; store top-32 plus tail.
  - Build the per-type averaged posteriors for N-type, and an in-sample mode for N-insample.
  - Diagnostics: g1-g3, band/POS mix, unigram KL.
- **`babylm/moonshot/xfit_stream.py` (~120 LOC).** Per-epoch draws with PCG64([seed, e, 2]).
  - Returns (input stream, target stream) for modes X / I / L / ntype / nunigram.
  - Asserts word ids, line starts and BOS are unchanged, and that the N-type change mask equals X's.
- **`train.py --xfit_post DIR --xfit_mode M --exposure_offset W` (~70 LOC).**
  - `EpochSampler.next_starts` also returns each window's epoch.
  - `gather_windows` takes x from that epoch's input stream and y from its target stream (PlanSampler path: `epoch_of[w]`). Swapping one stream per step would be a bug, because batches straddle epochs.
  - At most two epoch streams are live at once. Epoch 1 reads the real stream.
  - `pipeline_words` drives the R0.5 milestone labels.
- **R5 (~60 LOC).** 13-gram check restricted to the n-grams that cover changed tokens.
- **Tests (~200 LOC):**
  - Counts, lines and BOS preserved.
  - Only eligible slots change, and the spacing is respected.
  - The masked token never reaches the teacher: perturbing the original token leaves the slot's posterior unchanged.
  - The posterior is read at j-1.
  - Draws are deterministic per seed and epoch.
  - A batch straddling two epochs takes each window from its own epoch's stream.
  - With the flags off, the run is bit-identical to the base path.
  - N-type's change mask equals X's.

## Stacking and overlap

- **A3 (BPE-dropout):** orthogonal. Apply it after re-expression; slots are defined on canonical tokens.
- **A1:** not stackable as written, because its position-keyed targets would see a different token each epoch.
- **A2:** overlaps; I-only is an A2-family arm. A combination needs its own card, and the gains do not sum.
- **A4b:** shares the halves split and the exposure budget. It is an alternative use of the same budget, not additive.

## Skeptic fixes: disposition

**Adopted:**
- Relabel as known idea, with the added citations.
- Rename the sampler and drop the invariance claim.
- Honest ceiling.
- RB arm deferred.
- Clean-target (I-only) and label-only (L-only) twins.
- A proxy decision before any 10M run, at 2M rather than 1M.
- The base's per-epoch re-chunking stated as a headroom limit.
- The automatic teacher-NLL gate replaced by calibration and incoherence gates.
- Strict ledger with a pipeline counter and the E* contingency.
- Spacing >= 9 instead of a grouped sweep.
- Alphabetic slots only (no ' =' or punctuation).
- N-type null, and a change-mask-matched N-unigram.
- R5 13-gram check.
- One promotable arm, fade rule vs B7, size M registered.
- Per-window epoch streams and the j-1 read.
- TRAIN-sub data-vs-regulariser label.
- Proposal-class restriction with tail logging.

**Not adopted:**
- The 2-group sequential sweep at E_s = 3: a 3-epoch student against a 10-epoch base cannot win.
- An MH accept step: it needs extra forwards, which the strict ledger cannot pay for.
- The self-teacher variant: the base has no MNTP head, so it would change the student. Follow-up idea only.
- The lenient E_s = 8 arm: it needs a ruling first.
