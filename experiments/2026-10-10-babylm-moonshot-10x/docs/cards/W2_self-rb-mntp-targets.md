<!-- moonshot design round W2; status alive; ceiling 0.06 nats/word; expected 0.007; P(pass screen) 0.05; screen GPU 120 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card: self-RB, the hybrid's masked mode as a right-context teacher for its causal mode (`self-rb-mntp-targets`; card v1 after three skeptics, 2026-10-10)

**Status: alive, gated.**
- **Skeptic verdicts:** prior-art wounded, rules wounded, mechanism dead. That is 2 wounded and 1 dead, and no rule kills the card without a workable fix.
- **What survives is much narrower than the candidate:**
  - The zero-cost arm (a) is dropped as a screen arm. Once its label leak is fixed, its coverage is 2-5% of positions per epoch, with an arithmetic ceiling of 0.01 or less.
  - Arm b is redesigned so that it is legal under R0 and closer to unbiased.
  - The candidate's 0.45 ceiling is withdrawn.
  - The novelty label drops to "known idea".
  - The in-house GPU null, which the candidate left out, is now reported.
- **No 10M screen slot** is used until all of these hold: the CPU gates (F0) pass, the GPU mini-ruler gate (F1) passes, and ceiling-map C1 has landed.
- **Work so far:** none. Card only: no code, no runs.

## Known idea, not new
**Label: known idea.** The regime is untested: multi-epoch, exposure-capped LM pretraining, with the GPT-BERT hybrid's own masked mode used in place as a right-context teacher for its causal mode. The candidate's "known ideas, new combination" was wrong.

**Same mechanism, other tasks**
- **Dual-mode ASR** (Yu et al., ICLR 2021, arXiv 2010.06030). One shared-weight model's full-context mode is an in-place teacher for its streaming mode, at no extra compute, with a 3-bin KL. LibriSpeech TestOther WER is 8.5 with in-place KD and 10.2 without. Part of that gain is emission timing, which has no LM analogue.
- **Seer Forcing** (Feng et al., ACL 2021, 2106.06751). A co-trained decoder that sees y<i and y>i, but not y_i, is distilled into the conventional decoder. The gain shrinks with data: +1.84 BLEU on small NIST, +0.6-0.8 on large CN-EN. Sprint-1's novelty audit already called this "the same mechanism".
- **SBD-NMT** bidirectional-decoder self-KD (Zhang et al. 2022, 2203.05248).
- **BERT soft labels for autoregressive students:**
  - Chen et al., ACL 2020, 1911.03829. Their teacher was pretrained on 3.3B external words, which confounds external knowledge with right context.
  - Futami et al., Interspeech 2020, 2008.03822.
- **Context distillation** (Snell et al. 2022).
- **SelFusion** (ACL 2026, 2608.22898). Its more-context mode is overconfident on wrong tokens.
- **LUPI** (Lopez-Paz et al. 2016).

**Unconditional LM is where this has measured near nothing.** Twin Networks (Serdyuk et al., ICLR 2018) moved PTB from 58.8 to 58.3 and WT2 from 65.8 to 64.9, about 0.01 nats/token. The authors blame high-entropy targets.

**Theory**
- The candidate's "Rao-Blackwell" argument is Menon et al. (ICML 2021): a Bayes teacher reduces variance, so the quality of the teacher's probabilities is what matters.
- Dao et al. (ICLR 2021) show that both teacher underfitting and teacher overfitting (data reuse) destroy that benefit. Cross-fitting is the fix.

**In-house precedent: a NULL on GPU.** Sprint-1 B_rbdistill used a co-trained future-seeing teacher head in add mode, alpha 0.25, on an A100 with 152M FineWeb tokens in a single pass.

| | arm | shuffled-future null |
|---|---|---|
| full dev (nats/token) | 4.0306 | 4.0284 |
| context-new slice | 5.8111 | 5.8073 |

- Against the base it scored -0.018 [-0.072, +0.035].
- Its CPU pilot was 0.067 worse than its null at step 600, even though teacher_gain was 0.36. The calibration diagnostic it required was never run.
- Sources: sprint1/results/gpu_runs/screen/b_rbdistill/{rb_add,rb_add_null}/seed0/result.json and docs/sprint_1_report.md:59.
- The only argued difference is the regime. A single pass has no repeated one-hot labels to memorise, while 10 epochs refit the same label 10 times. That is the only reason to re-test.

**Same family**
- A1, which uses the same binned KL and is the same target-side regulariser family.
- PS-KD; Tf-KD (the smoothing confound); the Belief State Transformer.
- Kim/Kotha 2025: offline self-distillation, about -0.13 nats.
- Lu & Liu 2026 (2605.23857): distillation helps out-of-domain more readily than in-domain fit.

**BabyLM record**
- GPT-BERT (2024.conll-babylm.24) uses no cross-mode distillation.
- Self-distillation: +0.3-0.5 BLiMP (2024.conll-babylm.3).
- Mutual learning: near chance (2024.conll-babylm.17).
- BootBERT: -0.9 BLiMP (2023.conll-babylm.19).
- No entry uses the hybrid's masked head as a teacher for its causal head. Under R6 that gap does not make the idea new.

The web citations above were gathered by the prior-art skeptic. Their full texts are still to be confirmed by the prior-art check, which can only move the label further toward known.

## Mechanism (as amended)
**Precondition: MNTP.** The mechanism needs the GPT-BERT MNTP hybrid, and every provisional base cell is causal-only:
- `mntp_frac` is 0.0 in every config under shared/babylm/base_r1.
- make_base_jobs has no MNTP stage.

So every arm is **B+M + RB**. Here B+M is the base plus `--mntp_frac 0.25`, with GPT-BERT's p falling from 0.3 to 0.15, the 80/10/10 split, and vocab 16512. B+M is a known lever and is never credited to this card.

**RB teacher pass (arm RB-b, amended)**
1. **Write gate.** The pass starts at epoch 2, so there is no near-untrained epoch-1 teacher.
2. **Teacher forward.** From then on, each causal row gets one extra no-grad forward of the same window in bidirectional mode, under an RB mask:
   - Slots are j ≡ phi (mod 8), with a random phase phi per row per epoch.
   - Every slot is MASK. There are no keep or random slots, so the label never reaches the teacher.
   - Positions j-1 to j-7 are always clean. Right-side tokens are clean except at the other slots.
3. **Target.** The output at j-1 gives q_bi(.|l, r) for y_j. Its top-32 plus a tail bin is the target for the same row's causal target at the same stream position, in the same step. The target is fresh and detached, with no cache and so no staleness.
4. **Loss.**
   - Covered targets: (1-w) CE + w KL(q_bi || p_causal), with the KL over 33 bins (A1's `binned_kl`).
   - Uncovered targets keep plain CE.
   - w is 0.5 or 0.75. The candidate's CE + alpha KL, with alpha at most 1, kept at least half the weight on the one-hot label.

**The tower rule holds only approximately.** The teacher's left context has MASKs at j-8, j-16, and so on. So E[q_bi|l] is p(y | l with every 8th token masked), a slightly blurred predictor. The size of this bias is measured, not assumed (F0a, test b). A stride of 8 was chosen over the candidate's 25% spaced mask to bound this bias, at the cost of half the coverage.

**Coverage.** RB-b covers 12.5% of causal targets per epoch in epochs 2 to E_b. That is 0.75 x 12.5% = 9.4% of all positions, not the candidate's "about 25%".

**RB-a (the zero-cost cache from MNTP rows) is dropped as a screen arm:**
- Writing MASK slots only covers 0.25 x p x 0.8, about 3-6% of positions per epoch.
- Adding a clean-left filter (j-1 to j-3) leaves about 2%.
- Every write after epoch 1 is in-sample, so the ceiling is 0.01 or less.
- It runs only if F1's M-max margin is 0.15 or more.

Its write-time diagnostics on B+M's MNTP rows cost nothing and are logged.

**Logged**
- rb_cov, rb_H, rb_off and rb_t_label.
- The write-time privileged gain on MASK slots, measured against the same-step p_causal.
- Each epoch, on DEV-sub and TRAIN-sub slots: q_bi's label mass and entropy. These give the collapse index C = t_label(TRAIN) - t_label(DEV) and the ratio H_train/H_dev.

## Why (which measured slice)
**Target slice:** familiar words in new contexts.

| slice | nats/word |
|---|---|
| new text only | 0.61 |
| words seen 64-8k times | 0.36 |
| words seen 8k+ times | 0.22 |

**Why the label is the problem.** Dev-minus-train grows every epoch, from 0.51 to 1.11. After epoch 1, a repeated position's one-hot label is a memorisation target. RB replaces part of it with "the words that fit here, given what actually followed".

**Why the base cannot already capture this.** Today the right context reaches y_j's prediction only as context for later tokens, never as a target for y_j. WD, EMA, dropout and size all act in weight space and leave each label's noise intact.

**Honest limits**
- The overconfidence statistic (0.57 vs 0.39) is not headroom: the best temperature buys only 0.013.
- About 80% of the gap is abstraction and new contexts, which no change of target creates.
- The targets channel is bounded by ceiling-map C1, registered at f1 in [0.30, 0.75] with a 100M oracle teacher. A 10M self-teacher sits below that.

**Expected signature:** epoch-like, not data-like.
- Novelty tilt 0.6-0.9 (data sits at 1.04).
- Rare tilt below 1, because never-seen and rare words get poor q_bi.

## Ceiling arithmetic (the 0.45 is withdrawn)
**Noise removed at a covered target** = (1 - (1-w)^2) x V. V = H(y|l,r)/H(y|l), about 0.6 (range 0.5-0.75). That gives 0.45 at w = 0.5 and 0.56 at w = 0.75.

**Converting to nats/word:**
1. Effective data multiplier = 1/(1 - c x removed), where c is coverage.
2. Gain = 0.88 x log10(multiplier) nats/word.
3. Multiply by f1 (about 0.5), because the gap is not all noise-limited.

| design | c | gain before f1 | gain after f1 | note |
|---|---|---|---|---|
| RB-b, w 0.75, E_b = 6 | 0.104 | 0.023 | about 0.012 | |
| RB-b, w 0.75, E_b = 4 | 0.094 | 0.020 | about 0.010 | |
| Best legal design: E* <= 4, two phases per row, w = 1 | 0.19 | | about 0.04 | uses V = 0.75 and f1 = 0.75; 1.38x FLOPs, so it needs a FLOP-matched reference |
| M-max: all 8 phases, w 0.75 (out of track) | 1 | 0.32 (2.3x data) | about 0.16 | label only |

**In-track family ceiling: about 0.06.** This is generous: it allows for some regularisation beyond the noise/N exchange. The family cannot reach 0.88.

## R0 exposure ledger
E* is the frozen base's epoch count. The current provisional cells run 10 epochs.

| run | forwards | exposure | legal? |
|---|---|---|---|
| B, B+M | one per window | 10M x E* | yes (100M at E* = 10) |
| RB-b | plus one no-grad bidirectional forward per causal row, epochs 2 to E_b | 10M x E_b + 7.5M x (E_b - 1) | only if E_b <= 6: 97.5M at 6, 80M at 5, 62.5M at 4; 7 epochs would be 115M |
| nulls (N-left, N-shuffle, 4 LS) | the identical teacher pass, its output used or discarded | same as RB-b | yes |
| F0a | inference on published checkpoints | none (diagnostic) | R0.4 concerns training only |
| F0b, F1 (proxy) | M-max: 8 passes per causal row | about 6.4x the proxy cap | out of track, label only |

- **E_b = min(E*, 6).** The candidate's half-rows arm b (1.375 x E*) is illegal at the current E* = 10, where it comes to 137.5M.
- **If E* > 6:**
  - The confound arm B+M@E_b also runs (critic 18:43Z(c)).
  - The comparator is the better of B@E*, B+M@E* and B+M@E_b.
- **Counting the teacher pass:**
  - Its words count in `words_seen` (the stack.py ledger).
  - A run refuses to start if its registered total exceeds 100M.
  - RB-b's exposure-matched base is reported next to the FLOP ratio.

## FLOPs and the R4.4 plan
- **RB-b cost.** A no-grad forward costs about 1/3 of fwd+bwd. It runs on 75% of rows for (E_b-1)/E_b of training. That gives 1.19x at E_b = 4 and 1.21x at E_b = 6, under 1.25x, so no FLOP-matched reference is needed. Measured tokens/s and wall-clock are reported. If measured FLOPs come out above 1.25x, the arm either gets a FLOP-matched reference (a wider S model, same epochs) or is killed.
- **Two-phase variant.** It costs 1.38x and is not registered. It would need its own amendment plus a FLOP-matched reference.
- **M-max** costs about 3x and runs only as a proxy.

## Null twins
All twins share the arm's seed, data order, vocab 16512, E_b, teacher pass and ledger. Each removes the mechanism's information.

- **N-left.** The identical teacher pass, but under causal attention: the causal mode on the same slot-masked input. The target is p_causal(y_j | the same corrupted left).
  - It removes exactly the right context, and keeps the left corruption, the mode noise and the freshness.
  - A fresh detached target on a clean left would be a no-op (its KL gradient is zero), so the corrupted-left version is the informative twin.
  - The A1-style cached causal null is not needed. RB-b has no cache, so there is no temporal self-distillation to separate out. That null returns if RB-a is ever run.
- **N-shuffle.** Each slot's target is replaced by a q_bi with the same label token, drawn from a reservoir of the last 64 steps' targets.
  - Excluded draws: the slot itself, its window, and the same source file and 2,048-token block.
  - It keeps the label mass, the entropy and the word's typical substitutes, and removes the in-context information.
  - Fallback for types with no entry: a unigram label-smoothing target at the slot's rb_H. The fallback rate is logged.
- **Four position-matched label-smoothing nulls** (the critic 22:45Z family): uniform and unigram smoothing, each H-matched and off-mass-matched.
  - They apply only at the slot positions, from epoch 2.
  - Epsilon is set per epoch from the best arm's logged rb_H and rb_off at those positions.
  - This corrects the candidate's A1 nulls, which smoothed every position and so carried 4-20x the smoothing mass.

**The binding null is the best of these six.** At promotion, the best screen null is picked on DEV before seeds 1-2 and becomes the binding twin.

## Kill rule (registered before any run)
**Setup**
- Size S, seed 0, train_10M, the base recipe the critic accepted, with its hash recorded.
- Every arm uses vocab 16512 (critic 00:12Z).
- B, B+M, and B+M@E_b if E* > 6, are re-run in the same batch on the same GPU type.
- **Metric:** full-DEV nats/word from dev_bpw.py at the final checkpoint. No checkpoint picking.
- **Best arm:** the lower full-DEV score of RB-b w 0.5 and RB-b w 0.75. Ties within 0.005 go to the smaller w.

**Kill if any of these holds:**
1. The best arm's gain over the comparator (the better of B, B+M and B+M@E_b) is below 2x the registered seed floor.
2. It does not beat the best of the six nulls by 2x the floor.
3. BLiMP-fast drops by 2 sigma or more.
4. FLOPs exceed 1.25x and there is no FLOP-matched reference.

**"Mechanism not tested" flag** (reported, never a pass):
- **Trigger:** in epochs 2-4, the written q_bi has H_train/H_dev below 0.5, or a label-mass excess (TRAIN minus DEV) above 0.3.
- **Report:** "mechanism not tested: in-sample collapse".
- **Privileged gain:** it remains only as a "useless teacher" floor. The candidate's rule that "below 0.1 means collapse" had the logic inverted and is withdrawn.

**Mechanism signature** (reported): a novelty tilt below 0.85 or a rare tilt below 1 is labelled "epoch-like regulariser", never "data-like".

**Standard clauses**
- No kill or promotion before the floors are registered.
- **Provisional base:**
  - A kill on a provisional base reads "killed (provisional base)".
  - Any base change to MNTP, WD, LR, EMA or epochs forces one re-screen (critic 22:17Z(b)).
- **Trial accounting:** trial counts are logged per arm. The nulls get tuning parity: six null runs against two arm variants.
- **Promotion:**
  - Exactly one arm is promoted, picked on DEV before seeds 1-2, to 3 seeds, size M and the official eval.
  - Its R4 reference is B+M, or the causal base if that wins on DEV. Both are exported with the same pooling mode (critic 20:31Z and 20:52Z).
  - The 3-seed verdict reuses no screen numbers.

## Odds
- **Ceiling:** about 0.06 nats/word in track. M-max, out of track, is about 0.16 (a label).
- **Typical outcome:** median about 0.005, mean about 0.007.
- **P(pass), about 0.05 overall:**
  - P(passes F0) is about 0.6.
  - P(M-max clears F1, given F0) is about 0.25.
  - P(the in-track arm clears the 10M screen, given F1) is about 0.3.
- **P(gain of 0.1 or more):** below 0.01.

**Most likely result**
- RB-b acts as context-shaped label smoothing on about 10% of reads.
- It lands within 0.01 of B+M and about level with the label-smoothing and N-left nulls.
- The in-sample q_bi drifts toward the label by epochs 3-4.

## Fastest falsification
Each stage below can kill the card. None can confirm it.

### F0a: zero GPU (CPU, about 1-2 h)
**Inputs**
- The published GPT-BERT 10M causal-focus model (it has both modes) at checkpoints chck_10M, chck_30M and chck_100M.
- 1k DEV-sub and 1k TRAIN-sub slots, under the exact RB mask, with a 64-token left context.

**Measurements**
1. **Tower bias b** = CE(y; mean_K q_bi) - CE(y; p_causal), on DEV slots.
   - K = 8 sampled futures. Each future draws y' from p_causal, then an 8-token causal continuation.
   - This is the calibration test sprint-1 required and never ran.
2. **ECE** of q_bi on DEV MASK slots.
3. **Held-out privileged gain g.**
4. **Collapse index C** at chck_30M.

**Kill if any of these holds:**
- b > +0.02 nats/token and its document-bootstrap CI excludes 0. The teacher then implies a worse left-context predictor than the student.
- g < 0.1 nats/token: there is no right-context information to transfer.
- C > 0.2: the in-sample teacher has collapsed onto the label.

**Caveats**
- These are published checkpoints with a different tokenizer and recipe, so F0a gives a label on teacher pathology.
- **Fallback:** if the masked mode cannot be called, or the CPU is too slow, run F0a on F1's B+M per-epoch checkpoints (our tokenizer), or do the sampling on the GPU executor (about 5 A100-minutes).
- **Extra readout on the fallback:** R = mean KL(t || T100) / mean CE(y; T100) is also reported for q_bi, N-left, N-shuffle and the label-smoothing targets. T100 is an oracle diagnostic and out of track.

### F0b: CPU tiny (about 2.5 h on 4 CPUs)
- **Setup:** a 4L/128d hybrid trained on a 250k-word slice of train_1M for 10 epochs.
- **Runs:** B+M at seeds 0 and 1, M-max, M-max-N-left and M-max-N-shuffle.
- **Checks:** the mechanics, and the sign of the effect.
- **Kill** if M-max is behind both of its nulls by more than 3x the B+M seed spread. That was the wrong-sign pattern of sprint-1's pilot.

### F1: GPU mini-ruler
These are proxy labels, and every run counts as a trial.

**Setup:** size S, train_1M, 10 epochs, vocab 16512.

**Runs**
- B.
- B+M at seeds 0 and 1.
- B+M on train_2M (the 1M-to-2M rung).
- M-max: 8 phases, fresh 100% coverage from epoch 2, w 0.75.
- M-max-N-left and M-max-N-shuffle.
- RB-b-probe (one phase, w 0.75) and RB-b-probe-N-left.

**Gates**
- **Kill the card** unless M-max beats B+M, M-max-N-left and M-max-N-shuffle by at least max(0.05, 3x the B+M seed spread) nats/word, **and** sits at 1.3x data or more on the B+M 1M-to-2M rung.
- **No 10M slot** unless RB-b-probe also beats B+M and its own N-left by at least max(0.03, 3x the spread). The arithmetic says a one-phase arm keeps only about 10% of M-max's margin, so this gate also measures how the gain scales with coverage.
- **RB-a** runs only if the M-max margin is 0.15 or more.

### Scheduling gate
- No 10M screen before ceiling-map C1 lands.
- If C1-self does not beat B by 2x the floor, the 10M screen also needs an M-max margin of 0.10 or more.

### The 10M screen (conditional, about 100 A100-minutes)
- **First:** B, B+M, B+M@E_b if needed, RB-b w 0.5 and RB-b w 0.75.
- **Then:** the six nulls, calibrated on the best arm.

## Implementation plan in the harness
**Base branch.** Build on the stacker branch `wip-branch`. It carries A1's `self_distill.py` (`binned_kl`, `plan_positions`, `ls_eps_for_entropy`, `unigram_eps_for_entropy`) and the `stack.py` R0 ledger. The alternative is to merge `binned_kl` into the moonshot branch.

1. **`babylm/harness/self_rb.py` (about 250 LOC)**
   - **Masking:** `rb_mask(x, stride=8, phase, mask_id, gen)` asserts that every slot is MASK and that j-1 to j-7 are clean.
   - **Teacher:** `teacher_targets(raw_model, xt, mode)` runs under `torch.no_grad()`.
     - Mode 'bi' runs under `LV.BidirectionalAttention(raw_model)`; mode 'left' runs causal.
     - Logits come from `raw_model.logits(raw_model.hidden(xt))`; the target is their top-32 plus the tail.
   - **Loss:** `rb_loss(z, y, slots, t_idx, t_p, w)` applies (1-w) CE + w binned_kl at slots and plain CE elsewhere.
   - **N-shuffle:** `ShuffleReservoir`, per token type, over the last 64 steps, excluding the same source file, block and window, with the unigram label-smoothing fallback.
   - **Smoothing nulls:** the position-matched label-smoothing nulls.
   - **Flags:**
     - `--rb_mode {off,bi,left,shuffle,ls_u_H,ls_u_off,ls_g_H,ls_g_off}`
     - `--rb_w`, `--rb_stride 8`, `--rb_phases 1|8`
     - `--rb_start_epoch 1.0`
     - `--rb_null_from <run>`
2. **train.py wiring (about 60 LOC)** in the causal rows of the MNTP branch, next to the KD path:
   - The causal forward goes through hidden()/logits() so the logits are available.
   - The teacher pass runs on the same rows, and `words_seen` is increased by the teacher's words (stack ledger).
   - A pre-flight check refuses any run whose registered exposure exceeds 100M.
   - The teacher forward is compiled under `--compile`.
3. **Tests: `babylm/tests/test_self_rb.py` (about 150 LOC)**
   - `--rb_mode off`, and w = 0, are bit-identical to B+M.
   - Perturbing y_j, including as a keep or random token, never changes q_bi at slot j.
   - Positions j-1 to j-7 are clean.
   - The teacher output at j-1 is keyed to the causal target y[:, j-1].
   - N-left does not change when tokens right of j change.
   - N-shuffle preserves label mass, never draws from self or the same block, and logs its fallback rate.
   - The exposure counter equals the ledger formula and refuses 7 epochs.
   - Resume is bit-exact.
4. **`babylm/moonshot/rb_f0.py` (about 200 LOC):** the F0a tower, ECE and collapse diagnostics, run on published or harness checkpoints. The F0b tiny configuration reuses the harness.
5. **`babylm/moonshot/make_rb_jobs.py` (about 60 LOC):** the F1 and 10M job lists, each recording the recipe hash, E_b, exposure and trial counts.

**Total:** about 1-1.5 agent-days. Outputs go to shared/babylm/attacks/self_rb/<run>/, with curves saved as CSV and PNG.

## Stacking
- **Overlap.** It overlaps A1 and A2: all three are target-side or input-side regularisers, and N-left is A2-like consistency. Per critic 00:12Z, a combination needs its own card and the gains cannot be summed.
- **A1 comparison.** If A1's best arm is screened on the same base, its result is reported next to this card's.
- **Composability.** It composes in principle with A3 and A4.
- **Most plausible rescue: a separate card.** The skeptics rate a cross-fitted bidirectional teacher as the most likely to show a real effect:
  - two half-data hybrids, each supplying out-of-fold, full-coverage targets for the other half;
  - total exposure 80M at E* = 4;
  - it falls under the open "out-of-fold distillation" item.

  It is not part of this card.
