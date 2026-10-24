<!-- moonshot design round W1; status alive; ceiling 0.1 nats/word; expected -0.01; P(pass screen) 0.04; screen GPU 136 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card: paired cloze-teacher targets (rb-privileged-future-targets) — final card, 2026-10-10

**Status.** Card only. No code and no runs. This card folds in three skeptic verdicts, and all three said "wounded"; none said "dead". It stays alive at the **lowest priority**, behind three cheap gates (G0 to G2). It also waits for critic rulings on Q1 to Q4 at the end of this card.

**Label: known idea, untested regime.** The untested regime is same-data, exposure-capped LM pretraining on a GPT-BERT causal+MNTP hybrid. The candidate claimed it found "no use of a hybrid's own bidirectional mode as a teacher". That claim is withdrawn.

## Known, not new (prior art)

**The mechanism itself**
- Dual-mode ASR (Yu et al., ICLR 2021, arXiv 2010.06030). One set of shared weights runs a streaming mode and a full-context mode in the same batch. The full-context mode is distilled into the streaming mode in place, using `stop_gradient(fullcontext_pred)`. On LibriSpeech TestOther, streaming WER is 8.5 with in-place KD and 10.2 without it.
- This card's paired design is that same in-place distillation, with MNTP as the full-context mode and causal LM as the streaming mode.

**Bidirectional teachers for left-to-right text students**
- C-MLM, Chen et al. (ACL 2020): +1.54 BLEU on IWSLT De-En. The teacher was a BERT pretrained on external text. A causal-teacher ablation scored 35.20, against 35.27 for the baseline.
- Futami et al. (Interspeech 2020): WER 10.31 to 9.19. The teacher was trained on external text first.
- Seer Forcing (Feng et al., ACL 2021).
- Twin Networks (Serdyuk et al., ICLR 2018): PTB perplexity 58.8 to 58.3, which is about -0.01 nats/token. The authors called this "minor".
- Target-bidirectional agreement (Zhang et al., AAAI 2019).

**The Rao-Blackwell / variance argument**
- Menon et al. (ICML 2021), Lemma 1. The same paper notes that the result does "not directly hold for self- or co-distillation [where] the same sample is used for both models", and that is exactly the setting here.
- Learned MLM conditionals are mutually inconsistent (Young et al. 2023; Goyal et al., ICLR 2022), so the exact tower-property argument does not hold for a learned teacher.

**LUPI and self-distillation**
- LUPI and generalized distillation: Vapnik & Izmailov 2015; Lopez-Paz et al., ICLR 2016. Their fast-rate argument is assumed, not proven, and it is not used here.
- Self-distillation: Tf-KD (CVPR 2020) found self-KD gains are largely learned label smoothing. A1 (PR #55) is this card's in-flight relative.

**BabyLM**
- GPT-BERT (Charpentier & Samuel 2024) and AntLM combine the two objectives but do not distil one mode into the other.
- Syntactic Smoothing (Diehl Martinez et al., EMNLP 2024) uses substitute-class soft targets at 10M words: BLiMP 73.2, the same as plain label smoothing (73.2), against 71.4 for the base.
- BootBERT (2023.conll-babylm.19) uses an EMA teacher with a privileged view: BLiMP -0.9 and GLUE +1.4 at 100M.

**Local evidence against target smoothing**
- In the sprint-1 ladder, label smoothing hurt by 0.45-0.67 nats.
- Distillation tied at matched compute (prior_art.md, lines 72 and 81).

**Search limit.** The skeptics' shared web-search budget was exhausted. Their evidence comes from direct paper fetches and complete BabyLM 2023-2025 title lists, so any claim that something has not been done is weak. The label can only move further toward "known".

## What changed from the candidate (skeptic fixes adopted)

1. **Arms a and b are dropped.** These were the live-EMA teacher and the frozen epoch-1 EMA teacher, each run on a separate masked forward.
   - Under R0.2 (critic 19:11Z), a separate masked forward over the same text counts twice.
   - The frozen base already uses 99.54M of the 100M words, so the candidate's own formula gives teacher coverage f = 0.
   - The alternative, an E* <= 6 student plus teacher passes, pays an unmeasured epoch tax against a base that is still improving at e10. This is reported as a label only.
2. **The new primary design pairs rows.** Under the base+MNTP ledger, the teacher view is the MNTP training row the hybrid already pays for, co-scheduled on the same window as a causal row. No forward and no exposure are added.
3. **Spans (r = 1) are dropped.** On a language-like HMM with exact posteriors (skeptic 3's `hmm_rb.py`), bias makes up 52-70% of the r = 1 view's target error, so that arm is KD toward a distribution that ignores x_{j-1}.
   - Targets are now only MNTP's own iid single-token **MASK** positions. MNTP's keep (10%) and random (10%) positions are excluded, because they would leak the label or add noise.
   - The masking density is MNTP's own schedule, p = 0.3 to 0.15. The view stays in MNTP's training distribution, and about 15-30% of the left context is masked.
   - On the HMM, an r = 0 view at a 0.25 mask rate has bias^2 of 0.001, against an MSE of 0.103.
   - The words "unbiased" and "Rao-Blackwell" are dropped from the claims. This is **approximately-RB KD toward a cloze posterior**.
4. **The top-k plus tail-bin cache is dropped** because of the Anshumann 2025 bias. The cache arm instead stores 16 samples per position, drawn from the teacher's posterior, and trains toward their histogram. This is sampled KD, an unbiased estimate of training toward the full posterior.
5. **No EMA in any arm.** At train.py:531 the EMA shadow becomes the evaluated and exported model, which would change the scored weights. The teacher here is the live model, with its outputs detached.
6. **Twin 1 is dropped.** It ran causal attention over MASK input, which is off-distribution because causal rows never contain MASK. Entropy-matched and off-mass-matched smoothing nulls are added, following the A1 precedent (critic 22:45Z), plus a left-only self-teacher twin.
7. **R_hat is dropped as a mechanism diagnostic.** It diverges exactly when the teacher carries no privileged information. A teacher-memorisation test replaces it.
8. **The LUPI rate argument and the 0.9 ceiling are dropped.** The ceiling is restated below, including the coverage limit the skeptics did not price in.
9. **FLOPs are restated.** The separate-teacher arms would have cost about 1.25-1.6x the base. The paired arms cost about 1.0-1.05x.

## Premise and new evidence (checked 03:51Z)

**The frozen base has no MNTP.** It is the round-1 winner `base_S_wd0p4_lr0p5_r1_ve0_tied` (critic 02:53Z), and its config shows mntp_frac 0.0, ema_decay 0.0 and epochs 10. It scores 5.0367 on DEV (dev_bpw), with 99,541,960 words seen.

**The all-on cell landed at 03:25Z, after the skeptics' reads.** It is `base_S_wd0p4_lr0p5_r1_all`: MNTP 0.25, dropout 0.1, head_wd 0.1, tied, ve0, baseline_reg.

| | DEV (dev_bpw) | DEV-sub | train-sub | train-to-DEV gap | Throughput |
|---|---|---|---|---|---|
| All-on cell | **5.1023** | 5.20 | 4.41 | 0.79 | 278.6k tok/s, compile 154 s |
| Frozen causal base | 5.0367 | 5.12 | 3.83 | 1.29 | 366.2k tok/s, compile 37 s |

- The all-on cell is **0.066 nats/word worse** than the frozen base. That is one seed, and MNTP is confounded with dropout and head_wd.
- The hybrid memorises less.
- The winner rule (02:53Z) says only this cell could re-name the base, and only by more than 0.02. It did not, so the frozen base stays causal-only.

**Consequences**
- MNTP enters only as part of this attack, with its own confound arm N1. MNTP's own effect, positive or negative, is never attack credit.
- The binding comparator is **the better of the frozen causal base and N1**.
- If MNTP alone costs about as much as all-on did, the KL must recover the whole handicap. That is why gate G2 exists.

## Mechanism (registered)

The arms are built on the paired hybrid:
- mntp_frac 0.25, with mntp_rows giving k_m = mic/4.
- In each micro-batch, the sampler draws n_c = mic - k_m distinct windows. The k_m MNTP rows are **copies of the first k_m causal windows**, masked with `levers.mntp_corrupt` using MNTP's own p schedule and the 80/10/10 split.

**R-a (paired live cloze target)**
1. Run the MNTP row's forward, with grad, through the split API under `BidirectionalAttention`. Its MNTP loss is unchanged.
2. Read pi_j = softmax(z_bi[j-1]).detach() at the **MASK-replaced** positions only.
3. For the paired causal row: loss = CE + alpha * sum_{j in M} KL(pi_j || q_causal(.|x_<j)) / N_causal_positions.
   - The weighting is per position, so a targeted position's effective target is m = (onehot + alpha * pi) / (1 + alpha).
   - The KL is over the full vocabulary (16512 padded), at T = 1, with alpha = 1.0, starting at fed-epoch 1.0.

**R-b: R-a plus a sampled cache**
1. Each MASK target writes 16 samples from pi_j (int16) to a cache keyed by absolute stream token position, using A1's key scheme. That is 32 bytes per position, about 0.5 GB.
2. At later causal visits, a position with a cache entry and no live pi trains toward the histogram of its samples, with the same alpha.
3. Hit rate and staleness are logged.

**Coverage, the share of causal predictions that receive a target**
- R-a: (k_m/n_c) * mean(p) * 0.8 = (1/3)(0.2175)(0.8) = **0.058**.
- R-b: rises from about 0.04 at epoch 1 to about 0.35 at epoch 10, **about 0.2 on average**.

**Logged per step:** coverage, teacher mass on the label, sum of pi^2, H(m), mass moved off the label (for null calibration), KL, and cache hit rate and staleness.

## Why (which measured slices)

**Bottleneck 1: known words in new contexts.** The targets are the 64-8k band (0.36 nats/word) and the 8k+ band (0.22); content words make up 73% of the gap. One-hot labels teach the realised word in each repeated context. The cloze posterior, which sees the right context, spreads mass over the words that fit both sides (paradigmatic substitutes).

**Secondary targets:**
- overconfidence: loss minus own entropy is 0.57, against 0.39 for the 100M model;
- late-epoch memorisation: extra epochs have a novelty tilt of 0.43.

**Expected signature, measured against the best twin, not the base:**
- novelty tilt >= 0.8;
- **rare tilt < 1**, i.e. the gain lands on words seen 64+ times. The teacher knows words seen 1-63 times mostly at their own memorised occurrences, and distillation under-serves tail classes (Lukasik 2021).
- Never-seen words (0.10-0.11) are out of reach.

**Why a bigger, better-regularised base does not capture this:**
- WD, size, dropout and epochs change the fit, but never the target.
- MNTP shares weights, but it passes right-context knowledge to the causal mode only through shared representations, never as causal targets. N1 measures exactly that transfer, so the attack gets credit only for the KL's increment over N1.

## Ceiling (corrected)

**Per position (skeptic 2).** With an exact posterior, cross-entropy toward m = (onehot + alpha * pi)/(1 + alpha) has an effective variance ratio R_eff = (1 + alpha)^2 R / (R + alpha^2 + 2 alpha) <= (1 + alpha)^2.
- R is about 2-3 at the candidate's LM-typical collision probabilities, and about 7 on skeptic 3's language-like HMM.
- That gives R_eff of 1.6-2.8 at alpha = 1.

**Coverage, which the skeptics did not price in.** One bidirectional forward yields near-clean targets only at its own masked positions. Pushing past p of about 0.3, or adding spans, buys coverage with bias. Under the generous variance-limited model, the effective-sample multiplier is 1/((1-c) + c/R_eff) at 0.88 nats per decade:

| Variant | Coverage c | Multiplier | Nats/word |
|---|---|---|---|
| R-a | 0.058 | 1.02-1.05x | **0.008-0.018** |
| R-b | about 0.2 | 1.08-1.19x | **0.03-0.07** |
| Dense (c = 1; needs extra views, so out of track at the 10-epoch base) | 1 | 1.6-5x | 0.18-0.62 |

- Every variant sits under the C1 targets channel: f1 x 0.70, which is **0.21-0.53**.
- An in-sample self-teacher gets only a fraction of the oracle figures.

**In-track ceiling: about 0.05 nats/word with an oracle-quality teacher, and at most about 0.1** if soft targets act beyond label variance (C1's upper band x 0.2 coverage).

The family's dense version needs more views than R0 allows. One alternative is XLNet-style two-stream attention, which costs about 1.6-2x FLOPs (R4.4) and needs its own ruling; it is parked. **This is not a 10x mechanism.** At best it is a regulariser-sized cousin of A1.

## R0 exposure ledger (each arm)

| Arm | Training words fed | Teacher | Extra forwards | Total |
|---|---|---|---|---|
| Causal base B (re-run) | 99.54M causal rows | none | 0 | 99.54M |
| N1, R-a, R-b, N2, N3 x4, N4 | 99.54M: about 75% causal rows and about 25% MNTP rows; the paired MNTP rows duplicate same-step causal windows and **are counted again** (19:11Z) | the live model; its targets come from the MNTP row it already trains on | 0 | 99.54M |

- Each window gets about 7.5 causal and 2.5 MNTP views in expectation, the same as unpaired base+MNTP.
- R0.1: 10M unique words.
- No external model, no generated text, and nothing trained elsewhere.
- R5 leakage: clean, because no synthetic text is used.
- An assertion checks that words_seen is within 0.1% of N1's.
- The candidate's "zero or near-zero extra exposure" claim held only for this paired form.

## FLOPs and R4.4

**R-a and R-b cost about 1.00x by the harness's own FLOP count** (flops_per_token 193.76M against 193.46M, from the vocabulary pad).
- Counting full bidirectional attention scores in the MNTP rows, the cost is at most about 1.05x the causal base and 1.00x N1.
- The KL is element-wise over the vocabulary at under 25% of causal positions, so it adds under 1%.
- Wall-clock is reported, not judged (A3 ruling): the all-on cell ran at 0.76x the base's tokens/s.
- R4.4 does not bind at screen time. Promotion still needs the 01:20Z FLOP-matched reference if a measured multiple exceeds 1.25x.

## Comparators and null twins (same code path, seed, data order, vocab_pad 16512, same batch and GPU type)

| Run | What it removes |
|---|---|
| **B**: frozen causal base re-run (`--vocab_pad 16512`, A2 amendment) | — |
| **N1**: paired hybrid, alpha = 0 (confound arm) | the targets; it keeps MNTP and the pairing schedule |
| **N2**: label-keyed shuffle | context specificity. pi_j is replaced by a cloze posterior of another MASK position with the **same label token**, drawn from a cross-batch bank (FIFO of 4 per token id, stored as 16 samples). The KL is applied only where the bank has a partner, and the arm uses the same eligibility mask, so coverage is matched. Coverage is reported per frequency band. |
| **N3 x 4**: {uniform, unigram} smoothing x {H-matched, off-mass-matched} at the same positions and alpha, epsilon from the best arm's logs (A1 tool) | all context; it keeps the smoothing strength |
| **N4**: A1's temporal causal cache at the same positions, sample count and alpha | right-context (privileged) information. It is a left-only self-teacher that is in-distribution, and it is staleness-matched to R-b (for R-a it is stale; this is labelled). |

**Binding:** best arm vs max(B, N1), and best arm vs the best of {N2, N3 x 4, N4}.

## Kill rule (registered before any run)

**Setup:**
- Size S, train_10M, 10 epochs, seed 0.
- Metric: **full-DEV nats/word from dev_bpw.py on the fp32 HF export of the final 100M checkpoint**. There is no checkpoint picking, and every delta is read on the same instrument.
- Pad-cost guard (01:33Z): arms are scored unmasked; B and the nulls at the better of masked and unmasked.

**Kill if any of:**
- **K1:** the best arm minus max(B, N1) is under 2x the seed floor.
- **K2:** the best arm minus the best of {N2, N3 x 4, N4} is under 2x the floor.
- **K3:** BLiMP-fast drops by 2 sigma or more against the better comparator.
- **K4:** FLOPs exceed 1.25x with no FLOP-matched win, at promotion.

**Floor:** not registered yet, so a screen reports numbers only, with no verdict. Seeds 0 and 1 of the winner differ by 0.0035 on the harness instrument, which suggests 2x the floor is about 0.005-0.01. That is a hint, not a floor.

**Re-screen:** this is a regulariser-type, target-side attack. Any later base change to WD, EMA or MNTP forces a re-screen (22:17Z(b)).

**Promotion:** one arm, picked on DEV before seeds 1-2 run, goes to 3 seeds and size M. Its comparators are N1, B and the best screen null, 3 seeds each.
- GLUE uses the hybrid family's bidirectional AutoModel, declared in the export metadata before scoring (20:31Z(b)). AutoModelForCausalLM stays causal, so the DEV leg is unaffected.
- Against B, the GLUE leg carries the label "cross-family pooling", and the multiplier carries "cross-family" unless a hybrid ruler exists (19:11Z #1).

## Mechanism checks (reported; one can withhold promotion)

**Teacher memorisation.** At 10M, 50M and 100M words, measure the teacher's label mass and NLL at MNTP-style MASK positions on train-sub vs DEV-sub, and the causal mode's at the same positions.
- If the teacher's train-to-DEV gap exceeds the causal mode's, and the arm experimenters the best null by less than 0.03 on DEV-sub at 50M, stop the arm and label it "teacher memorised".
- An arm that passes K1-K3 while the teacher's gap is larger is reported as "passes, privileged-information mechanism not shown".

**Privileged-information signature.** Regress the per-word-type DEV gain (arm minus best twin) on that type's mean training-time H(q_causal) - H(pi). A smoothing-like gain gives a flat profile.

**Tilts.** Novelty and rare tilts on the arm-minus-best-twin difference.

## Fastest falsification (cheapest first; each gate's park rule is registered now)

**G0, CPU, about 1-2 hours on 4 cores, free.** `babylm/moonshot/rb_toy.py`.
- The language is an exact HMM: 40 states, 300 words, transitions Dirichlet(0.02), emissions Gamma(0.01) x Zipf, giving E sum p_c^2 of about 0.22.
- The model is a tiny 2L/64d paired hybrid with 25% MNTP rows. The base WD is tuned over 3 values on held-out data.
- The ruler: 25k, 50k, 100k, 200k and 400k tokens, each for 10 epochs.
- Arms at 50k tokens, 2 seeds each:
  - causal base and N1;
  - causal-oracle KD at the paired positions (the C1 analogue);
  - **RB-oracle-paired**: the exact posterior under the MNTP row's own masks (the in-track ceiling);
  - RB-oracle-dense: the exact p(x_t | x_-t) everywhere (the out-of-track family ceiling);
  - R-a self, R-b self, N2, and N3 (unigram, H-matched).
- Each arm's data multiplier is read off the toy ruler.
- **Park if** RB-oracle-paired is under 1.1x, **or** the best self arm minus max(N1, N2, N3) is under 25% of (RB-oracle-paired minus N1).
- Also report RB-oracle-dense / causal-oracle, which is predicted to be under 1.
- The HMM is the mechanism's best case, so failing here is decisive and passing is only a label.

**G1, GPU mini-ruler, about 12 A100-minutes, labelled proxy.**
- train_1M x 10 epochs at S: B with 3 seeds (giving sigma_1M), N1, R-a, R-b, N2, N3u-H, and B on train_2M as a ruler point.
- **Park ("killed (proxy)") if** the best arm minus max(N1, N2, N3u-H) is under max(0.02, 3 x sigma_1M).
- The proxy rules apply (00:40Z Q2): the trials count in the trial log, and the base recipe is unchanged for the proxy. The MNTP arms are attack-owned and labelled "attack-owned hybrid comparator".

**G2, at S, about 24 A100-minutes, run as the first two jobs of the screen batch.** B re-run and N1.
- **Park if N1 is more than 0.03 nats/word worse than B on dev_bpw.** The arm would then need a mechanism gain of at least 0.03 plus 2x the floor, about 4x its central guess.
- The result is logged as "parked (MNTP handicap)", not as a mechanism kill. It can be revived if the base builder ever adopts MNTP.
- N1 never feeds the base grid (ref-0086 firewall).

## Screen (size S, seed 0; launched only if G0-G2 pass)

| Run | Note |
|---|---|
| B | the frozen causal base, vocab_pad 16512 (from G2) |
| N1 | paired hybrid, alpha 0 (from G2) |
| R-a | paired live cloze target, alpha 1, MASK positions only |
| R-b | R-a plus the 16-sample cache |
| N2 | label-keyed shuffle, eligibility-matched |
| N3-uH, N3-uO, N3-gH, N3-gO | smoothing nulls; run after R-a and R-b to calibrate epsilon |
| N4 | A1 temporal cache, same positions |

**Cost:** 10 runs, about 136 A100-minutes in total. That is about 14 minutes per hybrid run, including the 2.5-minute compile and the eval, and about 10 minutes for B.

## Odds (guesses, stated plainly)

- **Ceiling:** about 0.05 nats/word in track with an oracle-quality teacher, and 0.1 at most.
- **Median**, best arm minus the binding comparator: **about -0.01**. The mechanism's own increment over N1 is +0.005 for R-a and +0.01 for R-b. MNTP's handicap against the causal base is the likely net-negative term.
- **P(pass K1-K3):**
  - about 0.3 x 0.12 if N1 is within 0.01 of B;
  - plus about 0.7 x 0.02 otherwise;
  - **about 0.04 in total.**
- **P(at least 0.05):** under 0.01.

A pass would be the first measurement of in-place cloze-to-causal distillation in exposure-capped pretraining.

## Implementation plan (harness; about 1.5-2 days of agent work; build M)

1. **`levers.mntp_corrupt(..., return_masks=False)`** also returns the MASK-replaced boolean. The default path is bit-identical.
2. **`babylm/moonshot/rb_teacher.py`, about 200 LOC.**
   - Pairing builder: the sampler draws n_micro x n_c windows and duplicates the first k_m of each micro-batch as MNTP rows. The exposure counter counts the duplicates.
   - The MNTP forward goes through the split API (`hidden()`/`logits()` under `BidirectionalAttention`), with CE reproducing `model(xm, ym)`.
   - Detached pi at MASK positions, aligned so that output j-1 predicts x_j.
   - Full-vocabulary KL with per-position weighting.
   - Flags:
     - `--rb_mode {off, pair, pair_cache, shuffle, smooth_uniform, smooth_unigram, temporal}`
     - `--rb_alpha`
     - `--rb_start_epoch`
     - `--rb_cache_samples 16`
     - `--rb_match {H, off}`
     - `--rb_eps`
   - Diagnostics as listed above.
3. **`babylm/harness/train.py`, about 70 LOC.** A branch inside the MNTP-hybrid step that mirrors the existing `--kd` path, with student logits from `KD.StudentLogits`. EMA is refused when `--rb_mode` is not off.
4. **Cache and N4, about 100 LOC.** Reuse A1's position-keyed cache from `self_distill.py` (PR #55). The sampled-histogram targets follow Anshumann 2025.
5. **N2 bank and the N3 epsilon calibration, about 80 LOC.**
6. **`babylm/moonshot/rb_toy.py`, about 200 LOC**, for G0.
7. **Tests, about 180 LOC.**
   - `--rb_mode off` is bit-identical to the base+MNTP path, and pair at alpha 0 equals an explicit paired reference.
   - The split-API MNTP loss equals `model(xm, ym)`.
   - The j-1 to x_j alignment.
   - Cache keys are exact under stream offsets.
   - The sampled cross-entropy is unbiased (Monte Carlo).
   - Shuffle eligibility is matched between arm and twin.
   - The exposure assertion.
   - On the toy HMM, the mean of the exact pi equals the causal posterior, and the masked view's bias is small.
8. **Job files:** `babylm/configs/jobs_moonshot_rb_{1M,S}.json`. Outputs go to `shared/babylm/attacks/rb/<run>/` with a CSV and PNG of the curves.

## Questions for the critic (before any code or GPU run)

- **Q1.** In the paired design, is reading the detached logits of an MNTP training row, as a target for the same-step causal row of the same window, an extra view? Our claim is that it is not: both rows are already counted, and the ledger equals base+MNTP's 99.54M.
- **Q2.** The binding comparator is max(B, N1), with MNTP owned by the attack as a confound arm (18:43Z(c)).
- **Q3.** The hybrid-family GLUE pooling at promotion.
- **Q4.** In G1, MNTP arms labelled as attack-owned proxies.

## Overlap

- The card is target-side, like A1 and label smoothing. A combination arm needs its own card, and gains cannot be summed.
- **The rescue route, parked as its own card:** an out-of-fold bidirectional teacher made from A4b's cross-fitted half-data hybrids. It removes in-sample memorisation, and its exposure is shared with A4b's ledger.
