<!-- moonshot design round W3; status alive; ceiling 0.1 nats/word; expected 0; P(pass screen) 0.04; screen GPU 60 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card: out-of-fold floor gate, revised after three skeptics (2026-10-10)

Status: card only, not yet sent to the critic. No code and no runs. **Verdicts folded:** prior-art said wounded, rules said wounded, mechanism said dead. That is one dead out of three, and every rule problem has a workable fix, so the card stays alive in a much narrower form. **Plain headline:** the most likely result is a kill at Stage 0, before any gated training (0 to 10 A100-minutes). Expected gain is about zero. This is a cheap test of whether per-token over-fitting can be detected at all on our base. It is not a 10x lever: at its ceiling it is worth about 1.3x data.

**Known idea, not new.**
- **Dropped arm b is published verbatim.** Two half-models each score the half they did not see. This is RHO-LOSS's no-holdout protocol (Mindermann+ ICML 2022, Sec. 4, App. Table 3).
- **Held-out loss as a per-sample floor is AdaFlood** (Bae+ 2023, arXiv 2311.02891: flood levels from held-out-fold auxiliary models). Also iFlood (Xie+ 2022) and flooding (Ishida+ ICML 2020). G1 at gamma = 0 is per-token AdaFlood/iFlood, with the gradient zeroed below the floor rather than reversed.
- **Self-scored token selection in from-scratch LM pretraining is ESLM** (Bal, Cevher, Muehlebach 2025, arXiv 2505.19893). Rho-1 (Lin+ NeurIPS 2024) also does it, and in Sec. 3.4 its reference is trained on the same corpus.
- **Measured LM evidence for this family is null.**
  - No Train No Gain (Kaddour+ NeurIPS 2023, arXiv 2307.06440): RHO-LOSS and selective backprop in BERT/T5 pretraining do not beat baseline validation loss at a fixed budget. With selection made free, GLUE drops 77.79 to 77.16.
  - ESLM Table 1 (124M, matched FLOPs): Rho-1 from scratch 0.3853 vs CLM 0.3852.
- **Related results that point against a gain:**
  - InfoBatch (Qin+ ICLR 2024) prunes low-loss samples but has to switch pruning off for the last 12.5% of epochs to stay lossless. This card gates in exactly those late epochs.
  - Sorscher+ 2022: with scarce data, keep the easy examples.
  - Ankner+ 2024: perplexity pruning fades past about 4 repeats.
  - MAT/XRM (Bayat+ 2024, arXiv 2412.07684): held-out half-models used against memorisation lower average accuracy.
  - Loss truncation (Kang & Hashimoto 2020) and focal/anti-focal loss (Lin+ 2017; Raunak+ 2020) are cited from memory and not verified.
  - Counterfactual memorisation (Zhang+ 2021, arXiv 2112.12938) defines memorisation with held-out models.
- **N1 is goldfish loss in spirit** (Hans+ 2024, arXiv 2406.10209: a consistent pseudo-random token drop that removes verbatim memorisation). So N1 may itself act on memorisation, and the arm has to beat it.
- **BabyLM and in-house record:**
  - ACLM surprisal selection: NLP 50.3 vs 57.3 (2025.babylm-main.34).
  - Batch-wise convergent pretraining: below baseline (2025.babylm-main.36).
  - AMLM: +1.0, inside noise.
  - In-house SEL: -0.003 / -0.011 (docs/sprint_1_prior_lessons.md:40).
  - Loss-weighted tokens and RHO/Rho-1 were killed in discovery session 2 (docs/discovery_session_2.md:268-269).
  - A5 (late-epoch rare-word focus) was accepted at 02:26Z.
- **TII 2607.04969** is a preliminary workshop paper with no numbers in its abstract, so the draft's "epoch-level" description is unverified.

**Label: known idea.** As far as we know, two parts are unpublished: using the token's own first-visit (prequential) loss as the floor, and calibrating that floor against a held-out quantile per frequency band. The setting is exposure-capped, multi-epoch, from-scratch, 10M words. "Multi-epoch" by itself is not new: RHO-LOSS, AdaFlood, iFlood and InfoBatch are all multi-epoch.

**Correction for the shared docs (the owner applies it; this card does not edit the repo).** The claim "RHO-LOSS in LM pretraining never measured" in refs/prior_art.md (RHO-LOSS row and section C) and in context_v1.md section 4 is false. No Train No Gain 2023 and ESLM 2025 both measured it.

## What changed from the draft (skeptic fixes adopted)
1. **Arm b dropped.** It gives up one student epoch: the last epoch's cooldown is worth 0.141 nats/word on the base, and a 9-epoch schedule is estimated to cost 0.02-0.05, more than the kill threshold. Its 5M-word, 1-epoch reference also treats rare words as unlearnable, and its 10M words of no-grad cross-scoring raise an open R0 question. Any out-of-fold-half follow-up needs its own card and an R0 ruling on scoring passes.
2. **No DEV read during training.** The draft's shift DEVsub(e-1) leaked the kill metric into the training signal. It is replaced by a constant table, Q[band][epoch] in nats/token, computed before the run from the frozen base and logged as one DEV tuning trial.
3. **Visit-time drift removed.** The first-visit loss is corrected for the epoch-1 learning curve. Positions first visited in the first 10% of epoch-1 steps are never gated. A visit-time correlation kill is registered.
   - The size of the drift on the base's train_log, which I verified: step-1 loss 9.70. In 49% / 26% / 12% of epoch-1 steps the batch loss is more than 0.3 / 0.6 / 1.0 nats/token above the end-of-epoch-1 level.
   - The draft's global shift therefore mostly encoded epoch-1 order. On DEV, where nothing is memorised, it fired on 58% of first-decile words and 30% of last-decile words.
4. **Per-band quantile calibration.** The draft used a global mean shift with margin mu. The new floor is calibrated so that, on the frozen base, the gate would fire on exactly q = 10% of held-out DEV-sub tokens in each band. Excess firing on train tokens over 10% is then the memorisation signal.
5. **Bands 1-63 exempt.** The draft's gate fired on 86% of 1-15 train words, which is a rare-word gradient cut and works against data's rare tilt of 1.83. It is now gated only on words seen 64+ times, the card's own target slice.
6. **One registered arm (gamma = 0, q = 0.10).** The draft had 8 single-seed arms, which gave roughly a 10% chance of a noise pass.
7. **Two count-matched nulls.** The draft's global shuffle did not match the gate's band or visit-time structure, and nothing checked whether the reference adds anything.
8. **Mechanism kills replaced.** The draft's "dev-minus-train gap shrinks" check passes by construction: TRAIN-sub is part of train_10M, so gating low-loss train tokens raises train loss.
9. **False claims removed.**
   - "Dev stalls after epoch 7": that was measured on published GPT-BERT, not our base.
   - "Replaces 4-epoch early stopping": E* = 10 (critic 02:53Z).
   - "A fixed denominator keeps the effective LR": false under Muon + AdamW. The surviving gradient is renormalised, which effectively raises the LR on ungated tokens. The count-matched nulls are the LR-matched control.
10. **Odds cut.** Ceiling 0.35 to 0.10, median 0.02 to 0.00, P(pass) 0.12 to 0.04.

## Mechanism (registered arm G1)
**Keys.** Each target token is keyed by p, its absolute position in the train_10M token stream (window start s, targets s+1..s+T). This is A1's keying, which holds under EpochSampler's per-epoch random offset.

**Band.** Each target is put in a band, b(p), by the train_10M count of the whitespace word containing it (data/babylm/analysis/wordcounts_train_10M.pkl). The bands are:
- 1-63: exempt;
- 64-8k: gated;
- 8k+: gated.

Tokens that are not inside a word (BOS, newline) are exempt.

**Epoch 1 (no weights; the base path).** Record two numbers per position from the training forward that already runs:
- l1(p): the per-token CE at the first visit;
- s1(p): the step of that visit.

**Reference (computed once at the epoch 1 to 2 boundary).**
- Let mbar(s) be the batch-mean training CE at step s, smoothed with an 11-step centred mean.
- Let mbar_end be the mean over the last 5% of epoch-1 steps.
- The reference is r(p) = l1(p) - [mbar(s1(p)) - mbar_end].
- Exempt: positions with s1(p) in the first 10% of epoch-1 steps (the near-random model), and positions epoch 1 never visited (offset edges, at most 2T tokens).

**Gate, epochs e >= 2.** Gate position p in band b at epoch e if its loss has fallen more than the band allows:
- CE_e(p) is its per-token CE at its epoch-e visit, computed by the training forward before the update.
- The gate fires if r(p) - CE_e(p) > Q[b][e].
- Weight w = gamma = 0 if the gate fires, else 1.
- Loss = sum(w * CE) / n_targets.

**Q table: 2 bands x 9 epochs = 18 constants in nats/token, registered before the run.**
- Q[b][e] is the 90th percentile, over DEV-sub tokens t in band b, of r_dev(t) - CEbar_e(t), measured on the frozen base.
- CEbar_e(t) is the mean of the base's per-token DEV-sub CE at the two milestones that bracket epoch e.
- r_dev(t) is the base's CE at milestone (k-1)M, minus [DEVsub_tok((k-1)M) - DEVsub_tok(10M)]. Here k is a pseudo first-visit decile drawn uniformly from 2..10 with seed 0.

On the base trajectory the gate therefore fires on 10% of unseen-data tokens in each band by construction. Any train firing above 10% is fit beyond what the band's held-out learning explains.

**Logged per step and band:** gate counts (gate_log.npz). **Logged per epoch:**
- gate rate per band;
- corr(fire, s1);
- share of gradient mass removed, sum over gated tokens of (1 - p_target) divided by the same sum over all tokens;
- gradient norm before Muon and Adam update RMS;
- the usual DEV-sub / TRAIN-sub curves.

**Cost.** One gather, one subtract and one compare per token. FLOPs 1.0x. Memory: 16M x (fp16 + int32 + int8) is about 110 MB.

## Why (measured slice), and what the bet now is
**Target slice.** Familiar words in new contexts, plus overconfidence:
- On published GPT-BERT, the gap on words seen 64-8k times is 0.36 and on 8k+ it is 0.22 nats/word. Overconfidence (loss minus own entropy) is 0.57 vs 0.39.
- On our base, DEV-sub against the 100M size-S twin, the gap is 0.701. Of that, 64-8k carries 0.307 and 8k+ carries 0.159 (A5 card).

**Signal on our base** (base_S_wd0p4_lr0p5_r1_ve0_tied @ ref-0065, curves.csv):
- Over epochs 1 to 10, TRAIN-sub falls 3.263 to 2.418 nats/token while DEV-sub falls 3.659 to 3.177. That leaves 0.36 nats/token of train-only fit.
- Per word, e1 to e10, train vs DEV: 64-8k 1.30 vs 0.65; 8k+ 0.58 vs 0.44 (mechanism skeptic, from the per-word dumps).

**The bet.** That excess sits on tokens the gate can identify one by one, and the gradient on them is harmful: it sharpens seen contexts and pushes down plausible alternatives.

**Why it is not already in the base.** WD 0.4, the cooldown and E are global. They cannot tell an over-fitted token from a still-learning one.

**Honest counterpoint (this is why the median is zero):**
- DEV improves in every band through epoch 10. Over e7-e10 the totals are 5.366 to 5.121 per word: 64-8k +0.22, 8k+ +0.16, with 0.141 of it in the last epoch's cooldown. So nothing on DEV is getting worse for the gate to win back.
- Per-token improvement on unseen text varies enormously (e1 to e10 SD of 3.07 nats/word), so a per-token signal of excess fit may be mostly noise.
- The draft's own falsification condition, "fires on bands whose DEV loss is still improving", is already met on this base. That withdraws the draft's premise. The card survives only as the narrower claim that the gate can find per-token excess fit while every band still improves, and Stage 0 measures exactly that, before any gated training.

## Null twins (same code path, exposure, gamma, seed and data order; gate count matched exactly per step and per band)
- **N1, stratified-permuted reference (goldfish-style).**
  - r(p) is permuted within cells of band x s1-decile, using a seeded permutation fixed at the epoch-1 boundary.
  - At each step and in each band, it gates exactly the arm's logged count k(s,b): the tokens with the largest r_perm - CE.
  - It keeps the gate's band and visit-time structure and its consistency across epochs, but removes the per-position reference information.
- **N2, reference-free easy-token drop** (Rho-1-from-scratch / InfoBatch / loss-truncation style).
  - At each step and in each band, it gates the k(s,b) lowest-CE tokens.
  - It tests whether the first-visit reference adds anything over "low loss now".

Both nulls renormalise through Muon/Adam in the same way as the arm, so they are also the LR-matched control. They read the arm's gate_log.npz, so they run after G1. If a null itself beats the base by 2x the floor, that is reported as a finding (a known idea), not promoted.

## Kill rule (registered before any run)
**Stage 0: falsification before any gated training.** Each condition below is a kill:
- **K0a, no memorisation signal.** Train firing rate at epoch 10 below 20% in both 64-8k and 8k+. That means precision 1 - q/rate is below 0.5 in both bands.
- **K0b, visit-time artefact.** |corr(fire, s1)| > 0.2 on the train sample at epoch 5 or 10.
- **K0c, nothing to act on.** Gradient-mass share removed below 5% at epoch 10, or the 64+ train firing rate at epoch 10 outside [10%, 60%].

**Screen (only if Stage 0 passes).**
- Setup: size S, train_10M x 10, the round-1 winner's recipe, seed 0.
  - Batch 1: G1, plus the base re-run at --gate_mode none.
  - Batch 2: N1 and N2, both count-matched to G1, plus a repeat of the base.
  - All on the same GPU type. If the two base runs differ by more than 0.005 nats/word, the screen is void and is re-run as one batch.
- Metric: full-DEV nats/word from dev_bpw.py (collapse protocol), final checkpoint chck_100M, no checkpoint picking.
- **Kill if any of these holds:**
  - G1's gain over the base is below 2x the seed floor;
  - G1 is not ahead of N1 by 2x the floor;
  - G1 is not ahead of N2 by 2x the floor;
  - fade rule (as A2): the DEV-sub gain at the end is more than 0.5 sigma below the gain 3 epochs earlier;
  - BLiMP-fast drops by 2 sigma or more.
- **Mechanism kills (screen):**
  - At least 60% of G1's full-DEV gain over N1 must fall in the 64-8k and 8k+ bands. This is the same-instrument harness per-word split, A5's band_split.py.
  - The ungated 1-63 bands must not lose more than 0.02 nats/word against the base.
  - K0b and K0c must also hold in the run's own gate log.
- **Reported only:** TRAIN-sub rise and the dev-minus-train gap (both expected by construction); the novelty and rare tilts. A data-like gain needs a novelty tilt of 0.8 or more. A gain with a tilt below 0.8 is read as epoch-like.

**Floor.** Not yet registered. Provisional: winner seeds 0 and 1 score 5.0367 and 5.0255 on dev_bpw, so SD is about 0.008 and 2x the floor is about 0.016 nats/word. Until the critic registers the floor, results are numbers only, with no kill or promote.

**Promotion.**
- One arm only (G1). The binding twin is the better of N1 and N2 on the screen, picked on DEV before seeds 1-2.
- 3 seeds, size M, official eval under R0-R6.
- Comparator: the base at E* = 10. If a shorter screening base is ever adopted, it is the better of base@E_s and base@10 (critic 02:53Z).
- No FLOP-matched reference is needed (1.0x).

**Provisional base.** G1 acts as a regulariser. Any later change to the base's WD, EMA, MNTP or dropout forces a re-screen. Until then a kill reads "killed (provisional base)".

## Odds (guesses)
**Ceiling: 0.10 nats/word, about 1.3x data by log-linear interpolation.**
- The gate adds no contexts and no targets. It can only stop gradient that over-fits beyond what WD 0.4 and the cooldown already prevent.
- A proxy for that bound: own-snapshot mix +0.048 plus temperature +0.013 (both measured on published GPT-BERT), plus part of the variance channel (C3 g(8), registered at 0.10-0.25, which a single run cannot capture in full).
- No path to never-seen words or new-context bias.

**Median: 0.00 (range -0.02 to +0.01).**
- Same-family LM results are null: No Train No Gain, ESLM's Rho-1-from-scratch, in-house SEL.
- The scarce-data regime favours keeping easy tokens.
- InfoBatch needs late pruning switched off.

**P(pass screen): 0.04.** That is P(Stage 0 survives), about 0.25, times P(screen pass given survival), about 0.15. The second figure includes about a 0.03 chance of a noise pass with one arm and two nulls.
- On the mechanism skeptic's proxy (global shift), precision is 0.26 in 64-8k and 0.13 in 8k+.
- A normal approximation under per-band quantile calibration gives about 0.4-0.5 in 64-8k and about 0.2 in 8k+. So K0a probably fires.

**Skeptics' adjusted odds:**

| Skeptic | Ceiling | Expected | P(pass) |
|---|---|---|---|
| Prior-art | 0.15 | 0.01 | 0.06 |
| Rules | 0.18 | 0.007 | 0.07 |
| Mechanism | 0.10 | -0.01 | 0.04 |

## R0 exposure ledger
| run | words fed (pipeline counter) | other trained models | no-grad passes | total |
|---|---|---|---|---|
| G1 | 10 x 9.95M = 99.5M (99,541,960 on the base) | none | none (the store reads the training forward) | 99.5M |
| N1, N2 | 99.5M | none | none | 99.5M |
| base (x2) | 99.5M | none | none | 99.5M |
| Stage-0b logging run | 99.5M (a base re-run, diagnostic) | - | - | not part of any arm's pipeline |

**Open question for the critic.** G1 takes only the 18 Q constants from the Stage-0b run. I treat them as hyperparameters chosen from the base's trajectory: one DEV tuning trial under R3, with the same status as A5's start epoch and focus set, which were chosen from the base's dumps and accepted at 02:26Z. They are not a model in the pipeline.

There are no generators and no synthetic text. Arm b is dropped, so the question of whether no-grad cross-scoring counts as exposure does not come up in this screen.

## FLOPs and R4.4
- G1 and the nulls are 1.0x the base's training FLOPs: no added matmuls. N2 adds a per-band top-k per step, which is negligible.
- Wall-clock and tokens/s are logged against the base arm. R4.4 does not trigger (ruling 01:20Z).
- The exported model and the official pipeline path are unchanged.

## Fastest falsification (cheapest first)
**Stage 0a: CPU only, zero GPU, a kill-only proxy labelled "per-word proxy".**
- Inputs: the base's per-word dumps (shared/babylm/base_r1/base_S_wd0p4_lr0p5_r1_ve0_tied/pw/pw_{dev,train}_{1..10,20..100}M.npy).
- Give each train_sub word its epoch-1 visit decile: replay EpochSampler(N, T, data_seed 0).plan(0) and map the documents in babylm/data/subsets/train_sub.json to stream positions.
- Use the Q definitions above, per word, and apply K0a-K0c.
- About an hour of CPU.

**Stage 0b: exact, about 10 A100-minutes. Runs only if 0a passes.**
- Re-run the base at --gate_mode log. Training is unchanged; it records l1 and s1 for all positions, CE at every visit for a seeded sample of 1M train positions, and per-token DEV-sub dumps.
- `python -m babylm.harness.repeat_gate --calibrate <run>` writes the Q table and the K0a-K0c statistics.
- K0 decides whether the screen runs at all. If Stage 0 kills, the cost is 0-10 A100-minutes.

## Screen (size S, seed 0, all 1 seed; about 9.5 A100-minutes per run)
| run | mode | note |
|---|---|---|
| S0b | log | base path plus logging; produces Q and the K0 statistics |
| G1 | self, gamma 0, q 0.10 | the only arm |
| base | none | batch 1 |
| N1 | null_perm, matched to G1 | batch 2 |
| N2 | null_easy, matched to G1 | batch 2 |
| base-repeat | none | batch 2, determinism check |

Outputs go to shared/babylm/attacks/OOF/<run>/ with curves CSV and PNG, gate.csv and gate_log.npz. Every run counts in the trial log. FINAL is not touched.

## Implementation plan (harness; about one day)
- **babylm/harness/repeat_gate.py**
  - `GateStore(n_tokens)` holds l1 (fp16[N]), s1 (int32[N], -1 = unvisited), band (int8[N], built from word_init and the wordcounts pkl) and mbar (float32[steps per epoch]). `state_dict()` saves it to gate_store.pt for resume.
  - `target_positions(starts, T)`: reuse A1's key helper if it is merged.
  - `record(ce, pos, step)`: epoch 0 only, writes where s1 == -1, reading before writing.
  - `finalize_epoch1()`: applies the de-trend and the exemptions, and builds N1's stratified permutation.
  - `weights(ce, pos, epoch, mode, gamma, qtab, match_counts)`: returns w and the per-band counts.
  - CLI `--calibrate <log run> --q 0.10`: writes `babylm/configs/gate_q_<base>.json` (Q, q, source run, sha) and prints the K0 verdicts.
  - CLI `--stage0a`: the per-word CPU proxy.
- **train.py changes**
  - New flags: `--gate_mode {none,log,self,null_perm,null_easy}`, `--gate_gamma`, `--gate_q_table`, `--gate_exempt_frac 0.1`, `--gate_match <arm run dir>`, `--gate_sample 1000000`.
  - Causal rows only; assert mntp_frac == 0, as in the base.
  - In the gated modes, `loss, _ = model(x, y)` is replaced by `z = raw_model.logits(raw_model.hidden(x))` followed by `F.cross_entropy(reduction="none")`. This is the same split API the KD path already uses under compile. Then loss = (w*ce).sum()/n_targets.
  - `none` keeps the untouched path, bit-identical.
  - Writes gate.csv and gate_log.npz.
- **Tests (babylm/tests/test_repeat_gate.py)**
  - `none` is bit-identical to the port on CPU.
  - `self` with Q = +inf reproduces the base losses.
  - Every position is recorded exactly once in epoch 1 under the random offsets.
  - The null counts match the arm exactly per step and per band.
  - Exempt positions are never gated.
  - Pause and resume is bit-exact with the store.
  - Calibration returns the right quantile on synthetic input.
- **Job list:** babylm/configs/make_gate_jobs.py for S0b, batch 1 and batch 2.

## Stacking and overlaps
- **A1** (changes the target on repeats) and this card (changes whether a repeat gets a gradient) overlap. A combination needs its own card, and the gains cannot be summed.
- **A5:** because Muon and Adam renormalise the surviving gradient, G1 shifts step share toward the ungated 1-63 tokens. That is A5's lever. The count-matched nulls control for it, but a combination with A5 needs its own card.
- **A2** (input noise) is orthogonal in mechanism, but both act as regularisers (critic's 00:12Z note).
- **Candidate 2:** if it survives, its neighbour loss is a natural zero-epoch out-of-fold reference and would replace r(p) in a follow-up card.
