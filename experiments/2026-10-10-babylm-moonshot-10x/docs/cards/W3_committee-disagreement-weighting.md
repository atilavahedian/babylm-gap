<!-- moonshot design round W3; status alive; ceiling 0.1 nats/word; expected 0.005; P(pass screen) 0.04; screen GPU 95 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card: committee-disagreement token weighting on a legal K=2 pair (`committee-disagreement-weighting`; card v1 after three skeptics, 2026-10-10)

**Status: alive, gated, low priority.**
- **Skeptic verdicts:** prior-art wounded, rules wounded, mechanism wounded. No skeptic found it dead.
- **The one fatal rules problem has a workable fix, adopted here.** As written, the population path either breaks R0 or never switches on. It is replaced by a legal, non-shared K=2 pair at 5 member-epochs each (critic ref-0110 Q1).
- **What survives is much narrower than the candidate:**
  - **One registered substrate.** The K=4 path and the "or base + A1 cache" option are withdrawn.
  - **The temporal fallback, KL(q_prev || q_now), is dropped.** It is Active Bias (2017). It weights positions that are mid-memorisation upward. A1's cache holds an EMA of the top 16 tokens plus a tail bin, not q_prev. Anyone who wants it needs a separate card with its own sign test.
  - **The mechanism is restated honestly.** Memorised positions already carry about 0.5% of the CE gradient, so the weight frees almost nothing from them. What it does is move gradient among positions that are not yet memorised.
  - **Four quantile-matched nulls plus a sign-flip check** replace the candidate's two nulls. A second arm targets the part of disagreement that entropy does not explain.
  - **Revised numbers.** The label drops to "known idea". The ceiling goes from 0.3 to 0.1, and P(pass) from 0.25 to 0.04.
  - **Three gates before any 10M screen slot:** a premise gate on CPU (F0, no GPU), a substrate gate (G0, about 9 A100-minutes), and a mini-ruler gate (F1, about 17 A100-minutes).
- **Work so far:** card only, plus one zero-cost premise proxy run while writing it (see "Premise proxy"). No code, no GPU.

## Known idea, not new
**Label: known idea.** The regime is untested, but this has not been verified by search. The regime is committee-disagreement (BALD) weights on token CE inside a legal population, in exposure-capped multi-epoch LM pretraining, switched on late. The candidate's "known ideas, new combination" was wrong.

**Same mechanism, already measured**
- **Wan et al., EMNLP 2020** (Self-Paced Learning for NMT, arXiv 2010.04505) do the same thing at token level.
  - Each target token's loss is weighted by (1 - Var over 5 MC-dropout members of P(y_j))^k.
  - They tested both signs. Weighting confident tokens up (k = 2) gave +0.6 to +1.2 BLEU over 5 runs.
  - "Unconfidence-first strategy (k < 0) performs worse than baseline." That is the candidate's direction.
- **The signal is BALD.** u = (1/K) sum_k KL(q_k || q_bar) = H(q_bar) - mean_k H(q_k), the mutual information of Houlsby et al. 2011 (1112.5745).
  - RHO-LOSS App. G (Mindermann 2022, 2206.07137) ran BALD, entropy and conditional entropy as online batch-selection baselines. They sped up MNIST but not CIFAR-10.
  - The authors note that high loss and high uncertainty are also caused by noisy or ambiguous labels.

**Updating on disagreement (noisy-label literature)**
- Decoupling (Malach & Shalev-Shwartz 2017, 1706.02613) updates only where two networks disagree.
- Co-teaching (1804.06872) points out that the disagreement area holds noisy labels.
- Co-teaching+ (1901.04215) therefore needs disagreement and small loss together.
- In an LM, the "noisy labels" are next tokens that are unpredictable in principle.

**Uncertainty-weighted token CE in LM pretraining**
- MiLe (NAACL Findings 2024, 2310.19531) weights tokens by predictive entropy at 468M-6.7B on the Pile.
- 6.7B Pile validation PPL: 5.473 with no weighting, 5.467 at gamma 0.5, 5.492 at gamma 1, 5.608 at gamma 2, 6.317 at gamma 5.
- That is at most about 0.001 nats/token, and stronger weighting hurts perplexity.

**Learnability selection in LM pretraining (corrects the prior-art map)**
- Kaddour et al., NeurIPS 2023 ("No Train No Gain", 2307.06440), ran RHO-LOSS and selective backprop in BERT pretraining.
- At matched compute both were worse: C4 validation loss 2.37 and 2.44 against 2.20 at 12h.
- With selection made free: 2.19 vs 2.21 (about 0.02 nats/token), but GLUE got worse (77.16 vs 77.79).
- **This corrects prior-art map row 87 and context section 4**, which say "RHO-LOSS never measured in LM pretraining".

**Loss-based weighting is a published positive method, so the loss-only null is not a strawman**
- Sow et al., ICLR 2025 (2502.06733): helps at 1.4B and 7B, mostly by down-weighting low-loss samples.
- Focal loss (Lin 2017).
- Rho-1 (Lin 2024, 2404.07965). Its self-reference entropy variant gives "a similar improvement".
- EL2N/GraNd (Paul 2021) and selective backprop (Jiang 2019).

**Temporal fallback (dropped):** Active Bias (Chang et al., NeurIPS 2017, 1704.07433), dataset-cartography variability (Swayamdipta 2020, 2009.10795) and forgetting events (Toneva 2019).

**Theory**
- Context-only weights leave the per-context ERM minimiser unchanged. They act only through the training trajectory and how capacity is shared.
- Importance-weight effects fade as over-parameterised networks keep training, and L2 restores them only partly (Byrd & Lipton, ICML 2019, 1812.03372).

**BabyLM record**
- Curricula driven by model uncertainty or surprisal were null or below baseline every year: 2023.conll-babylm.22, 2024.conll-babylm.21, 2025.babylm-main.34.
- Train-until-low-loss was below the baselines (2025.babylm-main.36).
- Peer-population KL (mutual learning) was near chance (2024.conll-babylm.17).
- AMLM adaptive masking: +1.0 NLP, 1 seed (2025.babylm-main.31).
- No 2023-2025 title covers disagreement-weighted token loss.

**In house:** A5 (late-epoch rare-word loss focusing, accepted 02:26Z) is the same family and the nearest lever in flight.

**Search caveat**
- Both skeptics found the shared web-search budget already used.
- Fetched from primary sources: Wan, RHO App. G, Kaddour, MiLe, Co-teaching and the BabyLM proceedings indexes.
- Cited from memory: Active Bias, Decoupling and Co-teaching+.
- The label can only move further toward known.

## Mechanism (as amended)
**Substrate P2 (the only one registered)**
- **Members:** two S models, non-shared, with separate optimizers.
  - Recipe: the round-1 winner, base_S_wd0p4_lr0p5_r1_ve0_tied @ ref-0065.
  - Init seeds 0 and 1.
- **Data:** one shared data loader, so both members see the same windows in the same order (data seed 0).
- **Schedule:** 5 epochs over train_10M per member, with the recipe's full LR schedule fitted to 5 epochs (cooldown 50%).
- **Deployed predictor:** the token-level mean of the two members' probabilities.
- **R0:** under ref-0110 Q1 each member is charged 100M/K = 50M words.

**Signal per scored causal position t** (detached, fp32, computed in chunks)
- q_k is the softmax of member k's softcapped logits from its own training forward.
- q_bar = (q_1 + q_2)/2.
- u_t = H(q_bar) - (H(q_1) + H(q_2))/2. This equals (1/2) sum_k KL(q_k || q_bar), the K = 2 JSD, i.e. BALD.
- For the nulls and the logs, the same step also computes:
  - Hbar_t = H(q_bar), the total entropy;
  - A_t = mean_k H(q_k), the aleatoric part;
  - C_t = mean_k CE_k,t, the loss.

**Weights**
- w_t = clip((u_t / u_mean)^beta, 0.25, 3), where u_mean is the mean over the micro-batch's scored positions.
- Weights are renormalised to mean 1, with clip-and-renormalise applied twice. The final range is logged.
- **beta = 0.5 only.** MiLe shows stronger weighting hurts perplexity, and a beta grid would add forking paths.
- **Loss.** The same w_t multiplies both members' CE: L_k = sum_t w_t CE_k,t / N.
- **Onset.** Weighting is off before member-epoch 3, i.e. for each member's first 20M words. Until then w = 1, and the run is bit-identical to the unweighted pair.

**Arms.** Each arm is a ranking rule. Every arm and every null hands out the same per-micro-batch set of weight values, so mean, variance and ESS are identical by construction. Only which position gets which weight differs.
- **CW-a (raw):** weights assigned by the rank of u. This is the candidate's mechanism.
- **CW-b (disagreement net of entropy):** rank of u within 16 equal-count Hbar strata of the micro-batch, i.e. each position's within-stratum percentile.
  - It puts weight where the members disagree more than other positions with the same total uncertainty.
  - This is the "reducible-only" arm the rules skeptic asked for.

**What the weight actually does** (mechanism skeptic's CPU probe on a round-1 S model)
- Train positions with p(y) > 0.9 are 19.8% of tokens and carry 0.5% of the CE gradient.
- The set of positions sent to the floor weight (21.5% of tokens) carries 4.7%.
- Weighting moves gradient among positions that are not yet memorised:
  - u-deciles 2-5 fall from 46% to 24% of the gradient;
  - the top three deciles rise from 37.5% to 63%. These are high-entropy content slots with p(y) about 0.18-0.22 and H about 3.5 nats.
- **Registered prediction:** dev-minus-train shrinks relative to the unweighted pair, with DEV falling. A gap that shrinks because TRAIN-sub rises does not count as the mechanism working.

**MNTP:** the base has none (mntp_frac 0.0). If an MNTP base is later frozen, causal and MNTP rows are normalised separately, because their u scales differ. That change forces a re-screen.

## Why: which measured slice, and what our base says
**Target slice:** generalisation.
- 0.61 of the 0.875 nats/word gap appears only on new text.
- The 64-8k band holds 0.36.

**Where the candidate's premise came from:** published GPT-BERT.
- Epochs 7-10 buy 6% of the dev gain.
- Their novelty tilt is 0.43 [0.30, 0.60].
- Dev-minus-train grows from 0.51 to 1.11.

**On our base the premise is only half true.** Round-1 winner, by epoch:

| | e3 | e5 | e7 | e10 |
|---|---|---|---|---|
| DEV-sub | 5.538 | 5.522 | 5.366 | 5.121 |
| TRAIN-sub | 4.72 | | | 3.83 |
| dev-minus-train | 0.82 | | 0.97 | 1.29 |

- DEV improves to the end: 0.42 nats/word from e3 to e10. A third of the e7-e10 gain comes in the last epoch's LR anneal.
- A5's premise check found that frequent words still improve in epochs 7-10, providing 43% of the late gain.
- So the gap grows, but learning does not stall.

**Why the base does not already do this**
- WD, tying and the anneal treat all positions alike.
- Nothing in the base steers gradient toward where an independent learner disagrees.
- Inside R0, the pair can only be built at 5 member-epochs, so the whole card depends on the substrate gate G0.

## Premise proxy (zero cost; run while writing this card; a label, not a gate result)
This is the prior-art skeptic's premise test, run on files already on disk.

**Setup**
- **Committee:** the winner's two seeds, s0 and s1. These are 10-epoch base runs, so as a pair they are out of track.
- **Signal:** per-whitespace-word loss disagreement d = |CE_s0 - CE_s1| at the 30M milestone (epoch 3), read from the harness per-word dumps.
  - DEV-sub has 993,244 words; TRAIN-sub has 1,004,072.
  - This is a word-level loss disagreement, not the token-level JSD.
- **Statistic:** the later drop in member-mean loss from 30M to 100M, on TRAIN-sub and DEV-sub separately. The ratio is DEV drop / TRAIN drop. A lower ratio means the later training looks more like memorisation.

| ranking | TRAIN drop | DEV drop | ratio |
|---|---|---|---|
| all words | 0.884 | 0.423 | 0.48 |
| top-d decile (raw) | 2.55 | 1.03 | 0.41 |
| bottom-d decile (raw) | 0.31 | 0.18 | 0.59 |
| top-CE decile (loss only) | 3.32 | 1.13 | 0.34 |
| top-d decile within 20 CE strata | 1.41 | 0.71 | 0.505 |
| bottom-d decile within 20 CE strata | 0.74 | 0.34 | 0.465 |

Measured from 50M (epoch 5) instead, the ratios are: all words 0.50; raw top-d 0.45; CE-matched top 0.546 vs bottom 0.472.

**Reading**
- **Raw ranking: fails.** The arm's ranking gives its biggest weights to the words where later training looks most like memorisation (0.41 vs 0.48). That fails the prior-art skeptic's pre-registered premise rule, as the noisy-label literature predicts. Ranking by loss alone is worse still (0.34).
- **At matched loss: a small positive signal.** Disagreement marks words that are still moving on both splits, about twice the drop. Their DEV share is slightly larger: +0.04 at e3, +0.07 at e5.
  - Under equal-variance noise, this matched comparison is free of regression to the mean: d depends on the difference between the seeds, while the strata depend on their mean.
- **Consequence:** the only part with a positive sign is the part net of loss and entropy. That is why CW-b exists, and why CW-a must clear F0 at token level before it gets GPU time.

**Files:** scratch/cw_card/ holds pw_disagree.py, pw_matched.py, pw_disagree.json and pw_matched.json.

**The mechanism skeptic's CPU probe**
- Files: committee_probe.py and grad_share.py, which wrote probe20.json and grad12.json in the same scratchpad.
- Its proxy committee is three round-1 S cells that share a seed.
- Results:
  - Spearman(u, H) = 0.62-0.73 and Spearman(u, CE) = 0.54-0.65.
  - K=2 and K=3 give the same signal (Spearman 0.97).
  - Weighted mean gap to T100 on train: 0.63 under u-weights, against 0.51 for loss, 0.51 for entropy and 0.38 uniform.
  - u adds about +0.06 rank-R2 over (CE, H).
- **Conclusion:** u carries some learnability information beyond loss and entropy, but only a modest amount.

## Ceiling and odds
**Ceiling: about 0.10 nats/word**, roughly 1.3x data at the log-linear exchange rate. The candidate's 0.3 is withdrawn:
- It assumed late epochs are used on memorised positions. They carry 0.5% of the gradient.
- It assumed reweighting could make late epochs act like fresh data. Reweighting adds no new context.

**Where 0.1 comes from (guess)**
- The pair's last three member-epochs plausibly buy 0.3-0.4 nats/word of DEV; the base buys 0.42 from e3 to e10.
- The matched proxy's best uplift in DEV share is 9-16%.
- Applied to the whole late gain, that gives 0.03-0.06. About 0.1 if everything goes right.

**Measured analogues in LM pretraining**
- MiLe: about 0.001 nats/token.
- Kaddour's RHO with free selection: about 0.02 nats/token.
- Both are null or negative at matched compute or with stronger weighting.

**Odds**
- Median about +0.003, inside the floor. Mean about +0.005.
- P(passes the 10M kill rule) is about 0.04, the product of these guesses:

| step | P (guess) |
|---|---|
| G0 holds | ~0.5 |
| F0 keeps an arm | ~0.7 |
| F1 passes | ~0.35 |
| the screen passes, given F1 | ~0.3 |

- P(gain >= 0.05) is about 0.01.

**Most likely result**
- CW-b ties N-H and N-ale within the floor.
- CW-a, if it runs at all, ties or loses to the permuted null.
- D-flip lands close to U.

## R0 exposure ledger

| run | models trained | words fed per model | total | legal |
|---|---|---|---|---|
| B: frozen base at E* = 10, re-run | 1 | 100M | 100M | yes |
| U, CW-a, CW-b, N-perm, N-H, N-ale, N-CE, D-flip (each) | 2, non-shared | 50M (5 epochs) | 100M | yes: ref-0110 Q1 charges K times, 100M/K each |
| the weights themselves | none | 0 extra | 0 | they come from the two training forwards on the same batch; no extra or no-grad forward |
| F0 | none | 0 | 0 | diagnostic: inference on existing s0/s1 checkpoints. That pair is 2 x 100M, so its readings are labelled out of track. T100 is an analysis instrument only, never training input |
| F1 | 2 per run | 5 x 1M | proxy | mini-ruler proxy under ref-0110 Q2; every run counts as a tuning trial |

- No window is dropped or added.
- The word counter logs 2 x the words per step, and a run refuses to start if its registered total exceeds 100M.
- Tokenizer: the committed train_10M BPE. No outside data or models.

**Withdrawn substrates**
- **K=4 non-shared population.** Each member gets 2.5 epochs, so the epoch-3 onset never fires. At 10 epochs per member it would be 400M, out of track.
- **Any shared-trunk variant.** It counts once, but its members differ only in their heads and init, so u comes close to the permuted null.

## FLOPs and the R4.4 plan
**Training FLOPs: 1.00x the base.**
- 2 members x 5 epochs pushes the same tokens through the same S model as the base's 1 x 10 epochs.
- The JSD adds about 3e5 FLOPs per token, against about 4e8 for the two members' forward and backward: under 0.1%.
- So no FLOP-matched reference is needed under R4.4, at the screen or at promotion.
- Measured tokens/s, wall-clock and peak memory are reported. If measured FLOPs came out above 1.25x, the arm would need the shared ref-0265 reference.

**Inference: 2x** (two members). This is allowed and reported.

**Memory**
- Log-probs of shape [2, micro, T, 16384] in fp32 take about 2 GB per 16 x 1024 micro-batch.
- The statistics are therefore computed in chunks of 2,048 positions under no_grad.
- Peak memory is about 2x the base (two models plus their optimizer states).

## Null twins and the direction check
**What every twin shares with the arms:** the code path, the seed pair, the data order, the onset, beta and the clip. Each twin also gets the same per-micro-batch set of weight values as CW-a, reassigned by rank. Only which position gets which weight differs.

- **N-perm: random assignment.** It removes the information entirely. It only adds gradient noise (ESS 0.61), so it is a weak null on its own.
- **N-H: rank by Hbar** (MiLe-style; context-only, a proper objective).
  - This is the binding entropy null.
  - If it does as well as the arm, the effect is entropy weighting, which is known, not committee disagreement.
- **N-ale: rank by A_t = mean_k H(q_k).** This isolates the aleatoric part, since u = Hbar - A.
- **N-CE: rank by C_t** (loss only).
  - It depends on the label (focal-like) and is a published positive method (Sow 2025).
  - Matching the weight values removes the histogram confound the rules skeptic raised.
- **D-flip: rank by -u.** This is a direction check, using Wan's winning sign. It is reported only.
  - If D-flip beats U by 2 sigma, that is a known-idea finding in Wan's direction and goes to its own card. It is never promoted from this one.

**Comparators, not nulls**
- U: the pair with w = 1.
- B: the frozen base at E* = 10, seed 0.

**At promotion,** the binding null is the best of {N-perm, N-H, N-ale, N-CE} on the screen, picked on DEV before seeds 1-2 run. It gets 3 seeds.

## Gates before any 10M screen slot
**F0: premise at token level (CPU, no GPU; registered now)**
- **Inputs**
  - Committee: the winner's s0 and s1 checkpoints at 30M, 50M and 100M words. These are R0.5 milestones and have to be copied from the executor.
  - As a pair this committee is out of track, so its readings carry a diagnostic label. T100 serves as an analysis instrument.
- **Positions:** 40 windows x 512 tokens per split (train_10M and DEV), scoring positions 128-511, so 15,360 positions per split.
- **Quantities:** token-level u, Hbar, A and C, and the member-mean CE drop from 30M to 100M and from 50M to 100M.
- **Rules**
  - **CW-a** gets GPU time only if its top-u decile's DEV/TRAIN drop ratio is above the all-position ratio, at both 30M and 50M.
  - **CW-b** gets GPU time only if, within 16 Hbar strata, the top-u decile's ratio is at least the bottom-u decile's + 0.02, at both 30M and 50M.
  - **If neither holds,** the card dies before any GPU time: "premise fails at token level".
- **Reported:** the top-u decile's enrichment for high T100 entropy (an aleatoric proxy) and its mean gap to T100 (a reducible proxy), each against the top-Hbar decile.
- **Fallback:** if the 30M and 50M checkpoints were not kept, the per-word proxy above stands in. CW-a is then dropped and CW-b is kept.

**G0: does the substrate hold up?**
- The legal pair U must exist at 10M. That can be V8's run if it uses this code path and data order; otherwise this card's own U run (1 run, about 9 A100-minutes, counted as a trial).
- If U's full DEV is more than 0.02 nats/word behind B's, the card is parked ("substrate fails"). CW would have to make up that deficit plus 2 sigma, which is several times its median.

**F1: GPU mini-ruler proxy (about 17 A100-minutes; labels only)**
- **Setup**
  - Data: train_1M, with the mini_ruler.py hashes.
  - Schedule: the legal per-member one, 5 epochs per member, onset at member-epoch 3.
  - Init seed pairs {0,1} and {2,3}.
- **Runs:** U, CW-b, CW-a (only if F0 kept it), N-H, N-perm and D-flip, each on both seed pairs: 10-12 pair runs.
- **Pass** only if all of these hold, on the 2-seed mean of ensemble full-DEV nats/word:
  - (a) the better CW arm beats U by at least max(0.03, 2 x SD_1M), where SD_1M is the SD of U's two seeds;
  - (b) it beats both N-H and N-perm by at least max(0.02, 2 x SD_1M);
  - (c) D-flip is not better than U;
  - (d) its dev-minus-train gap is smaller than U's, with DEV lower (not TRAIN higher).
- **Otherwise:** "killed (proxy)".
- **Notes**
  - The candidate's threshold of ">= 0.1 nats on the 1M cell" is withdrawn.
  - A pass earns only the 10M screen, because 1M favours anything that regularises.
  - Every F1 run counts as a tuning trial (ref-0110 Q2).

## Kill rule (registered before any run)
**Setup**
- Size S, train_10M, the round-1 winner's recipe (ref-0065).
- Pair init seeds {0,1}, data seed 0.
- B is re-run on the same GPU type and code commit. U and B form the comparator stage (G0), and the arms follow on the same hardware type. If the critic requires one batch, U and B are re-run alongside the arms (+2 runs).

**Metric**
- Full-DEV nats/word on the registered scorer (critic ref-0053): dev_bpw.py on each member's fp32 HF export, written out per token by `dev_bpw_dump`.
- The pair is scored as the token-level mixture (`ceiling_report.ensemble_tokens`). B is scored from its own dump.
- Each member's final checkpoint is used. No checkpoint picking.
- Harness numbers are never differenced against scorer numbers.

**sigma**
- sigma is the critic's registered DEV seed floor, which is not registered yet.
- A two-seed estimate from the winner's s0 and s1 harness evals (5.0406 and 5.0371) gives sigma about 0.0025 with 1 df, so 2 sigma is about 0.005.
- Until the floor is registered, the screen reports numbers only.

**Best arm:** the lower full DEV of CW-a and CW-b, among those that passed F0 and F1. Ties within 0.005 go to CW-b.

**Kill if any of these holds:**
1. The best arm is not ahead of the better of U and B by 2 sigma.
2. It is not ahead of the best of {N-perm, N-H, N-ale, N-CE} by 2 sigma. If a null clears (1) on its own, it is reported as the known method it is (entropy weighting per MiLe, or loss weighting per Sow and focal loss), and is never credited to this card.
3. BLiMP-fast of the pair export drops by 2 sigma (BLiMP floor) or more against U.
4. Anti-synergy: against U, the ensemble gain (member mean minus ensemble) shrinks by more than the arm improves the member mean.
5. Measured training FLOPs exceed 1.25x B's and there is no shared FLOP-matched reference.

**Reported, never a pass**
- D-flip against U.
- Weight histograms by the train_10M count of the target word: 1-63, 64-8k and 8k+. A count of 0 cannot occur on train rows, so it appears only in the DEV-sub diagnostics.
- ESS per micro-batch, and the fractions of weights at the floor and at the cap.
- At every milestone: mean u, and the mean pairwise JSD between members on DEV-sub.
- The train-CE slope on top-u vs bottom-u positions. A faster fall on top-u positions means memorisation.
- Dev-minus-train, with the direction each side moved.
- Novelty and rare tilts from `babylm.instruments.signature`, against both U and B.
  - A "data-like" claim needs a novelty tilt of at least 0.8 against B.
  - Expected: 0.5-0.8, i.e. epoch-like.

**Provisional base (critic ref-0086)**
- This is a regulariser-type attack, so a kill reads "killed (provisional base)".
- Any change to the frozen base's WD, dropout, EMA, MNTP, LR or epochs forces one re-screen.
- The screen runs at the winner's tuned WD of 0.4 only, because reweighting effects depend on L2 (Byrd & Lipton).

**Promotion**
- Exactly one arm is promoted, picked on DEV before seeds 1-2 run.
- It moves to 3 seed pairs ({0,1}, {2,3}, {4,5}), size M and the official eval, run through the pair export.
- It is compared with the better of U and B (3 seeds each) and with the binding null (3 seeds).
- The 3-seed verdict reuses no screen numbers.

## Screen (size S, 1 seed pair, after F0, G0 and F1)

| run | ranking | note |
|---|---|---|
| B | none | frozen base at E* = 10, one model, re-run |
| U | none (w = 1) | the legal pair; also serves as G0 |
| CW-a | u | only if F0 and F1 keep it |
| CW-b | u within Hbar strata | |
| N-perm | random | |
| N-H | Hbar | binding entropy null |
| N-ale | mean_k H(q_k) | |
| N-CE | mean_k CE_k | |
| D-flip | -u | direction check, reported only |

- **Size:** 9 runs, or 8 without CW-a.
- **Cost:** about 8-9 A100-minutes each, because a pair pushes the same tokens through the same S model as B. With F1, about 95 A100-minutes in all.
- **Outputs:** shared/babylm/attacks/CW/<run>/, with curves as CSV and PNG and trial counts per arm.
- **R3 parity:** B's tuning-trial count (the round-1 grid) already exceeds this card's, and both are logged.

## Implementation plan (harness)
1. **`babylm/moonshot/committee_weight.py`** (pure functions, testable on CPU):
   - `committee_stats(logits_list, y, chunk=2048)` returns u, Hbar, A and C. It runs under no_grad in fp32 per chunk, applying the model's softcap.
   - `jsd_weights(u, mask, beta=0.5, lo=0.25, hi=3.0)`: mean 1 over the scored positions, clip-and-renormalise twice.
   - `assign(w, score, mode, strata=16, gen)`: reassigns weights by rank for the modes {raw, res, perm, H, ale, ce, flip}, keeping the set of weight values exactly.
2. **Pair training in `babylm/harness/train.py`, behind `--pair_seed S2`.** With the flag off, the port path is bit-identical.
   - Build member 2 with its own init seed and its own Muon + AdamW optimizer, with the same recipe and LR schedule.
   - Feed both members the same micro-batches. Forward both through `hidden()` and `logits()`, as the KD path already does.
   - After `--cw_start_epoch` (member-epochs consumed, default 2.0), compute the statistics and weights.
   - Backprop sum(w x CE) / N for each member, then step both optimizers.
   - `words_seen` counts both members.
   - Checkpoint and resume save both models, both optimizers, the sampler, every RNG and the null generator.
   - Log per-member curves and the pair's DEV-sub at every milestone.
   - Flags: `--pair_seed`, `--cw_mode {off, raw, res, perm, H, ale, ce, flip}`, `--cw_beta`, `--cw_clip`, `--cw_start_epoch`, `--cw_strata`.
   - If the V8 or component-1 pair code lands first, reuse it.
3. **`babylm/moonshot/pair_export.py`:**
   - An HF wrapper that loads both members and returns the log of their mean probability. It serves BLiMP-fast and the official pipeline, and is declared before any scoring.
   - Plus fp32 exports of each member for `dev_bpw_dump`.
4. **Logged at every eval:** weight histograms by band, ESS, the fractions at floor and cap, mean u, pairwise JSD on DEV-sub, the ensemble gain, train-CE slopes for top-u and bottom-u positions, tokens/s and peak memory.
5. **`babylm/moonshot/cw_premise.py` (F0, CPU):** loads the s0 and s1 checkpoints and T100, and writes the registered table and the gate decision.
6. **Tests in `babylm/tests/test_committee_weight.py`:**
   - u equals (1/2) sum_k KL(q_k || q_bar) and also Hbar - A, to within 1e-6.
   - Chunked and unchunked results agree.
   - Weights have mean 1 after clipping.
   - Every mode keeps each micro-batch's set of weight values exactly.
   - With a pair, `--cw_mode off` is bit-identical to the unweighted pair; without `--pair_seed`, train.py is bit-identical to the port path.
   - Before the onset, every mode equals off.
   - Resume is bit-exact.
   - The exposure counter adds 2 x the words per step, and a run refuses to go past 100M.
   - A CPU smoke run (2 layers, 64 dims, synthetic corpus) checks mechanics only.
7. **`babylm/configs/make_cw_jobs.py`** writes the F1 and screen job lists with the winner's exact flags.

**Build size: M.** The pair trainer is most of the work; the weight function is small.

## Stacking and overlap
- **Not summable with A1, A2 or A5.** All four are loss-side regularisers, and A5 is the same family (late-epoch per-target loss weighting). Any combination needs its own card (critic note at A2's acceptance).
- **Depends on the pair substrate (V8 / component 1).** If the legal pair does not hold up against B (G0), this card has nothing to stand on.
- **May work against the ensemble.** The same weights pull both members toward the label exactly where they disagree, which can shrink the ensemble gain the pair exists for. Kill clause 4 measures this.
- **The temporal single-model version** (Active Bias) is not part of this card.

## Risks (most likely first)
1. **The raw signal is mostly entropy, and points the wrong way.** In the per-word proxy, the top-disagreement decile is where later training looks most like memorisation. The noisy-label literature and Wan 2020 predict exactly this.
2. **There is little to free.** Memorised positions carry 0.5% of the gradient, and our base is not stalled late.
3. **The legal pair may lose to the 10-epoch base** (G0).
4. **The effect may wash out.** Context-only weights cannot move the ERM minimiser, so any effect may fade by the end of training.
5. **The floor is small.** 2 sigma is about 0.005, so the four quantile-matched nulls are the real bar, and the screen is one seed.
6. **Re-screen risk.** Any base change forces a re-screen under the provisional-base rule.
7. **Novelty search is incomplete.** Web search was exhausted, so the label may move further toward known.
