<!-- moonshot design round W3; status alive; ceiling 0.15 nats/word; expected 0.01; P(pass screen) 0.05; screen GPU 370 A100-min. Draft card: not yet submitted to the critic. -->

# Attack A5: variational tied table with test-time posterior averaging (card, 2026-10-10)

Status: card only. No code and no runs yet. This card folds in three skeptic reviews: prior art "wounded", rules "wounded", mechanism "dead". One "dead" verdict out of three, and no rule kills the card outright, so it stays alive. It has been cut down hard:
- arm B is dropped;
- the ceiling falls from 0.88 to 0.15 nats/word;
- the prediction falls from +0.04 to +0.01;
- P(pass) falls from 0.20 to 0.05.

**No IVON code is written and no GPU training runs until the zero-training Stage 0 gate passes.** The base is the round-1 winner `base_S_wd0p4_lr0p5_r1_ve0_tied` @ ref-0065 (full DEV 5.0367). It stays provisional until the critic freezes it, because its EMA cells are still pending and this is a regulariser-type attack (ruling 22:17Z (b)).

**Known idea, not new.** The family is variational weight posteriors trained on one MC sample per step and predicted by MC averaging: Graves 2011 (practical VI, weight noise), Blundell et al. 2015 (Bayes by Backprop), IVON (Shen et al., ICML 2024, arXiv 2402.17641), SWAG (Maddox et al. 2019) and MC dropout.

It has already been measured in small-data, many-epoch word LMs:
- **Gal & Ghahramani 2016** (PTB, about 0.93M words). 1000-sample MC beats the mean by 0.009-0.024 nats/word, while 10-model ensembles buy 0.066-0.12.
- **Fortunato, Blundell & Vinyals 2017** (BBB LSTM on PTB). It gains 0.054 nats/word over the Variational LSTM, evaluated at the mean. Sampling was "not significantly better than taking the mean".
- **Xue et al. 2021** (arXiv 2102.04754), a 34M-word Transformer LM, evaluated at the mean. Variational embeddings went from 41.50 to 41.01 PPL and one variational FF block reached 40.65. Variational noise on all six FF blocks reached 74.50, which is 0.585 nats/word worse.

IVON itself:
- It was trained for several epochs on OpenWebText, not in a single pass: GPT-2 125M-773M on about 49.2B tokens. PPL went 18.1 to 17.9, 14.5 to 14.1 and 13.0 to 12.6 with a 64-sample average. Its Fig. 2(c) says averaging helps only with ">> 8" samples.
- It was also trained for 200 epochs on small-data vision.
- Daheim, Moellenhoff, Ang & Khan (arXiv 2512.01930, ICML 2026) report GPT-2 125M PPL of 17.4 / 18.0 / 18.4 for IVON-PoCo / IVON / AdamW. IVON-PoCo needs more gradient compute, so it is not a 1.0x lever.

What limits within-basin averaging:
- Samples within one basin stay in one mode (Fort, Hu & Lakshminarayanan 2019).
- In SWAG-LoRA (Onal et al. 2024), Gaussian sampling adds little beyond the SWA mean; the gains come from using several basins.
- The temperature-1 posterior predictive is often worse than SGD, and cold posteriors are needed to fix it (Wenzel et al. 2020).
- Variational learning acts as adaptive label smoothing (Yang et al. 2025, arXiv 2502.07273).

A title and abstract scan of the BabyLM 2023-2025 proceedings found no Bayesian, variational, IVON or weight-noise entry. Ensembles appear there only as distillation teachers (Baby Llama, BabyLlama-2). One caveat: the shared web-search budget was used up, so this check relied on direct fetches of primary sources and may miss work from 2025-26.

The only untested piece is IVON on a tied table in exposure-capped 10M x 10 pretraining. The original card's claims of a "single pass" and "no multi-epoch or data-limited result" were wrong and are withdrawn.

Labels used below:
- Kim/Kotha's 5.17x is extrapolated; the measured figure is 2.09x.
- Baby Llama's 4-seed ensemble (about 0.18 nats/word) costs K x exposure and is out of track.

## Mechanism
The frozen base has no value embeddings. It has one tied 16,384 x 384 table: 6,291,456 parameters, 22.9% of 27.5M, initialised N(0,1). The table is the input embedding (RMS-normalised before block 1) and also the output head (softcap 15). It sits in the AdamW "head" group at lr 0.0104 with adam_wd 0. The 21.2M block matrices stay on Muon, and the 54 scalars stay on AdamW.

**Arm.** IVON (Shen 2024, Alg. 1) replaces AdamW on the tied table only, including the positivity correction term 0.5(1-b2)^2 (h-h_hat)^2/(h+delta). Each entry keeps a mean m, a Hessian estimate h and a momentum g. Each step:
1. Draw eps ~ N(0, sigma^2), with sigma = 1/sqrt(lambda (h+delta)).
2. Write theta = m + eps into the table.
3. Run the ordinary single forward and backward. Assert train_mc_samples == 1.
4. Update the estimates: h_hat = g_hat * eps / sigma^2; g <- b1 g + (1-b1) g_hat; then the h update.
5. Update the mean: m <- m - alpha_t * clip((g_bar + delta m)/(h+delta), xi).
6. Restore m in the table.

Muon steps the other groups on the same gradient, which is taken at the sampled theta.

**Registered knobs.** None of these are tuned; every value goes in the trial log.
- **lambda**, two arms: 16M (tokens per epoch, 15.96M) and 160M (tempered, counts the repeats).
- **beta2 = 0.999.** The run has about 2,440 optimizer steps (159.6M tokens / 65,536). After that, 1 - b2^T = 0.91 of h is learned from data. With the paper's 1-1e-5, 97.6% of h would still sit at h0, sigma would be the same for every entry, and the arm would equal the fixed-sigma null by construction. The paper says 0.999 "can make training unstable at times", so instability is a kill, not a reason to re-tune.
- **h0, delta, b1, xi.** h0 = 1e-3 (the paper's GPT-2 value). delta = 1e-6 (the paper's value; it matches the base's adam_wd 0 up to a negligible shrinkage, and an improper delta = 0 is not allowed). b1 = 0.9, xi = 1e-3.
- **alpha.** A matched-step rule sets alpha * xi equal to the base's head lr of 0.0104 at peak, so alpha = 10.4, on the same schedule (cooldown 0.5, floor 0.1). One more trial runs at alpha x 0.5 for whichever lambda is better on DEV-sub. That makes 3 arm runs.
- **Prior sigma0** = 1/sqrt(lambda (h0+delta)): 7.9e-3 at lambda 16M and 2.5e-3 at 160M. Against the N(0,1)-initialised table that is about 1% relative noise.

**Prediction.** Draw K = 32 weight samples theta_k = m + sigma * eps_k from the final (m, h), using fixed common random numbers: eps_k comes from a CPU generator seeded with mc_seed + k from the export meta. Average the probabilities per token and return logits = logsumexp_k log_softmax(z_k) - log K, so dev_bpw.py runs unmodified. The K = 1 score at m is reported beside it. The curve over K = {1, 8, 32, 64} is a diagnostic. The original card used K = 8, but IVON's own GPT-2 averaging needs ">> 8", so K = 8 would register a setting known to sit below break-even.

**Arm B (IVON on every group, Muon removed) is dropped**, for three reasons:
- Xue 2021 measured variational noise on all FF blocks at -0.585 nats/word.
- Swapping out Muon with one LR trial confounds the optimizer with the posterior.
- IVON is very sensitive to h0 (Shen Table 8: 9.89% vs 92.71% accuracy), so one trial could not tell a dead mechanism from a mis-set optimizer.

Arm B would need its own card with a diagonal-optimizer control matched on FLOPs and tuning.

## Why (which measured slice)
The arm targets the variance part of the gap: the posterior channel that the ceiling map calls C3. Its registered bands are g(8) 0.10-0.25 and g_inf 0.12-0.30 nats/word. Those bands are unmeasured, and C3's members cost K x exposure.

A within-basin diagonal Gaussian captures only part of that channel. Gal & Ghahramani's MC-vs-ensemble numbers put the share at about 10-20%, and at most about 35%. The second leg is an adaptive per-entry regulariser at K = 1, from posterior-width noise during training. Against reasonable baselines that leg has measured 0.01-0.03 nats/word (Xue 2021; IVON on GPT-2; BBB gained 0.054, but only over 2016-era dropout).

Where it acts: sigma is widest where the Fisher is small, which means rare rows. Expect the averaging leg to land on the 1-63 and never-seen bands (0.16-0.18 and 0.10 of the published gap), not on the 64-8k band (0.36), where sigma is tight.

Not targeted:
- Calibration. The best global temperature buys only 0.013 on published GPT-BERT.
- The overconfidence figure (0.57 vs 0.39). It comes from the published checkpoints, not our base.

So the averaging leg must beat a vocabulary-band temperature null before it counts.

The base no longer overfits late. Its DEV-sub minimum is at e10 (final minus minimum = 0.000), and its gap to the 100M twin is 0.701 [0.604, 0.785]. The base builder has already removed the value-embedding memoriser. The original card's premise ("40-60% of parameters in memorising groups") described the old base and is withdrawn.

**Why the base does not already have this.** The base is a MAP point: Muon WD 0.4 on the matrices, no WD on the table, no dropout, no EMA. Free post-hoc averaging of the base at matched inference is the binding comparator Base-K below, so a pass cannot just be inference-time ensembling that needs no IVON training. The original card read Kim/Kotha ("ensembles beat the N->inf regularised model") as support. That result is about independent-init, multi-basin ensembles and does not transfer to within-basin samples, so that use is withdrawn.

## Null twins
All nulls share the arm's seed, data order, schedule, exposure and code path, and run at the better arm's (lambda, alpha). Any null with a posterior is scored at K = 32 with CRN.
- **N0, no posterior.** The IVON mean update runs with sigma = 0 in the forward and h frozen at h0, which is clipped, sign-like momentum on the table. It separates "IVON as an optimizer" from the posterior. It is scored at K = 1 and with its own post-hoc diagonal Laplace at K = 32; the better score counts.
- **N1, permuted width.** The full IVON path runs, but each step sigma is randomly permuted within the table before sampling. h_hat uses the sigma actually drawn, so the h estimate stays unbiased. This keeps the noise energy and the IVON mean update and removes which entry gets which width. The final K = 32 samples use permuted widths too.
- **N2a and N2b, generic weight noise (Graves 2011).** Base AdamW on the table plus fixed isotropic Gaussian weight noise each step. N2a uses sigma = sigma0, which can be computed from the prior. N2b uses the arm's logged training-time RMS sigma and runs after the arm on the same GPU type. Both are scored at K = 32 at the same sigma.

The best null on DEV-sub, plus any null within 2x floor of it, is scored on full DEV, and the best of those binds.

**Comparator Base-K** (matched inference, no extra exposure to the base itself). It is whichever of these two is better on DEV-sub:
- (a) The base's probability mix over all 10 milestone checkpoints. The weights are fit on DEV-sub, so the selection bias runs against the arm.
- (b) A diagonal Laplace at the base's final weights, with sigma^2 = 1/(lambda (F + delta)), lambda in {16M, 160M} and K = 32. F is the sampled-label Fisher of the tied table on 200k train_10M tokens. That Fisher pass is about 0.125M words and goes on the comparator's ledger only; it favours the comparator.

MC dropout does not apply because the base has no dropout.

## Kill rule (registered before any run)
**Setup.** Size S, train_10M x 10, the frozen base recipe and seed 0. The base is re-run in the same batch with `--ivon none`, which is the bit-identical path. Metric: full-DEV nats/word from dev_bpw.py, unmodified, on the HF export of the final checkpoint (m, sigma; K = 32; mc_seed in the export meta). No checkpoint picking.

**Main kill.** Kill if the best arm at K = 32 is not ahead of each of the following by 2x the registered DEV seed floor:
- (i) the base at K = 1;
- (ii) Base-K;
- (iii) the best null.

**Mechanism check (also a kill of this card).** All four must hold:
- the arm's K = 32 score beats its own K = 1 score at m by at least 1x floor;
- the document-bootstrap 95% lower bound of that gain is above 0;
- the gain beats a vocabulary-band temperature null: logits z_v / T_b(v) + c_b(v) over 5 train_10M frequency bands of the vocabulary, fit on one half of DEV-sub and scored on the other;
- the MC error across two mc_seeds is reported.

If only the K = 1 leg survives, the arm is relabelled "variational / weight-noise regulariser (Graves 2011, Blundell 2015, Xue 2021)". That is a known lever and cannot be promoted under this card.

**Other kills:**
- Training instability: a NaN, or any 50-step window whose mean training loss rises more than 0.5 nats above the previous window. No re-tune.
- A drop of 2 sigma or more on BLiMP-fast, scored with the K = 32 export with CRN fixed per pair.
- More than 1.25x training FLOPs (predicted 1.00x).

**Verdict conditions.**
- No verdict before the floor is registered (floor seeds 1-2 of the round-1 winner).
- While the base is provisional, a kill reads "killed (provisional base)". Any base change to WD, EMA or dropout forces a re-screen (22:17Z (b)).
- Best arm = lowest DEV-sub at K = 32. A tie within 0.005 goes to the larger lambda, then the smaller alpha.
- Exactly one arm is promoted, picked on DEV before seeds 1-2 run, to 3 seeds, size M and official eval. The 3-seed verdict re-uses no screen numbers.

**Pre-registered predictions** (each scored once):
- lambda = 160M wins at K = 1, and lambda = 16M has the larger averaging gain.
- The tempered arm's K32-minus-K1 gain is about 1/10 of the untempered arm's, since sigma^2 scales as 1/lambda.
- Arm minus base: -0.01 to +0.02. Averaging leg: 0.000 to 0.010. Arm minus best null: within +/-0.01.

**Signature (reported).** Gains by frequency band, seen vs new text, and the novelty tilt. Smoothing and regularisation both tilt toward novel trigrams, so a tilt near 1 cannot certify "data-like" here. Seen-text loss must not rise by more than 1 floor.

## Odds (guesses, stated plainly)
- **Ceiling: 0.15 nats/word**, if everything goes right. That is a within-mode share of at most 0.35 times g_inf of at most 0.30 (about 0.10), plus at most 0.05 from the regulariser, with some overlap. The original 0.88 assumed the full Bayes posterior over the model family; a mean-field Gaussian around one mean is not that.
- **Median: +0.01. P(pass screen): 0.05. P(Stage 0 passes): about 0.15.**
- **Most likely outcome:** Stage 0 kills the averaging leg for about 10 A100-minutes, and the card closes without any IVON code.

## R0 exposure ledger
| item | words |
|---|---|
| arm: one forward/backward per window per epoch, on one sampled theta | 10M x 10 = 100M |
| K = 32 test-time samples | 0 (inference on DEV and eval text only) |
| teacher, generator or synthetic text | none |
| each null (N0, N1, N2a, N2b) and the base re-run | 100M each, separate pipelines |
| Base-K Fisher pass | about 0.125M (comparator only; favours it) |
| Stage 0 Fisher on the existing base checkpoint | about 0.125M (diagnostic label, never a verdict) |

The arm's pipeline total is 100M. Assert train_mc_samples == 1 next to the exposure counter, and ask the critic to log that post-hoc posterior samples count as zero exposure.

**Trial ledger.**
- Arms: 3 (lambda 16M, lambda 160M, and the better lambda at alpha x 0.5).
- Nulls: 4.
- Stage 0 lambda settings: 2 registered plus 2 diagnostic.
- Base-K selection on DEV-sub (base side).
- Mini-ruler label runs: 6, if run.

The base has about 20 logged DEV trials (s1 plus round 1), so R3 parity holds; the ledger states it.

## FLOPs and R4.4
**Training.** IVON adds about 10 elementwise ops on 6.29M entries per step plus one Gaussian draw. That is under 0.1% of a step (6 x 27.5M x 65,536 tokens is about 1.1e13 FLOPs), so training costs 1.00x; wall-clock is logged.

**Memory.** h and g replace AdamW's exp_avg_sq and exp_avg, and one eps buffer of 25 MB is added. R4.4 is not triggered.

**Inference.** K = 32 forwards per scored token. This is allowed and reported, and matched-inference fairness is handled by Base-K.

## Fastest falsification: Stage 0
Stage 0 needs no training, costs about 10 A100-minutes, and runs before any IVON code exists. It uses the frozen base's final checkpoint (s0, chck_100M).
- **(a) Fisher.** Compute the sampled-label diagonal Fisher F of the tied table on 200k train_10M tokens.
- **(b) Arithmetic gate.** Simulate h_T = b2^T h0 + (1-b2^T) F at T = 2,440 under the registered (b2, h0, delta, lambda). Report sigma_T/RMS(m) by frequency band and the coefficient of variation of sigma_T. If under 10% of entries, or under 10% of the Fisher mass, have sigma_T/sigma0 outside [0.5, 2], the "learned width" claim is false and the card closes. The regulariser leg can then move to a weight-noise card.
- **(c) Diagonal-Laplace predictive.** Use sigma^2 = 1/(lambda (F + delta)) with lambda in {16M, 160M} registered ({1.6M, 1.6B} as diagnostics) and K in {8, 32, 64}, with CRN, probability-averaged on DEV-sub. Compare against the mean prediction, the best global temperature and the vocabulary-band temperature null. Kill the averaging leg, and with it this card, if either:
  - the best registered-lambda K = 32 gain over the band-temperature null is under max(0.02, 1x floor) nats/word; or
  - its document-bootstrap 95% lower bound is at or below 0.
- **(d) Milestone mix.** Report the base's milestone-checkpoint mix on DEV-sub, the other candidate for Base-K.

**Registered prediction:** the (c) gain is 0.000-0.015, and Stage 0 closes the card with P about 0.85.

**Caveat:** a Laplace around an AdamW point can be wider or narrower than IVON's own posterior. Stage 0 bounds the averaging leg; it cannot prove IVON's version empty. The card accepts that risk rather than compute about 6 A100-hours on a screen with P(pass) 0.05.

**Optional label, only if Stage 0 passes.** Mini-ruler at 1M x 10: the base vs fixed Gaussian weight noise on the table at three sigmas, 2 seeds each. That is about 6 A100-minutes, labelled "mini-ruler proxy" and counted as trials. The test is one-sided: if no sigma beats the base by at least 0.03 at 1M, the regulariser leg is presumed empty. It is never an IVON-vs-null test, because at 244 steps beta2 = 0.999 learns only 22% of h.

## Screen (only if Stage 0 passes; size S, seed 0)
| run | what |
|---|---|
| base | `--ivon none`, same batch |
| A5a | IVON table, lambda 16M, alpha 10.4 |
| A5b | IVON table, lambda 160M, alpha 10.4 |
| A5c | the better lambda on DEV-sub, alpha 5.2 |
| N0 | sigma = 0, h = h0 |
| N1 | permuted sigma |
| N2a / N2b | weight noise at sigma0 / at the arm's RMS sigma |

Cost:
- Training: 8 runs, about 64 A100-minutes.
- Eval, step 1: harness bf16 K = 32 on DEV-sub for every run, including the K-curve, about 1 A100-hour.
- Eval, step 2: dev_bpw fp32 K = 32 on full DEV for the best arm, the best null and Base-K. Time one fp32 pass first; expect about 1-2 A100-hours each. With the gate, the expected compute is about 10 + 0.15 x 360, or about 65 A100-minutes.

## Official-eval handling (declared in the export meta before any score)
- MC averaging lives only in HintonForCausalLM.forward. The weight samples are fixed (mc_seed + k), so scores are deterministic and do not depend on batch composition or padding. Both sentences of a BLiMP pair use the same K samples (CRN).
- HintonModel, the AutoModel used for GLUE, uses m only. sigma is a buffer, never an nn.Parameter, so GLUE fine-tuning is identical to the base's. GLUE is therefore scored at the mean; zero-shot is scored at K = 32.
- dev_bpw.py is unchanged. Every DEV and official score of the arm is labelled "K = 32 inference".

## Overlaps (not orthogonal)
- **A2.** The input side of the table noise overlaps A2's N-gauss null. The overlap is weak here, because embnorm makes 1%-scale row noise small.
- **A1 and the base's milestone mix.** The averaging leg overlaps A1's trajectory teacher and the base's own milestone mix.
- **WD.** delta overlaps weight decay.
- **A1's null family.** Yang 2025 ties variational learning to adaptive label smoothing, which is what A1's nulls test.

A gain here cannot be added onto A1 or A2; a combination needs its own card. The posterior samples could later serve as the teacher for ensemble-to-student distillation at no extra exposure, under a separate card.

## Implementation plan in the harness
Stage 0 takes about half a day; the IVON arm takes about one day after that.
- **`babylm/moonshot/posterior_eval.py`** (Stage 0 and Base-K): sampled-label Fisher, the sigma_T arithmetic, Laplace K-sampling with CRN, the milestone mix, the band-temperature null, the document bootstrap, and per-token dumps of the averaged probability (pt_devfull_final.npy).
- **`babylm/moonshot/ivon.py`** (about 200 lines, written from the public reference, not pip-installed): IVON on one tensor; a sample()/restore context around forward and backward; modes learned, perm_sigma and zero_sigma; fixed-sigma weight noise for N2.
- **`train.py` flags** (levers.py pattern; off = bit-identical): `--ivon {none,table}`, `--ivon_lambda`, `--ivon_h0`, `--ivon_beta1`, `--ivon_beta2`, `--ivon_delta`, `--ivon_clip`, `--ivon_alpha` (defaults to the matched-step rule), `--ivon_null {none,zero_sigma,perm_sigma}`, `--wnoise_sigma`. With `--ivon table`, the tied table leaves the AdamW head group.
- **Checkpoints and logging.** Checkpoints and resume.pt hold m (as the parameter), h, g, the step and the sampler RNG. In-training curves are computed at m. Each step logs sigma's RMS and coefficient of variation, and the 50-step loss-spike monitor runs throughout.
- **`export_hf.py`:** `--mc_samples K --mc_seed s` stores sigma as a buffer and writes the meta.
- **CPU tests (`babylm/tests/test_ivon.py`):**
  - the off path is bit-identical to the base commit;
  - h_hat is an unbiased estimate of diag H on a quadratic;
  - zero_sigma mode equals a hand-written clipped-momentum reference;
  - perm_sigma keeps the multiset of sigma values;
  - resume is bit-exact, RNG included;
  - the export at K = 1, sigma = 0 gives the base's logits;
  - two loads score identically;
  - scores are invariant to batch size and right padding;
  - HintonModel ignores sigma;
  - the train_mc_samples == 1 assert fires;
  - the exposure counter is unchanged.

## Skeptic fixes adopted
- **Prior-art line rewritten.** Gal 2016, Fortunato 2017 and Xue 2021 measured this family in small-data LMs; IVON ran multi-epoch on 49.2B tokens. Label: "known idea".
- **Ceiling 0.88 -> 0.15, prediction 0.04 -> 0.01, P(pass) 0.20 -> 0.05.**
- **K raised from 8 to 32 (binding), with the K-curve as a diagnostic.** The prior-art and mechanism skeptics asked for K >> 8, while the rules skeptic wanted K fixed. K = 32 is fixed before any run.
- **Arm B dropped.**
- **Premise rewritten for the frozen ve0+tied base.**
- **beta2, h0, delta, xi and alpha all registered.** beta2 = 0.999 so that widths are actually learned; instability kills the run.
- **Binding comparator at matched inference.** Base-K = the milestone mix or a diagonal Laplace.
- **Null set:** N0 (sigma = 0), N1 (permuted sigma), and N2 at two calibrations; the best null binds.
- **Stage 0 replaces the biased SWAG-on-2-milestones probe.**
- **Mini-ruler restricted** to a one-sided weight-noise label.
- **HF export:** CRN, MC averaging confined to the causal LM, GLUE at the mean, dev_bpw unmodified, "K = 32 inference" labels.
- **Full trial ledger and the train_mc_samples == 1 assert.**
- **Retracted "orthogonal to A2".**
- **Cold-posterior prediction registered.**
- **One change of my own.** The skeptic's "per-frequency-band temperature" null is redefined over vocabulary bands. A temperature keyed to the target word's band would use the label and would not be a valid predictive distribution.
