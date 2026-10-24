<!-- moonshot design round W1; status alive; ceiling 0.12 nats/word; expected 0.005; P(pass screen) 0.04; screen GPU 60 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card: spike-gated Muon, revised as a direction-wise SNR gate (wounded and gated; 2026-10-10)

Status: card only. No code, no runs. Skeptic verdicts: prior-art **wounded**, rules **wounded**, mechanism **dead**. One of three voted dead, and every rule problem has a workable fix, so the card stays alive. **The card as first written is dead, though.** Its scalar Marchenko-Pastur bulk edge failed a measured CPU probe (F0, below). What survives is a rewrite:
- a per-direction noise test,
- applied after Muon's Newton-Schulz step, with no RMS rescale,
- in the cooldown only,
- against noise-blind nulls.

It must pass three cheap gates before any harness code or attack GPU run: G0 (premise), F0' (calibration) and F1 (memorisation separation). **Priority: lowest in the moonshot queue.** Build only if G0 and F0' pass.

## Known ideas (label: **known idea**, pending a search that is still owed)
- **Base.** Muon (Jordan et al. 2024).
- **Shrinking Muon's tail as noise.** Pion (Fan et al., arXiv 2605.19282, 2026) uses a high-pass Newton-Schulz step that drives Muon's small singular values toward 0, at Muon's cost. Pion also benchmarks LRMuon, an exact-SVD top-k projection of the momentum: LRMuon beats Muon, loses to Pion at every rank, and pays for a per-step SVD.
- **Singular-value threshold from a noise level.** Gavish & Donoho: hard threshold (arXiv 1305.5870) and optimal shrinkage (arXiv 1405.7511).
- **RMT filtering against memorisation.** Staats, Thamm & Rosenow (arXiv 2206.03927) filter small singular values against label-noise memorisation.
- **Micro-batch spread as a noise estimate.** McCandlish et al. 2018 (arXiv 1812.06162).
- **Filtering updates by agreement across batches to cut memorisation.**
  - GAF (Chaubard & Kochenderfer, arXiv 2412.18052).
  - AND-mask (Parascandolo et al. 2020).
  - M3/RM3 (Zielinski et al., arXiv 2003.07422).
- **Related update rules.** Shampoo/SOAP epsilon-damping. Dion and GaLore (low-rank updates). HTMuon (Pang 2026) was cited by the proposer and has not been verified.

**Counter-evidence that must be cited with the card:**
- Muon's isotropic update spectrum is what makes it learn tail classes evenly; the gains sit in V/O and FFN matrices (Wang et al., arXiv 2509.26030).
- An update restricted to the dominant subspace stops reducing loss, and the bulk alone is as effective (Song, Ahn & Yun, arXiv 2405.16002).
- BLaLM (arXiv 2511.05560, Table 4; 100M words, up to 10 epochs, shared LR 4e-4): Muon val PPL 7.95 vs AdamW 11.21. On repeated data **Muon generalised better** (confounded; AdamW likely under-tuned). The proposal cited this -0.34-nat gap without its sign.
- Gaps between tuned optimisers shrink to 1.1-1.4x data, and SOAP/NAdamW beat Muon at 16x Chinchilla (Wen et al., arXiv 2509.02046).
- Muon speeds grokking (arXiv 2504.16041).
- arXiv 2605.06654 finds Muon memorises corrupted sequences more than AdamW in a synthetic test (65.0% vs 55.8% exact match). The paper is about SFT forgetting, labels its full-rank link a hypothesis, and our gap shows no verbatim recall.

**What is not yet found in prior art:** a per-direction cutoff set from the measured cross-micro-batch noise of the Nesterov vector, in multi-epoch, data-limited LM pretraining.

**Search owed.** The skeptics' web-search budget ran out. A targeted search is still owed: "Marchenko-Pastur / noise-thresholded / SNR-gated Muon", HTMuon, NorMuon, AdaMuon, GAF in a spectral basis. Until it is logged empty the label is "known idea"; at best it becomes "known ideas, new combination".

## Mechanism (revised)
Per Muon matrix (m x n, m <= n), there are 72 per step at S: q, k, v, o (384x384) and fc/proj (384x1536) in 12 layers. The harness forms the Nesterov vector u_t = (1-mu) g_t + mu buf_t and steps along O_t = NS5(u_t) (sprint1/harness/optim.py; mu ramps 0.85 to 0.95 over 150 steps). The arm leaves that path untouched and adds a filter after it.

1. **Basis.** Every R = 20 steps, an fp32 eigh of u_t u_t^T gives U and s^2. Then V = u_t^T U diag(1/s) for directions with s >= 1e-3 s_max. Smaller directions are left unfiltered (w = 1), because fp32 eigh error is O(1) there.
2. **Per-direction noise.** The harness already runs K = 4 micro-batches (16 x 1024 tokens each) per step.
   - After each micro backward, the micro-gradient is G_k = K x (increment of p.grad).
   - Only its diagonal in the stale basis is computed: a_{k,i} = u_i^T G_k v_i.
   - The noise variance of u_t along direction i is nu_i^2 = kappa(mu_t) Var_k(a_i) / K, with K-1 degrees of freedom.
   - kappa(mu) = (1-mu^2)^2 + mu^4 (1-mu)/(1+mu) is the Nesterov vector's variance per unit of per-step gradient variance. At mu = 0.95 it is 0.0304, 1.19x the buffer's 0.0256.
   - Noise is measured direction by direction, so Kronecker-structured, anisotropic noise is handled. No iid edge is assumed.
3. **Gate.** Signal s_i = u_i^T u_t v_i (the current Nesterov vector in the stale basis). Weight w_i = clip(1 - c^2 nu_i^2 / s_i^2, 0, 1). A direction whose momentum coefficient is within c noise SDs is shrunk toward 0.
4. **Filter.** O'_t = O_t - U diag(1-w) U^T O_t, with **no RMS rescale**.
   - A kept direction gets exactly Muon's step and a gated one gets less. No direction gets more than Muon would give it, which removes the RMS rescale's 2-4.5x amplification.
   - c = 0 gives O' = O bit for bit. So the arm's code path contains Muon exactly, and there is no NS5-vs-exact-polar confound.
5. **When.** One pre-registered arm, from the cooldown start (step 1218 of 2435, epoch 5.0) to the end.
   - Before the cooldown, decoupled weight decay at peak LR forgets in 1/(0.0325 x 0.4) = 77 steps, against 244 steps per epoch. A window's imprint has shrunk to about 4% before the window comes back.
   - Recurrence can only build up in the cooldown, where the LR falls to 0.1x and the memory grows to about 770 steps (about 3 epochs).
6. **c is not tuned.** F0' sets it: the smallest c at which at most 1% of pure-noise directions pass on both calibration checkpoints, stale-basis winner's curse included. It is registered before any GPU run.

The embeddings, head and scalars stay on AdamW, unchanged.

**Logged:**
- per matrix every 10 steps: pass share, kept energy share of O, median SNR, and the update-norm ratio |O'|/|O|;
- per step: k_t and the sorted w profile, saved for the replay null.

**What it measures, plainly (adopted from the mechanism skeptic).** Cross-micro-batch disagreement measures batch-sampling noise (g_batch - g_train). It does not measure the fixed finite-sample error of the corpus (g_train - g_pop).
- A window's own memorisation component looks like sampling noise in the step that contains it, so the gate shrinks it.
- The gate shrinks weak, consistent, generalising directions (tail learning) just as much.
- The corpus-level component g_train - g_pop passes as signal.

So the arm is a variance-reduction filter on Muon's whitened update. Whether it removes more memorisation than learning is an empirical question, and F1 answers it before any GPU run.

## Why (which slice), and the premise stated honestly
The target is bottleneck 1, known words in new contexts. New text carries an extra 0.61 nats/word; the 64-8k band is 0.36 and the 8k+ band 0.22 (all measured on published GPT-BERT checkpoints). The claim: Muon's whitening gives every bulk direction a full-size step, including per-window idiosyncratic directions that recur once per epoch. Damping the ones that sit within sampling noise should lower the memorisation equilibrium in the cooldown.

**The premise is unmeasured, and the evidence leans against it:**
- (a) The only BabyLM-scale data point (BLaLM) favours Muon on repeated data.
- (b) The 0.61 slice and the epoch-7 stall were measured on GPT-BERT, which was not Muon-trained (by chronology; not re-verified). No Muon-specific memorisation has been measured.
- (c) Our round-1 winner still reaches its DEV minimum at epoch 10 (5.0367). The A5 premise check found frequent words still improving in epochs 7-10. No AdamW cell exists in any base grid.
- (d) Wang et al. predict harm on the 1-63 and 64-8k bands, which are the bands this card claims.

Hence G0 runs first.

**What the base would not capture:**
- Tuned weight decay and a wider model leave Muon's per-direction step flat.
- EMA (queued in the base's reg stage) averages over time, not per direction.
- These knobs overlap the mechanism, so any base change in weight decay, EMA, LR, momentum, batch size or optimiser family forces a re-screen (ref-0086(b)).

## R0 exposure ledger
| Item | Training exposure | Note |
|---|---|---|
| Arm, N1, N2, N3, base re-run (each) | 99,541,960 words | One forward and one backward per window. The K = 4 micro-gradients come from the existing backward; no extra forward. |
| Noise statistics | 0 | Read from p.grad between micro backwards |
| F0' and F1 probes (CPU) | 0 (no parameter update) | Gradients at fixed weights on train_10M and DEV-sub windows, logged as calibration/diagnostic trials. DEV-sub is selection data. |
| G0 AdamW cells | Separate base models | Base builder's trials, not part of the attack pipeline |
| Adapted mini-ruler, CPU tiny | Separate proxy models on train_1M | Labels; counted as tuning trials (00:40Z Q2) |
| Promoted pipeline | 99.54M <= 100M | Single model |

Guard: the OOM fallback (train.py:777-793) halves the micro-batch and so changes K. For the arm and the nulls it must hard-fail instead.

## FLOPs, the R4.4 plan and wall-clock
- **Base cost.** 193.5M FLOPs/token x 65,536 tokens = 1.27e13 FLOPs per step, over 2,435 steps.
- **Added per gated step:**
  - micro-gradient projections: 4 x ~16 GFLOP = 65 GFLOP (0.5%);
  - post-filter: ~33 GFLOP (0.26%);
  - eigh plus u u^T every 20 steps: under 1 GFLOP per step, amortised.
- **Total.** The gate runs in half the steps, so the total is about **1.004x FLOPs** (1.01x as a bound). R4.4 is judged on FLOPs (A3 ruling) and is not triggered. Under 01:20Z the arm sits at the base's FLOP level; a FLOP-matched reference is needed only at promotion.
- **Wall-clock.** A per-step eigh is ruled out. Torch batches syevj only for n <= 32, so n = 384 loops cusolver syevd. CPU timings for the 72-matrix batch ranged from 0.6 to 66 s; the A100 cost is unmeasured. Amortised over 20 steps, the cost should be about 10-30 ms per gated step against the base's 179 ms step.
- **Timing gate before the screen.** A 60-step CUDA smoke inside the cooldown must show at most 1.15x the base's step time.
  - If it does not, set R = 50.
  - Failing that, the fallback is a Pion-style polynomial high-pass on u/tau_i, registered as a new arm before the screen.
- Wall-clock is reported next to FLOPs.

## Null twins
All nulls use the same base, seed, data order, steps, K, micro hook, eigh refresh and filter code path.
- **N1, noise-blind threshold (binding; the mechanism's information removed).**
  - Same basis and the same w formula, but c^2 nu_i^2 is replaced by (q_l x median s_i)^2: Gavish-Donoho's unknown-noise rule, the Pion family's cutoff relative to the momentum's own spectrum.
  - The micro hook still runs, so compute is the same. A test asserts the null never reads its statistics.
  - q_l per layer type is fixed before the screen on F1's epoch-7 momentum proxies, so that its pass share matches the arm's.
- **N2, magnitude replay (binding).**
  - Replays the arm's logged per-matrix, per-step k_t and sorted weight profile onto the top-k_t directions by s_i.
  - Same count and same weights; directions are chosen by magnitude, not by measured noise. This is plain low-rank Muon with the arm's own rank schedule.
  - Runs after the arm, because it needs the arm's log.
- **N3, high momentum (binding comparator; no spectral choice).** Base Muon with mu = 0.975 from step 1218. If plain time-averaging matches the arm, spectral gating is not needed.
- **Dropped:**
  - The random-subspace null. It keeps mostly bulk directions, is ill-posed in fp32 for tiny s_i, and loses by construction.
  - The fixed top-k null. Its k is constant, so a rank schedule would be credited to the noise measurement, and it can only be calibrated after the arm has run.
  - The exact-polar control. It is unnecessary, because c = 0 reproduces Muon bit for bit.
- **Reported, not binding:** tuned AdamW from G0.

## Kill rule (registered before any run)
**Setup.** Size S. Base = round-1 winner base_S_wd0p4_lr0p5_r1_ve0_tied @ ref-0065 (Muon lr 0.065 x 0.5, muon_wd 0.4, adam_wd 0, tied embeddings, cooldown 0.5, floor 0.1), train_10M x 10, seed 0. The base is re-run in the same batch, on the same GPU type, through the arm's code with --optimizer_variant muon.

**Metric.** Full-DEV nats/word from dev_bpw.py (collapse protocol), on the final checkpoint. No checkpoint picking.

**Comparator.** The better of:
- the base re-run;
- the base at lr_scale 0.25, run only if s1x has not bracketed the LR below 0.5;
- the tuned AdamW base, if G0 makes it the base.

**Kill if any of these holds:**
- **K1:** the arm's gain over the comparator is below max(2 x seed floor, 0.03 nats/word).
- **K2:** the arm's gain over the best of N1, N2 and N3 is below 2 x seed floor.
- **K3, band harm.** Measured on harness per-word dumps with band_split.py, the same instrument for every run (1000 paired bootstrap replicates, 2,000-word blocks, seed 0; A5's settings):
  - the 1-63 band (base 11.48) is worse than the base by more than 0.02 nats/word; or
  - the 64-8k band (base 6.33) is worse than the base, with the 95% CI entirely above 0.
- **K4, gate degenerate:** the arm's pass share lies outside [0.05, 0.90] for more than half of the gated steps.
- **K5:** BLiMP-fast drops by 2 sigma or more.

**Procedure:**
- No kill or promote before the critic registers the S seed floor (the C3 seeds of the round-1 winner). Until then the screen reports numbers only.
- There is one arm, so there is no best-arm selection.
- Promotion: 3 seeds, size M, official eval, and the FLOP-matched reference per 01:20Z, under R0-R6.

**Mechanism check (reported).** A pass that fails it is relabelled "not the claimed mechanism" and cannot be promoted without a new card.
- The DEV-sub minus TRAIN-sub gap at epoch 10 is at least 0.05 below the base's.
- Novelty tilt is at least 0.9.
- The gain lands on new text and on the 64+ bands.
- The update's stable rank in the cooldown is reported only, since it falls by construction.

**Provisional base.** A kill before the base is frozen reads "killed (provisional base)".

## Odds (guesses)
- **Ceiling: about 0.12 nats/word (extrapolated).** The spread between tuned optimiser families at 0.1B is 1.1-1.4x data, about 0.04-0.13 nats at 0.26 nats per doubling. A filter inside one family should not exceed it. Kim/Kotha's ensemble headroom (about 0.3) is not a single-run bound. Once G0 has run, the ceiling becomes min(0.12, Muon's measured excess train-dev gap over tuned AdamW).
- **Median: about 0.00** (range -0.03 to +0.02). **Expected: about +0.005.**
- **P(pass) ≈ 0.04.** This is P(G0 shows a Muon-specific excess >= 0.05) ≈ 0.3, times P(F0' and F1 pass) ≈ 0.35, times P(screen pass) ≈ 0.4.
- **Most likely failure.** The gate keeps the top 5-20 directions per matrix and shrinks the weak tail. The 1-63 and 64-8k bands regress slightly, DEV lands within ±0.02 of the base, and N3 does as well as the arm.
- **A clean null still says something:** Muon's whitened noise directions are not where our base memorises.

## Fastest falsification (in order; each step can kill)
- **F0, already run: the scalar-edge card is dead.**
  - Probe: the mechanism skeptic's noise_probe.py on the two local S checkpoints, port_S_10M_e10_s0 (WD 0, overfit) and grid_S_100M_e1_s0 (fresh). Verified from its JSON outputs.
  - Pure-noise micro-gradient differences have a top singular value at a median 6.95-7.43x the iid edge (min 2.2-3.4x over the 72 matrices).
  - 11.7-16.1 of 384 pure-noise directions pass at c = 1, and 4.7-5.9 at c = 1.5.
  - One real batch's mean passes 16.5 (c = 1) and 6.2 (c = 1.5) directions, against 15.3 and 5.5 for pure noise.
  - The overfit WD-0 model's DEV-gradient mean has 9.4x the noise energy (median), against 2.1x on its train text. So a generalisation-error gradient is measurable at fixed weights.
- **F0', calibration of the rewrite (CPU, about 1.5 h, scratch script only).**
  - Build a momentum proxy from 39 batch gradients through the real Nesterov recursion. Build a pure-noise momentum from within-batch micro differences through the same recursion. Use a stale basis as in the arm.
  - Pass: some c <= 4 gives a pure-noise pass share of at most 1% (4 of 384) on both checkpoints, **and** at that c the fresh model's momentum proxy passes at least 3x as many directions as pure noise.
  - Fail: the card is dead. Pass: register c.
- **G0, premise (base builder's stage under firewall (e);**
  - Needs a base lever, about 30 LOC: --hidden_optimizer adamw.
  - Tuned AdamW on the hidden matrices (a 2x2 LR x WD grid, at least the attack's trial count) against the Muon winner at S, 10M x 10.
  - Read the DEV-sub minus TRAIN-sub gap at epoch 10 and per-band DEV.
  - Proceed only if Muon's gap exceeds the best AdamW cell's by at least 0.05 nats/word.
  - If AdamW wins DEV outright, the base builder may switch the base, and K1 is then read against the AdamW base.
- **F1, memorisation separation (CPU, 1-2 h).**
  - Runs on chck_70M.pt weights from the ceiling-map C3 seed runs. These are already queued with --save; weights alone are enough.
  - At those weights: a momentum proxy as in F0'; g_dev from 32 DEV-sub micro-batches; m = g_train - g_dev from 32 train micro-batches. Energies come from split-half cross-products, so the noise bias is removed.
  - rho = (share of g_dev energy in the kept subspace) / (share of m's energy in the kept subspace).
  - Kill if the median rho over the 72 matrices is below 1.2, or if rho_gate - rho_topk (magnitude top-k at matched count) has a bootstrap 95% CI over matrices that includes 0.
  - N1's q_l is calibrated here.
- **CPU tiny (mechanics, labels).** Tiny size, train_1M, batch_tokens 8192. Base, arm and N2, 3 seeds each, about 1-2 CPU-h. Checks: c = 0 identity on a full run, and the pass share stays in range.
- **GPU smoke.** 60 steps inside the cooldown at S; step time must be at most 1.15x the base's. About 2 A100-min.
- **GPU mini-ruler, adapted (labels, counted as trials).**
  - Why not the standard 1M x 10 ruler: it has 24 steps per epoch, its 150-step momentum warmup covers 61% of the run, and it cannot test this mechanism.
  - Instead: train_1M x 10 at S with batch_tokens 8192 and micro 2 seqs (K = 4). That gives about 195 steps per epoch and a WD-memory/epoch ratio of 0.39, against 0.32 at the 10M screen.
  - Runs: base, arm, N1, N2; 4 runs, about 8 A100-min.
  - "Killed (proxy)" if the arm is not ahead of both the base and N2.

## Screen (S, 10M x 10, seed 0; two waves on 2 A100s)
| run | variant | note |
|---|---|---|
| base | muon | Same batch, arm code path |
| SG (arm) | snr; c from F0'; start 0.5; R 20 | The only arm |
| N1 | snr_blind | q_l from F1 |
| N2 | snr_replay | Wave 2; replays SG's log |
| N3 | muon; mu 0.975 from step 1218 | |
| base-lr (conditional) | muon; lr_scale 0.25 | Only if s1x has not bracketed the LR |

Cost: 5-6 runs at about 9 A100-min each, about 45-55 A100-min3). Adding the smoke and the mini-ruler gives about 60 A100-min. G0's cost is the base builder's.

## Implementation plan (harness)
- **babylm/moonshot/snr_muon.py (about 200 LOC).**
  - SNRGatedMuon(Muon), subclassing sprint1/harness/optim.py's Muon.
  - kappa(mu).
  - Per-parameter state in self.state[p]: U, V, floor mask, Welford sums of a_{k,i}, a previous-grad copy (fp32, about 85 MB total) and a refresh counter. Keeping it there puts it in resume.pt.
  - observe_micro(), and step() = Muon's update plus the post-filter when active.
  - Modes snr, snr_blind and snr_replay; logging; a replay npz writer and reader.
- **babylm/harness/train.py (about 40 LOC; off by default, so the port path stays bit-identical).**
  - Flags: --optimizer_variant {muon, snr, snr_blind, snr_replay}, --snr_c, --snr_start_frac 0.5, --snr_refresh 20, --snr_q, --snr_replay_from, and a late-momentum flag for N3.
  - After build_optimizer (line ~558), swap the Muon member of CombinedOptimizer for SNRGatedMuon over the same parameter groups.
  - One opt.after_micro() call after each backward in the three micro loops (~lines 717-774).
  - Hard-fail on the OOM-halving path for these variants.
  - Assert bf16 autocast, so no fp16 grad scaling enters the projections.
- **Base lever for G0 (base builder):** --hidden_optimizer adamw.
- **Tests: babylm/tests/test_snr_muon.py (about 180 LOC).**
  - c = 0 is bit-identical to Muon over 20 CPU steps and through a tiny train.py run.
  - kappa matches a Monte Carlo estimate within 5%.
  - iid pure noise: pass share at most 1%.
  - Kronecker noise with a power-law (alpha = 1) activation covariance: pass share at most 2% at the registered c.
  - A planted rank-4 signal at SNR 3: at least 3 of 4 directions recovered.
  - Micro-gradient differencing equals the per-micro gradients.
  - N1 never reads the micro statistics (checked with a poisoned buffer).
  - N2 reproduces SG's k_t exactly.
  - Pause and resume are bit-exact.
- **F0' and F1 probe scripts** extend noise_probe.py in scratch, with no repo change.
- **Effort.** About 1-1.5 agent-days, started only after G0 and F0' pass.

## Stacking
- **Orthogonal in target:** A1, A3, A4 and A5.
- **A2:** its token noise raises gradient noise, so the gate would cut more under A2. Combine only under a new card.
- **fold-consensus-gradients:** overlaps. Fold-consensus uses held-out folds, which is the right target for g_train - g_pop. If both survive, merge them under one card rather than stacking.
- **cross-microbatch-sam:** compatible; it uses the same micro-gradients.
- **Base knobs:** overlaps EMA, weight decay, LR and momentum.

## Skeptic fixes: adopted and not adopted
**Adopted:**
- Novelty set to "known idea", with the full citations and the owed search.
- The premise test runs first, as G0 under firewall (e).
- Runnable falsifications from weights plus a momentum proxy (no stored momentum buffers needed).
- A per-direction noise estimate that handles anisotropy.
- The K-1 estimator on the Nesterov vector, with K pinned.
- No per-step eigh, plus a timing gate.
- No RMS rescale.
- Binding noise-blind and replay nulls, plus the high-momentum comparator; the random null dropped.
- One pre-registered arm, in the cooldown only.
- A band-level kill.
- Ceiling lowered to about 0.12, labelled extrapolated.
- A step-matched mini-ruler.
- The BLaLM sign and the 2605.06654 paraphrase corrected.

**Not adopted as written:**
- An exact-polar arm. It is unnecessary, because c = 0 is bit-identical to Muon.
- The train-dev gap and novelty tilt as kills. They are mechanism checks that block promotion, as on A1 and A4.
- Storing 4 full micro-gradients. Projecting each one as it arrives gives the same per-direction variance with less memory.
- Merging into fold-consensus now. The CPU gates are cheap enough to settle this card on its own first.
