<!-- moonshot design round W1; status alive; ceiling 0.2 nats/word; expected 0.005; P(pass screen) 0.04; screen GPU 46 A100-min. Draft card: not yet submitted to the critic. -->

# Attack: fold-consensus gradients, v2 (fold-recurrence shrinkage of the Muon body step; card, 2026-10-10)

Status: card only. No code and no runs yet. All three skeptics (prior art, rules, mechanism) returned "wounded". None returned "dead", and no rule kills the attack. The v1 design is withdrawn as specified: K=4 hashed 1000-word blocks, beta 0.995/0.98 step-EMA per-fold accumulators, and a rho gate applied *before* unmodified Muon. It had three faults.
- Its cross-epoch memory was 0.995^244 = 0.29 and 0.98^244 = 0.007.
- Its random-fold null shared almost all of its statistic. Paired simulation: cos(arm step, null step) = 0.97-0.997.
- Muon's Frobenius normalisation erases the gate's magnitude (|NS(X) - NS(0.1X)| / |NS(X)| = 1.8e-6).

This v2 adopts the skeptics' fixes. It is a cheap probe of a known family in a new regime, and probably a negative one. It is **not** a 10x lever.

## Known idea, not new (corrected label)
The family is agreement-weighted or SNR-weighted aggregation across data partitions:
- AND-mask (Parascandolo et al. 2020, arXiv 2009.00329) and SAND-mask (Shahtalebi et al. 2021, 2106.02266).
- **Federated Gradient Masked Averaging** (Tenison et al., TMLR 2023, 2201.11986). It uses fixed client partitions kept across rounds, per-client updates accumulated over 1-12 local epochs, and a soft per-coordinate agreement mask. So v1's "new combination" already exists.
- Coherent Gradients (Chatterjee, ICLR 2020, 2002.10657) and M3/RM3 (Zielinski, Krishnan, Chatterjee 2020, 2003.07422): a coordinate-wise median of micro-batch gradients, rolled over steps.
- GAF (Chaubard, Eddy, Kochenderfer 2024, 2412.18052).
- The weight m^2/(m^2 + sigma^2) is Balles & Hennig's variance-adapted step (M-SVAG, ICML 2018, 1705.07774) and per-parameter GSNR (Liu et al., ICLR 2020, 2001.07384).
- Per-worker momentum fed into a robust aggregator: Karimireddy, He, Jaggi (ICML 2021, 2012.10333).
- Fixed hash partitions across repeats in LM pretraining: Goldfish loss (Hans et al. 2024, 2406.10209). Sample splitting: cross-fitting (Chernozhukov et al. 2018). Fixed shards: SISA (Bourtoule et al. 2021).
- Coordinate masks inside optimizers: Cautious optimizers (Liang et al. 2024) and Grokfast (Lee et al. 2024, with the opposite intent). These two IDs are from memory and unverified.

**The measured record is small or negative on natural data:**
- DomainBed averages (Fishr Table 4, Rame et al., ICML 2022, 2109.02934): ERM 68.7, AND-mask 67.5, SAND-mask 67.2. On DomainNet: 41.3 vs 37.9 and 32.2.
- FedGMA on IID splits, the setting closest to hash folds: CIFAR-100 56.80 -> 57.21, TinyImageNet 30.66 -> 30.45, CIFAR-10 with 100 clients 80.42 -> 80.07.
- RM3: a smaller ImageNet generalisation gap but no clean-label test gain, and momentum had to be 0.7 or lower.
- GAF's +18% comes under label noise, which clean LM text does not have.

**Evidence against the premise:**
- LM memorisation is entangled with general capability (Huang, Yang, Potts 2024, 2407.17817; MemSinks, Ghosal, Maini, Raghunathan, ICML 2025, 2507.09937).
- Memorising the long tail helps test accuracy (Feldman 2019, 1906.05271).
- The instruments found no verbatim recall in the gap.

**No LM evidence either way.** No gradient-agreement, fold or consensus paper appears in the BabyLM 2023-2025 proceedings (titles; 2025 abstracts too). The nearest LM work, Invariant Language Modeling (Peyrard et al., EMNLP 2022, 2110.08413), reports OOD robustness, not data efficiency.

**Label: known idea (agreement/SNR-shrunk aggregation across fixed partitions; untested in multi-epoch LM pretraining with Muon).** The skeptics' searches did not find the cross-epoch fold-recurrence statistic below, but they were budget-limited (web search was exhausted). It stays "pending prior-art" and defaults to known.

## Mechanism (v2)
1. **Folds (fixes the "document-local" error).** K = 8 document-level folds, source-stratified. Documents come from BOS boundaries in train_10M.bin and sources from .src.bin. Assignment is greedy LPT by token count within each source, in a seeded hash order. Measured on train_10M (13,713 docs), fold sizes are 1,987,955 to 2,024,508 tokens (max/min 1.018; K=4: 1.006; K=16: 1.023). A Gutenberg book (up to 194k tokens) stays whole in one fold, so document-specific content is fold-specific by construction. Under v1's 1000-word blocks, every book was spread over all folds.
2. **Fold-pure sampler (used by every run in the screen, the base included).**
   - Each fold's documents form their own stream.
   - In each epoch, each fold stream gets its own random offset in [0, T) and is tiled once, following EpochSampler semantics per fold.
   - Windows never straddle folds. v1's EpochSampler gave about 48% fold-mixed windows.
   - A micro-batch holds 8 windows of one fold (micro 8 at S and M, so 8 micro-batches per 64-window step). Micro-batch order is permuted every epoch.
   - Leftovers of fewer than 8 windows per fold carry over.
   - Every token gets exactly one pass per epoch. v1's fixed 16-per-fold design gave unequal folds 9.8-10.2 passes.
3. **Stash.**
   - After each micro-batch backward, the Muon-group gradients go into a step buffer and into that fold's epoch mean S_k(e). The Muon group is the body matrices, 12 x (4*384^2 + 2*384*1536) ≈ 21.2M parameters at S.
   - Embeddings, the tied head and scalars bypass the stash: they use Adam, and rare words live in one fold.
   - The step buffer is copied into .grad before the optimizer step.
   - It handles several micro-batches per fold per step (micro 4 after OOM halving).
   - Memory is about 19 body-sized buffers: S_k(e) and S_k(e-1) for 8 folds, the step buffer, and the two EMA statistics. That is ≈ 1.6 GB at S (fp32) and ≈ 3.2 GB at M (bf16). v1 budgeted 480 MB, which covered G only.
4. **Statistic, at the end of each epoch e >= 2, per coordinate of each Muon matrix.**
   - d_k(e) = S_k(e) - mean_j S_j(e).
   - Persistent fold variance: v(e) = clip((1/(K-1)) * sum_k d_k(e) ⊙ d_k(e-1), 0).
   - Signal: mu^2(e) = clip(Sbar(e) ⊙ Sbar(e-1) - v(e)/K, 0).
   - v and mu^2 are EMA'd over epochs (weight 0.5 on the old value). The horizon is defined in epochs, not steps, so no beta needs rescaling across corpus sizes.
   - Gate: w = mu^2 / (mu^2 + v/K + eps). This is the Wiener weight for the population gradient when the training set's own idiosyncratic part has fold-persistent variance v/K.
   - With fixed folds, the cross-epoch product estimates only variance that recurs in the same documents epoch after epoch, i.e. memorisation drift. Transient within-epoch noise averages out.
   - With mismatched fold pairing (the null), the expectation is -sigma_p^2/(K-1), so the information is removed.
5. **Applied after Newton-Schulz (fixes the Muon erasure).**
   - GatedMuon, a subclass of sprint1/harness/optim.py Muon, computes o' = w ⊙ NS(u) and does not renormalise, so shrinkage survives orthogonalisation. Decoupled WD is unchanged.
   - The gate is frozen within an epoch and first applied in epoch 3.
   - Epochs 1-2 are therefore identical to the stratified base (same seed, same data order).
6. **One arm, pre-declared for promotion: C8** (K=8, per coordinate, post-NS, epoch EMA 0.5).
   - Diagnostic arms are labelled, never promoted, and each counts as a trial:
     - **C8-gsnr**, the known-idea comparator. Same folds, but v is replaced by the within-epoch fold variance (1/(K-1)) * sum_k d_k(e)^2: FedGMA/GSNR across fixed folds, without recurrence.
     - **C8-adam**: the gated matrices on AdamW. It separates "Muon erases it" from "the statistic is blind".
   - Withdrawn from v1:
     - beta = 0.98: its cross-epoch weight was 0.007, so it was a second null.
     - The MLP-only scope: unfunded.
     - The pre-NS rho gate.

## Why (which measured slice)
**Target: the variance part of bottleneck 1** (published pair: an extra 0.61 nats/word on new text only; the 64-8k band is 0.36).

**On our round-1 winner** (base_S_wd0p4_lr0p5_r1_ve0_tied @ ref-0065, full DEV 5.0367, minimum at e10):
- DEV-sub falls every epoch, 5.90 -> 5.12.
- TRAIN-sub falls 5.17 -> 3.83, so the gap grows from 0.72 to 1.29.
- Epochs 7-10 buy 0.31 on DEV-sub against 0.68 on TRAIN-sub.

**Hypothesis:** part of each late body update fits idiosyncrasies that persist within documents. Shrinking the coordinates where that drift dominates should keep the DEV gain and drop part of the train-only gain.

**Bound:** a filter on gradients computed from 10M words adds no contexts and no targets. It can only reduce variance. Its reach is therefore at most the posterior channel (C3 registered band g_inf 0.12-0.30, not yet measured), minus what tuned WD 0.4 in the base already removes. A4's reading caps it further: ensembles gain only 0.06-0.09/word on the 64+ bands, so bottleneck 1 is mostly bias.

**Where any gain should land:** new text, on words seen 64-8k times.

**Pre-registered sign <= 0** on the 1-63 band and on bursty words. Bursty words are document-concentrated, hence fold-concentrated, hence shrunk; Feldman's result points the same way. Bursty words are dropped from the targets (v1 claimed them).

**Signature:** any regulariser raises the loss on train-like trigrams relative to novel ones, so novelty tilt alone cannot show "data-like". The arm's rare tilt and seen-text change are compared with the existing WD-increase cell (base_S_wd1p0_lr0p5_r1_ve0_tied, round 1, no extra run). If they match, the arm is labelled "regulariser-like, not data-like".

## Ceiling and odds (guesses, stated plainly)
**Ceiling: 0.20 nats/word** (the skeptics give 0.15, 0.18 and 0.25). It is set by the variance-channel bound above.

v1's 0.92 is withdrawn, for five reasons:
- Under Muon there is no fixed point: every step has unit spectral scale.
- A filter adds no information.
- train_100M contains train_10M, so the "fold-invariant 100M solution" also fitted these folds.
- The +0.094 (100M x10 vs x1) was double-counted.
- The 0.6 excess gap is GPT-BERT's, not our base's.

**Median 0.00** (sign uncertain), **expected +0.005**, **P(pass screen) 0.04**, P(gain >= 0.05) about 0.01.

Basis for the low odds:
- FedGMA's IID results and the DomainBed negatives.
- Per-coordinate statistics at K=8 are noisy: the relative SD of v is about sqrt(2/7) ≈ 0.53 per epoch before the EMA.
- Memorisation is dense and entangled, not aligned with coordinates.
- The effect overlaps tuned WD.

A null result is still informative: fold-persistent gradient drift either does not exist at 10M x 10 under Muon, or cannot be separated from population signal per coordinate.

## Null twins
All twins use the same fold-pure sampler, seed, data order, micro 8, stash, compute and exposure.
- **N-shuf (random-fold null without a data-order confound).** The statistic pairs each fold at epoch e with a different fold at e-1, using a seeded random derangement each epoch. This is the skeptics' "fold k at e with fold j at e-1" term. Fold persistence is removed, while mu^2 (gradient-magnitude persistence) stays informative. Its scores are mapped, per matrix and per epoch, onto the quantiles of the matched-fold gate computed in the same run on its own trajectory, so its gate distribution is identical to the arm's. This replaces v1's fold = hash(window, epoch). That null re-packs windows every epoch, which would add a seed-sized data-order confound.
- **N-perm.** The arm's own statistic on fixed folds, with the gate tensor randomly permuted within each matrix every epoch. It has the same gate distribution and the same mean effective body LR, with the coordinate information removed. It covers both the gradient-dropout reading and the "gate = lower body LR" confound. The winner sits at lr_scale 0.5, the edge of the grid, so the LR-schedule null is subsumed here.
- **Bases.**
  - **B-strat:** fold-pure sampler, micro 8, gate off, with shadow logging.
  - **B-frozen:** the round-1 winner as run (5.0367, seed 0, EpochSampler, micro 16).
  - The binding comparator is the better of the two (A4 amendment 1 analogue).

## Kill rule (registered before any run)
**Setup:**
- Size S, the round-1 winner recipe (muon_wd 0.4, lr_scale 0.5, ve0, tied, softcap 15, unet) @ ref-0065.
- Seed 0, train_10M x 10 epochs.
- Metric: full-DEV nats/word from dev_bpw.py (collapse protocol) on the final checkpoint, with no checkpoint picking.
- sigma is the critic's registered seed floor. Until it is registered, results are numbers only, with no kill or promote.

**Main kill:** kill if C8 is not ahead of **each** of B-strat, B-frozen, N-shuf and N-perm by >= 2 sigma.

**Also kill if any of these hold:**
- **(a) Shadow kill, before any arm runs (from B-strat's shadow log).** The shadow computes w_arm and w_shuf from the same sums every epoch, with no effect on the weights. Kill if any of the following:
  - Over Muon matrices and epochs 3-10, the median Δz < 0.05. Here z is the matrix-level correlation of d_k(e) with d_k(e-1), and Δz is matched minus deranged pairing.
  - The step barely moves: median ||w ⊙ O - O|| / ||O|| < 0.05.
  - The arm and N-shuf take the same step: median cos(w_arm ⊙ O, w_shuf ⊙ O) >= 0.95.

  These replace v1's "mean gate < 0.2 means starved". Under Muon that measured nothing, and at K=4 pure noise already gave a mean gate of 0.138.
- **(b)** The 1-63 band loses > 0.05 nats/word against the better base, or the bursty slice (instruments dispersion) loses > 0.05.
- **(c)** BLiMP-fast drops by >= 2 sigma.
- **(d)** FLOPs exceed 1.25x without a FLOP-matched reference. FLOPs are judged per the A3 condition; tokens/s is reported.

**Reported every epoch:** the DEV-train gap together with TRAIN-sub. A gap reduction that comes with no DEV gain is labelled "regulariser-only".

**Provisional-base clause (ref-0086 b).** This is a regulariser-type attack. A later base change in WD, EMA, dropout or MNTP, or a > 2x change in LR, forces a re-screen. Until then a kill reads "killed (provisional base)".

**Promotion.** Promote only C8, chosen before seeds 1-2. Then 3 seeds, size M and official eval. The FLOP-matched reference (ref-0265) applies at promotion only. Every GPU and proxy run is recorded in the trial log (R3 parity).

## Fastest falsification (cheapest first; stop at the first kill)
0. **CPU unit tests** (below).
1. **CPU power test, ≈25 CPU-minutes.**
   - Measured on this container: the tiny model trains at ≈5.5 s per 8,192-token step (≈1,500 tokens/s). v1's "train_1M x 30 epochs in 20 minutes" would take about 9 h per run.
   - Setup: tiny model (2L/64d, T=128, tied); a 200k-token prefix of train_1M (hash committed before use, per the ref-0110 proxy rules); K=8 chunk folds (train_1M docs are chunks of at most 2,049 tokens); gate off, shadow statistics on.
   - 5% canary windows: fixed random 64-token continuations, each pinned to one fold and recurring every epoch. At each epoch end, one backward with no optimizer step gives the canary gradient g_can.
   - Pass requires Δz >= 0.05 and Spearman(1-w_matched, |g_can|) - Spearman(1-w_deranged, |g_can|) >= 0.10.
   - This is a detection-power check only. A setup like this is biased toward passing, so it is never a pass on loss.
2. **CPU natural-text check, ≈1.5-2 CPU-hours (free).** The same tiny setup without canaries, with B-strat, C8, N-shuf and N-perm. Go on only if C8 beats both nulls on dev_sub by >= 0.01 nats/token. Labelled "CPU proxy".
3. **GPU shadow, ≈10 A100-minutes.** This is the screen's own B-strat run at S/10M, with shadow arm and N-shuf statistics. Apply kill (a).
4. **GPU mini-ruler, ≈6 A100-minutes (proxy label).** train_1M x 10 at S: B-strat at seeds 0 and 1 (for the proxy floor), plus C8, N-shuf and N-perm. The statistic is defined per epoch, which fixes v1's 1M-vs-10M regime mismatch (16.7x vs 1.84x). Kill if C8 minus the best of {B-strat, N-shuf, N-perm} is below max(0.03, 2x the proxy floor) nats/word on full DEV.
5. **S/10M screen, ≈30 A100-minutes.** C8, N-shuf and N-perm, plus B-strat from step 3 and B-frozen (an existing run). The arms may branch from B-strat's resume checkpoint at 20M words, with the full state including the stash: epochs 1-2 are identical and the comparison is paired. **Total ≈46 A100-minutes**

## Exposure ledger (R0)
| item | words |
|---|---|
| unique | 9,953,773 (train_10M only) |
| per epoch | each fold stream tiled once. At most 1 window per fold is lost to the offset or tail (≤ 8 x 1,024 tokens ≈ 5.1k words, 0.05%), carried over as in EpochSampler |
| 10 epochs | ≈ 99.5M ≤ 100M, asserted by the word counter. Per-fold pass counts are logged and asserted (10 each) |
| teacher, generator, second model, extra forward | none. The stash and statistics reuse gradients already computed |
| CPU canary backward | no update. Diagnostic run only, never a screen |
| N-shuf, N-perm, B-strat | the same ≈ 99.5M each, with the identical sampler and data order |

R5: no text is added. The HF export holds model weights only (the stash is excluded), so causal scoring is unchanged.

## FLOPs (R4.4)
Per step, the extras are 8 gradient adds plus one copy over 21.2M parameters (≈2e8) and one post-NS multiply. At epoch end, about 10 elementwise passes over the buffers (≈4e9 per epoch). Against ≈1.2-1.4e13 training FLOPs per step, the total is **≈1.00x**, so no FLOP-matched reference is needed for the screen. Wall-clock: micro 8 instead of 16 may cost 5-10% on every screen run. It is measured and reported against B-frozen.

## Implementation plan (harness, ≈2 agent-days)
- **New babylm/moonshot/consensus.py (≈300 LOC):**
  - `fold_plan` (BOS docs, sources, source-stratified LPT, seed).
  - `FoldStreamSampler` (per-fold offsets, fold-pure micro-batches, carry-over, state_dict, per-fold word counters).
  - `FoldStash` (per-fold epoch means and step buffer; fp32 at S, bf16 at M; several micro-batches per fold).
  - `ConsensusGate` (epoch-end d_k, v, mu^2, EMA, w; modes off|shadow|arm|shuf|perm|gsnr; quantile mapping; a per-epoch, per-matrix log of w quantiles, z_matched, z_deranged, the step-change ratio and the cosine to the null).
  - `GatedMuon` (with gate None it is bit-identical to Muon.step).
- **babylm/harness/train.py (≈40 LOC):**
  - Flags: --consensus MODE, --consensus_k 8, --consensus_fold_seed, --consensus_start_epoch 2, --consensus_ema 0.5, and --consensus_canary (CPU diagnostic only).
  - When the mode is not off: use the FoldStreamSampler with micro 8; after each micro backward, call stash.add(fold) and zero the grads; copy the step buffer into .grad before the step; call gate.update at each epoch boundary.
  - With the mode off, the port path is untouched.
  - The mode is asserted incompatible with PlanSampler and MNTP rows.
- **Tests, babylm/tests/test_consensus.py (≈200 LOC):**
  - The fold plan is deterministic and source-stratified, with balance ≤ 1.02 at K=8.
  - Every window is fold-pure.
  - Per-epoch words equal EpochSampler's within K windows.
  - Stash sums equal per-fold reference gradients.
  - A planted fold-persistent component gives v > 0 and w < 1 on its coordinates; deranged pairing gives E[v] ≈ 0.
  - A constant gate c scales the Muon update exactly by c (post-NS semantics).
  - A gate of 1 is bit-identical to the native Muon step.
  - Quantile mapping reproduces the reference quantiles.
  - The gate-off fold-pure path matches native accumulation to fp32 tolerance.
  - Resume is bit-exact (sampler, stash, gate, RNG).
  - Flag off is bit-identical to the port.
- **Jobs and outputs:** a JSON job list for run_jobs.py. Outputs go to shared/babylm/attacks/fold_consensus/&lt;run&gt;/ with curves CSV + PNG and consensus_log.csv.

## Stacks with
It acts on which gradient components are applied, not on what the gradient is computed from. Even so, it is a regulariser: per the critic's note, regularisers do not add. A combination with A1, A2 or tuned WD needs its own card. It partly substitutes for any spike-gated or cautious-Muon idea. It is orthogonal to A4 (new contexts) in mechanism.

## Skeptic fixes: adopted / not adopted
**Adopted:**
- The relabel and the full prior-art line (FedGMA, M-SVAG/GSNR, Karimireddy, RM3, Goldfish, the DomainBed and IID negatives).
- The recurrence statistic, with the horizon defined in epochs.
- Document folds, the fold-pure sampler and balanced passes.
- Post-NS gating.
- Distribution-matched N-shuf plus N-perm.
- The frozen-base comparator.
- One pre-declared arm; beta 0.98 and the MLP-only arm dropped.
- Muon-valid kill checks and the shadow kill.
- Bursty words dropped, with harm pre-registered.
- The ceiling rewritten to the variance bound.
- The natural-text CPU check before any GPU time.
- The regulariser re-screen clause.
- The corrected memory budget and OOM handling.

**Not adopted:**
- K=16 (needs 4-window micro-batches; a later card if K=8's shadow shows signal).
- Singular-direction gating (deferred; C8-adam diagnoses whether NS is the problem).
- A full per-step AND-mask arm (C8-gsnr is the known-idea comparator at the same horizon).
