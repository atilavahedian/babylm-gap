<!-- moonshot design round W3; status alive; ceiling 0.12 nats/word; expected 0.01; P(pass screen) 0.08; screen GPU 40 A100-min. Draft card: not yet submitted to the critic. -->

# Attack C-avg: cycle-and-average, a snapshot ensemble from warm-restart cycles inside one exposure (card v1, 2026-10-10)

Status: card only. No code and no runs. All three skeptics (prior art, rules, mechanism) rated it **wounded**; none rated it dead, and no rule kills it. Their fixes are folded in below. The odds are low and the test is cheap. Run it only if the free Stage 0 headroom gate passes. It waits for critic acceptance, the floor seeds (s1, s2) and the registered floor.

**A known idea, close to a reproduction.** The closest prior art is **q0 primitive 1** (Mandal, Berman, Vegesna, Dahal, arXiv 2606.03938, §2.1 and Table 3, as read by the prior-art skeptic). It uses a cyclic LR (plus WD anti-correlated with LR), probability-averages the cycle-end members, and trains with Muon+AdamW at peak WD 1.3 on 100M FineWeb tokens over many epochs with 2-epoch cycles. Its small-budget result is the main counter-evidence. At E = 4, 2-epoch cycles score 3.360, one normal model 3.365, and 4 snapshots from a normal run's last epoch 3.326. At E = 8, 4 cycles only "match" end-of-run snapshots. It also needed extra diversity levers: cycle-boundary weight noise, parallel trajectories and anti-correlated WD.

Same family:
- Snapshot ensembles: Huang et al. 2017, arXiv 1704.00109, built on SGDR (Loshchilov & Hutter, arXiv 1608.03983).
- FGE: Garipov et al. 2018, arXiv 1802.10026.
- Cyclical SG-MCMC: Zhang et al. 2020, arXiv 1902.03932.
- Checkpoint ensembles: Chen, Lundberg & Lee 2017, arXiv 1710.03282. For NMT: Sennrich et al. WMT17, arXiv 1708.00726; Denkowski & Neubig 2017, arXiv 1706.09733.
- Multi-epoch LSTM LMs (Mu et al. 2018, arXiv 1812.01216) compared cyclic snapshots with monotone checkpoints taken at the same epochs. The increment was 0.005-0.03 nats/token.
- Weight averaging: SWA (Izmailov et al. 2018, arXiv 1803.05407) and LAWA (Kaddour 2022, arXiv 2209.14981).

Limits on the diversity of one trajectory:
- Fort, Hu & Lakshminarayanan 2019 (arXiv 1912.02757).
- Ashukha et al. 2020 (arXiv 2002.06470): FGE is worth 1.2-1.5 deep-ensemble members; SSE reaches 2.2 only with 45-epoch cycles.
- Wang et al. 2021 (arXiv 2110.00959): checkpoint ensembles share errors, and SSE loses to a single model on ImageNet EfficientNet-B3.
- Frankle et al. 2020 (arXiv 1912.05671): runs become linearly mode-stable early in training.

Schedule penalty: Wen et al. 2024 (arXiv 2410.05192) find cyclic-cosine 0.04 nats/token behind WSD-S. Gupta et al. 2023 (arXiv 2308.04014) find that re-warming raises loss.

BabyLM: FORGETTER (2025.babylm-main.7) used LR restarts plus AdamW resets on 10M, single model, val 3.046 to 3.005, 1 seed. No BabyLM snapshot or checkpoint ensemble paper was found in 2023-2025.

Kim/Kotha 2025 (arXiv 2509.14786): ensembles beat the regularised single-model asymptote, and ensemble-optimal members use about half the WD and twice the epochs.

In-house: the GPT-BERT epoch-5/10 snapshot mix (+0.048), V8 and the C3 bands.

**Withdrawn:** the candidate's claim "untested in multi-epoch LM pretraining under an exposure cap". Only the BabyLM strict-small scale is untested: 21.2M non-embedding params and 10M words. The follow-on (distilling the snapshot predictive into one model) is also known: Snapshot Distillation (Yang et al. CVPR 2019, arXiv 1812.00123), q0 chain distillation and A1. It is not part of this card.

## Base for the screen
The round-1 winner is base_S_wd0p4_lr0p5_r1_ve0_tied @ ref-0065 (critic ref-0423).
- Full DEV 5.0367 nats/word (dev_bpw); harness 5.0406.
- E* = 10, the track cap; DEV minimum at epoch 10.
- Muon WD 0.4, lr_scale 0.5, tied embeddings, no value embeddings, EMA 0, MNTP 0.
- WSD schedule: peak LR for the first half of the steps, then a linear cooldown to 0.1x.
- 2435 steps (about 243.5 per epoch) at 65,536 tokens per batch; about 7.3 A100-minutes of training.

This is a regulariser-type attack (ref-0086(b)). Any later base change to WD, EMA or MNTP forces a re-screen. Until then a kill reads "killed (provisional base)".

## Zero-training read on this base (done for this card; DEV-sub, word-level mixture, labels only)
These come from the winner's DEV-sub per-word dumps (pw_dev_*.npy, 993,244 words), using an equal-weight probability mixture. The final model scores 5.121 on DEV-sub.

| Checkpoints mixed | Gain over the final model (nats/word) |
|---|---|
| {60, 80, 100M} | +0.044 |
| **{80, 90, 100M}** | **+0.054** |
| {70, 80, 90, 100M} | +0.069 |
| {90, 100M} | +0.030 |
| Tuned two-way mix (fit on the first half, scored on the second) | +0.041 to +0.058 |

- On TRAIN-sub, the {80, 90, 100M} mix loses 0.068. The earlier cooldown states memorised less, so much of the gain corrects overconfidence and is not proven mode diversity.
- On the other round-1 cells (wd 1.0, lr 1, all-on), {80, 90, 100M} gains only +0.011 to +0.030. The skeptics' read on the wd 2.0 cell used peak-LR members (e2 to e5) and lost 0.06 to 0.105 at equal weight.

**Consequence:** the winner's ordinary end-of-run checkpoints already give about +0.05 at 3x inference with no schedule change. The mechanism must beat that by 2x the floor. The screen re-measures all of this on full DEV, per token.

## Mechanism (registered)
Everything matches the base (model, data, seed, data order, steps, batch, optimiser) except the LR schedule after epoch 4 and one matched WD scalar.
- **Epochs 0-4 (steps 0-974):** identical to the base (peak LR).
- **K = 3 cycles of 2 epochs** (steps 974-1461, 1461-1948 and 1948-2435, rounded as in lr_factor).
  - Each cycle starts at 1.0x peak LR and decays linearly to the base's floor of 0.1x by its last step.
  - Restarts are instant, SGDR-style. Muon momentum and AdamW moments are kept.
  - Cycle 1 continues from peak with no jump. The restarts at epochs 6 and 8 are the perturbations.
- **Snapshots:** raw weights at the last step of each cycle (chck_step{1461,1948,2435}.pt, with words_seen). The base has no EMA, so there is no EMA member, and no member averages across a restart.
- **Predictor** (fixed before the run): the equal-weight token-level probability mean of the three snapshots, p(x_t) = (1/3) Σ_k p_k(x_t | x_<t), using each member's soft-capped logits. No fitted weights and no subset choice.
- **WD matched.** The arm's mean LR factor is 0.4 + 0.6 × 0.55 = 0.73, against the base's 0.5 + 0.5 × 0.55 = 0.775. Muon WD is scaled by 1.0616 (0.4 to 0.4247), so ∫lr·wd equals the base's. The code asserts and logs this. It removes the decay confound, which was 35-65% in the candidate's cosine design.
- **Anneal floor = 0.1, as in the base, not 0.** This removes the anneal-to-zero confound by construction.

| Arm | Schedule | Extra | Note |
|---|---|---|---|
| **C-a** | 3 × 2-epoch cycles from epoch 4, restart at 1.0x peak | none | plain warm-restart snapshot ensemble |
| **C-b** | same | Gaussian weight noise at each restart (epochs 6 and 8) on Muon matrices: N(0, (σ_k · std W)²), σ = 0.25 then 0.05, as in q0 §2.1, from its own RNG seeded by (seed, k) | q0's diversity lever (known); counted as a trial |

Dropped from the candidate:
- **The 0.5x-restart arm.** Huang 2017 found that a larger restart LR gives more diversity.
- **4 cosine cycles from epoch 1.** The early members would see 3.25 epochs or fewer at high LR, and on a WSD base the whole DEV gain comes during the cooldown.
- **The within-cycle EMA member.**
- **Log-mean and subset readouts as candidates.** They are now diagnostics only.
- **FORGETTER resets.**

Hypothesis in bits: a restart to peak LR followed by a full anneal ends at a different point of the basin. The cycle ends then make partly independent errors on new text, and each is fully annealed. The base's monotone checkpoints at 60M and 70M sit at 0.82x and 0.64x of peak LR.

Caveat from the mechanism skeptic: Muon's decoupled decay at peak is about 1.3% per step (lr 0.0325 × wd 0.4), a memory of about 77 steps. The stable phase therefore already re-randomises the matrix weights. What persists is the basin and the tied embedding, whose WD group is 0. Restarts can add diversity only by moving that.

## Why (which measured slice)
**Target: the posterior (variance) slice.**
- The C3 registered band is g_inf 0.12-0.30 nats/word.
- Baby Llama's 4-seed ensemble gave about 0.18 nats/word at 10M (a different tokenizer, converted).
- Recovering this slice would take K independent trainings, which is K x exposure and out of track.

**Overlaps:** A1 (the same slice, via distillation), V8 and C3. It does not overlap A2-A4. Critic 23:45Z(2)(b): inference-time ensemble headroom overlaps bottleneck 1 and is not additive. It is not a measured slice of the 10M-to-100M gap, because 100M models also gain from ensembling.

**What the base does not already capture:** the base has no restarts and no EMA. Heavy WD reduces seed variance without removing it (Kim/Kotha). But the base's own cooldown checkpoints already capture about +0.05, as measured above. The claim is therefore narrow: restarts plus full anneals must add diversity beyond what the cooldown checkpoints already have.

## Null twins (all from one base re-run in the same batch)
**The base re-run** uses the same code path with --lr_cycles 0 (WSD, bit-identical to the port path on CPU), the same seed, data order and GPU type. It saves raw checkpoints at steps 1461, 1948, 2191 and 2435 (epochs 6, 8, 9 and 10).

- **N-chk (same marks as the arm):** equal-weight mixture of the base at steps {1461, 1948, 2435}. It removes the restarts and keeps the same inference cost, exposure and code path.
- **N-end (q0's winning small-budget primitive):** equal-weight mixture of the base at steps {1948, 2191, 2435}. Its DEV-sub label is +0.054.
- **N-cyc (inside the arm):** the arm's own final snapshot. It measures the schedule penalty and makes the averaging carry the gain.

**Binding comparator = the best of {base final, N-chk, N-end}** on full DEV. Picking the best null biases the test against the arm. This is accepted, as with A1's best of four nulls.

Also dropped:
- **The candidate's N-chk at epochs 3.25/5.5/7.75/10.** Those members were at peak LR and lose 0.06-0.105 at equal weight (two skeptics measured this), so it was a straw man.
- **Base with lr_floor 0, and a WD-matched control arm.** The anneal and WD confounds are removed by construction.
- **N-lowrestart.** It is weaker than N-end and removes the same information.

## Kill rule (registered before any run)
**Setup:** size S, train_10M × 10 epochs, seed 0, the round-1 winner recipe, with the base re-run in the same batch.

**Instrument:** dev_bpw.py full DEV (ctx 512, stride 256, whitespace collapse), with per-token dumps for every member, arm and null (same-instrument rule, critic 03:16Z). Ensembles are formed offline from those dumps. nats/word = summed token nats / DEV words, exactly as dev_bpw computes it. No checkpoint picking: members are fixed by step above.

F = the registered seed floor (full-DEV nats/word). The best arm is the lower ensemble DEV of C-a and C-b; ties within 0.005 go to C-a.

1. **Kill** if G_tot = DEV(base final) − DEV(arm ensemble) < 2F.
2. **Kill** if min(DEV(N-chk), DEV(N-end)) − DEV(arm ensemble) < 2F.
3. **Kill** if G_ens = DEV(arm final snapshot) − DEV(arm ensemble) < 2F.
4. **Kill** if ρ < 0.5. ρ measures quality-matched diversity: ρ = [min(L2, L3) − L(mix of the cycle-2 and cycle-3 snapshots)] / [min(L_s0, L_s1) − L(mix of seeds 0 and 1)]. The seeds are the winner run and floor seed s1, read from dev_bpw full-DEV dumps. ρ_chk (the base pair at steps 1948 and 2435) is reported beside it.
   - This replaces pairwise KL, which needs full-vocabulary dumps and grows with quality gaps. On the winner, the {50M, 100M} pair has a Jensen gap of 0.21, yet the mix gains only 0.005.
5. **Kill** on a drop of 2σ or more on BLiMP-fast, for the ensemble (through the HF wrapper) or for the final snapshot, against the base.

**Process:**
- No verdict before F is registered. Until then results are numbers only, labelled "floor not registered".
- Exactly one arm is promoted, picked on DEV before seeds 1-2. The 3-seed verdict re-uses no screen numbers.
- Both arms count as trials (R3). The base is owed the same number of trials; its cooldown_frac and lr_floor can use them.

**Reported, not kill criteria:**
- Per-member DEV for the arm and the base.
- Mixture variants: weights fitted on even DEV documents and scored on odd ones; log-mean mixture; the K = 2 last-two subset; SWA of the three snapshots (1x inference, one extra dump).
- Diversity and memorisation: the Jensen gap, and each mixture's TRAIN-sub effect (the memorisation-correction check).
- Lever signature (rare and novelty tilts) of G_tot, beside a 2-seed ensemble's (s0 + s1). A novelty tilt of 1 or more is expected from variance reduction alone and is not evidence of data-equivalence.
- The ∫lr·wd ratio and tokens/s.

**Registered predictions (full DEV, nats/word):**

| Quantity | Prediction [range] |
|---|---|
| Arm final snapshot minus base final (schedule penalty: three 2-epoch anneals vs one 5-epoch anneal) | +0.03 [−0.01, +0.07] |
| G_ens | 0.08 [0.03, 0.12] |
| G_tot | +0.05 [−0.02, +0.09] |
| N-end gain over base final | +0.05 [+0.03, +0.07] |
| Arm minus best null | 0.00 [−0.05, +0.04] |
| ρ | 0.3 [0.15, 0.6] |
| C-b vs C-a | C-b better by 0.01 [−0.03, +0.04] |

The most likely outcome is a kill on rule 2.

## Odds (guesses, stated plainly)
**Ceiling.** If everything goes right, the restarts give seed-like members at K = 3 with no schedule penalty: g(3) = (2/3) g_inf = 0.08-0.20 across the C3 band, about 0.14 at its midpoint. The literature caps one-trajectory members at about 2 deep-ensemble equivalents (Huang 2017; Ashukha 2020, where FGE-like short cycles reach 1.2-1.5). That gives at most 0.5 g_inf.
- **Ceiling about 0.12 nats/word over the base final**, about 1.4x data by the log-linear rate.
- About 0.05 of that is already reachable by N-end, so the ceiling for this card's own increment is about 0.07.
- This is not a 20x channel. It is a stackable posterior-slice increment at 3x inference.

**Expected.** Gain over the base final is about +0.05, almost all of it reachable by N-end. The increment over the binding null is centred at 0.00 (spread about 0.025). Its mean, truncated at zero, is about 0.01.

**Probabilities.** P(pass the screen) ≈ 0.08. P(≥ 0.05 over the binding null) ≈ 0.02.

Even a pass is expected to end "partial" at promotion, because GLUE uses only the final snapshot.

## Exposure ledger (R0)
| Item | Words |
|---|---|
| C-a or C-b training (one model, one forward per window, 10 epochs of train_10M) | 99,541,960 (the base run's own counter) |
| Snapshots | 0: they are states of the same run, not separately trained models. ref-0110's K-times rule covers K models each fed the stream. The rules skeptic confirms no R0 kill. |
| Weight noise (C-b) | 0 |
| Teacher, generator, synthetic text | none |
| Test-time ensemble | inference only, 3x, reported |
| **Total** | **≤ 100M; unique text = train_10M** |

- Tokenizer: the committed train_10M BPE 16k.
- Leakage audit: identical to the base's, since no new text is fed.
- R0.5: the milestone checkpoints are the raw weights at each word milestone, as for every run. The ensemble is defined only at the 100M mark. The cycle-end saves are extra step-indexed checkpoints. All of this is declared in the export meta.

## FLOPs and the R4.4 plan
- **Training: 1.00x the base.** Steps, tokens and model are the same. The schedule and one scalar add no matmuls, and C-b adds one randn per matrix at two restarts. Measured tokens/s are reported for the 1.25x rule.
- **E* = 10** (ref-0423), so the arm runs at the base's E*. That is the same-FLOPs route allowed at ref-0265.
- **Inference: 3x**, which is allowed and reported.
- **Ruling requested:** does 18:43Z(c) ("an ensemble or teacher counts K x FLOPs under R4.4") cover a snapshot ensemble from one training run? The card's position is no, because R4.4 counts training FLOPs.
  - If the ruling is K x, promotion first needs the ref-0265 FLOP-matched reference at the 3x level.
  - Either way, the promotion report also compares against the tuned size-M base, whose per-token compute is roughly that of the 3-member S ensemble. If the arm beats S but not M, it is labelled "inference compute, not data efficiency", which is not a win.

## Official evaluation
- **Zero-shot tasks** (BLiMP, supplement, EWoK, entity tracking, COMPS): an HF wrapper whose CausalLM returns log(mean_k softmax(capped logits_k)) as logits. This is the same probability mixture as the DEV mixer; plain logit averaging would give the geometric mixture instead. A unit test checks that the wrapper's sentence log-probs equal the mixer's. The critic audits it like the lastpool adapter (20:52Z).
- **AutoModel and GLUE fine-tuning:** the final cyclic snapshot, declared in the export meta before any score (ref-0058). It carries the schedule penalty and none of the averaging. GLUE7 is a third of the reliable composite, so the expected outcome is "partial" (18:46Z(3)) unless a later, known-idea distillation card folds the ensemble into one model.

## Fastest falsification
**Stage 0 (free, when floor seeds s1 and s2 land).** Everything is measured on the same DEV-sub per-word instrument as the read above.
- Seed-pair gain: G2 = mean(L_s0, L_s1) − L(50/50 mix of the s0 and s1 finals).
- Seed-like upper bound for a 3-member arm: U = (4/3) G2, from g(K) = g_inf (1 − 1/K).
- N-end3 is measured on s0 and on s1, and the two are averaged.
- **Close the card with no GPU run if U − N-end3 < 2F.** In that case even a seed-diverse arm with no schedule penalty could not clear N-end. The label is "closed (stage-0 headroom, DEV-sub)", an owner close, not a screen verdict.

**Stage 1**
- C-a, C-b and the base re-run at size S, sharing one A100 under MPS (about 7.5 minutes of training each).
- 12 full-DEV dev_bpw dumps at about 1 minute each: 3 + 3 arm snapshots, 4 base checkpoints, the s1 final, and the C-a SWA.
- BLiMP-fast for the two ensembles and the base.
- Every readout runs on CPU afterwards.

**Never on the mini-ruler or on CPU-tiny models.** At 1M × 10, a 2-epoch cycle is about 49 steps, shorter than the 150-step Muon momentum warmup.

## Implementation plan (harness)
1. **Schedule** (`babylm/harness/levers.py`, `train.py`):
   - Flags `--lr_cycles K` (0 = off, the existing lr_factor path, bit-identical), `--cycle_start_frac 0.4` and `--cycle_restart 1.0`.
   - `cyclic_lr_factor(step, total, ...)` sits beside lr_factor; each cycle decays linearly to `--lr_floor`.
   - `--cycle_wd_match` scales every nonzero decoupled WD group by Σ_t f_base(t) / Σ_t f_cyc(t). It is logged and asserted.
2. **Step-indexed saves:** `--save_at_steps` writes `chck_step{n}.pt` (with words_seen) after the optimizer step, beside the word milestones. The base re-run gets 1461, 1948, 2191 and 2435 from the same rounding function, with an assert that they equal the arm's cycle ends.
3. **Weight noise (C-b only):** `--cycle_noise 0.25,0.05` adds noise to Muon matrices at each restart step from a dedicated torch.Generator seeded by (seed, k), so the data stream stays untouched. Each injection is logged in log.jsonl.
4. **Per-token dumps:** `babylm/eval/dev_bpw.py --dump PATH` writes per-token nats (float32, stream order) and a token-to-word index. An assert checks that the dump sums to the unchanged headline.
5. **Mixer:** new `babylm/moonshot/snapshot_avg.py`.
   - Takes registered member lists and forms the equal-weight mixture as logsumexp(−nats) − log K, with nats/word computed exactly as in dev_bpw.
   - Outputs the per-member and ensemble DEV, G_tot, G_ens, the arm-minus-best-null increment, ρ and ρ_chk, plus every reported label from the kill-rule section.
   - Computes the signature via babylm.instruments.signature, beside the s0+s1 ensemble's.
   - Writes CSV and JSON.
6. **HF export:** `export_hf --ensemble a,b,c`. The CausalLM returns the probability mixture as logits, and the AutoModel is the final snapshot. The meta declares the members, the pooling and the R0.5 definition before scoring.
7. **CPU tests** (`babylm/tests/test_cycle_avg.py`):
   - `--lr_cycles 0` reproduces the port losses bit for bit.
   - Cycle boundaries, floors and restarts are correct, and the WD integrals are equal.
   - Save-at-steps fires at the exact step.
   - Resume across a cycle boundary is bit-exact.
   - The noise RNG leaves the data order unchanged.
   - The mixer equals a dense brute-force mixture.
   - The wrapper's sentence log-probs equal the mixer's.
8. **Job list:** `babylm/configs/jobs_moonshot_cycle_S.json` (C-a, C-b, base re-run). Outputs, including curves CSV and PNG, go to `shared/babylm/attacks/cycle_avg/<run>/`.

## Skeptic fixes
**Adopted:**
- Novelty: relabelled as a near-reproduction of q0 §2.1 and Huang 2017. Mu 2018 and FORGETTER cited. The distillation follow-on labelled known.
- Nulls:
  - The q0-style N-end null added, with the zero-training read done first.
  - The straw-man N-chk replaced.
  - N-cyc made binding.
- Odds: ceiling cut to about 0.12; expected increment cut to about 0.01.
- Arms:
  - The 0.5x restart dropped.
  - q0 noise added as a separately registered arm.
- Schedule redesign:
  - Late start (epoch 4) with annealed members.
  - Anneal floor kept at the base's 0.1.
  - WD integral matched.
  - No EMA member.
- Readout:
  - One registered predictor.
  - The ρ gate in place of pairwise KL.
  - dev_bpw dumps for every number.
- Official eval: the wrapper uses a probability mixture and is tested and audited; GLUE uses the declared final snapshot; "partial" is expected.
- Reporting: SWA reported; the signature compared with a 2-seed ensemble.
- Process:
  - No mini-ruler.
  - The re-screen rule and the floor wait.
  - The R4.4 ruling requested, plus the size-M report.

**Changed:**
- K = 3 instead of 4: a late start leaves 6 epochs, and K = 2 is reported as a label.
- The lr_floor-0 control and N-lowrestart are not run, because their confounds are removed by construction.
- Kim/Kotha's half-WD ensemble members are not an arm here. They would be a third trial and belong on a possible later card.

## Questions for the critic
1. R4.4: snapshot ensemble FLOPs. Do they count 1x or K x?
2. Confirm that R0 counts the snapshots once.
3. Is the R0.5 definition above acceptable: raw weights at the milestones, with the ensemble defined only at 100M?
4. Is the Stage 0 owner close acceptable as a label, given it comes from DEV-sub?
