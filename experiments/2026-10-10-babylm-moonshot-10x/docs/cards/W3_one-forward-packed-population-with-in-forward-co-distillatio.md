<!-- moonshot design round W3; status alive; ceiling 0.2 nats/word; expected 0.03; P(pass screen) 0.1; screen GPU 125 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card (final, folded): shared-trunk fork ensemble with in-forward codistillation

This card started as the candidate "One-forward packed population with in-forward co-distillation". Critic ruling ref-0110 counts every packed arm K times. That leaves the candidate's own fallback, a shared-trunk fork, as the only arm that can count, so the card is rewritten around the fork and renamed.

**Status: alive, deprioritised.** This is a card only: no code, no runs.
- **Skeptic verdicts:** three (prior art, rules, mechanism). All three say "wounded"; none says "dead".
- **Rules:** R0.3 kills the headline arms P1a, P1b and P1c (400M, 400M and 200M words of exposure). The fix works: a fork whose branches all read 8 shared transformer blocks counts once.
- **Not a 10x lever.** The in-track ceiling is about 0.2 nats/word. The candidate's 0.9 ceiling and its 20x claim are withdrawn.
- **Order of work:** free CPU tests, then a free gate on the C3 seeds (F0), then a proxy of about 30 A100-minutes (F1). The 10M screen (about 95 A100-minutes) runs only if both pass.

**Facts that changed the design (checked 2026-10-10)**
- **The base** is the round-1 winner `base_S_wd0p4_lr0p5_r1_ve0_tied` @ ref-0065 (critic 02:53Z).
  - Score: full DEV 5.0367 nats/word (dev_bpw).
  - Recipe: Muon WD 0.4, lr_scale 0.5, tied embeddings, no value embeddings, U-net, softcap 15, no dropout, no EMA, no MNTP, 10 epochs (99.54M words).
  - Size and cost: 193.46 MFLOP/token, 21.2M non-embedding params.
  - Curves: DEV-sub is still falling at epoch 10 (5.262 at 90M words, 5.121 at 100M). TRAIN-sub is 3.833, so the gap is 1.29 nats/word.
- **Two premises in the inputs were wrong:**
  - The candidate assumed a base with heavy WD, EMA and MNTP rows. That base does not exist.
  - The skeptics' "5.30" is the WD-2 cell, which lost. The base is therefore 0.10 nats/word behind published GPT-BERT 10M (4.939), not 0.36.
- **Binding rulings:**
  - ref-0110: packed groups count K times. A real shared trunk counts once. It also sets the proxy rules.
  - ref-0265: the FLOP-matched reference must use at least the attack's FLOPs. It is the best of M at E* and the alternatives, each with WD re-tuned, with trial parity. It is needed only at promotion, and a tie there reads "compute, not data efficiency".
  - ref-0086: ensembling counts as a regulariser-type attack, so any base change to WD, EMA or MNTP forces a re-screen.
  - ref-0083: the best of four smoothing nulls is binding.
  - ref-0126: the C3 bands.
- **The seed floor is not registered yet.**

## Label and known ideas

**Label: known idea, untested regime** (exposure-capped 10M x 10 LM pretraining, FLOP-matched). This corrects the candidate's "known ideas, new combination".

**The same mechanism already exists:**
- **ONE** (Lan, Zhu, Gong, NeurIPS 2018, arXiv 1806.04606): a shared trunk with m branches; an on-the-fly ensemble teacher is distilled back into every branch on every mini-batch.
- **CL-ILR** (Song & Chai, NeurIPS 2018, arXiv 1805.11761): several heads on a shared trunk with consensus soft targets.
- **KDCL** (Guo et al., CVPR 2020) and **DML** (Zhang et al., CVPR 2018, arXiv 1706.00384).
- **Online codistillation** (Anil et al., ICLR 2018, arXiv 1804.03235). Its leave-one-out target and its burn-in before codistillation starts are adopted below.
- **Shared-trunk multi-head ensembles:** TreeNet (Lee et al. 2015, arXiv 1511.06314) and Hydra (Tran et al. 2020, arXiv 2001.04694).
- **Packing members on a K axis** is an implementation form only: Packed-Ensembles (Laurent 2023, arXiv 2210.09184), BatchEnsemble (Wen 2020, arXiv 2002.06715), MIMO (Havasi 2021, arXiv 2010.06610), torch.func ensembling.

**Shared-trunk ensembles are already measured in LM pretraining.** NeuroTrails (Grooten et al. 2025, arXiv 2505.17909) trained LLaMA-130M on 1B tokens of C4 (perplexities as the prior-art skeptic read them):

| model | perplexity | training FLOPs |
|---|---|---|
| dense | 29.06 | 1x |
| independent 3-ensemble | 26.88 | 3x |
| TreeNet (1/3 shared trunk, 3 heads) | 26.46 | 2.21x |
| dense LLaMA-250M (App. H.1) | 26.59 | about 2x |

So at matched FLOPs the fork about ties a wider single model. Only the regime (10 epochs on 10M words) is untested.

**Evidence against:**
- **Same-data codistillation is weak.** Anil's ~2x came from members trained on partitioned data. Same-data codistillation was only "slightly better than the baseline" (Fig. 2b), and that run covered under one epoch. Sodhani et al. 2020 (arXiv 2010.02838) read the gain as regularisation.
- **Consensus pulls peers together.**
  - ONE: the inter-branch variance is smaller than for independent nets, which "probably hurts the ensemble".
  - OKDDip (Chen et al., AAAI 2020, arXiv 1912.00350): mean aggregation homogenises peers.
  - Anil: codistillation cut prediction churn by 35%.
- **Training the average directly fails.** Allen-Zhu & Li (ICLR 2023, arXiv 2012.09816) and Jeffares et al. (NeurIPS 2023, arXiv 2301.11323, "learner collusion").
- **Small-data agreement regularisers gain little:**
  - Fraternal dropout (arXiv 1711.00066): PTB perplexity 57.3 to 56.8.
  - R-Drop (arXiv 2106.14448): WT103 perplexity 25.76 to 23.97, about -0.07 nats/token, using two forwards.
  - BatchEnsemble on WMT: perplexity 4.30 to 4.26.

**How large the ensemble channel is elsewhere:**
- **Kim/Kotha/Liang/Hashimoto 2025** (arXiv 2509.14786; independent seeds, no codistillation, K x exposure):
  - regularised single-model asymptote: 3.43;
  - ensemble of 300M members as K -> inf: 3.34, or 3.27 with retuned members;
  - joint asymptote: 3.17, extrapolated.
- **Baby Llama** (arXiv 2308.02019): a 4-seed ensemble took eval loss from 3.77 to 3.66, with unregularised members and 4x exposure.

**Corrections to the candidate's citations:**
- arXiv 2601.16936 ("Is BatchEnsemble a Single Model?") is a ResNet-18 study on CIFAR/MNIST. It is vision only, not a Transformer analysis.
- BabyLM 2024.17 (Iyer) is **uninformative**: its RoBERTa baseline sat at chance (BLiMP 49.6) and no loss is reported. It does not show that "mutual learning failed".
- MoS (arXiv 1711.03953) is the trained-mixture relative of the log-mean-exp head.

**In-house evidence:**
- GPT-2 10M + GPT-BERT 10M ensemble: +0.077 [0.070, 0.085] nats/word. Two architectures, at 2x exposure.
- Snapshot mix: +0.048.
- V8 (2 members x 5 epochs, legal): registered band [-0.02, 0.06] nats/token.
- C3 bands: g(8) in [0.10, 0.25], g_inf in [0.12, 0.30].

The search budget ran out after about 20 targeted searches. Under R6, unknown defaults to known.

## Mechanism (registered form)

**Architecture of the F4 arms (size S, one model)**
- **Embedding and head:** one tied table, as in the base.
- **Trunk (fed once per window):** blocks 0-7, i.e. all 6 U-net encoder blocks plus decoder blocks 6-7.
- **K = 4 branches.** Each branch owns:
  - 4 decoder blocks (positions 8-11), with its own block scalars;
  - its own 4 U-net skip weights, which pop the trunk's shared skips from blocks 3, 2, 1 and 0;
  - a final RMSNorm feeding the shared tied head, which gives softcapped logits z_k.
- **Initialisation:** branches are built in sequence from the run's RNG. Branch 0 takes the base's draws for blocks 8-11, so the K = 1 fork equals the base bit for bit. Output projections are zero-initialised as in the base, so the branches agree at step 0 and diverge through their Q/K/V/FC init.
- **Trunk depth is fixed at 8 of 12 and is never tuned downward.** A shallower trunk exploits the letter of ref-0110 ("at least one shared block") and needs its own card plus a ruling.

**Objective (one forward per step)**
- q_k = softmax(z_k).
- t_k = the mean of q_j over j != k, **detached** (Anil's leave-one-out form).
- L = (1/K) sum_k [CE(y, q_k) + λ·KL(t_k ‖ q_k)].
- λ = 0 in epoch 1 (steps < total/10, switched by `on_step_end`), then 0.3 (untuned).

**Arms**
- **F4-cd:** the objective above.
- **F4-0:** the same fork with λ = 0 throughout. This is plain TreeNet: K CE losses on one trunk.
- **F2s (exempt from R4.4):** an 8-block trunk plus K = 2 branches of 2 blocks. Each path is a 10-block U-net (5 encoder, 5 decoder), 12 blocks in total, 1.195x FLOPs, λ = 0. It is registered under ref-0265(2) as the arm at the base's FLOPs.

**Optimiser:** Muon on every 2-D block matrix; each branch matrix is its own parameter, so Newton-Schulz runs per member. Adam on the table and scalars. LRs, WD 0.4, schedule and seed are the base's. No EMA, and no "free snapshot members" (final-checkpoint rule).

**Inference and export**
- **Causal LM:** logits = log((1/K) Σ_k softmax(z_k)), computed in fp32 as logsumexp_k(log_softmax(z_k)) − log K. BLiMP, EWoK and COMPS then score the ensemble as one model, and dev_bpw scores the exported module.
- **AutoModel (GLUE):** returns the mean of the branches' final normed hidden states. hidden_size = n_embd, so the export_hf assert holds.
  - Mode: causal, by the family rule (no MNTP).
  - GLUE fine-tunes 2x the base's non-embedding params; this is reported.
- **Inference cost:** about 2.4x the base's forward (reported; allowed).

## Why (which measured slice)

**Channel: posterior averaging.** The base is one SGD solution with a large variance-like gap:
- TRAIN-sub minus DEV-sub is 1.29 nats/word at epoch 10.
- Mixing its own snapshots gains +0.048.
- Mixing two models gains +0.077.

The gain should land where members disagree: familiar words in new contexts (the 64-8k band, 0.36 nats/word; the new-text extra, 0.61) and rare words.

**Honest limits:**
- A4's measurement says this slice is mostly bias: the two-model mix gains only ~0.06-0.09 per word on the 64+ bands.
- Averaging hedges; it does not add knowledge.

**Why the base cannot capture it.** Tuning WD and size moves the model along the single-model curve. Kim/Kotha's K -> inf ensemble sits 0.09-0.16 nats/token below their regularised single-model asymptote. The base has no within-run averaging, and the in-forward KL is the only way to give branches soft targets at zero extra exposure.

**Why the fork may not capture it either.** Its branches share 8 of 12 blocks (every encoder block and every skip output), the tied table and head, and the data order. C3 seeds differ in both init and data order (data_seed = seed). So C3's g(K) is an **upper bound** for the fork, not its decider.

A λ > 0 consensus pull also works against the gain. With y drawn from p̄, mean_k KL(p̄ ‖ q_k) is the ensemble's Jensen gap, which is exactly the disagreement that produces the averaging gain.

## Ceiling

**Posterior channel at K x exposure (out of track): at most about 0.3 nats/word.**
- C3's registered g_inf tops out at 0.30.
- Kim/Kotha's ensemble plus member scaling over a regularised reference is 5.17/2.29 = 2.26x, about 0.31 by log-linear. Part of that is member scaling, which the FLOP-matched single model also gets.

**In-track fork: about 0.2 nats/word (about 1.7x data).** This is the best case: g_inf at the top of its band, and the fork keeping about 2/3 of the independent-seed diversity despite the shared trunk, table and data order.

**Withdrawn:** the 0.9 ceiling, "C3 decides 20x", and the extrapolation from 4x-exposure Baby Llama.

## R0 exposure ledger

| item | words |
|---|---|
| F4-cd, F4-0, F2s: one model. Every window is read once per epoch through the shared table and 8 shared blocks; the branches read the trunk output, not text | 10 x 9.954M = 99.54M |
| codistillation target t_k (same forward; several losses on one forward count once, R0.2) | 0 |
| teachers, generators, caches, second views | none |
| **total per arm** | **99.54M <= 100M; 9.95M unique; no outside data or model** |

- **Counter test:** words_seen equals the base's at every milestone (R0.5).
- **Dropped as out of track under ref-0110, not run:**
  - P1a (K = 4, own tables): 400M.
  - P1b (shared tables): 400M.
  - P1c (K = 2 at d = 272): 200M.
- **Not re-run:** P1a at λ = 0 is exactly C3 (8 independent seeds, labelled "K x exposure, out of track"), so g(K) is read from C3.
- **Not part of this card:** the legal "K members x 10/K epochs" split is V8. V8 plus the KL would need its own card.
- **No "official-letter" arm is requested.**

## FLOPs and the R4.4 plan

FLOPs use the harness formula: 6 x matmul params (heads included) plus causal attention. It reproduces the base's 193.46M.

| model | MFLOP/token | x S |
|---|---|---|
| base S (12L/384d) | 193.5 | 1.00 |
| F2s (8 + 2x2 blocks, 2 head passes) | 231.2 | 1.195 |
| F4-cd, F4-0 (8 + 4x4 blocks, 4 head passes) | 462.4 | **2.39** (not 2.0: the 16k head is about 20% of S and is paid 4 times) |
| W640 (12L/640d single) | 464.0 | 2.40 |
| M (12L/768d) | 641.7 | 3.32 |
| candidate's P1a (4xS) | 773.8 | 4.00 |
| fork at size M (8 + 4x4, d = 768) | 1434.5 | - |
| L (16L/1024d) | 1409.3 | 0.98x of fork-M |
| 18L/1024d | 1572.9 | 1.10x of fork-M |

The candidate's 12L/768d reference was 0.83x of P1a, which is invalid under ref-0265(a). At the second size, L is just short of fork-M, so the reference there is 18L/1024d or larger.

**At the screen:** W640 runs in the same batch at the base recipe, without retuned WD. This is stricter than ref-0265(2) requires. An untuned wider model is the weaker comparator, so losing to it is a clean kill.

**At promotion (F4 arms):**
- **Size S:** the reference is the best of M at E* = 10 and W640, each with WD re-tuned, at least the attack's trial count, frozen and hashed first.
- **Size M:** the reference needs at least 1434.5 MFLOP/token.
- **A tie is not a win.** It reads "compute, not data efficiency". NeuroTrails App. H.1 predicts exactly that tie.

**F2s** (1.195x, under 1.25x) is exempt from R4.4.

**KL overhead:** an estimated +5-10% wall-clock, measured as tokens_per_s.

## Null twins and comparators

- **B, the base re-run** (same seed, data order and GPU type).
  - It doubles as the identical-branch null. A fork whose branches start identical gets identical branch gradients scaled by 1/K, which Muon ignores and Adam nearly ignores. It is numerically the base at 2.4x the cost.
  - That null therefore runs only as a CPU test, which also asserts that disagreement stays at 0.
- **B-snap (binding): the base re-run's own zero-exposure inference ensemble.**
  - Construction: the final checkpoint is probability-mixed with one earlier epoch. `babylm/analysis/snapshots.py` picks the epoch and w on even DEV-sub documents, and the mix is scored on full DEV from dev_bpw dumps.
  - Purpose: it gives the base the same test-time averaging the fork gets (2x inference vs 2.4x).
  - MC dropout is not possible: the base has no dropout.
- **W640: same FLOPs, no branching.** This is the same-compute control for the averaging (screen kill, above).
- **F4-0: the KL null.** Same fork and code path with λ = 0. It separates codistillation from plain averaging.
- **Smoothing nulls (best of four binding, ref-0083).** These run only if F4-cd is at least 2σ ahead of F4-0.
  - Construction: the fork with t_k replaced by uniform or train_10M-unigram smoothing. ε is H-matched to F4-cd's logged `cd_H` or off-mass-matched to `cd_off`, with the same λ schedule and code path.
  - Purpose: to test whether the KL is only label smoothing (Tf-KD; Anil's own baselines).
- **Free decomposition (same checkpoints, reported):**
  - branch 0 alone (pre-registered, no picking);
  - the mean single-branch DEV;
  - the averaging gain (mean branch minus ensemble);
  - retention r = averaging gain / C3's g(K) at the same K.

## Gates before the 10M screen

**F0, free: C3 at the right scale.** C3 is already budgeted on the round-1 winner (seeds 0-7, 10M x 10).
- **g(4) gate:** g(4) must be at least 0.10 nats/word (raw dev_bpw; overlap-removed shown beside it). Below that, the fork's averaging is worth at most about 0.05, which cannot clear 2σ plus a single model at 2.4x FLOPs. The card is then labelled "gated out by C3 (diagnostic)".
- **M gate:** if a WD-tuned M single at E* exists and the 4-seed S ensemble (4x FLOPs, 4x exposure) is not at least 1σ ahead of it, no in-track fork can clear R4.4. Label: "R4.4 case dead (diagnostic)".

**F1, about 30 A100-minutes: mini-ruler proxy at train_1M x 10, kill-only.**
- **Labels and rules (ref-0110 Q2):** "mini-ruler proxy" and "tokenizer saw 10M". Every run is a trial, and the base recipe is not changed.
- **Why kill-only:** 1M favours ensembles, because the 10M-tuned WD under-regularises there. A pass means little; a fail is strong.
- **Runs:** base seeds 0-3, which give σ_1M and g_indep(4)@1M; F4-0, F4-cd, W640 and F2s at seed 0.
- **Kill rules:**
  - Kill (proxy) any arm that is not at least 2σ_1M ahead of base seed 0.
  - Also kill F4 arms that are not at least 2σ_1M ahead of W640.
  - Drop F4-cd if DEV(F4-cd) >= DEV(F4-0).
  - The card is "killed (proxy)" if no arm survives.
- **Reported:** r at 1M, the disagreement curves, and the mini-ruler position.

## Kill rule (registered before any run)

**Setup:**
- Size S, seed 0, round-1 winner recipe (hash recorded), train_10M x 10.
- All arms in one batch, on one GPU type, with the same data order.
- Metric: **full-DEV nats/word from `dev_bpw.py` on the exported final checkpoint**, with no checkpoint picking.
- σ = the critic's registered seed floor (planning value 0.01-0.015, unregistered). No kill or promotion before it is registered.

**Kill an F4 arm if any of these holds:**
- (a) DEV(arm) > DEV(B) − 2σ.
- (b) DEV(arm) > DEV(B-snap) − 2σ.
- (c) DEV(arm) > DEV(W640) − 1σ. This reads "compute, not data efficiency".
- (d) BLiMP-fast drops by at least 2σ vs B.

**F4-cd also:**
- It is killed if DEV(F4-cd) > DEV(F4-0) − 2σ. The KL is then dropped, and F4-0 is judged on its own.
- If it survives that, it must beat the best of the four smoothing nulls by 2σ. Those nulls run in batch 2, with their own base re-run.

**F2s:** rules (a), (b) and (d).

**Promotion: exactly one arm.**
- **Which arm:** the lowest full DEV among survivors. Ties within 0.005 go to the cheaper arm (F2s, then F4-0, then F4-cd).
- **Seeds and comparators:** 3 seeds of the arm, B, B-snap and the frozen ref-0265 reference.
- **Second size:** fork-M vs base M and a reference of at least 1434.5 MFLOP/token.
- **Official eval:** the R4 bar (> 2σ on the NLP score; minimal-pair tasks may not move), and FINAL once on the critic's call.

**Provisional base (ref-0086(b)):** a later change to the base's WD, EMA or MNTP forces one re-screen. Until then, a kill reads "killed (provisional base)".

**Mechanism checks** (reported, not kills). They replace "novelty tilt >= 0.8", which hedging passes automatically.
- Branch disagreement per epoch on DEV-sub: D(e) = mean_k KL(p̄ ‖ q_k), for F4-0 vs F4-cd, against C3's independent-seed D.
- Gain binned by member disagreement.
- Top-1 accuracy (log-rank) gained per nat, vs the ruler's 10M -> 20M step.
- The seen-text vs new-text split.
- BLiMP-fast and EWoK.
- Novelty and rare tilts, shown but not discriminating.

## Odds (guesses, stated plainly)

**Ceiling:** about 0.2 nats/word in track; about 0.3 for the whole channel at K x exposure.

**Median gain:**

| vs | median (nats/word) | 80% interval |
|---|---|---|
| B | +0.03 | [-0.01, +0.07] |
| B-snap | about +0.01 | - |
| W640 | about 0.00 | [-0.04, +0.03] |

**Probabilities:**
- F0 passes: about 0.55.
- F1 passes, given F0: about 0.45.
- The screen passes, given both: about 0.4.
- **Passes overall: about 0.10.**
- The KL beats λ = 0 by 2σ: about 0.1.
- Gain >= 0.26 (2x data): about 0.02.
- 10x: about 0.

**Most likely failure:**
1. Branch disagreement on DEV is 1/4 to 1/3 of the independent-seed level.
2. λ = 0.3 halves it from epoch 2.
3. The averaging gain is about 0.01-0.03.
4. The fork is +0.03-0.05 over B, mostly from 2.4x FLOPs and 2x params.
5. It ties W640: compute, not data efficiency.
6. BLiMP stays flat.

## Fastest falsification

1. CPU tests: free, about 1 hour.
2. F0: zero marginal GPU, once C3 lands.
3. F1: about 30 A100-minutes.

About 75% of plausible outcomes end at F0 or F1, before the 10M screen.

## Screen

| arm | what | est. A100-min |
|---|---|---|
| B | base re-run | 9 |
| B-snap | snapshot pick plus one extra full-DEV dump | 2 |
| F4-0 | fork 8 + 4x4, λ = 0 | 22 |
| F4-cd | fork plus leave-one-out codistillation, λ 0.3 from epoch 2 | 23 |
| W640 | 12L/640d single, base recipe | 22 |
| F2s | fork 8 + 2x2, λ = 0 | 11 |
| fast BLiMP/EWoK | all arms | about 7 |

- **Batch 2 (conditional, about 100 min):** the four smoothing nulls plus a base re-run.
- **Total with F1:** about 125 A100-minutes, or about 225 if batch 2 runs.
- **Logging:** every run logs exposure, tokens/s, FLOPs, peak memory and D(e), with curves as CSV and PNG under shared/babylm/attacks/fork/<run>/.

## Implementation plan (harness; about 800 LOC, about 1.5 agent-days)

Branch `wip-branch`, after the critic accepts the card.

- **`babylm/models/fork.py` (~250 LOC).** `build(cfg)` returns `ForkGPT`, reusing Block, CastedLinear, norm and Rotary from `sprint1.models.baseline`.
  - cfg.extra: `fork_trunk` (8), `fork_k`, `fork_branch_layers`, `cd_lambda`, `cd_start_frac` (0.1), `cd_target` (loo | uniform | unigram) and `cd_eps`.
  - It exposes tied `embed` and `lm_head`, so train.py's tied-head re-init applies.
  - Split API: `hiddens()` returns [K, B, T, d], `hidden()` the branch mean, plus `logits()`.
  - `forward`, training: the loss above, with the KL in fp32, chunked by branch.
  - `forward`, eval: the log-mean-exp CE, plus per-branch per-token losses and D in aux.
  - `on_step_end(step, total)` switches λ on.
  - `param_groups()`: every 2-D block matrix to Muon; the table and scalars to Adam.
  - `flops_per_token()` uses the table formula. It logs `cd_kl`, `cd_H`, `cd_off` and D.
- **Training:** run through the existing `train.py --model fork --size S --tie_embeddings --extra fork_k=4 ...`; no train.py change is expected. Micro-batch drops to 8 if `mem_estimate.py` predicts more than 35 GB (four fp32 [16, 1024, 16384] logit tensors).
- **Export (~150 LOC).** `hf_template/modeling_fork.py`: log-mean-exp causal logits and a mean-hidden AutoModel. `export_hf.py` detects the fork, and `verify_export` compares ensemble logits. A critic audit (as for ref-0252) is required before any score counts.
- **`babylm/moonshot/fork_report.py` (~120 LOC):**
  - per-branch and ensemble full-DEV from dev_bpw dumps;
  - averaging gain;
  - retention vs the C3 dumps (`ceiling_report c3`);
  - D curves;
  - gain binned by disagreement;
  - top-1 per nat;
  - B-snap via `snapshots.py` plus a full-DEV mix.
- **`babylm/moonshot/make_fork_jobs.py` (~60 LOC):** job lists for F1, batch 1 and batch 2, run by `run_jobs.py`.
- **`babylm/tests/test_fork.py` (~220 LOC):**
  - K = 1 is bit-identical to the base on the tiny config.
  - Identical-branch init reproduces the base's loss trajectory on CPU, with D = 0.
  - The exposure counter equals the base's.
  - The eval loss equals a dense −log mean_k softmax reference.
  - λ = 0 gives exactly the mean CE.
  - The target is detached and the leave-one-out mean is correct.
  - The Muon group count is trunk + K x branch.
  - The export reproduces the harness full-DEV.
  - Pause and resume are bit-exact.

## Stacking and overlap

- **A1, snapshot mixing and EMA** work on the same posterior channel; their gains do not sum (critic note at 00:12Z).
- **A2 and A3** act on the shared input and stack mechanically, but a combination needs a combined card.
- **The fork could serve as a one-exposure teacher** for ensemble-to-student distillation. That needs its own card.
- **Mutually exclusive** with any K x independent-seed arm under R0.

## Skeptic fixes: disposition

**Adopted:**
- **Label and citations:** the "known idea" label with the full citation set. Added NeuroTrails, ONE, OKDDip, Allen-Zhu & Li, Jeffares, R-Drop, fraternal dropout and the in-house +0.077.
- **Citation corrections:** the Anil claim corrected, with Sodhani added; the 2601.16936 and 24.17 descriptions fixed.
- **Exposure under ref-0110:** P1a, P1b and P1c dropped; the fork made primary; the trunk depth pre-registered.
- **Ceiling:** cut to about 0.2 (about 0.3 for the channel).
- **KL:** a λ = 0 arm, the leave-one-out target and disagreement curves.
- **FLOPs:** table corrected (fork 2.39x; P1a's reference invalid; L too small at M). A FLOP-matched single runs at the screen, and a tie with it kills.
- **Nulls:** the identical-init GPU null is replaced by B, B-snap, F4-0 and the best-of-four smoothing nulls.
- **Checkpoints and proxy:** snapshot members dropped; the proxy is kill-only and counted as trials.
- **Export and tests:** export plan (fp32 log-mean-exp, family pooling, critic audit) and a per-member Muon test.
- **Checks and gating:** discriminating mechanism checks, and the F0 gate on C3.

**Not adopted:**
- **Holding every screen until the base is "frozen".** The round-1 winner is selected (5.0367, not 5.303), and ref-0086 allows screens with a re-screen condition.
- **An R-Drop or fraternal-dropout null.** The base has no dropout, and two dropout streams on separate forwards count twice. F4-0 plus the smoothing nulls answer the same question.
- **The fold-masked out-of-fold CE variant.** Follow-up only: the trunk still sees every row.
- **An MC-dropout comparator.** The base has no dropout.
- **The r < 0.4 proxy kill.** r is reported instead; the W640 kill already covers it.

## Rulings requested before any GPU run

1. Confirm that the 8-of-12 shared trunk counts once under ref-0110.
2. B-snap as a binding comparator.
3. The untuned W640 as a screen kill, with the ref-0265 reference used at promotion.
4. C3 numbers used only to deprioritise (F0).
5. An audit of the fork export.
