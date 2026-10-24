<!-- moonshot design round W1; status KILLED (proxy) at Gate 2 (b), 2026-10-10; ceiling 0.3 nats/word; expected 0.02; P(pass screen) 0.15; screen GPU 170 A100-min. Built and CPU-tested; submitted to the critic 2026-10-10. -->

# Attack card PE: one-forward shared-trunk particle ensemble (bayes lens; bootstrap demoted to a gated add-on)

**Verdict (2026-10-10 ~16:10Z): killed (proxy) at Gate 2 (b).** Mini-ruler proxy, 1M x 10, torch 2.9.1 on both sides (PE3, NJ and the I3 = ms_f1_m0-m2 references; within-env). rho = J(PE3) / J(I3) = 0.3289 / 0.9030 = 0.364 on the full leg and 0.3283 / 0.8996 = 0.365 on the guarded leg, against the registered 0.5. p_bar(PE3) 7.6068, NJ 7.7267, B mean 7.9084, and the I3 three-seed mixture 7.005: three independent 1M models beat the one-forward population by 0.60 nats/word. (a) was not decided, because the F-deep run hit its VM timeout and was then skipped as moot (F-M alone reads 9.288). With the bootstrap arms unbuilt, no PE arm goes to Gate 3. This is the outcome the card itself forecast in its failure path (rho about 0.4 at 1M). Readout: executor 3 reads/pe_gate2/ (v2_readouts @ ref-0153).

**Status.** Built and CPU-tested on branch wip-branch (2026-10-10 ~04:50Z); no GPU run. It folds in three skeptic verdicts (prior-art, rules and mechanism). All three said "wounded" and none said "dead". Next steps, in order: critic acceptance, then Gate 0 (the C3 readout, already queued in the ceiling map), then the GPU proxy. The bootstrap arms (PE3b, PE3r) are not built: Gate 2 runs without them (see "As built").

**Names.** The card was submitted as "one-forward-bootstrap-particle-population". It is renamed because the bootstrap is no longer the primary arm. Arms use the prefix PE. The old names C3a/C3b clashed with the ceiling map's C3.

**Base.** The round-1 winner, `base_S_wd0p4_lr0p5_r1_ve0_tied` @ ref-0065:
- 12L/384d, 21.23M non-embedding parameters;
- tied embeddings, n_value_embeds = 0, U-net, x0 injection, softcap 15;
- Muon WD 0.4, lr_scale 0.5, E* = 10 (labelled "epochs not bracketed above");
- MNTP, EMA and dropout off;
- full DEV 5.0367 nats/word;
- 193,462,272 FLOPs/token; 99,541,960 words fed.

Because the base has no value embeddings, no dropout and no MNTP, the candidate's points about shared value-embedding tables, per-particle dropout masks and MNTP rows do not apply.

## Known ideas (label: **known idea, untested regime**)

This is a shared-trunk multi-head ensemble, with a bootstrap leg that is itself a known idea and has a negative prior. The label is not "new combination": under R6, an unknown novelty status defaults to known.

**The mechanism already exists:**
- **Bootstrapped DQN** (Osband et al. 2016, arXiv 1602.04621). A shared torso feeds K heads. Each head's loss carries its own bootstrap weight; the Exp(1) masks "closely resemble the standard Bayesian nonparametric posterior of a Dirichlet process". The torso gradient is normalised by 1/K. Their Atari runs used p = 1, which means no bootstrap ("performed similarly irrespective of p"), and they credit the diversity to random init.
- **TreeNet / "M heads"** (Lee et al. 2015, 1511.06314):
  - ILSVRC-NiN x5: random init 64.08%, single model 58.90%, bagging 55.02%;
  - sharing AlexNet through conv3/4/5 gave 58.39/57.73/55.25, against 59.47 for the full ensemble.
- **MoS** (Yang et al. 2018): a shared trunk, K softmaxes and a probability mixture, trained jointly. PTB 57.3 to 54.44 ppl, about -0.05 nats/token.
- Related designs: Hydra (Tran et al. 2020), Packed-Ensembles (Laurent et al. 2023), BatchEnsemble (Wen et al. 2020), MIMO (Havasi et al. 2021), Neural Bootstrapper (Shin et al. 2021).
- The bootstrap itself: Bayesian, weighted-likelihood and posterior bootstrap (Rubin 1981; Newton & Raftery 1994; Fong et al. 2019) and bagging (Breiman 1996).

**Standing counter-evidence for the bootstrap leg:**
- Lakshminarayanan et al. 2017: "bagging deteriorated performance".
- Nixon, Lakshminarayanan & Tran 2020 (ICBINB): bootstrap ensembles are dominated by plain deep ensembles, even with close to 100% unique points. One skeptic fetched this; two cite it from memory.
- Branch-Train-Merge (Li et al. 2022): LM ensembles trained on random data splits lose to a compute-matched dense LM (125M: 23.0 vs 19.9 ppl).
- Byrd & Lipton 2019: importance weights fade over epochs in deep nets.

**Ensembles in data-constrained LMs:**
- Kim/Kotha et al. 2025 (2509.14786). Independent members beat the regularised N->inf asymptote. App. C.1: init alone or data order alone gives most of the benefit, with data order helping more. A jointly trained 10-ensemble was only 0.02 better.
- q0 (Mandal et al. 2026, 2606.03938). At E = 4, 2 models x 2 epochs scored 3.501 against 3.365 for 1 model x 4 epochs: splitting exposure across models loses.
- Codistillation (Anil et al. 2018), ONE (Lan et al. 2018), learner collusion (Jeffares et al. 2023).

**BabyLM:**
- Baby Llama 2023: 4-seed logit ensemble, eval loss 3.77 to 3.66, at 4x exposure.
- BabyLlama-2 2024: gains "nowhere near" those of 10x data.
- Mutual learning 2024: BLiMP 51.7 / 51.6 / 51.6 with 1, 2 and 4 peers.
- refs/prior_art.md lists no shared-trunk or bootstrap ensemble in the 2023-2025 proceedings.
- Coverage caveat: the skeptics' web-search budget ran out, so this rests on prior_art.md plus direct fetches.

**What is untested:** whether a one-forward shared-trunk ensemble, counted once under R0, beats a FLOP-matched tuned single model at 10M words.

## Mechanism

One connected network, one forward per window.
- **Shared:** the tied embedding/unembedding, x0, and blocks 0..3. Ls = 4 is a registered constant.
- **Private:** K = 3 stacks of blocks 4..11, each initialised independently and each with its own U-net skip weights.
  - U-net with n_enc = 6: blocks 6 and 7 pop skips from the particle's own blocks 5 and 4; blocks 8-11 pop from shared blocks 3..0.
  - The final RMSNorm is parameter-free.
  - The tied unembedding and the softcap are shared.
- **Loss (primary arms):** L = (1/K) sum_k CE_k. Each particle is trained on its own CE, never on the mixture (Jeffares 2023). This gives one scalar and one backward pass, and the trunk receives the average of the K gradients.
- **Output:** p_bar = (1/K) sum_k p_k, computed as the log-mean-exp of the fp32 softcapped log-softmaxes. Nothing is fitted on DEV.

**Bootstrap add-on (gated at Gate 2; it enters the screen only if it survives):**
- Weights: w_k(u) ~ Gamma(4, 1/4). The unit is u = global stream position // 128, about 125k chunks of 128 tokens. Unit ids come from the sampler's global window starts, so they do not change with the random epoch offsets.
- Each particle has a Philox table keyed by (seed, k), renormalised to a token-weighted mean of 1.
- Weighted loss: L_k = sum_t w_k CE_k / sum_t w_k.
- Why chunks, not BOS documents: train_10M's 13,713 documents have a word-weighted effective count of about 487 (half of all tokens sit in 257 documents). Document weights would therefore mostly reweight books and sources: switchboard by about ±15% and gutenberg by about ±7% at a = 4.
- a = 1 is dropped: its per-token ESS is 0.50. At a = 4 the ESS is 0.80, which costs each member up to about 0.08 nats/word (at 0.26 per doubling) before any Jensen gain.

**Arms:**
- PE3: a = inf, base WD.
- PE3w: 0.5x Muon WD on the private stacks only. This is the WD half of Kim's member heuristic; the 2x-epoch half is out of track.
- Conditional: PE3b (a = 4, persistent weights) and PE3r (a = 4, weights re-drawn each epoch; an online reweighting that stands in for data-order diversity, as in Osband).

**Outside this card:** K > 3, wider private stacks, Ls < 4, the online-distilled student stack, and combinations with A1 or A2. Each needs its own card and its own FLOP level. Ls < 4 also needs a new critic entry, because it approaches the loophole that the 00:40Z Q1 ruling closes.

## Why (which measured slice)

**Target.** The variance part of the new-text-only slice (0.61 nats/word): familiar words in new contexts (the 64-8k band, 0.36) and the 1-63 band (0.16-0.18). The seen-text slice (0.24, partly model size) is not claimed; R4.4 credits it to the FLOP-matched reference.

**Measured size of the posterior channel:**
- GPT-2 10M + GPT-BERT 10M mix: +0.077 nats/word (2x exposure);
- own epoch-5/10 snapshot mix: +0.048;
- Baby Llama's 4 seeds: about 0.18 (4x exposure);
- registered C3 bands: g(8) in [0.10, 0.25], g_inf in [0.12, 0.30].

Averaging removes variance, not bias. About 80% of the gap is "abstraction" that even a 100M trigram table does not supply, so this channel is capped near g_inf.

**Why the base cannot capture it.** One network trained once cannot represent the average of K independently fitted functions. Kim/Kotha found that independent ensembles beat the regularised single-model asymptote. Corrected arithmetic:
- 2 x 300M = 3.597 - 0.115 = 3.482;
- one 600M model = 3.513;
- margin about 0.03 nats/token at matched parameters (the candidate said 0.055).

Separate members are not legal in track: q0 shows that splitting exposure loses, and 00:40Z Q1 counts separate members K times. A shared trunk is the only way to count the exposure once.

**Why it may fail:**
- The shared trunk and the shared tied unembedding (the main carrier of rare-word variance) both cut diversity. TreeNet found that sharing more than 1-2 early layers fell below the full ensemble.
- All particles see identical batch order, and Kim's larger diversity source, data order, is therefore missing.
- Expected retention: rho = J(PE3) / J(3 independent seeds) of about 0.3-0.6.
- The gains will be hedging-type, so the rare and novelty tilts may look data-like without data-like top-1 gains.

## R0 exposure ledger

Already ruled: critic 00:40Z Q1 (ceiling map ref-0110) counts a grouped design once if every group's output depends on at least one shared transformer block. Ls = 4 qualifies. The candidate's "ruling needed / withdraw" branch was stale and is removed.

| Run | Models trained | Words fed | Exposure | Label |
|---|---|---|---|---|
| PE3, PE3w, PE3b, PE3r, NJ | 1 connected net (3 particles on a 4-block trunk) | 10M x E* | 99.54M, counted once | in track |
| F-deep, F-M, B | 1 each | 10M x E* | 99.54M each | in track |
| Proxy I3 (B + 2 seeds) | 3 separate | 3 x 1M x 10 | 30M | "mini-ruler proxy, K x exposure", counts as tuning trials |
| Other proxy runs | 1 each | 1M x 10 | 10M each | "mini-ruler proxy", counts as tuning trials |
| Gate 0 (C3, ceiling map) | 8 separate | 8 x 100M | 800M | "K x exposure, out of track", diagnostic only |
| Inference | none | 0 | 0 | 2.46x inference cost, reported |

The bootstrap weights are loss multipliers drawn from a hash. They are never input and never data, the words-fed counter is unchanged, there is no teacher and nothing is generated.

## FLOPs and the R4.4 plan

Formula, from sprint1/models/baseline.py; the base value of 193,462,272 matches base_r1's result.json:

6 x (n_exec_blocks x 12d^2 + K x V x d) + 6 x n_exec_blocks x T x d

| Arm | FLOPs/token | x base |
|---|---|---|
| PE3/PE3w/PE3b/PE3r/NJ (K = 3, Ls = 4, 28 executed blocks, 3 head matmuls) | 476,577,792 | **2.463** |
| K = 4 / K = 8 (not in this card) | | 3.195 / 6.122 |
| F-deep 35L/384d | | 2.543 |
| 12L/672d | | 2.614 |
| F-M = size M 12L/768d | | 3.317 |

The candidate's 2.33x left out the K head matmuls and the attention in the private blocks.

**FLOP-ledger trap.** The harness fallback `harness_flops_per_token` counts parameters, not executed matmuls, so it would log the tied twin at about 1x. `ParticleGPT.flops_per_token` must count executed blocks plus the K head matmuls, and a test asserts 476,577,792 for every K = 3 variant, the tied twin included.

**Reference at size S.** A screen-level reference F runs in the batch. It is untuned (base WD), so a screen pass is necessary but not sufficient. This is stricter than 01:20Z(2) requires. For promotion (01:20Z), the reference must reach at least 2.463x and is the best of {M at E* (3.317x), 35L/384d (2.543x), 12L/672d (2.614x)}:
- WD re-tuned for each;
- at least the attack's trial count;
- frozen and hashed before the first promoted score, under the ref-0086 firewall.

The candidate's 28L/384d (2.07x) and ~590d (2.08x) fall below 2.463x and are dropped.

**R4.4 reference rule, shared with the legal-split family (critic ref-0135(4); one definition for every ensemble-shaped arm).** An arm whose submitted model executes F x the base's FLOPs/token (executed matmuls, the formula above) is compared, at promotion and claim, with the best (lowest v2 DEV, then official) of the registered plain base-family models below whose FLOPs/token are at least F x. Each is trained on train_10M at its own best epoch count E* <= 10 (the arm's exposure cap; the base itself is past its optimum at e10: experimenter 08:50Z, each e5 single beats the e10 pool by 0.02-0.03), WD re-tuned, with at least the arm's trial count, frozen and hashed before the arm's first promoted score (ref-0086 firewall). The menu holds, for each registered F, one deep model (d 384, layers added), one wide model (12 layers, d widened) and size M; a smaller F inherits every larger entry, so its reference can only be stronger. An exported K-member mixture executes K full forwards, K.000 x.

| Arm | F | Eligible references (x base) |
|---|---|---|
| legal split K = 2, branch(2) | 2.000 | 28L/384d (2.073), 12L/592d (2.092), and every entry below |
| PE3 family (this card) | 2.463 | 35L/384d (2.543), 12L/672d (2.614), M 12L/768d (3.317) |
| legal split K = 3 | 3.000 | 42L/384d (3.012), 12L/736d (3.074), M 12L/768d (3.317) |

The PE rows are the menu registered above, unchanged; the K = 2 and K = 3 rows apply the same rule. Accepted with clarifications (critic ref-0318): the reference is the lowest-loss of the 1x solo base pool (at its best E* <= 10; accepted, critic ref-0079: menu E* from {2, 4, 5, 6, 8, 10}, tuning counted as trials) and the eligible menu, so an overfitting large model never eases the bar; both legs and the calibrated control apply to the menu models too; every label states training and inference FLOPs (a split mixture: 1x training, K x inference). Beating the arm's base but not this reference is labelled "compute, not data efficiency".

**F = 2 menu, registered (critic ref-0419; babylm/moonshot/make_menu_jobs.py, babylm/configs/jobs_moonshot_menu_f2.json @ ref-0073).**
- Phase 1: 28L/384d and 12L/592d (8 heads of 74) at E 4, 5, 6, base WD, seed 0, solo. LRs are inherited from S, and the reference is labelled "LR inherited".
- Phase 2 fires when any entry at its E* comes within 0.03 of the K = 2 mixture on any leg under either metric (raw or calibrated_t). At that entry and E it runs WD 0.2 and 0.8 and lr_scale 0.35 and 0.7. It adds E 2 or 8 when E* sits at a grid edge. Every run is a counted trial.
- A K = 2 claim must beat the lowest of the base E* pool and the menu at E*, on both legs, raw and calibrated_t. The chosen entry gets seeds 1 and 2 at E* for the claim pool.
- The menu is frozen and hashed before K = 2's first promoted score.

**Reference at size M.** PE3 at 768d costs 1.548 GFLOP/token (2.412x M). Size L is only 2.196x M, so the M-level reference is the best of {32L/768d (2.471x), 20L/1024d (2.706x)}, WD re-tuned.

**No same-FLOP arm.** The 01:20Z option of a same-FLOP arm at the base's E* would need E of about 4, and the base's DEV is still falling at e10. Fewer epochs would forfeit more than the ensemble buys, so that arm is not registered.

**Memory.** K x [B,T,V] fp32 logits with chunked CE. The base peaks at 4.5 GB on a 40 GB A100.

## Null twins

- **NJ, joint twin (the binding R6 null).** Identical architecture: K untied, independently initialised stacks, the same forward, FLOPs, exposure, code path and p_bar output. It is trained on CE(log p_bar), the mixture (collusion) objective. This removes the per-particle independent objectives, i.e. the posterior sampling, and keeps the capacity.
- **F, FLOP-level reference in the screen batch.** F-deep (35L/384d) and F-M (12L/768d) at base WD; the comparison reads against the better of the two. It has zero diversity and matched or greater FLOPs.
- **B, base re-run.** The attack code at K = 1, Ls = 12 (a bit-identical path), with the same seed, data order and GPU type, in the same batch.
- **PE3r, bootstrap null.** Runs only if PE3b runs. It has the same Gamma(4) weight distribution, re-drawn each epoch (Philox key (seed, k, epoch)), so no particle keeps a persistent empirical distribution.
- **Tied twin.** The K stacks share their Parameters, so it computes the base function at 2.46x FLOPs. It is a CPU code-path and FLOP-ledger check only, never a GPU arm. It is not an R6 null, because it removes the extra parameters along with the diversity.
- **Reported beside B, with no change to the bar:** the base's own 3-snapshot inference mix at equal inference K (zero exposure).

## Kill rule (registered before any run)

**Metric:** full-DEV nats/word from dev_bpw.py (ref-0054 protocol, ws collapse) on the final checkpoint, with no checkpoint picking. Harness and official instruments are never mixed within one delta (03:16Z). Jensen gains are never used for a decision, except inside rho.

**Gate 0 (C3 readout, no new compute):**
- Kill the card before any GPU run if g(8) < 0.10 (the ceiling map's deprioritise line) or the interpolated g(3) < 0.06 nats/word. A one-forward population at equal K cannot exceed independent seeds, and F plus 2x the floor will take about 0.04-0.08.
- If 0.10 <= g(8) < 0.20, stop after Gate 2 and report.

**Gate 2 (mini-ruler proxy, 1M x 10).** Every run is a logged trial, and outcomes are "mini-ruler proxy" labels, at most "killed (proxy)". floor_1M is the SD of I3's three single seeds.
- (a) Killed (proxy) if the best p_bar among PE3/PE3b/PE3r is not below the better F by at least 2x floor_1M.
- (b) The shared trunk as designed is killed (proxy) if rho < 0.5. Here rho = J(PE3) / J(I3 3-ensemble), and J = mean-member DEV minus p_bar DEV.
- (c) The bootstrap is dropped unless p_bar(PE3b) beats both p_bar(PE3) and p_bar(PE3r) by at least 1x floor_1M. PE3r enters the screen as an arm only if it beats PE3 by at least 1x floor_1M.

**Gate 3 (10M screen: S, seed 0, E*, same batch and GPU type):**
- Kill if the best PE arm's p_bar is not ahead of each of B, NJ and the better F by 2x the registered seed floor.
- Kill on a BLiMP-fast drop of 2 sigma or more.
- A conditional PE3b must also beat PE3r and PE3 by 2x the floor, and PE3r must beat PE3 by 2x the floor.
- No kill or promote before the DEV and BLiMP-fast floors are registered; until then, numbers only.
- Ensembling is regulariser-type (22:17Z(b)), so any later change to the base's WD, EMA, MNTP or dropout forces one re-screen. Until then a kill reads "killed (provisional base)". If the critic adopts a shorter screening schedule E_s (02:53Z), PE screens at E_s against the base at E_s.

**Promotion:**
- Exactly one arm, picked on DEV before seeds 1-2. Ties within 0.005 go to PE3. The 3-seed verdict re-uses no screen numbers.
- 3 seeds at S and M plus official eval, against the R4.4 references above. Beating B but not F is labelled "compute, not data efficiency".

**Reported, not kills:**
- rho at 10M against C3's g(3);
- mean member DEV against B;
- pairwise JS between particles;
- rare and novelty tilts, plus top-1 accuracy gain per nat against the ruler's 10M->20M step. Below 0.5x the data step's ratio reads as hedging, not data-like.

## Odds

- **Ceiling about 0.30 nats/word:** the top of C3's g_inf band times rho (at most 1), net of the capacity credited to F. The candidate's 0.9 assumed the whole 0.61 new-text slice is estimator variance, and no evidence supports that.
- **Median about +0.02 over F.** That is about +0.04 over B, almost all of it known shared-trunk ensembling.
- **Bootstrap leg: 0 to -0.01.** A count-model toy (labelled "count-model toy, not neural") shows the pattern: Jensen gain +0.008 to +0.074 nats/token, while p_bar is 0.001-0.022 nats/token worse than the w = 1 fit, at 1M and at 10M.
- P(pass Gate 3 against the best of B, NJ and F) is about 0.15.
- P(full R4 at S and M, official score included) is about 0.05.
- P(bootstrap survives Gate 2) is about 0.1.
- P(at least 0.10 over F) is below 0.05.

**Most likely failure:**
1. CPU parity passes.
2. At 1M, rho comes out at about 0.4.
3. At 10M, p_bar beats B by about 0.03-0.05.
4. F comes within 0.01-0.02 of the population, and BLiMP-fast is flat.
5. Result: a kill against F.

## Fastest falsification

**(0) No new compute.** Read out Gate 0 from C3 (seeds 0-7, already queued in the ceiling map).

**(1) CPU, about 20 min.** Tests:
- K = 1 with Ls in {4, 12} is bit-exact to the port GPT (same RNG draw order);
- the tied twin stays one function and logs 2.463x;
- flops_per_token = 476,577,792;
- the exposure counter is unchanged;
- chunk ids are global and invariant to the epoch offsets, with token-mean weight 1;
- NJ differs only in its loss;
- Ls < 4 is refused;
- HF log p_bar matches the harness to 1e-5 (fp32).

The bigram toy stands as a sanity check. An optional 2L/128d imprint probe on train_1M can drop the bootstrap before any GPU time if its imprint is below 0.01 nats/token.

**(2) GPU mini-ruler, one batch, about 47 A100-min** Runs: B, I3 (B plus 2 seeds), PE3, PE3b, PE3r, NJ, F-deep and F-M, 10 runs in all. Training is about 14 min and eval about 33 min.

**(3) 10M screen, about 122 A100-min**

| Arm | A100-min (train + eval) |
|---|---|
| PE3 | 21 |
| PE3w | 21 |
| NJ | 21 |
| F-deep | 22 |
| F-M | 29 |
| B | 9 |

Conditional PE3b and PE3r add about 42 min.

**Promotion.** Estimated at 1,500-1,800 A100-min Ask the experimenter before promoting.

## As built (2026-10-10, replaces the implementation plan)

- **`babylm/models/particles.py`: ParticleGPT** (`--model particles`), a subclass of the sprint-1 GPT.
  - Particle 0 IS the GPT (same modules, names and init draws); particles 1..K-1 (`pblocks.{k-1}`, `pskip.{k-1}`)
    are built after it. K = 1 is the port model bit for bit (tested: state dict, train and eval loss, flops).
  - cfg.extra: `pe_k`, `pe_shared` (K > 1 needs 4 <= Ls <= L - L//2; Ls < 4 refused), `pe_objective`
    (`particles` = mean member CE, `mixture` = NJ), `pe_tied` (tied twin), `pe_private_wd_scale` (PE3w: the private
    block matrices get their own Muon group at scale x WD, through `model.make_optimizer`).
  - Eval loss and per_token_loss = -log p_bar(y), so evaluate_stream needs no change. Train logs `pe_member_ce` and
    `pe_mix_ce`. Only per-token CEs at the target are combined: no [K, B, T, V] mixture tensor.
  - `flops_per_token` counts executed matmuls: 476,577,792 for every K = 3 / Ls = 4 arm, tied twin included (tested
    on a meta-device size-S build; base 193,462,272).
  - At init the particles are the same function (zero-initialised output projections, as in the base); they diverge
    from their different input-side init after the first steps.
- **`train.py`:** refuses `--kd_teacher`, `--mntp_frac` and `--window_plan` for K > 1. Nothing else changes.
- **HF export:** `hf_template/modeling_hinton_pe.py` (imports blocks from modeling_hinton.py). CausalLM logits =
  log p_bar (fp32); AutoModel = mean of the K final-normed particle states; causal only. `export_hf --verify` compares
  with the harness's `mix_logprobs`: max diff < 1e-5 (tested, also tied twin), and the registered `dev_bpw.py` and its
  per-token dump agree on the export. K = 1 exports as the baseline.
- **Members:** `python -m babylm.moonshot.particle_members CKPT --out DIR` writes each particle as a plain baseline
  checkpoint (tested: same function as particle k; their mixture = the ensemble). J and rho then come from the
  registered scorer: J = `ceiling_report c3` over the K member dumps.
- **Jobs:** `python -m babylm.moonshot.make_particle_jobs BASE:JOB --gate {2,3}`;
  `babylm/configs/jobs_moonshot_pe_gate2_S.json` (B x seeds 0-2, PE3, NJ, F-deep, F-M on train_1M; 7 runs) and
  `jobs_moonshot_pe_gate3_S.json` (B, PE3, PE3w, NJ, F-deep, F-M on train_10M; 6 runs), on
  base_S_wd0p4_lr0p5_r1_ve0_tied @ ref-0065. Timeouts are about 2x the A100 estimates (worst case 150 and 260
  A100-min).
- **Not built:** the bootstrap weights (PE3b / PE3r; the card's own odds: 0 to -0.01, P(survive Gate 2) 0.1), and a
  separate per-particle dump in dev_bpw (the member split replaces it). Gate 2 (c) is therefore not run.
- **Tests:** `babylm/tests/test_particles.py` (16).

## Skeptic fixes

**Adopted:**
- the novelty relabel and the counter-evidence citations;
- the R0 citation and Ls registered at 4;
- the corrected FLOPs and the R4.4 levels;
- the seed-only population as the primary arm;
- bootstrap decisions made on p_bar only, against the re-drawn-weights null;
- chunk units and dropping a = 1;
- the NJ joint twin, with F in the screen batch;
- Gate 0 on C3;
- the rho gate;
- the export declaration;
- the corrected ceiling and Kim arithmetic;
- the arm rename;
- the provisional-base and floor-wait rules.

**Not adopted:**
- Private value-embedding tables: moot, because the base has n_value_embeds = 0.
- Breaking the U-net skips: it would change the K = 1 identity and add a trial, and the rho gate already catches lost diversity.
- The student leg: deferred to its own card.
