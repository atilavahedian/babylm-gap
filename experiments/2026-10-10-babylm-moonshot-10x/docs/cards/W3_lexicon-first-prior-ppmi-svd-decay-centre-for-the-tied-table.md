<!-- moonshot design round W3; status alive; ceiling 0.1 nats/word; expected 0.008; P(pass screen) 0.07; screen GPU 150 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card: lexicon-first prior (PPMI-SVD decay centre + lexicon-smoothed targets), 2026-10-10

Status: card only. No code and no runs. CPU gates G1/G2 run first; GPU waits for critic acceptance and the registered seed floor.
Base: round-1 winner `base_S_wd0p4_lr0p5_r1_ve0_tied` @ ref-0065 (full-DEV 5.037 nats/word). Its config: tied table in the AdamW "head" group, init N(0, 0.005), lr 0.0104, **head WD 0**, muon_wd 0.4, no value embeddings, no EMA, no MNTP, embnorm on the input, softcap 15, no output bias, 10 epochs. DEV-sub is still falling at epoch 10 (5.26 at 90M, 5.12 at 100M within one schedule).
Three skeptics (prior art, rules, mechanism) all returned "wounded". This card adopts their fixes. The draft's ceiling (0.35), expected gain (0.03) and P(pass) (0.2) are withdrawn.

## Known ideas (label: **known idea** per half; **known ideas, new combination** only for the joint arm)
- **Count lexicon.** PPMI-SVD with context-distribution smoothing and eigenvalue weighting 0.5: Levy, Goldberg & Dagan 2015 (Q15-1016); Levy & Goldberg 2014.
  - Sahlgren & Lenci 2016 (D16-1099) do not back the draft's "count beats predict" claim. At 10M words, truncated SVD loses to SGNS on TOEFL (38.75 vs 43.75), ESL (30 vs 42) and SimLex (19.27 vs 28.30), and wins only on MEN and RW. Asr et al. 2016 (CHILDES) measure word similarity, not LM loss. Both are dropped as support for LM gains.
- **Decay toward a pretrained point.** L2-SP (Li et al. 2018); the re-embedding penalty (Peng et al. 2015).
  - Welch, Mihalcea & Kummerfeld 2020 is the closest match: same-domain init plus anchoring in 10-100M-token LMs.
  - Their gain needs extra in-domain data. It is gone with BPE (dev ppl 37.8 baseline vs 38.3 with their method) and shrinks with more LM data. Frozen output embeddings gave PTB ppl 218-680.
- **Smoothing toward embedding neighbours.**
  - Inan, Khosravi & Socher 2017 (augmented loss). On top of tying it is worth only 0.029 / 0.0095 / 0.0073 nats/token on PTB small/medium/large, shrinking with size. The draft's -0.05/-0.06 came from untied baselines.
  - D2GPo (Li et al., ICLR 2020) and Elbayad et al. 2018 already use soft targets from *fixed* pretrained embeddings: WMT14 +0.13 BLEU.
  - BabyLM 10M: Syntactic Smoothing (Diehl Martinez et al. 2024) scores BLiMP 73.2 with similarity smoothing, the same as uniform label smoothing (73.2; base 71.4). Berend 2023 (MLSM) degraded BLiMP on its own.
  - Label smoothing raises Transformer perplexity (Müller et al. 2019). Pereyra et al. 2017: generic smoothing gains about as much as AL.
- **Static-embedding inits in BabyLM and transformers.**
  - No gain: Chen & Portelance 2023, Fields et al. 2023. WhatIf 2024 was matched by a generic augmentation.
  - Raw GloVe init is worse than random (Kim et al. 2024).
  - Kocmi & Bojar 2017: the same-train-set word2vec init is only about 0.024 nats/token better, an early head start.
  - BabyLM 2025.37 is dropped as support: its baselines are near chance (56.26 vs 54.91 untrained).
- **Rare-row geometry.** FRAGE (Gong 2018; 0.01-0.035 nats/token), Gao et al. 2019 (representation degeneration), AGG (Yu 2022), Pinto et al. 2024 (WD degrades low-frequency tokens).
- **In-repo (fails so far).**
  - The PPMI start on the dense toy base faded: 0.064 / 0.043 / 0.036 BPB at 500 / 1k / 1.5k updates, 0.0215 at 3k. Most of the held `ppmi_embnorm` experimenter is embnorm, which this base already has.
  - `vgraph` (whole-run coupling to PPMI neighbours) was KILLED; its shuffled-graph null was 0.0047 better (ref-0424; docs/sprint_1_prior_lessons.md).
  - sprint1/ideas/B_brainstorm.md #11 already listed similarity label smoothing as known (D2GPo), predicted ±0.005, "biased; fades".
  - Sprint-1 ladder: label smoothing hurt by 0.45-0.67 nats (refs/prior_art.md; source not re-verified).
- **What is untested.** A same-corpus PPMI-SVD *centre inside decoupled WD of a tied table*, under an exposure-charged count pass, against a WD-matched comparator. That is a detail, not a new mechanism.

## Mechanism (as amended)
1. **Lexicon (CPU, train_10M only).** The builder asserts the train_10M sha256 ref-0309... and refuses dev, dev_sub and test.
   - Token co-occurrence within ±5 tokens. Pair counts are clipped at 4 per 1000-word block. This is a block cap, not a document cap: 12,285 of the 13,713 documents are simple_wiki articles, and the burstiness measurement used 1000-word blocks. The 1M proxy uses the identical rule.
   - PPMI with context smoothing 0.75, then sparse truncated SVD to d = 384 with rows U·Σ^0.5.
   - **Row norms are kept** (not unit-normalised), times **one global scalar s** fixed before the run: the mean row norm of the round-1 winner's trained tied table. That is one number, logged as a trial. No per-row norms come from a trained model (R0.3).
   - The 128 never-seen types and any pad ids (16384-16511 if MNTP is ever enabled) get a zero centre and are excluded from every neighbour list.
2. **Decay centre (arm L-dec).** Decoupled decay of the tied table toward the prior: p ← p − lr_t·wd·(p − s·E_prior).
   - Same order as torch AdamW: the decay is applied before the Adam update. With wd = 0 it is a no-op, so the base's "same WD" was inert.
   - The knob is registered as the integrated pull κ = wd·Σ_t lr_t. The base schedule has Σ lr_t = 19.6 at 10 epochs, so κ ∈ {0.3, 1.0} means head_wd ≈ {0.015, 0.051}. The value is recomputed at start-up for 9 epochs and logged. κ = 0.3 is the weak-centre arm.
   - How it differs from vgraph: the centre is a fixed point, not the neighbours' current values. How it differs from the PPMI start: the pull lasts the whole run, so it cannot fade the way an init does, but it is a lasting bias if the lexicon is worse than SGD's table.
3. **Smoothing (arm L-sm), word-initial targets only.**
   - Target m = (1 − α_t)·onehot(y) + α_t·t_y, where t_y = softmax(cos(E_prior y, ·)/τ) over y's top-32 neighbours. Self is excluded, so the off-label mass is exactly α_t.
   - Word-initial means Ġ-prefixed, or the first token after a newline or document start. Word-internal pieces are never smoothed.
   - α = 0.03, annealed linearly to 0 over the last epoch, so the final epoch is pure cross-entropy.
   - Pre-registered bias bound: without the anneal, the optimum q* = (1−α)p + α·Σp(y)t_y costs at most −log(1−α) = 0.030 nats per word-initial token, about 0.03 nats/word. That is as large as the whole expected gain, so the anneal is mandatory.
   - τ is fixed from gate G2 (counted trials). The loss is computed on the same forward, so exposure is unchanged.
4. **Init only (L-init).** The table is initialised at s·E_prior with no decay or smoothing. This is a factorial cell (it is the in-repo PPMI start, expected to fade). It runs in batch 2 only.

## Why (corrected slice)
- **Draft claims withdrawn.**
  - The 1-63 band is out of reach. Words seen 1-63 times are only 8.3% single-token (0.7% for 1-15 times), and the median train count of their first piece is 528 (mechanism skeptic's measurement, not re-verified). The lexicon cannot give a rare word a row.
  - The 8k+ types are pinned by gradient.
  - "Heavy WD shrinks rare rows" is false here: the head WD is 0, and with a tied softmax every row gets gradient every step.
- **Where the centre could bind.** Token types with train counts 64-1023: 13,033 types, about 18% of DEV tokens (7.9% + 10.4%). This is part of the 0.36 "64-8k words in new contexts" slice.
- **What it can and cannot do.** It is a data-derived regulariser of the output geometry, not new information. Every bit comes from the same 10M words, and a count model of the same 10M adds only +0.051 nats/word at inference. The rare-word pattern (0.7-1.0 nats behind on train text, 2-6.5 on new text) looks like variance, which is the right target for a regulariser. But the base's muon_wd already regularises the body.
- **Frequency risk.** PPMI removes the marginals, so E_prior carries no unigram direction. The pull fights the head's frequency code just as decay to 0 does. B9-hwd isolates this.

## Exposure ledger (R0)
| item | words |
|---|---|
| Count pass over train_10M (a helper model under R0.3; charged one epoch by precedent) | 9.95M |
| Student: 9 epochs x 9,953,773 | 89.58M |
| **Total** | **99.54M ≤ 100M** |

- Counting online during epoch 1 does not avoid the charge, because R0.3 sums every model in the pipeline.
- A 10-epoch arm with an uncharged lexicon is out of track and **not run**. It needs a critic ruling and a re-registration first.
- If the frozen base lands at E* ≤ 9, the arms run at E* with no handicap, and the comparator is the base at E*.
- Gates G1 and G2 are inference on already-trained models, with zero new exposure.
- T100 (100M words x 10 epochs) appears only in labelled oracle diagnostics (G1 and arm O), never in a verdict.
- No outside data, no generated text, so the R5 13-gram check is trivial. HF eval is unaffected: no architecture or inference change, and the tied export already scores.

## FLOPs (R4.4)
- **Arm training.** 9 epochs = 0.90x the 10-epoch base (1.00x B9).
- **Overheads.**
  - Smoothing: a 32-way gather and softmax per word-initial position, under 0.1%.
  - Decay: one vector operation per step.
  - CPU: about 1-2e12 FLOP for counting, PPMI, SVD and the neighbour lists, against about 2.6e16 for training. Reported.
- **R4.4 is not triggered.** Wall-clock and tokens_per_s are logged against the base.

## Nulls and comparators (registered)
- **Binding base.** The better of **B10** (base, 10 epochs, re-run in the same batch) and **B9** (base, 9 epochs on its own full schedule; this measures the counting charge).
- **Tuned-regularisation floor for the new knob.** **B9-hwd(κ)**: the base with the same head_wd, decaying toward 0, run at each κ the arms use (trial parity).
- **Centre nulls.**
  - **N-permbin**: E_prior rows permuted *within 64 log-frequency bins*. Same per-bin norms and scale, no distributional identity.
  - **N-iso**: Gaussian rows with E_prior's per-row norms.
  - Both run at the best L-dec κ.
- **Smoothing nulls** (A1 precedent, critic 22:45Z). Uniform and unigram label smoothing on the same word-initial positions with the same anneal, each in two calibrations: off-mass matched (ε = α) and H-matched to L-sm's logged mean H(m). Plus smoothing over the N-permbin neighbour lists.
  - A1's null runs cannot be reused: the positions, the start epoch and the anneal all differ. The code is shared: A1's smooth path plus a position mask.
- **Oracle diagnostic O** (label "oracle lexicon (100M words; out of track)"). The decay centre is T100's lm_head rows, keeping their own norms, times one scalar matched to s, at κ = 1.0.

## Screen (size S, seed 0, round-1 winner recipe, same batch and GPU type)
| batch | run | what |
|---|---|---|
| 1 | B10, B9 | base comparators |
| 1 | B9-hwd0.3, B9-hwd1.0 | decay to 0 at matched κ |
| 1 | L-dec0.3, L-dec1.0 | E_prior centre |
| 1 | L-sm | α 0.03, word-initial, annealed; head_wd 0. Runs only if G2 passed |
| 1 | O | oracle centre, label only |
| 2 (only if a batch-1 arm passes) | N-permbin, N-iso | at the best decay arm |
| 2 | 5 smoothing nulls | if L-sm is alive |
| 2 | L-init | factorial cell |
| 2 | L-both | only if both halves passed batch 1 |

Batch 1 is 8 runs, about 70 A100-minutes The full screen is at most about 150 A100-minutes. Every run counts in the trial log, the base included (R3 parity).

## Kill rule (registered before any run; full-DEV nats/word from dev_bpw.py, final checkpoint, no checkpoint picking)
σ is the critic's registered seed floor at S (expected about 0.01-0.015, so 2σ is about 0.02-0.03). Until σ is registered, results are numbers only, with no verdict.
- **K1 (batch 1).** Kill the card if the best arm is not below min(DEV B10, DEV B9) by 2σ.
  - Kill a decay arm if it is not below its B9-hwd(κ) by 2σ.
  - Kill L-sm if it is not below B9 by 2σ.
- **K2 (decay ceiling premise).** Kill the decay half for the moonshot if min(B10, B9, B9-hwd1.0) − O < 0.05 nats/word: even a 100M lexicon buys less than 0.05. Also kill it if O ≥ 0.05 but L-dec recovers < 30% of O's gain.
- **K3 (batch 2).** Kill if the best arm does not beat every one of its nulls by 2σ. A decay arm faces N-permbin and N-iso. L-sm faces all 5 smoothing nulls. L-both faces all of them.
- **K4.** Kill on a ≥ 2σ BLiMP-fast drop, or if the arm's final gain over B9 is more than 0.5σ below its gain at epoch 6 (fade rule, 18:37Z).
- **Provisional base.** This is a regulariser-type attack. Any base change to WD, head_wd, EMA or MNTP forces a re-screen, and a kill reads "killed (provisional base)" until then (ref-0086(b)). If the frozen base re-enables value embeddings, that also forces a re-screen.
- **Promotion.** Exactly one arm (lowest full-DEV; ties within 0.005 go to the smaller κ or α) is picked on DEV before seeds 1-2. It goes to 3 seeds, size M and official eval. The best null becomes the 3-seed binding twin. The 3-seed verdict re-uses no screen numbers.
- **Mechanism check.** Required for the "lexical prior" label. If it fails, a passing arm is relabelled "generic regulariser (known)".
  - The gain must be enriched in token bands 64-1023 relative to those bands' share of the T100-vs-base gain (≥ 1.3x), and the enrichment must be absent in N-permbin.
  - The novelty-tilt CI must lie within [0.8, 1.4].
  - The gain on trigrams seen in train_10M must be non-negative.
  - Row drift ||E − s·E_prior|| is reported per band at every milestone. A table pinned to the prior is the frozen-output regime Welch warned about.

## Odds (folded from the three skeptics: ceiling 0.10 / 0.15 / 0.12, expected 0.010 / 0.015 / 0.008, P 0.06 / 0.10 / 0.08)
- **Ceiling: about 0.10 nats/word.** That is the case where the in-track lexicon behaves like T100's table. The realistic in-track cap is about 0.05 (the same-10M count model's +0.051). By log-linear interpolation that is at most about 1.3x data. **Not 20x-sized.**
- **Median expected: +0.008 nats/word** against the binding comparator, net of the 9-epoch charge.
  - Decay half: 0 to +0.015.
  - Smoothing half: about +0.005, overlapping A1/A2.
  - The charge itself costs about −0.005 to −0.015, measured by B9.
- **P(pass K1-K3) ≈ 0.07. P(the CPU gates kill it before any GPU run) ≈ 0.5.**
- A null result is still informative. It closes "static lexical priors" for this base, consistent with vgraph, the PPMI start and Syntactic Smoothing.

## Fastest falsification (in order; stop at the first kill)
1. **G1 (CPU, zero exposure, about 1 h). Decay half.**
   - Take the token types with train count 64-1023 (13,033 types).
   - For each type, compute top-20 cosine-neighbour overlap with T100's lm_head table (oracle, label only), once for E_prior and once for the round-1 winner's own tied table.
   - Continue only if E_prior beats the winner's table on a paired sign test (p < 0.01) with a mean overlap ratio ≥ 1.2. Otherwise the closed-form lexicon is no closer to where 10x data puts the rows than SGD already is: kill the decay half.
   - The draft's held-out co-occurrence test is withdrawn: it is circular (PPMI-SVD fits co-occurrence by construction) and its data was not held out.
   - The GPT-BERT-100M overlap check is dropped: it uses a different tokenizer.
2. **G2 (CPU, zero exposure). Smoothing half.**
   - Smooth the frozen winner's DEV-sub outputs post hoc: q'(z) = (1−a)q(z) + a·Σ_y q(y)·t_y(z), moving only word-initial mass, for a ∈ {0.01, 0.02, 0.05, 0.1} and τ ∈ {0.05, 0.1, 0.2}.
   - Controls: N-permbin neighbour lists, unigram mixing on word-initial mass, and the best global temperature.
   - Kill the smoothing half unless the real lexicon beats every control at some (a, τ) with a document-bootstrap 95% CI that excludes 0. The winning τ is carried into L-sm, counted as trials.
3. **Mini-ruler proxy (about 3 A100-minutes, label "mini-ruler proxy", kill only).** The lexicon is rebuilt from train_1M only, with the same block cap. Runs: L-dec1.0 vs B-hwd1.0 vs N-permbin at 1M x 9 epochs. Result "killed (proxy)" if L-dec is not ahead of both: by 2x the mini-ruler floor once one is registered, or by any margin at all before then. Priors should look best at 1M, so a failure here is telling.
4. Then batch 1.

## Implementation plan (harness; nothing written yet)
- **`babylm/moonshot/lexicon.py` (CPU, numpy + scipy).**
  - Reads the uint16 stream and asserts the sha256 from the meta. Splits it into 1000-word blocks at Ġ/newline word boundaries.
  - Within each block: pair ids w·V + c, then `np.unique`, clip at 4, accumulate into a sparse matrix.
  - Then PPMI (cds 0.75), `scipy.sparse.linalg.svds(k=384)`, U·Σ^0.5, the scalar s, and top-32 neighbours (with self, never-seen ids and pad ids excluded).
  - Writes E_prior.npy, nbr_idx/nbr_cos.npy, freq_bins.npy and a manifest (input/output sha256, params, CPU time, FLOP estimate).
  - `--null permbin|iso` writes null priors with a fixed seed. `--split train_1M` is for the proxy.
- **`babylm/moonshot/lexprior.py` (lever, off by default).**
  - Decay hook: the head group's weight_decay is set to 0, and p ← p − lr_t·wd·(p − c) is applied before `optimizer.step()`.
  - κ is converted to wd from the schedule at start-up.
  - Smoothing loss: on the same forward, gather the 32 neighbour log-probs at word-initial target positions, with the α anneal. Null targets are uniform, unigram or permbin, under `--match H|off`.
  - Logs smoothing mass, H(m) and row drift per band.
- **`train.py` flags.** `--lex_prior PATH --lex_kappa K --lex_init --lex_smooth_alpha A --lex_smooth_tau T --lex_smooth_anneal_epochs 1.0 --lex_smooth_null {uniform,unigram,permbin} --match {H,off}`.
- **G1/G2 scripts.** `babylm/moonshot/lexicon_gates.py` runs a CPU forward of the HF export on DEV-sub, plus the sparse mixing and bootstrap.
- **Tests (`babylm/tests/test_lexprior.py`).**
  - Flags off gives the bit-identical port loop.
  - A zero centre with κ > 0 equals `--head_wd`.
  - α = 0 reproduces the base losses.
  - The sparse smoothing loss equals the dense reference, with matching gradients.
  - The anneal reaches exactly 0 at the last step.
  - The builder refuses non-train inputs.
  - Pad, never-seen and self ids are absent from the neighbour lists.
  - N-iso keeps per-row norms; N-permbin keeps the bins.
  - Resume is bit-exact.
- **Jobs and outputs.** Jobs go through `run_jobs.py` JSON lists. Outputs (curves CSV + PNG, per-token dumps for prize.py band_shares and signature) go to shared/babylm/attacks/lexprior/&lt;run&gt;/.

## Stacking and overlap
- L-sm overlaps A1, since both soften targets. A combination needs its own card, and the gains do not add (critic note, 00:12Z).
- This card no longer claims rare rows. The experimenter's imprinting card owns those.
- If A3 promotes, the lexicon is rebuilt on canonical tokens only.
