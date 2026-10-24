<!-- moonshot design round W3; status alive; ceiling 0.15 nats/word; expected 0.01; P(pass screen) 0.05; screen GPU 220 A100-min. Draft card: not yet submitted to the critic. -->

# Attack A4-P: population-labelled re-expression (card, 2026-10-10, after three skeptics)

Status: card only. No code, no runs. **Alive, but demoted.** Skeptic verdicts: prior art wounded, rules wounded, mechanism dead. The mechanism skeptic's words were "dead as a component distinct from A4; the residual is a legitimate A4 arm". Only one of the three found it dead, and every rule defect has a workable fix, so the card survives in one form only: **an A4 arm that uses the population as generator and labeller**. It is not a separate component. Three gates stand before any size-S run:
- **G1.** The critic accepts component 1 as shared-trunk under ref-0110. Without that, K = 4 means 4 x 99.5M = 398M exposure, out of track, and this card has no in-track S arm.
- **G2.** The Stage-1 mini-ruler below passes.
- **G3.** If ceiling-map C2 lands first, its f2 must not be below 0.2. A 100M oracle writing contexts bounds a 10M writer from above.

**Removed from the candidate:**
- the cached "free label" as the mechanism (see the identity below);
- the base-as-teacher fallback, which is degenerate;
- burstiness-weighted frame-transplant seeding;
- the 0.5 ceiling;
- N-shuf.

**Label: known idea.** Not new, and not a new combination. The regime is untested: exposure-capped strict-small, a self-continued population, and synthetic rows used as KD inputs.

## The identity that removes the original premise
If y ~ p and the label is p, then E_y[grad CE(y; q)] = grad KL(p || q). A cached label equal to the sampling distribution therefore has N-hard's expected gradient, only with lower variance.

With the single base continuing as the student, q = p at the warm start, so both terms are exactly 0 at first. Later they act only as an anchor to the stale generator. The CE term adds an unlikelihood penalty on train_10M 8-grams through the copy block.

With a population and live co-distillation, a cached label adds nothing except staleness. So "every fed synthetic word carries dark knowledge" is empty as stated. It survives only as the Stage-1 arm L, whose registered prediction is a tie with N-hard.

## Mechanism (registered form)
1. **Generator.** The population P (component 1, shared trunk), trained as its own annealed run at E_G = 4 epochs. It uses the full schedule, and EMA weights if the recipe has EMA. This follows A4's run registration, not a mid-cooldown epoch-6 checkpoint. On base_r1 wd0p4_ve0_tied, DEV-sub is 5.43 at 60M words against 5.12 at 100M, so an epoch-6 teacher would be about 0.3 nats/word behind. P's full-DEV (the members' mean probability) is registered against the binding comparator before generation. If P is worse, the generator is labelled "weaker than comparator".
2. **Text.** A4's generator code and seeds: the first 64 tokens of document-aligned train_10M windows, T = 1.0, the 8-gram copy block against train_10M, and 192 new tokens per row.
   - Row i is sampled from member i mod K alone. Kim/Kotha App. E.1 found single members wrote better data than the logit-averaged ensemble.
   - Seeds are never fed: each row is BOS plus the continuation (A4 convention, 0 exposure).
   - Volume: 19.9M words.
3. **Phase B.** Warm start from P with A4's second LR cycle. The stream is 4 real epochs (39.8M words) plus 19.9M synthetic words, mixed uniformly. Synthetic text is 33% of phase B and 20% of all fed words; this share is registered now (A4 amendment 8).
   - **Real rows:** component 1's loss, unchanged.
   - **Synthetic rows: labels only.** Each member k minimises KL(sg(p_bar) || q_k) at tau = 1, where p_bar is the live mean of the members on the same forward, so it counts once. There is no CE on the sampled token: the sample only shapes the input.
   - **Consequences:** no target is ever the blocked sample, which fixes the copy-block/label mismatch. The label is unblocked and improves as the members train, so there is no stale anchor. There is no label cache, so the top-k bias issue (Anshumann ACL 2025) does not arise.
4. **Hypothesis (the only one left).** Ensemble-to-member transfer needs inputs the members have not memorised. On real rows repeated for 8 epochs, members agree more and co-distillation transfers less each epoch. On self-written rows they disagree more, so the same co-distillation moves more of the ensemble's posterior into each member. This is function matching on augmented inputs (Beyer et al. CVPR 2022).

## Why (which measured slice)
Self-written words carry no information about the true distribution beyond train_10M and P's weights, since both the inputs and the labels are functions of them. Two premises are therefore withdrawn:
- "synthetic word = fresh word";
- the Allen-Zhu & Li 3.1 analogy. Their paraphrases came from the ground-truth generator, not from a trained model.

The channel this arm can move is the **posterior** one:
- ceiling map C3: g_inf registered at [0.12, 0.30];
- Baby Llama's 4-seed ensemble: about 0.18 nats/word at 10M;
- own snapshot mix: +0.048.

It can also move **regularisation**, which overlaps the tuned-WD floor. Both would land on new text (0.61 of the gap) and on the 64-8k band (0.36). The motivating measurement is that dev-minus-train loss grows every epoch (0.51 to 1.11).

The bursty tilt is dropped. H7 puts the whole bursty slice at about 0.02 nats/word, which a 1-seed screen cannot resolve. Any future per-word weights must come from train_10M `type_stats`, never from `dispersion.json`, which holds DEV-gap ratios from published checkpoints.

The A4 thread's own prior on the round-1 base is -0.05 to +0.08 (central +0.02). It is low because our base still gains about 0.3 nats/word over epochs 6-10, so the real epochs given up in phase B are not wasted.

## Known ideas (corrected prior-art line)
- **Kim & Rush 2016, Seq-KD + Word-KD** (arXiv 1606.07947): teacher-generated sequences with a word-level KD term, L = (1-a)NLL + aKD. This is exactly the original candidate's loss. It gave gains for small students only (2x300 and 2x100, not 2x500).
- **Labelled pseudo-data:**
  - MUNGE (Bucila, Caruana & Niculescu-Mizil, KDD 2006): an ensemble labels pseudo-data built by swapping attributes between nearest neighbours, i.e. a transplant operator.
  - GAL (He et al., TACL 2022): an LM writes text and a classifier gives it soft labels.
  - GKD (Agarwal et al., ICLR 2024).
  - Tang et al. 2019 and TinyBERT 2020: augmented transfer sets.
- **Distillation family:**
  - function matching (Beyer et al. 2022);
  - online co-distillation (Anil et al. 2018);
  - LwF and DER (Buzzega 2020): stored-logit replay.
- **Seeding relatives:**
  - TinyStories (forced words);
  - Fadaee, Bisazza & Monz, ACL 2017: rare words in synthetic contexts, +2.9 BLEU;
  - EntiGraph;
  - SBP (arXiv 2509.15248): 42-58% of a 10-20x oracle at 3-6B;
  - RecombiText (BabyLM 2025.40): mixed results, worse as synthetic text dominates.
- **Kim/Kotha 2025, stated correctly** (arXiv 2509.14786, App. E):
  - hard tokens, sampled unconditionally at T = 1, into a fresh student;
  - real:synthetic 1:9 (ensemble distillation) and 1:3 (self-distillation), over 33.4B and 13.4B tokens;
  - individual members beat the averaged ensemble as generators;
  - the decisive ablation is mixing: 1:1 gives 3.437, no mixing 4.069, teacher 3.710.

  Their 83% retention is hard-label sequence-level KD. It is not evidence for soft labels.
- **Nakamura et al., EACL 2026 Findings** (https://aclanthology.org/2026.findings-eacl.213/):
  - OLMo-2 7B/13B continued on 1B self-generated tokens with no real data mixed in;
  - worse than the real-data control: 11.75 vs 11.10 (7B) and 10.60 vs 10.22 (13B);
  - neither 64-token real prefixes nor perplexity filtering helped;
  - soft labels were never tested.

  The confound it exposes is synthetic-only replacement, not hard labels. It must be added to prior_art.md section D with this link before acceptance.
- **Small-scale anchors:**
  - BabyLM 2025.2 (Ulm, Du & Snaebjarnarson): fresh 100M-parameter students on 100M real + 100M synthetic tokens (about 2B training tokens, exposure far past 100M), with per-task checkpoint selection. PPL 24.46 to 23.56 (about 0.037 nats/token); 23.42 at a 40% mix; 22% worse at 90% synthetic. It is an optimistic upper anchor.
  - BERTtime Stories (2024.conll-babylm.28): negative at 10M.
  - Kang et al., EMNLP 2025: rephrase-only text is not faster than natural text, the optimum is about 30% synthetic, and synthetic text hurts most at small budgets.
- **Novelty:** known idea. The "free-label trick is unmeasured" claim is withdrawn: it is standard Word-KD. The frame-transplant twist had no exact match, but WebSearch was unavailable, so that absence is weak. It defaults to known and is not claimed.

## R0 exposure ledger (whitespace words)
| item | words | note |
|---|---|---|
| P training, E_G = 4 | 39.8M | counted once only under accepted shared trunk (ref-0110); otherwise 4x |
| generation, seeds, labels | 0 | inference; seeds not fed; labels on the same forward as the members' loss |
| phase B real | 39.8M | 4 epochs |
| phase B synthetic | 19.9M | 166k rows x 192 tokens |
| **total** | **99.5M** | per-member ledger logged |

- N0' (phase B = 6 real epochs): 99.5M.
- N-hard, N-realKD and the smoothing nulls: 99.5M each.
- Component 1 alone: its own registered total, at most 100M.
- Out of track: a non-shared-trunk K = 4 population (398M).

## FLOPs and R4.4
The base's 10 epochs are 159.6M tokens x 6N. Generation is 31.9M tokens plus 10.6M of prefill, on one member per row: about 0.09x the base.
- The arm costs about (c1 + 0.09)x the base, where c1 is component 1's registered multiple: about 4.1x if c1 = 4.
- Against component 1 alone it costs at most 1.09x, so component 1 alone is the screen's compute-matched comparator.
- **Promotion** needs a frozen FLOP-matched reference at or above the arm's FLOPs (ref-0265): the best of size M at E* and of S on a longer schedule, with WD re-tuned and R3 trial parity. Without it, any gain is labelled "compute, not data efficiency".
- Stage-1 single-model arms: about 1.09x (H) and 1.16x (L, which adds a frozen-teacher forward). These are proxy only.
- Measured tokens/s and FLOPs are logged per source.

## Null twins
All nulls use the same seed, data order, schedule, exposure and code path.
- **N0':** phase B on real text only (6 epochs) with component 1's loss.
- **Component 1 alone,** at its own uninterrupted registered schedule.
- **N-hard:** the same synthetic text with CE on the sampled (blocked) token and the same total target weight; this is the A4 target. It removes the label.
- **N-realKD:** the same KL-only loss applied to a seeded, fixed 19.9M-word subset of real rows instead of the synthetic rows (phase B = 6 real epochs). It removes only "new inputs", and separates this arm from component 1's co-distillation getting extra steps.
- **Smoothing nulls** (A1 precedent, ref-0083): synthetic rows trained toward the sampled token, smoothed toward {uniform, train_10M unigram}, with epsilon matched to the arm's {mean label entropy, mean off-mass}. That is 4 runs, and the best of the four binds. They replace N-shuf: permuted labels keep the row's bag of words and contradict the sampled token, so they hurt by construction.
- **Dropped:** N-seed, together with the frame-transplant claim.

## Kill rule (registered before any run)
Setup: size S, seed 0, the round-1 base family, P = component 1 at E_G = 4. Metric: full-DEV nats/word from `dev_bpw.py` at the final checkpoint, with the population scored as the members' mean probability. No checkpoint picking. delta = 2 x the critic's registered DEV seed floor.

**Kill** if any of these holds:
1. The arm is not ahead by delta of the best of {N0', the frozen base, component 1 alone}.
2. It is not ahead of N-realKD by delta. Then the gain is component 1's co-distillation, not new inputs.
3. It is not ahead of N-hard by delta. Then the card collapses to A4 on the population. N-hard's own standing goes to the A4 thread as a label and is not promoted here.
4. It is not ahead of the best of the four smoothing nulls by delta.
5. BLiMP-fast drops by 2 sigma or more against the binding comparator.
6. A4's leakage audit is non-trivial. The audit (13-gram plus 6-12-word whole-item match, against every eval set and DEV) runs before any number is read.

**Process rules:**
- This is a regulariser-type attack, so a kill on a provisional base reads "killed (provisional base)" (66576f98b).
- No verdict before the floor is registered.
- There is one arm, so nothing is picked.
- Batch 1 (arm, N-hard, N-realKD, N0', component 1 alone, base re-run) decides kills 1-3. Batch 2 (the four smoothing nulls plus a second base re-run, same hardware type) runs only if batch 1 passes. This two-batch screen needs critic acceptance; otherwise all runs go in one batch.

**Promotion:**
- 3 seeds, size M, official eval;
- the population exported as one averaged-probability AutoModel;
- GLUE pooling fixed per family in the export meta before any score (20:31Z ruling);
- K x finetuning cost reported;
- the R4.4 reference must also be beaten.

**Mechanism checks (reported, not kills):**
- Member disagreement, mean KL(q_k || p_bar), on synthetic vs real rows at the start of phase B. The premise needs at least 1.5x.
- Novelty and rare tilts against the frozen run and against N0'. Registered prediction: epoch-like, 0.6-0.9.
- Blocked probability mass per position.
- 6-gram overlap and nearest-neighbour edit distance against train_10M and against each row's own seed document.
- Member diversity at the end of training.

## Odds
- **Ceiling about 0.15 nats/word.** It is bounded by the part of the ensemble gap that real-row co-distillation has not already transferred (g_inf at most 0.30 for the whole posterior channel), plus regularisation that overlaps the WD floor.
  - Even if a synthetic word were worth a fresh one, the arm would have about 30M unique-equivalent words, about 0.41 nats/word, not 0.5. That premise is rejected anyway.
  - This is not a 20x-sized lever.
- **Median:** +0.01 over the binding comparator (90% interval -0.06 to +0.06).
- **P(pass the S kill rule, if it runs):** about 0.05. P(gain of 0.1 or more): about 0.01. P(BLiMP-fast drop): about 0.2.
- **Value of running it:** it settles "labels on self-written text" cheaply, and it measures r, the exchange rate between synthetic and fresh words, which bounds every self-generation attack, A4 included.

## Fastest falsification** Run the base_r1 export through A4's `generate.py` and write 300 continuations. Measure:
- (i) blocked mass per position;
- (ii) the word-trigram novelty rate against equal-length DEV chunks;
- (iii) if two base seeds exist, the mean symmetric KL between seeds on continuations vs on train_10M rows;
- (iv) for the record only, how often w recurs under frame-transplant seeds vs plain seeds.

Decisions:
- If (iii) is below 1.2x, the input hypothesis is labelled unsupported and Stage 1b is skipped.
- If (ii) is below 0.6x of DEV's rate, the stream is labelled "epoch-like re-hash".

**Stage 1: GPU mini-ruler on train_2M.** Size S, seeds 0 and 1, about 70 A100-minutes Results carry the "mini-ruler proxy" label and every run counts as a trial. 2M is used rather than 1M because a 1M generator is too weak.

- **1a, single model.** B2 is 2M x 10 epochs. G2 is 2M x 4 epochs, annealed, and serves as the generator. Each phase B is 12M words:

  | run | phase B |
  |---|---|
  | N0' | 6 real epochs |
  | N-reg | as N0', with phase-B dropout 0.1 or 2x WD |
  | H | 4 real epochs + 4M synthetic words, CE on the samples |
  | L | as H, with alpha = 0.7 and the label = G2's unblocked distribution via an exact frozen-teacher forward (the original premise) |
  | N-fresh | 4 real epochs + 4M fresh words from train_10M minus train_2M (the exchange-rate anchor; proxy only) |
  | N-realKD1 | L's loss on 4M real words |

- **1b, only if component 1's code exists.** The population at 2M: the arm, N-hard, N-realKD and N0'.

Metric: delta = max(2 x paired seed SD, 0.04). L* = min(B2, N0', N-reg). r = (L* - H) / (L* - N-fresh).

Registered predictions:
- r in [-0.2, 0.2];
- L - H in [-0.02, +0.02];
- in 1b, arm - N-realKD in [-0.02, +0.03].

Decisions:
- If |L - H| < delta, the cached-label premise is "killed (proxy)". This is the expected outcome.
- If r < 0.3, the ceiling stays at or below 0.15.
- If H is within delta of N-reg, the effect is labelled "regulariser".
- The S screen is used only if the 1b arm beats N-hard, N-realKD and L* by delta at both seeds. Otherwise the card reads "killed (proxy)", which is a label, not a verdict.

## Implementation (harness)
- **`babylm/a4/generate.py`** (extend):
  - `--members round_robin`: row i is sampled from member i mod K, using component 1's per-member logits;
  - `--log_blocked_mass`;
  - a per-row seed document id for the audit.
  - It keeps the KV cache, the exact 8-gram block (`a4/ngram.py`), rows without fed seeds, and the `a4/data.py` mixed-stream writer.
- **`babylm/a4/synloss.py`** (new): a per-row source tag from the stream, plus:
  - `--syn_loss {ce, kl_live, kl_frozen, smooth}`, `--syn_alpha`;
  - `--syn_smooth {uniform, unigram}` and `--syn_match {H, off}`, with epsilon solved from the arm's logged `syn_H` / `syn_off`, like A1's `--null_eps_from`;
  - `--syn_teacher`, using `babylm/moonshot/kd.py` `load_teacher` / `kd_terms` (exact full-vocabulary KL, no cache);
  - `--kd_rows real --kd_rows_words 19.9e6` for N-realKD.
- **`babylm/harness/train.py`:**
  - loss chosen per row source;
  - exposure counters by source (real, synthetic, generator) fed to `a4/ledger.py`;
  - a FLOPs counter that includes generation and teacher forwards;
  - resume with the mixed stream and the second LR cycle (A4's path).
- **Population form:** uses component 1's model class (shared trunk, K members) and its averaged-probability export.
- **Audit and verdict:** `a4/audit.py` (leakage, 6-gram, edit distance against train_10M and the seed document); `a4/verdict.py` extended with the new nulls.
- **Jobs and outputs:** jobs `babylm/configs/gcp/a4p_stage1.json` and `a4p_screen.json` through `run_jobs.py`. Outputs go to `shared/babylm/attacks/A4/a4p_<run>/`, with curves as CSV and PNG.
- **CPU tests** (`babylm/tests/test_a4p.py`):
  - `syn_loss ce` with alpha = 0 reproduces A4a bit for bit;
  - `kl_frozen` at q = p gives zero gradient, and a Monte-Carlo check confirms E[grad CE] = grad KL (the identity as a unit test);
  - smoothing epsilon matches the logged H and off-mass;
  - the ledger stays at or under 100M, by source;
  - round-robin member assignment is correct;
  - resume is bit-exact;
  - the N-realKD subset is fixed by its seed.

## How each skeptic's fixes were folded in
- **Prior art (wounded).** Adopted:
  - relabelled as a known idea (Seq-KD + Word-KD, GAL, MUNGE);
  - the Kim/Kotha and EACL citations corrected;
  - the base fallback removed;
  - N-realKD added;
  - sampling from single members;
  - the copy-block mismatch fixed with alpha = 1 and an unblocked label;
  - the cache question removed (live or exact labels);
  - ceiling set to 0.15;
  - synthetic share at 33% of phase B (20% overall);
  - N-realKD added to the mini-ruler.
- **Rules (wounded).** Adopted:
  - the generator registered and gated on ref-0110;
  - seeds not fed;
  - an annealed generator with its DEV pre-registered;
  - component 1 alone added to the binding comparator;
  - N-shuf replaced by the four smoothing nulls, plus N-realKD;
  - the frame-transplant claim dropped;
  - type_stats-only weights;
  - A4's leakage audit;
  - FLOPs stated per form;
  - the provisional-base label;
  - the export rule;
  - proxy runs labelled "killed (proxy)" at most.
- **Mechanism (dead as a distinct component).** Adopted:
  - the identity stated;
  - stale cached labels replaced by live labels;
  - the bursty tilt dropped;
  - tilts measured against the frozen run;
  - Stage 0 and Stage 1, with N-reg and N-fresh, and 1b added.
- **Not adopted:** N-reg at size S. The frozen tuned-WD base and the smoothing nulls already test "just regularisation" there.
