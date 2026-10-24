<!-- moonshot design round W1; status alive; ceiling 0.05 nats/word; expected 0.004; P(pass screen) 0.04; screen GPU 45 A100-min. Draft card: not yet submitted to the critic. -->

# Attack NR: novelty-replay, consistent pseudo-word substitution of rare words (card, 2026-10-10)

**Status.** Card only: no code and no runs. All three skeptics (prior art, rules, mechanism) found it wounded and none found it dead. This card folds in their fixes.
- The original diagnosis, that repeated epochs erode the probability of new words, is **withdrawn**. It is false on the tuned base.
- The card survives on two smaller deficits, re-measured on the tuned base (F1 below). A GPU proxy gates it before any S screen.
- Renamed from "C1" so it does not collide with the ceiling map's C1/C2/C3 diagnostics.
- Base: the round-1 winner `base_S_wd0p4_lr0p5_r1_ve0_tied` ("b04"; provisional until frozen).
- Card writer's CPU scripts and dumps: `scratch/nrcard/` (f1.py, sub_dump.py, did.py, hapax.py, rate.py).

## Known ideas, new combination (corrected prior-art line)
Every part of the mechanism has been published before.

**Nearest relative.** Knowledgeless LMs (Cohen et al., arXiv 2607.12831, Jul 2026).
- Setup: from-scratch pretraining, SmolLM 135M-1.7B and Llama 1B/3B, 2.5-20B tokens. Entities are replaced by random 10-character strings that stay consistent within each document and are trained as ordinary targets.
- Results: closed-book LAMA falls from 34.7 to 3.9 (SmolLM-1.7B); contextual SQuAD rises from 55.9 to 59.8. Their eval loss is not comparable across text formats.

**Count-dependent noising, targets included.**
- Xie et al., ICLR 2017, "Data noising as smoothing" (arXiv 1703.02573). The Kneser-Ney variant noises both context and target. PTB medium 80.4 to 76.9.
- The rho = a/(a+c) shape is word dropout from Kiperwasser & Goldberg, TACL 2016.
- Dyer et al. 2015 replace singletons with UNK.
- SwitchOut (Wang et al. 2018) replaces targets as well.

**Fresh symbols per episode to force in-context learning.**
- Chan et al. 2022 (arXiv 2205.05055): bursty and rare items drive in-context learning, which "traded off against" in-weights learning.
- Kirsch et al. 2022.
- Lake & Baroni 2023 (MLC).
- Singh et al. 2023 (arXiv 2311.08360): in-context learning fades with overtraining.
- Minnow (Wang, Jiang, Linzen & Lake, EMNLP 2025, arXiv 2502.14791): BabyLM-10M from scratch, but 30 epochs (past our cap) and one shared placeholder token. It reports no LM loss and no from-scratch BLiMP; its only BLiMP check is a finetuned Llama-3 8B (83.5 vs 83.3).

**Consistent renaming.**
- DOBF (Lachaux/Roziere et al. 2021), for code identifiers.
- Entity anonymisation: Hermann et al. 2015, critiqued by Chen, Bolton & Manning 2016.
- Random stems copied consistently from input to output: Anastasopoulos & Neubig 2019.

**Generator and new-word models.**
- The splice generator is the Wuggy method (Keuleers & Brysbaert 2010).
- Spelling and reuse of new words: Kawakami, Dyer & Blunsom 2017; Mielke & Eisner 2019.
- Positional UNK copying: Luong et al. 2015.
- Good-Turing estimation: Good 1953.

**BabyLM.** Skeptic 1 checked the 2023-2025 proceedings tables of contents and found no nonce, pseudo-word or anonymisation pretraining entry. The nearest:
- WhatIf (2024.conll-babylm.20): lexical substitution, "small but consistent" gains.
- RecombiText (2025.babylm-main.40): mixed.
- Frequency-informed masking (2025.babylm-main.38): below baseline.

**In project.** `docs/discovery_session_3.md` line 558 (SCRIBE Phase A) already specified "rare words ... replaced, per example, by fresh pseudo-words".

**What is new.**
- Rare carriers chosen by frequency, not by NER.
- Spliced strings, not random ones.
- A capped Good-Turing rate.
- All inside a 10-epoch, 100M-exposure cap, scored on clean LM loss.

**Label: known ideas, new combination** (all three skeptics agree).

**Prior-art map update requested.** Entity anonymisation as pretraining *has* now been measured: recall collapses and contextual QA improves. Only its effect on clean LM loss is unmeasured.

## What the review changed (with the measurements behind it)
1. **The erosion diagnosis is withdrawn.**
   - The card's evidence (100M x10 vs x1: 19.45 vs 18.48 on words neither model saw) came from WD-0 port runs.
   - On b04, every new-word slice improves to the final checkpoint. DEV-sub, overlap removed, after 10M / 30M / 50M / 100M words seen (epochs 1/3/5/10):

     | Slice | 10M | 30M | 50M | 100M |
     |---|---|---|---|---|
     | Exact-string never-seen, first occurrences | 21.41 | 20.65 | 20.91 | 19.65 |
     | Key-unseen, first occurrences | 25.92 | 25.08 | 25.61 | 24.49 |
     | All words | 5.903 | 5.543 | 5.527 | 5.126 |

   - The wd2.0 tied cell also ends at its best on every slice.
   - Erosion appears only where everything overfits. In the WD-0 tied round-1 cell, never-seen first occurrences go from 20.01 to 27.11 between 20M and 100M words, while all words go from 5.368 to 6.348.
   - Measured by skeptic 3; the card writer re-ran it from the per-word dumps.
   - So skeptic 2's zero-GPU gate fires against the original diagnosis. The card is kept only in the form skeptic 3 proposed: calibrate new and rare word slots, and train in-window reuse.
2. **The rate was set on the wrong key.**
   - 2.05% is the exact-string hapax mass, mostly case and punctuation variants.
   - Registered key ("key A"): lowercase, with leading and trailing non-word characters stripped. Its Good-Turing mass is **0.79%** (78,565 hapax types / 9,953,773 words). The all-punctuation-stripped key gives 0.73%.
   - The 1% and 4% arms are dropped.
3. **Seen rare words were being starved.** At 2.05% the card replaced 28-30% of rare occurrences per visit and 77-79% of hapax visits. Rho is now capped at 0.3.
4. **Targets are re-sized on b04** (next section):
   - 100M-seen words removed (data, not novelty);
   - the double-counted count-0 repeats removed;
   - repeats more than 300 words back dropped (outside the eval context).
5. **Nulls rebuilt.**
   - Added N3 (novelty null) and N4 (spelling null).
   - N2 is relabelled as an anti-copy control, run at the proxy only.
   - All nulls share one occurrence set.
6. **Kill rule tightened.**
   - Key fixed now.
   - Harm guard on the 1-63 bands.
   - Proxy at 5M, judged against the base, not N1.
   - One registered arm.

## Mechanism (as registered)
**Key and carriers.**
- Key A: lowercase, with leading and trailing non-word characters stripped.
- Carriers: types whose key is purely alphabetic (`[^\W\d_]+`) with train_10M key count 1-63. They are 6.0% of training words.

**Window.**
- The unit is each training window's T+1 = 1025-token buffer, split at BOS into document segments.
- An occurrence is eligible only if all its tokens lie inside the buffer, so no word cut at either window edge is touched.

**Selection.**
- Applies from epoch 2; visit 1 is always clean.
- Each distinct carrier key in a segment is selected with rho(c) = min(0.3, a/(a+c)). a ≈ 1.76 is calibrated at build so the realised replaced mass, after feasibility drops, equals the key-A Good-Turing mass of 0.79% of words.
- Resulting rates: rho(1) 0.30, rho(5) 0.26, rho(20) 0.08, rho(63) 0.03. 13% of carrier occurrences are replaced per visit. A hapax is still seen intact about 7.3 of 10 visits.
- Draws come from PCG64([seed, epoch, window start, key hash]). They are independent of the arm.

**Replacement.**
- All occurrences of a selected key in the segment get one pseudo-word, drawn fresh for this (epoch, window, key).
- A pseudo-word is a prefix (>= 2 characters) of one carrier type spliced to a suffix (>= 2 characters) of another.
- It is rejected if it is a train_10M key, or if it was already used anywhere in the run, so no pseudo-word recurs across windows.
- Each occurrence keeps its own case pattern, attached punctuation and leading whitespace, and is tokenised with the canonical tokenizer.
- The pseudo-word must give the same token count as the original in every surface form in the segment. Up to 64 draws are tried; otherwise the key is dropped for **every** arm. This is the shared feasibility pass.

**Training.** Substituted tokens are ordinary targets:
- the first occurrence trains P(new word | rare slot) and a spelling prior for novel strings;
- later occurrences train multi-token in-context copying of a string known only from this window;
- the text that follows trains use of a word learned in the window.

**Invariants (asserted per window).** Token count, word-initial count, and word ids under prepare.py's rule are unchanged. Eval, export, scoring and the scored network are unchanged.

## Why (measured slices on b04)
Sizing comparator: full DEV, overlap removed, b04 vs the 100M x1 port model. That model is untuned; it is used here only for sizing, never as a 10x reference. The total gap is 0.647 nats/word.

**1. Key-unseen first occurrences** (all-punctuation key): 1.10% of words, 24.21 vs 19.00 nats, contribution 0.057.
- F1 split (card writer; 25% of DEV-sub windows, re-scored on CPU; matches the harness dumps, 5.1212 vs 5.1213 nats/word):
  - the head token (first non-whitespace token) carries a gap of 1.06 nats per occurrence;
  - the remaining tokens of the word (its spelling) carry 4.14.
- Head gap x 1.10% = **0.012 nats/word**.
- The head gap is the same on seen rare words (count 1: 1.00; counts 1-63: 1.04) and 0.67 on words seen 64+ times. So the novelty-specific part is about 0.4 nats per occurrence, roughly 0.004-0.005 nats/word.
- The spelling part (about 0.045) needs letter sequences from 10x more types. Splices of the 10M types cannot supply them.

**2. In-context repeats within 300 words.**

| Slice | Share of words | b04 | 100M x1 | Contribution |
|---|---|---|---|---|
| Key-unseen repeats | 0.33% | 7.93 | 5.39 | 0.008 |
| Key 1-63 repeats | 1.14% | 6.35 | 4.71 | 0.019 |

- Words neither model saw, repeated within 50 words: 5.76 vs 4.27 on the DEV-sub windows. The word-internal (copy) tokens are 2.32 vs 1.57. Total contribution is only 0.002.
- In absolute terms the base gains *more* from an earlier occurrence than the 100M model does (neither-saw words, first occurrence to repeat: 23.30 to 5.78 vs 18.48 to 4.13). So there is no gross copy deficit, only a residual multi-token one.

**3. A free copy cache already takes about a third of the repeat slice.**
- The cache: zero exposure, on b04, bigram pointer plus unigram cache inside the same 1024-token window. Lambda is fitted on 1/8 of the DEV-sub windows and scored on a disjoint 1/8.
- It gains +0.036 nats/word overall (context_v1 reports +0.037).
- On key 0-63 repeats within 300 words it gains +0.012, which is 37% of that slice's 0.031 gap. A training mechanism has to beat this.

**Excluded from the targets.**
- Words in train_100M but not train_10M (0.065): data, not novelty.
- Key-seen case and punctuation variants (1.50% of words).
- Repeats more than 300 words back.
- The 64+ bands.

**Why the base should not already have this.** The base keeps improving on these slices but never sees a string that is known only inside its window. A bigger or wider base would not create such events either.

**Untested bet.** That training events with in-window-only strings teach slot calibration and copying that transfer to real unseen DEV words.

## R0 exposure ledger
- **Base:** one model, one forward per window, 10 epochs x 9,953,773 = **99.54M words**.
- **NR and its nulls (N1, N3, N4, plus N2 at the proxy):** same windows and the same words fed. Each substitution is one whitespace word for one whitespace word. Token count, word-initial count and word ids are asserted equal per window, so the milestone counter is unchanged.
- **Carrier table, pseudo-word generator and N3 substitute table:** string operations on the train_10M type list and its counts. No model is involved and there is no training pass. These are the counts A3's rare scope uses, accepted without an exposure charge (ref-0063); this card still asks for its own explicit critic entry. If the critic charges them as a 10M count pass, every arm and the base run 9 epochs (89.6M + 10M = 99.6M).
- **Outside material:** no outside data, model or tokenizer. Pseudo-words are never filtered against DEV or eval items; collisions are reported.
- **Total:** <= 100M.

## FLOPs (R4.4)
- **1.00x the base's training FLOPs:** same tokens, same forward and backward passes, no extra model.
- N1 only masks targets, so it has the same FLOPs.
- The CPU patch build is reported separately; it is expected to take seconds per epoch.
- No FLOP-matched reference is needed.

## Null twins
**Shared across all nulls.**
- The identical selected occurrence set (shared selection and feasibility pass).
- Same seed, data order, schedule, GPU type, vocab (16384), init stream and code path.
- At S, the **best of N1, N3 and N4** binds.

**N1: target null.**
- Same substituted inputs, but the substituted target positions are set to -1 (ignored).
- The loss is the mean over the remaining targets, and the normalisation is logged.
- What it removes: the new-word prediction and reuse targets.
- What it keeps: the input corruption.

**N3: novelty null** (skeptics 1 and 3).
- Each selected key is replaced, consistently within the segment, by a real train_10M carrier type with:
  - the same count band (1 / 2-3 / 4-15 / 16-63);
  - the same token count per surface form;
  - no other occurrence in the window.
- Targets are kept.
- What it keeps: consistency, the slot, copying, and lexical-substitution regularisation (Xie 2017, WhatIf).
- What it removes: the novelty of the string.
- If N3 does as well as NR, the gain is substitution noise, not new-word learning.

**N4: spelling null** (skeptic 2).
- Consistent random token strings with the same token count.
- The first token is drawn from tokens with the original's leading-space and case class; the later tokens from word-internal tokens.
- Targets are kept.
- What it removes: English-like spelling.
- What it tests: generic tail mass in rare-word slots, which label smoothing or a temperature would also buy.

**N2: anti-copy control** (proxy only; diagnostic, never binding).
- Each occurrence gets its own independent pseudo-word.
- Skeptic 2 showed it adds anti-copy information, so beating it proves nothing. It is reported on the repeat slices only.

**Not adopted.** Skeptic 3's frequent-word variant of N3 (one frequent real word per window). The rare-band version matches the difficulty of the slot, and the screen already has three binding nulls.

## Kill rule (registered before any run)
**Protocol.**
- Size S, b04's recipe and seed 0, train_10M x 10 epochs.
- The base is re-run through the NR code path at rate 0, in the same batch on the same GPU type.
- Metric: **full-DEV nats/word from `dev_bpw.py` on the final checkpoint**, raw with the overlap-removed value beside it. No checkpoint picking; DEV-sub is diagnostics only.

**Kill if any of these holds:**
- (a) NR's gain over the base is < 2x the registered seed floor.
- (b) NR does not beat the best of N1, N3 and N4 by 2x the floor.
- (c) BLiMP-fast drops by >= 2 sigma.
- (d) Harm guard: on full DEV, the first-occurrence loss of key-A count 1-15 or 16-63 words is worse than the base's by > 1 sigma of that slice. The slice sigma comes from the floor seeds' per-word dumps and is registered with the floor.

**Veto at promotion.** A human-likeness or WUG drop of > 2 sigma (18:37Z rule).

**Mechanism check** (registered now; key A fixed).
- >= 40% of NR's full-DEV gain must fall on key-A count-0 words (first occurrences and repeats) plus key-A 1-63 in-context repeats within 300 words.
- Otherwise the result is reported as a win for another reason and is not promoted under this card.
- Bands reported under key A: 0 / 1 / 2-3 / 4-15 / 16-63 / 64+. Exact-string case and punctuation variants are reported separately.

**Process.**
- No verdict before the floor is registered.
- Provisional base, ruling ref-0086(b): a base change in WD, EMA or MNTP forces one re-screen. Until then a kill reads "killed (provisional base)".
- Promotion: the single arm, with the screen's best null as the binding twin. 3 seeds, size M, official eval, R0-R6.

## Odds
- **Ceiling: about 0.05 nats/word.** Everything would have to go right:
  - head-token parity on key-unseen first occurrences, 0.012;
  - the key 0-63 in-context repeat gap, 0.027, including the share the free cache already takes;
  - at most about 0.01 from a novel-string spelling prior.
  - It can do nothing for the 64+ bands or for the spelling knowledge the 100M model has.
  - At best that is about 1.15x data, a small, well-aimed lever, not a 10x.
- **Median: +0.004 nats/word** (range -0.015 to +0.02). That is 0.005-0.015 of gain minus 0.005-0.015 of cost to seen rare words.
- **P(pass at S): about 0.04.** Skeptics' estimates: 0.09, 0.08 and 0.05.

## Fastest falsification (in order; each step only if the previous passes)
**F0 (done, zero cost).** The erosion diagnosis is false on b04 (above).

**F1 (done; CPU; about 45 CPU-minutes).** Thresholds were registered by skeptic 3 before the measurement. Neither leg is killed; the first only narrowly.
- First-occurrence leg: head gap x 1.10% = **0.0116 >= 0.010**, so it survives.
- Reuse leg: the copy cache recovers **35-37% < 50%** of the in-context rare-repeat gap, so it survives.

**F1b (CPU, about 30 minutes, needs full logits; before any GPU).**
- A zero-exposure word-start control on b04: one scalar is fitted on half of DEV-sub that shifts mass toward word-initial rare-piece tokens. Either a word-start temperature or a Good-Turing tail reweighting of word-initial tokens. It is scored on the other half.
- If it recovers >= 50% of the head gap on key-unseen first occurrences (>= 0.5 nats per occurrence), kill the first-occurrence leg.
- If both legs are dead, kill the card.

**CPU tiny run plus unit tests (about 15 minutes).** Checks the invariants listed under the implementation plan.

**F2: GPU mini-ruler proxy.** A label only; it counts as trials (ref-0110).
- Data: train_5M x 10 epochs at S with the b04 flags. 5M rather than 2M, because priors inflate at 2M.
- Six runs on one GPU type: base (NR path, rate 0), NR, N1, N3, N4, and N2 as a diagnostic.
- Carriers, the rate and the pool come from **train_5M counts only**.
- Slices are read from the DEV-sub per-word dumps at the final checkpoint, overlap removed, defined by train_5M key counts.

**F2 kill rule.** Kill unless all of these hold (per-occurrence deltas):
- (i) NR minus base on key-unseen first occurrences <= -1.0 nats;
- (ii) NR minus base on key-unseen repeats within 300 words <= -0.3 nats. This is judged against the base because N2 is contaminated; N2 is reported beside it;
- (iii) NR minus base on key 1-63 first occurrences <= +0.2 nats;
- (iv) NR beats the base and the best of N1, N3 and N4 on total DEV-sub, with no slack;
- (v) the projected full-DEV gain is >= 0.02 nats/word. The projection is the sum over slices of (NR minus base per occurrence) x the full-DEV share. Shares: key-unseen first 1.10%, key-unseen repeats within 300 words 0.33%, key 1-63 first 4.23%, key 1-63 repeats within 300 words 1.14%, and 64+ for the rest. `nr_slices.py` recomputes the shares under key A before the run.

A pass experimenters to the S screen: base at rate 0, NR, N1, N3 and N4.

## Implementation plan (harness; about 1.5 agent-days)
1. **`babylm/harness/novelty_replay.py`** (about 280 LOC).
   - `build_tables`: for each word id of the split, the key id, count, carrier flag, case class and punctuation spans. Also the N3 band lists and the splice material, with a recorded hash.
   - `plan_epoch(mode, seed, epoch)`, over `EpochSampler.plan(epoch)`, which is deterministic. For each window: build the T+1 buffer, split it into segments at BOS, find eligible carriers, make the arm-independent selection draws, run the shared feasibility pass, and produce replacement ids for mode `nr|mask|real|rand|indep|off` plus the mask of substituted targets.
   - Output per (mode, epoch): a patched uint16 token stream (about 32 MB) plus a target-mask bitmap, cached with sha256 like A3's per-epoch streams. They are built on the VM's CPUs before training. Windows of one epoch tile the stream and edge words are never replaced, so one patched copy serves every window of that epoch.
   - A run-level used-string set and these stats: replaced share by band, reuse events per 1k words, feasibility drop rate, and collisions with DEV types and with eval nonce vocabulary (WUG stems), reported and never filtered.
2. **`stream.py`** (about 15 LOC). Add `EpochSampler.next_starts_with_epoch`, so a batch that straddles an epoch boundary reads each window's own epoch stream. PlanSampler is not supported in v1 (asserted).
3. **`train.py`** (about 50 LOC).
   - Flags: `--nr_mode {off,nr,mask,real,rand,indep}`, `--nr_mass gt|<float>`, `--nr_cap 0.3`, `--nr_max_count 63`, `--nr_from_epoch 2`, `--nr_seed`.
   - Gather from the epoch's patched stream; in mask mode set y = -1 at substituted positions.
   - `off` is bit-identical to the base path; rate 0 runs through the same path for the base re-run.
   - Table and stream hashes go into config.json; the used-string set and generator state go into resume.pt.
   - MNTP is off in b04. If the frozen base adds it, patching comes first, and the effective causal-row rate is logged.
4. **`babylm/analysis/nr_slices.py`** (about 120 LOC). Key-A slices from the per-word dumps, the head/rest split from the per-token dump, the F2 projection and the mechanism check. It ports the card writer's F1 scripts and adds F1b.
5. **`babylm/tests/test_novelty_replay.py`** (about 180 LOC). Tests:
   - token count, word-initial count and word ids unchanged per window;
   - consistency within a segment;
   - edge words never replaced;
   - no pseudo-word is a train_10M key, and no string is reused across windows;
   - identical occurrence sets across `nr`, `mask`, `real` and `rand`;
   - rate 0 is bit-identical to `off` on a CPU tiny run;
   - resume is bit-exact;
   - N1's loss normalisation is correct.
6. **`babylm/configs/make_nr_jobs.py`**. The F2 proxy job list (train_5M counts) and the S screen job list, mirroring `make_a3_jobs.py` (PR #61).

## Stacks with
- **Experimenter's imprinting card.** Imprinting covers seen rare words; NR covers unseen words and in-window reuse. Both draw on the rare band's visits, so a combined arm needs its own harm check.
- **A1 temporal self-distillation.** Orthogonal, since soft targets add no new strings. A combination must skip substituted positions in A1's position-keyed cache.
- **A2 input noise.** Conflicts, since both corrupt inputs. A combined arm caps the total corrupted share and needs its own card.
- **A3 BPE-dropout.** Same rare band. Combine only after both pass alone, with NR's token-count matching run after A3's re-segmentation.
- **Band-factored-softmax candidate** (formerly "C3"). Pseudo-words carry the count-0 band label.
- **Lexical-gibbs-resample candidate** (formerly "C2"). It cannot create contexts for never-seen words; NR targets exactly those.
