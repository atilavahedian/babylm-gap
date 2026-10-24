<!-- moonshot design round W3; status alive; ceiling 0.25 nats/word; expected 0.02; P(pass screen) 0.05; screen GPU 240 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card: trained train_10M memory reader (TRIME_ext with document-excluded reads, strict-small)

Status: card only. No code, no runs. Gated by a zero-training falsifier (F0) that must pass before any GPU training.
Folded from three skeptic verdicts (prior-art, rules, mechanism). All three said **wounded** and none said dead. No R0 or R4.4
rule kills the card once the fixes below are applied, so it stays alive. It is **no longer a 20x-class mechanism**:
its honest ceiling is about 0.25 nats/word, and its median sits about at the known kNN-LM floor it extends.

**Known idea, untested regime. Not new.** The mechanism is TRIME_ext (Zhong, Lei & Chen, EMNLP 2022, arXiv 2205.12674)
almost verbatim:
- the memory-augmented softmax exp(h·E_y) + Σ_{m: y_m = y} exp(sim(k, k_m));
- in-batch training memory with BM25-similar segments in one batch;
- local memory dropped with p = 0.9 to force cross-segment use;
- backprop into the memory keys;
- at test, a FAISS datastore over the whole training set, interpolated with a tuned λ.

The one delta, document-level exclusion at p = 1, is RETRO's training rule ("we filter out neighbours originating from
the same document as the training sequence", Borgeaud+ 2022 §2) and REALM's "prohibiting trivial retrievals" (Guu+ 2020).
NPM (Min+ 2022, arXiv 2212.01349) uses the same in-batch nonparametric objective over other sequences. The
"corpus-specific bits in memory, regularised trunk learns to use evidence" framing is also published: Khandelwal+ 2020
("Generalization through Memorization"), RetroPrompt (Chen+ 2022), Memory^3 (2024), Pouransari+ 2025.

Its closest direct test at BabyLM-like size was null or negative for the trunk. Samuel, Charpentier & Wold (NAACL 2024),
retrieval-augmented pretraining at 9M/28M/98M params, scored without retrieval:
- BLiMP: 73.4→73.3, 78.6→78.8, 82.9→83.2 (noise)
- LAMBADA: 25.3→19.3, 35.1→26.2, 46.0→37.5
- GLUE at 98M: 82.2→81.2

Caveats on that test: a masked LM, 400M Wikipedia words, an idealised retriever.

No BabyLM 2023-2025 submission used TRIME or kNN-LM. The closest BabyLM-style measurement is a post-hoc kNN-LM, Liu & Kim
2026 (arXiv 2608.23851): 124M params, ~50M words, 10 epochs, PPL 14.88→14.30 (−0.040 nats/token, 1 seed). Minimal
pairs: low-frequency +0.14, high-frequency −0.013 (CI spans 0).

The R6 label is therefore **"known idea (TRIME_ext + kNN-LM; document exclusion = RETRO/REALM filter), untested at
strict-small under an exposure cap"**. It may move to "known ideas, new combination" only if the N-noexcl arm shows
the document exclusion adds > 2x floor. It can never be "new".

In-house flag: `docs/sprint_1_prior_lessons.md` §4 lists post-hoc kNN/count memory and state-keyed memory tables as
closed families ("do not propose again without a specific new twist"). On the pi memory-toy bench:
- the kNN cache gave +0.0012;
- the S2 trunk-keyed count memory faded 0.0057→0.0036→0.0033 as the trunk trained;
- M2 one-shot writes were killed by their scrambled null;
- Bet 1 "train through the count mixture" (MT) was −0.034 against the post-hoc blend.

That is a different bench, so it transfers weakly. The twist claimed here is a *trained* reader on the real BabyLM bench.
F0 decides whether that twist earns any GPU training.

## Mechanism (as it will be built)
1. **Trunk + key head.** Size S, base recipe. A key head W_k (384→128, no bias) reads the head input h_i (the final
   RMS-normed hidden state from `hidden()`), so k_i = W_k h_i. The keys come from the **same forward** as the logits:
   no momentum encoder, no EMA key encoder, no recomputed keys.
2. **Training loss (TRIME memory softmax).** For position i with target y:
   `L_i = −log[exp(z_{i,y}) + Σ_{m∈S_i, y_m=y} exp(s·k_i·k_m)] + log[Σ_v exp(z_{i,v}) + Σ_{m∈S_i} exp(s·k_i·k_m)]`.
   - z are the base's softcapped logits; memory scores are softcapped at the same 15.
   - s is a learnable scale, initialised to 1/√128.
   - S_i is 8,192 positions sampled from the *other* windows of the same micro-batch, minus every position whose
     **document cluster** equals i's.
3. **Document clusters.** Document ids come from the cumsum of BOS positions in the train stream (13,713 docs, int32).
   They are **never** taken from `src.bin`, which is a uint8 source id; masking by it would exclude a whole source.
   Near-duplicate documents (CHILDES transcripts, subtitle episodes) are merged into clusters by 13-gram Jaccard
   ≥ 0.5, a threshold fixed before any run, so a near-copy of one's own document is not readable.
4. **BM25 batching.** BM25 runs over window-sized chunks (the offset-0 tiling, ~15.6k chunks of 1,024 tokens), not over
   documents: Gutenberg is 51 books holding ~25% of words, so book-level BM25 is meaningless. It is implemented in
   numpy/scipy, with nothing installed. Each epoch's EpochSampler window list is **permuted** so that half of each
   batch pairs with its unused BM25 neighbours. Every window still appears exactly once per epoch (asserted).
5. **Weight decay = the base's.** The "WD tuned up so the trunk leans on memory" step is **dropped from the screen**. It
   collides with the trunk-alone gate and the trunk-only GLUE leg, because the base WD is already DEV-optimal. It may
   come back later as its own registered arm, with the base's WD grid containing the same value under an equal trial
   count.
6. **Datastore.** After training, one inference pass over train_10M (dev_bpw windows, ctx 512 / stride 256, every
   position once) stores the 128-d fp16 keys and uint16 next-token values: 16M × 128 × 2 B ≈ 4.1 GB. **One exact
   pure-torch index** (the knn_diag search) is used identically for DEV and for the official suite. There is no
   FAISS, IVF or PCA in any scored path; FAISS is not in the certified ref-0243 env and approximate search threatens
   R5 determinism.
7. **Test-time scoring.**
   - Read top-k' neighbours (k' ∈ {16, 64, 256}) and form p_knn = softmax over −d/T aggregated by value.
   - Mix p = (1−λ)·p_model + λ·p_knn, with p_model the trained parametric softmax.
   - T, λ and k' are fitted by **2-fold document cross-fitting on DEV**: fit on even docs and score odd docs, then the
     reverse. Every full-DEV token is scored with knobs not fitted on it. DEV-sub (a subset of DEV) is never used for
     knobs.
   - The λ-free joint softmax is reported beside it.
   - Memory = train_10M only. There is no in-batch memory at eval, and no DEV or eval token ever enters memory. The
     512-token protocol is unchanged.

What changed from the proposal, and why: the "learning as inference / two-system" rationale is withdrawn. With document
exclusion, the parametric term must still fit every document's own tokens for 9 epochs, so per-document memorisation
pressure is unchanged. What memory can absorb is *cross-document, generic* evidence, which is the part that
generalises. The gradient therefore moves generalising regularities out of the weights, and the document-specific
residual (the 0.51→1.11 dev-minus-train excess) stays in them. That is the opposite of the stated aim. The rules and
mechanism skeptics both found this, and Samuel 2024 is consistent with it. The trunk-damage gate (K6) tests it directly.
What remains is a trained kNN reader whose keys are learned for retrieval.

## Why (which measured slice)
A kNN reader over the same 10M (h, next-token) pairs adds no information absent from training. It can only move mass to
tokens that followed similar training contexts. On GPT-BERT 10M (published), the train_10M count model gains per word:

| Band | Gain (nats/word) |
|---|---|
| Seen once | 0.389 |
| Seen 2-3 | 0.368 |
| Seen 4-7 | 0.163 |
| 64-255 | 0.038 |
| 256-1k | 0.041 |
| 1k-8k | 0.041 |
| Never seen | −0.072 |

The copy cache gains +0.037 (repeats +0.114, first occurrences −0.025). Count model and cache together give +0.076.

Slices it can act on: words seen 1-63 times (0.16-0.18), repeats and near-copies (~0.04), wiki entity names, and
CHILDES formulas. It acts only weakly on the biggest band (words seen 64-8k times, 0.36), which is familiar words in
*new* contexts, where retrieving old contexts is weakest.

The document structure limits the cross-document story. 12,285 of 13,713 docs are simple_wiki (14% of tokens; 13.5% of
DEV words). For ~85% of tokens the "other documents" are a weak random pool.

Expected signature: rare tilt high, novelty tilt ~0.3-0.6. That looks like extra epochs (0.43-0.76), not like data
(1.04). The novelty check is therefore a **binding kill** here, not a label.

External relatives, in nats/token (word-level on WT-103):
- TRIME_ext without long memory over its base: 18.70→15.55 = 0.18. This is at 247M params with a 103M-word datastore,
  ~6x ours.
- TRIME_ext over its own post-hoc kNN-LM: 16.37→15.55 = 0.051.
- At 7M params: 83.66→42.36 = 0.68 over the base, but only 0.079 over kNN-LM + continuous cache.

Ablations at 7M (dev PPL, TRIME Table 11): 41.50 with everything; 45.71 without BM25 batching; 45.15 without backprop
into memory.

Note: TRIME's 18.70→17.76 (0.052) is the *local, same-segment* memory, which document exclusion removes, so it does not
transfer. The proposal's citation of it is withdrawn.

Datastore scale counts against us. Singh+ 2026 (arXiv 2604.00715) find ~91% of the gain arrives by ~1 datastore token
per parameter. 16M tokens against ~30-37M params is at the low end.

## Ceiling (mechanism class, not the generic Bayes class)
**About 0.25 nats/word (range 0.18-0.30).** It adds up as follows:
- the rare 1-63 band, up to 0.17;
- copying, ~0.04;
- at most a quarter of the 64-8k band, ~0.09;
- the WT-103 247M analogue gives 0.18 with a 6x larger datastore.

The proposal's 0.9 is the bound for *any* learner on 10M words, the weight-only base included, so it says nothing about
this mechanism. It is withdrawn. So is the Minnow "10-100x in-context sample efficiency" claim: Minnow needed its own
meta-training objective at 30 epochs (over the cap) and measured 4-way word discrimination, not LM loss. A retrieved-chunk
(RETRO/RALM) design that puts evidence into the context would be the class the 0.9 rhetoric needs. Under R0 those fed
words count as exposure, so it would be a different card.

## R0 exposure ledger (registered reading; ruling requested before any run)

| Item | Words | Note |
|---|---|---|
| Unique words | 9,953,773 | train_10M only |
| Trunk training, 9 epochs | 89,583,957 | in-batch memory reads use the same forward's h: several losses on ONE forward count once (R0.2) |
| BM25 ordering | 0 | uncharged only because each epoch is a permutation of the EpochSampler windows; the exposure counter must be bit-equal to a 9-epoch base |
| Datastore pass (one inference forward over train_10M) | 9,953,773 | charged 1 epoch under the count-model precedent; the proposal's "if ruled free, 10 epochs" branch is **refused** |
| **Total** | **99,537,730** | ≤ 100M |

Further rules:
- One datastore per scored model.
- Intermediate and AoA checkpoints are scored **trunk-only**, labelled; rebuilding a datastore for them is not free.
- No teacher, no generated text.
- A4 synthetic text never enters memory.
- DEV, test and eval items never enter memory (enforced by a test).

Ledgers of the other rows:
- **Nulls:** N-shuf and N-noexcl carry the identical ledger. N-self uses in-window memory at test, so no datastore:
  89.6M.
- **Known-idea comparators:** B9 + post-hoc kNN-LM = 89.6M + 10M; B9 + count model = 89.6M + 10M.
- **Bases:** B10 = 99.5M. The frozen base = its own ledger.

## FLOPs and the R4.4 plan
Estimate at S (12L/384d, tied 16k vocab, seq 1,024), about 190 MFLOP per training token. The memory term adds:
- key projection: ~0.3 MFLOP/token;
- scores over 8,192 sampled keys, forward and backward into both sides: ~6.3 MFLOP/token;
- together ~+3.5% per step.

Per-run total: 9 epochs × 1.035 + the datastore forward (~⅓ of a training epoch) ≈ 9.65 epoch-equivalents, **≈ 0.97x the
10-epoch base**. The proposal said 1.12x. R4.4 is **not expected to trigger**. FLOPs are *measured* per run (key head,
memory softmax and datastore forward included); a FLOP-matched reference is needed only if the measured value exceeds
1.25x. BM25 runs on CPU.

Test-time exact search is inference. It is uncapped and reported: about 25-40 A100-minutes per model on full DEV at
128-d, and ~60 at 384-d.

## Null twins and comparators (all size S, seed 0, base WD, same batch, same GPU type)

| Row | What it is | What it removes or tests |
|---|---|---|
| **M** | the arm (above) | — |
| **N-shuf** | identical objective, reads and ledger, but memory values are randomly re-paired with keys (fixed-seed permutation) at training **and** test | the information in memory; runs at base WD, so it is not handicapped |
| **N-self** | training memory = same-document positions only; test memory = the current 512 window's own earlier positions only (never DEV text beyond the window) | if N-self ≈ M, the gain is copying |
| **N-noexcl** | TRIME_ext-faithful: same-window/segment memory dropped with p = 0.9, no document mask | measures the document-exclusion delta; decides the novelty label only, not a kill |
| **B9-bm25** | base, 9-epoch full schedule, *identical BM25 window order*, no memory | data-order confound (critic 18:43Z(c)); exposure match |
| **B10** | base re-run, 10 epochs, standard order | same-batch base |
| frozen base | round-1 winner base_S_wd0p4_lr0p5_r1_ve0_tied @ ref-0065 (harness full-DEV 5.041 nats/word raw) | frozen-base rule (A4 amendment 1) |
| **K-knn** | B9-bm25 + post-hoc kNN-LM on the same train_10M datastore (its own 384-d head-input keys, exact), T/λ/k' with the same grid size, 2-fold cross-fit | the binding known-idea floor |
| **K-knnT** | the same over a cross-fitted temperature-scaled LM (knn_diag knn_over_temp) | recalibration control |
| **K-count** | B9-bm25 + 10M count-model interpolation, cross-fitted weights | the second known-idea floor |

Every base row also gets a cross-fitted temperature (knn_diag lm_temps grid), so recalibration is never credited to
memory. Trial counts are logged per row, and the bases get at least the arm's knob trials.

M and every null are decomposed into **trunk-only** (memory off) and **trunk + memory**.

## Kill rule (registered before any run)
**Metric.** 2-fold document-cross-fitted **full-DEV nats/word** (dev_bpw protocol, ctx 512 / stride 256, collapse) at the
final checkpoint, with no checkpoint picking, domain-reweighted headline. The gain must also hold on **overlap-removed
DEV** (positions whose 13-gram context occurs in train_10M dropped). The gain on positions whose 8/13-gram context is in
train_10M is reported separately.

Let B* = the best of {frozen base, B9-bm25, B10}, each with its cross-fitted temperature. **Kill** if any of these holds:
- **K1.** M (with memory) beats B* by < 2x the registered seed floor.
- **K2.** M beats the better of K-knn and K-knnT by < 2x floor. *Beating the bare base and N-shuf is not enough: a pass
  without K2 would just be kNN-LM.*
- **K3.** M beats K-count by < 2x floor.
- **K4.** M beats N-shuf by < 2x floor.
- **K5.** M beats N-self by < 2x floor (the gain is copying, reported as the known kNN-LM/cache effect).
- **K6.** M **trunk-only** is worse than B* (trunk-only, temperature-matched) by > 2x floor (memory only compensates a
  damaged trunk; the GLUE leg loses).
- **K7.** A ≥ 2σ BLiMP-fast drop, memory on as declared.
- **K8.** Novelty tilt of (M vs B*) < 0.8, **or** the gain on words seen 64+ times on new text ≤ 0. This is binding: the
  pre-registered likely failure *is* the known copy effect.
- **K9.** Measured training FLOPs > 1.25x without a FLOP-matched reference.

The floor is not yet registered, so until it is, results are numbers only, with no verdict.

N-noexcl: if M − N-noexcl < 2x floor, the label stays "known idea (TRIME_ext)". Any gain concentrated on repeated
n-grams or high-frequency tokens is reported as the known kNN-LM/copy effect. It overlaps the copy-cache (+0.037) and
count-model (+0.051) headroom rather than adding to them. A per-domain table (wiki vs the rest) is required.

Promotion would be exactly one arm, picked on DEV before seeds 1-2. **Realistic best outcome is "partial":** GLUE7 runs
on the trunk, and kNN-type memory is about flat on BLiMP.

## Odds (guesses, stated plainly)
- Ceiling: ~0.25 nats/word (mechanism class).
- Median: **+0.02 nats/word over B*** (with memory), and **~0.00 beyond K-knn** (range −0.02 to +0.015).
- P(F0 says go) ≈ 0.25.
- P(passes K1-K9 at the S screen) ≈ **0.05**.
- P(≥ 0.1 over B*) ≈ 0.03.

Likeliest failure, step by step:
1. From ~epoch 4 the trunk has memorised its windows, the parametric logit wins on train positions, and the memory
   gradient fades.
2. Keys drift toward output-embedding directions.
3. At test, λ buys 0.03-0.05 over the base, of which 0.01-0.02 survives the temperature control, on repeats and wiki.
4. K-knn lands within ±0.01 of M.
5. M trunk-only is 0.02-0.04 worse than base.
6. The run dies on K2, K6 or K8.

A null is informative: it closes the trained-memory branch of the memory family on the real bench.

## Fastest falsification (in order; nothing trains until F0 passes)
**F0: zero training, ~60 A100-minutes.** This is the already-coded `babylm/analysis/knn_diag.py` (the experimenter's P7/V5 job) on
the frozen round-1 base, eval_split dev, train_10M datastore, run as "card F0, not V5". Additions (~50 LOC):
- the reverse cross-fit fold;
- recall@256 (share of DEV tokens whose target is among the neighbour values);
- a per-token oracle-λ bound U (a loose upper bound for any reader in this key space);
- overlap buckets by left n-gram overlap with train_10M (Drozdov/Norlund);
- frequency and novelty slices via p2_slices (Nishida).

Thresholds are **registered now in nats/word, before V5 lands** (V5 is in nats/token; 0.03 nats/word ≈ 0.019 nats/token).
Go only if all four hold:
1. gain_knn_beyond_temp ≥ 0.03 nats/word, cross-fitted and overlap-removed;
2. novelty tilt of that gain ≥ 0.6;
3. < 50% of it sits on positions with ≥ 4-token left overlap;
4. U ≥ 0.3.

Otherwise the card is withdrawn to a known-idea lever ≤ 0.1, with no training compute.

**F1: mini-ruler proxy, ~15 A100-minutes.** train_1M, S, base WD, seed 0, one batch: base@1M (10 epochs); M (9 epochs + a
1M datastore pass); N-shuf; N-self; base@1M + post-hoc kNN on a train_1M datastore; base@2M as the exchange-rate anchor.
Falsified if any of these holds:
- (i) M − (base@1M + kNN) < max(0.03, 2x floor);
- (ii) M − N-shuf < 0.03;
- (iii) M − N-self < 0.015;
- (iv) novelty tilt < 0.8;
- (v) trunk-only worse than base@1M by > 0.03.

It is a proxy label, never a verdict. A 1.6M-entry datastore understates any datastore-size effect.

## Screen F0 + F1 add ~75)
- Training: M, N-shuf, N-self, N-noexcl, B9-bm25, B10, about 6 × 9 min.
- Datastore passes: 4 × ~1.5 min.
- Exact full-DEV kNN scoring: M, N-shuf, N-noexcl at 128-d (~35 min each) and K-knn at 384-d (~60). N-self (in-window)
  and K-count are cheap or CPU.
- BLiMP-fast with memory: ~10-15 min.

Outputs go to shared/babylm/moonshot/mem/<run>/ with curves CSV + PNG. DEV-sub training curves are
trunk-only, labelled.

## Implementation plan (harness; ~1,000 LOC, 2-3 agent-days; F0 alone ~50 LOC)
- `babylm/moonshot/memory.py` (~200 LOC):
  - key head;
  - the in-batch memory softmax via logsumexp over [z_i, s_i] and the masked numerator, with cluster masks;
  - samplers for the 8,192 memory keys;
  - null modes `shuffle` (fixed-seed value permutation) and `selfdoc`;
  - the N-noexcl p = 0.9 segment drop.

  It hooks into `train.py` through the split API hidden()/logits(). Flags: `--mem_mode none|inbatch --mem_dim 128
  --mem_keys 8192 --mem_exclude cluster|segment_p0.9 --mem_null none|shuffle|selfdoc --mem_order bm25`. `--mem_order`
  also works with mem_mode none, for B9-bm25.
- `babylm/moonshot/doc_clusters.py` (~80 LOC): BOS-cumsum doc ids and 13-gram-Jaccard clusters (threshold registered),
  as an int32 array aligned to the stream.
- `babylm/moonshot/bm25_order.py` (~120 LOC):
  - scipy-sparse BM25 over the offset-0 window chunks, giving top neighbours;
  - a per-epoch greedy pairing that permutes the EpochSampler plan (window i ↔ chunk i under any offset);
  - an assert that every window appears once.
- `babylm/moonshot/build_datastore.py` (~80 LOC): reuses knn_diag's `encode_stream` to dump keys and values, and logs
  +10M exposure words.
- knn_diag extension, or `memory_score.py` (~150 LOC):
  - `--key_source head_input|mem_head`;
  - 2-fold cross-fit;
  - recall@k and the U bound;
  - overlap buckets;
  - shuffle and in-window null modes;
  - the trained-softmax base.
- Count-model scorer ported to harness full-DEV per-token dumps (~100 LOC, CPU; instruments' ngram_mix, PR #52) for
  K-count.
- HF export wrapper, `export_hf --memory` (~150 LOC):
  - trust_remote_code CausalLM returning log p_mix as logits (identity under the pipeline's log_softmax);
  - the 128-d fp16 index bundled as safetensors, with exact torch search on GPU;
  - memory on/off and λ declared **per family before scoring**: BLiMP/EWoK memory-on; GLUE, AoA and intermediate
    checkpoints trunk-only.
- Tests (~200 LOC):
  - flags off = bit-identical port loop;
  - exposure counter equals the 9-epoch base's (bm25 sampler included);
  - the cluster-exclusion mask is correct (no same-cluster read; a known toy case);
  - the BM25 order is a permutation per epoch;
  - no DEV or eval token in memory;
  - batch invariance (bs 1 = bs 16) of the wrapper;
  - pipeline_check-style sentence equivalence (wrapper vs memory_score);
  - N-shuf keeps the value marginal;
  - cross-fit knobs never see their scored documents.
- R5, before any suite number: a 13-gram audit of train_10M against every eval set (memory makes verbatim overlap
  retrievable; 0.57% of long CHILDES dev lines are verbatim in train_10M).

## Critic rulings requested before any GPU compute
1. Does memory-on BLiMP/EWoK through a trust_remote_code wrapper with a bundled ~4 GB index count toward the official NLP
   score (R2)? If not, the card can claim at most the loss leg ("partial").
2. Confirm the datastore pass is charged one epoch (registered here; the free branch is refused).
3. Confirm the BM25 permutation is uncharged ordering, and that the base's comparator set is B* as defined.

## Stacks with / overlaps
- Overlaps the copy cache (+0.037), the count model (+0.051) and the kNN-LM row. Those are its known-idea floor, not
  additive.
- Could stack with A1 or A3 (trunk-side regularisers), each needing its own combination card.
- A4: synthetic text never enters memory.

## Skeptic fold-in (what changed and where the skeptics disagreed)
Adopted from all three:
- the relabel;
- ceiling 0.9 → ~0.25;
- withdrawal of the Minnow/ICL claim;
- binding K-knn/K-knnT;
- the datastore charged and the free branch refused;
- nulls at base WD;
- the WD-up arm dropped from the screen;
- N-noexcl;
- B9 with an identical BM25 order;
- B* = best of three bases with temperature;
- the trunk-only gate;
- the 13-gram audit;
- doc ids from BOS, never src.bin;
- window-level BM25;
- the permutation-only sampler;
- an exact index for both DEV and the suite;
- F0 thresholds registered in nats/word before V5.

Resolved conflicts:
- **Index.** The mechanism skeptic asked to score DEV with the PCA-64/IVF index the wrapper uses; the rules skeptic asked
  for one exact index everywhere. Rules adopted: it satisfies "same index for both" without the R5 determinism and
  FAISS-certification risk.
- **Where knobs are fitted.** The mechanism skeptic offered "fit knobs on train_sub with the document excluded" as an
  alternative. Rejected: train keys are memorised states, so the fit would be biased. Document cross-fitting on DEV keeps
  the full-DEV metric free of in-sample knobs.
- **Novelty check.** Made a binding kill (K8), per the mechanism skeptic, rather than a reported label as in A1/A4.
  Here the pre-registered failure mode is exactly the known copy effect.
- **K-count.** Kept binding, per the rules skeptic. No verdict is reported until its scorer runs on the harness dumps.
