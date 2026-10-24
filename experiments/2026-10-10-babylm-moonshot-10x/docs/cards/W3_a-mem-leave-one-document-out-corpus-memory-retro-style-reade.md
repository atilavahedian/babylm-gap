<!-- moonshot design round W3; status alive; ceiling 0.2 nats/word; expected 0.015; P(pass screen) 0.05; screen GPU 65 A100-min. Draft card: not yet submitted to the critic. -->

# Attack card (candidate): leave-one-document-out corpus memory (card, 2026-10-10)

Status: card only. No code, no runs. All three skeptics found it **wounded, not dead** (prior-art, rules, mechanism). No rule kills it outright, because each blocking rule has a workable fix. Every fix is folded in below, and the odds are cut to match. Before any code is written, three things must happen in this order: the critic rules on R0.2 (section 9), the R5 overlap check runs, and the zero-training probe F0 passes (section 8).

**Plain summary:** this is a known lever, RETRO/TRIME-style retrieval from the training set, being tested in a regime nobody has tried: strict-small with exposure counted. Its ceiling is about 0.2 nats/word, roughly 1.7x data. It is **not** a 20x mechanism. As designed, it can move only the loss leg: retrieval never fires on official-eval items, which are all shorter than 257 tokens. Its job is to be a cheap, decisive test of the "weights under-use exact corpus statistics" slice, and a possible stack component.

## 1. Known-ideas line and corrected novelty label
**Label: known idea (RETRO/REALM/TRIME_ext/kNN-LM family), untested regime (strict-small, 10M words x <= 10 epochs, neighbour reads charged as exposure).** The draft's "new here" items 1, 2 and 5 are deleted. Each one is published:
- RETRO (Borgeaud+ 2022, arXiv 2112.04426) already has all of these:
  - the training data as the retrieval database;
  - neighbours of 64-token chunk N plus 64-token continuation F;
  - previous-chunk causality;
  - "filter out neighbours originating from the same document";
  - k=2 in training with a larger k at eval.
  - Its App. C.2 is a negative result on our question: training from scratch on WikiText-103 (103M tokens, about 6x our stream) was *weaker than the baseline* because of over-fitting.
- Wang+ 2023 (arXiv 2304.06762) re-pretrained RETRO with the pretraining corpus as the only datastore (k=2, chunk 64). They also note that inputs shorter than one chunk never retrieve.
- REALM (Guu+ 2020, arXiv 2002.08909) excludes "trivial retrievals" from the query's own document and refreshes an index of the model's own embeddings asynchronously. A 30x staler index cut NQ from 38.2 to 28.7.
- TRIME (Zhong+ 2022, arXiv 2205.12674) uses the training set as external memory and drops local memory with p=0.9.
  - WikiText-103 test perplexity: vanilla 18.70, kNN-LM 16.12, TRIME_ext 15.37, i.e. about 0.05 nats/token over kNN-LM plus cache.
  - Without its memory, TRIME_ext is no better than vanilla (dev 17.99 vs 18.04).
- SPALM (Yogatama+ 2021, arXiv 2102.02557) trains a gate over a training-set memory and scores 18.8, *worse* than kNN-LM's 18.5.
- Memorizing Transformers (Wu+ 2022, arXiv 2203.08913) read cached, non-differentiable own states, and their large memories trained from scratch suffered from staleness.
- kNN-LM (Khandelwal+ 2020, arXiv 1911.00172) cuts 0.146 nats/token on WikiText-103. Its "retrieval beats training" result needs a datastore 30x the training set.
- **Primary anchor:** Liu & Kim 2026 (arXiv 2608.23851), a BabyLM-like setup (50M words, 10 epochs, post-hoc kNN only).
  - The gain is −0.040 nats/token (about 0.064 nats/word), from 1 seed.
  - Low-frequency minimal pairs gain +0.14; the high-frequency band moves −0.013, which is not significant.
  - They report that "semantic similarity alone provides little benefit".
- Studies that say what the gains are made of:
  - In-Context Pretraining (Shi+ ICLR 2024, arXiv 2310.10638): their retrieved-neighbour pretraining baseline did not beat standard perplexity, and near-duplicates caused copying and instability.
  - Norlund+ 2023 (arXiv 2302.12128): RETRO's gains are almost all verbatim copying.
  - Xu, Alon & Neubig 2023 (arXiv 2301.02828): 55% of the kNN-LM gain comes back without any retrieval, and exact search over the model's own training set overfits.
  - MassiveDS (arXiv 2407.12854): gains come mostly from lexical overlap.
- Internal (docs/sprint_1_prior_lessons.md):
  - neighbour cache +0.0012;
  - trunk-keyed post-hoc kNN/count memory 0.0057, 0.0036, 0.0033, shrinking as the trunk trains;
  - "post-hoc kNN or count memory" and "state-keyed memory tables" are on the do-not-repropose list.
  - Relative to sprint-1, the only twist is a *trained* cross-document reader, and RETRO and TRIME have already published that.
- BabyLM 2023-2025 proceedings (title scan): no paper does training-time corpus retrieval. The regime is untested, not shown to fail.
- What remains ours is small: the reader reads the trunk's own cached layer-6 states (Memorizing-Transformers style, at corpus scale with document exclusion) instead of re-encoding neighbour tokens, under an exposure-charged budget.

## 2. Mechanism
- **Base.** The round-1 winner, `base_S_wd0p4_lr0p5_r1_ve0_tied`:
  - 12L/384d, seq 1024, 65,536-token batches (64 windows, 1,024 stream chunks per step);
  - Muon WD 0.4, lr_scale 0.5, tied embeddings, no value embeddings;
  - causal only: no MNTP, dropout or EMA.
  - So the cached states come from plain causal bf16 training forwards.
  - Full-DEV dev_bpw is 5.037 (seed 0) and 5.026 (seed 1).
- **Memory (ChunkMemory, on the GPU).** The stream (15.96M tokens) is cut into 249,333 aligned 64-token stream chunks. The memory holds:
  - values: per-token 128-d states (a fixed random projection R of the RMS-normalised output of block index 5, the U-net encoder/decoder boundary) for all 15.96M positions, bf16, 4.09 GB;
  - keys: per stream chunk, R(RMSnorm(mean-pool of the chunk's block-5 states)), 64 MB;
  - per chunk: the parent-document set, the source id, and the epoch it was last written.
  - Parent document = the blank-line block in `babylm/data/corpus.py:doc_ranges` *before* the switchboard piece cut, so pieces of one conversation share an id.
- **Writes.**
  - Every training forward writes its window's values at exact stream positions; the window starts come from `EpochSampler.next_starts`.
  - Keys are written only for stream chunks fully inside the window. Each epoch has a random offset, so about 6% of chunks per epoch straddle a window edge and keep their previous key.
  - The memory is therefore a by-product of counted forwards: no extra forward, and staleness of at most about one epoch.
  - Writes happen after the optimizer step.
- **Default design M1c (exposure-charged; used unless the critic rules neighbour reads free).**
  - Each 1,024-token window splits into four 256-token query blocks. Blocks 1-3 retrieve; block 0 never does.
  - The query key for block b is computed from the *current* forward's block-5 states of the 64 window tokens just before the block (tokens 192-255, 448-511, 704-767).
  - A top-16 shortlist comes from one matmul over the key table. The first admissible candidate is used (k = 1). A candidate is admissible when all of these hold:
    - its key has been written at least once;
    - neither its parent documents nor those of chunk c+1 intersect any document the query window touches;
    - it is more than 2,048 stream positions from the window;
    - **near-duplicate filter:** no 13-gram of its value span appears anywhere in the 1,024-token query window (precomputed rolling 13-gram hashes per position, 128 MB; `torch.isin`). This is a training-time data filter. It cannot be applied at test, because it reads the future.
    - **hub cap:** the chunk has been retrieved at most 8 times this epoch. The retrieval-frequency Gini is logged.
  - The neighbour value span is the matched chunk's last 16 tokens plus its 64-token continuation: 80 cached states.
  - **Reader.** After block index 5, a MemoryCrossAttention module, added to the residual:
    - 6 heads; q = Linear(384, 384) on norm(x) for the block's 256 tokens; k, v = Linear(128, 384) on the *detached* neighbour states; no positions on the memory side;
    - output = g · W_o(attention), where g is a scalar initialised at 0, so step 0 equals the base exactly.
  - Blocks 6-11 and the U-net skips are unchanged.
  - The reader's parameters live outside `self.blocks`, so they train with **AdamW, not Muon**: matrices at the adam_head lr (0.0104 effective), g at the adam_scalar lr (0.026), WD 0. This is one registered value with no tuning. It avoids Muon giving full-size orthogonalised updates to a zero-signal reader.
  - The reader's parameters and R are initialised from a separate generator, so the trunk's initialisation is bit-identical to the base.
  - Retrieval runs on all 3 blocks of every window from step 0. During epoch 1, only filled chunks are candidates.
  - **8 own epochs** (section 4).
- **Conditional design M1f (only on an explicit "free" ruling).** RETRO layout:
  - every 64-token chunk 1-15 retrieves, keyed on chunk i-1;
  - k = 2; neighbour [N, F] = 128 states; 10 epochs.
  - The ruling picks M1c or M1f before code. They are never both run, and never chosen after a result.
- **Test (scored).**
  - Datastore = **the final training cache, checkpointed with the weights**. It is never rebuilt: a rebuild pass is a corpus statistic built for inference, charged one epoch under the count-model precedent, and would take the run to 108M.
  - At test there is no document exclusion and no 13-gram filter.
  - Registered k = 1 (M1c) or 2 (M1f). k = 4 or 8 is a label only.
  - The dev_bpw protocol (ctx 512 / stride 256) scores positions 256-511, which always fall in a retrieving block.
- **Key choice, fixed by F0 before training.** One of two keys, never both:
  - the state key above;
  - a lexical key: a fixed random projection (16,384 → 128) of the chunk's TF-IDF token vector. It is stale-free and answers Liu & Kim's "semantic alone gives little".
  - The registered rule picks the key with the higher F0 hit lift H/H0. The other is not run.

## 3. Why: which measured slice, and the honest ceiling
- **Target slice.** The "weights under-use exact corpus statistics" headroom, measured on tuned published checkpoints:
  - count model of the same 10M +0.051;
  - copy cache +0.037;
  - all add-ons together +0.125, which overlaps the gap rather than adding to it;
  - plus the bursty-word excess: 1.29x gap at equal frequency, about 0.02 nats/word.
- **Hypothesis.** A trained reader turns that inference headroom into a learned skill over the word bands 1-63 (0.16-0.18) and 64-8k (0.36).
- **What it cannot reach.** The memory holds nothing beyond train_10M.
  - 42-43% of the gap sits on word trigrams found in neither corpus, and 20-21% on trigrams found only in train_100M. At those positions no instance of the pattern exists to retrieve.
  - A perfect reader therefore caps near the seen-trigram slice, about 0.37 x 0.875 = 0.32 nats/word. The 100M trigram count model shows most of that slice is abstraction, not lookup.
- **Hit rate.** With 2 random neighbour spans out of 250k chunks, a word seen c times is present with probability about 1.6e-5 x c. So the key has to enrich hits 150-800x to matter for the 1-63 band.
- **Where the overlap payoff sits.** It is formulaic CHILDES speech (dev_sub 6-gram train matches: CHILDES 46%, Gutenberg 2%, wiki 6%; about 90% of 9-gram matches are CHILDES). That is already the base's lowest-loss source (4.01 vs 5.04 nats/word).
- **Ceiling 0.2 nats/word** (about 1.7x data by log-linear; hard cap about 0.32). The draft's 0.6 is withdrawn: the "retrieval beats training" evidence needs a datastore 30x the training set. Measured anchors with datastore = training set:
  - BabyLM-like −0.040 nats/token (about 0.064/word, 1 seed, 5x larger datastore);
  - WikiText-103: −0.146 (kNN-LM) and about −0.196 (TRIME_ext) nats/token, i.e. about 0.23-0.31/word, inflated by overlap on a non-decontaminated split.
- **Why a bigger regularised base would not already capture this.** It stores each word only in weights estimated from its n_w occurrences, and the +0.051 and +0.037 add-ons were measured on top of tuned models.
- **Counter-evidence (taken seriously, reflected in the odds):**
  - TRIME_ext's weights are no better than vanilla without the memory;
  - SPALM lost to plain kNN-LM;
  - from-scratch RETRO lost on WikiText-103;
  - In-Context Pretraining's retrieved-neighbour baseline did not help.
- **Multiplier caveat.** The 100M twin also gains from add-ons (copy cache +0.013 at 100M). The ruler is therefore reported both with and without post-hoc kNN over its own training set.

## 4. R0 exposure ledger (conservative reading = card default)
Neighbour token states read by the trained reader count as exposure. The rules skeptic's reductio is accepted: cache layer-0 states and you get unlimited free context. Words are counted exactly from the words memmap (word-initial tokens among the states read), logged per step. The run aborts if the projected total exceeds 100M.

| Item | M1c | M1f (only if ruled free) |
|---|---|---|
| Own forwards | 8 x 9.954M = 79.63M words | 10 x 9.954M = 99.54M |
| Neighbour reads | 3 blocks x 80 states per 1,024 trained tokens = 23.4% → ≈18.66M words | 15 x 2 x 128 = 3.75x (≈373M words) logged, ruled free |
| Test datastore | final training cache = activations of counted forwards: 0 | 0 |
| Cache rebuild | none (would be +10M under the count-model precedent) | none |
| **Total** | **≈98.3M ≤ 100M** | 99.5M (+ ruled-free reads) |
| Null twins N1, N2 | identical ledger | identical |
| Comparator base + kNN-LM | base 99.5M + datastore pass 10M (charged): out of track, a bar only, never claimed | same |
| F0 probe | inference on the frozen base; no training | — |

Resume restores `memory.pt` (values, keys, counters, hub counts), saved at every 10M milestone. The memory is never recomputed, so spot preemption costs no extra forwards. R5: the datastore is train_10M only.

## 5. FLOPs and the R4.4 plan
- **Base cost.** About 193 MFLOP/token for training (`flops_per_token`).
- **M1c additions per token.** About +1.7 MFLOP (reader, fwd+bwd, 3 of 4 blocks), +0.19 MFLOP (retrieval matmul: 3 queries x 249k x 128 per window), +0.1 MFLOP (write projection). That is about 1.01-1.02x per token.
- **M1c total.** Over 8 epochs, about **0.82x** the reference's training FLOPs. M1f is about 1.03x.
- **R4.4.** Not triggered; it is judged on FLOPs.
- **Wall-clock.** Reported separately, expected 1.2-1.4x per token: gathers, top-k and isin run eager, and only the trunk blocks are compiled.
- **GPU memory.** About 4.3 GB of memory buffers per run. Three S runs per A100 under MPS fit on 80 GB; on 40 GB, run two.

## 6. Null twins and binding comparators
Every null has the same module, spans, k, cache writes, gate, optimizer groups, epochs, exposure and FLOPs, with the retrieval matmul still run and its result discarded.
- **N1 (binding): same-source random neighbours.** The neighbour is a uniformly random admissible filled chunk from the same corpus source as the query's key chunk, under the same exclusions and hub cap. It removes query-specific retrieval and keeps domain priming. With an ignorable memory, N1 is about the base at 8 epochs, so it also prices the epoch cut.
- **N2 (binding): shuffled query.** Each block receives the arm's real admissible top-1 for another block in the same step (a fixed derangement; same source where possible), with exclusions re-checked against the receiving window. It keeps the neighbour-set statistics (hubs, coherent near-duplicate-like sets) and removes relevance.
- **Comparator C1 (binding): base + post-hoc kNN-LM** over the same train_10M datastore. It uses `babylm/analysis/knn_diag.py` on the base re-run with `--eval_split dev`, beyond-temperature (kNN over the τ-scaled LM), with λ, T and k' picked on even DEV documents. The arm and C1 are compared on odd DEV documents, on the same tokens. TRIME's whole claim is beating kNN-LM, and SPALM failed to.
- **Labels, not kill conditions:**
  - an in-document online cache: a different protocol, since dev_bpw's window batching has no online cache;
  - a uniform random null: dropped as redundant with N1;
  - base + copy cache, if the instruments' cache tool runs on the same protocol.

## 7. Kill rule (registered before any run)
**Setup.** Size S, the round-1 winner's recipe and seed 0, with the base re-run in the same batch (module absent, the bit-identical path, same GPU type).

**Metric.** Full-DEV nats/word from `dev_bpw.py` on the **final checkpoint**, with the final training cache and the registered k. No checkpoint picking.

**Kill if any of these holds:**
- **(a)** The arm is not ahead of the base re-run (10 epochs) by 2σ, where σ is the registered seed floor.
- **(b)** The arm is not ahead of the better of N1 and N2 by 2σ.
- **(c)** The arm is not ahead of C1 (base + kNN-LM, odd documents) by 2σ.
- **(d)** (a) fails on overlap-removed DEV, i.e. DEV lines with any train_10M 13-gram match removed.
- **(e) Mechanism** (vs the base re-run, on per-token dumps):
  - novelty tilt < 0.8; **or**
  - less than 50% of the gain lands on tokens whose 8-gram (7 context tokens + target) is absent from train_10M.
  - Reported alongside, not a kill: rare tilt (expected ≥ 1.5) and Norlund overlap buckets. If 70% or more of the gain falls on tokens with ≥ 3 preceding tokens overlapping the neighbour span, the result is labelled "copy mode / cache lever".
- **(f) Fade.** On DEV-sub, the arm's gain over N1 at the final milestone sits more than 0.5σ below its value at 70% of training.
- **(g) Early kill (stop the run).** Retrieval-information gauge = DEV-sub with N1-style random neighbours minus DEV-sub with retrieved neighbours, on the same checkpoint. Kill if it is below 1σ at the epoch-4 milestone. The gauge is logged every epoch on DEV-sub and on TRAIN-sub, so the multi-epoch double bind is visible.
- **(h) Other kills:**
  - a BLiMP-fast drop of 2σ or more, since the trunk may offload to the memory;
  - more than 1.25x training FLOPs without a FLOP-matched reference.

**Before any number is reported (R5):**
- the 13-gram check of train_10M against every official eval set (none exists in the repo yet);
- the planted-duplicate leak test passes, including its positive control (section 10);
- the TEST overlap count is run by the critic only; attack threads never read TEST.

**Floor.** No verdict before σ is registered. The indicative value is 0.0079 nats/word (winner seeds 0 and 1: 5.0367 vs 5.0255), so 2σ ≈ 0.016. The registered σ governs.

**Official eval.** M1c and M1f are **registered loss-leg-only**: inputs under 257 tokens (M1c) or 65 tokens (M1f), which is every BLiMP, Supplement, EWoK and COMPS item and most GLUE items, never retrieve. The predicted NLP-score delta is 0 ± σ (trunk effects only). The best possible label is "partial" (ref-0088). The only route to the R4.2 leg is a prefix-keyed first-block path (M1p: re-query every 16 tokens from the running mean-pooled prefix, causal), which needs its own card and is built only if M1 passes.

**Export.** HF wrapper: memory as bf16 buffers exempt from the eval fp32 cast, retrieval code under trust_remote_code, right padding asserted.

**Promotion.** If the arm passes, exactly one arm and key go to 3 seeds and size M under R4. Screens count in the trial log.

## 8. Fastest falsification: F0 thresholds registered before running)
- **(a) kNN-LM cell.** `python -m babylm.analysis.knn_diag --ckpt <winner chck_100M.pt> --train_split train_10M --eval_split dev_sub`.
  - This is the registered V5 cell (v2 was re-timed to this winner). If the experimenter's V5 job has already run, reuse it; otherwise this run *is* V5 and is reported as such.
  - Slice the lmtemp vs knntemp dumps: novelty tilt, rare tilt, per-source split, and the gain on tokens with vs without a train_10M 8-gram match.
- **(b) Exact-key chunk probe** (`babylm/analysis/chunk_probe.py`, about 100 lines).
  - It uses the arm's retriever exactly: block-5 mean-pooled, RMS-normed, projected by R, keyed on the previous chunk; same exclusions, 13-gram filter and spans; even/odd documents.
  - It runs once with the state key and once with the lexical key.
  - H = P(the target word type appears in the neighbour span), for dev_sub content words in the 1-63 and 64-8k bands. H0 = the same rate for random same-source neighbours.

**Build the reader only if ALL of these hold:**
1. gain_knn_beyond_temp on the held-out odd documents ≥ 0.04 nats/word (≈ 0.025 nats/token).
2. ≥ 50% of that gain lands on tokens without a train_10M 8-gram match.
3. Its novelty tilt is ≥ 0.6. This is looser than 0.8 because post-hoc interpolation pays log(1−λ) on misses.
4. For the better key, in the 64-8k content-word band, H/H0 ≥ 3 and H ≥ 0.06 for the M1c span (k=1, 80 tokens), or H ≥ 0.10 for the M1f span (k=2, 128 tokens).

**Predictions:**
- (1) 0.02-0.06
- (2) 30-45%
- (3) 0.2-0.5
- (4) H about 3-8%
- **P(F0 kills) ≈ 0.8**

**CPU fallback (proxy label, criteria 2-4 only).**
- Datastore: train_1M, about 1-2 CPU-hours on 4 CPUs.
- Model: the winner's local weights `shared/babylm/base_r1/base_S_wd0p4_lr0p5_r1_ve0_tied/hf/model.safetensors`.
- Queries: about 100k dev_sub tokens from odd documents.

**Do not use the 1M mini-ruler as the first falsifier.** The datastore shrinks with the data, and BOS-keyed leave-one-document-out leaks same-book pieces there. If it is ever used, carry the train_10M parent document ids.

## 9. Gates and rules questions (in order)
- **G0, the critic rules on:**
  - (Q1) R0.2: do cached states of other windows, read by a trained reader, count as exposure? The card's reading is yes, which gives M1c.
  - (Q2) The final training cache as the test datastore costs 0, being activations of counted forwards (A1 precedent).
  - (Q3) The loss-leg-only registration, with "partial" as the best label.
  - (Q4) The C1 comparator is out of track by 10M; it is accepted as a bar only.
- **G1:** R5 13-gram check of train_10M against all official eval sets.
- **G2:** F0.
- **G3:** CPU tests.
- **G4:** GPU screen.
- **G5:** promotion only via R4.

## 10. Implementation plan (harness; about 2 days, eager first, compile the trunk only)
**New code:**
- `babylm/moonshot/chunk_index.py`: offline, CPU, no model, no exposure. Builds per-chunk parent-doc sets (switchboard parent blocks), source ids, 64-bit rolling 13-gram hashes per position, TF-IDF lexical keys, and the overlap-removed DEV mask.
- `babylm/moonshot/memory.py`: ChunkMemory with:
  - `write(starts, h5)`;
  - `shortlist(qkeys)` via one matmul and top-16;
  - `admissible(...)`: doc, range, filled, hub-cap and 13-gram masks;
  - `pick(mode = retrieved | samesource | shufquery)`;
  - `gather` → [B, 3, k·80, 128];
  - `state_dict` / `load_state_dict`;
  - exact neighbour-word counting.
- `babylm/moonshot/memory_reader.py`: MemoryGPT(GPT). It overrides `hidden()` to run blocks 0-5 with the identical skip, x0 and value-embedding bookkeeping, then compute the block query keys, call MemoryCrossAttention, and continue with blocks 6-11. It is registered for `train.load_model_module` so that knn_diag, dev_bpw and export can load it.
- `babylm/analysis/chunk_probe.py` (F0b).
- `babylm/analysis/overlap13.py` (R5), if no existing tool is found.

**Harness changes:**
- `train.py` flags: `--memory {off,M1c,M1f} --memory_null {none,samesource,shufquery} --memory_key {state,lex}`.
  - Write positions come from `EpochSampler.next_starts` (not PlanSampler).
  - `memory.pt` is saved at the 10M milestones and restored on resume.
  - The exposure counter includes neighbour words.
  - A dedicated AdamW "memory" group.
- `dev_bpw.py`: loads `memory.pt` with the checkpoint; `--memory_k` (label) and `--memory_mode` (for the gauge).
- `export_hf.py`: the wrapper described in section 7.
- Jobs: `babylm/configs/jobs_moonshot_mem_S.json`.

**Tests** (`babylm/tests/test_memory.py`):
- `--memory off` is bit-identical to the port path; with the module on and g = 0, step-0 logits equal the base's and the trunk initialisation is bit-identical.
- Admissible picks never share a parent document, the ±2,048 range, or a 13-gram with the window.
- **Planted-duplicate leak test** (lesson of ref-0422). Document D is copied in as a different document D'.
  - Positive control: with only the same-document exclusion, D' is retrieved for D and the loss on D drops (the leak is visible).
  - With the 13-gram filter on, D' is never retrieved.
- Distribution checks: N1 is uniform over admissible same-source chunks (chi-square test); N2 is a derangement.
- Keys are written only for fully contained chunks, and values land at exact positions under the random epoch offsets.
- `memory.pt` round-trip on resume is bit-exact.
- The exposure counter is exact, and the run aborts above 100M.
- dev_bpw with memory equals a slow reference implementation.
- HF export logits equal the harness logits.

**Screen**:

| run | what |
|---|---|
| base | module absent, 10 epochs |
| M1c | (or M1f, per ruling), key per F0 |
| N1 | same-source random |
| N2 | shuffled query |
| C1 | knn_diag on the base re-run, full DEV |

Diagnostics: per-epoch gauge, reader attention mass on memory, g, hub Gini, the training-time hit rate H, the rate at which a neighbour continuation contains the target 8-gram, per-source gains, and the gain split by 8-gram match.

## 11. Odds (stated plainly)
- **Ceiling:** 0.2 nats/word (about 1.7x data), hard cap about 0.32.
- **Gain distribution for M1c vs the base re-run on full DEV:**
  - 40% at ≤ +0.01: the reader is ignored, or the 8-epoch cut costs more than the memory buys;
  - 35% at +0.01 to +0.03;
  - 20% at +0.03 to +0.06;
  - 5% above +0.06.
- **Median and mean:** median about +0.01, mean about +0.015.
- **Where any gain should land:** 50-70% on CHILDES and subtitle tokens with a train 8-gram match. Novelty tilt 0.3-0.7.
- **P(pass):** P(F0 pass) ≈ 0.2 x P(screen pass | F0) ≈ 0.25 → **P(pass) ≈ 0.05**. M1f, if ruled free: mean about +0.02, P(pass) about 0.07.
- **P(official NLP leg moves):** about 0 by construction.
- **What a null result means:** in this regime, retrieval over one's own 10M words adds nothing that kNN-LM at inference does not already give. Inference headroom then stays a test-time add-on (P7/V5), not a training lever.

## 12. Skeptic fixes: adopted, modified, declined
- **Adopted:**
  - relabel and cut the "new here" list;
  - C1 kNN-LM as a binding comparator;
  - F0 first, with registered thresholds and the exact-key probe;
  - Norlund overlap buckets;
  - 13-gram near-duplicate filter, hub cap and Gini;
  - R0.2 ruling first, conservative charge by default (M1c);
  - test datastore = the checkpointed final training cache, no rebuild;
  - loss-leg-only registration and the M1p route;
  - R5 checks, the overlap-removed DEV kill, and the planted-duplicate test with a positive control;
  - N1 same-source and N2 shuffled-query as binding nulls;
  - mechanism kill on novelty tilt plus the no-8-gram-match share, replacing the band-share check;
  - reader on AdamW with a zero-initialised scalar gate;
  - fact fixes: EpochSampler with its random offset, 1,024 chunks per step, block indexing, causal-only states;
  - ceiling cut to 0.2; Minnow and WikiText-103 dropped as primary anchors.
- **Modified:**
  - The cross-fitted memory is replaced by the per-epoch retrieval-information gauge with an early kill. The weights memorise both halves under any split, so cross-fitting removes no memorisation; the gauge measures the actual risk.
  - The lexical retriever is a pre-training key choice decided by F0, not an extra screen arm.
  - TEST overlap is counted by the critic, not by the attack thread.
- **Declined:** in-batch TRIME as the R0 fallback. A 65k-token batch holds about 1,000 random chunks, so it is a near-null. The charged M1c design replaces it.
