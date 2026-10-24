# Attack A3: stochastic segmentation for rare words (BPE-dropout) (card, 2026-10-10)

Code `babylm/harness/bpe_dropout.py` (`train.py --bpe_dropout 0.1x8,0x2 [--bpe_dropout_scope all|rare|frequent]`),
tests `babylm/tests/test_bpe_dropout.py`, jobs `babylm/configs/make_a3_jobs.py`.
Status: **KILLED (base E10), critic ref-0002: rare-word claim fails the registered null** (see Result); accepted by the critic
(ref-0063). Conditions: the 1.25x rule is judged on training FLOPs (train_time reported too); the base arm runs through
the stream builder at p = 0 (`--bpe_dropout 0`) in the same batch on the same GPU type; a helpful frequent-only null
is reported but cannot be promoted under this card.

**Known, not new.** BPE-dropout (Provilkov, Emelianenko & Voita, ACL 2020) is the mechanism; subword regularisation
(Kudo 2018, unigram LM sampling) is its parent. Close follow-ups: StochasTok (arXiv 2506.01687, 2025: stochastic
re-splitting during LLM pretraining, gains on subword-level tasks) and a 2026 low-resource BERT pretraining study
(arXiv 2605.13436: stochastic tokenisation helps most when used in both pretraining and fine-tuning). Not in our
prior-art map and no BabyLM entry found in a quick search (prior-art thread to check). What we test is only whether
it helps a tuned 10M causal base under canonical-only scoring.

## Result (10-10, seed 0, E = 10, solo; numbers before calibration; kill confirmed by the critic, ref-0002)
**Rare-word claim: KILLED on the registered null leg.** a3a − a3n(p 0.1185, ratio 1.1503 vs 1.1504) = +0.0039 v2
twin (+0.0038 harness), below the 0.0074 kill line. Twin gains vs the same-batch base (5.0077): a3a +0.0632, a3n(0.1185)
+0.0593, a3c +0.0494, a3n(0.1139, diagnostic) +0.0356, a3b −0.1575. Guard: unseen-character words get slightly worse in
every arm (−0.002 to −0.004), so the gain is not byte-row artifact. The two nulls differ by 0.024 for a 4% change in p,
so run-to-run noise in this family is ~3x the base seed SD. What survives is generic: annealed segmentation noise of any
scope helps the overtrained E10 base by ~0.06, and is ~0.02 raw vs the E* candidate. That claim was carded separately as
A3g (`A3g_segmentation_noise.md`), CLOSED 11-10 with no claim: +0.043 single at ~1.9x FLOPs loses to depth (R4.4);
~+0.013 inside the legal split, below its bar. Band and run files: shared/babylm/attacks/A3/.

## Mechanism
Each training epoch e reads its own re-tokenisation of train_10M with dropout p_e; evaluation (DEV, TRAIN-sub,
official suite, export) stays on the canonical tokenizer. Segmentation = Provilkov Alg. 1: per merge step, candidates
are visited in (rank, position) order, each dropped with probability p; the first survivor is applied; if all are
dropped the pre-token stops. p = 0 reproduces the committed tokenizer on all 176,724 train_10M pre-token types, and
the p = 0 rebuild of the stream is bit-exact (tokens and word ids). Seeded: PCG64([seed, epoch, chunk]), so a stream
depends on (data, p, scope, seed, epoch) only; streams are cached with sha256s in `config.json`.
Note: HF tokenizers' dropout is weaker (stale queue entries consume draws): 1.09x tokens at p = 0.1 vs 1.18x here.
Cost: p = 0.1 makes 1.188x tokens per epoch (same words; exact count, an earlier 3-chunk estimate said 1.173). A3a =
1.150x training tokens and steps (measured on the GPU run, 2802 vs 2435 steps); under 1.25x (R4.4).

## Why (bottleneck 2 in the experimenter's ranking: rare words)
Words seen < 64 times in train_10M are 8.3% of DEV words and 32% of the GPT-BERT 10M→100M gap (never-seen 6.12,
1-63 band 2.69 nats/word; instruments PR #52, published checkpoints). A rare word is mostly a rare or unique token
sequence, so its handful of uses train a few rare embeddings and memorised continuations. Under dropout the same word
is also shown as smaller, frequent pieces, whose embeddings are trained on many words; the model must learn to
compose pieces, which is what an unseen or rare DEV word needs. Second effect: the input sequence of every repeated
sentence changes each epoch, which may slow memorisation of frequent contexts (the base peaks at epoch 2-3).
Risk: canonical-only scoring charges the model for any mass on other segmentations; anneal (last 20% of epochs
canonical) is the fix under test, constant p the control for it. Honest prior: BPE-dropout's gains are in MT and
subword tasks, and LM perplexity under canonical scoring usually gets slightly worse. Guess: −0.03 to +0.05
nats/word over the base, P(passes kill rule) ≈ 0.15. If it passes, the gain should sit in the rare bands.
**Prior on the ve0 + tied base (10-10 01:45Z, corrected 01:55Z, before any run; the registered prediction above
stands):** dropping the value-embedding tables, plus tying once they are gone, already took part of the rare-word
headroom (ve0 + tied vs the port's best: +0.87 nats/word on never-seen words, +0.51 on words seen once; instruments
ref-0091; tying alone fixes nothing). Fewer token-keyed lookup tables overlaps A3's mechanism, so I now expect −0.03
to +0.03 and P(pass) ≈ 0.10.

## Null twin
`a3n`: dropout only on words with train_10M count > 63, p = 0.1185, annealed like A3a. It matches A3a's token count
(epoch-0 streams 18,957,525 vs 18,957,645 tokens; corrected 10-10 08:55Z before the null ran: the registered p = 0.1139
came from a biased 3-chunk estimate and would have given 0.64% fewer tokens than A3a), compute, exposure, code path and noise rate on frequent words, but never re-segments a
rare word, so the information the mechanism claims (alternative segmentations of rare words) is removed. If the null
does as well, the gain is generic input-noise regularisation, not rare-word learning.

## Kill rule (registered before any run)
Size S, train_10M × 10 epochs, the base winner's recipe and seed 0, base re-run in the same batch (no flag), same GPU
type. Metric: **full-DEV nats/word from `dev_bpw.py` on the final checkpoint** (= `score.json` dev_nats_per_word on the fp32 HF
export, critic ref-0053; never result.json full_eval, which differs per model by −0.020 to +0.034) (collapse protocol; no checkpoint
picking; DEV-sub is diagnostics only). **Kill** if the best arm's gain over the base is below 2× the critic's seed
noise floor, **or** it does not beat the null by 2× the floor. **Floors registered (critic ref-0019, superseding ref-0021 and ref-0022; before any A3 run):** primary metric is the
blank-line twin, `score.json` dev_nats_per_word_twin (v1 dev_nats_per_word reported for information); interim screen
floor 0.0036. Screen (1 arm seed vs the same-seed base; same for the null): pass at a gain of 0.010 nats/word or more;
0.0074-0.010 is inconclusive and may buy 2 more seeds of arm and base once; kill below 0.0074. Claim floor: pooled SD of
plain base seeds (n >= 5), registered before any claim is read. A BLiMP-fast drop above 2.0 points kills a screen
(above 1.4 a claim).
**Added by critic ref-0001 (before any A3 run):** a pass must clear the bar on v2 full DEV **and** on v2 DEV minus
the 10,275 words containing characters absent from train_10M (frozen list, unseen_chars.py); this guard binds for A3,
since dropout exposes rare byte tokens. Screens must be same-condition: solo screens pass at 0.010 or more; packed
screens pass at 0.0144 or more (inconclusive 0.0102-0.0144, kill below 0.0102). Requested: all five arms solo. Also kill on a ≥ 2σ drop on BLiMP-fast, or > 1.25x training FLOPs (tokens x flops/token; train_time reported
too) without a compute-matched reference. **Best arm** = lowest full-DEV nats/word among A3a-c; ties within 0.005 go to the lower token count.
Promote exactly one arm (picked on DEV before seeds 1-2) to 3 seeds, size M, official eval (R0-R6 bar).
Mechanism check (diagnostic, reported with the verdict): per-word gain by train-count band (0, 1-63, 64-8k, 8k+,
instruments' `prize.py band_shares` on per-token dumps); the rare-word claim needs ≥ 40% of the gain in bands 0 and
1-63 (their share of the gap is 32%). Also reported, diagnostic only (added 10-10 at the experimenter's request, after
acceptance, before any run): the gain on the first DEV occurrence of words seen 1-63 times in train_10M, the core of
bottleneck 2 in the instruments' split (0.125 of 0.284 nats/word). Provisional base (critic ref-0086): a kill on a provisional base reads
"killed (provisional base)" until a base change in WD/LR/EMA > 2x or in tokenisation forces one re-screen.

## Screen (one seed; ~8-9 A100-minutes per S run)
| arm | scope | p plan (E = 10) | tokens vs base |
|---|---|---|---|
| base | — | `--bpe_dropout 0` (p = 0 through the A3 sampler) | 1.000 |
| A3a | all | 0.1 × 8, 0 × 2 | 1.150 (measured) |
| A3b | all | 0.1 × 10 | 1.188 |
| A3c | rare (≤ 63) | 0.2 × 8, 0 × 2 | ~1.04 (estimate) |
| A3-null | frequent (> 63) | 0.1185 × 8, 0 × 2 | 1.150 |

5 runs ≈ 45 A100-minutes. Stream building runs on the VM's CPUs before training (~2 min table + ~10-20 s per
epoch), reported separately from train_time. FINAL untouched. Each run counts in the A3 trial log.

## Exposure and legality (R0)
Every epoch stream has exactly the canonical stream's whitespace words in order (asserted at build: same
word-initial count and last word id), so words seen per epoch, the milestone counter and checkpoints are unchanged;
only tokens per word grow. Same tokenizer (trained on train_10M only), no extra data, no outside model.

## Verified on CPU (tests)
p = 0 equals the HF tokenizer on hand-picked hard strings and on every train_10M type; the p = 0 stream rebuild is
bit-exact; dropout streams decode to the same text per document and their word ids equal prepare.py's rule applied
to the new tokens; seeded, chunk-local and worker-count independent; scopes re-segment only their words;
`--bpe_dropout 0` reproduces the port's losses bit for bit; flag off is bit-identical to base commit ref-0092 (tiny
run: 60 losses, curves, full DEV); dropout run sees the same words with more steps, and pause + resume is bit-exact.
