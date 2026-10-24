# Attack A3: stochastic segmentation for rare words (BPE-dropout) (card, 2026-10-10)

Code `babylm/harness/bpe_dropout.py` (`train.py --bpe_dropout 0.1x8,0x2 [--bpe_dropout_scope all|rare|frequent]`),
tests `babylm/tests/test_bpe_dropout.py`, jobs `babylm/configs/make_a3_jobs.py`.
Status: built and CPU-tested; **accepted by the critic (ref-0063)**; no GPU run yet, waits for the base round-1
winner. Conditions: the 1.25x rule is judged on training FLOPs (train_time reported too); the base arm runs through
the stream builder at p = 0 (`--bpe_dropout 0`) in the same batch on the same GPU type; a helpful frequent-only null
is reported but cannot be promoted under this card.

**Known, not new.** BPE-dropout (Provilkov, Emelianenko & Voita, ACL 2020) is the mechanism; subword regularisation
(Kudo 2018, unigram LM sampling) is its parent. Close follow-ups: StochasTok (arXiv 2506.01687, 2025: stochastic
re-splitting during LLM pretraining, gains on subword-level tasks) and a 2026 low-resource BERT pretraining study
(arXiv 2605.13436: stochastic tokenisation helps most when used in both pretraining and fine-tuning). Not in our
prior-art map and no BabyLM entry found in a quick search (prior-art thread to check). What we test is only whether
it helps a tuned 10M causal base under canonical-only scoring.

## Mechanism
Each training epoch e reads its own re-tokenisation of train_10M with dropout p_e; evaluation (DEV, TRAIN-sub,
official suite, export) stays on the canonical tokenizer. Segmentation = Provilkov Alg. 1: per merge step, candidates
are visited in (rank, position) order, each dropped with probability p; the first survivor is applied; if all are
dropped the pre-token stops. p = 0 reproduces the committed tokenizer on all 176,724 train_10M pre-token types, and
the p = 0 rebuild of the stream is bit-exact (tokens and word ids). Seeded: PCG64([seed, epoch, chunk]), so a stream
depends on (data, p, scope, seed, epoch) only; streams are cached with sha256s in `config.json`.
Note: HF tokenizers' dropout is weaker (stale queue entries consume draws): 1.09x tokens at p = 0.1 vs 1.18x here.
Cost: p = 0.1 makes 1.173x tokens per epoch (same words). A3a = 1.138x training FLOPs; under 1.25x (R4.4).

## Why (bottleneck 3: rare words)
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

## Null twin
`a3n`: dropout only on words with train_10M count > 63, p = 0.1139, annealed like A3a. It matches A3a's token count
(1.1732x vs 1.1731x per epoch), compute, exposure, code path and noise rate on frequent words, but never re-segments a
rare word, so the information the mechanism claims (alternative segmentations of rare words) is removed. If the null
does as well, the gain is generic input-noise regularisation, not rare-word learning.

## Kill rule (registered before any run)
Size S, train_10M × 10 epochs, the base winner's recipe and seed 0, base re-run in the same batch (no flag), same GPU
type. Metric: **full-DEV nats/word from `dev_bpw.py` on the final checkpoint** (collapse protocol; no checkpoint
picking; DEV-sub is diagnostics only). **Kill** if the best arm's gain over the base is below 2× the critic's seed
noise floor, **or** it does not beat the null by 2× the floor. No kill or promote until the floors are registered; until
then numbers only. Also kill on a ≥ 2σ drop on BLiMP-fast, or > 1.25x training FLOPs (tokens x flops/token; train_time reported
too) without a compute-matched reference. **Best arm** = lowest full-DEV nats/word among A3a-c; ties within 0.005 go to the lower token count.
Promote exactly one arm (picked on DEV before seeds 1-2) to 3 seeds, size M, official eval (R0-R6 bar).
Mechanism check (diagnostic, reported with the verdict): per-word gain by train-count band (0, 1-63, 64-8k, 8k+,
instruments' `prize.py band_shares` on per-token dumps); the rare-word claim needs ≥ 40% of the gain in bands 0 and
1-63 (their share of the gap is 32%). Provisional base (critic ref-0086): a kill on a provisional base reads
"killed (provisional base)" until a base change in WD/LR/EMA > 2x or in tokenisation forces one re-screen.

## Screen (one seed; ~8-9 A100-minutes per S run)
| arm | scope | p plan (E = 10) | tokens vs base |
|---|---|---|---|
| base | — | `--bpe_dropout 0` (p = 0 through the A3 sampler) | 1.000 |
| A3a | all | 0.1 × 8, 0 × 2 | 1.138 |
| A3b | all | 0.1 × 10 | 1.173 |
| A3c | rare (≤ 63) | 0.2 × 8, 0 × 2 | 1.036 |
| A3-null | frequent (> 63) | 0.1139 × 8, 0 × 2 | 1.138 |

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
