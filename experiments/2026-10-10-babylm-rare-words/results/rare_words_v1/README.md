# Rare-word (bottleneck 2) anatomy, 2026-10-10

Companion data for `docs/babylm/attacks/A5_rare_word_focus.md`. Produced by the rare-word attack thread on CPU; no
GPU, no training, no compute.

## What was measured

Two published models, `babylm-baseline-10m-gpt-bert-causal-focus` and `babylm-baseline-100m-gpt-bert-causal-focus`
(Hugging Face, the BabyLM 2025 GPT-BERT baselines, each with its own tokenizer), scored on a seeded subsample of the
official dev split: 287 pieces of at most 2,500 words from whole documents (at most 2 pieces per document), words per
source proportional to the dev split's word shares, 215,312 whitespace words, seed 20261010. Scoring reused the
bottleneck-instruments scorer (PR #52 branch, `babylm/instruments/score.py` and the vendored GPT-BERT code):
512-token windows, stride 256, BOS, collapsed whitespace, newline line join; per-token nats are assigned to
whitespace words. GPT-BERT 10M: 5.019 nats/word on the raw sample (the instruments' own sample: 5.020).

Word features come from `train_10M` and `train_100M` normal-form counts (lowercased, surrounding punctuation stripped).
Words whose line appears verbatim in `train_100M` (13.0% of the sample) are dropped in the `_filtered` tables, as in
the instruments' protocol; the unfiltered `decomp.md` is kept for comparison. Confidence intervals are piece bootstraps.

## Files

- `decomp_filtered.md`, `decomp_filtered.json`: the gap by train_10M frequency band, never-seen category, count
  sub-band, first use vs repeat, first piece vs continuation, rare words as context, case, source, top words.
  `decomp.md`: the same without the verbatim-line filter.
- `calib_summary_gptbert_10m.json`: realised vs predicted token-frequency-band mass, cross-fitted per-band logit
  offsets (headroom 0.0012 nats/word), word-boundary mass at rare-word continuation positions.
- `trie_oracle_gptbert_10m.json`: lexicon-trie continuation prior mixed into the 10M model (best +0.0095 nats/word).
- `context_novelty_filtered.json`: 1-63 first uses by whether their bigram/trigram occurs in train_10M / train_100M.
- `matched_count.txt`: loss minus own-unigram surprise at matched own counts; same-word comparison; 1-63 gap by
  piece count.
- `rare_words_fulldev_anatomy.json`: counts-only anatomy of the full dev split (no model), used for shares.
- `scoring_log.txt`: the scoring run log (throughput, nats/word per model).

The scripts are in `babylm/analysis/rare_words/`. They were run from a scratch directory laid out as
`osf/raw/{train_10M,train_100M,dev}`, `ckpt/<model>` (HF checkpoints) and `instr/` (the instruments' package); the
paths are arguments. Run order: `score_rare.py` (sample + scoring), `overlap_flag.py` (verbatim-line mask),
`analyze_rare.py` (tables), `calib_pass.py` then `analyze_calib.py` (calibration oracle), `trie_oracle.py`,
`context_novelty.py`, `matched_count.py`, `window_density.py` (rare exposure under window reweighting).

## Two findings that are not attacks

1. `train_10M` contains none of the CHILDES conversation-analysis markup (overlap brackets ⌊ ⌋ ⌈ ⌉, rising pitch ↗,
   ∆, ∙, ⁎; counts 0), while `train_100M` childes has about 7,000 of each bracket and the dev childes split about
   2,500 of each (0.94% of its lines). Words carrying these symbols are about 0.045 nats/word of the 10M-vs-100M dev
   gap (90% of it in childes), unreachable by any 10M-only method except in-context learning at test time. Rulers
   should report the gap with and without these words.
2. The Latin accented letters (à á â ã ä å æ ç è é ê í ñ ó ô ö ú ü ā č ł ō š ū) are already single tokens in the 10M
   tokenizer; accented words carry 0.0075 nats/word of the gap and 93% of them are spelled with single-token letters.
   Character coverage in the tokenizer has nothing to buy.

## a5_bands_fulldev/ (added 03:15Z, before the diagnostic runs)

`bands_base.md` / `bands_base.json`: the round-1 winner's (base_S_wd0p4_lr0p5_r1_ve0_tied) full-DEV per-word nats split by train_10M
frequency band, by the A5 focus set (rare / after-rare / focus / frequent), by source, and with the CHAT markup words
excluded (labelled second number). Produced by `babylm/analysis/rare_words/band_split.py` from
`shared/babylm/base_r1/base_S_wd0p4_lr0p5_r1_ve0_tied/pw/pw_devfull_final.npy` and the OSF raw text (dev words reconstructed in
the harness's stream order: 10,418,979 words, equal to the dump; per-source means match the harness's result.json to
four decimals). The same script, given the arm and the twin run dirs, writes the diagnostic's verdict table with paired
block-bootstrap CIs and the registered kill rule (`--floor` = pooled seed SD).
