# GPT-BERT (BabyLM 2025 strict-small baseline), single-GPU port

Target: `BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus` (published checkpoint in
`~/babylm/models/babylm-baseline-10m-gpt-bert-causal-focus`). Code: `github.com/ltgoslo/gpt-bert @ ref-0493`.
`official/` holds `model_extra.py`, `dataset.py`, `lamb.py`, `utils.py` copied **verbatim** (attribution and upstream
sha256 in each header, MIT licence in `official/LICENSE_gpt-bert`). `hf_files/` holds the published HF files.

| file | role |
| --- | --- |
| `prepare_data.py` | tokenise train_10M / dev (/ train_100M) with the published tokenizer; doc tables; fixed 1M-word train sample |
| `train.py` | trainer (one GPU, or CPU for `--smoke`), checkpoints, resumable states, dumps, CSV/PNG |
| `export_hf.py` | train-side weights -> HF dir loadable by the published `modeling_gpt_bert.py`; `--verify` |
| `scoring.py` | per-token causal scoring used for the dumps |
| `../../eval/dev_bpw.py` | the critic's held-out metric; imported for the in-training `dev_bpw_curve` column |

## Commands (cloud GPU)

```bash
cd babylm/recipes/gptbert                       # needs torch (CUDA), transformers 4.51.3, tokenizers, numpy, matplotlib
python prepare_data.py --data_root /path/to/text_data --splits train_10M,dev           # ~3 min, 1 CPU
# (optional) prove the train-side model == published model:  python export_hf.py --verify --published /path/to/published_dir
python train.py --seed 42 --out_dir runs/gptbert_cf_s42 --dev_text_dir /path/to/text_data/dev/dev
```
Siblings: mixed `--hybrid_numerator 8`, masked-focus `--hybrid_numerator 1` (denominator 16).
Published-model reference on the same dev subset / dumps / curve: `python train.py --eval_only --init_from_hf /path/to/published_dir --out_dir runs/published_ref`.
100M track: `python prepare_data.py --splits train_100M,dev,train_10M` then `--train_dir train_100M --epochs 10` (1B words, ~99k steps) or `--epochs 1`.
Resume: `python train.py <same flags> --resume runs/X/states/chck_40M.pt` (bit-exact on CPU; tested). If memory is short at
seq 512 (e.g. L4 24 GB) add `--grad_accum 2` (same batch, two micro-batches). Disk: one resumable state is 555 MB
(~19 per run = ~10.5 GB); `--resume_states last` keeps only the newest.

## What the default run does (train_10M, seed 42)

- corpus 16,242,414 tokens / 9,953,773 whitespace words -> `max_steps = round(10 x 16,242,414 / 16,384) = 9,914` (the card's number exactly).
- 16,384 token positions per step (128 x 128, then 64 x 256 from step 5,948, then 32 x 512 from step 7,931); 162.4M positions in total.
  Of these ~1.1% are `<s>`/padding, so ~160.6M data tokens = ~9.9 epochs = **~98.5M words consumed** (measured by the run,
  not estimated): BabyLM checkpoints chck_1M..chck_9M, chck_10M..chck_90M fire; chck_100M is not reached; `hf/final` is the end.
- per step: 120 causal + 8 masked sequences at seq 128 (15/16), one forward through the official hybrid path.
- LAMB lr 0.007, betas (0.9, 0.98), eps 1e-8, wd 0.1 (not on biases/LayerNorm, as official), clip 2.0, z-loss 1e-4,
  warmup 158 / cooldown 158 steps, cosine to 0.1 then linear to 0 (official schedule fn), mask p 0.3 -> 0.15 linear,
  random 0.1 / keep 0.1, dropout 0.1, bf16 autocast, EMA 0.999 kept as in train_10m.py.
- Outputs in `--out_dir`: `loss_curve.csv` + `.png` (shared plotter), `run_meta.json` (all explicit values, the 2024 repo
  defaults NOT used, git SHA, versions, file hashes), `summary.json`, `hf/chck_*M/` (raw weights, eval-ready), `hf/final`,
  `hf/final_ema`, `states/*.pt` (weights, EMA, optimizer, scheduler, data-sampler state incl. the official datasets'
  visit counters, data/CPU/CUDA/python/numpy RNG), `dumps/`.
- CSV columns: `step, tokens_seen, words_seen` (exact word-exposure counter), `words_seen_est, epoch, lr, seq_len,
  train_loss_causal, train_loss_masked, dev_loss_causal, dev_loss_masked` (fixed dev subset, nats/token, every 200 steps
  and at checkpoints), `dev_bpw_curve` (dev_bpw.py `--slice curve` at ctx 128 / stride 64, every 1000 steps + checkpoints),
  `dev_dump_loss_causal, dev_dump_bpw, train_sample_loss_causal, train_sample_bpw` (epoch checkpoints + end),
  `elapsed_s, data_tokens_seen, grad_norm, z_term, mask_p, checkpoint`.
- Dumps (every 10M words and at the end): `dumps/<chck>/dev_nll.npy` and `train_sample_nll.npy` (float16 per-token causal
  NLL); static arrays once in `dumps/`: `*_token_ids.npy` (int32), `*_doc_ids.npy` (int32, row of `data/<split>_docs.npy`
  = [file index, first line, n lines]), `*_line_ids.npy` (int32, 0-based line in its file), `*_file_ids.npy`,
  `*_pos_in_chunk.npy`; `train_sample_lines.npy` = (file, line) pairs of the fixed sample. Protocol: training-format text,
  each document cut into 127-token chunks scored after `<s>` (context resets every 127 tokens), fp32.

## Data preparation (`prepare_data.py`)

Documents = runs of non-blank lines joined with `\n`, files in sorted order; **whitespace inside a line collapsed**
(`" ".join(line.split())`). Encoding as `corpus_tokenization/tokenize_corpus.py` (`text.strip()`, no special tokens, int16).
The official repo does not ship its jsonl builder; the rule was inferred and checked against the published checkpoint:
it assigns 13-23 nats to the TAB after CHILDES/Switchboard speaker labels and ~7 nats to a double space (never seen in
training) but ~0 to `\n`; with this rule the corpus has 16,242,414 tokens and 9,914 x 16,384 = 10 x 16,243,098 (raw text:
16,971,177); and the published model's MNTP loss on our dev subset is 3.12 (32k-token CPU check; card: 3.10), versus 3.78
on raw-whitespace dev. Train sample: 13,546 chunks of <=127 tokens, 1,000,040 words, seed 20251009 (independent of runs).

| artefact | value |
| --- | --- |
| tokenizer.json sha256 | e5109a7fbe7ed59be6da3dd1184a466abfa18918ba4796f0d57f079eb0d17cab |
| train_10M token stream sha256 (version-independent) | 0ebf4e97477cf599109405f7e31bf88a4205051d5f2aacf17944bf4bd05e68ef |
| dev token stream sha256 | 03c064ee6fdf59dc9316c10e889361a76a535f2aa391467abf419bda6eeceb56 |
| train sample chunk-table sha256 | 1ff87da9632b5d4d61d5a7896fb8f3672278d071f33e59c227d03b7bf966b884 |
| dev: docs / words / tokens | 12,092 / 10,418,979 / 16,791,105 |

## Verification

- `export_hf.py --verify`: published weights -> train-side `Bert` -> logits vs published HF model on 5 sentences:
  max |dlogit| = **0.0** (causal and masked); export -> reload via AutoModelForCausalLM / AutoModelForMaskedLM
  (trust_remote_code) -> **0.0**; non-default size through the patched config -> **0.0**. So eval2025 (causal, mntp)
  scores our exports exactly like the published one.
- Train-side model through `dev_bpw.nll_of` == exported HF model through `dev_bpw.nll_of` (identical bpw); dumps
  re-scored from the HF export: max diff 0.0.
- Smoke (`python train.py --smoke --device cpu --out_dir X`, ~2.5 min on 1 CPU thread incl. evals): dev-subset loss
  9.86 -> 7.90 (causal), 9.86 -> 8.03 (masked) in 24 steps; checkpoints/exports/dumps/states written; a run stopped at
  step 10 and resumed from the step-7 state ends bit-identical to the uninterrupted run (weights, EMA, optimizer, data RNG, dumps).

## Deviations from the official code / card, and why

1. One process instead of 16 ranks: masked and causal sequences share one batch through the official hybrid forward
   (`num_masked`, `ratio`); loss = m*mean(masked) + (1-m)*mean(causal). Official averages per-rank means (8 sequences each);
   pooled means differ only by per-rank token-count weighting.
2. Sequence length switches by step at 60% / 80% (card). The code switches only at epoch boundaries at 70% / 90% (2024),
   and with 16 ranks its single masked rank (full-data shard, 8 seqs/step) never finishes an epoch, so it would stay at 128;
   here both objectives switch.
3. Constant 16,384 tokens/step, no 4x batch ramp (`batch_reduction` 1): implied by 9,914 steps being exactly 10 epochs.
4. LayerNorm eps 1e-7 (published config) instead of 1e-5 (repo `configs/small.json`); vocab 16,384, LR 0.007 (card).
5. Data shuffling: own per-objective shuffled index streams (randperm per pass, drop_last) with a private RNG instead of
   per-rank DataLoaders; same distribution, not the same order (the official order cannot be reproduced anyway).
6. The batch for step s uses mask p(s); official prefetches it with p(s-1). Negligible.
7. EMA kept like train_10m.py; **raw weights are exported as the primary model** (`hf/final`, all `chck_*`); which one the
   published checkpoint is cannot be determined. `hf/final_ema` and both dev-subset losses are saved for comparison.
8. Dropped: wandb, ModelLogger, DDP. Dev evaluation is extra (fp32, eval mode) and does not touch training.
9. `--hybrid_numerator` is the CAUSAL share here; in train_10m.py it counts MASKED ranks (printed at start-up).

## File hashes (sha256)

See `HASHES.txt` (regenerate with `sha256sum $(git ls-files .)` after commit).
