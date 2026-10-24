# sprint1_port: the frozen sprint-1 recipe on BabyLM 2025 strict-small

The frozen sprint-1 baseline (`origin/sprint-1:sprint1/configs/frozen_baseline.json`, config `s2_B_base`, a
modded-nanogpt speedrun recipe at 6 layers x 384 wide) trained on BabyLM train_10M for 10 epochs, with checkpoints
at the BabyLM word milestones, dev-split held-out loss, and an HF export that the official 2025 pipeline loads with
`--backend causal`.

## Commands (one seed on one CUDA GPU)

```bash
cd .            # repo root; venv with torch 2.x, numpy, tokenizers, transformers 4.51.3, safetensors
# 1. data, once (~2 min CPU). Writes data/ (gitignored, 100 MB). Check meta.json hashes against the table below.
python babylm/recipes/sprint1_port/prep_data.py \
    --train_dir /path/to/text_data/train_10M/train_10M --dev_dir /path/to/text_data/dev/dev
# 2. train (frozen args are built in; nothing else to pass)
python babylm/recipes/sprint1_port/train.py --seed 0 --save_per_token \
    --out babylm/recipes/sprint1_port/runs/s1port_small_s0
# 3. HF export of every checkpoint (chck_1M..chck_90M, final; final also as chck_100M)
python babylm/recipes/sprint1_port/export_hf.py --run_dir babylm/recipes/sprint1_port/runs/s1port_small_s0 \
    --out babylm/recipes/sprint1_port/runs/s1port_small_s0/hf --final_as chck_100M
# 4. (optional, CPU, a minute) re-verify an export against the training module and the official loader
python babylm/recipes/sprint1_port/verify_export.py --ckpt <run>/ckpt/final.pt --scratch /tmp/verify
# 5. official evaluation (from eval2025/): ./eval_zero_shot.sh <run>/hf/chck_100M causal
```

Seeds 1 and 2: `--seed 1`, `--seed 2` (the seed sets model init and data order; `--data_seed` decouples them).
CPU smoke: `python babylm/recipes/sprint1_port/train.py --smoke --out /tmp/smoke` (tiny model, 30 steps, real
tokens, milestone unit 1000 "words" so every checkpoint path runs).

## Expected budget (seed 0; seeds 1-2 differ by one window)

| | frozen sprint-1 (FineWeb) | this port (BabyLM train_10M) |
|---|---|---|
| train tokens available | 2 shards x 100M | 17,191,976 (GPT-2 BPE, incl. 13,705 EOT) |
| budget | 152,043,520 tokens = 2,320 steps | 10 epochs: 167,880 windows -> **2,623 steps = 171,900,928 tokens** (**1.131x** the frozen budget), 9.9995 epochs |
| words seen (est.) | - | 171.9M x 0.57898 words/token = **99.53M words** |
| batch | 64 x 1024 = 65,536 tokens | unchanged (one epoch = 16,788 windows = 262 steps, so no change needed) |
| steps per 1M words | - | 26.35 (one step = 37,944 words) |

Wall time: sprint-1 measured the frozen baseline at **494k-498k tokens/s on a spot A100-40GB, compiled**
(`docs/ladder_result.md`, `docs/ruler_result.md` on origin/sprint-1). 171.9M tokens / 495k = **~5.8 min of training**,
plus ~1 min compile, ~45 curve evals of 0.52M tokens (~0.5 min), the full-dev final eval (17.4M tokens, ~0.3 min)
and 20 checkpoint writes (~8 GB, ~1 min): **about 9-10 min per seed on an A100**. Not measured here (no GPU on this
host). Disk per seed: 19 milestone checkpoints x 352 MB (fp32 model only) + final.pt with optimizer (~1 GB) = ~7.7 GB,
and the same again for the HF exports. Params: 87,883,803 (10.6M non-embedding; 4 vocab tables of 50304 x 384:
embedding, 2 value embeddings, lm head).

## Outputs (`<out>/`)

- `curve.csv`: one row per step: `step, tokens_seen, words_seen_est, epoch, lr_scale, train_loss, dev_loss, elapsed_s`
  (`dev_loss` filled at step 0, every 100 steps, at every word milestone and at the end; curve evals use the dev
  subset below). `curve.png` is drawn by the shared `babylm/tools/plot_curve.py`.
- `result.json`: final full-dev loss / bpb / new-seen split / per-domain loss, every curve eval (also per domain),
  checkpoint list with exact step, tokens and words, tokens/s, config (incl. sha256 of data and vendored code).
- `log.jsonl` (per step), `config.json`, `per_token.npz` (with `--save_per_token`: loss, target, window, pos, seen,
  domain for every scored dev target, for the bottleneck slices).
- `ckpt/chck_<n>M.pt` (model state, fp32, step/tokens/words) and `ckpt/final.pt` (+ optimizer, sampler pointer, RNG,
  source).

## Held-out loss

Official BabyLM dev split (`text_data/dev/dev/*.dev`, 10.42M words, 17,413,401 tokens), tokenised exactly like train.
- Curve: 512 windows of 1,024 targets at `round(linspace(0, N - 1025, 512))`, i.e. evenly spaced over the whole dev
  stream (all six domains in token proportion), 524,288 targets (sprint-1's small curve-eval size). Deterministic,
  identical for every seed and arm; fingerprint (sha256 of starts + first/last window) in `config.json`.
- Final: every non-overlapping window `k * 1024`, k < 17,005 -> 17,413,120 targets (the full dev split).
- Mean next-token nats over all targets of the windows (no BOS added: windows start mid-stream, as sprint-1), bits
  per byte (EOT = 1 byte), sprint-1's new/seen split, and per-domain loss (domain of the target token). Fixed eval
  batch (32 on CUDA, 8 on CPU), padding rows score nothing, bf16 autocast on CUDA as in sprint-1.

## What is unchanged

- Model: `vendor/sprint1/models/baseline.py`, `vendor/sprint1/harness/model_api.py`, `vendor/sprint1/harness/optim.py`
  are byte-identical to origin/sprint-1 (git blobs `ref-0494`, `ref-0495`, `ref-0496`; identical at the frozen-final
  commit `ref-0497`). Size small: 6 layers, 6 heads, 384 wide, MLP 4x ReLU^2, seq 1024, vocab 50304, untied, RMSNorm,
  half-truncated RoPE, QK-norm, 2 value-embedding tables, x0 injection, U-net skips, softcap 15.
- Optimizer and schedule: Muon (NS5, Nesterov) on the 2-D block matrices at 0.065; AdamW embed 0.78, head 0.0208,
  scalars 0.052, betas (0.8, 0.95), eps 1e-10; no weight decay, no grad clip; no warmup; constant then linear
  cooldown over the last 50% of steps to 0.1x; Muon momentum 0.85 -> 0.95 over 150 steps; no batch scaling.
  `train.py` parses the frozen args list (`FROZEN_ARGS`, copied from frozen_baseline.json) before the command line.
- Training step: copied line for line from `sprint1/harness/train.py` (16-sequence micro-batches with gradient
  accumulation, bf16 autocast, tf32, `torch.compile` on CUDA, `lr_factor`, momentum warmup).
- Seeding: `torch.manual_seed(seed)` before the model is built; data RNG `PCG64(data_seed + 1_000_003)`.

## Deviations from the frozen recipe, and why

1. **Data.** BabyLM train_10M instead of FineWeb, tokenised with GPT-2 BPE (`openai-community/gpt2` tokenizer.json from
   the HF hub; tiktoken is not installed here; the BPE is the same as tiktoken gpt2). Join rule (`prep_data.py`): files
   in sorted name order; a document = a maximal run of non-blank lines (blank lines are the only document boundaries
   the files carry: 12,285 simple_wiki articles, a few hundred breaks elsewhere, none in switchboard); each document
   is `<|endoftext|>` + the lines joined with `\n` (+ final `\n`), text verbatim (speaker tags kept). Documents are
   **not shuffled** in the stream; data order comes only from the sampler. As in sprint-1, training windows cross
   document boundaries with full causal attention (sprint-1 dropped FlexAttention document masking).
2. **Sampler.** Sprint-1 drew 1,025-token windows uniformly at random with replacement, which has no exact epoch. The
   BabyLM cap ("at most 10 epochs, counting every pass") needs one, so each epoch here takes a random phase offset in
   [0, 1024) and visits all non-overlapping windows from it in a random permutation; epochs run back to back, a step
   may straddle two. Every target token is seen at most once per epoch, at most 10 times in all.
3. **Budget.** 10 epochs = floor(total windows / 64) steps (2,623, the tail < 64 windows of epoch 10 is dropped),
   instead of 2,320 steps. The schedule shape is the same (it is defined in fractions of total steps); only the
   momentum warmup is in absolute steps (150), unchanged.
4. **Held-out set.** BabyLM dev instead of the FineWeb val half; curve evals every 100 steps and at each word milestone
   (sprint-1: every total/8 steps) plus step 0; final eval on the full dev split.
5. **Checkpoints.** `chck_1M..chck_10M`, `chck_20M..chck_90M` saved at the first step whose words_seen_est (tokens x
   train words/token, 0.57898; words = whitespace-split strings, 9,953,773) reaches the milestone, so each overshoots
   by less than one step (< 37,944 words, < 3.8% at 1M). 10 epochs of 9.95M words is 99.53M words, under 100M, so
   `chck_100M` is the final model (`--final_as chck_100M`); its exact word count is in `export_info.json`.
6. **HF export** (`export_hf.py`, `hf_template/modeling_sprint1.py`): the baseline network copied with two edits
   that do not change the computed function (HF config instead of ModelConfig; RoPE tables computed on the CPU per call
   size instead of registered buffers, because HF can build modules on the meta device). `AutoModelForCausalLM` ->
   fp32 softcapped logits [B, T, 50304]; `AutoModel` -> final normed hidden state (for the official finetuning
   head; `hidden_size` = 384). `tie_word_embeddings=False` is explicit (HF ties by default, which silently replaced the
   lm head on the first try). Tokenizer: GPT-2 vocab/merges/pre-tokenizer unchanged; the post-processor prepends
   `<|endoftext|>`; bos = eos = pad = unk = `<|endoftext|>`, padding right. The official causal collate feeds
   `tokens[:-1]` and scores `tokens[1:]`, so without a BOS the first word of every sentence would never be scored (the
   official GPT-2 baseline's tokenizer prepends `<s>` the same way). Only right padding is accepted (the network has
   no padding mask; with causal attention, right padding cannot change real-token logits); other masks raise.
   `config.autocast_bf16` (default false) runs the CUDA forward in bf16 as in training; the default fp32 forward is the
   exact function of the weights, so official scores and the trainer's bf16 dev loss differ at the bf16-rounding level.

## Verification (CPU, this host)

`verification/verify_export_small_cpu.json` (size small, 3 smoke steps) and `verify_export_tiny_cpu.json` (tiny,
30 steps):
- logits, training module vs `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`: **max |diff| 0.0** on
  4 dev windows x 1,024 and on 10 BOS-prefixed sentences; AutoModel hidden state max |diff| 0.0; training-forward
  loss 6.437486 vs CE from HF logits 6.437486 (diff 4.8e-7).
- right padding (official collate, pad 50256): real-token logits change by at most 1.0e-5 (fp32 kernel rounding for a
  different sequence length; causal attention cannot see the pads).
- official loader (`AutoProcessor` + `sentence_zero_shot.dataset.get_dataloader`): BOS 50256 prepended with offset
  (0, 0), the first word is inside the scored phrase, and the pipeline's sentence log-probs equal the training
  module's to 3.8e-6 nats.
- official `python -m evaluation_pipeline.sentence_zero_shot.run --task blimp --backend causal` on a 5-pair
  BLiMP-format smoke file (written to scratch, not the real data): runs to completion and writes its report.
- smoke curve: `verification/smoke_tiny_curve.{csv,png}`.
Not verified here: the CUDA path (compile, bf16) — no GPU on this host. The first GPU run should be read for
tokens/s near 495k and for a dev curve that starts at ~10.83 nats (ln 50304 = 10.83).

## File hashes (sha256)

| file | sha256 |
|---|---|
| train.py | f292a4e0c01e0f1d306b59a26da9ec6c1e3ea5dd88f904bd434ad19906a85c4d |
| prep_data.py | c07ca0a53901aaaae2deee959f57d8458269a4f60347494100c92502b593bb33 |
| export_hf.py | ef4c7cae9dab5a2b5213aae247d48159683e25dcba2f5c5eda7a297abc6bd36b |
| verify_export.py | 777331305b89fbb34862bdbd0c394047d34930768daed669958df0c97168eaa8 |
| hf_template/modeling_sprint1.py | 7ad485ea1c6ea94e657439ff55a169ed0c8f1d694bbd01dff2b4eba8df538f5c |
| vendor/sprint1/models/baseline.py | 25f5eabd1b1b9297eab3ab007c0512afc11cfa0aeea2407d3943259cc9cb5b64 (git blob ref-0494) |
| vendor/sprint1/harness/model_api.py | 998e5b02cfc8a46402e07d9dbf975d67574dbbed768e347a248ff535852a0d3f (git blob ref-0495) |
| vendor/sprint1/harness/optim.py | a66df6dea7b0f578b7314f9479c145623bbe3e9860ab20d951485d8c054284ac (git blob ref-0496) |
| gpt2_tokenizer/tokenizer.json | 8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6 |
| gpt2_tokenizer/vocab.json | 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783 |
| gpt2_tokenizer/merges.txt | 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5 |
| data/train.bin (prep output) | a6b1898dfc474a51c79cdbbe0226e930cb6e0f6888e74c184a8a8726b8912f19 |
| data/dev.bin (prep output) | f052b5dde5f4ff06e040a5c382a79ece01c6e2fa0a605bb14b11fda556252053 |
| data/train_domain.bin | 5f98751539581326c3822cf322e0f5dc6fd0903c63e18d3e8304068a4eec35cb |
| data/dev_domain.bin | 43f839abda6c76b743e36e7583f384afcb4a6687e8c0dbeb4ce0af58fb2ddf27 |
| data/token_bytes.npy | 04c953be0702d45358aa19991151620dbd40b9b4939da1884ae2be860b565cc3 |

Source text file hashes and per-domain words/documents/tokens are in `data/meta.json` (train: bnc_spoken 1.26M tokens,
childes 6.93M, gutenberg 3.47M, open_subtitles 3.26M, simple_wiki 2.03M, switchboard 0.25M). Recipe source:
origin/sprint-1 at ref-0498.

## Open faithfulness questions (for the critic)

- **BOS at evaluation vs EOT density in training.** The official causal eval puts BOS = `<|endoftext|>` before every
  sentence; our stream has an EOT only every ~1,250 tokens on average (13,705 in 17.2M; childes and switchboard almost
  never). Sprint-1 had one per FineWeb document (density not measured here), so the convention matches sprint-1, but
  "EOT then a sentence" is rare in BabyLM training text. It affects both sentences of a minimal pair alike, yet it
  could cost BLiMP points. `prep_data.py --doc_unit line` (EOT before every line, 1.18M EOTs) exists as an attack, not
  as the port.
- **Tokenizer and strict-track rules.** GPT-2 BPE was trained on WebText, not on BabyLM data. Check whether the 2025
  strict rules accept an externally trained tokenizer before reporting this as a strict-small entry; the vocab (50k)
  is also 3x the official baseline's 16k, so ~88% of the 88M parameters are vocabulary tables.
- **Repeated data.** The frozen recipe was tuned for single-pass FineWeb with no weight decay or dropout; 10 passes of
  10M words is a different regime (the per-epoch dev curve will show whether it overfits). Kept frozen on purpose.
- Nats per GPT-2 token are not comparable with models using other tokenizers; compare bpb or the official scores.
