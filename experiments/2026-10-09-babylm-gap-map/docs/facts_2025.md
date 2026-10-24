# BabyLM 2025 strict-small (10M): facts for reproduction (checked 2026-10-09)

Sources (each fact below cites one of these):
- [AMLM] Edman & Fraser 2025, https://aclanthology.org/2025.babylm-main.31.pdf (arXiv 2510.20475)
- [AMLM-code] https://github.com/Leukas/babylm25 (HEAD d1438b077761a891ec2b47c6854ea34be6e6e6aa, last commit 2026-05-20)
- [FIND] Findings of the 3rd BabyLM Challenge, https://aclanthology.org/2025.babylm-main.28.pdf
- [CFP] https://arxiv.org/pdf/2502.10645
- [EVAL] https://github.com/babylm/evaluation-pipeline-2025 (main = bf55c1131e53654c2a87418f5629c26959acd710, 2025-08-19 "Update eval_finetuning.sh"; from `git ls-remote` + shallow clone, because the GitHub API was blocked here)
- [LB] https://huggingface.co/spaces/BabyLM-community/babylm-leaderboard-2025-all-tasks (sha ref-0231…, src/about.py, src/leaderboard/read_evals.py)
- [HF-x] model cards and config.json at https://huggingface.co/BabyLM-community/babylm-baseline-10m-{gpt-bert-causal-focus,gpt-bert-mixed,gpt-bert-masked-focus,gpt2}
- [OSF] https://osf.io/ryjfm/ (BabyLM_2025 node), read through https://api.osf.io/v2/nodes/ryjfm/files/osfstorage/

## 1. Winner: AMLM-Hard-Decay (NLP-score winner, strict-small) [FIND §6.1]
**Method** [AMLM §3.1, code train_mask.py]:
- Every vocabulary type i keeps a mask weight w_i, initialised to p_mlm.
- Every 200 batches (one "timestep"): w_t,i = λ·w_{t-1,i} + (1-λ)·p_mlm·(1-score_{t-1,i}), with λ = 0.2. In the code this is `0.2*old + 0.8*new`.
- "Hard" score = smoothed accuracy (correct+0.5)/(total+1) on the masked predictions of type i in the last window. "Soft" score = 1 - min-max-normalised mean CE loss.
- Weights are renormalised so that the mean over the vocabulary is p_mlm, so a single type can exceed p_mlm.
- Masking draws tokens in proportion to these per-type weights. The usual 80/10/10 replacement applies: Random 0.1, Keep 0.1.
- **Decay** = p_mlm falls linearly from 0.4 to 0.15 (`--mlm_prob 0.4 --mask_decay 0.25`).
- Other changes vs GPT-BERT: DeBERTa-v2 architecture (HF DebertaV2ForMaskedLM) instead of GPT-BERT; MLM only, with no causal/MNTP mix; vocab 40k BPE with byte-fallback (SentencePiece `spm.model`); **a different training corpus** (see the caveat below).
- n-hot sub-token embeddings appear only in the separate "N-hot Hard" model, not in Hard-Decay.

**Hyperparameters** [AMLM Table 4; AMLM-code README command]

| item | value |
|---|---|
| arch | DeBERTa-v2, 12 layers, hidden 384, FFN 1280, dropout 0.1. HF config: 12 heads, rel-pos buckets 256, max_pos 1024 |
| params | not stated in the paper (UNVERIFIED; roughly 15M non-embedding + 40k×384 ≈ 15.4M embedding) |
| tokenizer | BPE + byte-fallback, vocab 40000 (`tokenizers/bb24.model`, sha256 ref-0232…) |
| seq len | 64 for epochs 0-4, then 256 (`--max_seq_len 0:64,5:256`). Code divides the batch by the length ratio, so tokens per batch stay constant |
| batch | 16384 tokens (256×64) |
| steps / epochs | 8325 / 10 |
| LR / sched | 7e-3, cosine, 1% warmup (`get_cosine_schedule_with_warmup`, warmup = total//100) |
| optimizer | LAMB (bitsandbytes). Paper says β2 0.95; **code uses betas (0.9, 0.98)** for LAMB (0.95 only on the AdamW path). eps 1e-8, wd 0.01, clip 1.0, z-loss 1e-4 (code) |
| mask ratio | 0.4 → 0.15 linear |
| seeds | dev results (Table 1) "averaged over 3 runs", with no std reported. Submission = seed 0 only |
| run cmd | `python train_mask.py --train_data data/bb24.train --valid_data data/bb25_small.dev --tokenizer tokenizers/bb24.model --mask_update_steps 200 --logging_steps 200 --intermediate_size 1280 --hidden_size 384 --max_seq_len 0:64,5:256 --lamb --all_checkpoints --mlm_prob 0.4 --mask_decay 0.25 --seed 0` |

**HF models**: leukas/amlm_hard_decay (main sha ref-0233…, branches chck_1M…chck_100M), leukas/amlm_hard, leukas/amlm_hard_nhot, leukas/amlm_hd_fail (the failed run), collection https://huggingface.co/collections/leukas/babylm-2025-68a74f07d8570c914c69e4be. Model cards are empty auto-generated templates.

**Scores (strict-small)**. AMLM Table 2 holds the submitted seed-0 model. AMLM Table 3 and [FIND] Table 3 hold the leaderboard numbers.

| task | AMLM-HD T2 | AMLM-HD T3 (leaderboard) | 3-run avg HD (T1) | GPT-BERT masked MNTP (T3) | GPT-BERT causal (T3) |
|---|---|---|---|---|---|
| BLiMP | 71.4 | 71.4 | 71.3 | 70.4 | 71.7 |
| BLiMP Suppl | 59.2 | 59.2 | 58.3 | 63.7 | 63.2 |
| EWoK | 51.0 | 51.0 | 50.9 | 50.0 | 49.5 |
| Entity Tracking | 44.2 | 44.2 | 44.4 | 40.1 | 34.6 |
| Adj nominalization (wug_adj) | 34.0 | 22.3 | 14.3 | 2.7 | 59.2 |
| Past tense (wug_past) | 6.0 | 6.2 | 1.3 | 28.7 | 12.9 |
| COMPS | 54.2 | 54.2 | 54.0 | 53.6 | 52.8 |
| Reading (mean of ET, SPR) | ET 8.3 / SPR 3.5 | 5.9 | 8.9 / 3.7 | 6.4 | 6.7 |
| AoA | -0.9 | -0.9 | n/a | 0.3 | -3.9 |
| (Super)GLUE avg | 70.1 | 69.8 | 70.1 | 66.0 | 65.1 |
| GLUE: BoolQ/MNLI/MRPC/QQP/MultiRC/RTE/WSC | 69.5/62.3/90.5/73.1/68.6/63.3/63.5 | – | 69.2/62.3/90.6/73.1/68.2/64.3/63.5 | 67.6/51.4/86.1/67.4/71.6/57.5/61.5 (T2) | – |
| 10-task avg | 38.3 | 38.3 | – | 38.2 | 41.2 |

[FIND] Table 3, strict-small: **AMLM-Hard-Decay human-likeness 8.4, NLP 58.3, macro 33.3**. AMLM-Hard: 26.7 / 58.0 / 42.3. MoEP (human-likeness winner): 31.5 / 53.2 / 42.3.

**CAVEATS found in this check (not stated in [FIND])**
- (a) **AMLM's training data is not the official 10M corpus.** `data/bb24.train` has 10,009,099 whitespace words, slightly over 10M.
  - Only 32.2% of its words come from lines of the official train_10M; I computed this myself.
  - The paper says CHILDES was "removed, and replaced with data from Zhang et al. (2023a)", i.e. SynCSE. That data is "synthetically generated" by large language models (https://arxiv.org/abs/2305.15077).
  - The file holds NLI triplets: premise, entailment paraphrase, contradiction.
  - I checked the eval pipeline's MNLI fine-tuning data ([OSF] glue_filtered) against this file:
    - **14,313 of 17,646 MNLI-train sentences (≥6 words) appear verbatim in bb24.train**
    - 78 of 6,887 MNLI-valid sentences appear verbatim
    - MRPC, RTE and QQP valid: 0, 0 and 1.
  - AMLM Table 5: switching to this data alone lifts regular-MLM MNLI from 44.0 to 56.6 (16k vocab), and fine-tune avg from 64.8 to 67.5.
  - So **much of AMLM's GLUE edge comes from data, not from masking**.
- (b) The NLP-score margin over the baselines is small: 58.3 vs 57.3 for GPT-BERT mixed-MNTP and masked-MNTP.
- (c) Macro (33.3) is *below* every GPT-BERT baseline.
- (d) The authors themselves show that wug_adj (≈77-78 from an untrained model), AoA, EWoK and COMPS are unreliable (AMLM App. C, Table 3).

## 2. Official 2025 strict-small baselines [FIND §3.1; HF cards]
| | GPT-BERT (causal-focus / mixed / masked-focus) | GPT-2 small |
|---|---|---|
| HF id | BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus (sha ref-0234), -mixed (ref-0235), -masked-focus (ref-0236) | BabyLM-community/babylm-baseline-10m-gpt2 (ref-0237) |
| code | https://github.com/ltgoslo/gpt-bert (HEAD ref-0238; this is the 2024 code, whose defaults differ: vocab 8192, LR 1.41e-2, batch 32768, 7812 steps. The exact 2025 launch config is UNVERIFIED) | https://github.com/momergul/babylm-gpt2-baseline (HEAD ref-0239) |
| params | ~31M | 124M |
| layers/hidden/heads/FFN | 12 / 384 / 6 / 1280 (GEGLU, LTG-BERT/DeBERTa disentangled attn, NormFormer LN), pos buckets 32 | 12 / 768 / 12, n_positions 1024 |
| vocab | 16384 (tokenizer.json in repo) | 16384 |
| causal:masked ratio | 93.75:6.25 / 50:50 / 6.25:93.75 | 100% causal |
| batch | 16384 tokens | 16 × 512 = 8192 tokens |
| steps | 9914 (≈10 epochs; ckpt every 99.14 steps = 1M words) | 21300, warmup 213 |
| LR / sched | 7e-3, warmup 1.6% + cosine + cooldown 1.6% | 5e-5, cosine, 1% warmup |
| optimizer | LAMB β(0.9, 0.98), eps 1e-8, wd 0.1, clip 2.0, z-loss 1e-4 | AdamW β(0.9, 0.999), eps 1e-8, clip 1 |
| mask ratio | 0.3 → 0.15 linear; random 0.1, keep 0.1 | – |
| seq len | 128 (60%) → 256 (20%) → 512 (20%), tokens per batch constant | 512 |
| train time | "40 minutes … 10.67 GPU-hours" (8 MI250X = 16 GCDs) | "~0.5 GPU hours" (1 H100) |
| val loss (1M dev words) | 3.10 / 2.78 / 2.79 | n/a |

Note: the findings paper says the GPT-2 batch is 16 (8192 tokens). The prompt's "16 × 512" is the same thing. The card says 10 epochs.

**Leaderboard scores** ([FIND] Table 3; Human-likeness / NLP / Macro):

| model | Human | NLP | Macro |
|---|---|---|---|
| causal MNTP | 17.4 | 56.7 | 37.1 |
| causal AR | 18.7 | 56.1 | 37.4 |
| mixed MNTP | 13.4 | 57.3 | 35.4 |
| mixed AR | 16.6 | 54.3 | 35.4 |
| masked MNTP | 9.5 | 57.3 | 33.4 |
| masked AR | 18.9 | 54.0 | 36.5 |
| GPT2 | 19.8 | 49.1 | 34.5 |

**Per-task scores from the model cards** (BLiMP / Suppl / EWoK / ET-reading / SPR / EntityTr / WUG-adj [old accuracy metric]):

| model | scores |
|---|---|
| causal-focus, Causal | 71.66 / 63.21 / 49.49 / 9.89 / 3.45 / 33.96 / 43.00 |
| causal-focus, MNTP | 69.07 / 64.33 / 49.62 / 9.47 / 3.48 / 39.17 / 43.00 |
| mixed, Causal | 69.62 / 61.56 / 50.23 / 9.50 / 3.37 / 22.27 / 45.00 |
| mixed, MNTP | 71.29 / 63.30 / 49.93 / 9.78 / 3.33 / 39.95 / 16.00 |
| masked-focus, Causal | 65.22 / 59.49 / 49.47 / 9.52 / 3.44 / 30.60 / 68.00 |
| masked-focus, MNTP | 70.36 / 63.71 / 49.95 / 9.40 / 3.37 / 40.02 / 57.5 |
| GPT-2 | 66.36 / 57.07 / 49.9 / 8.66 / 4.34 / 13.9 / 52.5 |

**GLUE from the model cards** (BoolQ / MNLI / MRPC-F1 / QQP-F1 / MultiRC / RTE / WSC):

| model | scores |
|---|---|
| causal-focus | 68.07 / 46.86 / 83.44 / 72.03 / 68.28 / 56.12 / 65.38 |
| mixed | 66.67 / 53.30 / 84.56 / 76.56 / 67.04 / 55.40 / 63.46 |
| masked-focus | 67.58 / 51.38 / 86.08 / 67.37 / 71.63 / 57.55 / 61.54 |
| GPT-2 | 66.91 / 51.12 / 80.77 / 62.45 / 65.72 / 56.83 / 61.54 |

Note: the [FIND] leaderboard values (e.g. masked BLiMP 70.4, GLUE 66.0) differ slightly from the card values. The leaderboard is the reference.

## 3. Evaluation pipeline [EVAL README + scripts]
- **Zero-shot (full)** `./eval_zero_shot.sh <model> <backend> [evaluation_data/full_eval]`, covering:
  - blimp_filtered, supplement_filtered, ewok_filtered, entity_tracking, wug_adj_nominalization, wug_past_tense, comps, reading (reading_data.csv)
- **AoA**: `./eval_aoa.sh <model> <backend> <track>`. It runs on all checkpoints and uses CDI words with c4-en-10k contexts.
- **Finetune**: `./eval_finetuning.sh <model> [LR 3e-5] [BSZ 32] [BIG_BSZ 16] [EPOCHS 10] [WSC_EPOCHS 30] [SEED 42]`, with seq len 512.
  - Tasks: boolq and multirc (bsz 16); rte, wsc (30 ep), mrpc and qqp (valid metric F1), mnli (3-class).
  - MNLI and QQP are subsampled to 10k train examples. CoLA, SST-2, MNLI-mm and QNLI were removed.
- **Fast** = 20% of each zero-shot task [FIND §4.2]: `./eval_zero_shot_fast.sh <model> <revision> <backend>`. It covers blimp_fast, supplement_fast, ewok_fast, wug_adj, wug_past, entity_tracking_fast and reading; there is no COMPS and no finetune. It is required on the checkpoints chck_1M…chck_10M and chck_10M…chck_100M (strict-small).
- **Backends**: `causal`, `mlm`, `mntp` (MLM shifted like causal), `enc_dec_mask`, `enc_dec_prefix`. The scorer supports min/max temperature sweeps (default 1.0).
- **Leaderboard aggregation** [LB read_evals.py, about.py]:
  - Benchmarks: BLiMP, BLiMP-Suppl, EWoK, Entity Tracking, wug_adj (Spearman), wug_past (Spearman), COMPS, Reading (mean of SPR and eye-tracking ΔR²), AoA, (Super)GLUE (mean of 7: acc, F1 for MRPC/QQP).
  - The "text average" = sum of the 10 benchmarks / 10. A missing task counts as 0.
- **NLP score vs human-likeness** (I derived these and they reproduce [FIND] Table 3 exactly; [FIND] does not spell them out):
  - **NLP** = mean(BLiMP, Suppl, EWoK, EntityTracking, COMPS, GLUE)
  - **Human-likeness** = mean(wug_adj, wug_past, Reading, AoA)
  - **Macro** = (NLP + Human)/2
  - Check: AMLM-HD gives (71.4+59.2+51.0+44.2+54.2+69.8)/6 = 58.3 and (22.3+6.2+5.9-0.9)/4 = 8.4.
- **Eval data**: OSF https://osf.io/ryjfm/ → `evaluation_data/{full_eval,fast_eval}`.
  - EWoK full is gated on HF (ewok-core/ewok-core-1.0) and is built with `python -m evaluation_pipeline.ewok.dl_and_filter`.
  - The ewok_fast.zip password is "BabyLM2025" (stated in the README).
  - GLUE file md5s are listed via the OSF API, e.g. mnli.valid ref-0240…, mnli.train ref-0241….
- **Versions** (requirements.txt): Python 3.13 (README), torch==2.7.0, transformers==4.51.3, datasets==3.6.0, tokenizers==0.21.1, numpy==2.2.5, scikit-learn==1.6.1, statsmodels==0.14.4.
- **Run time**: not documented. UNVERIFIED.

## 4. Training data [OSF; data README]
- 2025 node https://osf.io/ryjfm/, `text_data/` (uploaded 2025-05-06). The zips are **byte-identical to 2024** (https://osf.io/ad7qg/, same md5).

| file | download | md5 | sha256 (prefix) |
|---|---|---|---|
| train_10M.zip | https://osf.io/download/y7djq/ | dc67e5d2db96ed4f76cb072e2504a18c | 6d9d2cb3b3feccf0 |
| train_100M.zip | https://osf.io/download/ywea7/ | f63cde477afd5673f3367fd009ea70cb | 37a92db26f391e43 |
| dev.zip | https://osf.io/download/uyd7v/ | ad712e2f943ddbbbc94036178144e810 | b5af75aeb1ee5a36 |
| test.zip | https://osf.io/download/ftwu3/ | dc7f838fe026dfd9781c6f41cba1ebd9 | 12bd1c9c89f4f821 |

- Six files per split: bnc_spoken, childes, gutenberg, open_subtitles, simple_wiki, switchboard (`.train` / `.dev`).
- `wc -w` counts I measured:

| file | train_10M | train_100M |
|---|---|---|
| bnc_spoken | 932,468 | 7,759,945 |
| childes | 2,839,590 | 28,894,784 |
| gutenberg | 2,539,445 | 26,367,028 |
| open_subtitles | 2,036,948 | 19,924,117 |
| simple_wiki | 1,451,320 | 14,653,591 |
| switchboard | 146,789 | 1,342,029 |
| **total** | **9,946,560** | **98,941,494** |

  dev total: 10,402,202.
- **10M ⊂ 100M**: 100% of the non-empty lines in every 10M file occur verbatim in the matching 100M file (line-set check). The order is not a prefix for simple_wiki.
- No HF dataset mirror from BabyLM-community for English 2025 text was found. ltg hosts its own 2024 variants (ltg/babylm-2024-baby-cosmo-fine-10m).

## 5. Rules on auxiliary models and synthetic data [CFP §4.1, §4.2, FAQ]
- Strict and Strict-Small rules are "unchanged from last year": "The STRICT and STRICT-SMALL tracks require that submissions be trained on a corpus of 100M words or less (in STRICT) and 10M words or less (in STRICT-SMALL). These tracks do not require that participants use the official BabyLM corpus…"
- Exposure cap: "we require the submission of results for models trained on a fixed amount of input (counting repeated exposures), in particular after at most 100M words for the STRICT-SMALL track … we instead quantify training by the number of whitespace-separated input words."
- FAQ: "Any training objective/regime is permitted as long as the data restrictions are followed. If you use ancillary models, for example, in the case of reranking or data augmentation, the training data for these models is counted towards your 100M word budget. This applies to all tracks, including INTERACTION track; so, for example, while you can use the external model to produce POS tags, you cannot use an off-the-shelf POS tagger in your pipeline."
- FAQ: "Can I use external tools? Yes, but if they are learned on language, their tokens are counted towards the 100M. … the sum of all text seen by all training can not surpass the amount of text allowed. This raises the question of synthetic data, which is allowed under some restrictions. You may generate the 100M tokens in any legal way you like (yes, distilling or writing your own is fair …), you may also train eventually on more than 100M words by augmentation, however, that only works in a closed system, i.e., the augmenters' training data counts toward the limit, so, for example, training two LMs on half of the words, and then having them generate more words and training a model on both the original data and the new one is legit."
- Interaction track only: "this word count includes text generated by external models and pre-existing corpora. Additionally, the submission model may not generate more than 100M words during the training process." [CFP §4.1]
- Checkpoints: every 1M words up to 10M, then every 10M up to 100M (strict-small). [CFP §4.2]
- **Implication**:
  - A teacher or augmenter for strict-small must itself be trained only within the 10M-word budget, as a closed system.
  - Generated words the student trains on count as input exposures toward the 100M cap (whitespace words). Per the FAQ's closed-system rule this is how I read it; strict-small does not say so as explicitly as the Interaction track text does.
  - AMLM's LLM-synthesised SynCSE data seems to conflict with the letter of the "augmenters' training data counts" rule, yet the organisers ranked it first. Treat this as a judgement call for our critic.
