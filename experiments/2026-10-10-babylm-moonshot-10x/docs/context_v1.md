# BabyLM 10x moonshot: shared context for every agent (v1, 2026-10-10)

Owner: the 10x moonshot thread (ultracode). Atila (owner) asked for an unconstrained, max-effort attempt at **10x data
efficiency**: a model trained on BabyLM's 10M words that matches a model trained on 100M words, with the gain holding at
two model sizes. He wants mechanisms with **20x-sized ceilings**, and honest odds. "Nobody has shown it" is not a reason
to stop; it is also never a reason to overstate a result. Never call a known idea new.

Repo: atilavahedian/hinton (private). Branch base: the research experimenter's harness `wip-branch`.
Moonshot branch: `wip-branch`. Local checkout `.`, venv `~/venv`
(CPU torch 2.7.0, numpy, tokenizers 0.21.1, safetensors, matplotlib, pytest, scipy). Data prepared at
`data/babylm` (train_10M + dev + dev_sub/train_sub; **test is never touched**).
Container: 4 CPUs, 15 GB RAM, no GPU. GPU only through the "BabyLM GPU executor 3" thread (A100s, spot).

## 1. The yardstick

- Prize (published GPT-BERT causal 10M vs 100M, overlap-removed, reweighted to dev sources): **0.875 nats per
  whitespace word [0.82, 0.93]**. GPT-2 pair: 1.13. Our own port pair (best epoch): 1.01.
- Log-linear interpolation between 10M and 100M: **2x data ~ 0.26 nats/word, 4x ~ 0.53, 10x ~ 0.88, 20x ~ 1.14**.
  (Unmeasured until the 20M/30M ruler cells land; use it as a rough exchange rate.)
- Published GPT-BERT 10M (causal) scores **4.939 nats/word** on our DEV protocol (`babylm/eval/dev_bpw.py`: full dev,
  ctx 512 / stride 256, whitespace collapsed). The 100M twin is ~0.88 lower (~4.06).
- At size S (12L/384d, ~30M non-embedding), 100M words x 10 epochs beats 100M x 1 epoch by only 0.094 nats/word.
  So the target is essentially "**10M unique words x 10 epochs as good as 100M fresh words x 1 epoch**".
- Our own sprint-1 port base at weight decay 0 overfits catastrophically (DEV 9.9 nats/word at 10 epochs, best 5.48 at
  epoch 2; 60% of params are embedding tables that never saw weight decay). The base builder is fixing this (WD on
  embeddings, tied embeddings, dropout, EMA, MNTP hybrid, 4 epochs). Attacks are judged against the fixed base.

## 2. Where the gap lives (measured on published checkpoints; instruments thread, PR #52)

| Slice | nats/word | Notes |
|---|---|---|
| Seen text (train text both models saw) | 0.24 | the 10M model is behind even on its own training text (part model size: 31M vs 120M params) |
| Extra on new text only | 0.61 | generalisation: the 10M model memorises its contexts |
| Never-seen words | 0.10-0.11 | not in train_10M at all; mostly out of reach (spelling/copying only) |
| Words seen 1-63 times | 0.16-0.18 | few contexts per type; on train text only 0.7-1.0 behind, on new text 2-6.5 behind |
| Words seen 64-8k times | 0.36 | the biggest band: familiar words in new contexts |
| Words seen 8k+ times | 0.22 | |
| Content words | 73% of gap | 44% of words |
| Trigram count model of the 100M words | supplies only 0.16 | ~80% of the gap is not n-gram statistics ("abstraction") |

Learning dynamics: learning on new text stalls after epoch 7 (epochs 7-10 buy 6% of the dev gain) while train loss keeps
falling; dev-minus-train loss grows every epoch (0.51 -> 1.11). Overconfident (loss minus own entropy 0.57 vs 0.39).
No verbatim recall. Long context is NOT a bottleneck (gap flat after the 4th word). Calibration is not (best temperature
0.013). Bursty words (few documents) carry a 1.29x larger gap than evenly spread words of equal frequency.

**Lever signatures** (how to tell what a lever buys): rare tilt = share of loss removed on words seen 1-15x vs 64+x;
novelty tilt = trigram in neither corpus vs in train_10M. 10x data: rare 1.83, **novelty 1.04**. Extra epochs: novelty
**0.43-0.76** (they buy memorisation). GPT-2 -> GPT-BERT recipe change at 10M: 0.41 nats/word gain (about 2.3x data by
log-linear), rare 3.48, novelty 0.96: recipe changes CAN buy data-like gains. A real 10x mechanism should look like data
(novelty tilt near 1, gain on new text and on words seen 64+ times), not like more epochs.

Inference-time headroom at zero exposure (overlaps the above, not additive): own epoch-5/10 snapshot mix +0.048, copy
cache +0.037, temperature +0.013. Count model of the same 10M +0.051 (costs one epoch of exposure). 4-seed logit
ensemble (Baby Llama, 10M): eval loss 3.77 -> 3.66 (-0.11 nats/token, about -0.18 nats/word).

## 3. Rules that make a 10x claim mean anything (critic R0-R6; binding)

- **R0 exposure**: <= 10M unique words; **<= 100M words total exposure**, summed over every pass, every objective, and
  **every model in the pipeline** (teachers, generators, helpers, tokenizer-free models). One sequence used twice in a
  step (two views / two objectives on separate forwards) counts twice; several losses on ONE forward count once.
  Synthetic text from a model trained on the corpus is allowed and counts as exposure when fed. A count model built
  from the corpus was charged one epoch (10M). No external pretrained weights/tokenizers/LLM text (R0.4).
- **R4 win bar**: 3 seeds of attack and reference; > 2 sigma (seed floor) on the official NLP score AND held-out
  bits/word (FINAL split, touched once at promotion) AND the reliable composite; holds at two sizes >= 3x apart in
  non-embedding params ("fades at second size" otherwise). **R4.4: if an attack uses > 1.25x the reference's training
  FLOPs, it must also beat a FLOP-matched reference (wider/deeper model, same epochs).**
- **Tuned-regularisation floor (binding)**: every attack's comparator is its base with weight decay and size tuned on
  DEV under the same trial budget (Kim/Kotha 2025 heavy WD = 2.29x is already in the reference, not our gain).
- **Data multiplier** (headline, not the win bar): the base recipe re-tuned and trained on nested 10/20/30/50/100M
  subsets (the "ruler"), multiplier = D*/10M by monotone interpolation; >10x is reported as "beyond ruler".
- **R6 cards**: before any GPU run, a card (mechanism, why it closes its bottleneck, null twin with the mechanism's
  information removed but same compute/exposure/code path, numeric kill rule, prior-art line, novelty label
  {reproduction, known idea, known ideas new combination, new}) is committed and accepted by the critic. Screens: 1 seed
  at size S vs null twin and base re-run in the same batch; kill if not ahead of both by 2x the seed floor on full-DEV
  nats/word at the final checkpoint (no checkpoint picking). Survivors: 3 seeds + size M + official eval.
- Inference compute is not capped (ensembles at test time cost K x inference, which is allowed and reported).
- Selection uses DEV and the official fast suite only; test (FINAL) is never read by anyone but the critic's promotion.

## 4. What is already known (prior-art map, PR #46; priors table on babylm-1)

Big, measured, same-data levers: Kim/Kotha 2025 heavy WD 2.29x (in the base); **ensembles beat any single model: K=3
ensemble beats the N->inf regularised single-model asymptote; ensemble+param scaling asymptote 5.17x (extrapolated;
measured 2.09x); distilling an 8x ensemble into one model keeps 83%; self-distillation 300M->300M beats its teacher by
0.13 nats** (but their students trained on 13-33B tokens: far past our exposure cap). Muennighoff: repetition worth
~7.8x unique data at 10 epochs (large-scale fit; does NOT hold at BabyLM scale, where models are over-parameterised).
Diffusion LMs vs AR: the gap vanishes once AR is properly regularised (Xu 2026). AR token-noise augmentation -0.21 nats
but only past 60+ epochs. Synthetic bootstrapped pretraining (SBP): 42-58% of a 10-20x oracle at 3-6B params.
GPT-BERT causal+masked hybrid: the only objective change that won big and repeatedly in BabyLM (it is the baseline).
Dead in BabyLM: data-order and vocabulary curricula (3 years), linguistic/syntactic biases, vision, exotic tokenizers
(char/phoneme lose to BPE), MoE, RNN/linear attention, external-teacher RL. Small/inside noise: adaptive masking,
distillation (+1-3 BLiMP), contrastive synthetic data, MTP (hurts below ~1B), ELC layer mixing, FORGETTER resets.
Genuinely open (never measured on LMs): out-of-fold (cross-fitted) distillation; RHO-LOSS in LM pretraining;
nonce-word meta-learning's effect on LM loss; entity anonymisation; Muon on a fixed corpus over many epochs.
Packed ensembles / BatchEnsemble / MIMO (Laurent 2023, Wen 2020, Havasi 2021), IVON variational learning (Shen 2024),
TRIME in-batch memory (Zhong 2022), GECA compositional augmentation (Andreas 2020), SSMBA (Ng 2020), AND-mask (Parascandolo
2020), MC dropout, snapshot ensembles, looped/universal transformers: known ideas; none measured in BabyLM-style
exposure-capped pretraining as far as we know (verify before claiming).

## 5. Attacks already in flight (do not duplicate; you may build on them)

- A1 temporal self-distillation (targets from the model's own past epochs; PR #55). Prediction +0.02-0.08.
- A2 input-noise augmentation (token replacement/masking, R2L rows; PR #60). +0.03-0.12.
- A3 BPE-dropout for rare words (PR #61). -0.03 to +0.05.
- A4 model writes its own new contexts (closed-system re-expression: generator = the base, copy-blocked or cross-fitted
  half-data generators, student continues on real+synthetic within 100M total exposure). Ceiling ~0.5, +0.03-0.15.
- Experimenter: rare-word "imprinting" card in progress. Stacker thread merges A1-A4. Base builder: WD, tied embeddings,
  dropout, EMA, MNTP, epochs, size {S, M}.

## 6. Compute and the harness

- A100 size S: ~364k tokens/s; 10M x 10 epochs (~160M tokens) = ~8 min. 2-3 S runs can share one A100 (MPS).
  Size M ~3-4x slower. spot5/h. A full screen batch
- **Fast proxy ("mini-ruler")**: the base trained on nested 1M/2M/5M/10M-word subsets of train_10M at 10 epochs (S on
  A100: 1M x 10 = ~45 s). An attack trained on 1M words is falsified if it does not move up this mini-ruler. Proxy
  results are labels, never verdicts; mechanisms that rely on priors may look better at 1M than at 10M.
- Harness: `babylm/harness/train.py` (Muon + AdamW, word milestones, DEV-sub/TRAIN-sub curves, resume, export),
  levers in `babylm/harness/levers.py` (each off by default, off = bit-identical port path), model
  `sprint1/models/baseline.py` (GPT: RMSNorm, half-RoPE, QK-norm, ReLU^2 MLP, value embeddings, U-net skips, x0
  injection, logit softcap 15, untied embeddings unless --tie_embeddings; split API hidden()/logits()/in_embed()),
  `babylm/models/baseline_reg.py` (dropout), jobs via `babylm/harness/run_jobs.py` + JSON job lists.
  Tokenizer: byte-level BPE 16k trained on train_10M; 1.603 tokens/word.

## 7. What "20x-sized ceiling" means here

A mechanism whose theoretical best case (everything goes right) closes more than the whole 0.88 nats/word, i.e. one
that could in principle make 10M words worth 20x. Most known levers cap at 0.03-0.3. To reach 10x a portfolio needs
either one lever with a >1 nat ceiling or several large ones that stack multiplicatively on different bottleneck slices
(bias vs variance; frequent vs rare; seen vs new text). Be explicit about which slice each mechanism attacks, where its
ceiling comes from, and why it would not already be captured by a bigger, heavily regularised base.
