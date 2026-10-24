# Attack A1: temporal self-distillation (card, 2026-10-09)

Code `babylm/harness/self_distill.py` (`train.py --sd_mode temporal|smooth`), tests `babylm/tests/test_self_distill.py`.
Status: built and CPU-tested; **no GPU run yet**, waits for the frozen base.

**Known, not new.** Temporal ensembling (Laine & Aila 2017: EMA of each example's past-epoch predictions as a target)
and PS-KD (Kim et al. 2021: the previous epoch's predictions as soft targets) are the mechanism; born-again networks
(Furlanello et al. 2018: a student distilled from an identical earlier generation) and Kim/Kotha 2025 (ensemble →
self-distillation; offline, on teacher-generated text, so extra exposure) are the same family. Tf-KD (Yuan et al.
CVPR 2020: self-KD gains are largely label-smoothing-like; hence the two nulls) and online codistillation (Anil et al.
ICLR 2018, LM) are the closest analyses. Prior-art check (2026-10-09): "known mechanism, untested regime
(within-run, LM pre-training, 10M x 10 epochs)"; PS-KD's only text task is MT, DynSDPB (2024) is fine-tuning only,
q0 hyper-epoch chain distillation (arXiv 2606.03938, ~4.6x on 100M FineWeb tokens) uses separate models. Closest BabyLM rows:
Baby Llama (Timiryasov & Tastet 2023, ensemble logit KD on 10M, +3.7 BLiMP, teachers cost extra training) and an
2024 born-again/self/reverse-distillation entry (Nair et al., CoNLL-BabyLM 2024: offline, "consistent, modest",
BLiMP +0.3-0.5) and BootBERT 2023 (EMA teacher, BLiMP -0.9 at 100M). What we test is only whether the zero-exposure, single-run version helps a tuned 10M base.

## Mechanism
Every training forward already computes the model's next-token distribution q at each corpus position p. We keep a
top-k target t_p per position (int16 ids + fp16 mass; k = 16 is 65 bytes per position, 1.04 GB for train_10M's
15.96M tokens), with no extra forward. Visit 1: t_p = top-k(q). Later visits: t_p = top-k over supp(t_p) ∪ top-k(q)
of d·t_p + (1−d)·q (q is exact for every candidate; the leftover 1 − Σt_p is kept as a tail bin). From epoch
`--sd_start_epoch` (default 1.0, i.e. epoch 2) on, loss = CE + α·KL(t_p ‖ q), the KL over k + 1 bins, using the target
from earlier visits only (read before write). Knobs: α (0.5), k (16), d (0.6), start epoch (1.0), and
`--sd_write_until E` (stop merging from epoch E, so late memorising epochs never enter the target). Works with the
stream and line samplers (keys are exact stream positions under the random epoch offsets) and with MNTP (causal rows
only). Known bias: caching only a top-k biases the teacher distribution and hurt calibration in Sparse Logit Sampling
(Anshumann et al., ACL 2025, top-k renormalised); A1's tail bin keeps the total mass exact but not the tail's shape, and
`sd_mass` logs the cached share. If A1 survives, an importance-sampled cache is the follow-up arm.
Training cost: one log-softmax, a top-k and a gather per step; FLOPs ≈ 1.0x (no matmuls added), wall-clock
logged.

## Why (bottleneck 1: generalisation)
72% of the GPT-BERT 10M→100M gap shows only on unseen text (0.6 of 0.84 nats/word), and our port recipe peaks on DEV
at epoch 2 (5.48 nats/word) then climbs to 9.98 while train falls to 1.83. After epoch 1 the one-hot label of a
repeated position is a memorisation target. The cached target replaces part of it with the model's own earlier belief
about that position, which (a) was formed when the model had seen that exact token fewer times (at visit 1, never),
so it encodes what generalises from other contexts, and (b) averages several points of the trajectory, a free
ensemble. Direct evidence (instruments pre-read, PR #52 @ ref-0082, dev sample): mixing GPT-BERT 10M's own epoch-5
predictions into its final model gains +0.048 [0.042, 0.057] nats/word (epoch 7: +0.040), so past-epoch predictions
carry information the final model has lost. Epochs 7-10 buy only 6% of its dev gain, dev-minus-train grows every epoch
(0.51 → 1.11; the 100M model sits at 0.51), and frequent words (8k+ occurrences, 46% of dev words) stall from epoch 7
while their train loss keeps falling: those contexts are what the soft targets must protect. Kim/Kotha 2025: the regularised recipe alone (heavy WD, tuned LR/epochs) is 2.29x; their offline 300M→300M
self-distillation reaches about −0.13 nats vs the regularised model (the ensemble figure 5.17x is an extrapolated
asymptote). A1 is the cheapest single-run, zero-exposure cousin. Honest prior: one trajectory's ensemble has far less diversity than independent models,
and the gain may overlap with tuned weight decay. Guess: +0.02 to +0.08 nats/word over the tuned base, P(passes
kill rule) ≈ 0.25. **Upper reference:** the inference-time probability mix of the run's own mid-epoch snapshot with
its final model (2x inference, zero extra exposure; +0.048 on GPT-BERT). The card reports what share of that mix's gain
one distilled model recovers; if it is a small share, we say so.

## Null twin
`--sd_mode smooth`: from the same start epoch, loss = (1+α)·CE with label smoothing ε. The temporal loss has gradient
(1+α)(q − m) with m = (onehot + α t)/(1+α); the twin has (1+α)(q − m0), m0 = label smoothing. ε is set so H(m0) equals
the temporal run's mean H(m) (logged as `sd_H`, tail spread uniformly):
`python -m babylm.harness.self_distill --null_eps_from <temporal run dir>`. Same base, LR, seed, steps. If smoothing
of the same strength does as well, the mechanism is just smoothing.
Second null (critic ref-0060): `--sd_mode smooth_unigram`, the same but smoothing toward the train_10M unigram
distribution, ε entropy-matched with the label drawn from the unigram (`... --null unigram`). The best arm must beat
the **best of four nulls** (critic ref-0083): {uniform, unigram} × {H-matched, off-mass-matched (`--match off`: same
mean mass moved off the label, logged as `sd_off`)}. At promotion the best screen null, picked on DEV before seeds
1-2, is the binding twin and gets 3 seeds.

## Kill rule (registered before any run)
Size S, train_10M × 10 epochs, the base's recipe and seed, base re-run in the same batch with `--sd_mode none` (same
seed, data order, GPU type). Metric: **full-DEV nats/word from `dev_bpw.py` on the final checkpoint** (DEV-sub is
diagnostics only). **Kill** if the best arm's gain over the base is below 2× the critic's seed noise floor, **or** it
does not beat the best of the four nulls by 2× the floor. No kill or promote until the floors are registered; until then
numbers only. Kill also on a ≥ 2σ drop on BLiMP-fast (human-likeness rule) or > 1.25x wall-clock without a compute-matched
reference. **Promote exactly one arm**, picked on DEV before seeds 1-2, to 3 seeds, size M, official eval (R0–R6 win
bar); the 3-seed verdict re-uses no screen numbers. **Best arm** (for the nulls and the promote) = lowest full-DEV
nats/word (dev_bpw.py, final checkpoint) among A1a-e; ties within 0.005 go to the smaller α, then the smaller d.
Compute: the SD forward (hidden + logits) is torch.compile'd under `--compile`; the log-softmax, KL, top-k and cache
merge run eager. Each verdict reports measured tokens_per_s vs the base arm for the 1.25x rule.

Provisional base (critic ref-0086): screens may run on the s1/reg winner, with the base arm, A1 arms and the null
twin at the same recipe, seed and data order at size S, all counted in the trial log. Since this is a regulariser,
any later base change to WD, EMA or MNTP forces one re-screen; until then a kill reads "killed (provisional base)" and
a promote stays provisional until the frozen-base verdict.

## Screen (small first; about 8 A100-minutes per S run, all 1 seed)
| arm | α | d | start | note |
|---|---|---|---|---|
| A1a | 0.5 | 0.6 | 1.0 | default (Laine & Aila's EMA weight; the epoch-10 target gives epoch 5 a weight of 5%) |
| A1b | 0.5 | 0.85 | 1.0 | long EMA (the epoch-10 target gives epochs 1-5 52% of the weight) |
| A1c | 1.0 | 0.85 | 1.0 | stronger pull |
| A1d | 0.5 | 0.0 | 1.0 | PS-KD: last visit only |
| A1e | 0.5 | 0.6 | 1.0 | `--sd_write_until 5`: target = EMA of epochs 1-5, frozen after |
| A1-nullH | best α | — | 1.0 | uniform smoothing, ε from the best arm's sd_H |
| A1-nullUH | best α | — | 1.0 | unigram smoothing, ε from sd_H |
| A1-nullO | best α | — | 1.0 | uniform smoothing, ε from sd_off |
| A1-nullUO | best α | — | 1.0 | unigram smoothing, ε from sd_off |
| base | — | — | — | `--sd_mode none`, same batch |

10 runs ≈ 95 A100-minutes. Floor: the v2b seeds (S, tuned recipe), registered by the critic; a read before that is
labelled "floor not registered". Diagnostics per arm: the upper-reference snapshot mix on the frozen base's per-epoch
checkpoints, and the instruments' lever signature (`python -m babylm.instruments.signature`, novelty tilt; extra
epochs score 0.43-0.66, 10x data ~1.04). FINAL untouched. Each run's DEV trial count is logged.

## Exposure and legality (R0)
No extra forward passes, no teacher trained elsewhere, no generated text: the teacher is the student's own forward on
the same window, so exposure = the base's exposure. No outside data or models.

## Verified on CPU (tests)
Off and `--sd_mode none` run the port's unchanged loop; temporal α = 0 and smooth α = 0, ε = 0 reproduce the port's
losses bit for bit; a tiny port run is bit-identical to the base commit ref-0087. Cache keys are exact under both
samplers' random offsets, visit counts match, the EMA merge equals the dense reference, the binned KL equals the
coarsened KL with matching gradients, pause + resume is bit-exact (cache in `sd_cache.pt`).

## CPU smoke (mechanics only, not evidence)
Tiny model (2L/64d) on the 20k-token synthetic test corpus (random words from a 24-word list, so there is nothing to
generalise and no overfitting: train−dev gap ≈ 0.06), 6 epochs, seed 0, full DEV nats/word: port 3.456, A1a temporal
3.481, uniform null (ε 0.146) 3.754, unigram null (ε 0.319) 3.976. It runs end to end; the target covers 99.9% of
positions once on, sd_t_label 0.34, sd_kl 0.19. This data has no generalisation gap to close, so the numbers say
nothing about the attack.
