# BabyLM 2025 winner reproduction (AMLM-Hard-Decay)

## Question
Does the 2025 strict-small NLP winner, AMLM-Hard-Decay (published NLP 58.3, macro 33.3; GPT-BERT mixed 57.3), beat the
GPT-BERT baselines when it is retrained on the official train_10M and scored on our certified pipeline? Its 3-seed
mean was to enter the reference set: the best of {GPT-BERT causal-focus, GPT-BERT mixed, AMLM on official data}.

## Method
- **Code.** The authors' repo @ ref-0405, kept pristine; wrappers are in `code/babylm/winner/amlm/`. The model is a
  12-layer DeBERTa-v2 (hidden 384, FFN 1280, 34.68M params), trained as a pure MLM. Optimiser and schedule follow the
  code, not the paper: LAMB betas (0.9, 0.98), lr 7e-3, cosine, z-loss 1e-4, bf16. Sequence length is 64, then 256
  from epoch 5. Hard-Decay masking uses mlm_prob 0.4 and mask_decay 0.25.
- **Arms (registered).**
  - `hard_decay_off`: official train_10M (9,953,773 words, 10 epochs = 99.5M exposure), sentencepiece retrained
    with the authors' spec (98.6% vocab overlap on their data), seeds 0-2.
  - `hard_decay_pub`: one fidelity run on `bb24.train`. It must land within 1.5 points of the published BLiMP,
    Supplement and EWoK. It is a label, never a reference.
  - Released `leukas/amlm_hard_decay` scored as a pipeline check.
- **Metrics.** The official pipeline (`mlm` backend) gives zero-shot and GLUE. The per-token MLM pseudo-loss on DEV
  (7 fixed mask partitions) is reported as a label only, since an MLM cannot claim the causal loss leg.
- **Data audit.** `nli_overlap.py` measures exact-sentence overlap of `bb24.train` with GLUE MNLI and RTE splits.

## Results
| item | value | bar / plan | status |
|---|---|---|---|
| `hard_decay_off` seed 0 training | 10,155 steps, 99,537,730 words, 3,144 s train on A100 | 3 seeds | done (1 of 3) |
| final in-training eval (MLM loss / mask acc) | 2.4272 / 59.30% | - | label |
| DEV MLM pseudo-loss, epoch 10 | 3.79 nats/word (2.2786 nats/token) | - | label, not causal |
| official zero-shot + GLUE for seed 0 | failed (AutoProcessor error; `ModuleNotFoundError: wandb`) | within max(2 SD, 1.5) of published | **no score** |
| eval-only rerun, seeds 1-2, `hard_decay_pub` fidelity, released-checkpoint score | not run (no ledger admission) | registered | **not run** |
| `bb24.train` share that is NLI triplets | first 485k lines, about 5.85M of 10.01M words (58%) | - | finding |
| MNLI-train overlap: premises / hypotheses / full pairs | 99.3% / 64.7% / 61.6% verbatim | - | finding |
| MNLI validation/test overlap, RTE overlap | about 0% of pairs (2.9% of matched premises); RTE 0% | - | no val/test leak found |

## What it rules out
- Using AMLM's published 58.3 as a bar in this programme. The critic ruled (2026-10-09T18:46Z) that training on
  LLM-synthesised NLI text overlapping the MNLI fine-tune set violates R0.4 in our arena, so it is reported only
  with that caveat.
- A verbatim MNLI validation/test or RTE leak in the authors' data, at the exact-sentence level.

## What it does not rule out / caveats
- It says nothing about AMLM's masking method on official data. No official score exists for any arm, so AMLM never
  entered the reference set, and GPT-BERT remained the reference.
- **Flagged, not ruled on:** whether the MNLI/SNLI-heavy training data is an evaluation leak under BabyLM's own
  rules. The authors' ablation (paper Table 5) credits this data with MNLI 47.3 to 59.1. Their margin over GPT-BERT
  is all (Super)GLUE. The line order encodes the entailment/contradiction label. Some hypotheses may be LLM-generated
  (SynCSE): this is inferred, not verified. The finding accuses a published winner. Escalating it is the owner's call.
- The seed-0 checkpoint exists (see ARTIFACTS), so the eval-only job could still produce a score.
- Exposure: `bb24.train` is 10,009,099 words, so 10 epochs are 0.09% over the strict-small cap. This applies to the
  published-data label arm only.

## References
- [docs/winning_recipes.md](docs/winning_recipes.md): the runner's plan, code facts and finding (verbatim).
- [governance/critic_rulings.md](governance/critic_rulings.md): critic entries 18:33Z, 18:43Z, 18:46Z and the GLUE audit (verbatim).
- [code/babylm/winner/amlm/PIN.json](code/babylm/winner/amlm/PIN.json): code, data, tokenizer and checkpoint pins.
- [results/train_hard_decay_off_seed0/](results/train_hard_decay_off_seed0/): run.json, curve, per-token index and eval logs.
- Ruler certification: [../2026-10-09-babylm-official-ruler/SUMMARY.md](../2026-10-09-babylm-official-ruler/SUMMARY.md).
