# BabyLM 2025 winner reproduction: AMLM-Hard-Decay (2026-10-09 to 2026-10-10)

## Question
Can we retrain the 2025 strict-small NLP winner, AMLM-Hard-Decay (Edman & Fraser 2025; NLP 58.3, about 1 point
over GPT-BERT), on the official train_10M with our certified ruler, so that it can join the reference set?

## Setup
- Upstream code is github.com/Leukas/babylm25 @ ref-0405 (MIT), pinned in `code/babylm/winner/amlm/PIN.json`. The
  model is a 12-layer DeBERTa-v2 (hidden 384, 34.68M params) trained as a pure MLM with LAMB.
- Plan (critic ruling, 09 Oct):
  - `hard_decay_off`: 3 seeds on official train_10M, with a retrained sentencepiece tokenizer. These are the
    reference-set entry.
  - `hard_decay_pub`: one fidelity run on the authors' own data.
  - Score the released checkpoint as a pipeline check.
- Evaluation: the ruler's pipeline (`common/babylm/eval`, `mlm` backend). Only the MLM pseudo-loss is reported, as a
  label: an MLM cannot claim the causal loss leg.
- `code/babylm/winner/amlm/nli_overlap.py` measures exact-sentence overlap between the authors' data and GLUE MNLI/RTE.

## Result
- Only `hard_decay_off` seed 0 ran: 10 epochs, 99.
  - The final in-training eval was MLM loss 2.4272 at 59.30% mask accuracy.
  - The DEV MLM pseudo-loss was 3.79 nats/word (label only).
- The official evaluation of that run **failed**. Zero-shot hit an AutoProcessor error and GLUE hit a missing
  `wandb`. The eval-only rerun (`amlm_eval_ckpt_off_s0`), seeds 1-2, the fidelity run and the released-checkpoint
  score never ran. The ledger shows no admission for any of them.
- **Data finding.** The authors' training file `bb24.train` has 10.01M words. About 58% of it (the first 485k lines,
  about 5.85M words) is MNLI/SNLI triplets: premise, entailment, contradiction, so the line order encodes the label.
  It contains 99.3% of MNLI-train premises and 61.6% of full MNLI-train pairs verbatim. Overlap with MNLI
  validation/test and with RTE is about 0%. The authors' own ablation credits this data with MNLI 47.3 to 59.1, and
  their margin over GPT-BERT is all (Super)GLUE.

## Verdict
Incomplete: no official score was produced (1 of 3 seeds trained; eval failed and was never rerun). The ~58% MNLI/SNLI
composition was ruled out-of-track for *our* arena (critic 09 Oct 18:46Z: R0.4, so AMLM's 58.3 is not a bar). Whether
it is an evaluation leak under BabyLM's own rules is **flagged, not ruled on**.

## Reproduce
    # build a scratch tree: gpu_job.sh expects babylm/eval next to babylm/winner
    T=$(mktemp -d); cp -r common/babylm $T/; cp -r experiments/2026-10-09-babylm-winner-repro/code/babylm/winner $T/babylm/
    bash $T/babylm/winner/amlm/cpu_smoke.sh /tmp/amlm_smoke hard_decay            # CPU, 4 steps
    AMLM_CPU_TEST=1 PT_DEV_WORDS=1500 PT_TRAIN_WORDS=1500 OFFICIAL_ZIPS=<osf zips> \
      bash $T/babylm/winner/amlm/vm_job.sh hard_decay_off 0 /tmp/amlm_cpu             # CPU end-to-end (~6 min)
    bash $T/babylm/winner/amlm/gpu_job.sh train_hard_decay_off_seed0 out/train_hard_decay_off_seed0   # GPU (configs/gcp_jobs)
    python $T/babylm/winner/amlm/nli_overlap.py <repo>/data/bb24.train <glue parquet dir>   # the data finding

## Provenance
- Branch wip-branch @ ref-0406. Key commits: ref-0407 (pin + finding), ref-0408 (VM job),
  ref-0409 (seed-0 results), ref-0406 (eval fixes, never run).
- GCP jobs from wip-branch @ ref-0009. Critic: wip-branch @ ref-0010.
- Archive tags: `wip-branch`, `wip-branch`.
