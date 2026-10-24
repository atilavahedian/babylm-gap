# BabyLM 10M: the bottlenecks, ranked (v1)

2026-10-09 ~23:55Z, research experimenter. Critic audit ref-0425: no veto, labels below applied.

**Where the sizes come from.** They're measured on the published GPT-BERT 10M vs 100M checkpoints (causal; instruments pre-read, PR #52,
`results/babylm/preread_v1/prize_anatomy.csv` and `prize_bands.csv`). With overlap removed and reweighted to dev sources, the gap is
**0.875 nats/word [0.82, 0.93]**. The same shape shows up on GPT-2 (1.13) and on our own port pair at its best epoch (1.01: content
words 75% of the gap, words seen 64+ times 69%). These are published-checkpoint sizes, not our base. v2 at our base will confirm or
revise them.

| # | Bottleneck | Size, nats/word (share) | Cause (hypothesis) | Attacks, all known ideas | Chance (my guess) |
|---|---|---|---|---|---|
| 1 | Known words in new contexts | ~0.40 (~45%) | The model memorises the contexts it has seen instead of learning word associations that transfer | A1 temporal self-distillation (Laine & Aila 2017, PS-KD); A2 input-noise augmentation (arXiv 2606.16246; Xu+ 2026); regularised scaling in the base (Kim/Kotha 2025) | 0.35 |
| 2 | Rare and never-seen words | 0.28 (32%): never seen 0.11, seen 1-63 times 0.18 | Too few examples to learn their embeddings and subword composition | A3 stochastic segmentation (BPE-dropout, Provilkov 2020); tied embeddings (base) | 0.15 |
| 3 | Inference-time headroom (overlaps 1, not additive) | Legal at zero exposure: own epoch-5 snapshot mix 0.048, copy cache 0.037, temperature 0.013. Count model 0.051 costs one epoch of exposure. The 0.077 ensemble uses two 10-epoch models, which is 2x exposure and out of track; V8 is the legal test | Variance within one run; unused document history | A1 (distils snapshots into one model); V8 | 0.4 |

Ranked by prize x chance: 1 (0.14), 2 (0.04), 3 (~0.03). "Chance" is my guess of P(an attack captures a quarter of that prize, beyond
2x noise).

**Evidence behind the ranking**
- Seen vs new text: on text both models trained on, the 10M model is only 0.24 behind. The other 0.61 shows up only on new text.
- Content words are 44% of words and carry 73% of the gap.
- Late epochs memorise: epochs 7-10 buy 6% of the dev gain. Extra epochs have a novelty tilt of 0.43, against 1.04 for 10x data.
- On the rare-word size, the 21:30Z "~0.16" was the 1-63 band alone on the raw dev sample (0.162). The 0.28 here adds never-seen
  words (0.108 + 0.175, overlap removed and reweighted).

**What this means for the goal**
Capturing a quarter of a bottleneck is worth 0.03-0.10 nats/word. Twice the data is worth about 0.26 nats/word: that's an estimate
from log-linear interpolation between 10M and 100M, and v2's 30M cell measures it. So one attack hitting its chance is not a 2x
win. Getting to 2x needs several levers to stack, or one bigger than anything published. The only published lever with 2x-class
evidence is Kim/Kotha regularised scaling (2.29x at 200M tokens), and it now runs as a priority base stage.

**Next**
- Base round 1 runs on 2 A100s after the WD sweep. It has 11 cells: value-embedding tables off, tied embeddings, embedding decay,
  dropout, MNTP, 4 epochs, and all of them together. The sweep showed why the base overfits: weight decay never touched the 60% of
  parameters in the embedding tables.
- After that: the A1 screen and v2 at the round-1 base, then M/L sizes with re-tuned regularisation.
- Cards for A2 and A3 go to the critic.
