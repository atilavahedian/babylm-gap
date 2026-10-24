# A5: rare-word gap anatomy and late-epoch rare-word focus lever

**Status:** incomplete · **Dates:** 2026-10-10 (card, anatomy, lever, registration; the GPU diagnostic was never launched)

## Question
How much of the BabyLM 10M-to-100M loss gap sits on rare and never-seen words ("bottleneck 2")? Is any of it reachable
from 10M words alone? Does using compute the late epochs' loss on rare words and the 3 words after them recover some of it,
at unchanged exposure and FLOPs?

## Method
- **Anatomy (CPU, no compute).** The published GPT-BERT causal-focus 10M and 100M baselines were scored on a seeded DEV
  sample: 287 pieces, 215,312 words, or 187,326 after dropping words on lines found verbatim in train_100M. The sample
  was split by train_10M frequency band (0 / 1-63 / 64-8k / 8k+), never-seen word type, first use vs repeat, and
  context novelty. Four cheap oracles were run: calibration offsets, a lexicon trie, matched counts, and window
  reweighting.
- **Lever (card arm 1).** `--focus_plan 7:0.25`: from epoch 7 on, non-focus targets stay in the loss with probability
  0.25. The focus set is the words seen at most 63 times in train_10M plus the 3 words after each one (27.4% of DEV
  words). Exposure and FLOPs are unchanged.
  - Null twin N1 drops the same share of targets at random.
  - N2 (rare tokens only) was skipped.
  - Arm 2 (tokenizer character coverage) was dropped, because accented Latin letters are already single tokens.
- **Planned diagnostic.** The arm and N1 at size S, seed 0, on the ve0/tied/Muon-WD-0.4 base. Two runs, alone and
  sequential on one A100, labelled "diagnostic", which means they cannot be promoted.
  - GCP ticket `a5_diag`, pinned at ref-0428, about 0.5 A100-hours per cell.
  - Critic clearance: ref-0429 (03:36Z), then the 06:04Z ruling.
- **Registered bars (critic ref-0019 / ref-0001, before any run).**
  - Primary metric: score.json `dev_nats_per_word_twin`. The base at seed 0 is 5.0023.
  - Screen pass: at least 0.010 over both the same-seed base and N1. 0.0074 to 0.010 is inconclusive; below 0.0074 kills.
  - Consistency gate: 0.0036 with sign agreement.
  - 8k+ guard: no worse than 0.0051 on the harness dumps.
  - Never-seen-character guard leg.
  - A BLiMP-fast drop above 2.0 points kills.
  - Mechanism check (reported, not a kill): at least 60% of the gain lands in the focus bands.

## Results

The anatomy is complete. The lever experiment was never run.

| Quantity | Value | Source |
|---|---|---|
| Gap, GPT-BERT 10M vs 100M (filtered sample) | 1.003 nats/word (5.141 vs 4.138) | `results/rare_words_v1/decomp_filtered.md` |
| Bottleneck 2 (rare + never-seen) | ~0.34 on this sample; 0.28 on the instruments' protocol | doc, headline |
| of which CHILDES markup absent from train_10M | ~0.045 | doc |
| general-competence share of the rest | ~40% | doc |
| occurrence-driven remainder | ~0.10-0.12 | doc |
| Oracle: calibration headroom | 0.0012 | `calib_summary_gptbert_10m.json` |
| Oracle: lexicon trie | +0.0095 | `trie_oracle_gptbert_10m.json` |
| Own base vs 100M twin (DEV-sub): gap / bottleneck 2 / focus-set share | 0.701 / 0.235 / 0.351 | doc, "New base" |
| Base band split, full DEV, harness dumps (all; 1-63; 8k+) | 5.0406; 11.48; 3.04 | `a5_bands_fulldev/bands_base.md`, critic ref-0266 |
| Registered prior (binding, critic ref-0267) | -0.01 to +0.04 nats/word, centre +0.01, P(pass) ~0.3 | doc |
| **Arm vs base / arm vs N1** | **not run** | no ledger entry, result file or verdict |

## What it rules out
- **Representation fixes at readout.** A clever output layer, recalibration, or a spelling prior cannot recover a
  meaningful share of the rare-word gap. Calibration headroom is 0.0012 and the trie adds +0.0095.
- **The 0.15 target.** A target of at least 0.15 nats/word from bottleneck 2 alone is out of reach. The
  occurrence-driven part is only about 0.10-0.12.

## What it does not rule out / caveats
- **The lever itself is untested.** Neither arm 1 nor N1 was ever trained.
  - The executor ledger has no `a5_diag` admit or close.
  - No branch holds a file under `attacks/A5/`.
  - No critic verdict on A5 exists. The later "A5" rows in the critic log are the unrelated A5 yardstick.
- **Anneal confound.** Epochs 7-10 sit in the LR cooldown, and frequent words still carry 43% of the late gain on the
  new base. The instruments therefore read the late rare-word gain as mostly the anneal. A close twin comparison was
  expected.
- **Anatomy model.** The anatomy uses the public GPT-BERT checkpoints, not the lab's base. The own-base ceiling
  numbers are DEV-sub diagnostics, not ranking sizes.
- **Redaction.** For licensing reasons, the per-word lists of top rare words were removed from `decomp*.md` and
  `decomp_filtered.json`, and the 560 dev-split example words were removed from `rare_words_fulldev_anatomy.json`
  (all are BabyLM corpus words). The originals stay in the archive tag (ARTIFACTS.md).
- **Missing harness wiring.** The `--focus_plan` wiring in `babylm/harness/train.py` is not in `common/`. It ships as
  `code/patches/focus_plan_train.patch`, which does not apply cleanly to common's newer train.py.

## References
- Card, anatomy, prior and kill rule (verbatim): [docs/A5_rare_word_focus.md](docs/A5_rare_word_focus.md)
- Critic rulings: [governance/critic_log_a5_rare_words.md](governance/critic_log_a5_rare_words.md)
- Anatomy outputs: [results/rare_words_v1/README.md](results/rare_words_v1/README.md)
- Ticket: [configs/gcp_jobs/a5_diag.json](configs/gcp_jobs/a5_diag.json), jobs: [configs/babylm_configs/jobs_a5_S.json](configs/babylm_configs/jobs_a5_S.json)
- Parent program: [../2026-10-09-babylm-gap-map/SUMMARY.md](../2026-10-09-babylm-gap-map/SUMMARY.md)
