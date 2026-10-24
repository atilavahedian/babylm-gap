# BabyLM official ruler: pipeline certification and bottleneck instrumentsjsonl`, jobs smoke/certify*/ftverify/gptbert_causal_seed0).

## Question
Can our install of the official BabyLM 2025 evaluation reproduce published scores exactly? If so, it is the one ruler
on which every BabyLM attack is judged. A second question: which instruments locate the 10M-to-100M gap, and so tell
an attack where to aim?

## Method
- **Pipeline.** Official `evaluation-pipeline-2025` @ `bf55c1131e53654c2a87418f5629c26959acd710`, transformers 4.51.3, torch 2.7.0, A100 40GB.
  Data comes from OSF `ryjfm`, with every file sha256-checked (`common/babylm/data/osf_listing.txt`). Full EWoK is
  read from a private bucket and pinned by sha.
- **Certification (critic rule R1).** The published 10M checkpoints are scored with our install:
  - GPT-BERT causal-focus, mixed and masked;
  - GPT-2;
  - an untrained control.
- **Gates.** Zero-shot must land within 0.5 points of published. GLUE counts only when all 7 tasks finish and their
  results.txt files exist (critic audit 22:51Z).
- **Concurrency check.** Sequential and concurrent GLUE are checked for byte identity (ftverify).
- **Causal GLUE pooling.** Official BOS pooling was tested. Then a last-token adapter was certified on base_S_e6
  against an untrained twin (critic ref-0214).
- **Reading.** The tokenizer alignment was checked (critic 81702719).
- **Instruments.** Frozen DEV/train samples: 419 DEV chunks, 138,745 words. Per-word causal loss on these samples
  feeds the slices, signatures, prize, stall, n-gram, dispersion and headcheck probes. The hypotheses (H1-H7) were
  registered before correct-format scores (`docs/preread_hypotheses.md`).
- **Cut. Published
  checkpoints stand in as reference rows.

## Results
| check | ours | published / bar | verdict |
|---|---|---|---|
| GPT-BERT causal-focus BLiMP (zero-shot, causal backend) | 71.66 | 71.66 | exact |
| GPT-BERT causal-focus Supplement | 63.21 | 63.21 | exact |
| GPT-BERT causal-focus EWoK (full, sha-checked) | 49.49 | 49.49 | exact |
| GPT-BERT causal-focus Entity tracking | 34.55 (OSF v2) / 33.96 (OSF v1 files) | 33.96 | certified on v1 (critic 10-10 22:50Z); the gap is a data version |
| GPT-BERT GLUE, 7 tasks x {causal-focus, mixed} | 12/14 cells exact to 2 dp | published README | MultiRC +0.24 / +0.04 only |
| GPT-BERT mixed QQP | F1 72.24, acc 76.56 | 76.56 | published cell is accuracy |
| base_S_e6 GLUE, official BOS pooling | mcc 0 on every binary task, MNLI .357 | - | not a score (critic, 11 Oct 01:51Z) |
| base_S_e6 GLUE7, last-token pooling | 67.24 (7/7 off majority, mcc > 0.18) | registered trained-half criterion | certified (ref-0214) |
| untrained base_S, last-token pooling | 62.6 mean of 6 tasks | "untrained ≈ majority" | half failed; reference row only |
| concurrent vs sequential GLUE (base_S_e6) | 6/7 bit-identical, MultiRC mismatch | identity | all later GLUE sequential |
| Reading_rt / Reading_spr, our models | 0.00-0.29 | GPT-BERT 9.5-9.9 / 3.3-3.5 | token-form artifact; amended column (add_prefix_space) |

Final-table outcome measured on this ruler (RULER_REPORT, official scores, n=3 vs 3, bar = 2 x combined SE):

| comparison | zero-shot | GLUE7 | compute |
|---|---|---|---|
| S split vs S single | +1.39 vs bar 1.84: tie | +1.29 vs bar 0.72: beats | ~1.67x, not matched |
| S split vs 28L single | n/a | +0.63 vs bar 0.75: tie | ~matched (critic ref-0220) |
| 28L split vs 28L single | +1.05 vs bar 2.04: tie | -0.11 vs bar 1.09: tie | ~1.67x, not matched |

The data-equivalent multiplier is a lower 95% bound (full / guarded): A1 S same size 1.37 / 1.39, A2 S FLOPs-matched
1.10 / 1.15, A3 28L same size 1.39 / 1.41, A4 28L FLOPs-matched 1.22 / 1.25. These are upper bounds, because the
100M reference was not exhaustively tuned. 2x is not met.

Instrument findings (`docs/instruments.md`, `results/instruments/`):
- Text format must match training. Wrong line joins or wrong tab handling cost 1-4 nats/word on the affected text.
- Extra epochs score 0.43-0.76 on the novelty signature. 10x data scores 1.04.
- The port's final model puts 39% of targets seen 1-15 times below a logit of -14, and is overconfident by 4.6
  nats/token (headcheck).

## What it rules out
- That our pipeline install explains a gap to published numbers. Zero-shot reproduces exactly and GLUE is within
  ≤0.24 on one task.
- Citing BOS-pooled GLUE for any causal model. It is degenerate (majority class) for ours and for published GPT-2.
- Citing any GLUE number made before the 7/7 sequential rule. The audit of 10 Oct found none complete.
- An official-score win for the K=2 split at matched compute. Every compute-matched comparison ties.

## What it does not rule out / caveats
- The reproduction of published GPT-BERT is a certification of released checkpoints, not a 3-seed retrain. The
  brief's "reproduce baseline and winner, 3 seeds each" was met only as certification. GPT-2's last-token GLUE was
  cut at 2/7.
- The GPT-BERT GLUE certification uses whole-input (bidirectional) BOS pooling with one fine-tune seed. Our causal
  GLUE7 uses last-token pooling and a 2-seed ensemble, so the two GLUE scales are not comparable.
- Our absolute composites are about 1.6 points low because of the Reading token-form issue. They are comparable
  between our own arms, not to the leaderboard. The amended column is "amended after data".
- The CPU zero-shot runs in `results/zs_cpu/` are provisional (critic ref-0350: CPU counts only where exact vs A100).
- The instruments are document-cluster bootstraps on a 139k-word DEV sample, not the official held-out number.

## References
- [docs/RULER_REPORT.md](docs/RULER_REPORT.md): the ruler's final report (verbatim).
- [results/certify_v3/glue_certification_1011.md](results/certify_v3/glue_certification_1011.md), [results/certify_v3/glue_audit_1010.md](results/certify_v3/glue_audit_1010.md), [results/certify_v3/partial_a/README.md](results/certify_v3/partial_a/README.md), [results/certify_v3/entity_diag/README.md](results/certify_v3/entity_diag/README.md)
- [results/final_table_lines/glue_lastpool_certification.md](results/final_table_lines/glue_lastpool_certification.md), [results/ftverify/ftverify_base_S_e6_lastpool/README.md](results/ftverify/ftverify_base_S_e6_lastpool/README.md)
- [results/readouts/pub_gptbert_reference.md](results/readouts/pub_gptbert_reference.md), [results/readouts/reading_tokenizer_check.md](results/readouts/reading_tokenizer_check.md), and the other readouts.
- [docs/instruments.md](docs/instruments.md), [docs/preread_hypotheses.md](docs/preread_hypotheses.md), [docs/preread_bottlenecks.md](docs/preread_bottlenecks.md), [docs/facts_2025.md](docs/facts_2025.md), [docs/ruler_branch_README.md](docs/ruler_branch_README.md)
- [governance/critic_rules.md](governance/critic_rules.md) (R0-R6) and [governance/critic_log.md](governance/critic_log.md). These cover every BabyLM experiment.
