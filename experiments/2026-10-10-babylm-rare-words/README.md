# A5: rare and never-seen words, and a late-epoch rare-word focus lever (2026-10-10)

## Question
How much of the 10M-to-100M loss gap sits on rare and never-seen words ("bottleneck 2"), is any of it reachable from
10M words alone, and does focusing the late-epoch loss on rare words (and the 3 words after them) recover some of it?

## Setup
- CPU anatomy (no GPU, no compute): the published GPT-BERT causal-focus 10M and 100M baselines scored on a seeded DEV
  sample (287 pieces, 215,312 words; 187,326 after dropping words on lines found verbatim in train_100M), split by
  train_10M frequency band, never-seen word type, first use vs repeat, context novelty; four cheap oracles
  (calibration offsets, lexicon trie, matched counts, window reweighting).
- Lever (card arm 1): `--focus_plan 7:0.25`: from epoch 7, non-focus targets stay in the loss with probability 0.25;
  focus = words seen <= 63 times in train_10M plus the 3 words after them; exposure and FLOPs unchanged. Null twin N1
  drops the same share of targets at random. Kill rule registered before any run (critic ref-0427, 2026-10-10T02:20Z).
- Planned run: arm + N1 as a two-run labelled diagnostic on the round-1 base (experimenter 03:01Z), GCP ticket `a5_diag`
  pinned at ref-0428; critic cleared it to launch (ref-0429 at 03:36Z; card commit ref-0430 at 06:04Z).

## Result
- Anatomy (GPT-BERT, filtered sample): gap 1.003 nats/word (10M 5.141 vs 100M 4.138). Bottleneck 2 is ~0.34 on this
  sample (0.28 on the instruments' protocol), of which ~0.045 is CHILDES markup absent from train_10M; ~40% of the
  rest is general competence; the occurrence-driven remainder is ~0.10-0.12. Oracles: calibration headroom 0.0012,
  lexicon trie +0.0095; accented characters already single tokens (arm 2 dropped).
- New-base premise check: frequent words still improve in epochs 7-10 (43% of the late gain), so the card was
  re-priced before any run: prior -0.01 to +0.04 (centre +0.01), P(pass) ~0.3 (critic ref-0267 binds it).
- The lever is implemented and tested on CPU (7 focus tests pass; the existing suite 43 passed, 3 CUDA-only skipped; recorded at ref-0431) and is off by default.
- **No A5 run is on record.** The executor ledger has no `a5_diag` admit or close, no A5 result file exists in any
  branch, and no critic verdict on A5 exists (every later "A5" row is the unrelated A5 yardstick).

## Verdict
Incomplete: lever built and off by default, diagnostic cleared but never run; no verdict on record (registered prior
+0.01 nats/word, P(pass) ~0.3).

## Reproduce
    export PYTHONPATH=common      # from the repo root; analysis code and the focus mask live in common/babylm
    python common/babylm/analysis/rare_words/score_rare.py <osf_raw_root> <hf_ckpt_dir> <out_dir>   # then overlap_flag,
        # analyze_rare, calib_pass, analyze_calib, trie_oracle, context_novelty, matched_count, window_density
        # (order and inputs in results/rare_words_v1/README.md)
    python -m babylm.data.focus_mask --split train_10M                          # build the focus mask
    # the --focus_plan wiring in babylm/harness/train.py is NOT in common/: apply code/patches/focus_plan_train.patch
    # (diff vs ref-0426; does not apply cleanly to common's newer train.py) or use the archived branch tree
    python -m babylm.harness.run_jobs experiments/2026-10-10-babylm-rare-words/configs/babylm_configs/jobs_a5_S.json  # ticket: configs/gcp_jobs/a5_diag.json
    python common/babylm/analysis/rare_words/band_split.py --base <base run> --runs <arm> <N1> --raw_dir <dev raw> --out <dir>
    python -m pytest -q experiments/2026-10-10-babylm-rare-words/code/babylm/tests/test_focus.py  # needs torch + the patch

## Provenance
- Branch `wip-branch` @ ref-0430 (card), results ref-0432, lever ref-0433, tests ref-0431;
  archive tag wip-branch. Ticket: `wip-branch` ref-0434 / ref-0435.
- Rulings: `governance/critic_log_a5_rare_words.md` (excerpt of `wip-branch` @ ref-0010).
