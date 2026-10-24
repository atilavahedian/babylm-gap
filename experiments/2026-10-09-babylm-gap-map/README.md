# BabyLM data efficiency: the 10M-to-100M gap map (2026-10-09 to 2026-10-13)

## Question
Under BabyLM 2025 strict-small rules (10M words, at most 10 epochs, at most 100M words of total exposure), where does
the loss gap between a model trained on 10M words and the same recipe on 100M words live, and how much of it can a
legal recipe close? The mission bar was 10x data efficiency; 2x was the realistic target.

## Setup
- Base: 12L/384d causal Transformer (size S) from the sprint-1 modded-nanogpt port, tied output, no value embeddings,
  Muon weight decay 0.4, 6 epochs (E6). Second size: 28L/384d. Spot A100s on GCP, three at most.
- DEV metric: nats per whitespace word on full DEV (v2 blank-line twin scorer), calibrated (cal_t), full and guarded
  legs (guarded drops the 10,277 words with a character never seen in training). Reference: E6 pool, 5 seeds.
- Yardstick: x = 10^(gain / Y), Y = S E6 pool minus the best 100M point (port recipe E10, 3 seeds) = 0.715 / 0.690.
- Gap map (D1): 8 seed ensemble, C1-oracle student (100M teacher on 10M contexts) and the 100M model on the same DEV words.
- Official scores: pinned BabyLM upstream pipeline bf55c1131e53654c2a87418f5629c26959acd710, 3 runs per arm. An independent critic registered every bar first.

## Result
- Gap (raw, torch 2.9.1, D1 map) 0.743 nats/word = 0.289 single-model variance + <=0.233 learnable bias (upper bound)
  + 0.221 information absent from 10M words. A perfect legal learner reaches at most ~4x (0.45 of 0.743, log-linear).
- Best legal recipe: K=2 split (two models x 5 epochs on the same 10M words, probabilities averaged).
  S: +0.112 / +0.110 vs E6 pool, x >= 1.37 (point 1.44). 28L: +0.1145 / +0.1116, x >= 1.39 (point 1.45).
  At equal compute: +0.058 / +0.059 at S (x >= 1.10 as registered), +0.086 / +0.083 at 28L (x >= 1.22, point 1.32).
- Official BabyLM scores: ties. Zero-shot composite +1.39 (bar 1.84) at S, +1.05 (bar 2.04) at 28L; GLUE7 at S beats
  the S single (+1.29, bar 0.72) but ties the compute-matched 28L single (+0.63, bar 0.75); 28L GLUE7 -0.11 (bar 1.09).
- 2x not met at either size (highest interval upper end 1.52).

## Verdict
Best legal = K=2 split, x >= 1.37 on DEV loss (a known idea, replicated at two sizes), ties on official scores; gap =
0.289 variance + <=0.233 learnable bias + 0.221 absent information, so 10x is not in the data.

## Reproduce
Code is shared: `common/babylm` (harness, eval, tools, ledger) plus the moonshot readers in
`experiments/2026-10-10-babylm-moonshot-10x/code`. From the repo root:

    export PYTHONPATH=common:experiments/2026-10-10-babylm-moonshot-10x/code
    python -m babylm.data.prepare --download --splits train_10M,dev            # OSF zips; never committed
    python -m babylm.harness.run_jobs <job list>                               # GCP tickets: configs/gcp_jobs/*.json
    python -m babylm.moonshot.dev_bpw_dump <hf_export> <dev_dir> --out <run>/score/dev_pt.npz --twin --temps <T grid>
    python -m babylm.moonshot.ceiling_report bias --col member=mean:D0,..,D7 --col ens8=mix:D0,..,D7 \
        --col c1self=D --col c1oracle=D --col t100=D --guard --out results/d1_bias   # D1 gap map
    python -m babylm.eval.a5_yardstick_fixedT --config <a5_yardstick_final.json> --out a5_out.json   # Y and x
    python -m babylm.eval.official_readout <eval result dirs> --arm S_split=mixS_p01,mixS_p23,mixS_p45 \
        --arm L28_E6=L28_e6_s0,L28_e6_s1,L28_e6_s2 --beats S_split:L28_E6                 # official ties
The dumps and checkpoints these read live on GCS (see ARTIFACTS.md); torch is needed for every training/dump step.

## Provenance
- Experimenter report and paper: `wip-branch` @ ref-0008 (wip-branch).
- Executor reads/results: `wip-branch` @ ref-0009 (D1 map ref-0175, A5 yardstick ref-0176, 28L z3 ref-0009).
- Ruler readouts and plain-English report: `wip-branch` @ ref-0177 (report ref-0178).
- Critic log/rules: `wip-branch` @ ref-0010 (final rulings ref-0179, ref-0180, ref-0181, ref-0149, ref-0182).
- Prior art: `wip-branch` @ ref-0104. Facts/priors: `babylm-1` @ ref-0183. Base sweep: `wip-branch` @ ref-0184.
