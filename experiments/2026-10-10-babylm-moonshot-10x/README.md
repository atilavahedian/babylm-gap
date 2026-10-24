# BabyLM 10x moonshot: is any mechanism legal under the rules worth 10x? (2026-10-10 to 2026-10-11)

## Question
Inside BabyLM strict-small (R0: 10M words, <= 10 epochs, <= 100M words of exposure counting every model), is there a
mechanism whose ceiling could reach 10x data efficiency, and if not, what is the honest legal ceiling?

## Setup
- Design portfolio: three rounds, 92 agents, adversarial skeptics; 18 surviving cards (`docs/cards/W*`), ranked in
  `docs/portfolio_v1.md`. Every portfolio number is a judgement, not a measurement.
- Ceiling map (`docs/ceiling_map.md`, critic ref-0126; diagnostics, out of track): C3 = up to 8 seed models averaged
  (posterior channel); C1 = KD from a 100M-trained teacher onto the 10M contexts (targets channel, f1).
- Cards run on GPU: PE (one-forward shared-trunk particle ensemble; Gate 2 on a 1M mini-ruler proxy), F1 (8-model
  ensemble distilled into one student; stage 1 on the proxy), F2 (heterogeneous K=2 split; Stage 0 from existing dumps).
- Legal-split family (registered 2026-10-10T07:19Z): K=2 (2 x 5 epochs), K=3 (3 x 3.33), trunk + 2 branches; and the
  R4.4 compute-matched menus F=2 (S) and F=4.146 (28L) used as references by the gap-map claims.
- Base, metric (cal_t, full / guarded DEV) and critic as in `../2026-10-09-babylm-gap-map/`.

## Result
- Portfolio: no surviving card has a 10x-sized ceiling (each 0.05-0.30 nats/word, expected 0.00-0.03); union of all
  survivors ~0.40-0.45 nats/word, roughly 3x, with P(all pass) ~1e-5.
- Ceiling map: C3 g(8) = 0.2893 / 0.2815 (g_inf 0.333 raw); C1 f1 = 0.697 raw / 0.704 overlap-removed (full), inside
  the registered [0.30, 0.75]. A perfect 100M teacher closes ~70% of the gap; both channels break R0.
- PE killed at Gate 2 (b): shared trunk keeps rho = 0.364 / 0.365 of independent models' gain (bar 0.5).
- F1 killed at stage 1: 8-teacher student +0.0502 / +0.0506 over the 1-teacher student vs bar 0.107 (2 floor, proxy).
- F2 killed at Stage 0: heterogeneous pairs add d <= +0.0068 cal_t g(2) (bar 0.03); lane B +0.003 to +0.005 (bar 0.016).
- Trunk + 2 branches -0.037 / -0.037 vs the plain K=2 split; K=3 reads void (scored mid-anneal), no valid number.
- The plain K=2 split survived and became the gap-map claim; F=2 menu: P-F2d HIT (+0.058 / +0.059, critic ref-0186);
  F=4.146 menu: best cell 36L/512d e4 (4.8728 / 4.8461), the 28L split beats it by +0.086 / +0.083 (critic ref-0182).
- P-F2e (size M vs the split) scored a MISS: M is 0.14-0.20 worse than the S single; mixture minus M = -0.330.

## Verdict
No 10x mechanism exists in the portfolio: every card's ceiling is 0.05-0.30 nats/word and the union is ~3x
(~0.40-0.45); PE, F1 and F2 were killed at their gates, and only the plain K=2 split (x >= 1.37) survived.

## Reproduce
    export PYTHONPATH=common:experiments/2026-10-10-babylm-moonshot-10x/code
    python -m babylm.moonshot.make_ceiling_jobs ...; make_f1_jobs; make_particle_jobs; make_split_jobs; make_menu_jobs  # write job lists
    python -m babylm.harness.run_jobs common/babylm/configs/<jobs_moonshot_*.json>   # copies in configs/babylm_configs/
    python -m babylm.moonshot.dev_bpw_dump <hf_export> <dev_dir> --out <run>/score/dev_pt.npz --twin --temps <T grid>
    python -m babylm.moonshot.ceiling_report c3 <dumps...> --guard --calib --out results/ceiling_v2/c3_gate0
    python -m babylm.moonshot.ceiling_report c1 --self D --oracle D --t100 D --out results/ceiling_v2/c1
    python -m babylm.moonshot.pe_gate2_read ...; python -m babylm.moonshot.f1_read ...   # gate reads
    bash code/babylm/moonshot/v2_readouts.sh; bash code/babylm/moonshot/hetero_pairs.sh  # executor read scripts (GCS paths)
    python -m pytest -q experiments/2026-10-10-babylm-moonshot-10x/code/babylm/tests   # needs torch
GPU job tickets as launched are in `configs/gcp_jobs/`. Dumps and checkpoints are on GCS (ARTIFACTS.md).

## Provenance
- Moonshot code/docs: `wip-branch` @ ref-0377 (newest per file; `ceiling_report.py` and
  `hetero_pairs.sh` from `wip-branch` @ ref-0414, which carries the D1 `bias` read ref-0415).
- Card F1: `wip-branch` @ ref-0416. Card F2 + design agent memo: `wip-branch` @ ref-0414.
- Reads: `wip-branch` (ceiling_v2 ref-0335, f2_stage0 ref-0329, f4_menu ref-0204, pe_gate2).
- Rulings: `governance/critic_log_moonshot.md` (excerpt of `wip-branch` @ ref-0010).
