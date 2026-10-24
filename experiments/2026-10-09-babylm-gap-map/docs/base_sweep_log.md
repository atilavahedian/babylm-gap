# Attack-base sweep log (base builder thread)

Every tuning trial of the attack base, in launch order, with its score. Selection = DEV nats/word (babylm/eval/dev_bpw.py,
ctx 512 / stride 256, full dev) plus the official fast zero-shot suite; seed 0 unless stated. The trial count goes to
the critic at freeze time (critic 19:11Z: base frozen by commit hash before any attack screen).

## s1: size S, muon_wd x lr_scale (pre-normalisation pipeline, ref-0221)

Pipeline label: **pre-normalisation pipeline (ref-0221)**: raw in-line whitespace, tokenizer sha ref-0222. Code
ref-0157. Scored in its own training format (dev_bpw from ruler ref-0223: raw whitespace, newline line join). Rows
wd {0, 0.1, 0.4, 1.0} at lr_scale 1 score the experimenter's registered prediction P4. Launched 2026-10-09 ~20:03Z, A100
spot VM bl-base-s1-1009-200254 (executor 3). Per the experimenter's 20:12Z decision, stage 2 re-runs the s1 winner once on the
final pipeline before anything else, so the frozen base never mixes pipelines.

Target (ruler certify, babylm/results/certify/score_pub_gptbert_causal/ on ij746t): published GPT-BERT 10M (causal)
scores 4.939 nats/word on dev_bpw full dev in the FINAL format (ws collapse). The s1 rows use the pre-normalisation
format, so compare them with it only roughly. The final-pipeline re-run of the s1 winner is the first like-for-like number.

Columns: DEV = dev_bpw full dev (selection); harness = trainer's own full-DEV pass; min = lowest DEV-sub over milestones.

| trial | muon_wd | lr_scale | DEV nats/word | harness full DEV | DEV-sub min (at) | final train | fast suite | train s | note |
|---|---|---|---|---|---|---|---|---|---|
| base_S_wd0_lr1 | 0 | 1 | 9.9329 | 9.6996 | 5.4761 (e2, 20M) | 1.83 | BLiMP 59.44 / suppl 66.0 / EWoK 52.64 / ent 32.28 | 464 | port control; DEV rises every epoch after 20M words |
| base_S_wd0p1_lr1 | 0.1 | 1 | 7.1331 | 7.0594 | 5.5772 (e2, 20M) | 2.75 | BLiMP 63.16 / suppl 62.8 / EWoK 51.36 / ent 15.94 | 465 | P4; still overfits after epoch 2 |
| base_S_wd0p4_lr1 | 0.4 | 1 | 6.5483 | 6.5127 | 5.8482 (e2, 20M) | 3.37 | BLiMP 60.84 / suppl 59.6 / EWoK 53.64 / ent 19.57 | 463 | P4; rises after epoch 2 |
| base_S_wd1p0_lr1 | 1.0 | 1 | 6.3697 | 6.3444 | 6.1346 (e2, 20M) | 3.69 | BLiMP 59.86 / suppl 56.0 / EWoK 51.09 / ent 27.53 | 463 | P4; rises after epoch 2 |
| base_S_wd2p0_lr1 | 2.0 | 1 | 6.3994 | 6.3877 | 6.3377 (e3, 30M) | 3.92 | BLiMP 56.31 / suppl 56.4 / EWoK 52.09 / ent 18.99 | 465 | rises after epoch 3 |
| base_S_wd0_lr0p5 | 0 | 0.5 | 8.4007 | 8.2607 | 5.4363 (e2, 20M) | 1.70 | BLiMP 62.37 / suppl 57.6 / EWoK 52.64 / ent 14.39 | 462 | rises after epoch 2 |
| base_S_wd0p1_lr0p5 | 0.1 | 0.5 | 7.0403 | 6.9687 | 5.4686 (e2, 20M) | 2.36 | BLiMP 61.84 / suppl 58.0 / EWoK 51.64 / ent 21.8 | 466 | rises after epoch 2 |
| base_S_wd0p4_lr0p5 | 0.4 | 0.5 | 6.3551 | 6.3325 | 5.6081 (e2, 20M) | 3.08 | BLiMP 61.12 / suppl 58.8 / EWoK 51.82 / ent 20.6 | 468 | rises after epoch 2 |
| base_S_wd1p0_lr0p5 | 1.0 | 0.5 | 6.2468 | 6.2252 | 5.7564 (e2, 20M) | 3.43 | BLiMP 59.98 / suppl 56.0 / EWoK 51.0 / ent 16.29 | 464 | rises after epoch 2 |
| base_S_wd2p0_lr0p5 | 2.0 | 0.5 | 6.1619 | 6.1462 | 5.9272 (e2, 20M) | 3.67 | BLiMP 59.73 / suppl 56.0 / EWoK 51.18 / ent 28.02 | 468 | **s1 winner** (lowest final DEV); rises after epoch 2 |

## Round 1 (tokenizer ref-0224; compare to the control, not to s1)

| trial | DEV nats/word | harness full DEV | DEV-sub min (at) | final − min | final train | fast suite |
|---|---|---|---|---|---|---|
| base_S_wd0p4_lr0p5_r1_ve0_tied_s1 | 5.0255 | 5.0371 | 5.1171 (e10, 100M) | 0.000 | 3.84 | BLiMP 68.52 / suppl 61.6 / EWoK 49.45 / ent 23.01 |
| base_S_wd0p4_lr0p5_r1_ve0_tied_s2 | 5.0283 | 5.0384 | 5.1257 (e10, 100M) | 0.000 | 3.84 | BLiMP 67.23 / suppl 63.6 / EWoK 51.55 / ent 22.96 |
| base_S_wd0p4_lr0p5_r1_ve0_tied | 5.0367 | 5.0406 | 5.1210 (e10, 100M) | 0.000 | 3.83 | BLiMP 68.37 / suppl 66.0 / EWoK 50.45 / ent 23.21 |
| base_S_wd0p4_lr0p5_r1_all | 5.1023 | 5.1225 | 5.2037 (e10, 100M) | 0.000 | 4.41 | BLiMP 64.78 / suppl 60.0 / EWoK 51.0 / ent 19.52 |
| base_S_wd1p0_lr0p5_r1_ve0_tied | 5.1232 | 5.1403 | 5.2225 (e10, 100M) | 0.000 | 4.19 | BLiMP 68.03 / suppl 62.0 / EWoK 49.0 / ent 19.07 |
| base_S_wd0p4_lr1_r1_ve0_tied | 5.1278 | 5.1416 | 5.2224 (e10, 100M) | 0.000 | 4.14 | BLiMP 66.61 / suppl 62.4 / EWoK 50.09 / ent 22.95 |
| base_S_wd0_lr0p5_r1_ep2 | 5.1616 | 5.1628 | 5.2320 (e2, 20M) | 0.000 | 4.00 | BLiMP 64.15 / suppl 58.4 / EWoK 50.45 / ent 20.52 |
| base_S_wd0_lr0p5_r1_ep3 | 5.2453 | 5.2423 | 5.3178 (e3, 30M) | 0.000 | 3.54 | BLiMP 63.76 / suppl 62.4 / EWoK 50.55 / ent 15.94 |
| base_S_wd2p0_lr0p5_r1_ve0_tied | 5.3031 | 5.3177 | 5.4051 (e10, 100M) | 0.000 | 4.46 | BLiMP 63.14 / suppl 60.4 / EWoK 50.91 / ent 21.79 |
| base_S_wd2p0_lr0p5_r1_ve0 | 5.5662 | 5.5660 | 5.6530 (e10, 100M) | 0.000 | 4.26 | BLiMP 62.33 / suppl 58.4 / EWoK 52.45 / ent 27.14 |
| base_S_wd2p0_lr0p5_r1_do0p1 | 5.8304 | 5.8225 | 5.8843 (e2, 20M) | 0.044 | 3.77 | BLiMP 61.97 / suppl 56.8 / EWoK 52.27 / ent 22.27 |
| base_S_wd2p0_lr0p5_r1_tied | 5.8966 | 5.8806 | 5.9130 (e2, 20M) | 0.084 | 3.80 | BLiMP 58.97 / suppl 62.4 / EWoK 51.45 / ent 35.74 |
| base_S_wd2p0_lr0p5_r1_ewd0p005 | 5.9729 | 5.9518 | 5.9452 (e4, 40M) | 0.122 | 3.60 | BLiMP 61.78 / suppl 58.8 / EWoK 51.64 / ent 28.4 |
| base_S_wd2p0_lr0p5_final | 6.1212 | 6.0914 | 5.9128 (e2, 20M) | 0.298 | 3.62 | BLiMP 61.4 / suppl 54.4 / EWoK 52.18 / ent 37.18 |
| base_S_wd0_lr0p5_r1_ve0_tied | 6.2471 | 6.2127 | 5.3038 (e3, 30M) | 1.037 | 2.66 | BLiMP 65.07 / suppl 60.8 / EWoK 48.64 / ent 17.01 |

## Round 1 winner (experimenter, 02:50Z)

**`jobs_base_r1_S.json:base_S_wd0p4_lr0p5_r1_ve0_tied` @ ref-0065**: full DEV 5.0367 (harness 5.0406), with its DEV-sub minimum at the final point (e10). Config: no value embeddings, tied input/output table, Muon WD 0.4, lr_scale 0.5, 10 epochs, everything else at the port recipe.
- Muon WD curve for ve0+tied: 0 = 6.2471 (overfits, min at 30M), 0.4 = 5.0367, 1 = 5.1232, 2 = 5.3031. lr 1 at WD 0.4 = 5.1278. ep2 = 5.163.
- Instruments (ref-0225): gap to the 100M twin 0.701 [0.604, 0.785], beating ep2 by 0.111 [0.102, 0.121] paired, mostly on rare words. Most of the epoch 5-10 gain comes from the cooldown (inferred); epoch 9 to 10 alone gives -0.141.
- Floor seeds 1 and 2: `jobs_base_seeds_base_S_wd0p4_lr0p5_r1_ve0_tied.json` (`make_base_jobs --seeds`). Seed 0 is the original cell.
- **Seed floor (3 seeds, 04:03Z):** full DEV 5.0367 / 5.0255 / 5.0283, mean 5.0302, sd 0.0058, range 0.011; every seed has its DEV-sub minimum at e10. Fast suite mean (sd): BLiMP 68.04 (0.71), supplement 63.73 (2.20), EWoK 50.48 (1.05), entity tracking 23.06 (0.13).
- Confirmed: ep3 5.2453 and all-on wd0.4 5.1023 (0.065 worse; instruments: the extra levers slow learning, worse on seen contexts) do not re-name it.
- **Against the new-tokenizer control** base_S_wd2p0_lr0p5_final (registered full DEV 6.1212, DEV-sub min at e2 and then overfits): the winner gains **-1.085 nats/word** on full DEV (5.0367). By lever on full DEV vs the control: ve0 -0.555, ve0+tied -0.818, ve0+tied at WD 0.4 -1.085; tied alone -0.225, dropout -0.291, embed WD 0.005 -0.148 (these three still overfit). Instruments, vs the control's best epoch on DEV-sub: tied 0.000, dropout +0.028, embed WD -0.032, ve0 +0.260, ve0+tied +0.508, WD 0.4 ve0+tied +0.792, ep2 +0.681 (ref-0226). Tying pays only once the value embeddings are gone.
- Critic 26839124: the seeds match the registered floor (ref-0022). At freeze, the R3 trial count comes from executor 3's ledger (runs executed and looked at). r2 is under the ref-0086 firewall, and its knob choices may not use attack or moonshot results. If r2 re-names the base by >0.02, the affected attacks re-screen and the new base needs its own 3 floor seeds.
- **Round 2 (experimenter 09:37Z/09:39Z, `jobs_base_r2_S.json` @ ref-0190):** the locked base is overtrained at E=10. Its E=5 twin (v2_S_10M_e5_s0/s1, jobs_gpu_p2v2.json @ ref-0161) scores 4.9681 / 4.9807 against 5.0367 at E=10. So E becomes a knob: a one-knob star at E=5 around that twin (WD 0.2, 0.7; cooldown 0.75, 1.0; lr 0.35) plus E=10 cooldown 0.75 and WD 0.7. Every cell runs solo with the ref-0034 temps dump (critic ref-0031: raw and calibrated winners picked separately).
- **Size M, 2 of the 4 cells run reads in babylm/gcp/reads/base_size_m/ @ ref-0227):** wd0.4 scored registered 5.1587 (twin 5.1323), wd0.8 scored 5.1342 (twin 5.1059). Both are **worse than S** (E10 twin pool 5.0006). M memorises: final train nats/word is 3.37 at wd0.4 and 3.85 at wd0.8, against S's 3.83 with better DEV. DEV-sub keeps falling into the cooldown (wd0.8: 5.48 at e8, 5.41 at e9, 5.23 at e10). More WD helps at M (0.8 beats 0.4 by 0.025), so the M optimum is above 0.8. wd1.6 and wd0.8/lr0.25 have not run.
- **M at short E (experimenter 15:51Z, `jobs_base_size_M_shortE.json` @ ref-0167, torch 2.7):** wd1.6 E5, wd3.2 E5, wd1.6 E3. If the best cell beats S at E6 (twin 4.9635), its E neighbours and a second seed get added. The first two M cells ran on torch 2.9, so they show direction only. The R4.4 reference is the best of tuned M and the S base (Kim/Kotha 2509.14786 regularised scaling, a known idea).
- **M short-E result (executor 3 21:45Z, reads @ ref-0228 on wip-branch, SHORT_E.md):** twin full/guarded scores: wd1.6 E5 5.1747/5.1492, wd3.2 E5 5.3577/5.3334, wd1.6 E3 5.1745/5.1502. Heavier WD hurts M, and shortening E at wd1.6 doesn't recover it. Best M is still E10 wd0.8 (5.1059 twin), 0.14 worse than S E6 (4.9635). At 10M words, a 3.3x-FLOP model with this recipe loses to S at every WD/E tried: Kim/Kotha-style regularised scaling does not pay here. NOTE: SHORT_E.md records system torch 2.9.1, not the 2.7 the experimenter required, so these cells show direction and cannot bind as an R4.4 reference.
- Selection uses score.json dev_nats_per_word (registered dev_bpw); the harness's own full-DEV pass differs by -0.020 to +0.034 per cell (instruments, results/babylm/base_r1/scorers.csv).

## Next stages (23:45Z, experimenter calls 23:28Z and 23:40Z)

Every s1 cell bottoms at epoch 2-3 and rises after it. More Muon WD gives stability at a worse best point, while lower LR helps at both ends. s1 decays only the 21M Muon matrices. The ~31.5M AdamW lookup-table params (input embedding, untied head, 3 value-embedding tables) run at adam_wd 0 and are the likely memoriser.
- wd {4, 8}: dropped.
- **Round 1** (`jobs_base_r1_S.json`, runs first on both A100s right after s1, at the s1 winner). It starts with the 60-step CUDA lever smoke, then 11 cells: ve0, tied, ve0+tied, embed wd {0.001, 0.005} (value embeds sit in the embed group), head wd 0.1, embed 0.001 + head 0.1, dropout 0.1, MNTP 0.25 (mask p 0.3->0.15), 4 epochs, and all-on.
- 01:05Z fix (ref-0229): all-on originally also passed --embed_wd 0.001. With tied embeddings and no value embeds, no embed group exists, so the three all-on cells died at startup. All-on is now tied + head wd 0.1 (which decays the tied table) + ve0 + dropout 0.1 + MNTP 0.25, the same as the "tied" cell plus the other levers. The cells rerun after the plain control base_S_wd2p0_lr0p5_final (new tokenizer ref-0224; s1 ran on the old ref-0222, so every r1 gain is measured against this control, not s1).
- s1x lr {0.25, 0.125} x wd {1, 2} (`jobs_base_grid_S.json`): runs only if the round-1 winner still peaks at epoch 3 or earlier.
- `jobs_base_reg_S.json`: superseded by round 1. Its EMA cells come back on top of the round-1 winner.
- Then M and L with the regulariser retuned per size (Kim/Kotha: optimal WD grows with size), as a priority stage.

## Not tuning trials
- base_gpu_smoke_levers (ref-0230): 60-step CUDA smoke of all levers; queued.
