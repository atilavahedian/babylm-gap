# Train-loss check: locked vs port recipe, 100M S, E10 s0 (critic item 4, experimenter 23:15Z item 3)

Both runs use the same train_100M data, torch 2.7.0 and harness eval. Values at the final step 24409 (990M words):

| | locked a5_S_100M_e10_s0 | port grid_S_100M_e10_s0 |
|---|---|---|
| final train loss (nats/token) | 2.721 | 2.242 |
| train nats/word (harness) | 4.317 | 3.523 |
| dev nats/word (harness) | 4.673 | 4.420 |
| dev - train gap | 0.357 | 0.897 |
| twin full DEV raw | 4.5467 | 4.2612 |

- Verdict: the gap is already present in train loss (0.79/word). The locked recipe underfits at 100M; this is not a DEV-only or scoring effect. The critic PASSED this check.
- Config differences, port vs locked: muon_wd 0 vs 0.4; lr_scale 1.0 vs 0.5 (all LRs halved); value embeds default vs 0; untied vs tied. The cause among these is untested.
- Files: locked_vs_port_e10.csv (both runs' curves.csv milestones) and locked_vs_port_e10.png (200-step mean train loss; train vs dev nats/word).
