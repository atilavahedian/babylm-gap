# A4 batch 1, partial read (2026-10-11 03:15Z): arm a only
Inputs: executor 3's copy-back @ ref-0106 (babylm/gcp/reads/a4_b1/ on wip-branch) of VM
bl-a4-screen-1010-235629 (cut by its 3h max-run-duration during a4_a's per-token dump; a4_a's training and score.json
finished). Reader: `python -m babylm.a4.verdict` @ ref-0105, base = v2_S_10M_e10_s0 (v2e), E*_raw = v2_S_10M_e6_s0,
both solo; no --calib (the per-token dumps are not all done), so the calibrated gates read "?".
A4a cannot pass: a pass needs every reference cleared, and two references (N0'_a, base) already kill. N1 and N2 can
only add kills. The card's judged arm is the lower-DEV one of A4a and A4b (amendment 5); A4b is not run.
Audits (amendment 4, before any number): leak.json and copy_{a4a,n1a,a4b}.json are summarised in the JSON's "audits".

| family | arm DEV (v2) | reference | ref DEV | gain | gain v1 / v2 | gain without never-seen-char words: v2 twin / harness cross-check | status | BLiMP-fast arm / ref |
|---|---|---|---|---|---|---|---|---|
| a (a4_a) | 5.0188 | N0' | 4.9809 | -0.0380 | -0.0340 / -0.0380 | -0.0450 / ? | kill | 68.72 / 67.57 |
| a (a4_a) | 5.0188 | base | 5.0088 | -0.0101 | -0.0066 / -0.0101 | -0.0191 / ? | kill | 68.72 / 66.69 |

| family | promotion gate (its metric's kill line, both legs) | gain full | gain without never-seen-char words | lam-only full / without |
|---|---|---|---|---|
| a (a4_a) | calibrated_t over N0' | ? | ? | ? / ? |
| a (a4_a) | calibrated_t over base | ? | ? | ? / ? |
| a (a4_a) | raw over the same-seed base at E*_raw | -0.0553 | -0.0590 |  |
| a (a4_a) | calibrated_t over the same-seed base at E*_cal | ? | ? |  |

judged arm: a4_a -> NO VERDICT: missing ['N1', 'N2'] [critic ref-0019, interim v2 screen floor 0.0036; solo] (gain over N0' -0.0380 < 0.0074; gain over base -0.0101 < 0.0074)
environment: torch 2.7.0+cu126 / NVIDIA A100-SXM4-40GB / solo (ENV_RULE, critic ref-0037 / ref-0038 / ref-0039)
instrument evidence (critic ref-0035): 6 of 7 compared runs corroborated solo from VM evidence (4 by a complete instrument.json sidecar, critic ref-0036); not corroborated: a4_a; 4 instrument.json sidecar(s) agree
scope: a screen only (critic ref-0039): no claim until the arms' environment has a solo E10 pool of 3 runs or more (torch 2.7: v2e, v2f and one more), whose SD on the gated metric sets the claim floors; the torch 2.9 pool's SD is not borrowed
E*: experimenter 14:28Z: E*_raw = E*_cal = 6 (v2c calib_control)
