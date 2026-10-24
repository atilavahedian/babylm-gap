# 10x moonshot: design portfolio v1 (2026-10-10 ~04:30Z)

Three design rounds, 92 agents in all (W1 and W2 on Opus with 9 lenses, W3 on design agent 5.1 with 3 whole-system lenses).
Each round generated mechanisms, triaged them, sent the top ones to three adversarial skeptics (prior art, rules, mechanism)
and wrote cards for the survivors. Raw outputs: shared/babylm/moonshot/design/W{1,2,3}_result.json. Cards:
docs/babylm/moonshot/cards/ (drafts, not yet submitted to the critic). Every number below is a judgement call by the
round's agents, not a measurement.

## Bottom line
- No surviving mechanism has a 10x-sized ceiling. After verification every ceiling is 0.05-0.30 nats/word (1.1-2.2x on
  its own), and every expected gain is 0.00-0.03 nats/word.
- Everything-goes-right union of all survivors, with overlaps counted once: about 0.40-0.45 nats/word, roughly 3x. The
  product of the P(pass) values needed for that is about 1e-5. Expected stacked contribution: about 0.03-0.05 nats/word,
  roughly 1.1x.
- What still decides whether 10x is physically reachable inside R0: the ceiling map (C1 oracle targets, C3 seed
  ensembles), now running on executor 3. Only f1 >= 0.6 together with g(8) >= 0.2 opens a channel with a 10x-sized
  ceiling; the rounds put P(both) near 0.1-0.15 and P(10x | both) in low single digits.
- The bits lens (W3) found the one class with a ceiling >= 0.88: a learner whose prior is better than a transformer's
  (Bayes-with-a-learned-prior). The 100M transformer is itself far from the data floor (fitted L_inf about 3.2-3.65
  nats/word vs 4.3 at 100M), so 10x is not ruled out by information. Nobody has built that learner at this scale.

## Survivors, ranked (duplicates across rounds merged)
| # | mechanism (card) | slice | ceiling | expected | P(pass) | screen A100-min | known? |
|---|---|---|---|---|---|---|---|
| 1 | Shared-trunk particle ensemble, one forward (W1 PE; W3 packed population with co-distillation is the same family) | posterior variance | 0.30 | 0.02-0.03 | 0.15 | 170 | known idea, untested regime |
| 2 | Trained train_10M memory reader, document-excluded reads (W3; W3 A-mem is the same family) | memorisation offload | 0.25 | 0.02 | 0.05 | 240 | known idea, untested regime |
| 3 | Cross-fitted Gibbs re-expression (W2 xfit) | contexts | 0.20 | 0.005 | 0.03 | 56 | known ideas, new combination |
| 4 | Optimizer filters: fold-consensus / spike-gated Muon (W1, count once) | memorisation in updates | 0.20 | 0.005 | 0.04 | 46-60 | known family |
| 5 | Procedural "overhypothesis worlds" prior (W2) | prior | 0.15 | 0.003 | 0.02 | 105 | known family (formal-language pre-pretraining) |
| 6 | Cycle-and-average snapshots (W3) | posterior variance | 0.12 | 0.01 | 0.08 | 40 | known idea |
| 7 | IVON weight posterior (W3) | posterior variance | 0.15 | 0.01 | 0.05 | 370 | known idea |
| 8 | Masked-mode targets for the causal mode (W1 rb + W2 self-rb, count once) | targets | 0.06-0.10 | ~0 | 0.04-0.05 | 120-136 | known idea |
| 9 | Population-labelled re-expression (W3, folded into A4 as a gated arm) | contexts | 0.15 | 0.01 | 0.05 | 220 | known idea |
| 10 | Lexicon-first prior (W3), frame-mate substitution (W2), committee weighting (W3), out-of-fold gate (W3), novelty replay (W1) | various | 0.05-0.10 | 0.00-0.01 | 0.03-0.07 | 45-150 | known |

Dead or dropped with reasons: half-data jackknife extrapolation (exact convexity identity says every step loses;
published as contrastive decoding), MemSinks variants, LOO Kneser-Ney mixture, cyclic tail ensemble (merged into 6),
tied-loop core, cross-microbatch SAM, band-factored softmax, IVON-frequency (merged into 7). Details in the W*_result.json
"dropped" lists.

## What runs next (cheapest decisive first)
1. Ceiling map C3 + C1 on GPU (registered, running; dumps on the registered scorer).
2. Zero-GPU gates: plain checkpoint averaging on the floor seeds (prices every snapshot card), frame-mate CPU gate,
   overhypothesis Gate 0 + CPU transfer test, self-rb F0 calibration read on GPT-BERT 10M.
3. Build PE (rank 1) now so it can screen the moment C3 reads: it is the only card whose odds move with C3
   (P about 0.25 if g(8) >= 0.2, about 0.05 if g(8) < 0.1). Needs the shared FLOP-matched reference F (critic 01:20Z).
4. Escalate the one problem with a 10x-sized ceiling (a learner with a better prior than a transformer, legal under R0)
   to a dedicated design agent 5.1 max-effort thread, per Atila's rule.
5. Everything else waits on C1/C3: f1 low parks the target-side cards; g(8) low parks the posterior cards.
