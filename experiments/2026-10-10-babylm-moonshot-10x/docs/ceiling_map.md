# Moonshot ceiling map: which channel could physically carry a 10x? (diagnostics; registered before any run)

Status: diagnostic card, 2026-10-10. Labels only: never a screen verdict, never a win, never an attack. Owner: the 10x
moonshot thread. Design credit: the first moonshot thread's handoff (shared/babylm/moonshot/handoff_v0.md).

## Why
Every mechanism on the table improves one of three things a 10M learner lacks relative to a 100M learner: the
**targets** it is trained toward on its 10M contexts, the **contexts** it gets to see, or the **posterior averaging**
over the hypotheses its data supports. Before betting the moonshot on mechanisms, measure how much of the 10M -> 100M
gap each channel can carry when it is fed by an oracle (or, for the posterior, by brute force). A channel with a small
oracle ceiling cannot carry a 10x however clever the mechanism; a channel with a large one is where the moonshot goes.

## The 10x yardstick (registered by the critic, ref-0126)
- **Y-a (primary, mission wording).** Recipe X on train_10M (<=10 epochs) is "10x" only if, at both sizes S and M with 3
  seeds each: (i) X's mean full-DEV nats/word (dev_bpw, overlap-removed; raw shown beside it) <= that of R@100M;
  (ii) X's official NLP and reliable composite scores >= R@100M's; (iii) TEST is confirmed once, on the critic's call.
  R@100M = the base family re-tuned at 100M words (its own WD and epochs, up to 10) with at least X's tuning-trial count,
  not the 10M recipe carried over.
- **Y-b (partial results).** Data multiplier read off the base's own ruler under the ref-0261 ruler rules; no
  extrapolation beyond 10x.
- The p2 port-recipe 100M runs (100M x1 = 4.384, x10 = 4.290) are untuned and weak: no 10x or multiplier claim may be made
  against them. They serve only as the C1 oracle teacher below.

## Common setup
Size S, the base recipe the critic accepts for screens at run time (round-1 winner when frozen; otherwise the best
provisional base, its recipe hash recorded and every number labelled "provisional base"), seed 0, train_10M x 10
epochs, full-DEV nats/word (dev_bpw.py protocol: ctx 512 / stride 256, ws collapse) of the final checkpoint, no
checkpoint picking. The oracle teacher T100 = `p2/ckpt/grid_S_100M_e10_s0/chck_1000M.pt` (size S, port recipe, 100M words
x 10 epochs; dev_bpw DEV 4.290 nats/word raw; harness full-DEV 4.298 raw, 4.312 overlap-removed). Every number that uses T100 carries
"oracle teacher (100M words; out of track)". Code: `babylm/moonshot/kd.py` (`--kd_teacher`, full-vocabulary token-level
KD, no top-k cache), branch `wip-branch`.

## C1: targets channel (3 runs, ~45 A100-minutes; out of track)
- **B**: the base, re-run in the same batch (no KD).
- **C1-self**: KD with alpha = 1.0, tau = 1, teacher = a 10M-trained model of the base recipe (seed 1, final
  checkpoint). Measures what soft targets do by themselves (regularisation, born-again effect).
- **C1-oracle**: the same with teacher = T100.
- Quantities: G = DEV(C1-self) - DEV(T100); **f1 = (DEV(C1-self) - DEV(C1-oracle)) / G**, the share of the remaining gap
  that the 10M contexts carry when their targets come from a 100M learner. Also reported: DEV(B) - DEV(C1-self), the
  seen/new-text split and the lever signature (rare and novelty tilts) of C1-oracle vs C1-self.
- **Registered prediction: f1 in [0.30, 0.75]**, point guess 0.5. The student will match T100 on seen-like contexts
  (the 0.24 seen-text slice) and inherit part of its smoother function on new text.

## C3: posterior channel (5 extra base seeds, ~40 A100-minutes; seeds 0-2 shared with the experimenter's V6 if run at the same recipe)
- K-seed probability average of the base at 10M x 10 (seeds 0..7; seed 0 = the round-1 winner's run, seeds 1-2 = the
  floor seeds; this thread adds seeds 3-7; every K-seed number carries "K x exposure, out of track"), on full DEV from per-token dumps (an ensemble's
  probability of the observed token is the members' mean; `pt_devfull_final.npy`).
- Quantities: g(K) = mean single-seed DEV - ensemble DEV, for K = 2, 4, 8; fit g(K) = g_inf (1 - c / K).
- **Registered prediction: g(8) in [0.10, 0.25] nats/word; g_inf in [0.12, 0.30].** Baby Llama's 4-seed logit ensemble
  at 10M gave -0.11 nats/token (about -0.18 nats/word); Kim/Kotha found ensembles beat the regularised N->inf single
  model. The ensemble's DEV is placed on the base ruler (10/20/30M cells) to read it as a data multiplier.

## C2: contexts channel (deferred; needs KV-cache generation, coordinate with A4)
T100 writes 192-token continuations seeded by 64-token train_10M windows; the student trains on real + synthetic within
100M words of student exposure (5 real epochs + 50M synthetic words). **Registered prediction: f2 in [0.2, 0.6]**
(same definition as f1 with C2 in place of C1-oracle). Runs only after A4's generator exists, to avoid duplicate code.

Readout (experimenter's scorer rule; critic ref-0053): every ceiling number uses THE registered scorer, dev_bpw.py on the fp32
HF export. `python -m babylm.moonshot.dev_bpw_dump` writes its per-token losses (`<run>/score/dev_pt.npz`; it reproduces the
run's registered nats/word or refuses), and `python -m babylm.moonshot.ceiling_report {c3,c1}` reads the dumps: raw =
the registered number, plus overlap-removed. The harness's own end-of-run eval is never used for a ceiling number.

Critic ref-0126: accepted as diagnostics only, bands registered as above; T100 and oracle numbers reported
overlap-removed with the raw value beside them.

## What each outcome means for the moonshot (decided now)
- f1 >= 0.6: the 10M contexts already hold most of what a 100M learner knows how to predict; the binding constraint
  is the learner's targets and generalisation. Target-side mechanisms (posterior averaging, cross-fitted distillation,
  co-distillation inside one exposure) have a ceiling in 10x territory and become the moonshot's main line.
- f1 <= 0.3: contexts are binding. Only new contexts (A4-style or structural recombination) or stronger priors can
  move the needle much; target-side mechanisms are capped near f1 x 0.88 nats.
- g(8) >= 0.2 nats/word (~1.7x by log-linear): the posterior channel is large; pursue one-exposure ensembles (pending
  the critic's R0.2 ruling on grouped models) and ensemble-to-student distillation. g(8) <= 0.1: deprioritise it.
  Critic ruling ref-0110: a packed/grouped ensemble counts K times under R0.2 unless its groups share a real trunk (at
  least one shared transformer block); only shared-trunk designs are candidates inside the track.

## Proxy rules (critic ruling ref-0110)
Mini-ruler subsets (train_1M/2M/5M, `babylm/moonshot/mini_ruler.py`, hashes in their meta files) are allowed for
falsification only: results carry "mini-ruler proxy", the tokenizer is labelled "saw 10M", every proxy run counts as a
tuning trial in the trial log, the dev_bpw protocol is unchanged, and the base recipe is never changed for a proxy.

## Exposure and FLOPs (labels)
C1-oracle and C2 use a teacher trained on 1B words: out of track by construction. C1-self's teacher is a separate 10M x 10
run (100M words): as a pipeline it is 2x the cap, so it is a diagnostic too. C3 members are each 100M words; the
ensemble is K x the cap. None of these is a strict-small result. KD runs cost about 1.4-1.6x the base's training FLOPs
(the teacher forward plus full-vocabulary KL).
