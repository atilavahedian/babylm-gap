# Attack A2: input-noise augmentation (card, 2026-10-09)

Code `babylm/harness/input_noise.py` (`train.py --noise_mode replace|mask|gauss --noise_p --noise_dist --noise_r2l`),
tests `babylm/tests/test_input_noise.py`. Screen jobs `python -m babylm.configs.make_a2_jobs`.
Status: **killed (calibration-equivalent), see Verdict**; built and CPU-tested; **accepted by the critic with one amendment (ref-0062, applied)**. Screen base = round-1 winner
`jobs_base_r1_S.json:base_S_wd0p4_lr0p5_r1_ve0_tied` @ ref-0065 (tied, no value embeddings, muon_wd 0.4, lr_scale 0.5;
full DEV 5.0367); jobs regenerated and CPU-smoked 2026-10-10 03:0xZ, ticketed to executor 3.

**Known, not new.** Training-time token augmentation for data-constrained AR pretraining (Chen et al., arXiv
2606.16246: 150M params, 75M tokens, 100+ epochs; uniform random replacement 15% is their best single method, minimum
val loss 4.015 → 3.841 nats/token (−0.17), optimum moves from epoch 16 to 28; R2L 50% with a direction token −0.10;
mask 15% −0.10; best combo random 5% + R2L 50% + offsets −0.21; no WD/dropout comparison). Masked-input
regularisation (Xu et al., arXiv 2606.06888: an auxiliary loss on masked inputs, i.e. a second forward, on top of
strong WD; ≈1.3x unique data). Older relatives: word dropout (Gal & Ghahramani 2016), data noising as smoothing (Xie
et al. 2017), SwitchOut (Wang et al. 2018), the 10% random replacement inside BERT/GPT-BERT masking. Prior-art map
row: "AR token-noise augmentation, untested in BabyLM, needs >10 epochs to pay". What we test is only whether the
single-forward, clean-target version helps a tuned 10M base inside the 10-epoch cap.

## Mechanism
On every causal row, each eligible input position (a real token that is not BOS) is replaced with probability p by a
token drawn uniformly from the vocabulary (BOS excluded) or from the train_10M unigram distribution. Targets keep
the clean tokens, so at a corrupted position the model must predict the next token through a wrong input, and every
later position must predict through a context that contains it. Optional `--noise_r2l q`: a row is reversed with
probability q and starts with a direction token (Chen et al.'s format); its targets are exactly the L2R row's tokens.
Noise is fresh at every visit (private CPU generator, state in resume.pt). Eval is always clean, L2R. MNTP rows are
untouched. Cost: CPU-side indexing only; FLOPs 1.0x; wall-clock logged.

## Why (bottleneck 1: known words in new contexts)
72% of the GPT-BERT 10M→100M gap shows only on unseen text (0.6 of 0.84 nats/word; on shared train text the 10M
model is only 0.24 behind), and the experimenter's ranking puts known-word-in-new-context at ~45% of the prize. Our port
recipe peaks on DEV at epoch 2 (5.48) and climbs to 9.98 while train falls to 1.83: from the second visit on, the
model fits each exact window. A 10M model sees each context ~10 times; a 100M model sees ten times as many distinct
contexts per word. Input noise turns every revisit of a window into a slightly different context (p = 0.15 gives
~150 corrupted tokens per 1024-token window), so the model is pushed to predict from the remaining evidence rather
than from a memorised exact prefix; this is the cheapest way to manufacture "new contexts" without new words. Chen
et al. measure the effect in the closest regime on record, but their gain sits at the far side of the baseline's
optimum (epochs 16-68). Our base overfits much earlier (port: epoch 2), so the regime may pay inside 10 epochs; a
tuned round-1 base (embedding WD, dropout, tied/fewer lookup tables) may remove part of that headroom, and dropout
(Xue et al. 2023) is the same kind of regulariser. **Prediction (registered):** +0.03 to +0.12 nats/word on full DEV
over the base arm; replace beats mask at the same p; P(passes kill rule) ≈ 0.3.
**Prior for the tied, no-VE base (written 2026-10-10 01:4xZ, before any run; the registered prediction above stands):**
that base is under-trained and calibrated (still improving at epoch 10), and dropping the value-embedding tables (with
tying on top) already removed most new-context damage (instruments: gap 0.985 ve0+tied vs 1.493 tied only and 1.464
untied + dropout 0.1; attribution corrected 01:51Z, before any run), so noise mostly costs signal here:
−0.03 to +0.04 nats/word, best at p = 0.05, P(pass) ≈ 0.12, rising back toward 0.3 only if the winner is a low-WD
cell that overfits again before epoch 10.

## Null twins (same seed, base, positions logic, rate)
- **N-mask** `--noise_mode mask`: the same positions at the best arm's p get the MASK id. Information is removed but
  no misleading token is inserted (word dropout).
- **N-gauss** `--noise_mode gauss`: no token changes; every eligible raw input embedding gets isotropic Gaussian
  noise with σ_x² = p·E_r‖e_r − e_x‖²/d (r from the best arm's distribution, live table each forward), i.e. the same
  expected squared perturbation as replacement, spread continuously over every position.

If either null does as well, the active ingredient is generic input noise of that strength, not discrete
distractors. For the R2L arm the replace-only arm at the same p (A2a) is its twin.

## Kill rule (registered before any run)
Size S, train_10M × 10 epochs, the base round-1 winner's recipe and seed 0; the base is re-run in the same batch with
the noise code at default flags (same seed, data order, GPU type). Metric: **full-DEV nats/word from `dev_bpw.py`
(ref-0054 protocol, collapse) on the final checkpoint**, no checkpoint picking: the registered number is `score/score.json` dev_nats_per_word (fp32 HF export, critic
ref-0053), masked twin `score/dev_bpw_masked.json`; result.json full_eval and DEV-sub are diagnostics only.
**Metric v2 (experimenter 04:59Z, decided on base seeds only under a pre-registered rule, before any A2 number was read):** the primary number is `score.json` dev_nats_per_word_twin (blank-line twin, `dev_bpw_twin.py` @ ref-0049, seed SD 0.0036), judged against the critic's v2 floor; v1 above is reported alongside. A gain on v1 that is absent on v2 is the p(newline | newline) artifact that random-token noise can move, and does not count. Pad guard on v2: the same twin call plus `--vocab_mask 16384` (the twin passes dev_bpw.py's flags through). **Screen floor (critic ref-0019, 05:34Z, supersedes ref-0021 and ref-0022, before any A2 number was read):** interim floor 0.0036 on v2; screen pass >= 0.010 nats/word vs the same-seed base; 0.0074-0.010 inconclusive (2 more seeds of arm and base, once); kill below 0.0074. Null comparisons use the same bars. Claim floor = pooled SD of >= 5 plain base seeds, registered before any claim. BLiMP-fast kill > 2.0 (screen), > 1.4 (claim). These numbers replace the v1 thresholds below. **Critic ref-0001 (05:59Z, before any v2 A2 number):** the A2 screen is PACKED (2 per A100 under MPS), so it reads at the packed bars: floor 0.0051, pass >= 0.0144, inconclusive 0.0102-0.0144, kill < 0.0102, on v2 for BOTH legs: full DEV and DEV minus the 10,275 words holding characters absent from train_10M (`babylm/tools/unseen_chars.py`, frozen list). Claims run solo only, against the solo s0-s4 base pool.
**Kill** if the best arm's gain over the base is below 2× the critic's seed floor, **or** it does not beat the
better of N-mask and N-gauss by 2× the floor, **or** its DEV-sub gain at the end is more than 0.5σ below its gain 3
epochs earlier (fade rule, 18:37Z). The R2L arm is promotable only if it also beats A2a by 2× the floor. Kill also
on a ≥ 2σ drop on BLiMP-fast, or > 1.25x the base's wall-clock without a compute-matched reference. A null that
itself beats the base by 2× the floor is reported to the experimenter as a finding (a known idea), not promoted under this
card. **Floors (critic ref-0022, registered 04:05Z, before any A2 number):** screen 2x floor = 0.016 nats/word (seed SD 0.0058 on score.json); BLiMP-fast kill = a drop > 2.0 points; screen gain 0.012-0.016 = inconclusive, may buy seeds 1-2 of arm and base once (R3), judged on 3-seed means at 0.012 / BLiMP 1.4. Nulls use the same thresholds. Under the executor's 2-per-A100 MPS packing (04:02Z) wall-clock is read on tokens/FLOPs (noise is CPU-side, FLOPs 1.0x). **Promote exactly one arm**,
picked on DEV before seeds 1-2, to 3 seeds, size M and official eval (R0–R6, ref-0088); the 3-seed verdict re-uses
no screen numbers. **Best arm** = lowest full-DEV nats/word among A2a-e; ties within 0.005 go to the smaller p, then
no R2L. At promotion the better screen null is the binding twin and gets 3 seeds.

Provisional base (critic ref-0086): noise is a regulariser-type attack, so any later base change in WD, EMA, MNTP or
dropout forces one re-screen; until then a kill reads "killed (provisional base)" and a promote stays provisional.

## Screen (1 seed, ~8 A100-min per S run)
| arm | flags | note |
|---|---|---|
| A2a | `replace 0.05 uniform` | Chen's combo rate |
| A2b | `replace 0.15 uniform` | Chen's best single rate |
| A2c | `replace 0.30 uniform` | our base overfits harder than theirs |
| A2d | `replace 0.15 unigram` | plausible (frequent) distractors |
| A2e | `replace 0.05 uniform, r2l 0.5` | Chen's best combo minus offsets |
| A2-nM | `mask`, best arm's p | null |
| A2-nG | `gauss`, best arm's p and dist | null |
| base | default flags | same batch |

Arms first (6 runs incl. base, ~50 min); I pick the best arm on full DEV and the two nulls follow (~16 min). Total
8 runs ≈ 70 A100-minutes, all counted in the trial log. **Every arm, the base re-run and both nulls run at
`--vocab_pad 16512`** (MASK 16384, R2L 16385), so all share one init stream (critic ref-0062: in a 1-seed screen an
init difference is as large as the kill threshold). Pad rows stay in the softmax: with a tied head the MASK/R2L rows are trained as
inputs, so each verdict reports the DEV softmax mass on ids >= 16384 per arm (CPU tiny smoke on the tied, no-VE base:
base 4.5e-4, replace 2-8e-4, mask null 1.8e-3, i.e. at most ~0.002 nats/token). **Pad guard (experimenter; amended by critic ref-0061, before any run):** every arm is scored
both unmasked and with ids >= 16384 masked at eval (logits -inf, i.e. nats + log(1 - mass) per token) and both are
reported. Attack arms (A2a-e) count unmasked (what the model predicts and the official pipeline sees); the base and
the nulls count at the better of the two. The kill uses those numbers. Masked score: the same `dev_bpw.py` call as score_trial.sh plus
`--vocab_mask 16384` (default 0 leaves the registered protocol unchanged).
Diagnostics per arm: dev-minus-train gap per epoch, the instruments' lever signature
(`python -m babylm.instruments.signature`, novelty tilt), and the frequency slice (does it help frequent words in new
contexts, rare words, or neither). FINAL untouched.

**Screen result (v1, 05:50Z; verdict reads the v2 twin):** packed base 5.0417; A2a +0.025 (harness +0.022, broad across
domains), A2b -0.077, A2c -0.258, A2d -0.081, A2e -0.094; masked = unmasked to 4 dp; BLiMP-fast A2a -0.6. Best arm A2a.
**Batch 2** (`make_a2_jobs --batch2 A2a` -> `jobs_a2_batch2.json`, experimenter 05:53Z, cancelled if the twin gain < 0.010):
claim seeds 1-2 of A2a and the base, packed in pairs; the two nulls at p 0.05 (seed 0, packed together); a labelled
refinement pair p 0.025 / 0.035 (seed 0), exploratory only, which does not touch A2a's claim. All comparisons are
packed vs packed (critic ref-0027).
**Solo batch (experimenter 06:10Z, replaces batch 2; `make_a2_jobs --solo A2a` -> `jobs_a2_solo.json`):** A2a s0-s2, gauss null s0 and the refine pair run SOLO at the base job's own vocab (replace and gauss need no pad ids), so they are code-, init- and data-order-identical to the solo base seed pool (train.py at default flags is bit-identical to ref-0065's harness, which differs only by the A2 files) and are read against that pool. This honours ref-0062's intent (one init stream per comparison) without the pad. The mask null needs the MASK id, so it runs at --vocab_pad 16512 against its own padded solo base s0.

**VERDICT (final 2026-10-10 15:35Z): KILLED (calibration-equivalent).** Formal read on the pinned v2 twins (ref-0026,
executor cross-check ref-0089 `babylm/gcp/reads/calib_a2_check.txt`): A2a minus base, full DEV raw +0.0220, temperature-
calibrated +0.0047; guarded (no never-seen-character words) raw +0.0128, calibrated -0.0025; both below the packed
kill line 0.0102. Agreed with the experimenter 07:59Z; critic ref-0011(2): a calibrated single-seed gain below the packed kill line is not promoted.
A2a's raw gain over the packed base, +0.022 harness / +0.025 v1 (+0.0133 on DEV without never-seen-character words),
is mostly calibration: a train_10M unigram interpolation leaves +0.0139 / +0.0062 guarded, and a post-hoc temperature
(base T=1.10, A2a T=1.05) leaves +0.0024 +- 0.0053 (CPU DEV sample, experimenter, temp_check.py @ ref-0085). Higher rates,
unigram distractors and R2L all lose 0.08-0.26. Lesson: on this tied, no-VE, WD 0.4 base, input noise only buys
smoothing (Xie et al. 2017), not "known words in new contexts". Solo seeds and nulls cancelled; the formal read uses
the experimenter's eval-only temperature dumps with calib_control @ ref-0033 once the critic pins it.

## Exposure and legality (R0)
One forward per window, as the base; targets are the base's tokens (R2L rows: the same tokens in reverse order); the
MASK/R2L ids are not data. No teacher, no generated text, no outside data or models.

## Verified on CPU (tests)
Default flags build nothing; `replace` and `gauss` at p = 0 (objects built and wired) reproduce the port's losses and
full-DEV nats bit for bit; a tiny run with default flags is bit-identical to the base commit ref-0090. Targets are
never changed; BOS and padding are never corrupted and never inserted; the measured corruption rate matches p; the
unigram sampler follows the counts; R2L rows keep the target multiset and stay causal (x'[j+1] = y'[j]); the gauss
null's E‖Δe‖² matches uniform replacement within 5% and is off in eval and on MNTP rows; the private generator leaves
the global RNG untouched; pause + resume is bit-exact with noise on; every mode runs with line windows and MNTP;
words and tokens seen equal the base's.
