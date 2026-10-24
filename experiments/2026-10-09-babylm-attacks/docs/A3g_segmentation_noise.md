# Attack A3g: annealed segmentation noise as a generic regulariser (card, 2026-10-10; CLOSED 2026-10-11)

Status: **CLOSED, no claim on any registered path (critic ref-0012, ref-0013; experimenter 11-10 00:56Z).** Gate history: PASSED (experimenter, v2c calib_control,
split/reads/calib_epochs.txt): E\* = 6 on both raw and cal_t. E6 s0 cal_t is 4.9466 full / 4.9209 guarded. a3a vs E6:
+0.032 / +0.039; a3n(0.1185) vs E6: +0.020 / +0.025. Both clear 0.0074 on both legs. The formal one-invocation read
(A3A, A3N vs E5 and E6) is attacks/A3/reads/calib_a3_estar.txt. Both reads are pinned here; the drive is full, so the
byte-exact copies live in git at ref-0093 (branch wip-branch), babylm/gcp/reads/, shas verified.
Pinned: split/reads/calib_epochs.txt sha256 eb74197add456346643bed958570087c135918974841ac9a316f8a2a75798eac
(cal_t full / guarded: E5 4.9528 / 4.9245, E5 s1 4.9627 / 4.9334, E6 4.9466 / 4.9209, E10 4.9684 / 4.9405).
Pinned: attacks/A3/reads/calib_a3_estar.txt sha256 adefce042d85487d95b250af52bdce544531546802998665bced36a0d3f77200
(one calib_control @ ref-0033 invocation: E5, E6, A3A, A3N(0.1185); cal_t full / guarded: E5 4.9528 / 4.9245, E6
4.9466 / 4.9209, A3A 4.9143 / 4.8817, A3N 4.9271 / 4.8955; E5 − A3A +0.0385 / +0.0428, E5 − A3N +0.0257 / +0.0290;
vs E6 by subtraction: A3A +0.0323 / +0.0392, A3N +0.0195 / +0.0254). **Launch blocker cleared.** It is a
successor to A3 (`A3_bpe_dropout.md`), whose rare-word claim failed its registered null (killed, final, critic
ref-0094): a3a − a3n(0.1185) = +0.0039 raw twin, below the 0.0074 kill line.
Same code (`train.py --bpe_dropout`), no new mechanism.

**Known, not new.** BPE-dropout (Provilkov, Emelianenko & Voita, ACL 2020), subword regularisation (Kudo 2018), and
StochasTok (arXiv 2506.01687, 2025). What is new here is only the measurement: does annealed segmentation noise beat a
tuned, *epoch-optimal* 10M causal base under canonical scoring, after free post-hoc calibration?

## What the A3 screen showed (seed 0, E = 10, solo; before calibration)

Twin gains vs the same-batch E10 base (5.0077):

| arm | twin gain | token ratio |
|---|---|---|
| a3a (all, p 0.1, anneal) | +0.063 | 1.150 |
| a3n (frequent-only, p 0.1185) | +0.059 | 1.150 |
| a3c (rare-only, p 0.2) | +0.049 | 1.051 |
| a3n (p 0.1139) | +0.036 | 1.144 |
| a3b (constant p 0.1, no anneal) | −0.158 | 1.188 |

- **Scope barely matters.** Re-segmenting frequent words only (+0.059) does about as well as re-segmenting all words
  (+0.063).
- **Not the rare-word composition idea.** Rare words gain in every arm, frequent-only included, so the effect is generic.
- **The anneal is required.** Without it the frequent bands lose ~0.18.
- **Raw against the E\* candidate (s0 e4/e5, 4.968):** a3a and the exact null are both ~0.02 better, roughly 6x
  smaller than against E10. Much of the E10 gain is therefore anti-overtraining that fewer epochs also deliver.
- **Noise.** The two frequent-only nulls differ by 0.024 for a 4% change in p (a new random stream), with overlapping
  mid-training curves. This suggests a run-to-run SD of ~0.01 in this family, ~3x the base seed SD. No single-seed
  difference below ~0.03 is trustworthy here.

Calibrated (calib_control @ ref-0033, v2 temps dumps; cal_t, full / guarded): base E10 4.9719 / 4.9433; a3a 4.9143 /
4.8817; a3n(0.1185) 4.9271 / 4.8955. So base − a3a = +0.0576 / +0.0616, and calibration takes only ~0.005 of the raw
gain. a3a − a3n = +0.0128 / +0.0138 cal_t vs +0.0040 / +0.0057 raw, because a3a gains more from temperature than
the null (raw to cal_t: 0.0302 vs 0.0213). Diagnostic arms, cal_t full / guarded: a3c 4.9237 / 4.8920, a3n(0.1139)
4.9360 / 4.9041, a3b 5.1602 / 5.1288. Full text: shared/babylm/attacks/A3/reads/calib_a3.txt.

## Final verdict (2026-10-11)
A3g is closed with no claim. Annealed BPE-dropout (`--bpe_dropout 0.1x(E-r),0xr`, all scope) helps a 10M-word model,
but it did not clear any registered bar. All numbers are cal_t, full / guarded, solo.

| path | gain | bar | verdict | why |
|---|---|---|---|---|
| single model, E10, s1-s3 vs plain E6 pool | +0.043 / +0.046 | claim floor + arm variance | clears | |
| same, vs R4.4 F≥1.9 entry (28L/384d e6) | −0.013 / −0.007 | must beat best of | **FAIL** (ref-0012) | ~1.9x FLOPs; depth spends them better |
| legal split: A3K2 vs plain K = 2, 3 reps | +0.014 / +0.013 (reps .020/.016/.005) | ≥ 0.020 both legs | **FAIL** (ref-0013) | below bar, shrinking across reps |
| singles E5, s0-s5 vs same-seed e5 | +0.026 / +0.024 | screen | pass (diagnostic) | |
| second size, 28L e5 + a3a (Amendment 1) | not run | G = max(0.024, 2·seed gap) | dropped by the experimenter | no claim path left; no more A3 GPU compute |

**What survives (reportable, not citable).**
- As a single-model regulariser at equal words: ~+0.043 over the epoch-optimal base, at ~1.9x compute. A deeper model at
  matched FLOPs beats it.
- Inside the legal 2 x E5 split: a likely small positive add-on of ~+0.013, not established and below the registered bar.
  Roughly half the single-model gain is lost because noisy members are less diverse.
- Not rare-word learning: the token-matched frequent-only null did as well (A3 killed, ref-0002).
- BLiMP: no consistent change. The E5 s2-s5 mean is 68.1 vs base 68.4; s4's 66.2 is a single seed, reported only.

**So what for the mission:** ~5% of the 2x bar inside the winning recipe, and ~17% as a stand-alone regulariser that
depth beats. A recipe footnote, not a lever.

## Result: secondary, 3 whole-mixture replicates (11 Oct 01:10Z; executor 3, git ref-0095 reads/a3g_stack/; solo, torch 2.7)
Plain K = 2 mix minus A3K2 mix, cal_t full / guarded: rep 1 (s0+s1) +0.0203 / +0.0184, rep 2 (s2+s3) +0.0156 / +0.0134,
rep 3 (s4+s5) +0.0054 / +0.0070. **Mean +0.0138 / +0.0129 (SD 0.0076 / 0.0057)**; raw mean +0.0189 / +0.0174.
- That is below the registered 0.020 claim bar on both legs, so **the stack claim FAILS** (critic applies the bar).
- The sign is positive in all 3 reps on both legs, but n = 3 (t ≈ 3, df 2, p ≈ 0.09) and the reps shrink: a likely small
  positive add-on (~0.013, ~5% of the 2x bar), not established and below the registered bar, inside
  the legal split. That is about half the same-seed single-model gain (s0-s5 mean +0.0255 / +0.0239; s4 only +0.009).
- BLiMP-fast a3a s2-s5: 68.0 / 68.6 / 66.2 / 69.5 (base 68.4; s4 −2.2 single-seed, the same seed as the weakest gain).

## Result: primary (claim), 10-10 22:20Z (executor 3, git ref-0096 babylm/gcp/reads/a3g_claim/; solo, torch 2.7)
a3a E10, fresh seeds s1-s3, cal_t full / guarded: mean 4.9103 / 4.8793 (SD 0.0058 / 0.0067); raw 4.9368 / 4.9034;
BLiMP-fast 69.4-70.7 (base 68.4, no drop).
- **vs the plain E6 pool (s0-s4): +0.0431 / +0.0458 cal_t**, raw +0.0325 / +0.0364. Clears the claim floor (0.0112 /
  0.0064) and the arm-variance bar (0.0084 / 0.0082).
- **vs the R4.4 F~2 entry (28L/384d, best e6 s0, torch 2.9.1): −0.0128 / −0.0072 cal_t. A3g is WORSE.** The card's
  reference is the best of both, so the claim **FAILS** against the binding reference (critic ref-0012; ENV_SYS 0.012 off still leaves ~0 / +0.004; entries are scored by seed mean, ref-0097). The
  caveats are cross-env (the measured 2.7 to 2.9 shift is ~0.002, far less than the gap) and a single 28L seed (28L e6 s1
  lands in rep 2).
- Reading: annealed segmentation noise is a replicated gain of ~0.04 nats/word (~17% of the 2x bar) over the
  epoch-optimal base at the same words, BLiMP intact. But it spends ~1.9x the compute, and using compute that compute on a deeper
  model buys more. So it is not the best use of FLOPs. It is still open whether it stacks on the deeper model, since
  noise and depth may be complementary; that is untested.

## Amendment 1: second size (experimenter 22:27Z; REGISTERED, critic ref-0098; NOT RUN, dropped by the experimenter 11-10 00:56Z)
The mission rule requires the gain at a second size. The deeper F~2 cell is the natural one.
- **Arm**: 28L/384d (the menu F2 deep cell, WD 0.4) at E5 with a3a's E5 schedule (`--bpe_dropout 0.1x4,0x1`, scope
  all), seed 0, torch 2.9.1 system python, solo. Same instrument as menu F2 and the 28L split. Job file
  `babylm/configs/jobs_a3g_28l_e5_s0.json`: the trainer flags of `jobs_gpu_ms_28l_e5_s1.json` with only seed, tag
  and the two dropout flags changed. ~1.15 x 2.07 = 2.4x base FLOPs;
- **Reference**: plain 28L e5 s0 (same seed) and s1 (raw 4.9203 / 4.9325; their 0.012 seed gap sets this family's
  noise at 28L).
- **Read**: calib_control @ ref-0033 cal_t both legs (raw reported), within torch 2.9.1; BLiMP-fast.
- **Bars (critic ref-0098, cal_t within 2.9.1, per leg)**: G = max(0.024, 2·|cal_t(s0) − cal_t(s1)|). PASS if arm − s0
  ≥ G on both legs AND the arm beats s1 on both legs; KILL if below G/2 on either leg or BLiMP-fast drops > 2.0;
  otherwise INCONCLUSIVE. VOID without instrument.json solo. Screen only; family trial. A claim later needs 3 seeds plus
  R4.4 at ~2.4x FLOPs. It would justify an a3a-member 28L K = 2 split (not ticketed) and seeds, not a claim.
- **Prior**: depth and noise both regularise a 10M-word run past E\*, so they may overlap. At 12L the noise gave ~0.03
  cal_t at E5 (singles); at 28L, which has more capacity to memorise, I expect +0.01 to +0.035, P(pass) ≈ 0.35.

## Result: secondary (stacking), 10-10 21:10Z (executor 3, git ref-0099 babylm/gcp/reads/k2_estar/; solo, torch 2.7)
cal_t gains, full / guarded, a3a E5 vs plain e5 (same seed):
- s0 +0.0290 / +0.0272; s1 +0.0349 / +0.0331. Both singles PASS (>= 0.020 on both legs).
- K = 2 mixture a3a(s0+s1) vs plain p01 (e5 s0+s1), one shared T each: cal_t +0.0203 / +0.0184, raw +0.0251 / +0.0224.
  The full leg passes and the guarded leg is 0.0016 short, so by the letter this is **INCONCLUSIVE** (critic rules).
  vs the 15-pair e5 pool mean: +0.0226 / +0.0213.
- Reading: noise stacks on fewer epochs, and most of it survives ensembling. It shrinks from ~0.032 (singles) to ~0.020
  (mixture), because noisy members are less diverse: the mixture's gain over its member mean is 0.107 vs 0.118 (SD 0.002)
  for plain seed pairs. A mixed pair (one plain e5 + one a3a) beats A3K2 by a further 0.003-0.005.
- Context: the plain legal split (2 x e5) beats the E6 pool by +0.112 / +0.110 cal_t. A3K2 adds ~0.02 to that, putting
  the combined recipe at ~0.13, about half the 2x bar (0.26) on DEV nats/word. Single-seed pairs only; a claim needs 3
  whole-mixture replicates.
- Experimenter 21:12Z: the stack stays homogeneous A3K2 as registered (the hetero edge is inside noise). Replicate 2 = a3a e5
  s2+s3 vs plain p23 (`babylm/configs/jobs_a3g_stack_S_a3a_e5_s23.json`, torch 2.7, solo); rep 3 (s4+s5) only if the
  critic's combination rule needs it. Critic ref-0076: singles PASS, K = 2 screen INCONCLUSIVE; the claim path is the
  3-replicate mean (p01, p23, p45 with `jobs_a3g_stack_S_a3a_e5_s45.json`) >= 0.020 on both legs plus the arm-variance
  bar; the experimenter decides between running reps 2-3 and INCONCLUSIVE-final. "~0.13 combined" is a screen-level figure, not a claim.

## Mechanism

Each training epoch reads a fresh seeded BPE-dropout re-tokenisation of train_10M. The last max(1, round(0.2 E)) epochs
are canonical. Evaluation is canonical.

Why it might help a tuned base: the input sequence of every repeated sentence changes between epochs. This slows
memorisation of exact token sequences, the way data noising does (Xie et al. 2017). The anneal then re-specialises on
canonical splits.

Why it might not: E\* already stops before memorisation. Free temperature calibration removed most of A2's similar
regulariser gain (+0.019 → +0.002).

## Design (experimenter 11:57Z)

### Secondary, runs first: does noise stack on the epoch lever? (E5, because the legal split is 2 x E5; 2 x E6
would break the 100M cap)
- **Arm**: a3a's schedule scaled to E = 5: `--bpe_dropout 0.1x4,0x1`, scope all, seeds 0 and 1. Job file
  `babylm/configs/jobs_a3g_stack_S_a3a_e5_s01.json`.
- **Read 1**: each seed vs the same-seed plain e5 (split/v2_S_10M_e5_s0, _s1), cal_t, both legs.
- **Read 2**: the K = 2 mixture of the two arm seeds (shared T) vs the K = 2 mixture of the plain legal split (e5 s0 +
  s1), raw and cal_t, both legs, one shared T per mixture (experimenter 14:48Z). Instrument: `python -m
  babylm.moonshot.ceiling_report c3 <two dev_v2.npz> --calib --guard` @ ref-0100 (calib_control @ ref-0033 inside; members
  tempered at one shared T, then mixed), run once per pair; the gain is plain-mixture minus arm-mixture from the same
  instrument.
- **Question**: is segmentation noise additive to "fewer epochs plus ensembling", or the same regularisation in another
  form?

### Primary: the measured E10 recipe vs the best plain base
- **Arm**: a3a at E = 10 (`0.1x8,0x2`). The pick rule from the A3 screen was applied once, on run 2's calibrated reads: a3c cal_t is 4.9237 / 4.8920 vs
  a3a 4.9143 / 4.8817, so a3a stays the arm. Claim seeds are fresh: s1, s2, s3 (critic amendment 1). s0 prompted the card, so it is reported, not
  confirmatory. Job file `babylm/configs/jobs_a3g_claim_S_a3a_e10_s123.json`.
- **Reference** (critic ref-0043): the best of the plain E6 pool and every R4.4 entry with FLOP ratio ≥ 1.9 at its
  own E\* (the F2 menu and size M). Compute is ~1.9x E6 (10 x 1.150 vs 6), ~2.3x E5; the ratio is reported.
- **No null.** The rare-word mechanism is dead, and a compute-matched plain null (base past E\*) is worse than E\*, so
  the reference above binds.

**Environment (critic ref-0101):** the stack and claim runs use p2venv torch 2.7, as the E6 pool. The gate (A3 VM runs vs
E6) stays provisional until executor 3 shows the torch builds match or a same-seed cross-env pair agrees within 0.002.

## Kill rule (critic ref-0094 rules, to be registered before any run)

- **Gating metric, every leg**: calibrated_t on pinned instruments, i.e. the v2 twin @ ref-0026 `--temps` dumps read by
  `calib_control @ ref-0033`, both legs (full DEV and ex_unseen_chars). Raw twin is reported but never gates.
- **Primary (claim)**: a3a mean over fresh seeds s1-s3 (E10) vs the reference above, cal_t, both legs. It must clear
  the critic's claim floor AND gain ≥ 2·sqrt(sd_arm²/3 + sd_pool²/5), where sd_arm is the arm's seed SD (s1-s3) and
  sd_pool is the reference pool's. This is because the family's noise is ~3x the base's. Below 0.0074 reads
  "epoch-equivalent". BLiMP-fast drop > 1.4 kills.
- **Secondary (stacking)**: single-seed reads vs the same-seed plain e5. Kill below 0.0074, pass ≥ 0.020, inconclusive
  in between. The K = 2 mixture read is a screen with the same single-seed bars. A claim on it needs 3 whole-mixture replicates.
- **Trial count**: the 6 A3 screen arms count as 6 trials of this family (critic amendment 3).
- **Honesty**: S-only unless re-tested at M. Known idea (Provilkov 2020; StochasTok 2025). Gains vs E10 are never
  quoted as the result.
- **Pre-registration fact** (A3 screen, E10, s0): calibration widened, not narrowed, the gap between all-scope and
  frequent-only dropout: a3a − a3n(0.1185) was +0.0040 raw and +0.0128 cal_t on full DEV, +0.0057 and +0.0138 guarded.
  Both gaps are single-seed against a family whose run-to-run spread looks like ~0.01-0.02.

**Prior.** Primary: a3a 3-seed cal_t vs the E\* pool, +0.015 to +0.04. P(clears the claim floor) ≈ 0.5. Secondary:
noise on top of e5, −0.01 to +0.015. P(pass ≥ 0.020) ≈ 0.1. Even a primary pass is ~10-15% of the 2x bar (0.26): a
recipe add-on, not a lever.

## Cost

Secondary: 2 S runs at E5 (~1. Primary: 3 S runs at E10 Plus eval and temps dumps
for each.
