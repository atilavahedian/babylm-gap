# Attack A4: closed-system re-expression (bold bet on bottleneck 1; card, 2026-10-10)

Status: **CLOSED 2026-10-11, A4a killed** (see "Result" below). Accepted with amendments (critic ref-0102, 00:05Z;
prior-art ref-0103, 00:08Z; both binding). Tooling built by the A4 attack thread (branch
wip-branch). Owner: A4 attack thread (max effort). Experimenter: research experimenter thread.

**Label: known family, untested twist at strict-small (prior-art map, section D). Not new.** Synthetic bootstrapped pretraining (Yang+ 2025, arXiv 2509.15248: a synthesiser trained on
the corpus writes related documents; captures 42-58% of a 10-20x unique-data oracle at 3-6B, growing with scale);
sequence-level / self-distillation via samples (Kim & Rush 2016; born-again nets); BabyLM 2025 paper 25.2 (synthetic
text from contrastive decoding, 100M track, 10 seeds: PPL 24.46 -> 23.73, about -0.03 nats/token: our pessimistic
anchor). Knowledge needs varied phrasings to generalise (Allen-Zhu & Li 2023, Physics of LMs 3.1: 1 phrasing per fact
is memorised but not extractable; 5 paraphrases lift out-of-format extraction from ~10% to ~90%). Candidate twists:
novelty-forced generation (8-gram copy-blocking against train_10M, or cross-fitted generators), at strict-small scale,
under exposure-counted R0. Prior-art check (prior_art.md @ ref-0104): known family, twists unmeasured as pretraining
augmentation, but neither mechanism is new. Copy-blocked decoding is MemFree (Ippolito+ INLG 2023), a privacy defence
that near-verbatim paraphrase gets past (hence the edit-distance audit, amendment 6). The map notes that cross-fitted
generation is the BabyLM FAQ's own example. Real text stays in every mix: accumulating synthetic text beside real text
avoids the collapse that replacing it causes (Shumailov+ Nature 2024, Gerstgrasser+ 2024).

## Result (2026-10-11, final): CLOSED, A4a killed
Batch 1 (VM bl-a4-screen-1010-235629: solo, torch 2.7, A100-SXM4-40GB, pin ref-0048) ran G, G_A, G_B, all generation,
the build, the audits, N0'_a and A4a. Its 3h max-run-duration then cut it during A4a's per-token dump, after A4a's
training and score were done. Read with the registered reader @ ref-0105 on executor 3's copy-back ref-0106
(`A4_reads/verdict_a4_b1_partial.md`):

| run | DEV twin (v2) | twin without never-seen-char words | v1 | BLiMP-fast |
|---|---|---|---|---|
| A4a | 5.0188 | 4.9957 | 5.0390 | 68.72 |
| N0'_a | 4.9809 | 4.9508 | 5.0050 | 67.57 |
| base v2e (E10, seed 0) | 5.0088 | 4.9766 | 5.0324 | 66.69 |
| E6 seed 0 (E*) | 4.9635 | 4.9367 | 4.9897 | 67.31 |

- **Gains.** A4a vs N0'_a: -0.0380 (without never-seen-character words -0.0450; v1 -0.0340). Vs the base: -0.0101
  (-0.0191). Vs E6 seed 0: -0.0553 (-0.0590).
- **Why N1 and N2 don't matter.** Both judged references kill (line 0.0074), so N1 and N2 could only add kills.
- **No BLiMP kill.** BLiMP-fast is +1.15 against N0'_a.
- **Step check OK.** 1449 and 1460 steps, both end-fire.
- **Audits clean.** Synthetic text's 13-gram share of DEV is about 1e-6 (train_10M: 1.4e-3). Synthetic-only whole-item
  hits are generic 6-12-word phrases, none in BLiMP or EWoK. No near-copies. A4a's 8-gram overlap with train_10M is
  0.02% (DEV: 8.3%), so the block worked.
- **Curves.** In-loop DEV is worse than N0'_a at every milestone of the second cycle.

**Verdict: KILL.** The experimenter closed A4 (2026-10-11 03:18Z) without running A4b or T2:
- A4b: the same mechanism with weaker half-data generators and no block, about 1% odds.
- T2: KD on the same generator's text, about 4% after A4a.

At 10M words, text written by a model trained on those words carries no information the student lacks. At a 50% share
it displaces real epochs, at a cost of 0.038 nats/word. This matches the priors (BERTtime Stories: negative at 10M;
Kang+ 2025: rephrased-only text no better than natural). A known family and a clean negative.

**Measured on the side:**
- Generation ran at 21-24 ms per decode step at batch 1024, about 2x the plan's "~10 A100-minutes per arm"
  (`A4_reads/gen_speed_b1.txt`).
- Each student's scoring (score_trial, twin, per-token dump) took 17.7 min against about 5 min of training. The
  batch-1 runtime plan left this out.
- The warm-restart null N0'_a (G at 4 epochs, then 6) beat the single-cycle E10 base by 0.028 but trails E6 by 0.017.
  No schedule gain.

## Why (bottleneck 1: known words in new contexts, ~0.40 nats/word)
On text both models trained on, GPT-BERT 10M is only 0.24 nats/word behind 100M; the other 0.61 appears only on new
text. Content words carry 73% of the gap. Words concentrated in few documents (bursty) have a 1.29x [1.21, 1.37] larger
gap than evenly spread words of the same frequency (instruments `dispersion.json`), while left-neighbour diversity does
not matter (0.91). Late epochs memorise: their novelty tilt is 0.43 vs 1.04 for 10x data, and epochs 7-10 buy 6% of
the dev gain. Reading: the 10M model ties each word's knowledge to the few documents and phrasings it saw. Fresh data
helps because it re-expresses the same words in new documents and frames. Ensembles help bottleneck 1 little (the
GPT-2 + GPT-BERT mix gains ~0.06-0.09 per word on 64+ bands, under 10% of the 1.07 per-word gap), so this is mostly
bias, not variance: an attack must add new contexts, not just average models. A4 asks whether a model trained only on
train_10M can write those new contexts for itself.

## Mechanism
1. Generator G = the base at its selected recipe (its training exposure counts under R0).
2. Seeds: the first 64 tokens of train_10M windows (document-aligned where possible).
3. G writes a 192-token continuation per seed, sampled at T = 1.0 (calibrated, not greedy), with **novelty forcing**
   so it re-expresses instead of reciting: arm A4a blocks any sampled token that would complete an 8-gram present in
   train_10M (rejection with that token masked); arm A4b uses two cross-fitted generators instead (G_A trained on
   document half A writes for seeds from half B, and vice versa, so neither can recite its seed's document).
4. The student continues from G's weights with a second LR cycle over the spare budget, on a mix of re-expressed text
   and real text. Total exposure (G's training + generators + every fed synthetic and real word) <= 100M whitespace
   words, logged. Default split if the base selects E* real epochs: spare = 100M - 10M x E*, half synthetic, half real.
   Generation is inference (KV cache): about +10% FLOPs, logged against the 1.25x rule.

## Null twins (same seed, data order, schedule and total exposure)
- **N0' (binding):** the same second LR cycle on real text only. It separates "re-expressed text" from "a warm
  restart plus more real epochs".
- **N1:** unseeded samples from G (same volume, temperature and copy-blocking). Does seeding by training contexts
  matter, or is this plain self-distillation through samples?
- **N2:** A4's own synthetic text with words shuffled within each sentence (same words and topics, no syntax). Does
  coherent re-expression matter, or only topical word exposure?

## Kill rule (registered before any run)
Size S, the base round-1 winner, seed 0. Metric: full-DEV nats/word from `dev_bpw.py` on the final checkpoint. **Kill**
if the best arm's gain over N0' is below 2x the critic's seed floor, or if it does not beat N1 and N2 by 2x the floor.
Kill also on a >= 2 sigma BLiMP-fast drop, or > 1.25x FLOPs without a compute-matched reference. No verdict before the
floor is registered. **Mechanism check** (reported, not a kill): the instruments' signature of the best arm vs N0'
should look like data, not like epochs. Novelty tilt >= 0.8 (10x data sits at 1.04, extra epochs at 0.43), with the
gain landing on words seen 64+ times on new text. If it passes, promote one arm (picked on DEV) to 3 seeds, size M and
official eval under R0-R6.
**Floor (critic ref-0022, noise floor v1, registered 2026-10-10 04:05Z, before any A4 number):** per-seed SD of the
round-1 winner 0.0058 nats/word, BLiMP-fast 0.71. This screen (one seed against one seed) passes at a gain >= 0.016
nats/word over every reference, is inconclusive at 0.012-0.016 (it may buy 2 more seeds of arm and base, once, judged
on 3-seed means at 0.012 and BLiMP-fast 1.4), and is killed below 0.012 or by a BLiMP-fast drop above 2.0 points
against the comparator. `babylm/a4/verdict.py` applies exactly this.
**Metric v2 (critic 2026-10-10 05:02Z under rule ref-0018, before any A4 number).** The primary DEV metric is the
blank-line twin, score.json `dev_nats_per_word_twin` (babylm/eval/dev_bpw_twin.py @ ref-0049, here verbatim); v1
(`dev_nats_per_word`) stays reported. Interim v2 screen floor (critic ref-0019, 0.0036, registered before any A4
number): a gain >= 0.010 nats/word passes, 0.0074-0.010 is inconclusive (2 more seeds of the arm, once; its 3-seed solo
mean is then judged against the solo s0-s4 base pool at E* on the claim floor, critic ref-0042 / ref-0030), below
0.0074 kills; a BLiMP-fast drop above 2.0 kills. `babylm.a4.verdict` and `babylm.a4.t2` default to it. The blank-line guard
(ref-0020: a gain counts only if its v2 gain is > 0 and within 0.0058 of its v1 gain, else "protocol-artifact", no
pass, no promotion, the number reported) holds by construction under v2 (critic ref-0021); it applies to --metric v1
reads, which are information only. The executor runs the twin after every A4 student's score.
**Never-seen-character leg and run condition (critic ref-0001 / ref-0027 / ref-0015, adopted before any A4 number;
instrument ref-0023 / ref-0024 / ref-0025, experimenter 06:43Z).** Every gain is also read on DEV without the words that hold
one of the 85 characters absent from the train_10M raw text. The binding number is the v2 twin @ ref-0026's
`ex_unseen_chars_nats_per_word` (the literal 85-character rule over 10,277 words, the span token rule, (N - NU) /
(W - WU)), v2 against v2, read from score.json (`dev_nats_per_word_twin_ex_unseen`, merged at copy-back) or the
dev_bpw_twin.json beside it; a twin json counting other characters or words is not read. Executor 3 runs that twin in
every score and back-fills the base seeds; the T2 files put it over the vendored ref-0049 copy at setup, sha256
checked. `babylm/a4/unseen.py` (the pinned `unseen_chars.py` ref-0017 as functions: 85 characters, 10,275 words, on
the harness per-word dumps pw/pw_devfull_final.npy) is reported beside it as a cross-check and never binds; base seeds
0-2 carry 0.0360, 0.0364 and 0.0363 nats/word on those words. A pass needs the bar on both legs. A leg below the kill
line kills, a leg in the band makes the reading inconclusive, and a missing leg leaves it pending (critic ref-0025).
Same condition: arm, nulls and base run under one execution condition. A4 runs SOLO against the solo seed-0 base
(batch 1 unpacked; the T2 10M files carry "solo"); `--packed` compares a packed screen with a packed base only, on the
packed floor (pass >= 0.0144, inconclusive 0.0102-0.0144, kill below 0.0102); the packed T2 gate is labels only.
Claims run solo only.
**Null bar (critic ref-0028, registered before any A4 number).** Stream-randomized nulls are noisy (the A3 nulls at
near-identical p differed by 0.024), so against a null (N0', N1, N2, N0'_a; the T2 nulls too) a single-seed pass needs
a gain >= 0.020 on each leg; from the kill line (0.0074) to 0.020 the reading is inconclusive, and 2 more seeds of the
arm and that null decide. The base keeps the 0.010 pass line.
**Promotion gates (critic ref-0011 amendment 2 / ref-0029 and ref-0030 / ref-0031, registered before any A4 number; read
by `babylm.a4.verdict` and `babylm.a4.t2 verdict` after the screen, never changing its status).** Every number is read
on both legs. (1) Calibration: the calibrated_t gain over every graded reference, i.e. softmax(z / T) mixed with lam x
the train_10M unigram, T in {0.90, ..., 1.20} and lam picked jointly and cross-fitted on DEV document parity
(`babylm/tools/calib_control.py` @ ref-0033 over `dev_bpw_dump.py` @ ref-0034 --twin --temps dumps of full DEV, base and
arm on one GPU type; `--calib`). Each gate reads on the kill line of its own metric (critic ref-0032: a bar read on
calibrated_t is the raw bar x the leg's calibrated_t/raw solo E10 SD ratio, 1.28 full / 1.27 guarded). Below the
calibrated_t kill line (solo 0.0095 full / 0.0094 guarded; packed 0.0131, provisional) a pass is
"calibration-equivalent" and is not promoted. (2) Epochs: the raw gain over the same-seed base at E*_raw (`--estar_raw`,
raw kill line 0.0074) and the calibrated_t gain over the same-seed base at E*_cal (calib name `estar_cal`, calibrated_t
kill line); below its line a pass is "epoch-equivalent" and is not promoted. A4's arms keep the registered schedule
below (an E* variant would be one more registered trial). (3) Instrument (critic ref-0035): every compared run's packing
instrument corroborated by executor 3 from VM evidence (`--vm_evidence`, below) or, for a solo run, by its complete
runner sidecar (critic ref-0036, below). A missing number, an uncorroborated instrument, or a T pick at the grid edge
(the grid is extended for every model first), leaves promotion pending. The lam-only calibrated gain is reported beside.
Claims read only against solo base pools of 3 runs or more in the arms' environment and instrument, raw and calibrated
gains both at the claim bar on both legs (2x that pool's SD on the gated metric), and print both comparisons (experimenter
2026-10-10 19:55Z): the E10 pool and the E\* pool (E6 s0-s4, torch 2.7, solo: raw SD 0.0057 full / 0.0031 guarded,
executor 3 19:37Z). The E\* pool binds: a gain that clears the E10 pool but not the E\* pool is epoch-equivalent and is
not promoted. The claim's BLiMP-fast line (1.4 on 3-seed means) reads against the better (higher-BLiMP) of the two
pools' means, both reported, and a null that sat in the band at the screen has its 3 seeds read as a pool on the same
rule (critic ref-0044, before any A4 number). E* picks (experimenter, 2026-10-10 14:28Z): E*_raw = E*_cal = 6 (v2_S_10M_e6_s0:
twin 4.963547, guarded 4.936749).
**Step rule (critic, experimenter 2026-10-10 14:48Z; the K3 bug).** A score is VOID unless its scored checkpoint's step (its
curves.csv row) is the run's final step (result.json steps). Both readouts refuse while any score they read is VOID
(`babylm.a4.verdict.STEP_RULE`); the remedy is a re-score from resume.pt (EMA off) or a rerun at a commit with
ref-0045 (which saves chck_final.pt). Before any run, the pinned samplers were replayed on CPU (exact for real data;
they reproduce the E4 and E5 runs' 39,816,934 and 49,791,035 words): every A4 batch 1 and 2 run, every T2 10M run and
the T2 gate's G, base, size-M reference and N0' end on a scored checkpoint at their final step. The gate's three
synthetic students (train_1M is 1,001,443 words, so they end about 38k words past 20M) cross 20M inside their last
step only if their synthetic text counts at most about 3k words over the planned 10M (expected: about 10k under, from
the windows' untargeted first tokens); the readout catches a miss.

**Environment rule (critic ref-0037, ref-0038, 76151841, 48436034, ref-0039; experimenter 2026-10-10 15:07Z, 19:19Z).** The
torch/CUDA build, the GPU model and the packing instrument (solo or packed) are part of the recipe. Every A4 job runs
solo in executor 3's p2venv (torch 2.7.0+cu126; its batch 1 file runs this file's jobs unchanged under that venv). G
comes from that build too: it is trained in batch 1. The round-1 base s0 (base_r1, torch 2.9.1+cu129) and its E10 pool
are another build; E6 s0 (E\*) and the E-sweep are 2.7.0+cu126. The GPU model binds the same way (critic 15:34Z):
result.json's `gpu` (the device name torch reports) must match across the batch and its references; another instance of
the same model is the same environment. The readouts compare the exact device name, so an A100-SXM4-40GB and an
A100-SXM4-80GB count as two models (critic 15:50Z). All 74 runs on the drive (10-10 15:40Z) record NVIDIA
A100-SXM4-40GB. Packing binds too (critic 48436034 / ref-0039: within one environment means within one instrument). Run
files do not record it, so the batch's instrument is its ticket's (solo; `--packed` for a packed ticket) and each
reference's is declared from its ticket (`--ref_instrument base=solo estar_raw=solo`; the T2 verdict adds `flop_ref`,
the T2 gate declares `oracle`; executor 3's run table names each run's VM, and a packed ticket's VM name carries
`-p2-`). Both readouts refuse when the batch's runs and generators span environments, when a run lacks its build or GPU
record, when a reference (the E10 base, E\*_raw, the T2 oracle and size-M reference) sits in another environment, or
when a reference's instrument is undeclared or not the batch's. The ticket's declaration is enough for a screen;
promotion and every claim also need each compared run's instrument corroborated by executor 3 from VM evidence (critic
ref-0035): `--vm_evidence`, a tab-separated table with columns run, vm, machine_type, gpu, instrument and evidence, one
row per run (the VM whose copy is scored): the VM's job list or startup.log showing one job at a time (solo) or two
(packed), its machine type and its GPU name. A row whose instrument disagrees with the ticket, or whose GPU is not the
run's result.json GPU, makes the read VOID; a run without a complete row leaves promotion pending. Since executor 3's
ref-0040 the runner also writes `instrument.json` into each run dir (solo or packed from the job's pack log, pack
partners, VM, machine type, nvidia-smi GPUs). For a solo run a complete sidecar stands in for the row (critic ref-0036):
(a) startup.sh writes it on the VM during the run and records its own commit (`source`, `startup_commit`; a `+dirty`
copy is no commit); (b) it is copied back with the run's outputs and never regenerated; (c) it states solo positively:
PACK recorded as unset or 1 (`pack_env`), no pack partners and one training process at launch (`train_procs_at_launch`),
since no pack lines alone is not solo; (d) it lists the VM, the machine type and the nvidia-smi GPUs, the run's
result.json GPU among them. Solo must hold through the whole training window (critic ref-0041): exactly 1 GPU compute
process in every sample from the first train step to the final checkpoint write (`train_window_unix` over the job's
`gpu_proc_log`; scoring after the final checkpoint is excluded by time), and a second process in that window makes the
read VOID; a sidecar that cannot isolate the window needs `gpu_procs_max` = 1 over the job, else the row decides.
ref-0040's record lacks `startup_commit`, `pack_env` and `train_procs_at_launch`, so those runs and every run from
before it still need the row. Executor 3's ref-0047 adds them from the next launch: launch.sh passes the last commit of
startup.sh (`+dirty` when the copy sent differs), `pack_env` is the job's PACK (null when unset), and
`train_procs_at_launch` is the first non-zero sample of the GPU's compute processes, taken every 30 s through the job.
Any disagreement between the sidecar, the row, the ticket and the run's GPU record makes the read VOID: another
instrument, a recorded sign of packing in a solo read, a GPU outside its list, another VM or machine type than the row,
or a sidecar written for another run. The 0.002 single-pair admission that `--env_pair` implemented is retracted (critic
ref-0038: same-seed reruns differ by up to 0.006, so one pair can neither pass nor fail it) and the flag is gone. A
cross-environment read would stand only if its margin cleared its bar + ENV_SYS on both legs, raw and calibrated_t (raw
0.014 full / 0.013 guarded, calibrated_t 0.014 / 0.014), plus PACK_SYS across instruments (critic ref-0046: 0.012 /
0.010; A4 changes the token stream, so the packed-lift bars of ref-0107 do not apply), so A4 reads within one
environment and one instrument only. The E10 reference is therefore v2e (`v2_S_10M_e10_s0`, E10 s0 under torch 2.7,
solo; experimenter 19:19Z): `--base_score <v2e score.json> --ref_instrument base=solo`, and the calib json's `base` is v2e's
temps dump. The calib json's `estar_cal` must be the `--estar_raw` run (E\*_raw = E\*_cal = 6), else promotion stays
pending. A read against the single-seed v2e is a screen only (critic ref-0039; both readouts print it as their scope). A
claim needs 3 seeds per side in the arms' environment (experimenter 19:19Z) and its floors from solo pools there of 3 runs or
more: at E10, v2e, v2f and one more under torch 2.7 (the torch 2.9 pool's SD is not borrowed). Each gain sits on the
claim floor of its own metric and pool (critic 76151841 / ref-0032: 2x that pool's SD on the metric being gated;
E\*-referenced claims use the E6 pool's SDs, and the E10 floors do not transfer to E6) and on the arm-variance bar on
the same metric's SDs (critic 14:30Z / ref-0043).

## Odds (guesses, stated plainly)
Ceiling: if synthetic words were worth as much as fresh real words, 30M of them would be worth ~0.5 nats/word. That's
the default split at E* = 4, i.e. 40M unique-equivalent words, two doublings at ~0.26 each by log-linear interpolation.
Reaching 0.2 needs each synthetic word worth about a quarter of a fresh real one (17.5M equivalent): SBP got more than
that at 3B, while BabyLM 25.2 got far less at small scale. Guess: +0.03 to +0.15
nats/word, P(passes kill) ~0.15, P(>= 0.2) ~0.05. A null result is informative: it says a 10M-scale model cannot
bootstrap new contexts for itself, and that bottleneck 1 needs a different mechanism.

**Prior on the round-1 base (A4 thread, 2026-10-10 02:00Z, before any A4 run):** gain over the binding comparator -0.05 to +0.08 nats/word (central +0.02), P(passes kill) ~0.08, P(>= 0.2) ~0.02: lower than above because the new base still improves at epoch 10, so its late real epochs are not wasted on memorisation, the warm restart is paid against a base that gains most in one long cooldown, and N1 must be beaten too; the calibrated generator (+0.26 nats/token overconfidence, like the 100M twin) writes better samples, which is the main upside.

## Rules note for the critic
R0 counts synthetic words as exposure, our reading of the 2025 exposure cap. The official FAQ also says closed-system
augmentation may go past 100M words ("training two LMs on half of the words, and then having them generate more words
and training a model on both the original data and the new one is legit"; facts_2025.md section 5). If A4 shows a
real gain under R0, a separately labelled arm with synthetic volume past 100M would need the critic's and Atila's
call. It is not part of this card.

## Screen (small first; size S, seed 0, about 1.5 A100-hours)
Runs: N0', A4a, A4b (+ 2 half-data generators), N1, N2, with the base round-1 winner as G. Plus generation (KV cache,
~10 A100-minutes per arm). Every run logs exposure by source (real / synthetic / generator), tokens/s and FLOPs.
Outputs go to shared/babylm/attacks/A4/<run>/ with curves CSV + PNG, as all runs do.

## Amendments (critic ref-0102 and ref-0103, binding)
1. **Binding comparator** = the better of N0' and the round-1 base itself. The arm must clear 2x floor over both, and
   over N1 and N2.
2. **Exposure ledger per arm**: G's training + each A4b half-generator's training + every synthetic and real word fed to
   the student <= 100M. Generator training comes out of the arm's own budget. Seeds read at inference do not count.
   N0' gets the same student-fed total as its arm (so A4a and A4b each have their own N0').
3. **FLOPs**: generator trainings count. An arm above 1.25x its reference's FLOPs needs a FLOP-matched reference (R4.4).
4. **R5 leakage**: a 13-gram check of all synthetic text against every eval set runs before any screen number is reported.
5. One arm is promoted, picked on DEV. No verdict before the floor is registered.
6. A4 arms and N1 report 6-gram overlap and nearest-neighbour edit distance to train_10M (an 8-gram block lets
   one-token-edit copies through). A near-copy rate that differs materially from the blocked target is labelled.
7. Priors to cite: Kim/Kotha 2025 sequence-level self-distillation, about -0.13 nats at 200M tokens and 300M params with
   1:3 real:synthetic, on par with heavy weight decay; synthetic-only 4.07 vs teacher 3.71, so pure synthetic hurts
   (priors_data_efficiency.md row 1d). BERTtime Stories (2024.conll-babylm.28): synthetic stories negative at 10M (62.2
   -> 58.3-61.5), null at 100M. Kang+ EMNLP 2025: optimal synthetic share about 30%; rephrased-only is not better than
   natural text. Our registered 50% sits above Kang's optimum.
8. The synthetic share (50% of student-fed words) is fixed before the run. A 30% arm only if registered as its own arm
   before any screen; it counts in the trial log.
- Past 100M (official-FAQ closed-system augmentation) is a separate label outside R0.2, needs Atila's OK, out of scope.

## Run registration (A4 thread, 2026-10-10, before any run; the experimenter may change any line before launch)
**G and budgets.** E* = epochs of the base round-1 winner (seed 0). G = that run if E* <= 5; otherwise the winner's
recipe re-trained at E_G = 4 epochs (full schedule, seed 0), and the base itself stays the comparator. A4b's two
half-generators: the same recipe on document halves A and B (per source, a seeded word-balanced document split),
E_g = 2 epochs each. Ledger at E_G = 4 (one epoch of train_10M = 9.95M words; real cycle-2 text = whole epochs; the
synthetic target equals the real part, so the 50% share holds to within a batch):

| arm | G | generators | cycle-2 real | cycle-2 synthetic | total |
|---|---|---|---|---|---|
| N0'_a | 39.8M | 0 | 59.7M (6 ep) | 0 | 99.5M |
| A4a | 39.8M | 0 | 29.9M (3 ep) | 29.9M | 99.5M |
| N0'_b | 39.8M | 0 | 39.8M (4 ep) | 0 | 79.6M |
| A4b | 39.8M | 19.9M (2 x 5M x 2 ep) | 19.9M (2 ep) | 19.9M | 99.5M |
| N1, N2 | as their arm | as their arm | as their arm | same volume | as their arm |

FLOPs: A4a vs N0'_a = 1 + generation (about 0.1x). A4b trains about 1.25x N0'_b, so N0'_a (same model, about A4b's
training FLOPs) is A4b's compute-matched reference at the screen; R4.4's wider-or-deeper reference applies at promotion.

**Second LR cycle (student and every null).** Warm start from G's evaluated weights (EMA weights if the recipe has EMA),
fresh optimizer state, the base recipe's flags and LR shape (constant, linear cooldown over cooldown_frac to lr_floor)
over the cycle-2 steps, plus a linear warmup over the first 2% of steps (a warm restart; identical in every arm and
null). Window format = the recipe's last window spec. Real and synthetic windows are tiled per source and epoch, then
interleaved in proportion, so every batch carries both sources in the registered share. Seed 0 everywhere.

**Generation.** Seeds: 64-token windows at line starts, tiling train_10M greedily without crossing a document boundary
(a document start includes its BOS), shuffled with seed 0 and used in order until the word target is reached.
Continuation: 192 tokens, T = 1.0, no top-k or top-p, ids >= 16384 masked, a sampled BOS ends the sample. A4a's block
removes, at every step, every token that would complete a token 8-gram of the train_10M stream (an exact draw from
the renormalised distribution, the same as rejecting and resampling). A4b: G_A writes for half-B seeds and G_B for
half-A seeds, no block. Kept text: the continuation only (the seed is real text and is not fed to the student), cut to
whole words, blank lines dropped, documents shuffled, then tokenized like train_10M (BOS before each document).
N1: from BOS, 64 self-generated tokens (discarded, BOS masked) then 192 kept, so only the seed's origin differs from
the arm. N2: the arm's own documents, words shuffled within each sentence (sentence-final punctuation and a leading
speaker label stay in place), identical word counts.

**Audits (before any screen number).** (4) 13-gram overlap (words lowercased, non-alphanumerics to spaces) of all
synthetic text with every file of the official evaluation_data and with DEV, next to train_10M's own overlap as the
base rate; eval items hit only by synthetic text are listed. (6) Token 6-gram and 8-gram overlap with train_10M; the
nearest-neighbour edit distance of 2,000 sampled documents to train_10M around their best 6-gram anchors (Myers
semi-global distance on tokens); near-copy rate = share within 10% of their length. Baselines: 2,000 DEV chunks of the
same length, and 2,000 unblocked continuations from the same G and seeds (a probe, never trained on).

**Screen order.** Batch 1: G (if needed), G_A, G_B, generation, audits, N0'_a, A4a, N1 (for A4a), N2 (for A4a),
N0'_b, A4b. Batch 2 only if A4b has the lower DEV: N1 and N2 for A4b. Each student is scored by score_trial.sh
(dev_bpw full DEV, the kill metric, plus the fast suite for the BLiMP-fast kill line).

**Implementation details (A4 thread, before any run; code in babylm/a4/ and babylm/harness/cycle.py).** Halves: per
source, one PCG64(4242) permutation of train_10M's documents dealt to the half with fewer words. If the word target
needs more rows than there are seeds, the seed order repeats (each continuation is an independent draw); the number
of passes is logged. N1's 64 discarded tokens are drawn without the block; the block applies to the 192 kept ones, as
in its arm. The block hashes token 8-grams (two polynomial hashes, a collision about 3e-8 per lookup): a collision can
only remove one more token, never let a copy through. The leak check also matches whole items of 6-12 words (BLiMP
pairs are shorter than 13 words, so a 13-gram check alone cannot see them). G = the generator run's final weights
(resume.pt; the EMA weights if the recipe has EMA); the cycle's words_seen continues G's, so its milestones continue
and score_trial.sh scores the last one (100M for the a family, 80M for the b family). Every generation run first
checks that cached decoding matches the model's forward pass and refuses if it does not.

## T2 arm: closed-system augmentation past 100M (registered 2026-10-10, before any T2 run)
**Mandate and rulings.** Atila 04:36Z ("yes run", to the coordinator's post naming "the A4 T2 synthetic arm, pool Critic ref-0014: T2 is the official-FAQ tier, label "closed-system augmentation (official FAQ),
outside R0.2"; corpus-word exposure summed over every trained model <= 100M, synthetic words to the student uncapped,
inference forwards count in FLOPs only. Critic ref-0108 accepted the allocation ref-0109 (F1 card Part C) with V_syn
fixed at 100M; it runs under this card. Never a strict-small (R0) entry, never "new".

**Allocation (ref-0109 as amended by the experimenter, 04:47Z, before any T2 run).** G = the base recipe at 5 epochs, seed 0,
annealed on the recipe's own schedule (constant LR, then cooldown to 0.1 over the second half): the round-1 winner's
args with --epochs 5, i.e. the experimenter's v2_S_10M_e5_s0 args, trained in batch 1 as a4_t2_G (1,218 steps; 49,791,035
words, the round-1 base's count at that step). This replaces ref-0109's G = the base's chck_50M, which sits at the end
of the constant-LR half, unannealed (harness dev 5.52 there vs 5.12 at the end), so its text and KD targets would have
been far weaker than the model the arm must beat. Student = G (its resume.pt; the recipe has no EMA) + A4's second LR
cycle (fresh optimizer, 2% warmup, the recipe's constant + cooldown shape) on 5 real epochs of train_10M (49,763,942
words) interleaved with V_syn = 100M words of G's A4a text (64-token train_10M seeds, 192-token continuations kept,
T = 1.0, 8-gram copy block against train_10M). Loss: CE on real rows; on synthetic rows (1 - alpha) CE + alpha tau^2
KL(G_tau || student_tau), alpha 0.5, tau 1, full vocabulary (the moonshot's kd.py terms restricted to the synthetic
rows, babylm/a4/t2.py; the teacher reads only those rows). Corpus exposure 49.79M + 49.76M = 99.55M <= 100M; the
trainer refuses a plan above the cap (--exposure_rule t2).

**References.** Exposure-matched: the round-1 base itself (seed 0, 10 epochs, 5.0367 under v1). Nulls with the same
cycle and KD rule: N0' (real only, no teacher), N1 (100M unseeded words from G), N2 (the arm's text, words shuffled
within sentences). FLOP reference: the base builder's size-M base (its score.json, when it exists; the arm costs about
2.8x the base's training FLOPs: G 0.5, cycle 1.5, teacher forwards 0.33, generation about 0.45, and the ledger reports
the measured ratio). Binding comparator = the lowest DEV among base, N0', N1, N2 and the FLOP reference. G is scored
and reported (did the student beat its teacher?), never a comparator.

**Mini-ruler gate (1M scale; labels only, ruling ref-0110; runs first).** train_1M from babylm.moonshot.mini_ruler
(--sizes 1, seed 20261010, 2048-token pieces): 1,001,443 words, 1,601,735 tokens, sha256 ref-0111...cd83 (the job
refuses any other subset). G@1M = a4_t2g_G = the base recipe on train_1M at 5 epochs, seed 0, annealed (123 steps,
5.04M words); a4_t2g_base = the base recipe on train_1M x 10 (the exposure-matched comparator). a4_t2g = G@1M + 5 real
epochs of train_1M (5.00M words) + 10M words of G@1M's text (seeds and copy block from train_1M only) under the same
KD rule; a4_t2g_n0, a4_t2g_n1, a4_t2g_n2 likewise; a4_t2g_M = the base recipe at size M on train_1M x 10 (3.32x FLOPs
per token). Cap at this scale: 100M x 1,001,443 / 9,953,773 = 10,060,938 corpus words (10 epochs of train_1M are
already 10.01M); the arm uses 10.04M. phi_stu = (DEV(base@1M) - DEV(arm)) / (DEV(base@1M) - DEV(round-1 base at
10M)), all under the primary metric. **Killed (proxy)** if phi_stu < 0.20 or the arm experimenters the binding comparator
(lowest DEV of base@1M, N0', N1, N2, M@1M) by less than 0.016 nats/word on the twin (the experimenter kept the gate's bar,
read on v2); under --metric v1 a experimenter that fails the blank-line guard reads "protocol-artifact" (no proceed); else the
10M stage may run (a label, not a verdict). G@1M
is reported, not compared. Known bias: a 5M-word generator writes poor text, so the gate under-reads the synthetic leg.

**10M stage.** Batch 1: G, generation (100M words and a 2,000-row probe), split, audits, a4_t2 and a4_t2_n0, scores
(G's too). Batch 2 only if a4_t2 clears the base and N0' (pass or inconclusive on both): N1, N2, their audits and
students. Every score runs the blank-line twin @ ref-0026 and merges v2 and the never-seen-character leg into
score.json, and each 10M student writes its harness per-word dumps (pw/) for the leg's cross-check. The 10M files run
SOLO (the comparator is the solo seed-0 base);
the gate may run packed (labels only, critic ref-0066). The A4 thread reads gate and verdict after copy-back.

**Verdict (`python -m babylm.a4.t2 verdict`).** Metric v2 primary, v1 reported (see the kill rule above); thresholds =
the interim v2 screen floor (critic ref-0019: >= 0.010 passes, 0.0074-0.010 inconclusive, < 0.0074 kills; a --metric
v1 read uses floor v1 ref-0022, 0.016/0.012, for information). Gain over each reference = DEV(ref) - DEV(arm); a BLiMP-fast drop above
2.0 below the binding comparator kills; corpus exposure over 100M kills; under --metric v1 a gain failing the
blank-line guard (ref-0020) reads "protocol-artifact"; a pass also needs the never-seen-character leg (kill rule above). A kill or an artifact against any reference already scored is final; missing references
leave it pending. Audits first (all of A4's: 13-gram R5 check on every eval set, 6-gram and edit-distance reports, copy
block, no DEV/TEST conditioning); score.json only. Promotion (3 seeds, size M, official eval, Y-a
for any 10x wording) needs >= 0.26 over the binding comparator; a pass below that is reported as a reproduction at
strict-small, never a win. Every number carries "closed-system augmentation (official FAQ), outside R0.2", "known
ideas, reproduction at strict-small" and "compute-heavy" with the FLOP ratio. Mechanism check as above, reported only.

**Odds.** Registered (ref-0108): expected 0.05-0.15 nats/word, P(>= 0.26) 0.10, P(>= 0.5) 0.02, P(10x) < 0.01. A4
thread, lower: with chck_50M as G (0.40 nats/word behind the final base) central -0.03 to +0.05, P(screen pass) ~0.08.
With the annealed 5-epoch G (expected a few hundredths behind the base, so its text and KD targets are close to the
model the arm must beat): central 0.00 to +0.06, P(screen pass) ~0.12, P(>= 0.26) ~0.03. The binding comparator is
likely N1 (unseeded text with the same KD), which takes away the teacher's share of any gain.
