<!-- design agent 5.1 escalation card, 2026-10-10; ruled by the critic at ref-0014 (T1 rejected, T2 accepted; see Part C). Strict-R0 ceiling of the
best legal design: 0.35-0.45 (below the 0.5 asked). Official-FAQ tier ceiling: ~0.6. Needs one critic ruling and one
Atila decision before any GPU run. -->

# F1: the answer to "a learner with a better prior, legal under strict R0, ceiling >= 0.5" and the one card that clears 0.5

## Part A. Under strict R0 (R0.2 + R0.3 as registered) no design reaches a 0.5 ceiling. Three measured numbers say why.

1. **The information floor.** The bits lens puts the information absent from train_10M at 0.25 +/- 0.08 nats/word
   (never-seen words 0.10 of which ~0.08 unrecoverable; the unrecoverable half of the 1-63 band ~0.08; collocations and
   facts present only in the 100M corpus 0.08-0.12). Every learner, whatever its prior, is bounded at about 0.88 - 0.25 =
   0.63 nats/word, which is about 5x by the registered log-linear rate (0.26 per doubling). **A 10x under R0 needs this
   estimate to be wrong by more than 2x.** C1-oracle's f1 is also the test of this floor: f1 >= 0.85 would say the floor
   is lower than estimated; f1 in the registered [0.30, 0.75] band confirms it. No new run is needed for this.
2. **The in-context channel is capped near 0.05.** The W2 Step 0 table (round-1 base vs T100, per-word dumps): on words
   seen 1-63 times the base is 1.57 nats/word behind T100 even when the word occurred in the previous 320 words, vs 1.82
   when it did not; recent repeats carry 0.21 of the 0.74 gap and the base already gains as much from recency as T100.
   So every "Bayes at prediction time" design (meta-learned nonce learner + retrieved occurrences, memory readers, kNN,
   lexinvariant/Minnow-style episodes) has a lexical ceiling of about 0.15-0.2 and a realistic reach near 0.05. The
   100M model's edge on a rare word comes from 10x more *training* occurrences, which one or four in-context examples do
   not replace. The portfolio's memory-reader (0.25) and lexicon-first (0.35) cards already bound this family.
3. **In-weights prior transfer is measured at <= 1.5x and fades.** Formal/procedural pre-pretraining: Hu 2025 ~33% token
   saving at 1B, 0.013 nats at 160M; McCoy & Griffiths MAML prior 0.004 nats at full data; BabyLM 2023 bracket/emergent
   pre-pretraining +/-0.7 BLiMP. The W2 overhypothesis-worlds card already sets this ceiling at 0.15. Nothing legal under
   R0.4 (no outside text, no outside models) supplies a stronger prior than train_10M itself plus procedure, and both
   sources are carded.

What is left is variance (C3 band g(8) 0.10-0.25) and the memorisation dynamic (0.2-0.35), both carded, both capped
by R0.2 at 1x exposure. **Honest strict-R0 verdict: the union of everything legal is ~3x; 10x is not reachable, and
probably not in the data.** The literal "Bayes with a learned prior" class exists only as theory; nobody can build it
in days, and its legal brute-force approximation under the transformer prior is exactly what R0.2/R0.3 forbid.

## Part B. The one card with a measured >= 0.5 ceiling: closed-system Bayes by brute force (official-FAQ tier)

**Label: known ideas, reproduction at strict-small scale.** Kim, Kotha, Liang & Hashimoto 2025 (arXiv 2509.14786: fixed
200M tokens, ensembles beat the N->inf regularised single model; joint ensemble+size asymptote 5.17x, measured 2.09x;
distilling an 8x ensemble keeps 83%); Baby Llama 2023/2024 (same-data teacher ensembles distilled at BabyLM 10M, +3.7
BLiMP, 4-seed logit ensemble -0.11 nats/token); SBP (Yang+ 2025, 42-58% of a 10-20x oracle at 3-6B); A4 (experimenter's card,
closed-system re-expression). Counter-evidence: BERTtime Stories negative at 10M; BabyLM 25.2 -0.03 nats/token at 100M.

**Rule tier (the fork; nothing runs before it is settled).** The official 2025 FAQ (facts_2025.md section 5): "you may
also train eventually on more than 100M words by augmentation ... in a closed system ... training two LMs on half of
the words, and then having them generate more words and training a model on both ... is legit". Our R0.2 caps every
word fed to every model at 100M summed; the official text caps the *augmenters' training data* and leaves the student's
synthetic exposure open. Two readings for the critic to rule on: (T1) unique text <= 10M, each model <= 10 epochs of
corpus words, synthetic words uncapped (K independent 10-epoch teachers legal); (T2) the sum of corpus-word exposure
across all models <= 100M, synthetic uncapped (teachers must share the 100M, e.g. two half-data generators x 10 epochs
or 4 x 2.5 epochs). Either tier is "closed-system augmentation (official FAQ), outside R0.2", the label the critic
created at 00:05Z for A4's past-100M arm. **It needs Atila's one typed line to run at 10M.** It can never be a strict-R0
win; it can be a BabyLM strict-small submission under the organisers' own FAQ.

**Mechanism (size S, round-1 base recipe @ref-0065 unchanged for every member).**
1. Population: K = 8 members, seeds 0-7, train_10M x 10 each. Under T1 these are the C3 seeds already paid for by the
   ceiling map; under T2 they are replaced by the FAQ's two half-data generators (A4b's G_A, G_B) plus K = 4 at 2.5 epochs.
2. Teacher = the members' probability average (8x inference, out of the student's FLOP count as inference).
3. Synthesis: A4a's code path once it exists (64-token train_10M seeds, 192-token continuations, T = 1.0, 8-gram
   copy-block against train_10M, 13-gram R5 check against every eval set, 6-gram overlap and edit-distance report).
   Volume V_syn in {100M, 300M} words, fixed before the run. No new generator code: this card depends on A4's.
4. Student: size S (then M), real train_10M x 10 (100M, inside the cap) + V_syn synthetic, loss = CE + KD to the
   ensemble (kd.py, alpha 0.5, tau 1, full vocabulary, no top-k cache; kd.py gains a multi-teacher average, ~30 LOC).
   The submission is the single student at 1x inference. The raw ensemble is reported beside it, labelled 8x inference.

**Why it should work (which slice).** Variance: g(8) of the C3 band, 0.10-0.25, of which distillation keeps ~0.8.
Bias on familiar words in new contexts (0.36-0.40): the student sees each word in V_syn/10M new frames written by a
model that is itself a better estimator than any member; SBP measured this at 42-58% of the oracle gap at scale, A4's
own ceiling argument puts 30M synthetic words at ~0.5 if worth fresh text. Memorisation: soft targets on repeats remove
the one-hot gradient that keeps paying after a window is learned (novelty tilt should move from 0.43 toward 1.0).
Nothing here reaches never-seen words (0.10) or 100M-only facts: the ceiling is the information floor, ~0.6, which is
also where Kim/Kotha's 5.17x asymptote (~0.63 by our rate) sits. Those two numbers agreeing is the whole case.

**Why a bigger tuned base does not already have it.** Kim/Kotha: a K = 3 ensemble beats the N->inf single regularised
model; the ruler's size and WD tuning cannot reach the posterior average. The FLOP-matched reference (R4.4) is run anyway.

**Falsification screen, ~30 A100-minutes, mini-ruler, before any 10M compute.** The mini-ruler has a free 10x oracle:
base@train_1M x 10 vs the existing base@train_10M (5.0367). G1 = DEV(base@1M) - DEV(base@10M). Runs, one batch, seed 0,
full-DEV dev_bpw.py at the final checkpoint: base@1M (45 s); 8 members at 1M x 10 (6 min); student S at 1M x 10 with
8-teacher KD on real text only (3 min; stage 1 has no synthetic leg, so it measures the T1 ensemble-distillation route
and needs no A4 code); null N-single: KD from one member, seed 1 (soft-target regularisation without diversity; the C1-self
analogue); null N-perm: 8-teacher KD with teacher logits permuted across positions (same 8 forwards, information removed);
FLOP reference: size M base at 1M x 10 (3 min). Readouts: g(8) at 1M beside C3's g(8) at 10M (does the variance share grow as
data shrinks?); phi_ens = (DEV(base@1M) - DEV(ens8)) / G1; phi_stu likewise for the student; rare and novelty tilts.
Stage 2 (after A4's generator exists, ~25 A100-min): add V_syn = 10M synthetic words from the 1M ensemble; nulls N1 and
N2 from the A4 card (unseeded samples; within-sentence word shuffle).

**Kill rule (registered now; the critic may tighten, never loosen).** Kill if phi_stu < 0.20 (the route cannot deliver
even 2x-equivalent at the scale most favourable to variance), or if the student is not ahead of both N-single and N-perm
by 2x the 1M seed floor (measured from the 8 members), or if it does not beat the size-M FLOP reference by 2x that floor.
Advance to the 10M stage (C3 members + 8-teacher KD student, ~35 A100-min; then V_syn) only if phi_stu >= 0.30 and the
rule tier is ruled and Atila has typed his line. At 10M, kill if the student gains < 2x the 0.0058 floor over the
round-1 base or over the nulls; promote to 3 seeds, size M and official eval only if the gain is >= 0.26 (2x by the
rate), and label anything smaller as a reproduction of Kim/Kotha at strict-small, never a win. Mini-ruler numbers are
labels, never verdicts (ruling ref-0110); the tokenizer "saw 10M"; every proxy run is a logged trial. Known bias of the
proxy: a 1M-word generator writes poor text, so stage 2 under-reads the synthetic leg; regularisers over-read at 1M.

**Odds (guesses, stated plainly).** Ceiling 0.6 (the floor). Expected at 10M, size S, T1 tier: ensemble g(8) 0.15,
distilled 0.12, synthetic leg +0.05-0.15, joint 0.15-0.30 (1.5-2.2x). P(student >= 0.26, a 2x) 0.30; P(>= 0.5) 0.08;
P(10x under any tier) < 0.02, because of the floor. Under strict R0.2 this card is not runnable at 10M at all, and
nothing replaces it: the honest strict-R0 answer stays ~3x. What a null result buys: a measured g(8) and phi at two
data sizes, which prices every posterior card in the portfolio and closes the brute-force route with numbers.

**Exposure ledger (10M stage, T1).** 8 members x 99.54M = 796M corpus words; student 99.54M real + V_syn synthetic;
unique text 9.95M. Training FLOPs about 8 + (1 + 8/3) x (1 + V_syn/100M) times the base: R4.4 triggers; the FLOP-matched
reference is the experimenter's size-M WD-tuned base (a 3-4x partial match, labelled as such; a 12x model does not exist in the grid).

## Part C. Critic ruling ref-0014 (2026-10-10 04:32Z) and the committed T2 allocation

**Ruling.** T1 rejected: the FAQ's "the sum of all text seen by all training can not surpass the amount of text
allowed" is broken by 8 members x 100M. T1 may run only as "T1 diagnostic, not BabyLM-legal", never a claim or a
submission. T2 accepted under the label "closed-system augmentation (official FAQ), outside R0.2": corpus-word exposure
summed over every trained model <= 100M; synthetic words from in-system models to the student uncapped; inference
forwards are not exposure but their FLOPs count. Amendments: the 10M gate is read on a T2-budgeted mini-ruler (summed
exposure <= 10M at 1M scale) with the allocation committed first; the binding comparator is the better of N-single,
N-perm and the size-M FLOP reference, and the round-1 base is the exposure-matched comparator; floor v1 thresholds
(0.016 one-seed screen, 0.012 three-seed claim, BLiMP-fast drop above 2.0 kills), blank-line guard, official score.json
only; all A4 leakage rules (13-gram check, 6-gram and edit-distance reports, copy-block, no DEV/TEST conditioning);
promotion needs >= 0.26, 3 seeds, size M and Y-a for any 10x wording, labels "reproduction at strict-small" and
"compute-heavy" with the FLOP ratio. Part A is registered as a prediction: C1-oracle f1 >= 0.85 contradicts the 0.25 floor.

**What T2 does to the ceiling, said plainly.** The Kim/Kotha argument needs K members that each see the whole corpus
for its full epochs; T2 forbids that. Splitting 100M corpus words among K models makes every member weaker than the
base, so the variance leg collapses, and the two-half-generator FAQ example (two 5M-word models) is weaker still. What
survives under T2 is the synthetic leg only: a generator trained within the budget writes uncapped text, and the
student continues on it. That is A4's past-100M arm, nothing more. Its ceiling is SBP's (42-58% of a 10-20x oracle at
3-6B params), about 0.45 here as an everything-goes-right number; BERTtime Stories is negative at exactly our scale.
**Revised odds under T2 at 10M, size S: expected 0.05-0.15, P(>= 0.26) 0.10, P(>= 0.5) 0.02, P(10x) < 0.01.** The
Part B numbers (expected 0.15-0.30, P(>= 0.5) 0.08) stand only for the T1 diagnostic, which is not BabyLM-legal.
So the "official tier" does not open a 5x route; the honest reach under either tier is about 3x, with the T1
diagnostic as the out-of-competition upper reference for the paper ("what unlimited compute on 10M words buys").

**T2 allocation (committed now; it is A4's arm and should run under the A4 card, owned by the experimenter, not as a separate card).**
- Generator G = the round-1 base, seed 0, at 5 epochs (chck_50M.pt, already on disk): 50M corpus words.
- Student = G continued for 5 more real epochs (50M) on a fresh LR cycle, interleaved with V_syn synthetic words from G.
  **V_syn = 100M, fixed here (critic ref-0108: one value, never picked after results).** 300M is not an arm of this
  allocation; if anyone wants it, it is registered as its own arm before any screen and counts in the trial log.
  Hard CE on real tokens, token-level KD to frozen G (alpha 0.5, tau 1) on
  synthetic tokens only. Synthesis = A4a (64-token train_10M seeds, 192-token continuations, T = 1.0, 8-gram copy-block).
- Summed corpus exposure 100M (50M G + 50M student). Synthetic uncapped. Training FLOPs ~ 1 + V_syn/100M x (1 + 1/3) + generation.
- Exposure-matched comparator: the round-1 base (10 epochs, 5.0367). Nulls: A4's N0' (same second cycle, real only),
  N1 (unseeded samples), N2 (within-sentence shuffle). Binding comparator per amendment (2).
- Mini-ruler T2 gate at 1M scale, summed exposure <= 10M: G = base@1M at 5 epochs (5M), student 5 real epochs (5M) +
  10M synthetic words; base@1M x 10 is the exposure-matched comparator and base@10M the oracle; phi as in Part B.
  Known bias: a 1M-word generator writes poor text, so this gate under-reads the synthetic leg; a pass is a label, a
  fail is a kill. Kill if phi_stu < 0.20 or the student is not ahead of the binding comparator by 0.016.

**T1 diagnostic (not BabyLM-legal, recommended as a priced upper reference).** Stage 1 of Part B as written, at 1M scale
(~30 A100-min), then at 10M using the paid C3 members (~35 A100-min): the distilled 8-teacher student is the closest
buildable approximation to the Bayes predictive under the transformer prior on 10M words. Its DEV, placed on the ruler,
says how far "unlimited compute, 10M words" reaches. Every number carries "T1 diagnostic, not BabyLM-legal".

**Honesty lines.** Never call this new. Never present the raw ensemble (8x inference, 9x exposure) as a strict-small
result. Report g(8), phi and the tier on every number. "The brute-force route closes X of the 1M->10M gap" is the
only claim the screen can make.
