# Attack stacking: one harness, every flag composable

Branch `wip-branch` = the experimenter's harness (`wip-branch` @ ref-0112) +
A1 temporal self-distillation (PR #55 @ ref-0113) + A2 input noise (PR #60 @ ref-0114) + A3 BPE-dropout
(PR #61 @ ref-0115) + A4 second-cycle tooling (branch wip-branch @ ref-0116, no PR yet).
Code: `babylm/harness/stack.py`. Tests: `babylm/tests/test_stack.py`.

## What holds (CPU tests)

- **Off = base, bit for bit.** Every attack built and wired but inert (A1 alpha 0, A2 p 0, A3 p 0), alone and all
  together, reproduces the base's step losses and full-DEV nats exactly. Explicit off flags build nothing.
  Checked also across branches: the same tiny runs on this branch and on the source branches give identical loss
  curves and final weights (base vs ouf5ji; A1-on vs PR #55; A2-on vs PR #60).
- **Every pair and the full stack run** (A1 x {A2 replace, mask, gauss, R2L}, A1-smooth x A2, A3 x {A1, A2 replace,
  R2L}, A1+A2+A3, A1+A2+MNTP+EMA+line windows, A4 cycle x {A1, A2 replace, A2 gauss, A1+A2}) and resume bit-exact
  across a pause. An inert A4 (`--init_from` a step-0 checkpoint) with inert A1 and A2 is the base bit for bit.
- **R0 exposure unchanged.** A1 and A2 work on the same forward: words and tokens equal the run without them.
  A3 changes tokens and steps, never canonical words. A1/A2 inside an A4 cycle leave the cycle's words per kind
  unchanged. `result.json` gets an `exposure_r0` ledger whenever an attack is on (real input words, prior lineage
  included, + synthetic words + generator words vs the cap; under A4 it equals A4's own `exposure` total).

## Choices made (the interactions)

| Interaction | Choice | Why |
|---|---|---|
| Order in a causal micro-batch | A3 segmentation, then A2 R2L, then A2 token noise, then the forward (A2 gauss hook), then A1 read/KL/write | Noise must corrupt what the model actually reads; R2L reverses the segmented row; the marker is protected from noise |
| What A1 caches when input is noised | The prediction of the **noised** forward; no extra clean forward | Zero extra compute and exposure. The target becomes an EMA over noise draws, which is how temporal ensembling (Laine & Aila 2017) was defined |
| A1 on R2L rows | Neither read nor written; CE only | The R2L prediction conditions on the right context: a different distribution that would pollute the L2R teacher |
| A1 under A3 | Cache keyed by canonical stream position. A piece keeps the key of the canonical token with the same byte span; a re-segmented piece gets no key (CE only) | Positions shift every epoch under dropout; byte spans do not. An intact token's teacher is shared across segmentations |
| A1 epoch clock under A3 | Read off A3's sampler (A3 streams are longer than N tokens) | `tokens_seen / N` would switch the KL on early |
| A2 mask + MNTP | Share the MASK id 16384 | Same meaning (information removed); no new embedding row |
| A2 replacement vocabulary under A3 | The canonical vocabulary (BOS excluded); unigram counts from the canonical stream | Every A3 piece is already in the vocabulary |
| MNTP rows | Untouched by A1, A2 and the A1 cache | Unchanged from each attack's own design |
| A1 in an A4 cycle | Keys only rows of a real source that is the training split (same sha256); synthetic rows CE only. Cache starts empty (not carried by `--init_from`); epoch clock = cycle tokens / split tokens | Synthetic text has no canonical position; a fresh cache avoids a teacher from the pre-cycle LR schedule |
| A2 in an A4 cycle | Applies to real and synthetic rows alike | The mix is not confounded by treating the two kinds differently |
| A2 mask/R2L in an A4 cycle | Needs a generator/base trained at `--vocab_pad 16512` (else `--init_from` refuses: cfg differs) | The padded vocab is part of the weights; critic ref-0062 already runs A2 arms padded |
| A3 + A4 | Refused by train.py for now | The cycle samples canonical streams; A3 streams for synthetic sources do not exist yet |

One code change outside the glue: `self_distill.loss` accepts a real target with key -1 (CE only, KL averaged over
all real targets). With every target keyed it is the old code path bit for bit (A1 suite unchanged and green).
A2 and A3 each record the rows of their last call (`InputNoise.last_r2l`, `MultiStreamSampler.last_rows`).

## A4 notes

- Synthetic rows enter through the same micro-batch path, so A2 treats them as real text.
- If A4 wins and A3 wins, composing them needs A3 streams for the cycle's sources (real and synthetic): a small extension of CanonicalKeys and the cycle sampler.

## Costs to know before a stacked GPU run

- A1 + A3: one canonical byte-span map per epoch stream (two kept), about 0.5 GB host RAM at 16M tokens.
- A1 coverage drops under A3 and R2L (the logged `sd_cov`), so A1's effective alpha is smaller in those epochs;
  that is the defined behaviour, not a bug. Screen stacks at the promoted single-attack settings first.

## Scoring stacked runs

Stack verdicts use the registered number only. Since floor v2 (critic ref-0021) that is `score.json`
`dev_nats_per_word_twin`, the blank-line twin from the dev_bpw protocol on the fp32 HF export: base seeds 0-2 have a
mean of 5.0002. Floors: interim screen 0.0036 (ref-0019); the claim floor is the pooled n>=5 base SD. `dev_nats_per_word` (critic ref-0053; 5.0367) is reported for information only. Never `result.json` `full_eval` (the harness eval differs per model
by -0.020 to +0.034, instruments ref-0084). The composition tests use `full_eval` only as an equality check between
runs that must be identical, never as a score.

## Stack jobs (babylm/configs/make_stack_jobs.py)

The stack launches once two or more attacks are promoted (pool: promotion):

    python -m babylm.configs.make_stack_jobs --arm A1=FILE.json:JOB --arm A2=FILE.json:JOB [...] --loo

- Each `--arm` names that attack's promoted screen job. Only its attack flags are copied onto the winner job, `jobs_base_r1_S.json:base_S_wd0p4_lr0p5_r1_ve0_tied` @ ref-0065, for seeds 0-2.
- An arm whose other settings parse differently from the winner's is refused, as are A3 + A4 and fewer than two arms.
- `--loo` adds every leave-one-out stack, so the final report can attribute the gain to each arm.
- If any arm pads the vocab (A2), every job is padded too, and three base re-runs at that pad are added (`stack_base_s*`) as the comparator.
- With A4 in the stack, each job is a second LR cycle and `--epochs` is dropped. By default every seed starts from A4's seed-0 G. That is fine for a screen, but it hides the generator's own variance (experimenter, 03:48Z).
- `--own_g` is for claim runs: seed k reads every A4 run directory as its `_s{k}` twin (`a4_G_s1/`, `a4_data_s1/`, `a4_GA_s1/`, ...), so each seed starts from its own G_k. The A4 thread builds those directories first, and the note lists them. Critic ref-0117 ruled on this:
  - A shared G is allowed only for a screen labelled "shared-G screen", and G_0's full exposure is charged to every run that uses it.
  - A claim needs its own G_k per seed. Exposure is counted per seed, never pooled: G_k's training, the words G_k generated, the seed's own words and the generator FLOPs.
  - G_k is frozen to the seed-0 card's recipe, generation settings and mix. There is no per-seed tuning and no picking among the G_k.
- A stack is judged as a claim (critic ref-0021, floor v2). Its 3-seed means of score.json `dev_nats_per_word_twin` are compared with both the base and the best single promoted arm, at the claim floor (critic ref-0019): the pooled SD of at least 5 plain base seeds, registered before any claim is read. BLiMP-fast may drop by at most 1.4. Stacks wait for that floor. Under critic ref-0001, the bar must hold on v2 full DEV and also on v2 DEV without the 10,275 words that contain characters never seen in train_10M. Claims run solo, one job per GPU and never packed, against the solo s0-s4 pool. The job file marks this with `"solo": true`.
- An arm whose own runs used a different `--vocab_pad` has no same-init comparator, so it is re-run alone at the stack's settings (`stack_a1_s*`). `--singles` re-runs every arm alone.
- `babylm/tests/test_make_stack_jobs.py` checks that every job differs from the winner only in attack flags, seed, tag and the shared pad. It checks this both after parsing and token by token.
