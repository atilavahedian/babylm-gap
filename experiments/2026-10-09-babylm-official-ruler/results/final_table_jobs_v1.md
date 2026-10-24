# Final table: official-eval job list (draft v1, ruler, 11 Oct 05:40Z)

This is a draft for the critic to register. Ruler code is at the commit that adds this file: ZS_ONLY, DEV_BPW=0, BOS_ENS=0 and the score_mix twin guard. Everything runs on spot A100 40GB (a2-highgpu-1g), because GLUE numbers must all come from one GPU type.

## Metrics per model (registered rules)
- **Zero-shot:** the full official 2025 suite (BLiMP, Supplement, EWoK full, Entity, WUG, Reading, COMPS) on upstream pipeline bf55c1131e53654c2a87418f5629c26959acd710.
  - The early read comes from the CPU batch now running.
  - The table uses the GPU run, the same device as the reference scores. The CPU run of base_S_e6 measures the drift between the two.
- **GLUE7, claim metric:** final-token pooling (critic ref-0212), from the 2-fine-tune-seed ensemble (seeds 42+43, critic 44552922).
  - For a K=2 mixture: member 0 is fine-tuned with seed 42 and member 1 with seed 43, then the probabilities are averaged.
  - The score exists only if all 7 tasks have results.txt.
- **GLUE7 BOS-pooled:** a side number, never a claim. Seed 42 only (BOS_ENS=0), for singles. See question Q2 for mixtures.
- **Held-out:** twin DEV nats/word from the existing reads, so it is not recomputed (DEV_BPW=0).
- **Mixture guard (binding):** the twin DEV of the exported mixture at T=1 must be within 1e-4 of the read's raw mixture DEV.
  - Set TWIN_REPO and MIX_REF_NPW.
  - The result is written to MIX_GUARD_<name>.txt. FAIL makes the official leg VOID.

## Jobs
Cost basis:
- One S GLUE fine-tune (7 tasks, sequential official script) took 63 min on A100. This was measured on base_S_e6 in certify_b.
- A 28L fine-tune is estimated at about 2.1 h by compute scaling (2.05x non-embedding FLOPs). This is unmeasured.
- Zero-shot takes about 5 min per model on GPU. The guard takes about 10 min for an S mixture and about 25 min for a 28L mixture.

| # | Model(s) | Job (env on top of EWOK_ZIP, FT_PARALLEL=0, DEV_BPW=0, GLUE_ENS=2) | Fine-tunes | GPU-h |
|---|---|---|---|---|
| 1 | S E6 singles s0 s1 s2 | `score_hf`, HF_NAME="S_e6_sK S_e6_sK_lastpool" (same URI twice), BOS_ENS=0, FMT_JSON=/opt/hinton/babylm/recipes/harness/format.json | 3 x (2 last + 1 BOS) = 9 S | 9.6 |
| 2 | S K=2 mixtures p01 p23 p45 | `score_mix`, HF_URI="m0 m1", TWIN_REPO, MIX_REF_NPW | 3 x 2 = 6 S | 7.0 |
| 3 | 28L K=2 split reps 1 2 3 | `score_mix` as in row 2 | 3 x 2 = 6 28L | 14.4 |
| 4 | 28L e6 singles s0 s1 s2 | as row 1 | 3 x 3 = 9 28L | 19.2 |
| 5 | T100 (S, 100M, e10) | as row 1 | 3 S | 3.2 |
| 6 | Published GPT-2 | seed-43 runs only (seed 42 BOS and lastpool are already in certify_b) | 1-2 small | ~1.5 |
| 7 | Published GPT-BERT causal | seed-43 run only (seed 42 is in certify_a) | 1 GPT-BERT | ~2.5 |
| - | Our GPT-BERT seeds | already inside the held gptbert training jobs (GLUE_ENS=2) | - | (ruler pool) |
| - | Untrained-S sanity pair | already queued (lastpool pair job) | - | (queued) |

Totals:
- **Claim-grade (rows 1-5 without the BOS runs, plus 6-7):** about 47 GPU-h.
- **With the BOS side numbers:** about 58 GPU-h.5/h). So these run in waves, each admitted as earlier VMs close.

## VM bins
Each bin is 6-7 h of work, so maxdur is 10 h Rows 3-4 are split by rep and seed, so a preemption loses at most one model's unfinished tasks. Per-task GLUE resume works with GLUE_RESUME_FROM.
- **B1:** row 1 (S singles).
- **B2:** row 2 + row 5.
- **B3:** 28L rep1 + rep2 mixtures.
- **B4:** 28L rep3 mixture + 28L e6 s0.
- **B5:** 28L e6 s1 + s2.
- **B6:** rows 6-7. These can ride on whichever bin finishes early.

With 3 GPUs this is about 19 h of wall time if run sequentially.

## The big lever: concurrent GLUE (Q1)
Sequential GLUE leaves the A100 mostly idle on S and 28L models. The concurrent script (all 7 tasks at once) does not run out of memory for these sizes; that OOM was GPT-BERT at seq 512.
- Critic ref-0293 requires a byte-identity check (concurrent vs sequential, same GPU type) before concurrent GLUE counts.
- Proposal: one check on base_S_e6_lastpool, whose sequential seed-42 run is already in the queued sanity pair job. The check reruns it with the concurrent script and compares sha256 of every output.
- If IDENTICAL, rows 1-5 use FT_PARALLEL=1. The estimate is about 2.5x less GPU time, so roughly 58 GPU-h becomes about 23.

## Questions for the critic
- **Q1:** Is the concurrent-GLUE identity check above enough to register FT_PARALLEL=1 for Hinton models?
- **Q2:** Mixture BOS side number: the ruler proposes none. A BOS "mixture" would need its own 2-member ensemble, and it carries no claim.
- **Q3:** Can the CPU zero-shot stand for the table instead of the GPU rerun?
  - Only if the base_S_e6 drift check is exact on every task.
  - If it is, the GPU jobs can add ZS_ONLY-style skipping of zero-shot. Not built yet.

## v2 re-price (05:30Z, after critic ref-0350 / ref-0397 and experimenter priority)
Assumptions:
- Concurrent GLUE is registered per size after one sha256 identity check.
- It is assumed to run about 2.5x faster than sequential. This is unmeasured; the S check measures it. One fine-tune then costs about 0.42 h for S and about 0.85 h for 28L.
- Zero-shot for S singles comes from CPU only if base_S_e6's CPU-vs-A100 check is exact on every task. S mixtures and 28L run zero-shot on GPU, about 5-10 min per model, so no extra exactness checks are bought.

| prio | rows | GPU-h |
|---|---|---|
| (i) | Lastpool sanity pair (base_S_e6 + untrained, sequential, seed 42) + S identity rerun (concurrent), then S split p01/p23/p45 (6 FT) + S E6 s0-s2 (5 new FT; s0 seed 42 = the pair) + zero-shot/guards | 8.0 |
| (ii) | 28L split reps 1-3 (6 FT) + 28L e6 s0-s2 (6 FT). The first retrained member runs sequential + concurrent (the 28L identity check). Plus zero-shot/guards | 14.6 |
| (iii) | GPT-2 lastpool seed 43 (~1), GPT-BERT causal seed 43 (sequential, ~2.5), T100 lastpool ens2 anchor (0.85) | 4.4 |
| (iv) | BOS side numbers, FT seed 42 only: S e6 s0-s2 + 28L e6 s0-s2 | 3.8 |
| total | ||

If either identity check fails, that size falls back to sequential. Worst case is about 58 GPU-h

## v3 re-price at FT_CONC=3 (09:40Z, experimenter: every GLUE job uses FT_CONC=3)
Assumptions:
- With 3 slots for 7 tasks, wall time is about max(longest task, total/3). The estimate is about 2x faster than sequential (unmeasured; the S identity job measures it). One fine-tune costs about 0.5 h for S, about 1.0 h for 28L and about 1.3 h for GPT-2.
- GPT-BERT stays sequential, because it OOMed at 7-way and its 3-way memory is untested.
- certify_b was cut before pub_gpt2_lastpool finished (2/7 done, both off majority). So the GPT-2 final-token reference now needs both seeds (42 and 43) in row (iii).

| prio | rows | GPU-h |
|---|---|---|
| (i) | ftverify base_S_e6_lastpool (concurrent 0.5 + sequential 1.05) + untrained 0.5; S split 6 FT; S E6 s0-s2 6 FT (ftverify dumps no probabilities, so s0 reruns); zero-shot/guards | 9.1 |
| (ii) | 28L: first member sequential 2.1 + concurrent 1.0 (identity), then 11 FT x 1.0; zero-shot/guards | 16.4 |
| (iii) | GPT-2 lastpool seeds 42+43 (2 x 1.3), GPT-BERT causal seed 43 (sequential, 2.5), T100 anchor (2 x 0.5) | 6.1 |
| (iv) | BOS side numbers, seed 42: 3 S + 3 28L | 4.5 |
| total | ||

## v4 (12:50Z): S identity MISMATCH on multirc, all remaining GLUE sequential
- The S identity check (ftverify @ref-0194) came back MISMATCH. Six tasks were bit-identical; multirc differed. Concurrent GLUE ran in 3533 s against 3942 s sequential, a 10% saving. So every remaining GLUE line runs at FT_PARALLEL=0, and the 28L identity check is dropped.
- Sequential fine-tune cost: about 1.1 h for S (measured) and about 2.2 h for 28L (scaled). The v3 figures were about 2x too low.
- Line files are in final_table_lines/:
  - final_i_s_e6.json: S split p01 + E6 singles. The executor splits it into B1/B2.
  - final_ii_28l_{x0,x1,x2,z1,z2,z3}.json: one 28L model per VM.
  - a5_100m_*: the A5 100M point.
  - BOS (iv) is cut.

## v5 budget rule (experimenter, 11 Oct 18:35Z)
- Re-project when z1 closes, using its measured 28L fine-tune time (the 2.2 h figure is scaled, not measured).
  1. x2 + z3. 28L GLUE7 drops to n=2 per arm, so no GLUE7 claim at 28L.
  2. A5 s1/s2. T100 becomes single-seed and is flagged as an upper bound.
- The A5 grid is protected over the 28L GLUE7 column.
- Ledger reservations stay worst-case; admits are planned on the actual-rate projection.

## v6 A5 ruling and re-projection (experimenter 23:15Z / 23:16Z, critic 23:12Z)

- A5 grid_b and grid_c are CUT. grid_a finishes E1 and closes. The WD 0.1 probe is dropped.
- The 100M yardstick is the port recipe at E10 (T100 = grid_S_100M_e10_s0, 4.261 raw), made a 3-seed mean with s1/s2. Spec: a5_100m_port_seeds.json, which also re-dumps s0 on A100 with the same instrument. Its label is "best of registered 100M points; not fully tuned, so x is an UPPER BOUND".
- Trainer code ref-0165 adds levers that are all off by default over the ref-0158 that trained s0. Torch is 2.7.0 in both.
- Required before T100 is used: a train-loss check, locked vs port.
- Locked-recipe 100M points are reported as a finding: the 10M-tuned recipe loses about 0.29 nats/word at 100M; the cause is untested.4 h, from z1's boolq/mnli timing.
  - So 28L GLUE7 stays at n=3. The cut rule still applies if real closes run over.
- Admit order:
  1. A5 port seeds (project-b, after grid_a).
  2. 28L z2, x0, x1.
  3. 28L z3, x2.
- Critic 23:17Z:
  - Spec ref-0196 ACCEPTED. Footnote: "trainer commit differs, training-equivalent (critic diff check)". Torch is 2.7.0 in both, so no torch footnote.
  - Train-loss check done by executor 3 (ref-0364): locked 4.32 vs port 3.52 train nats/word, so the locked recipe underfits at 100M. That is not a scoring bug.

## v7 re-projection at z1r close (23:50Z, 11 Oct)
- Measured: z1r GLUE wall 15114 s for 13 28L member-tasks, so about 2.1 h per 28L member (7 tasks). A 2-member 28L GLUE job is about 4.3 h plus about 0. 28L GLUE7 stays at n=3 per arm.
- Admission gate: each 28L job reserves its worst case Run them serially as earlier jobs close. Suggest maxdur 6 h for x0, x1, z3, x2; GLUE timeout 19800 s fits inside. With that, the last admit
- Critic ref-0220 registered S_split GLUE7 vs L28_e6 s0-s2 GLUE7 (compute-matched about 2x) before any L28_e6 GLUE cell existed; claim-grade.

## v8 earmarked rung-0 pot (10:45Z, 12 Oct)
- Atila 05:35Z (<message>): "run rung 0 only.
- ledger. "pool=rung0" in the note selects it, and a close inherits its admit's pool. The main pot is unchanged with rung 0 excluded.2h After that, z3 at 6h All fit. Order: x1, x2, z3 (experimenter 10:36Z: x2 first, because it completes the claim-grade check, critic ref-0220). It is gated on worst case like the main pot.
