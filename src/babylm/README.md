# common/babylm: the shared BabyLM package

This is the code shared by every BabyLM experiment in this repo, assembled into one tree on 2026-10-13. The package
name is still `babylm`, so imports such as `babylm.harness.train` work unchanged. No single source branch held all
of `babylm/`. For each file found on several branches, this tree takes the version from the branch whose last
commit touching that file is newest. The full per-file record is in "Provenance" below.

## What is here

| dir | what | main source (newest commit wins per file) |
|---|---|---|
| `harness/` | trainer (`train.py`, frozen sprint-1 recipe on BabyLM), streams, levers, A3 BPE-dropout, A4 second LR cycle (`cycle.py`), HF export + template, job runner, `score_trial.sh`; library modules for A1 `self_distill.py`, A2 `input_noise.py` and the attack stacker `stack.py` | ouf5ji; `levers.py` from wx2meb (strict superset of ouf5ji's); `self_distill.py`, `input_noise.py`, `stack.py` from taxy02; `hf_template/modeling_hinton_pe.py` from sc490a |
| `models/` | baseline_reg, looped, switchable, particles (moonshot PE) | ouf5ji; `particles.py` from sc490a |
| `data/` | corpus rules, `prepare.py` (sha256-checked against `manifest.json`), the committed 16k tokenizer `tok16k_10m/tokenizer.json`, DEV-sub/TRAIN-sub doc-index lists, OSF listing + `fetch.py`, rare-word `focus_mask.py`. **No corpus text.** | ouf5ji, ij746t, o6j641, wx2meb |
| `eval/` | official-pipeline wrappers: `setup.sh`, `zero_shot_all.sh`, `official_readout.py`, `dev_bpw.py` (registered DEV metric), `dev_bpw_twin.py` (blank-line twin, metric v2), GLUE last-token adapters (`adapters/`), ensemble/readout tools | ij746t, o6j641; `dev_bpw_twin.py` from sc490a/o3stxz |
| `recipes/` | official GPT-2 and GPT-BERT baseline recipes (GPT-BERT vendored with its license), the sprint-1 port, rotary fix | ij746t |
| `instruments/` | bottleneck instruments: frozen DEV/train sample indices, slices, signatures, prize, stall, n-gram, headcheck, vendored GPT-BERT modeling | o3stxz |
| `analysis/` | post-run reports, kNN diagnostics, per-word dumps, snapshots; `rare_words/` analysis | ouf5ji; `rare_words/` from wx2meb |
| `configs/` | job-list generators (`make_*_jobs.py`, attacks included because tests import them) and the shared base/GPU/frozen-port job lists, `base_sweep_log.md`, `a5_yardstick_final.json`, `gcp/lead_p2*.json` | ouf5ji, ij746t, taxy02, gd7wjc, wx2meb |
| `tools/` | calibration control (`calib_control.py`), unseen-character list, temperature tools, curve plot | ouf5ji, sc490a, ij746t |
| `ledger/` | append-only hash-chained cost ledger of the ruler (`runs.jsonl`, `ledger.py`) | ij746t |
| `jobs/vm_job.sh` | the ruler's GPU VM job script (score_published, certify, gptbert, lastpool GLUE, ...) | ij746t |
| `gcp/` | GPU executor: `launch.sh`, `startup.sh`, `vm_runtime.sh`, `auth.sh` (reads a key from `GCP_SA_KEY_JSON`, none committed), `ledger.py` + `runs.jsonl` (compute), `HANDOFF.md` (executor log), `rt28/`, `dw/` helpers | byi6gg (supersedes rmb0k3, djsgaz) |
| `a4/t2.py`, `moonshot/kd.py` | the two experiment modules that `harness/train.py` imports at top level (see below) | ouf5ji |
| `tests/` | CPU tests for the shared code, plus `test_babylm_instruments.py` and `test_dev_bpw_twin.py` (moved from the repo-root `tests/` of o3stxz) | ouf5ji, o3stxz |

Moved out of this package on purpose:
- Experiment sub-packages `babylm/a4` (all except `t2.py`), `babylm/moonshot` (all except `kd.py`) and `babylm/winner`
  live in their experiment folders under `code/babylm/...`.
- Experiment job lists: A1-A4 and A3g job JSONs and `configs/gcp/a4_*.json` are in
  `experiments/2026-10-09-babylm-attacks/configs/`. Moonshot (`jobs_moonshot_*.json`, `gcp/rt28/jobs_moonshot_28l_retrain.json`)
  and rare-word (`jobs_a5_S.json`) job lists belong to those experiment folders.
- The executor's 118 job JSONs and 198 reads (`babylm/gcp/jobs`, `babylm/gcp/reads`) are split across the experiment
  folders' `configs/gcp_jobs/` and `results/`.
- Tests that need an experiment package or an unmerged trainer: `test_a4.py`, `test_stack.py`, `test_self_distill.py`,
  `test_input_noise.py`, `test_make_stack_jobs.py` (attacks); `test_moonshot_kd.py`, `test_particles.py`,
  `test_bias_map.py` (moonshot); `test_focus.py` (rare words).

## How to run

The code assumes the current directory contains `babylm/`. Scripts call `babylm/eval/dev_bpw.py` by path, and runs
write `babylm/runs/` and `data/babylm/`. It also imports `sprint1` (from `common/sprint1`). So run from `common/`:

    cd common
    export PYTHONPATH=$PWD                      # babylm + sprint1
    python -m babylm.data.prepare --download --splits train_10M,dev
    python -m babylm.harness.train --size S --split train_10M --epochs 10 --seed 0 --save --out babylm/runs/S_e10_s0
    python -m pytest -q babylm/tests            # needs torch, tokenizers, transformers

To add an experiment's code, append its folder: `export PYTHONPATH=$PWD:$PWD/../experiments/<exp>/code`.
`babylm/__init__.py`, `babylm/a4/__init__.py` and `babylm/moonshot/__init__.py` call `pkgutil.extend_path`, so
`experiments/<exp>/code/babylm/a4/*.py` (for example) is importable as `babylm.a4.*`. An experiment folder does not
need its own `babylm/__init__.py`.

**Overlays (unmerged trainers).** Some attacks and side threads changed `harness/train.py` on branches that were
never merged back into ouf5ji: A1 (2szb4k), A2 (gd7wjc), the stacker (taxy02), rare-word focus (wx2meb) and moonshot
particles/PE (sc490a/mfsfhs). This tree keeps their library modules, but its `train.py` (ouf5ji) does not expose
their flags (`--sd_mode`, `--noise_mode`, `--focus_plan`, `--model particles`). To run one, put that branch's
differing files first on the path as an overlay. Each overlay `__init__.py` uses extend_path, so everything else
falls through to `common/`. The attack overlays are in `experiments/2026-10-09-babylm-attacks/code/overlays/`.

    PYTHONPATH=$PWD/../experiments/2026-10-09-babylm-attacks/code/overlays/stack_taxy02:$PWD:$PWD/../experiments/2026-10-09-babylm-attacks/code

`eval/a5_yardstick_fixedT.py`, `gcp/rt28/rt_member.sh`, `gcp/dw/make_dw_r0_jobs.py`, `tools/calib_control.py` (v2
input) and `configs/make_a4_t2_jobs.py` (job text) refer to `babylm.moonshot.*` modules that live in the moonshot
experiment folder.

## Verification (2026-10-13, this container: Python 3.13, numpy, no torch/tokenizers/pytest)

- `python3 -I -m py_compile` passes on every `.py` here.
- Pure-python tests run with a stub runner: `test_job_configs` (1/1 pass), `test_dev_bpw_twin` (5 pass; the torch test
  was skipped), `test_data` (2 pass; the others need tokenizers or fixtures). Everything else imports torch and was
  not run.
- Exact reproduction of a past run uses the commit pinned in its job file (archive tags keep every original commit).

## Notes and caveats

- `data/fetch.py`, `jobs/vm_job.sh` and `gcp/rt28/zs_official.sh` contain the EWoK zip passwords that BabyLM and EWoK
  publish. These are anti-scraping passwords, not credentials. No EWoK data is in this repo. The full EWoK zip is
  read at run time from the private bucket `${BUCKET}/ewok/`. Never copy
  it into git.
- `recipes/gptbert/official/` is vendored upstream GPT-BERT code under its own license (`LICENSE_gpt-bert`).
- `gcp/runs.jsonl` and `ledger/runs.jsonl` are compute records. They are kept because experiment reports cite them.

## Provenance

Branch tips used (archive tags `archive/<branch>` keep the originals): ouf5ji ref-0008, ij746t ref-0177,
o3stxz ref-0314, o6j641 ref-0406, byi6gg ref-0009, babylm-attack-stacker-taxy02 ref-0007, babylm-10x-ultracode-sc490a
ref-0377, wip-branch ref-0430, babylm-attack-input-noise-gd7wjc ref-0004, project-thread-mfsfhs
ref-0414. A3 (edm4oi ref-0005) and A4 (1mvwf5 ref-0006) are ancestors of ouf5ji. rmb0k3 and djsgaz are superseded by
byi6gg. The original harness and ruler READMEs follow, verbatim. Their "run from the repo root with `PYTHONPATH=.`"
now means "from `common/`".

---

## Original harness README (wip-branch:babylm/README.md)

### BabyLM harness: the frozen sprint-1 recipe on BabyLM 2025

Model, optimizer, LR schedule imported unchanged from `sprint1/`; `configs/frozen_port.json` lists what changed.
Python 3.12+, `torch==2.7.0 numpy tokenizers==0.21.1 transformers==4.51.3 safetensors matplotlib pytest` (+ `datasets==3.6.0
scipy tqdm aiohttp` for the official pipeline). Run everything from the repo root with `PYTHONPATH=.`.

**Prepare** (md5-verified OSF zips → `data/babylm/{zips,raw,bin,ruler}`; raw dev text = `data/babylm/raw/dev/*.dev`;
committed tokenizer; every output's sha256 is checked against `data/manifest.json`):

    python -m babylm.data.prepare --download                          # all splits, ~11 min, 2.6 GB of bins
    python -m babylm.data.prepare --download --splits train_10M,dev   # laptop: 10M training + DEV only

**Train** (`--device auto` = cuda > mps > cpu; run dir `babylm/runs/<name>`: `train_log.csv`, `curves.csv`,
`pt/pt_{dev,train}_<n>M.npy`, `pt/pt_devfull_final.npy`, `checkpoints/chck_<n>M.pt`, `resume.pt`, `result.json`):

    python -m babylm.harness.train --size small --split train_10M --epochs 10 --seed 0 --compile --save \
        --resume auto --extra softcap=15 unet=true --out babylm/runs/port_small_10M_e10_s0
    python -m babylm.harness.run_jobs babylm/configs/jobs_a_frozen_port.json            # GPU job lists (a)-(c)
    python -m babylm.harness.run_jobs babylm/configs/jobs_a_frozen_port.json --device mps --only port_S_10M_e10_s0

**Export** (one checkpoint → HF dir; a whole run → per-revision dirs plus an offline hub cache with `chck_*M` refs):

    python -m babylm.harness.export_hf --ckpt babylm/runs/R/checkpoints/chck_100M.pt --out exports/R_final --verify
    python -m babylm.harness.export_hf --run_dir babylm/runs/R --out exports --name R --verify

**Evaluate** with the official pipeline (`git clone https://github.com/babylm/evaluation-pipeline-2025`, commit
`bf55c1131e53654c2a87418f5629c26959acd710`, eval data from OSF `ryjfm`), from the pipeline directory:

    bash eval_zero_shot.sh /abs/exports/R_final causal                                   # final model, full eval
    HF_HUB_OFFLINE=1 HF_HUB_CACHE=/abs/exports/hub bash eval_zero_shot_fast_all_revisions.sh hinton/R causal strict-small
    HF_HUB_OFFLINE=1 HF_HUB_CACHE=/abs/exports/hub bash eval_aoa.sh hinton/R causal strict-small
    python -m babylm.harness.pipeline_check --export exports/R_final --ckpt babylm/runs/R/checkpoints/chck_100M.pt \
        --pipeline <pipeline> --blimp_dir <pipeline>/evaluation_data/fast_eval/blimp_fast --run_official

The pipeline loads checkpoints with `from_pretrained(name, revision="chck_XM")`, which ignores `revision` for a
local directory, hence the hub-cache layout. GLUE: `eval_finetuning.sh` passes neither `--causal` nor
`--take_final`, so a causal model is pooled at position 0 (the BOS, which sees nothing); `--take_final` indexes the
wrong dimension. Causal GLUE scores from the unmodified script are not meaningful (applies to every causal entry).

**Plot**: `python -m babylm.harness.plot_curves babylm/runs/R`. **Test**: `python -m pytest -q babylm/tests` (~2 min).

#### Facts and conventions
- Words = `str.split()` items (`wc -w` in the POSIX locale gives BabyLM's quoted 9,946,560; str.split gives 9,953,773).
  words_seen counts word-initial target tokens; milestones 1M..10M, 20M..100M, 200M..1B; the first unreached one
  fires at the end if ≥99% reached (10 epochs of train_10M = 99.5M words → `chck_100M`).
- Documents = blank-line blocks, each line normalised by `corpus.normalize_line` (" ".join(line.split()));
  switchboard = 2000-line pieces; BOS `<|endoftext|>` (id 0) before each. Tokens/word 1.603 (train_10M), 1.588 (dev).
  DEV-sub/TRAIN-sub: ~1M words of whole docs (`data/subsets/`). Eval text is not normalised by the HF tokenizer.
- `train_20M/30M/50M`: train_10M + whole train_100M docs sharing no line position with train_10M, nested.
- FINAL (`test`) loads only with `train.py --final` (clean tree, warning, `FINAL_SPLIT_USED`).

#### Memory (fp32, seq 1024, `python -m babylm.harness.mem_estimate --math_sdpa`, CPU-measured)
Peak ≈ fixed + micro × per-sequence: small 0.22 + 0.87 GiB, S 0.24 + 1.44 GiB, M 0.45 + 2.77 GiB (math attention
stores [B,H,T,T]; the CPU flash kernel needs ~1/3 less). Default micro on MPS: small 8 (~7.2 GiB), S 4 (~6.0),
M 2 (~6.0); batch stays 65,536 tokens via accumulation. CUDA defaults 16/16/8, bf16, OOM halves micro.

**Untested on MPS (no Mac here)** `--device mps`: fp32 default, no compile, `PYTORCH_ENABLE_MPS_FALLBACK` must be unset (refused, so unsupported ops error
instead of silently using the CPU), `torch.mps.synchronize` timing, MPS RNG in `resume.pt`, MPS OOM halving, `--amp_dtype
bf16|fp16`. First run `mem_estimate --device mps` and a `--steps 20` run.

---

## Original ruler README (wip-branch:babylm/README.md)

### BabyLM ruler (branch babylm-1)

Official BabyLM 2025 data + evaluation, shared by every BabyLM thread. One pipeline, pinned:

- **Data**: OSF node `ryjfm` (BabyLM_2025). `babylm/data/osf_listing.txt` lists all 183 files with OSF sha256;
  `python3 babylm/data/fetch.py DEST` downloads, verifies every hash and unzips (train_10M 9,946,560 words by `wc -w`,
  train_100M, dev, test; fast EWoK unzipped with the password published in the eval README).
- **Eval**: github.com/babylm/evaluation-pipeline-2025 @ `bf55c1131e53654c2a87418f5629c26959acd710`;
  `babylm/eval/setup.sh WORKDIR` clones it at that commit, fetches data, builds a venv (torch 2.7.0, pinned
  requirements). `babylm/eval/zero_shot_all.sh MODEL BACKEND` runs the full zero-shot suite (BLiMP, supplement, EWoK,
  entity tracking, WUG adj/past, reading, COMPS). Finetuning: the pipeline's own `eval_finetuning.sh`.
  - Full EWoK is a gated HF dataset (ewok-core/ewok-core-1.0); until a HF token is provided, EWoK uses the official
    fast set. AoA needs per-checkpoint hub revisions and is not run yet.
- **Recipes**: `babylm/recipes/gpt2` (official GPT-2 baseline), `gptbert` (official GPT-BERT baseline),
  `sprint1_port` (our frozen sprint-1 recipe on BabyLM data).
- **VM jobs**: `babylm/jobs/vm_job.sh JOB OUT_DIR` (JOB = score_published | gpt2_seedN | gptbert_causal_seedN |
  sprint1_seedN). Every training run writes `curve.csv` + `curve.png`.
- **Ledger**: `python -m babylm.ledger.ledger status`
- Plot any curve CSV: `python babylm/tools/plot_curve.py run/curve.csv`.
