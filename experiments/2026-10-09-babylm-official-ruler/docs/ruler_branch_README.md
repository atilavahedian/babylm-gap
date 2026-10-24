# BabyLM ruler (branch babylm-1)

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
