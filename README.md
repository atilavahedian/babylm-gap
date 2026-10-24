# Where the 10M-Word Gap Lives

Code, registration records and per-run results for the paper

**Where the 10M-Word Gap Lives: Noise, Unlearned Structure and Missing Information in BabyLM Data Efficiency**
Atila Vahedian, Independent researcher

arXiv: <to be added>

The paper (`paper/main.tex`, `paper/main.pdf`) measures where the gap between a 10M-word and a 100M-word language model
lives under the BabyLM 2025 strict-small rules, reports ten pre-registered attacks that failed, and one measured point
(a two-model seed ensemble).

## Layout

| path | contents |
|---|---|
| `paper/` | LaTeX source and PDF |
| `data/` | one CSV per table and one for Figure 1: every printed number, its full-precision value, derivation and source record |
| `scripts/build_tables.py` | regenerates `data/*.csv` from the records under `experiments/` |
| `scripts/check_tables.py` | checks every number printed in the paper's tables against `data/*.csv` |
| `src/babylm/` | the training, DEV-scoring and official-evaluation code (package `babylm`) |
| `src/sprint1/` | the model and optimizer code the BabyLM recipe was ported from (package `sprint1`) |
| `experiments/<date>-babylm-*/` | one folder per study: `README.md` (protocol), `SUMMARY.md` (outcome), `configs/` (every job, with its exact command), `code/` (study-specific modules), `docs/` (attack cards and designs), `governance/` (registration log and reviewer rulings), `results/` (per-run records, loss curves, readouts) |
| `ledger/` | the job ledgers: every GPU job admitted and closed |

Pre-registration: every bar was registered before the result it judged by an independent automated reviewer that
wrote no model code. Its rulings, timestamped, are in `experiments/2026-10-09-babylm-official-ruler/governance/critic_log.md`
(the full log), with per-study excerpts in each `governance/` folder. Attack cards are in
`experiments/2026-10-09-babylm-attacks/docs/` and `experiments/2026-10-10-babylm-moonshot-10x/docs/cards/`.

## Tables and figures

Each CSV is regenerated with one command from the repository root; `python3 scripts/build_tables.py` writes all six.

| paper | data file | regenerate | main records |
|---|---|---|---|
| Table 1 (the three parts of the gap) | `data/table1_gap_map.csv` | `python3 scripts/build_tables.py --table tab:map` | `experiments/2026-10-09-babylm-gap-map/results/d1_bias/bias.json` |
| Figure 1 (slices by frequency band) | `data/figure1_bands.csv` | `python3 scripts/build_tables.py --table fig:bands` | same |
| Table 2 (ten failed attacks) | `data/table2_failed_attacks.csv` | `python3 scripts/build_tables.py --table tab:failed` | critic log rows, `experiments/2026-10-09-babylm-attacks/results/gcp_reads/a4_b1/`, `experiments/2026-10-09-babylm-gap-map/results/base_size_m/`, `experiments/2026-10-10-babylm-moonshot-10x/results/f4_menu/`, `experiments/2026-10-12-babylm-rung0-data-wall/results/` |
| Table 3 (two-model ensemble) | `data/table3_ensemble.csv` | `python3 scripts/build_tables.py --table tab:ensemble` | `experiments/2026-10-09-babylm-official-ruler/results/readouts/a5_xfactor_final_*.json` |
| Table 4 (official BabyLM scores) | `data/table4_official.csv` | `python3 scripts/build_tables.py --table tab:official` | `experiments/2026-10-09-babylm-official-ruler/results/readouts/{zs,glue}_*.json` |
| Table 5 (data-wall rung 0) | `data/table5_rung0.csv` | `python3 scripts/build_tables.py --table tab:rung0` | `experiments/2026-10-12-babylm-rung0-data-wall/results/calib_*.json` |

Then `python3 scripts/check_tables.py` checks all 130 printed numbers (it prints "all match").

Loss curves (CSV and PNG) sit beside each run's record as `curves.csv` / `*.png` under `experiments/*/results/`.

## Rerun one arm end to end (size S, 10M words, one seed)

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python -m babylm.data.prepare --download --splits train_10M,dev        # BabyLM 2025 data, sha256-checked
python -m babylm.harness.train --size S --split train_10M --epochs 6 --seed 0 --save --out runs/S_e6_s0
python src/babylm/eval/dev_bpw_twin.py runs/S_e6_s0 <data_dir>/dev --slice full --out twin.json   # DEV nats/word, full and guarded legs
python3 -I src/babylm/tools/calib_control.py --format v2 BASE=<dump> ARM=<dump>   # calibrated score (cal_t), from per-word dumps
```

The two-model ensemble is two such runs at `--epochs 5` with seeds 0 and 1, averaged per token. The exact command of
every run in the paper is in its job file under `experiments/*/configs/`. Official zero-shot and GLUE scores use the
BabyLM 2025 evaluation pipeline at upstream commit bf55c1131e53654c2a87418f5629c26959acd710: see `src/babylm/eval/setup.sh`
and `src/babylm/eval/zero_shot_all.sh`. The GPT-2 baseline recipe needs the public GPT-2 tokenizer:
`bash scripts/fetch_gpt2_tokenizer.sh`.

## Not included

- Per-word DEV loss data: the per-word loss arrays (10.4M words per run) behind Tables 1, 3 and 5, because of their size.
  The scored summaries and calibration reads derived from them are included.
- Per-item predictions of the official BabyLM evaluation beyond 3 runs (the `zs_cpu/` runs under
  `experiments/2026-10-09-babylm-official-ruler/results/`); for the other runs only the task scores are included.
- Loss curves beyond the 25 we hold (CSV and PNG beside each run's record).
- Model weights.
- The code has not been test-run in the export environment. Every Python file compiles, and the table scripts
  (`scripts/build_tables.py`, `scripts/check_tables.py`) were run here; training and evaluation were not.

## Notes on this release

- Internal revision ids are replaced by stable `ref-NNNN` tokens (the same token is the same revision everywhere),
  cloud project and bucket names by placeholders (`gcp-project-a`, `${BUCKET}`), and cost fields are removed from the
  ledgers. The cloud launcher is not included; every job's command is in its config file.

## License

Code: Apache License 2.0 (`LICENSE`). Data, records and the paper's tables: CC BY 4.0. The vendored GPT-BERT code in
`src/babylm/recipes/gptbert/official/` keeps its own license.
