#!/bin/bash
# VM entry for the sprint-1 port: prep data, train one seed, export every checkpoint to HF, evaluate the final one.
# Called by babylm/jobs/vm_job.sh: run_job.sh SEED OUT_DIR WORKDIR (WORKDIR from babylm/eval/setup.sh)
set -uo pipefail
S=$1; OUT=$2; W=$3; R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PY=$W/venv/bin/python
$PY "$R/prep_data.py" --train_dir "$W/osf/text_data/train_10M/train_10M" --dev_dir "$W/osf/text_data/dev/dev" || exit 1
$PY "$R/train.py" --seed "$S" --save_per_token --out "$OUT/run" || exit 1
$PY "$R/export_hf.py" --run_dir "$OUT/run" --out "$OUT/run/hf" --final_as chck_100M || exit 1
FINAL=$(ls -d "$OUT"/run/hf/chck_100M 2>/dev/null || ls -d "$OUT"/run/hf/* | tail -1)
echo "[sprint1] final export $FINAL"
evaluate "$FINAL" causal "sprint1_s$S" finetune
