#!/bin/bash
# VM entry for the GPT-BERT 2025 strict-small port: prep data, train one seed, evaluate hf/final (causal + mntp + GLUE)
# and hf/final_ema (zero-shot causal). Called by babylm/jobs/vm_job.sh: run_job.sh SEED OUT_DIR WORKDIR
# Env: GB_NUM  causal share numerator over 16 (15 causal-focus [default], 8 mixed, 1 masked-focus)
#      GB_TRAIN train_10M [default] | train_100M ; GB_EPOCHS 10 [default] ; GB_TAG name prefix (default gptbert)
#      GB_PUBREF=1 also score the published checkpoint with --eval_only (same dev subset, curve, dumps)
set -uo pipefail
S=$1; OUT=$2; W=$3; R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PY=$W/venv/bin/python
NUM=${GB_NUM:-15}; TR=${GB_TRAIN:-train_10M}; EP=${GB_EPOCHS:-10}; TAG=${GB_TAG:-gptbert}
DEVTXT=$W/osf/text_data/dev/dev
SPLITS=$TR,dev; [ "$TR" = train_100M ] && SPLITS=train_100M,dev,train_10M   # train_10M = fixed 1M-word sample source
$PY "$R/prepare_data.py" --data_root "$W/osf/text_data" --splits "$SPLITS" || exit 1
if [ -n "${GB_PUBREF:-}" ]; then
  P=$W/hf/babylm-baseline-10m-gpt-bert-causal-focus
  $PY -c "from huggingface_hub import snapshot_download as s; s('BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus', local_dir='$P')"
  $PY "$R/train.py" --eval_only --init_from_hf "$P" --out_dir "$OUT/published_ref" --dev_text_dir "$DEVTXT" || echo "[gptbert] pubref failed"
fi
$PY "$R/train.py" --seed "$S" --hybrid_numerator "$NUM" --train_dir "$TR" --epochs "$EP" --out_dir "$OUT/run" \
  --dev_text_dir "$DEVTXT" --resume_states last ${GB_EXTRA:-} || exit 1
N=${TAG}_s$S
evaluate "$OUT/run/hf/final" causal "$N" finetune
evaluate "$OUT/run/hf/final" mntp "$N"
evaluate "$OUT/run/hf/final_ema" causal "${N}_ema"
