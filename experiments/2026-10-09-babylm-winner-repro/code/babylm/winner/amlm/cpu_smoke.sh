#!/usr/bin/env bash
# CPU smoke test: pinned AMLM code, patched only for device (cuda:0 -> cpu) and the CUDA-only LAMB import,
# 100 training lines, 1 epoch. Proves the pipeline runs end to end; says nothing about scores.
set -euo pipefail
WORK="${1:?usage: cpu_smoke.sh <work_dir> [hard_decay|hard]}"; ARM="${2:-hard_decay}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -d "$WORK/code" ] || bash "$HERE/fetch.sh" "$WORK/code"
rm -rf "$WORK/cpu" && cp -r "$WORK/code" "$WORK/cpu" && rm -rf "$WORK/cpu/.git"
sed -i -e 's/"cuda:0"/"cpu"/g' -e 's/^from bitsandbytes.optim import LAMB/LAMB = None/' \
       -e 's/torch.cuda.memory_allocated()/0/g; s/torch.cuda.memory_reserved()/0/g' "$WORK/cpu/train_mask.py"
EXTRA=(--mlm_prob 0.4 --mask_decay 0.25); [ "$ARM" = hard ] && EXTRA=(--mlm_prob 0.15)
cd "$WORK/cpu"
python3 train_mask.py --train_data data/bb24.train --valid_data data/bb25_small.dev --tokenizer tokenizers/bb24.model \
  --output_path "$WORK/out_$ARM" --mask_update_steps 2 --logging_steps 1 --intermediate_size 1280 --hidden_size 384 \
  --max_seq_len 0:64 --seed 0 --debug --epochs 1 --batch_size 8 --cpus 1 --eval_steps 2 --save_steps 1000 "${EXTRA[@]}" \
  2>&1 | tee "$WORK/smoke_$ARM.log"
python3 "$HERE/../curves.py" "$WORK/smoke_$ARM.log" "$WORK/smoke_$ARM" --epochs 1 --title "AMLM $ARM CPU smoke"
