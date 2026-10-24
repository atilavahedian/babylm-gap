#!/usr/bin/env bash
# Per-token MLM pseudo-losses at each epoch checkpoint of an AMLM run: full official DEV + the 1M-word train sample.
#   usage: pt_epochs.sh <run_dir> <tokenizer.model> <dev_dir> <train_sample_1M.txt>
# Epoch checkpoints are read from <run_dir>/train.log's total steps; tokenized windows are cached across checkpoints.
set -euo pipefail
OUT="${1:?run_dir}"; TOK="${2:?tok}"; DEV="${3:?dev_dir}"; TR="${4:?train_sample}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TOTAL=$(grep -m1 -oP 'Total steps: \K\d+' "$OUT/train.log")
mkdir -p "$OUT/per_token"; C=$(mktemp -d)
for c in $(ls -d "$OUT"/checkpoint-* | awk -F"checkpoint-" '{print $NF" "$0}' | sort -n | cut -d" " -f2-); do
  s=${c##*-}; E=$(python3 -c "e=$s*10/$TOTAL; print(round(e) if abs(e-round(e))<0.02 and round(e)>=1 else '')")
  [ -z "$E" ] && continue
  python3 "$HERE/per_token.py" "$c" "$TOK" "$OUT/per_token/dev_ep$E" "$DEV"/*.dev --max_words "${PT_DEV_WORDS:-0}" --cache "$C/dev.npz"
  python3 "$HERE/per_token.py" "$c" "$TOK" "$OUT/per_token/train1M_ep$E" "$TR" --max_words "${PT_TRAIN_WORDS:-0}" --cache "$C/train.npz"
  echo "[pt] epoch $E done $(date -u +%T)"
done
rm -rf "$C"
