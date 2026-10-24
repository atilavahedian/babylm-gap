#!/bin/bash
# AMLM job on a GPU VM. Run from the repo root of a checkout of this branch.
#   babylm/winner/amlm/gpu_job.sh JOB OUT_DIR
# JOB:
#   score_pub_hard_decay | score_pub_hard   released leukas/amlm_* checkpoint (pinned sha): full zero-shot + fast suite
#                                           (mlm backend) + GLUE finetune. Certifies the pipeline vs Findings Table 3.
#   train_<arm>_seed<N>                     amlm/vm_job.sh <arm> N (train + per-token saves), then evaluate the final
#                                           checkpoint the same way, then per-token scoring (pt_epochs.sh), so a time
#                                           limit can cut only the per-token tail. arm: hard_decay_off | hard_off | hard_decay_pub | hard_pub
#   eval_ckpt_<name>                        evaluate CKPT=<GCS uri or local dir> (weights + tokenizer) as <name>
# Eval = the ruler runner's setup (babylm/eval, pipeline ref-0243, OSF data sha256-verified; files copied from ref-0410).
set -uo pipefail
JOB=${1:?job}; OUT=${2:?out_dir}
REPO=$(cd "$(dirname "$0")/../../.." && pwd); HERE=$REPO/babylm/winner/amlm
W=${BABYLM_WORK:-/opt/babylm}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); exec > >(tee -a "$OUT/job.log") 2>&1
echo "[job] $JOB start $(date -u +%FT%TZ) commit $(git -C "$REPO" rev-parse HEAD 2>/dev/null || cat "$REPO/GIT_COMMIT")"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
"$REPO/babylm/eval/setup.sh" "$W" || { echo "[job] eval setup failed"; exit 1; }
PY=$W/venv/bin/python
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled   # finetune/trainer.py imports and inits wandb
# Full (gated) EWoK, as in the ruler's babylm/jobs/vm_job.sh: EWOK_ZIP=${BUCKET} or a local path; else fast EWoK.
EWOK_SHA=cea403b8c5b87d5bb62ce423dba5914458dfe3a5e3e74a78361893f910e16db4
if [ -n "${EWOK_ZIP:-}" ] && [ ! -d "$W/eval2025/evaluation_data/full_eval/ewok_filtered" ]; then
  z=$W/ewok.zip
  case "$EWOK_ZIP" in ${BUCKET}) gcloud storage cp "$EWOK_ZIP" "$z" || gsutil cp "$EWOK_ZIP" "$z";; *) cp "$EWOK_ZIP" "$z";; esac
  if [ "$(sha256sum "$z" | cut -d' ' -f1)" = "$EWOK_SHA" ]; then
    $PY - "$z" "$W/eval2025/evaluation_data/full_eval/ewok_filtered" <<'PYEOF'
import sys, zipfile, os
z, dst = zipfile.ZipFile(sys.argv[1]), sys.argv[2]
names = [n for n in z.namelist() if '/full_eval/ewok_filtered/' in n and n.endswith('.jsonl')]
os.makedirs(dst, exist_ok=True)
for n in names: open(os.path.join(dst, os.path.basename(n)), 'wb').write(z.read(n, pwd=b'ewok'))
print('[job] full EWoK files:', len(names))
PYEOF
    ( cd "$W/eval2025/evaluation_data/full_eval/ewok_filtered" && sha256sum *.jsonl ) > "$OUT/ewok_filtered.sha256"
  else echo "[job] EWoK zip sha mismatch; using fast EWoK"; fi
fi
fast_tok() {  # AMLM checkpoints lack tokenizer.json and the eval venv lacks sentencepiece (see add_fast_tok.py)
  python3 -c "import sentencepiece, google.protobuf, transformers" 2>/dev/null || python3 -m pip install -q "transformers==4.54.1" sentencepiece protobuf
  python3 "$HERE/add_fast_tok.py" "$@"
}

evaluate() {  # evaluate MODEL_DIR NAME : full zero-shot + fast suite + GLUE finetune, mlm backend
  local m=$1 name=$2 link=$W/models/$2; mkdir -p "$W/models"; ln -sfn "$m" "$link"
  export BABYLM_TIE_LOG=$OUT/ties_${name}_mlm.jsonl
  ( cd "$W/eval2025" && PY=$PY "$REPO/babylm/eval/zero_shot_all.sh" "$link" mlm ) > "$OUT/eval_zero_shot_${name}.log" 2>&1
  BABYLM_TIE_LOG=$OUT/ties_fast_${name}_mlm.jsonl bash -c "cd '$W/eval2025' && PATH=$W/venv/bin:\$PATH bash eval_zero_shot_fast.sh '$link' fast mlm" > "$OUT/eval_fast_${name}.log" 2>&1
  ( cd "$W/eval2025" && PATH=$W/venv/bin:$PATH bash eval_finetuning.sh "$link" ) > "$OUT/eval_finetune_${name}.log" 2>&1
  mkdir -p "$OUT/eval"; cp -r "$W/eval2025/results/$name" "$OUT/eval/" 2>/dev/null
  echo "[job] evaluated $name $(date -u +%T)"
}

t0=$(date +%s)
case "$JOB" in
  score_pub_hard_decay|score_pub_hard)
    n=${JOB#score_pub_}; m=amlm_$n
    rev=$($PY -c "import json;print(json.load(open('$HERE/PIN.json'))['released_checkpoints']['leukas/$m'])")
    $PY -c "from huggingface_hub import snapshot_download as s; print(s('leukas/$m', revision='$rev', local_dir='$W/hf/$m'))"
    echo "$rev" > "$OUT/published_revision.txt"; ( cd "$W/hf/$m" && sha256sum *.safetensors *.json *.model ) > "$OUT/published_files.sha256"
    evaluate "$W/hf/$m" "pub_amlm_$n" ;;
  train_*_seed*)
    arm=${JOB#train_}; seed=${arm##*_seed}; arm=${arm%_seed*}
    PT=$W/pt_stage; rm -rf "$PT"
    PT_STAGE=$PT OFFICIAL_ZIPS=$W/osf/text_data bash "$HERE/vm_job.sh" "$arm" "$seed" "$OUT/run" || { echo "[job] training failed"; exit 1; }
    last=$(ls -d "$OUT"/run/checkpoint-* | awk -F"checkpoint-" '{print $NF" "$0}' | sort -n | tail -1 | cut -d" " -f2-)
    # Self-contained model dir for the pipeline: final weights + tokenizer (vm_job saves both per checkpoint).
    fast_tok "$last"; evaluate "$last" "amlm_${arm}_s${seed}"
    bash "$HERE/pt_epochs.sh" "$OUT/run" "$PT/tok.model" "$PT/dev" "$PT/train_sample_1M.txt" ;;
  eval_ckpt_*)
    name=${JOB#eval_ckpt_}; src=${CKPT:?set CKPT}
    if [[ $src == ${BUCKET} ]]; then mkdir -p "$W/ckpt/$name"; gsutil -m cp -r "${src%/}/*" "$W/ckpt/$name/"; src=$W/ckpt/$name; fi
    fast_tok "$src"; ( cd "$src" && sha256sum * ) > "$OUT/ckpt_files.sha256"; echo "${CKPT}" > "$OUT/ckpt_source.txt"
    evaluate "$src" "$name" ;;
  *) echo "unknown job $JOB"; exit 2 ;;
esac
echo "[job] $JOB done in $(( $(date +%s) - t0 ))s"
