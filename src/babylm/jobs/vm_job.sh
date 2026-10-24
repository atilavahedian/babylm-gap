#!/bin/bash
# One BabyLM ruler job on a GPU VM. Run from the repo root of a checkout/tarball of this branch.
#   babylm/jobs/vm_job.sh JOB OUT_DIR
# JOB:
#   score_pub_<name>         certify the pipeline on a published strict-small checkpoint, full + fast zero-shot,
#                            GLUE finetune. name: gpt2 | gptbert_causal | gptbert_mixed | gptbert_masked
#                            (GPT-BERT is scored with both the causal and the mntp backend)
#   score_hf                 official eval (full zero-shot + GLUE, causal) of any HF dir: env HF_URI, HF_NAME [FMT_JSON];
#                            (GLUE resumes per task from $OUT/eval or GLUE_RESUME_FROM=<earlier OUT_DIR copy>)
#                            a Hinton export named *_lastpool gets last NON-PAD token pooling (make_hinton_lastpool, unit
#                            test first); a name containing _untrained scores a fresh random init of the same export
#                            a name ending in _lastpool gets GLUE only (after smoke_hf --pool last)
#   score_mix                K=2 mixture (critic 44552922): env HF_URI = 2 causal Hinton member exports, HF_NAME;
#                            zero-shot on the one-model per-token mixture, GLUE = members fine-tuned (last-token
#                            pooling, seeds 42/43), probabilities averaged -> <HF_NAME>_ens2. GLUE_ENS=2 on any job
#                            gives each scored model its matched 2-fine-tune-seed ensemble <name>_ens2
#                            Binding guard (critic 44552922): with TWIN_REPO (a checkout holding the moonshot's
#                            babylm/eval/dev_bpw_twin.py + its dev_bpw.py) the export's twin DEV nats/word at T=1 is
#                            computed; with MIX_REF_NPW (the read file's raw mixture DEV) |diff| <= 1e-4 -> MIX_GUARD
#                            PASS, else FAIL + VOID_<name> (official leg void). No ref -> NO_REF (not claim-grade).
#                            GLUE_PREV=<earlier score_mix OUT> [GLUE_RERUN=multirc] [GLUE_SPOT=m0:rte]: GLUE fix-up of a
#                            concurrent run (critic ref-0191): reuse its zero-shot/guard and identical tasks, rerun the rest
#   READING_PS=1 (score_hf, score_mix): ONLY the amended Reading score (critic 81702719): the model dir with its tokenizer
#                            set to add_prefix_space=True (weights and every other file unchanged), official reading
#                            task only -> eval/<name>_ps; alignment check of the copy first (reading_align_<name>_ps.json)
#   ZS_ONLY=1 (score_hf, score_mix): zero-shot only, no GLUE (CPU runs); DEV_BPW=0 skips the registered dev_bpw passes
#   BOS_ENS=0 with GLUE_ENS=2: only *_lastpool names get the 2-seed ensemble (BOS GLUE of a causal model is a side number)
#   ftverify_hf              the same identity check on a causal Hinton export: env HF_URI, HF_NAME=*_lastpool (adapter)
#   ftverify_pub_<name>      GLUE byte-identity: concurrent vs official sequential script on the published checkpoint
#                            (FT_PARALLEL=0: sequential official GLUE only, identity NOT_TESTED)
#                            (gpt2 | gptbert_causal); reuses a concurrent run an earlier job left on this VM
#   entity_v1_pub_gptbert_causal  published GPT-BERT causal on the pre-15-May-2025 OSF entity files (diagnosis)
#   score_controls           chance floor and failure check: untrained GPT-2 and GPT-BERT (random init from the
#                            published configs, seed 0) and a NaN-weight GPT-BERT, full zero-shot
#   gpt2_seed<N>             train the official GPT-2 baseline (babylm/recipes/gpt2) with seed N, then evaluate it
#   gptbert_causal_seed<N>   train the official GPT-BERT causal-focus baseline (babylm/recipes/gptbert), evaluate
#                            (seed 0 also scores the published checkpoint with the trainer's own dev subset and dumps)
#   gptbert_mixed_seed<N>    same, mixed (8/16 causal)
#   gptbert_100M_e10 / gptbert_100M_e1   prize runs: causal-focus on train_100M, 10 epochs / 1 epoch, seed 0
#   sprint1_seed<N>          train the sprint-1 port (babylm/recipes/sprint1_port), evaluate
# Writes everything under OUT_DIR (run dir, curve.csv/png, eval results, logs, job_meta.json). Needs: NVIDIA
# driver, python3 with venv, network to github.com, osf.io, huggingface.co, PyPI.
set -uo pipefail
JOB=${1:?job}; OUT=${2:?out_dir}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)   # absolute: evaluate() cd's into the pipeline
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${BABYLM_WORK:-/opt/babylm}; mkdir -p "$W/models"   # GLUE-only jobs (lastpool, GLUE_PREV) skip evaluate(), which used to create it
mkdir -p "$OUT"; exec > >(tee -a "$OUT/job.log") 2>&1
echo "[job] $JOB start $(date -u +%FT%TZ) commit $(cat "$REPO/GIT_COMMIT" 2>/dev/null || git -C "$REPO" rev-parse HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

[ -n "${SKIP_SETUP:-}" ] || "$REPO/babylm/eval/setup.sh" "$W" || { echo "[job] setup failed"; exit 1; }   # SKIP_SETUP: local tests only
PY=$W/venv/bin/python
$PY -c "import matplotlib" 2>/dev/null || $W/venv/bin/pip install -q matplotlib==3.10.3
$PY - <<EOF > "$OUT/job_meta.json"
import json, torch, transformers, platform
print(json.dumps({"job": "$JOB", "torch": torch.__version__, "cuda": torch.version.cuda,
  "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
  "transformers": transformers.__version__, "python": platform.python_version()}))
EOF
cat "$OUT/job_meta.json"
# the certified official pipeline runs on the pinned transformers (requirements_eval.txt); anything else is refused
TF_PIN=$(sed -n 's/^transformers==//p' "$REPO/babylm/eval/requirements_eval.txt")
TF_HAVE=$($PY -c "import transformers; print(transformers.__version__)")
[ "$TF_HAVE" = "$TF_PIN" ] || { echo "[job] transformers $TF_HAVE != pinned $TF_PIN; refusing to score"; exit 3; }
TRAIN=$W/osf/text_data/train_10M/train_10M; DEV=$W/osf/text_data/dev/dev
# GLUE finetuning: the official eval_finetuning.sh runs its 7 tasks one after another and leaves an A100 mostly idle
# (one 30M-param model took >2h40m). FT_PARALLEL=1 (default) runs the SAME script with each task launched in the
# background and a final wait: identical commands, arguments, seeds and output paths (results/<model>/.../<task>).
# The generated copy may differ from the official file only by those additions (checked here).
FT_SCRIPT=eval_finetuning.sh
# FT_CONC=N (experimenter 11 Oct, after concurrent OOMs): at most N tasks at a time; a throttle line (wait for a free slot) is
# inserted before each task's python call, nothing else changes.
if [ "${FT_PARALLEL:-1}" = 1 ]; then
  if [ -n "${FT_CONC:-}" ]; then
    sed -e 's/--verbose$/--verbose \&/' -e "s/^\( *\)python -m evaluation_pipeline.finetune.run/\1while [ \"\$(jobs -rp | wc -l)\" -ge $FT_CONC ]; do wait -n; done\n&/" \
      "$W/eval2025/eval_finetuning.sh" > "$W/eval2025/eval_finetuning_parallel.sh"
  else sed 's/--verbose$/--verbose \&/' "$W/eval2025/eval_finetuning.sh" > "$W/eval2025/eval_finetuning_parallel.sh"; fi
  echo wait >> "$W/eval2025/eval_finetuning_parallel.sh"
  nd=$(diff "$W/eval2025/eval_finetuning.sh" "$W/eval2025/eval_finetuning_parallel.sh" | grep -c '^>')
  nv=$(grep -c -- '--verbose$' "$W/eval2025/eval_finetuning.sh")
  np=$(grep -c 'python -m evaluation_pipeline.finetune.run' "$W/eval2025/eval_finetuning.sh")
  want=$((nv + 1)); [ -n "${FT_CONC:-}" ] && want=$((nv + 1 + np))
  [ "$nd" = $want ] && FT_SCRIPT=eval_finetuning_parallel.sh || echo "[job] parallel finetune script check failed ($nd vs $want); using the sequential official script"
fi
echo "[job] GLUE finetune script: $FT_SCRIPT${FT_CONC:+ (at most $FT_CONC tasks at a time)}"
# GLUE_ENS=2: every GLUE run also dumps validation probabilities and each scored model gets its matched 2-fine-tune-seed
# ensemble <name>_ens2 (critic 44552922); score_mix always ensembles its members.
[ "${GLUE_ENS:-}" = 2 ] && export GLUE_DUMP_PROBS=1
# Full (gated) EWoK: EWOK_ZIP=${BUCKET} or a local path to ewok-core-1.0_ref-0155.zip (private bucket, staged by the executor).
# Without it zero_shot_all.sh falls back to fast EWoK (scores marked provisional).
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
    [ "$(ls "$W/eval2025/evaluation_data/full_eval/ewok_filtered" 2>/dev/null | wc -l)" -ge 11 ] || \
      unzip -o -j -P ewok "$z" '*/full_eval/ewok_filtered/*.jsonl' -d "$W/eval2025/evaluation_data/full_eval/ewok_filtered"
    ( cd "$W/eval2025/evaluation_data/full_eval/ewok_filtered" && sha256sum *.jsonl ) > "$OUT/ewok_filtered.sha256"
  else echo "[job] EWoK zip sha mismatch; using fast EWoK"; fi
fi
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled   # finetune/trainer.py imports and inits wandb

evaluate() {  # evaluate MODEL_DIR BACKEND NAME [finetune] ; FAST=1 also runs the fast suite
  local m=$1 b=$2 name=$3 ft=${4:-}
  local link=$W/models/$name; mkdir -p "$W/models"; ln -sfn "$m" "$link"
  export BABYLM_TIE_LOG=$OUT/ties_${name}_${b}.jsonl   # argmax ties per batch (patch_ties.py)
  ( cd "$W/eval2025" && PY=$PY "$REPO/babylm/eval/zero_shot_all.sh" "$link" "$b" ) > "$OUT/eval_zero_shot_${name}_${b}.log" 2>&1
  if [ -n "${FAST:-}" ]; then
    BABYLM_TIE_LOG=$OUT/ties_fast_${name}_${b}.jsonl bash -c "cd '$W/eval2025' && PATH=$W/venv/bin:\$PATH bash eval_zero_shot_fast.sh '$link' fast '$b'" > "$OUT/eval_fast_${name}_${b}.log" 2>&1
  fi
  mkdir -p "$OUT/eval"; cp -r "$W/eval2025/results/$name" "$OUT/eval/" 2>/dev/null   # zero-shot results out before GLUE
  if [ "$b" = causal ] && [ "${DEV_BPW:-1}" != 0 ] && [ ! -f "$OUT/dev_bpw_${name}.json" ]; then   # registered held-out metric (dev_bpw.py)
    # headline = the model's own training format (critic ref-0051), from babylm/recipes/<r>/format.json;
    # secondary = the common format (newline + collapse) if different, else raw whitespace
    local r=gptbert; [[ $name == *gpt2* ]] && r=gpt2; [[ $name == sprint1* ]] && r=sprint1_port
    local fj=${FMT_JSON:-$REPO/babylm/recipes/$r/format.json}
    local lj=$($PY -c "import json;print(json.load(open('$fj')).get('line_join','newline'))" 2>/dev/null || echo newline)
    local ws=$($PY -c "import json;print(json.load(open('$fj')).get('ws','collapse'))" 2>/dev/null || echo collapse)
    local ts=$($PY -c "import json;print(json.load(open('$fj')).get('tab_space',''))" 2>/dev/null)
    $PY "$REPO/babylm/eval/dev_bpw.py" "$link" "$DEV" --slice full --ws $ws --line_join $lj --tab_space "$ts" --out "$OUT/dev_bpw_${name}.json" > "$OUT/dev_bpw_${name}.log" 2>&1
    local pts=$($PY -c "import json;print(json.load(open('$fj')).get('probe_tab_space',''))" 2>/dev/null)
    if [ -n "$pts" ]; then   # second labelled number: probe format, data copy unverified (critic ref-0056)
      $PY "$REPO/babylm/eval/dev_bpw.py" "$link" "$DEV" --slice full --ws $ws --line_join $lj --tab_space "$pts" --out "$OUT/dev_bpw_${name}_probefmt.json" > "$OUT/dev_bpw_${name}_probefmt.log" 2>&1
    fi
    if [ "$lj/$ws/$ts" != newline/collapse/ ]; then
      $PY "$REPO/babylm/eval/dev_bpw.py" "$link" "$DEV" --slice full --out "$OUT/dev_bpw_${name}_common.json" > "$OUT/dev_bpw_${name}_common.log" 2>&1
    else
      $PY "$REPO/babylm/eval/dev_bpw.py" "$link" "$DEV" --slice full --ws raw --out "$OUT/dev_bpw_${name}_wsraw.json" > "$OUT/dev_bpw_${name}_wsraw.log" 2>&1
    fi
  fi
  if [ -n "$ft" ]; then   # GLUE last: by far the longest step, so the cheap metrics above survive a cut
    echo "[job] GLUE finetune $name start $(date -u +%T)"
    run_glue "$link" "$OUT/eval_finetune_${name}.log"
    [ "${BOS_ENS:-1}" = 0 ] || glue_seed2 "$name"   # BOS_ENS=0: the BOS-pooled side number of a causal model, seed 42 only
  fi
  mkdir -p "$OUT/eval"; cp -r "$W/eval2025/results/$name" "$OUT/eval/" 2>/dev/null
  echo "[job] evaluated $name ($b${ft:+, finetune}) $(date -u +%T)"
}

reading_ps() {  # reading_ps MODEL_DIR NAME: amended Reading (critic 81702719), labelled "amended after data". The official
  # get_p2 encodes the target word without a leading space; our GPT-2 BPE export (add_prefix_space=False) then scores the
  # word-internal piece. A copy of MODEL_DIR whose tokenizer_config sets add_prefix_space=True (GPT-BERT's convention;
  # weights and all other files are the same files, symlinked) runs the unchanged official reading task as <NAME>_ps.
  local m=$1 name=${2}_ps d=$W/hf/${2}__ps f
  mkdir -p "$d"; for f in "$m"/*; do [ "$(basename "$f")" = tokenizer_config.json ] || ln -sfn "$(readlink -f "$f")" "$d/"; done
  $PY - "$m/tokenizer_config.json" "$d/tokenizer_config.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1])); c['add_prefix_space'] = True; json.dump(c, open(sys.argv[2], 'w'), indent=1)
PYEOF
  local data=$W/eval2025/evaluation_data/full_eval/reading/reading_data.csv
  sha256sum "$data" > "$OUT/reading_data_${name}.sha256"
  $PY "$REPO/babylm/eval/reading_alignment_check.py" "$d" "$data" --out "$OUT/reading_align_${name}.json" > "$OUT/reading_align_${name}.log" 2>&1
  $PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['prefix_space']['critic_mismatch_pct'] <= 1.0 else 1)" \
      "$OUT/reading_align_${name}.json" 2>/dev/null \
    || { echo "[job] reading alignment > 1% mismatch or missing for $name (critic STOP rule); not scored"; echo FAILED > "$OUT/READING_ALIGN_FAILED_${name}"; return 1; }
  mkdir -p "$W/models"; ln -sfn "$d" "$W/models/$name"
  ( cd "$W/eval2025" && $PY -m evaluation_pipeline.reading.run --model_path_or_name "$W/models/$name" --backend causal \
      --data_path evaluation_data/full_eval/reading/reading_data.csv ) > "$OUT/eval_reading_${name}.log" 2>&1 || echo "[job] reading FAILED for $name"
  mkdir -p "$OUT/eval"; cp -r "$W/eval2025/results/$name" "$OUT/eval/" 2>/dev/null
  echo "[job] amended reading $name: $(tr '\n' ' ' < "$OUT/eval/$name/main/zero_shot/causal/reading/report.txt" 2>/dev/null)"
}
glue_done() { find "$W/eval2025/results/$1/main/$2" -name results.txt 2>/dev/null | wc -l; }   # glue_done NAME DIR
# Per-task GLUE resume (experimenter, 10 Oct): a `python` shim first on PATH while the UNCHANGED official script runs. For a
# finetune call (--task T) whose results.txt and predictions.json already exist for this model it skips the call; after
# each finished task it copies the task dir to $OUT/eval/<model>/main/finetune/<T>, so a cut VM keeps finished tasks.
# Every other call passes straight to the venv python. A relaunch resumes when $OUT/eval (or GLUE_RESUME_FROM, a local
# copy of an earlier OUT_DIR) holds those task dirs: run_glue restores complete ones before starting.
mkdir -p "$W/glue_shim"; cat > "$W/glue_shim/python" <<'SHIM'
#!/bin/bash
a=("$@"); t=; m=
for ((i = 0; i < ${#a[@]}; i++)); do case ${a[i]} in --task) t=${a[i+1]};; --model_name_or_path) m=$(basename "${a[i+1]}"); m=${m%.*};; esac; done
d=results/$m/main/finetune/$t
fin() { [ -f "$d/results.txt" ] && [ -f "$d/predictions.json" ] && { [ "${GLUE_DUMP_PROBS:-0}" != 1 ] || [ -f "$d/valid_probs.json" ]; }; }
if [ -n "$t" ] && fin; then echo "[glue-resume] $m $t already finished; skipped"; exit 0; fi
if [ "${GLUE_DUMP_PROBS:-0}" = 1 ] && [ "${a[0]}" = -m ] && [ "${a[1]}" = evaluation_pipeline.finetune.run ]; then
  "$GLUE_REAL_PY" "$GLUE_PROBS_RUNNER" "${a[@]:2}"; rc=$?   # official run + valid_probs.json (glue_probs_run.py)
else "$GLUE_REAL_PY" "$@"; rc=$?; fi
if [ -n "$t" ] && fin; then mkdir -p "$GLUE_OUT/$m/main/finetune"; cp -r "$d" "$GLUE_OUT/$m/main/finetune/"; fi
exit $rc
SHIM
chmod +x "$W/glue_shim/python"
glue_restore() {  # glue_restore NAME: copy complete task dirs from $OUT/eval and GLUE_RESUME_FROM into the results dir
  local name=$1 src t dst=$W/eval2025/results/$1/main/finetune
  for src in "$OUT/eval/$name/main/finetune" ${GLUE_RESUME_FROM:+"$GLUE_RESUME_FROM/eval/$name/main/finetune"}; do
    for t in "$src"/*/; do t=${t%/}; [ -f "$t/results.txt" ] && [ -f "$t/predictions.json" ] && { [ "${GLUE_DUMP_PROBS:-0}" != 1 ] || [ -f "$t/valid_probs.json" ]; } \
        && [ ! -f "$dst/$(basename "$t")/results.txt" ] || continue
      mkdir -p "$dst"; cp -r "$t" "$dst/"; echo "[job] GLUE resume $name: restored $(basename "$t") from $src"; done
  done
}
run_glue() {  # run_glue MODEL_LINK LOG: GLUE with $FT_SCRIPT; all 7 tasks must finish. Concurrent runs can run out of GPU
  # memory (GPT-BERT, seq 512, 10 Oct: 6 of 7 tasks OOM on a 40 GB A100): then the incomplete output is kept aside under
  # finetune_concurrent_incomplete and the official sequential script reruns all 7. Fewer than 7 after that -> GLUE_INCOMPLETE_<name>.
  # optional 3rd arg SEED: the official script's positional args with its own defaults and that fine-tuning seed
  local link=$1 log=$2 name; name=$(basename "$1"); local fd=$W/eval2025/results/$name/main
  local sa=(); [ -n "${3:-}" ] && sa=(3e-5 32 16 10 30 "$3")
  local genv=(GLUE_REAL_PY=$W/venv/bin/python GLUE_OUT=$OUT/eval GLUE_PROBS_RUNNER=$REPO/babylm/eval/glue_probs_run.py PATH=$W/glue_shim:$W/venv/bin:$PATH)
  rm -f "$OUT/GLUE_INCOMPLETE_${name}"; glue_restore "$name"
  ( cd "$W/eval2025" && env "${genv[@]}" bash "$FT_SCRIPT" "$link" "${sa[@]}" ) > "$log" 2>&1
  local k; k=$(glue_done "$name" finetune)
  if [ "$k" -lt 7 ] && [ "$FT_SCRIPT" != eval_finetuning.sh ]; then
    { echo "concurrent GLUE finished $k/7 tasks; errors:"; tr '\r' '\n' < "$log" | grep -E '^[A-Za-z.]*(Error|Exception)' | cut -c1-200; } > "$OUT/ft_concurrent_incomplete_${name}.txt"
    echo "[job] GLUE concurrent $name incomplete ($k/7); rerunning the official sequential script $(date -u +%T)"
    rm -rf "$fd/finetune_concurrent_incomplete"; mv "$fd/finetune" "$fd/finetune_concurrent_incomplete"
    rm -rf "$OUT/eval/$name/main/finetune_concurrent_incomplete"   # keep concurrent task copies out of any later resume
    [ -d "$OUT/eval/$name/main/finetune" ] && mv "$OUT/eval/$name/main/finetune" "$OUT/eval/$name/main/finetune_concurrent_incomplete"
    ( cd "$W/eval2025" && env "${genv[@]}" bash eval_finetuning.sh "$link" "${sa[@]}" ) > "${log%.log}_sequential.log" 2>&1
    k=$(glue_done "$name" finetune)
  fi
  [ "$k" -ge 7 ] || { echo "GLUE finished $k/7 tasks; see $log" > "$OUT/GLUE_INCOMPLETE_${name}"; echo "[job] GLUE_INCOMPLETE $name ($k/7)"; }
}

glue_ens() {  # glue_ens OUT_NAME LINK_A SEED_A LINK_B SEED_B: two fine-tunes with probability dumps, then the averaged
  # ensemble (critic 44552922). Single model: the same model, seeds 42 and 43. K=2 mixture: member 0 seed 42, member 1
  # seed 43. Results under results/OUT_NAME/main/finetune (official results.txt format) + ensemble.json.
  local o=$1 la=$2 sa=$3 lb=$4 sb=$5
  GLUE_DUMP_PROBS=1 run_glue "$la" "$OUT/eval_finetune_$(basename "$la").log" "$sa"
  GLUE_DUMP_PROBS=1 run_glue "$lb" "$OUT/eval_finetune_$(basename "$lb").log" "$sb"
  mkdir -p "$OUT/eval"; cp -r "$W/eval2025/results/$(basename "$la")" "$W/eval2025/results/$(basename "$lb")" "$OUT/eval/" 2>/dev/null
  $PY "$REPO/babylm/eval/glue_ensemble.py" --pipeline "$W/eval2025" --out "$W/eval2025/results/$o/main/finetune" \
    "$W/eval2025/results/$(basename "$la")/main/finetune" "$W/eval2025/results/$(basename "$lb")/main/finetune" \
    > "$OUT/glue_ensemble_${o}.log" 2>&1 || { echo "[job] GLUE ensemble $o FAILED (see glue_ensemble_${o}.log)"; echo FAILED > "$OUT/GLUE_ENS_FAILED_${o}"; }
  cp -r "$W/eval2025/results/$o" "$OUT/eval/" 2>/dev/null; echo "[job] GLUE ensemble $o $(date -u +%T)"
}
glue_seed2() {  # glue_seed2 NAME: GLUE_ENS=2 -> NAME's matched 2-fine-tune-seed ensemble NAME_ens2 (seeds 42 + 43)
  [ "${GLUE_ENS:-}" = 2 ] || return 0
  # member A = NAME's own seed-42 run (GLUE_ENS=2 exports GLUE_DUMP_PROBS=1, so it already has valid_probs.json and
  # the resume shim skips all 7 tasks); member B = the same weights fine-tuned with seed 43
  ln -sfn "$(readlink -f "$W/models/$1")" "$W/models/${1}_ft43"
  glue_ens "${1}_ens2" "$W/models/$1" 42 "$W/models/${1}_ft43" 43
}

ft_verify() {  # ft_verify MODEL_DIR NAME: concurrent GLUE counts only after sha256 byte-identity of every output file
  # with the official sequential script on the same GPU type (critic ref-0293). Reuses this VM's concurrent results
  # for NAME when an earlier job left them, else runs the concurrent script first.
  local m=$1 name=$2 link=$W/models/$2 fd=$W/eval2025/results/$2/main
  mkdir -p "$W/models" "$OUT/eval"; ln -sfn "$m" "$link"
  if [ "$FT_SCRIPT" = eval_finetuning.sh ]; then   # FT_PARALLEL=0: sequential official GLUE only, no identity test
    echo "[job] GLUE sequential $name start (FT_PARALLEL=0, identity not tested) $(date -u +%T)"
    run_glue "$link" "$OUT/eval_finetune_${name}_sequential.log"
    cp -r "$fd/finetune" "$OUT/eval/finetune_sequential" 2>/dev/null
    echo "NOT_TESTED (FT_PARALLEL=0)" > "$OUT/ft_identity_${name}.txt"; return 0
  fi
  if [ "$(glue_done "$name" finetune)" -lt 7 ]; then
    echo "[job] GLUE concurrent $name start $(date -u +%T)"
    ( cd "$W/eval2025" && PATH=$W/venv/bin:$PATH bash "$FT_SCRIPT" "$link" ) > "$OUT/eval_finetune_${name}_concurrent.log" 2>&1
  else echo "[job] GLUE concurrent $name: reusing this VM's results"; fi
  rm -rf "$fd/finetune_concurrent"; mv "$fd/finetune" "$fd/finetune_concurrent"
  cp -r "$fd/finetune_concurrent" "$OUT/eval/" 2>/dev/null
  echo "[job] GLUE sequential $name start $(date -u +%T)"
  ( cd "$W/eval2025" && PATH=$W/venv/bin:$PATH bash eval_finetuning.sh "$link" ) > "$OUT/eval_finetune_${name}_sequential.log" 2>&1
  cp -r "$fd/finetune" "$OUT/eval/finetune_sequential" 2>/dev/null
  ( cd "$fd/finetune_concurrent" && find . -type f | sort | xargs sha256sum ) > "$OUT/ft_sha_${name}_concurrent.txt"
  ( cd "$fd/finetune" && find . -type f | sort | xargs sha256sum ) > "$OUT/ft_sha_${name}_sequential.txt"
  local kc ks; kc=$(glue_done "$name" finetune_concurrent); ks=$(glue_done "$name" finetune)
  if [ "$kc" -lt 7 ]; then echo "CONCURRENT_INCOMPLETE $kc/7 (sequential $ks/7)" > "$OUT/ft_identity_${name}.txt"
  elif [ -s "$OUT/ft_sha_${name}_sequential.txt" ] && cmp -s "$OUT/ft_sha_${name}_concurrent.txt" "$OUT/ft_sha_${name}_sequential.txt"
  then echo IDENTICAL > "$OUT/ft_identity_${name}.txt"
  else { echo MISMATCH; diff "$OUT/ft_sha_${name}_concurrent.txt" "$OUT/ft_sha_${name}_sequential.txt"; } > "$OUT/ft_identity_${name}.txt"; fi
  nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT/ft_identity_${name}_gpu.txt" 2>/dev/null
  echo "[job] GLUE concurrent vs sequential $name: $(head -1 "$OUT/ft_identity_${name}.txt") $(date -u +%T)"
}

t0=$(date +%s)
case "$JOB" in
  score_pub_*)
    n=${JOB#score_pub_}
    case $n in gpt2) m=babylm-baseline-10m-gpt2;; gptbert_causal) m=babylm-baseline-10m-gpt-bert-causal-focus;;
      gptbert_mixed) m=babylm-baseline-10m-gpt-bert-mixed;; gptbert_masked) m=babylm-baseline-10m-gpt-bert-masked-focus;;
      *) echo "unknown $n"; exit 2;; esac
    $PY -c "from huggingface_hub import snapshot_download as s; p=s('BabyLM-community/$m', local_dir='$W/hf/$m'); print(p)"
    $PY -c "from huggingface_hub import HfApi; print(HfApi().model_info('BabyLM-community/$m').sha)" > "$OUT/published_revision.txt"
    ( cd "$W/hf/$m" && sha256sum *.safetensors *.bin *.json *.py 2>/dev/null ) > "$OUT/published_files.sha256"
    export FAST=1
    if [ $n = gpt2 ]; then evaluate "$W/hf/$m" causal "pub_$n" finetune
      # critic ref-0058: also GLUE with the pooling-only last-token adapter (reported beside the official BOS-pooled GLUE)
      ! grep -q -- '--padding_side' "$W/eval2025/eval_finetuning.sh" && \
      $PY "$REPO/babylm/eval/adapters/test_lastpool.py" "$W/hf/$m" > "$OUT/lastpool_test.log" 2>&1 && \
      $PY "$REPO/babylm/eval/adapters/make_lastpool.py" "$W/hf/$m" "$W/hf/${m}_lastpool" && \
      ln -sfn "$W/hf/${m}_lastpool" "$W/models/pub_gpt2_lastpool" && \
      run_glue "$W/models/pub_gpt2_lastpool" "$OUT/eval_finetune_pub_gpt2_lastpool.log"
      cp -r "$W/eval2025/results/pub_gpt2_lastpool" "$OUT/eval/" 2>/dev/null
    else evaluate "$W/hf/$m" causal "pub_$n" finetune; evaluate "$W/hf/$m" mntp "pub_$n"; fi
    ;;
  ftverify_pub_*)   # sequential-vs-concurrent GLUE byte-identity on a published checkpoint (critic ref-0293)
    n=${JOB#ftverify_pub_}
    case $n in gpt2) m=babylm-baseline-10m-gpt2;; gptbert_causal) m=babylm-baseline-10m-gpt-bert-causal-focus;; *) echo "unknown $n"; exit 2;; esac
    $PY -c "from huggingface_hub import snapshot_download as s; s('BabyLM-community/$m', local_dir='$W/hf/$m')"
    ft_verify "$W/hf/$m" "pub_$n"
    ;;
  score_mix)   # K=2 mixture of causal Hinton exports (critic 44552922): env HF_URI = the 2 member exports, HF_NAME = its name
    # zero-shot + dev_bpw: the per-token mixture exported as ONE causal model (make_hinton_mix.py), unchanged pipeline;
    # GLUE: each member fine-tuned separately with last-token pooling (member 0 seed 42, member 1 seed 43), member
    # probabilities averaged -> results/<HF_NAME>_ens2 (glue_ensemble.py). Unit tests run on the real members first.
    read -ra uris <<< "${HF_URI:?HF_URI}"; nm=${HF_NAME:?HF_NAME}; mem=()
    [ ${#uris[@]} = 2 ] || { echo "score_mix needs exactly 2 member URIs"; exit 2; }
    for i in "${!uris[@]}"; do
      u=${uris[$i]%/}; d=$W/hf/${nm}_m$i; mkdir -p "$d"
      case "$u" in ${BUCKET}) gcloud storage cp -r "$u/*" "$d/" || gsutil -m cp -r "$u/*" "$d/";; *) cp -r "$u/." "$d/";; esac
      ( cd "$d" && sha256sum * 2>/dev/null ) > "$OUT/hf_files_${nm}_m$i.sha256"; echo "$u" > "$OUT/hf_uri_${nm}_m$i.txt"; mem+=("$d")
    done
    $PY "$REPO/babylm/eval/adapters/test_hinton_mix.py" "${mem[0]}" > "$OUT/test_hinton_mix_${nm}.log" 2>&1 \
      || { echo "[job] mixture unit test FAILED; not scored"; echo FAILED > "$OUT/MIX_TEST_FAILED_${nm}"; exit 4; }
    $PY "$REPO/babylm/eval/adapters/make_hinton_mix.py" "$W/hf/$nm" "${mem[@]}" > "$OUT/make_hinton_mix_${nm}.log" 2>&1 \
      || { echo "[job] make_hinton_mix FAILED"; exit 4; }
    ( cd "$W/hf/$nm" && sha256sum * 2>/dev/null ) > "$OUT/hf_files_${nm}.sha256"
    $PY "$REPO/babylm/eval/smoke_hf.py" "$W/hf/$nm" > "$OUT/smoke_hf_${nm}.json" 2>"$OUT/smoke_hf_${nm}.log" \
      || { echo "[job] VOID $nm: mixture smoke failed"; echo VOID > "$OUT/VOID_${nm}"; exit 4; }
    if [ "${READING_PS:-}" = 1 ]; then reading_ps "$W/hf/$nm" "$nm"; echo "[job] $JOB done $(date -u +%FT%TZ) wall_s=$(( $(date +%s) - t0 ))"; exit 0; fi
    if [ -n "${GLUE_PREV:-}" ]; then   # GLUE fix-up of an earlier concurrent run (critic ref-0191): its zero-shot, guard
      # and twin stand and are copied into this OUT (the model's one source for the readout); the GLUE_RERUN tasks
      # (default multirc) are set aside, never reported, and rerun with $FT_SCRIPT; GLUE_SPOT=m<i>:<task> also reruns that
      # member's task and must be sha256-identical to the set-aside concurrent copy, else GLUE_INCOMPLETE_<name>_ens2.
      P=$W/prev; A=$W/prev_aside; mkdir -p "$P" "$A"
      case "$GLUE_PREV" in ${BUCKET}) gsutil -m -q cp -r "${GLUE_PREV%/}/*" "$P/";; *) cp -r "${GLUE_PREV%/}/." "$P/";; esac
      for f in "MIX_GUARD_${nm}.txt" "VOID_${nm}" "dev_bpw_twin_${nm}.json" "dev_bpw_twin_${nm}.log" "eval_zero_shot_${nm}_causal.log"; do
        [ -e "$P/$f" ] && cp "$P/$f" "$OUT/"; done
      [ -d "$P/eval/$nm" ] && { mkdir -p "$OUT/eval"; cp -r "$P/eval/$nm" "$OUT/eval/"; }
      echo "$GLUE_PREV" > "$OUT/glue_prev_${nm}.txt"
      for i in 0 1; do for t in ${GLUE_RERUN:-multirc}; do d=$P/eval/${nm}_m${i}_lastpool/main/finetune/$t
        [ -d "$d" ] && { mkdir -p "$A/${nm}_m${i}_lastpool"; mv "$d" "$A/${nm}_m${i}_lastpool/$t"; }; done; done
      if [ -n "${GLUE_SPOT:-}" ]; then sm=${nm}_${GLUE_SPOT%%:*}_lastpool; st=${GLUE_SPOT#*:}; d=$P/eval/$sm/main/finetune/$st
        if [ -d "$d" ]; then mkdir -p "$A/spot/$sm"; mv "$d" "$A/spot/$sm/$st"; else echo "[job] spot $GLUE_SPOT: no concurrent copy"; fi; fi
      export GLUE_RESUME_FROM=$P
      echo "[job] GLUE fix-up $nm from $GLUE_PREV: rerun ${GLUE_RERUN:-multirc}${GLUE_SPOT:+, spot $GLUE_SPOT} ($FT_SCRIPT)"
    else
    evaluate "$W/hf/$nm" causal "$nm"
    if [ -n "${TWIN_REPO:-}" ]; then   # binding guard: twin DEV of the export at T=1 vs the read file's raw mixture DEV
      $PY "$TWIN_REPO/babylm/eval/dev_bpw_twin.py" "$W/hf/$nm" "$DEV" ${TWIN_ARGS:---slice full} --out "$OUT/dev_bpw_twin_${nm}.json" \
        > "$OUT/dev_bpw_twin_${nm}.log" 2>&1 || echo "[job] twin DEV failed for $nm"
      $PY - "$OUT/dev_bpw_twin_${nm}.json" "${MIX_REF_NPW:-}" > "$OUT/MIX_GUARD_${nm}.txt" <<'PYEOF'
import json, sys
try: got = json.load(open(sys.argv[1]))['nats_per_word']
except Exception as e: print(f'FAIL twin DEV missing ({e})'); sys.exit()
ref = sys.argv[2]
if not ref: print(f'NO_REF export twin DEV {got:.6f} nats/word (no MIX_REF_NPW; not claim-grade)'); sys.exit()
d = abs(got - float(ref))
print(f"{'PASS' if d <= 1e-4 else 'FAIL'} export {got:.6f} vs read {float(ref):.6f} nats/word, |diff| {d:.2e} (tolerance 1e-4)")
PYEOF
      echo "[job] mixture guard $nm: $(cat "$OUT/MIX_GUARD_${nm}.txt")"
      grep -q '^FAIL' "$OUT/MIX_GUARD_${nm}.txt" && echo VOID > "$OUT/VOID_${nm}"
    fi
    fi
    [ "${ZS_ONLY:-}" = 1 ] && { echo "[job] ZS_ONLY: $nm GLUE skipped"; echo "[job] $JOB done $(date -u +%FT%TZ) wall_s=$(( $(date +%s) - t0 ))"; exit 0; }
    lps=()
    for i in 0 1; do
      $PY "$REPO/babylm/eval/adapters/test_hinton_lastpool.py" "${mem[$i]}" --pipeline "$W/eval2025" > "$OUT/test_hinton_lastpool_${nm}_m$i.log" 2>&1 \
        || { echo "[job] lastpool unit test FAILED for member $i; GLUE not scored"; echo FAILED > "$OUT/LASTPOOL_TEST_FAILED_${nm}_m$i"; exit 4; }
      $PY "$REPO/babylm/eval/adapters/make_hinton_lastpool.py" "${mem[$i]}" "$W/hf/${nm}_m${i}_lastpool" > "$OUT/make_hinton_lastpool_${nm}_m$i.log" 2>&1 || exit 4
      ln -sfn "$W/hf/${nm}_m${i}_lastpool" "$W/models/${nm}_m${i}_lastpool"; lps+=("$W/models/${nm}_m${i}_lastpool")
    done
    glue_ens "${nm}_ens2" "${lps[0]}" 42 "${lps[1]}" 43
    if [ -n "${GLUE_PREV:-}" ] && [ -n "${GLUE_SPOT:-}" ]; then   # spot identity: concurrent copy vs this sequential rerun
      a=$W/prev_aside/spot/$sm/$st; b=$OUT/eval/$sm/main/finetune/$st; r=$OUT/spot_identity_${sm}_${st}.txt
      if [ ! -d "$a" ] || [ ! -f "$b/results.txt" ]; then echo "MISSING (concurrent copy $( [ -d "$a" ] && echo present || echo absent ), rerun $( [ -f "$b/results.txt" ] && echo present || echo absent ))" > "$r"
      else ( cd "$a" && find . -type f | sort | xargs sha256sum ) > "${r%.txt}_concurrent.sha256"
        ( cd "$b" && find . -type f | sort | xargs sha256sum ) > "${r%.txt}_sequential.sha256"
        # only results.txt + predictions.json decide (critic ref-0350 definition); the full lists are informational
        k() { grep -E '  \./(results\.txt|predictions\.json)$' "$1"; }
        if [ "$(k "${r%.txt}_concurrent.sha256" | wc -l)" = 2 ] && [ "$(k "${r%.txt}_concurrent.sha256")" = "$(k "${r%.txt}_sequential.sha256")" ]
        then echo IDENTICAL > "$r"
        else { echo MISMATCH; diff "${r%.txt}_concurrent.sha256" "${r%.txt}_sequential.sha256"; } > "$r"; fi; fi
      echo "[job] spot identity $sm $st: $(head -1 "$r")"
      grep -q '^IDENTICAL' "$r" || echo "spot identity $sm $st not IDENTICAL: rerun the bin sequentially (critic ref-0191)" > "$OUT/GLUE_INCOMPLETE_${nm}_ens2"
    fi
    ;;
  score_hf)   # env HF_URI and HF_NAME (space-separated lists of equal length, scored in turn); optional FMT_JSON
    read -ra uris <<< "${HF_URI:?HF_URI}"; read -ra names <<< "${HF_NAME:?HF_NAME}"
    [ ${#uris[@]} = ${#names[@]} ] || { echo "HF_URI/HF_NAME length mismatch"; exit 2; }
    for i in "${!uris[@]}"; do
      u=${uris[$i]%/}; nm=${names[$i]}; m=$W/hf/$nm; mkdir -p "$m"
      case "$u" in ${BUCKET}) gcloud storage cp -r "$u/*" "$m/" || gsutil -m cp -r "$u/*" "$m/";; *) cp -r "$u/." "$m/";; esac
      ( cd "$m" && sha256sum * 2>/dev/null ) > "$OUT/hf_files_${nm}.sha256"; echo "$u" > "$OUT/hf_uri_${nm}.txt"
      if [[ $nm == *_untrained* ]]; then   # GLUE sanity pair (critic 44552922): same export, fresh random init (seed 0)
        $PY "$REPO/babylm/eval/adapters/make_untrained.py" "$m" "${m}__untrained" 0 > "$OUT/make_untrained_${nm}.log" 2>&1 \
          || { echo "[job] make_untrained FAILED for $nm"; continue; }
        m=${m}__untrained; ( cd "$m" && sha256sum * 2>/dev/null ) > "$OUT/hf_files_${nm}_untrained.sha256"
      fi
      if [[ $nm == *_lastpool ]] && grep -q '"model_type": *"hinton_gpt"' "$m/config.json" && ! grep -q '"automodel_pool": *"last"' "$m/config.json"; then
        # last NON-PAD token pooling for a causal Hinton export (experimenter 11 Oct; critic 44552922): pad-side unit test first
        $PY "$REPO/babylm/eval/adapters/test_hinton_lastpool.py" "$m" --pipeline "$W/eval2025" > "$OUT/test_hinton_lastpool_${nm}.log" 2>&1 \
          || { echo "[job] hinton lastpool unit test FAILED for $nm; not scored"; echo FAILED > "$OUT/LASTPOOL_TEST_FAILED_${nm}"; continue; }
        $PY "$REPO/babylm/eval/adapters/make_hinton_lastpool.py" "$m" "${m}__lastpool" > "$OUT/make_hinton_lastpool_${nm}.log" 2>&1 \
          || { echo "[job] make_hinton_lastpool FAILED for $nm"; continue; }
        m=${m}__lastpool; ( cd "$m" && sha256sum * 2>/dev/null ) > "$OUT/hf_files_${nm}_lastpool.sha256"
      fi
      $PY "$REPO/babylm/eval/smoke_hf.py" "$m" > "$OUT/smoke_hf_${nm}.json" 2>"$OUT/smoke_hf_${nm}.log" || echo "[job] smoke_hf FAILED for $nm (see smoke_hf_${nm}.json)"
      # a model whose causal logits are not finite (or that does not load) would only produce void scores (critic R-step 0)
      $PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('causal_logits_finite',{}).get('ok') else 1)" \
        "$OUT/smoke_hf_${nm}.json" 2>/dev/null || { echo "[job] VOID $nm: non-finite logits or load failure; not scored"; echo VOID > "$OUT/VOID_${nm}"; continue; }
      if [ "${READING_PS:-}" = 1 ]; then [[ $nm == *_lastpool ]] || reading_ps "$m" "$nm"; continue; fi
      if [[ $nm == *_lastpool ]] && [ "${ZS_ONLY:-}" = 1 ]; then echo "[job] ZS_ONLY: $nm is GLUE-only; skipped"; continue
      elif [[ $nm == *_lastpool ]]; then   # same weights as the BOS export: GLUE only (last-pooled adapter, critic ref-0491)
        $PY "$REPO/babylm/eval/smoke_hf.py" "$m" --pool last > "$OUT/smoke_hf_${nm}_poollast.json" 2>&1 || { echo "[job] $nm fails --pool last; skipped"; continue; }
        ln -sfn "$m" "$W/models/$nm"
        run_glue "$W/models/$nm" "$OUT/eval_finetune_${nm}.log"
        mkdir -p "$OUT/eval"; cp -r "$W/eval2025/results/$nm" "$OUT/eval/" 2>/dev/null; echo "[job] finetuned $nm $(date -u +%T)"
        glue_seed2 "$nm"
      elif [ "${ZS_ONLY:-}" = 1 ]; then
        evaluate "$m" causal "$nm"
      else
        evaluate "$m" causal "$nm" finetune
      fi
    done
    ;;
  ftverify_hf)   # concurrent-vs-sequential GLUE sha256 identity on a causal Hinton export with last-token pooling, FT
    # seed 42 (critic ref-0350: one check per model size registers FT_PARALLEL=1 for that size). env HF_URI, HF_NAME
    # (*_lastpool). Needs FT_PARALLEL=1 (the default). Result: ft_identity_<HF_NAME>.txt IDENTICAL | MISMATCH | ...
    u=${HF_URI:?HF_URI}; u=${u%/}; nm=${HF_NAME:?HF_NAME}; m=$W/hf/$nm; mkdir -p "$m"
    [[ $nm == *_lastpool ]] || { echo "ftverify_hf: HF_NAME must end in _lastpool"; exit 2; }
    [ "$FT_SCRIPT" != eval_finetuning.sh ] || { echo "ftverify_hf needs the concurrent script (FT_PARALLEL=1)"; exit 2; }
    case "$u" in ${BUCKET}) gcloud storage cp -r "$u/*" "$m/" || gsutil -m cp -r "$u/*" "$m/";; *) cp -r "$u/." "$m/";; esac
    ( cd "$m" && sha256sum * 2>/dev/null ) > "$OUT/hf_files_${nm}.sha256"; echo "$u" > "$OUT/hf_uri_${nm}.txt"
    $PY "$REPO/babylm/eval/adapters/test_hinton_lastpool.py" "$m" --pipeline "$W/eval2025" > "$OUT/test_hinton_lastpool_${nm}.log" 2>&1 \
      || { echo "[job] hinton lastpool unit test FAILED for $nm"; echo FAILED > "$OUT/LASTPOOL_TEST_FAILED_${nm}"; exit 4; }
    $PY "$REPO/babylm/eval/adapters/make_hinton_lastpool.py" "$m" "${m}__lastpool" > "$OUT/make_hinton_lastpool_${nm}.log" 2>&1 || exit 4
    ft_verify "${m}__lastpool" "$nm"
    cp -r "$W/eval2025/results/$nm" "$OUT/eval/" 2>/dev/null
    ;;
  entity_v1_pub_gptbert_causal)   # entity +0.59 diagnosis (critic ref-0293): the published 33.96 predates OSF's 15 May 2025
    # replacement of the entity files (v1: 53,904 items, 7 May; v2 = the current 9,483). Score the published model on v1.
    m=babylm-baseline-10m-gpt-bert-causal-focus; d=$W/entity_v1/entity_tracking; mkdir -p "$d"
    $PY -c "from huggingface_hub import snapshot_download as s; s('BabyLM-community/$m', local_dir='$W/hf/$m')"
    for f in 86n79:ambiref:2be84c350d2a101c7e47a9c4e2fbd0d03f2a5d575cdf5b6ff6b32fdc5e5038af \
             9nvtd:move_contents:cdc7ba816786881e6be5cb107e6355ec24532652bc6be23de5ddb1f99bad17c0 \
             qfx7h:regular:ff6c4280e3b0fbc5a93d11ed83c6487aadda8d1d64976d4cac4507e33f30cd7a; do
      IFS=: read -r id nm sha <<< "$f"
      curl -sfL --retry 5 --max-time 600 -o "$d/$nm.jsonl" "https://osf.io/download/$id/?revision=1"
      [ "$(sha256sum "$d/$nm.jsonl" | cut -d' ' -f1)" = "$sha" ] || { echo "[job] entity v1 $nm sha mismatch"; exit 1; }
    done
    ( cd "$W/eval2025" && $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$W/hf/$m" --backend causal \
        --task entity_tracking --data_path "$d" --save_predictions --output_dir "$OUT/entity_v1" ) > "$OUT/entity_v1.log" 2>&1
    tail -3 "$OUT/entity_v1.log"
    ;;
  score_controls)
    for m in babylm-baseline-10m-gpt2 babylm-baseline-10m-gpt-bert-causal-focus; do
      $PY -c "from huggingface_hub import snapshot_download as s; s('BabyLM-community/$m', local_dir='$W/hf/$m')"
    done
    $PY "$REPO/babylm/eval/make_controls.py" "$W/hf" "$W/controls" || exit 1
    evaluate "$W/controls/untrained_gpt2" causal ctl_untrained_gpt2
    evaluate "$W/controls/untrained_gptbert" causal ctl_untrained_gptbert; evaluate "$W/controls/untrained_gptbert" mntp ctl_untrained_gptbert
    evaluate "$W/controls/nan_gptbert" causal ctl_nan_gptbert
    ;;
  gpt2_seed*)
    s=${JOB#gpt2_seed}
    $PY "$REPO/babylm/recipes/gpt2/train.py" --train_dir "$TRAIN" --dev_dir "$DEV" --out_dir "$OUT/run" --seed "$s" || exit 1
    $PY "$REPO/babylm/tools/plot_curve.py" "$OUT/run/curve.csv"
    evaluate "$OUT/run/final" causal "gpt2_s$s" finetune
    ;;
  gptbert_causal_seed*)
    s=${JOB##*_seed}; [ "$s" = 0 ] && export GB_PUBREF=1
    GB_TAG=gptbert_causal source "$REPO/babylm/recipes/gptbert/run_job.sh" "$s" "$OUT" "$W" || exit 1
    ;;
  gptbert_mixed_seed*)
    s=${JOB##*_seed}
    GB_NUM=8 GB_TAG=gptbert_mixed source "$REPO/babylm/recipes/gptbert/run_job.sh" "$s" "$OUT" "$W" || exit 1
    ;;
  gptbert_100M_e*)
    e=${JOB##*_e}
    GB_TRAIN=train_100M GB_EPOCHS=$e GB_TAG=gptbert_100M_e$e source "$REPO/babylm/recipes/gptbert/run_job.sh" 0 "$OUT" "$W" || exit 1
    ;;
  sprint1_seed*)
    s=${JOB##*_seed}
    source "$REPO/babylm/recipes/sprint1_port/run_job.sh" "$s" "$OUT" "$W" || exit 1
    ;;
  *) echo "unknown job $JOB"; exit 2;;
esac
echo "[job] $JOB done $(date -u +%FT%TZ) wall_s=$(( $(date +%s) - t0 ))"
