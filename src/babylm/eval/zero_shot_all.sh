#!/bin/bash
# Official 2025 zero-shot suite (eval upstream pipeline bf55c1131e53654c2a87418f5629c26959acd710). Usage: zero_shot_all.sh MODEL_PATH BACKEND [EWOK_DIR]
# Uses full_eval everywhere; EWoK falls back to fast_eval/ewok_fast when the gated full set is absent.
set -u
M=$1; B=$2; E=evaluation_data/full_eval
EW=${3:-$E/ewok_filtered}; [ -d "$EW" ] || EW=evaluation_data/fast_eval/ewok_fast
BR=$B; [[ "$B" == *enc_dec* ]] && BR=enc_dec
PY=${PY:-python}
run(){ echo "== $(date -u +%H:%M:%S) $*"; $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $M --backend $B --save_predictions "$@" || echo "FAILED $*"; }
run --task blimp --data_path $E/blimp_filtered
run --task blimp --data_path $E/supplement_filtered
run --task ewok --data_path $EW
run --task entity_tracking --data_path $E/entity_tracking
run --task wug_adj --data_path $E/wug_adj_nominalization
run --task wug_past --data_path $E/wug_past_tense
echo "== $(date -u +%H:%M:%S) reading"; $PY -m evaluation_pipeline.reading.run --model_path_or_name $M --backend $BR --data_path $E/reading/reading_data.csv || echo "FAILED reading"
run --task comps --data_path $E/comps
echo "== $(date -u +%H:%M:%S) done"
