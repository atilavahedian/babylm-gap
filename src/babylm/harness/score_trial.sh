#!/bin/bash
# Score one finished base-sweep run with the critic-accepted protocol: HF export of the final checkpoint (--verify),
# DEV nats/word (babylm/eval/dev_bpw.py, ctx 512 stride 256, full dev) and the official FAST zero-shot suite
# (causal backend; an MNTP-trained export has a bidirectional AutoModel, see export_hf). DEV nats/word uses this repo's
# babylm/eval/dev_bpw.py in the committed training format (babylm/recipes/harness/format.json: --ws collapse
# --line_join newline). The eval pipeline set-up tools come from the ruler runner's pinned commit (RULER_SHA).
#   babylm/harness/score_trial.sh RUN_DIR WORKDIR        (WORKDIR: eval2025/ + venv/ + osf/ from babylm/eval/setup.sh;
#                                                        created on first use)
# Writes RUN_DIR/score/{hf,dev_bpw.json,eval_fast.log,eval/,score.json}.
set -uo pipefail
RUN=$(cd "${1:?run_dir}" && pwd); W=${2:?workdir}; REPO=$(cd "$(dirname "$0")/../.." && pwd)
RULER_SHA=${RULER_SHA:-c09ef37ac16bbe0a9f3ab9def66ad000ea40cef5}
T=$W/ruler_tools_$RULER_SHA
if [ ! -d "$T/babylm/eval" ]; then
  mkdir -p "$T" && git -C "$REPO" fetch -q origin "$RULER_SHA" 2>/dev/null
  git -C "$REPO" archive "$RULER_SHA" babylm/eval babylm/data/fetch.py babylm/data/osf_listing.txt | tar -x -C "$T" || exit 1
fi
[ -x "$W/venv/bin/python" ] && [ -d "$W/eval2025" ] || "$T/babylm/eval/setup.sh" "$W" || exit 1
PY=$W/venv/bin/python; S=$RUN/score; mkdir -p "$S"; NAME=$(basename "$RUN")
CK=$RUN/checkpoints/chck_final.pt  # end weights of a run that stopped short of its next milestone (fractional epochs)
[ -f "$CK" ] || CK=$(ls "$RUN"/checkpoints/chck_*M.pt | awk -F"chck_" '{print $NF" "$0}' | sort -n | tail -1 | cut -d" " -f2)
( cd "$REPO" && $PY -m babylm.harness.export_hf --ckpt "$CK" --out "$S/hf" --verify ) > "$S/export.log" 2>&1 || { echo "[score] export failed"; exit 1; }
FMT=$($PY -c "import json;f=json.load(open('$REPO/babylm/recipes/harness/format.json'));print('--ws',f['ws'],'--line_join',f['line_join'])")
# shellcheck disable=SC2086
$PY "$REPO/babylm/eval/dev_bpw.py" "$S/hf" "$W/osf/text_data/dev/dev" --slice full $FMT --out "$S/dev_bpw.json" > "$S/dev_bpw.log" 2>&1
L=$W/models/$NAME; mkdir -p "$W/models"; ln -sfn "$S/hf" "$L"
( cd "$W/eval2025" && BABYLM_TIE_LOG=$S/ties_fast.jsonl PATH=$W/venv/bin:$PATH bash eval_zero_shot_fast.sh "$L" fast causal ) > "$S/eval_fast.log" 2>&1
mkdir -p "$S/eval"; cp -r "$W/eval2025/results/$NAME" "$S/eval/" 2>/dev/null
REPO="$REPO" $PY - "$RUN" "$CK" <<'PYEOF' > "$S/score.json"
import json, sys, glob, os
run, ck = sys.argv[1], sys.argv[2]
r = json.load(open(os.path.join(run, "result.json")))
d = json.load(open(os.path.join(run, "score", "dev_bpw.json")))
sys.path.insert(0, os.environ["REPO"])
from babylm.harness.fast_summary import summarize
fast = summarize(os.path.join(run, "score", "eval"))  # accuracies; predictions stay in score/eval
print(json.dumps({"run": os.path.basename(run), "checkpoint": os.path.basename(ck),
                  "dev_nats_per_word": d.get("nats_per_word"), "dev_bits_per_word": d.get("bits_per_word"),
                  "dev_format": {k: d.get(k) for k in ("ws", "line_join", "ctx")},
                  "automodel_attention": json.load(open(os.path.join(run, "score", "hf", "hinton_export.json")))
                  .get("automodel_attention"),
                  "harness_full_dev_nats_per_word": r.get("full_eval", {}).get("nats_per_word"),
                  "train_time_s": r.get("train_time_s"), "tokens_per_s": r.get("tokens_per_s"),
                  "params_nonembedding": r.get("params_nonembedding"), "levers": r["config"].get("levers"),
                  "fast": fast}, indent=1))
PYEOF
echo "[score] $NAME dev nats/word $($PY -c "import json;d=json.load(open('$S/score.json'));print(d['dev_nats_per_word'], d['fast'])")"
