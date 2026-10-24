#!/bin/bash
# Set up the official BabyLM 2025 evaluation pipeline at a pinned commit, with verified data.
# Usage: babylm/eval/setup.sh WORKDIR   -> WORKDIR/eval2025 (pipeline), WORKDIR/osf (data), WORKDIR/venv
set -euo pipefail
W=${1:?workdir}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"; cd "$W"
[ -d eval2025 ] || timeout 600 git clone -q https://github.com/babylm/evaluation-pipeline-2025 eval2025
git -C eval2025 checkout -q bf55c1131e53654c2a87418f5629c26959acd710
timeout 3600 python3 "$HERE/../data/fetch.py" "$W/osf"   # hard stop: a hung fetch fails the job instead of idling a GPU
ln -sfn "$W/osf/evaluation_data" eval2025/evaluation_data
python3 "$HERE/patch_ties.py" "$W/eval2025"   # logs argmax ties; scores unchanged
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
  venv/bin/pip install -q --timeout 60 --retries 5 torch==${TORCH_VERSION:-2.7.0} ${TORCH_INDEX:+--index-url $TORCH_INDEX}
  venv/bin/pip install -q --timeout 60 --retries 5 -r "$HERE/requirements_eval.txt"
  venv/bin/python -c "import nltk; nltk.download('punkt_tab',quiet=True); nltk.download('punkt',quiet=True)"
fi
echo "eval ready: cd $W/eval2025 && PY=$W/venv/bin/python $HERE/zero_shot_all.sh MODEL causal"
