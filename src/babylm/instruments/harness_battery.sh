#!/bin/bash
# The instruments' battery on the harness's per-word dumps (DEV-sub / train-sub; see harness.py): one 10M run with
# a dump per epoch and its 100M twin. Writes results/babylm/<tag>/ (CSV + PNG).
#   bash babylm/instruments/harness_battery.sh TAG RUN10 REF100 [FRESH]
#   e.g. bash babylm/instruments/harness_battery.sh harness_port port_S_10M_e10_s0 grid_S_100M_e10_s0/chck_1000M \
#        grid_S_100M_e1_s0/chck_100M
# FRESH (optional): a 100M-data run/rev that has seen as many words as RUN10's final, each once (fresh vs repeated text).
# Needs: python -m babylm.instruments.harness features (once) and harness import --pack for both runs.
# The 10M run is read at its final checkpoint and, when an earlier epoch has the lowest DEV-sub loss, there too.
cd "$(dirname "$0")/../.." || exit 1
export BABYLM_SAMPLE=harness
T=$1 R10=$2 REF=$3 FRESH=$4
DATA_ROOT=${BABYLM_DATA:-~/babylm_data}
has() { [ -f "data/babylm_scores/$R10/$1/devsub.npz" ]; }
read -r FINAL BEST <<<"$(python3 -c "
import json
from babylm.instruments import slices as S
revs = S.scored('$R10')
nats = lambda v: json.load(open(f'data/babylm_scores/$R10/{v}/meta.json'))['splits']['devsub']['nats_per_word']
print(revs[-1], min(revs, key=nats))")"
[ -n "$FINAL" ] || exit 1
EP() { echo "${1//[!0-9]/}M"; }
PAIRS=(--pair "$R10/$FINAL:$REF" --pair "$R10/chck_10M:$R10/$FINAL")
LEVERS=(--lever "10x data=$R10/$FINAL:$REF")
CACHE="$R10/$FINAL,$REF" MODELS="$R10/$FINAL,$REF"
JOINT=(--joint "$R10/$FINAL+kn:train_10M+cache")
PRIZE=(--pair "$R10 vs ${REF%%/*}=$R10/$FINAL:$REF")
if [ "$BEST" != "$FINAL" ]; then
  PAIRS+=(--pair "$R10/$BEST:$REF" --pair "$R10/$BEST:$R10/$FINAL")
  LEVERS+=(--lever "10x data, best epoch ($(EP "$BEST") words)=$R10/$BEST:$REF")
  CACHE+=",$R10/$BEST" MODELS+=",$R10/$BEST"
  JOINT+=(--joint "$R10/$BEST+kn:train_10M+cache")
  PRIZE+=(--pair "$R10 best epoch vs ${REF%%/*}=$R10/$BEST:$REF")
fi
if [ -n "$FRESH" ]; then
  PAIRS+=(--pair "$R10/$FINAL:$FRESH")
  LEVERS+=(--lever "fresh vs repeated text ($(EP "$FINAL") words seen)=$R10/$FINAL:$FRESH")
fi
for pair in chck_10M:chck_20M chck_20M:chck_50M chck_50M:$FINAL chck_10M:$FINAL; do
  a=${pair%%:*} b=${pair##*:}
  if has "$a" && has "$b" && [ "$a" != "$b" ]; then
    LEVERS+=(--lever "epochs $(EP "$a") to $(EP "$b") words=$R10/$a:$R10/$b")
  fi
done
MIX=()
for e in chck_20M chck_50M; do has "$e" && [ "$e" != "$FINAL" ] && MIX+=(--mix "$R10/$e+$R10/$FINAL"); done
echo "final $FINAL, best $BEST"
set -x
python3 -m babylm.instruments.overlap --data-root "$DATA_ROOT"
python3 -m babylm.instruments.slices --tag "$T" --runs "$R10,${REF%%/*}${FRESH:+,${FRESH%%/*}}" "${PAIRS[@]}"
python3 -m babylm.instruments.stall --tag "$T" --run "$R10" --ref "$REF"
python3 -m babylm.instruments.signature --tag "$T" --no-defaults "${LEVERS[@]}"
python3 -m babylm.instruments.anatomy --tag "$T" --cache "$CACHE" "${MIX[@]}"
python3 -m babylm.instruments.ngram --tag "$T" --data-root "$DATA_ROOT" --models "$MODELS" "${JOINT[@]}"
python3 -m babylm.instruments.prize --tag "$T" --data-root "$DATA_ROOT" "${PRIZE[@]}"
