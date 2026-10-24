#!/usr/bin/env bash
# Card F2 Stage 0 (docs/babylm/moonshot/cards/F2_heterogeneous_legal_split.md): pairwise g(2) of existing v2 dumps,
# raw and calibrated, full and guarded legs. Zero GPU. For a session with GCS read access and data/babylm set up as
# v2_readouts.sh needs. Read-only on GCS.
#
#   PAIRS="name1=${BUCKET}//B.npz name2=${BUCKET}//D.npz ..." babylm/moonshot/hetero_pairs.sh OUT_DIR
#
# Each pair is read with `ceiling_report c3 A B --guard --calib`; OUT_DIR/summary.txt gets one line per pair:
# g(2) raw / calibrated_t on both legs and each member's DEV. Compare a heterogeneous pair only with a seed pair of
# the same torch env and epoch count (the torch-env rule); the script does not check that for you.
set -euo pipefail
OUT=${1:?usage: PAIRS=... hetero_pairs.sh OUT_DIR}
: "${PAIRS:?set PAIRS=\"name=${BUCKET}//B.npz ...\"}"
PY=${PY:-python}
mkdir -p "$OUT"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

fetch() {
  local u=$1 n
  case $u in
    */pt_v2/dev_v2.npz) n=$(basename "$(dirname "$(dirname "$u")")") ;;
    */dev_v2.npz) n=$(basename "$(dirname "$u")") ;;
    *) n=$(basename "$u" .npz) ;;
  esac
  [ -e "$W/$n.npz" ] || gcloud storage cp -q "$u" "$W/$n.npz" >&2
  echo "$W/$n.npz"
}

for spec in $PAIRS; do
  name=${spec%%=*}
  IFS=, read -r ua ub <<< "${spec#*=}"
  a=$(fetch "$ua"); b=$(fetch "$ub")
  $PY -m babylm.moonshot.ceiling_report c3 "$a" "$b" --guard --calib --out "$OUT/$name" > "$OUT/$name.brief.json"
  $PY - "$name" "$OUT/$name/c3.json" >> "$OUT/summary.txt" <<'PYEOF'
import json, sys
name, path = sys.argv[1], sys.argv[2]
r = json.load(open(path))
def g2(d):
    rows = {x["K"]: x for x in d.get("rows", [])}
    return rows.get(2, {}).get("g_npw_raw", float("nan"))
def cal(d):
    for row in (d.get("calibrated") or {}).get("rows", []):
        if row.get("K") == 2:
            return {k: v for k, v in row.items() if isinstance(v, float) and k.startswith("g_")}
    return {}
members = [x.get("npw_raw", float("nan")) for x in r.get("single", [])]
line = f"{name}: g2_raw={g2(r):.4f}"
if "never_seen_char_guard" in r:
    line += f" g2_raw_guarded={g2(r['never_seen_char_guard']):.4f}"
c = cal(r)
if c:
    line += " " + " ".join(f"{k}={v:.4f}" for k, v in c.items())
if members:
    line += " members=" + ",".join(f"{m:.4f}" for m in members)
print(line)
PYEOF
done
cat "$OUT/summary.txt"
