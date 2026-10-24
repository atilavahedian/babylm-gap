#!/usr/bin/env bash
# Moonshot readouts on v2 twin dumps, for a session that has GCS read access and data/babylm (raw/dev, raw/train_100M or
# analysis/linehash_train_100M.npy, bin/{dev.bin,dev.words.bin,dev.meta.json,train_10M.bin}). Read-only on GCS.
#
#   babylm/moonshot/v2_readouts.sh OUT_DIR
#
# Each env var is a space-separated list of GCS dev_v2.npz (or _dumps_v2/<name>.npz) paths; unset ones are skipped.
#   C3          the 8 C3 base seeds, s0-s2 (solo) FIRST, then s3-s7 (packed); label mixed. Gate 0 = g(3) >= 0.06 and
#               g(8) >= 0.10 on both legs; the first three are also read alone (solo-only K=3)
#   C1          three paths: self oracle t100 (final f1)
#   LS_<TAG>    one legal-split mixture's members (e.g. LS_K2 = the two e5 seeds, LS_K3, LS_BR = the branch children)
#   LS_REF      the five 10-epoch base seeds (temps dumps): vs_ref = their mean minus the mixture, raw and calibrated
#   ESTAR_REF   the base's best-epoch singles (temps dumps): the vs-E* bar, same read
#   PE3 PE3_ENS I3 F [F_JSON] [NJ_ENS]   card PE Gate 2 (babylm.moonshot.pe_gate2_read): PE3's member dumps (particle_members),
#               PE3's own dump, F1's ms_f1_m0-m2, F-deep (+ F-M) dumps, local dev_bpw_twin.json of an F run
#               with no dump (F_JSON, e.g. ms_f1_FM), NJ's own dump
#   F1_MEMBERS F1_STU F1_NSINGLE F1_NPERM F1_FM F1_BASE10M   card F1 stage 1 (babylm.moonshot.f1_read): ms_f1_m0-m7,
#               the student, N-single, N-perm, F-M, and the 10-epoch 10M base seed(s) for G1
# Every read runs with --guard (dev_bpw_twin ex_unseen_chars leg) and, for mixtures, --calib ((T, lam) control on
# temps dumps). Outputs (small json/csv/png) land in OUT_DIR/<read>/ and a one-line-per-read summary.txt.
set -euo pipefail
OUT=${1:?usage: v2_readouts.sh OUT_DIR}
PY=${PY:-python}
CALIB=${CALIB---calib}  # CALIB= drops the calibrated control (it needs v2 twin dumps)
mkdir -p "$OUT"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

fetch() {  # ${BUCKET}/<run>/[pt_v2/]dev_v2.npz or ${BUCKET}/<name>.npz -> $W/<run or name>.npz (printed)
  local u=$1 n
  case $u in
    */pt_v2/dev_v2.npz) n=$(basename "$(dirname "$(dirname "$u")")") ;;
    */dev_v2.npz) n=$(basename "$(dirname "$u")") ;;
    *) n=$(basename "$u" .npz) ;;
  esac
  [ -e "$W/$n.npz" ] || gcloud storage cp -q "$u" "$W/$n.npz" >&2
  echo "$W/$n.npz"
}
many() { local x; for x in $1; do fetch "$x"; done; }

read_c3() {  # read_c3 NAME "MEMBERS" "REFS" [--calib]
  local name=$1 members refs calib=${4:-}
  members=$(many "$2")
  refs=$(many "${3:-}")
  # shellcheck disable=SC2086
  $PY -m babylm.moonshot.ceiling_report c3 $members ${refs:+--ref $refs} --guard $calib --out "$OUT/$name" \
    > "$OUT/$name.brief.json"
  echo "== $name: $(echo $members | wc -w) members, $(echo $refs | wc -w) refs" >> "$OUT/summary.txt"
  $PY - "$OUT/$name/c3.json" >> "$OUT/summary.txt" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
def legs(d, tag):
    rows = {x["K"]: x for x in d["rows"]}
    g = " ".join(f"g({k})={rows[k]['g_npw_raw']:.4f}" for k in sorted(rows) if k >= 2)
    print(f"  {tag}: {g}" + (f"  g_inf={d['g_inf_npw_raw']:.4f}" if "g_inf_npw_raw" in d else ""))
    v = d.get("vs_ref")
    if v:
        print(f"  {tag}: ref mean {v['ref_npw_raw']:.4f} (sd {v['ref_sd_npw_raw']:.4f}), mixture {v['mix_npw_raw']:.4f}, "
              f"gain {v['gain_npw_raw']:+.4f}")
legs(r, "full")
if "never_seen_char_guard" in r:
    legs(r["never_seen_char_guard"], "guarded")
c = r.get("calibrated")
if c:
    for row in c.get("rows", []):
        print("  calibrated K=%s " % row["K"] + " ".join(f"{k}={v:.4f}" for k, v in row.items()
                                                        if isinstance(v, float) and k.startswith("g_")))
    v = c.get("vs_ref") or {}
    if v:
        print("  calibrated vs_ref " + " ".join(f"{k}={x:.4f}" for k, x in v.items() if isinstance(x, float)))
EOF
}

$PY - <<'EOF'
from pathlib import Path
from babylm.data import corpus as C
d = Path(C.DEFAULT_DATA_DIR)
need = ["raw/dev", "bin/dev.bin", "bin/dev.words.bin", "bin/dev.meta.json", "bin/train_10M.bin"]
miss = [x for x in need if not (d / x).exists()]
if not (d / "analysis" / "linehash_train_100M.npy").exists() and not (d / "raw" / "train_100M").exists():
    miss.append("raw/train_100M or analysis/linehash_train_100M.npy")
if miss:
    raise SystemExit(f"data/babylm is missing {miss}: run this where the harness data is prepared")
EOF
: > "$OUT/summary.txt"
git rev-parse HEAD > "$OUT/commit.txt"

if [ -n "${C3:-}" ]; then
  read_c3 c3_gate0 "$C3" "" $CALIB
  read_c3 c3_solo_s0s2 "$(echo $C3 | cut -d' ' -f1-3)" "" $CALIB  # the solo-only K=3 beside the mixed K=8 (critic ref-0319)
fi
if [ -n "${C1:-}" ]; then
  set -- $(many "$C1")
  $PY -m babylm.moonshot.ceiling_report c1 --self "$1" --oracle "$2" --t100 "$3" --guard --out "$OUT/c1" \
    > "$OUT/c1.brief.json"
  echo "== c1" >> "$OUT/summary.txt"
  $PY -c "import json,sys; r=json.load(open(sys.argv[1])); print('  ' + json.dumps({k: v for k, v in r.items()
    if isinstance(v, (int, float))}))" "$OUT/c1/c1.json" >> "$OUT/summary.txt"
fi
for v in $(compgen -v LS_ | grep -v '^LS_REF$' || true); do
  [ -n "${LS_REF:-}" ] && read_c3 "ls_${v#LS_}_vs_e10" "${!v}" "$LS_REF" $CALIB
  [ -n "${ESTAR_REF:-}" ] && read_c3 "ls_${v#LS_}_vs_estar" "${!v}" "$ESTAR_REF" $CALIB
done
if [ -n "${PE3:-}" ]; then
  # shellcheck disable=SC2046
  $PY -m babylm.moonshot.pe_gate2_read --pe3 $(many "$PE3") --pe3_ens "$(fetch "$PE3_ENS")" --i3 $(many "$I3") \
    --f $(many "${F:-}") ${F_JSON:+--f_json $F_JSON} ${NJ_ENS:+--nj_ens "$(fetch "$NJ_ENS")"} --guard \
    --out "$OUT/pe_gate2" > /dev/null
  echo "== pe_gate2 (mini-ruler proxy)" >> "$OUT/summary.txt"
  $PY - "$OUT/pe_gate2/pe_gate2.json" >> "$OUT/summary.txt" <<'EOF2'
import json, sys
r = json.load(open(sys.argv[1]))
for leg in ("full", "guarded"):
    x = r.get(leg)
    if x:
        print(f"  {leg}: p_bar(PE3) {x['pe3_pbar_own_dump']:.4f} better F {x['better_F']:.4f} (vs F {x['pe3_vs_better_F']:+.4f}, "
              f"bar 2 floor {x['a_bar']:.4f}) B mean {x['B_mean']:.4f} J(PE3) {x['J_pe3']:.4f} J(I3) {x['J_i3']:.4f} "
              f"rho {x['rho']:.3f} (a) killed {x['a_killed']} (b) killed {x['b_killed']}"
              + (f" NJ {x['nj_pbar_own_dump']:.4f}" if "nj_pbar_own_dump" in x else ""))
EOF2
fi
if [ -n "${F1_MEMBERS:-}" ]; then
  # shellcheck disable=SC2046
  $PY -m babylm.moonshot.f1_read --members $(many "$F1_MEMBERS") --stu "$(fetch "$F1_STU")" \
    --nsingle "$(fetch "$F1_NSINGLE")" --nperm "$(fetch "$F1_NPERM")" --fm "$(fetch "$F1_FM")" \
    --base10m $(many "$F1_BASE10M") --guard --out "$OUT/f1" > /dev/null
  echo "== f1 stage 1 (T1 diagnostic, mini-ruler proxy)" >> "$OUT/summary.txt"
  $PY - "$OUT/f1/f1.json" >> "$OUT/summary.txt" <<'EOF2'
import json, sys
r = json.load(open(sys.argv[1]))
for leg in ("full", "guarded"):
    x = r.get(leg)
    if x:
        print(f"  {leg}: base@1M {x['base_1M']:.4f} base@10M {x['base_10M']:.4f} G1 {x['G1']:.4f} g(8) {x['g8']:.4f} "
              f"phi_ens8 {x['phi_ens8']:.3f} student {x['student']:.4f} phi_stu {x['phi_stu']:.3f} floor {x['floor_1M']:.4f} "
              "ahead " + " ".join(f"{k} {v:+.4f}" for k, v in x["student_ahead_by"].items())
              + f" killed {x['killed']} build_10M {x['build_10M_student']}")
EOF2
fi
cat "$OUT/summary.txt"
