#!/usr/bin/env bash
# Fetch the AMLM authors' code at the pinned commit into $1 and verify every pinned hash. Pristine: no patches.
set -euo pipefail
DEST="${1:?usage: fetch.sh <dest_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
COMMIT=$(python3 -c "import json;print(json.load(open('$HERE/PIN.json'))['code']['commit'])")
git clone -q https://github.com/Leukas/babylm25 "$DEST"
git -C "$DEST" checkout -q "$COMMIT"
python3 - "$HERE/PIN.json" "$DEST" <<'PY'
import hashlib, json, sys, os
pin, dest = json.load(open(sys.argv[1])), sys.argv[2]
for f, h in pin["code"]["sha256"].items():
    got = hashlib.sha256(open(os.path.join(dest, f), "rb").read()).hexdigest()
    assert got == h, f"hash mismatch {f}: {got} != {h}"
print("AMLM code + data + tokenizer hashes verified at", pin["code"]["commit"])
PY
