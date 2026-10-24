"""Executor 3: calibrated reads for a named set of v2 temps dumps. argv: OUT.json NAME=PATH ... -- MIX=A+B ...
Per single and per mixture (token-level probability average): raw, calibrated, calibrated_t on full and guarded legs
(calib_control @ref-0033 via ceiling_report.calib_tool, same as ceiling_report c3 --calib)."""
import json, sys
import numpy as np
from babylm.moonshot import ceiling_report as R
out = sys.argv[1]; args = sys.argv[2:]; k = args.index("--")
paths = dict(a.split("=", 1) for a in args[:k]); mixes = dict(a.split("=", 1) for a in args[k + 1:])
ds = {n: R.Dump(p) for n, p in paths.items()}
cc = R.calib_tool(); S_, tok = cc.shared(), cc.tokenizer()
first = next(iter(ds.values()))
_, ids, words, ok, drop, *_ = cc.load_v2(first.path, S_, tok)
temps = first.temps
assert all(d.temps == temps for d in ds.values())
def cal(members):
    x = R.ensemble_tokens([d.nats for d in members]) if len(members) > 1 else members[0].nats
    o = cc.calibrate(x, ids, words, ok, drop, S_)
    cols = [d.nats_at() for d in members]
    mixed = [R.ensemble_tokens([c[j] for c in cols]) if len(cols) > 1 else cols[0][j] for j in range(len(temps) + 1)]
    for leg, v in cc.calibrate_t(mixed[0], np.stack(mixed[1:], 1), temps, ids, words, ok, drop, S_).items():
        o[leg].update(v)
    return {leg: {q: float(o[leg][q]) for q in ("raw", "calibrated", "calibrated_t")} | {"t": o[leg].get("t_even")} for leg in ("full", "guarded")}
import os
res = json.load(open(out)) if os.path.exists(out) else {"paths": paths, "mixes": mixes, "single": {}, "mix": {}}
res["mixes"].update(mixes)
for n, d in ds.items():
    if n in res["single"]: continue
    res["single"][n] = cal([d]); print("single", n, json.dumps(res["single"][n]), flush=True)
    json.dump(res, open(out, "w"), indent=1)
for m, spec in mixes.items():
    if m in res["mix"]: continue
    res["mix"][m] = cal([ds[x] for x in spec.split("+")]); print("mix", m, json.dumps(res["mix"][m]), flush=True)
    json.dump(res, open(out, "w"), indent=1)
