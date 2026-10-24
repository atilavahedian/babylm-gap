"""C3 seed-pair reference for card F2 (executor 3): per-pair g(2), raw on all 28 pairs of the 8 C3 seeds (f1 _dumps_v2,
no temps) and raw + calibrated + calibrated_t on the 10 pairs of s0-s4 (temps back-fill dumps). Mean and SD over pairs."""
import itertools, json, sys
from pathlib import Path
import numpy as np
from babylm.data import corpus as C
from babylm.moonshot import ceiling_report as R

out = Path(sys.argv[1]); raw_paths = sys.argv[2].split(","); cal_paths = sys.argv[3].split(",")
dd = C.DEFAULT_DATA_DIR; dev_dir = dd / "raw" / "dev"
res = {}

def legs_raw(ds):
    keep = R.keep_mask(ds[0].files, dev_dir, dd)
    mask, wu = R.guard_tokens(ds[0], dev_dir, R.DEFAULT_TOK)
    gd = [d.zeroed(mask, d.n_words - wu) for d in ds]
    o = {}
    for leg, xs in (("full", ds), ("guarded", gd)):
        single = [xs[0].per_word(x.nats).mean() for x in xs]
        o[leg] = {"single": single, "pairs": {}}
        for i, j in itertools.combinations(range(len(xs)), 2):
            mix = xs[0].per_word(R.ensemble_tokens([xs[i].nats, xs[j].nats])).mean()
            o[leg]["pairs"][f"{i}-{j}"] = {"mix": float(mix), "g": float((single[i] + single[j]) / 2 - mix)}
    return o

r = legs_raw([R.Dump(p) for p in raw_paths])
res["raw28"] = {"dumps": raw_paths, **r}
ds = [R.Dump(p) for p in cal_paths]
cc = R.calib_tool()
S_, tok = cc.shared(), cc.tokenizer()
_, ids, words, ok, drop, *_ = cc.load_v2(ds[0].path, S_, tok)
temps = ds[0].temps
def cal(members):
    x = R.ensemble_tokens([d.nats for d in members]) if len(members) > 1 else members[0].nats
    o = cc.calibrate(x, ids, words, ok, drop, S_)
    cols = [d.nats_at() for d in members]
    mixed = [R.ensemble_tokens([c[j] for c in cols]) if len(cols) > 1 else cols[0][j] for j in range(len(temps) + 1)]
    for leg, v in cc.calibrate_t(mixed[0], np.stack(mixed[1:], 1), temps, ids, words, ok, drop, S_).items():
        o[leg].update(v)
    return o
single = [cal([d]) for d in ds]
pairs = {}
for i, j in itertools.combinations(range(len(ds)), 2):
    m = cal([ds[i], ds[j]])
    pairs[f"{i}-{j}"] = {leg: {q: float((single[i][leg][q] + single[j][leg][q]) / 2 - m[leg][q]) for q in ("raw", "calibrated", "calibrated_t")}
                         | {"mix_" + q: float(m[leg][q]) for q in ("raw", "calibrated_t")} for leg in ("full", "guarded")}
    print("pair", i, j, json.dumps(pairs[f"{i}-{j}"]), flush=True)
res["cal10"] = {"dumps": cal_paths, "single": [{leg: {q: float(s[leg][q]) for q in ("raw", "calibrated", "calibrated_t")} for leg in ("full", "guarded")} for s in single], "pairs": pairs}
out.mkdir(parents=True, exist_ok=True)
(out / "c3_pairref.json").write_text(json.dumps(res, indent=1))
L = []
for leg in ("full", "guarded"):
    g = np.array([v["g"] for v in res["raw28"][leg]["pairs"].values()])
    L.append(f"raw28 {leg}: g2 mean {g.mean():.4f} sd {g.std(ddof=1):.4f} min {g.min():.4f} max {g.max():.4f} (n {len(g)})")
for leg in ("full", "guarded"):
    for q in ("raw", "calibrated_t"):
        g = np.array([v[leg][q] for v in pairs.values()])
        L.append(f"cal10 {leg} {q}: g2 mean {g.mean():.4f} sd {g.std(ddof=1):.4f} min {g.min():.4f} max {g.max():.4f} (n {len(g)})")
(out / "summary.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
