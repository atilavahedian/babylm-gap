"""Count-based probe: how much of the gap is local word statistics?

Builds interpolated Kneser-Ney word trigram models (surface whitespace words, one
discount per order from count-of-counts) on train_10M and on train_100M, scores the dev
sample with them, and mixes each into the neural models' word probabilities with a
cross-fitted weight (fit on even documents, score odd, and vice versa).
  neural 10M + KN 10M   : local statistics in the 10M words that the network under-uses
                          (a legal strict-small attack: the count model sees each word once)
  neural 10M + KN 100M  : local statistics that only the 100M words hold (an oracle, not legal)
Words the count model never saw get probability 0 from it. The count model predicts only
the word, not the whitespace before it, so its share is slightly optimistic.
--joint fits one cross-fitted mixture of several components (neural models, count models,
the copy cache) and reports its gain over the best neural member next to the sum of the
two-way gains, so the overlap between add-ons is measured rather than assumed.

Usage:
  python -m babylm.instruments.ngram --tag preread_v1 --data-root ~/babylm_data \
      --models gpt2_10m/chck_100M,gptbert_cf_10m/chck_100M,gpt2_100m/chck_1000M \
      --joint gptbert_cf_10m/chck_100M+gpt2_10m/chck_100M+kn:train_10M+cache
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from babylm.instruments import corpus as C
from babylm.instruments import slices as S
from babylm.instruments.anatomy import cache_logp, fit_weights, mix_nll
from babylm.instruments.score import load_rows

M1, M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * M1
        z = (z ^ (z >> np.uint64(27))) * M2
        return z ^ (z >> np.uint64(31))


def h2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _mix(_mix(a.astype(np.uint64)) ^ b.astype(np.uint64))


def h3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return _mix(h2(a, b) ^ c.astype(np.uint64))


def encode_surface(files: list[Path], vocab: dict[str, int]) -> np.ndarray:
    out = []
    for path in files:
        buf: list[int] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if C.is_header(line):
                    continue
                for w in line.split():
                    i = vocab.get(w)
                    if i is None:
                        i = vocab[w] = len(vocab)
                    buf.append(i)
        out.append(np.asarray(buf, dtype=np.int32))
    return np.concatenate(out)


def _lookup(keys: np.ndarray, vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(keys, q)
    pos[pos == len(keys)] = 0
    hit = keys[pos] == q
    return np.where(hit, vals[pos], 0)


def _discount(counts: np.ndarray) -> float:
    n1, n2 = int((counts == 1).sum()), int((counts == 2).sum())
    return n1 / (n1 + 2 * n2) if n1 + n2 else 0.5


class KN3:
    def __init__(self, ids: np.ndarray, nv: int):
        u, v, w = ids[:-2], ids[1:-1], ids[2:]
        key, first, cnt = np.unique(h3(u, v, w), return_index=True, return_counts=True)
        U, V, W = u[first], v[first], w[first]
        self.tri_k, self.tri_c = key, cnt.astype(np.float64)
        self.D3 = _discount(cnt)
        ck, inv = np.unique(h2(U, V), return_inverse=True)
        self.ctx_k, self.ctx_c, self.ctx_n = ck, np.bincount(inv, weights=self.tri_c), np.bincount(inv).astype(np.float64)
        sk, sfirst, sn = np.unique(h2(V, W), return_index=True, return_counts=True)
        self.suf_k, self.suf_n = sk, sn.astype(np.float64)
        self.D2 = _discount(sn)
        self.mid_tot = np.bincount(V, minlength=nv).astype(np.float64)
        self.mid_types = np.bincount(V[sfirst], minlength=nv).astype(np.float64)
        bk, bfirst = np.unique(h2(ids[:-1], ids[1:]), return_index=True)
        cont = np.bincount(ids[1:][bfirst], minlength=nv).astype(np.float64)
        self.p1 = cont / cont.sum()
        self.stats = {"tokens": int(len(ids)), "trigram_types": int(len(key)), "bigram_types": int(len(bk)),
                      "D3": round(self.D3, 4), "D2": round(self.D2, 4)}

    def prob(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """u, v, w: int arrays, -1 = missing (chunk start) or unknown word."""
        nv = len(self.p1)
        known = (w >= 0) & (w < nv)
        wc = np.where(known, w, 0)
        p1 = np.where(known, self.p1[wc], 0.0)
        vok = (v >= 0) & (v < nv)
        vc = np.where(vok, v, 0)
        tot = np.where(vok, self.mid_tot[vc], 0.0)
        n_vw = np.where(vok & known, _lookup(self.suf_k, self.suf_n, h2(vc, wc)), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p2 = np.where(tot > 0, np.maximum(n_vw - self.D2, 0) / tot + self.D2 * self.mid_types[vc] / tot * p1, p1)
            uok = (u >= 0) & (u < nv) & vok
            uc = np.where(uok, u, 0)
            c_uv = np.where(uok, _lookup(self.ctx_k, self.ctx_c, h2(uc, vc)), 0.0)
            n_uv = np.where(uok, _lookup(self.ctx_k, self.ctx_n, h2(uc, vc)), 0.0)
            c_uvw = np.where(uok & known, _lookup(self.tri_k, self.tri_c, h3(uc, vc, wc)), 0.0)
            p3 = np.where(c_uv > 0, np.maximum(c_uvw - self.D3, 0) / c_uv + self.D3 * n_uv / c_uv * p2, p2)
        return np.where(known, p3, 0.0)


def dev_contexts(vocab: dict[str, int], split: str = "dev") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    us, vs, ws = [], [], []
    for r in load_rows(split):
        ids = np.array([vocab.get(x, -1) for x in r["text"].split()], dtype=np.int64)
        ws.append(ids)
        vs.append(np.r_[-1, ids[:-1]])
        us.append(np.r_[-1, -1, ids[:-2]][: len(ids)])
    return np.concatenate(us), np.concatenate(vs), np.concatenate(ws)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="preread_v1")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--models", default="gpt2_10m/chck_100M,gptbert_cf_10m/chck_100M,gpt2_100m/chck_1000M")
    ap.add_argument("--corpora", default="train_10M,train_100M")
    ap.add_argument("--joint", action="append", default=[],
                    help="components joined by +: run/rev (neural), kn:<corpus> (count model), cache")
    args = ap.parse_args()
    out = S.REPO / "results" / "babylm" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    F = S.load_feats("dev")
    par = F["unit"] % 2
    vocab: dict[str, int] = {}
    kn_logp, info = {}, {}
    for corpus in args.corpora.split(","):
        t0 = time.time()
        ids = encode_surface(C.split_files(args.data_root, corpus), vocab)
        kn = KN3(ids, len(vocab))
        del ids
        u, v, w = dev_contexts(vocab)
        p = kn.prob(u, v, w)
        del kn
        with np.errstate(divide="ignore"):
            kn_logp[corpus] = np.log(p)
        info[corpus] = {"seconds": round(time.time() - t0, 1), "dev_oov": float((p == 0).mean()),
                        "kn_alone_nats_on_known": float(-np.log(p[p > 0]).mean())}
        print(corpus, info[corpus], flush=True)
    rows = []
    for spec in [x for x in args.models.split(",") if x]:
        r, rv = spec.split("/")
        s = S.load_score(r, rv, "dev")
        if s is None:
            print(f"skip {spec}: missing scores")
            continue
        lm = -s["word_nll"].astype(np.float64)
        for corpus, lk in kn_logp.items():
            mixed = np.zeros(len(lm))
            lams = []
            for fit in (0, 1):
                wts = fit_weights(np.vstack([lm[par == fit], lk[par == fit]]))
                lams.append(round(float(wts[1]), 4))
                ev = par != fit
                mixed[ev] = mix_nll(np.vstack([lm[ev], lk[ev]]), wts)
            gain = -lm - mixed
            ci = S.bootstrap_gap(gain, F["unit"], np.zeros(len(gain), dtype=int), 1)[4]
            fb, flabels = S.dims(F)["freq10"]
            row = {"model": spec, "count_model": corpus, "nats": float(-lm.mean()), "mixed": float(mixed.mean()),
                   "gain": float(gain.mean()), "gain_lo": ci[0], "gain_hi": ci[1], "kn_weight": json.dumps(lams)}
            for k, lab in enumerate(flabels):
                row[f"gain_freq10_{lab}"] = float(gain[fb == k].mean()) if (fb == k).any() else np.nan
            rows.append(row)
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items() if not k.startswith("gain_freq")}), flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out / "ngram_mix.csv", index=False)
        (out / "ngram_info.json").write_text(json.dumps(info, indent=1))
    jrows = [r for r in (joint(spec, kn_logp, F) for spec in args.joint) if r]
    if jrows:
        pd.DataFrame(jrows).to_csv(out / "joint_mix.csv", index=False)


def _xfit(L: np.ndarray, par: np.ndarray) -> tuple[np.ndarray, list]:
    mixed, ws = np.zeros(L.shape[1]), []
    for fit in (0, 1):
        w = fit_weights(L[:, par == fit])
        ws.append([round(float(x), 4) for x in w])
        ev = par != fit
        mixed[ev] = mix_nll(L[:, ev], w)
    return mixed, ws


def joint(spec: str, kn_logp: dict, F: dict) -> dict | None:
    """One cross-fitted mixture of every component in spec, its gain over the best neural member,
    and each component's two-way gain with that member (their sum minus the joint gain = overlap)."""
    names, comps = spec.split("+"), []
    for c in names:
        if c == "cache":
            comps.append(cache_logp("dev"))
        elif c.startswith("kn:"):
            if c[3:] not in kn_logp:
                print(f"skip joint {spec}: {c} not built (add it to --corpora)")
                return None
            comps.append(kn_logp[c[3:]])
        else:
            s = S.load_score(*c.split("/"), "dev")
            if s is None:
                print(f"skip joint {spec}: missing scores for {c}")
                return None
            comps.append(-s["word_nll"].astype(np.float64))
    L, par = np.vstack(comps), F["unit"] % 2
    neural = [i for i, c in enumerate(names) if c != "cache" and not c.startswith("kn:")]
    best = min(neural, key=lambda i: -L[i].mean())
    base = -L[best]
    mixed, ws = _xfit(L, par)
    gain = base - mixed
    ci = S.bootstrap_gap(gain, F["unit"], np.zeros(len(gain), dtype=int), 1)[4]
    pair = {names[i]: float((base - _xfit(L[[best, i]], par)[0]).mean()) for i in range(len(names)) if i != best}
    row = {"joint": spec, "best_member": names[best], "best_nats": float(base.mean()), "joint_nats": float(mixed.mean()),
           "gain": float(gain.mean()), "gain_lo": ci[0], "gain_hi": ci[1], "sum_of_pair_gains": sum(pair.values()),
           "pair_gains": json.dumps({k: round(v, 4) for k, v in pair.items()}), "weights": json.dumps(ws)}
    fb, flabels = S.dims(F)["freq10"]
    for k, lab in enumerate(flabels):
        row[f"gain_freq10_{lab}"] = float(gain[fb == k].mean()) if (fb == k).any() else np.nan
    print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items() if not k.startswith("gain_freq")}), flush=True)
    return row


if __name__ == "__main__":
    main()
