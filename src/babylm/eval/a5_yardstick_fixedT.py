"""a5_yardstick.py (moonshot @ref-0377) with critic bind ref-0385 (1): the overlap-removed ("ovr") view reuses the
(T, lam) picks that calib_control.calibrate_t made on full DEV ("all"), unchanged, instead of re-fitting them on the
masked subset. Everything else (members, mixing, legs, Y, x, intervals) is the moonshot tool as is.

Run from a checkout of wip-branch with this file copied to babylm/eval/:
    python -m babylm.eval.a5_yardstick_fixedT <the a5_yardstick.py arguments, e.g. the config and --json a5_out.json>
"""
import numpy as np

from babylm.moonshot import a5_yardstick as A


def calibrate_t_fixed(cc, x, xt, temps, ids, words, ok, drop, S, picks):
    """calibrate_t's read with given picks: the even-half pick is read on the odd half and vice versa."""
    ut, odd = S["u"][ids], S["odd"][words]
    tg = [1.0] + list(temps)
    cols = [x] + [xt[:, j].astype(np.float64) for j in range(len(temps))]

    def loss(t, lam):
        xc = cols[tg.index(t)]
        return xc if lam == 0 else -np.log((1 - lam) * np.exp(-xc) + lam * ut)

    out = {}
    for leg, m, den in (("full", ok, S["W"]), ("guarded", ok & ~drop, S["W"] - int(S["bad"].sum()))):
        p = picks[leg]
        v = (loss(p["t_even"], p["lam_t_even"])[m & odd].sum() + loss(p["t_odd"], p["lam_t_odd"])[m & ~odd].sum()) / den
        out[leg] = dict(p, calibrated_t=float(v))
    return out


class FixedTReader(A.Reader):
    def _read(self, members):
        cc, S = self.cc, self.S
        loaded = [cc.load_v2(p, S, self.tok) for p in members]
        _, ids, words, ok, drop, *_ = loaded[0]
        has_t = all(z[6] is not None for z in loaded)
        mix = (lambda cols: A.R.ensemble_tokens(cols)) if len(loaded) > 1 else (lambda cols: cols[0])
        x = mix([z[0] for z in loaded])
        xt = temps = None
        if has_t:
            temps = loaded[0][7]
            assert all(z[7] == temps for z in loaded), "members' temps differ"
            xt = np.stack([mix([z[6][:, j].astype(np.float64) for z in loaded]) for j in range(len(temps))], 1)
        out, picks = {}, None
        for view in A.VIEWS:  # "all" first: its picks are reused on "ovr"
            s, m = (S, ok) if view == "all" else (
                dict(S, W=int(self.keep.sum()), bad=S["bad"] & self.keep), ok & self.keep[words])
            raw = cc.calibrate(x, ids, words, m, drop, s)
            ct = None
            if has_t:
                if view == "all":
                    ct = picks = cc.calibrate_t(x, xt, temps, ids, words, m, drop, s)
                else:
                    ct = calibrate_t_fixed(cc, x, xt, temps, ids, words, m, drop, s, picks)
            out[view] = {leg: {"raw": raw[leg]["raw"], "cal_t": ct[leg]["calibrated_t"] if ct else None,
                               "t": [ct[leg]["t_even"], ct[leg]["t_odd"], ct[leg]["t_edge"]] if ct else None}
                         for leg in A.LEGS}
        return out


assert tuple(A.VIEWS)[0] == "all", A.VIEWS
A.Reader = FixedTReader

if __name__ == "__main__":
    A.main()
