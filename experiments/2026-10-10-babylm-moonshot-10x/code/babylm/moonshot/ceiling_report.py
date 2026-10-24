"""Ceiling-map readouts (docs/babylm/moonshot/ceiling_map.md). Diagnostics only: every number is labelled, never a
verdict.

    python -m babylm.moonshot.ceiling_report c3 DUMP... --out DIR [--ensemble DUMP] [--ref DUMP...]  # g(K), g_inf
    python -m babylm.moonshot.ceiling_report c1 --self DUMP --oracle DUMP --t100 DUMP --out DIR
    python -m babylm.moonshot.ceiling_report bias --col member=mean:D0,..,D7 --col ens8=mix:D0,..,D7
        --col c1self=D --col c1oracle=D --col t100=D [--col f1stu=D] [--diff A-B ...] --out DIR [--guard]

Headline numbers use THE registered scorer (dev_bpw.py, critic ref-0053) through its per-token dumps
(babylm.moonshot.dev_bpw_dump, `<run>/score/dev_pt.npz`; a DUMP argument may be the npz or a run dir holding it). A dump
reproduces its run's registered dev_bpw nats/word (checked when written). Each number is reported raw (the registered
number) and overlap-removed (words of DEV lines found verbatim in train_100M dropped, critic ref-0260).

C3: an ensemble's probability for each observed token is the members' mean (token-level mixture; members share one
tokenizer, so their token streams are identical and are asserted so). For K = 1..n, g(K) = mean member DEV - ensemble
DEV over member subsets of size K (all subsets when there are <= 30, else 30 fixed ones); fit g(K) = g_inf - b / K by
least squares over K >= 2. Label: "K x exposure, out of track".
--ref DUMP... adds "vs_ref": the reference runs' mean DEV (and SD) minus the all-member mixture's DEV, e.g. the legal
split (2 runs x 5 epochs mixed, 100M words) against the 10-epoch base seeds, or PE against B.
C1: f1 = (DEV(self) - DEV(oracle)) / (DEV(self) - DEV(T100)).
--blank_tok HF_DIR (any export's tokenizer) adds every number again under "blank_excluded": the nats of blank-line
newlines (a newline token after a newline token, never seen in training) and of the token after a blank line are
zeroed, words unchanged (the instruments' scorers.py events; critic ref-0020). Label: "approximate blank-line twin,
diagnostic" (the instruments' twin re-scores without the event; this drops it).
--guard [TOK_DIR] adds every number again under "never_seen_char_guard": the registered ex_unseen_chars rule of
babylm/eval/dev_bpw_twin.py @ ref-0026 (critic ref-0023), its own unseen_mask on the dump's own documents: a token is
dropped when any character of its span lies in a whitespace word holding one of the 85 never-seen characters
(whitespace-only tokens and BOS separators stay), and npw_raw = (N - NU) / (W - WU). A single dump's npw_raw is checked
against the ex_unseen_chars_nats_per_word of a dev_bpw_twin.json beside it, when there is one. npw_ovr on this leg
also has those tokens zeroed (diagnostic). Mixtures (no scorer json of their own) are read on the same rule, and a
gain is only ever taken within this instrument (critic ref-0023, same-instrument rule).
bias (the D1 bias map, design agent escalation 2 / experimenter 17:28Z): per band of train_10M whitespace-word count (0, 1-63, 64-8k,
8k+) x text slice (all; seen = the word's line, >= 8 words, found verbatim in train_10M; new = not) x leg (raw; ovr =
overlap-removed vs train_100M; guarded, with --guard), the nats/word of each --col (NAME=DUMP, NAME=mean:DUMPS = mean of
the singles, NAME=mix:DUMPS = their probability average) and each --diff A-B (A minus B; defaults: member-ens8,
ens8-c1oracle, c1oracle-t100, member-f1stu where those names exist). Labels: ens8 is K x exposure; c1oracle and t100
carry 100M information (ens8 - c1oracle is an UPPER bound on learnable bias); columns from different torch envs are
cross-env and void until the experimenter's v2e pair test passes; on the guarded leg the guarded tokens' nats are zeroed and
their words stay in each band's denominator. Writes bias.csv and bias.json.
--calib (c3 on v2 twin dumps) adds "calibrated": the calibrated-gain control (critic ref-0292) through the pinned
instrument babylm/tools/calib_control.py (experimenter ref-0033, calibrate() on the mixed per-token nats): g(K) raw and
calibrated on both legs (full, guarded) over the first --calib_subsets member subsets of each K, and vs_ref likewise.
When every dump carries nats_temps (dev_bpw_dump --temps, the registered grid), each side also gets calibrated_t
(calibrate_t, (T, lam) cross-fitted): a mixture's members are tempered at one shared T and then mixed, so (T, lam) is
picked on the mixture's own loss (experimenter 07:59Z).
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from babylm.data import corpus as C

MAX_SUBSETS = 30
C3_LABEL = "K x exposure, out of track"
CALIB_TOOL = Path(__file__).resolve().parents[1] / "tools" / "calib_control.py"
CALIB_SHA256 = "70c0eb5b2922c1bca300a9f3f3f1494c66fad85c69d2bc9c0a84dc5509089ca0"  # experimenter ref-0033, critic ref-0292
DEFAULT_TOK = "shared/babylm/base_r1/base_S_wd0p4_lr0p5_r1_ve0_tied/hf"


class Dump:
    """One model's registered-protocol per-token dump."""

    def __init__(self, path: str | Path):
        p = Path(path)
        if p.is_dir():
            p = p / "score" / "dev_pt.npz" if (p / "score" / "dev_pt.npz").exists() else p / "dev_pt.npz"
        z = np.load(p, allow_pickle=False)
        self.path = str(p)
        self.nats = z["nats"].astype(np.float64)
        fw, ft = z["file_words"].astype(np.int64), z["file_tokens"].astype(np.int64)
        self.files = [str(x) for x in z["files"]]
        self.file_words, self.file_tokens = fw, ft
        step = z["word_step"].astype(np.int64)
        word = np.empty(len(step), dtype=np.int64)
        start, woff = 0, 0
        for n_tok, n_w in zip(ft.tolist(), fw.tolist()):
            word[start:start + n_tok] = np.cumsum(step[start:start + n_tok]) + woff
            start, woff = start + n_tok, woff + n_w
        self.word = word  # global whitespace-word index in sorted-file order
        self.n_words = int(fw.sum())
        self.scale = 1.0  # per-word values x scale: a leg divided by fewer words than the dump holds (the guard)
        self.protocol = json.loads(str(z["protocol"])) if "protocol" in z.files else None
        self.temps = [float(t) for t in z["temps"]] if "temps" in z.files else None
        self._z = z if "nats_temps" in z.files else None

    def nats_at(self) -> list[np.ndarray]:
        """[nats at T = 1] + [nats at each of self.temps] (float64), for the temperature control."""
        xt = self._z["nats_temps"]
        return [self.nats] + [xt[:, j].astype(np.float64) for j in range(xt.shape[1])]

    def zeroed(self, mask: np.ndarray, n_words: int | None = None) -> "Dump":
        """Those tokens' nats zeroed; with n_words, every mean over all words becomes a sum / n_words."""
        d = copy.copy(self)
        d.nats = np.where(mask, 0.0, self.nats)
        d.scale = self.n_words / n_words if n_words else 1.0
        return d

    def per_word(self, nats: np.ndarray | None = None) -> np.ndarray:
        pw = np.bincount(self.word, weights=self.nats if nats is None else nats, minlength=self.n_words)
        return pw * self.scale if self.scale != 1.0 else pw


def keep_mask(files: list[str], dev_dir: Path, data_dir: Path | None = None) -> np.ndarray:
    """True for DEV words outside lines found verbatim (>= 8 words) in train_100M, in the dumps' word order (files
    sorted by name, words as text.split(); whitespace collapse does not change words)."""
    from babylm.analysis import p2_slices
    hashes = p2_slices.line_hashes("train_100M", data_dir or C.DEFAULT_DATA_DIR)
    keys = []
    for f in files:
        for line in (dev_dir / f).read_text(encoding="utf-8").split("\n"):
            lw = line.split()
            key = p2_slices.line_key(lw) if len(lw) >= p2_slices.OVERLAP_MIN_WORDS else 0
            keys.extend([key] * len(lw))
    k = np.array(keys, dtype=np.uint64)
    return ~((k != 0) & np.isin(k, hashes))


def guard_tokens(d: Dump, dev_dir: Path, tok_dir: str) -> tuple[np.ndarray, int]:
    """(token mask, WU) of dev_bpw_twin's ex_unseen_chars rule in the dump's token order: the DEV text re-tokenised with
    the dump's own protocol (documents for a twin dump, separators never dropped); token counts asserted equal."""
    from transformers import AutoTokenizer
    from babylm.eval import dev_bpw as D
    from babylm.eval import dev_bpw_twin as T
    tok = AutoTokenizer.from_pretrained(tok_dir)
    pr = d.protocol or {"ws": "collapse", "line_join": "newline", "tab_space": ""}
    out, wu = [], 0
    for f, n in zip(d.files, d.file_tokens.tolist()):
        text = D.load_text(str(dev_dir / f), "full", pr["ws"], pr["line_join"],
                           tuple(x for x in pr.get("tab_space", "").split(",") if x))
        parts = []
        for k, doc in enumerate(T.documents(text) if pr.get("twin") else [text]):
            enc = tok(doc, add_special_tokens=False, return_offsets_mapping=True)
            m, w = T.unseen_mask(doc, enc["offset_mapping"])
            parts += [np.zeros(1 if k else 0, dtype=bool), m]
            wu += w
        m = np.concatenate(parts)
        if len(m) != n:
            raise SystemExit(f"{f}: guard re-tokenised {len(m)} tokens, dump has {n}")
        out.append(m)
    return np.concatenate(out), wu


def twin_json_guard(path: str) -> float | None:
    """ex_unseen_chars_nats_per_word of a dev_bpw_twin.json beside a dump (its dir, the parent, or the parent's score/)."""
    p = Path(path).parent
    for q in (p, p.parent, p.parent / "score"):
        if (q / "dev_bpw_twin.json").exists():
            return json.loads((q / "dev_bpw_twin.json").read_text()).get("ex_unseen_chars_nats_per_word")
    return None


def blank_events(d: Dump, dev_dir: Path, tok_dir: str) -> np.ndarray:
    """True at blank-line newlines and at the token after a blank line, in the dump's token order (DEV re-tokenised
    with the dump's own protocol; the token counts are asserted equal)."""
    from transformers import AutoTokenizer
    from babylm.eval import dev_bpw as D
    tok = AutoTokenizer.from_pretrained(tok_dir)
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    assert len(nl) == 1, nl
    pr = d.protocol or {"ws": "collapse", "line_join": "newline", "tab_space": ""}
    if pr.get("twin"):
        raise SystemExit(f"{d.path} is a blank-line twin dump (metric v2): it has no blank-line events to drop")
    out = []
    for f, n in zip(d.files, d.file_tokens.tolist()):
        text = D.load_text(str(dev_dir / f), "full", pr["ws"], pr["line_join"],
                           tuple(x for x in pr.get("tab_space", "").split(",") if x))
        r = np.asarray(tok(text, add_special_tokens=False)["input_ids"])
        if len(r) != n:
            raise SystemExit(f"{f}: re-tokenised {len(r)} tokens, dump has {n}")
        blank = np.r_[False, (r[1:] == nl[0]) & (r[:-1] == nl[0])]
        out.append(blank | np.r_[False, (r[1:] != nl[0]) & blank[:-1]])
    return np.concatenate(out)


def dev(pw: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    return {"npw_ovr": float(pw[keep].mean()), "npw_raw": float(pw.mean())}


def ensemble_tokens(nats: list[np.ndarray]) -> np.ndarray:
    """Token-level probability average: -log mean_k exp(-nats_k)."""
    x = np.stack(nats).astype(np.float64)
    return -(np.logaddexp.reduce(-x, axis=0) - math.log(len(nats)))


def subsets(n: int, k: int) -> list[tuple[int, ...]]:
    allc = list(itertools.combinations(range(n), k))
    if len(allc) <= MAX_SUBSETS:
        return allc
    idx = np.random.Generator(np.random.PCG64(20261010 + k)).choice(len(allc), MAX_SUBSETS, replace=False)
    return [allc[i] for i in sorted(idx)]


def fit_ginf(ks: list[int], gs: list[float]) -> tuple[float, float]:
    """g(K) = g_inf - b / K by least squares; returns (g_inf, b)."""
    A = np.stack([np.ones(len(ks)), -1.0 / np.asarray(ks, dtype=np.float64)], axis=1)
    (g_inf, b), *_ = np.linalg.lstsq(A, np.asarray(gs, dtype=np.float64), rcond=None)
    return float(g_inf), float(b)


def c3(dumps: list[Dump], keep: np.ndarray, refs: list[Dump] = ()) -> dict:
    ref = dumps[0]
    for d in [*dumps[1:], *refs]:
        if d.files != ref.files or not np.array_equal(d.file_tokens, ref.file_tokens) or not np.array_equal(
                d.word, ref.word):
            raise SystemExit(f"{d.path}: token stream differs from {ref.path} (members must share a tokenizer)")
    single = [dev(d.per_word(), keep) for d in dumps]
    rows = []
    for k in range(1, len(dumps) + 1):
        ens, mem = [], []
        for s in subsets(len(dumps), k):
            nats = ensemble_tokens([dumps[i].nats for i in s]) if k > 1 else dumps[s[0]].nats
            ens.append(dev(ref.per_word(nats), keep))
            mem.append({q: float(np.mean([single[i][q] for i in s])) for q in ("npw_ovr", "npw_raw")})
        row = {"K": k, "subsets": len(ens), "label": C3_LABEL}
        for q in ("npw_raw", "npw_ovr"):
            row[f"ens_{q}"] = float(np.mean([e[q] for e in ens]))
            row[f"member_{q}"] = float(np.mean([m[q] for m in mem]))
            row[f"g_{q}"] = row[f"member_{q}"] - row[f"ens_{q}"]
        rows.append(row)
    out = {"dumps": [d.path for d in dumps], "single": single, "rows": rows, "label": C3_LABEL}
    fit_rows = [r for r in rows if r["K"] >= 2]
    if len(fit_rows) >= 2:
        for q in ("npw_raw", "npw_ovr"):
            g_inf, b = fit_ginf([r["K"] for r in fit_rows], [r[f"g_{q}"] for r in fit_rows])
            out[f"g_inf_{q}"], out[f"b_{q}"] = g_inf, b
    if refs:
        rs = [dev(r.per_word(), keep) for r in refs]
        v = {"refs": [r.path for r in refs], "label": f"mean of {len(refs)} reference runs - the {len(dumps)}-member "
             "mixture (positive = mixture better)"}
        for q in ("npw_raw", "npw_ovr"):
            x = np.array([r[q] for r in rs])
            v[f"ref_{q}"], v[f"ref_sd_{q}"] = float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")
            v[f"mix_{q}"] = rows[-1][f"ens_{q}"]
            v[f"gain_{q}"] = v[f"ref_{q}"] - v[f"mix_{q}"]
        out["vs_ref"] = v
    return out


def calib_tool():
    import hashlib
    import importlib.util
    if hashlib.sha256(CALIB_TOOL.read_bytes()).hexdigest() != CALIB_SHA256:
        raise SystemExit(f"{CALIB_TOOL}: sha256 differs from the pinned {CALIB_SHA256[:8]}")
    spec = importlib.util.spec_from_file_location("calib_control", CALIB_TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def calib_c3(dumps: list[Dump], refs: list[Dump], max_subsets: int, cc=None) -> dict:
    """g(K) and vs_ref, raw and calibrated, on the full and guarded legs (calib_control.calibrate on mixed nats)."""
    cc = cc or calib_tool()
    if not (dumps[0].protocol or {}).get("twin"):
        raise SystemExit("--calib reads v2 twin dumps (dev_bpw_dump --twin)")
    S, tok = cc.shared(), cc.tokenizer()
    _, ids, words, ok, drop, *_ = cc.load_v2(dumps[0].path, S, tok)
    allx = [*dumps, *refs]
    temps = allx[0].temps if all(getattr(d, "temps", None) for d in allx) else None
    if temps and any(d.temps != temps for d in allx):
        raise SystemExit("--calib: the dumps' temperature grids differ")

    def cal(members: list) -> dict:
        """calibrate (and calibrate_t with shared-T mixing) one side: a single dump or the mixture of several."""
        x = ensemble_tokens([d.nats for d in members]) if len(members) > 1 else members[0].nats
        out = cc.calibrate(x, ids, words, ok, drop, S)
        if temps:
            cols = [d.nats_at() for d in members]
            mixed = [ensemble_tokens([c[j] for c in cols]) if len(cols) > 1 else cols[0][j] for j in range(len(temps) + 1)]
            for leg, v in cc.calibrate_t(mixed[0], np.stack(mixed[1:], 1), temps, ids, words, ok, drop, S).items():
                out[leg].update(v)
        return out
    single = [cal([d]) for d in dumps]
    legs, qs = ("full", "guarded"), ("raw", "calibrated") + (("calibrated_t",) if temps else ())
    rows, mix = [], None
    for k in range(1, len(dumps) + 1):
        ss = subsets(len(dumps), k)[:max_subsets]
        ens = [cal([dumps[i] for i in s]) if k > 1 else single[s[0]] for s in ss]
        row = {"K": k, "subsets": len(ss)}
        for leg in legs:
            for q in qs:
                mem = np.mean([np.mean([single[i][leg][q] for i in s]) for s in ss])
                row[f"g_{leg}_{q}"] = float(mem - np.mean([e[leg][q] for e in ens]))
            row[f"lam_{leg}"] = [[e[leg]["lam_even"], e[leg]["lam_odd"]] for e in ens]
            if temps:
                row[f"t_{leg}"] = [[e[leg]["t_even"], e[leg]["t_odd"], e[leg]["t_edge"]] for e in ens]
        rows.append(row)
        mix = ens[0]
    out = {"label": "calibrated-gain control (critic ref-0292): comparator, never an entry", "tool": str(CALIB_TOOL),
           "tool_sha256": CALIB_SHA256, "temps": temps, "single": single, "rows": rows}
    if refs:
        rc = [cal([r]) for r in refs]
        out["vs_ref"] = {f"gain_{leg}_{q}": float(np.mean([r[leg][q] for r in rc]) - mix[leg][q])
                         for leg in legs for q in qs}
        out["vs_ref"]["refs"] = rc
    return out


def c1(self_d: Dump, oracle_d: Dump, t100_d: Dump, keep: np.ndarray) -> dict:
    d = {k: dev(x.per_word(), keep) for k, x in (("self", self_d), ("oracle", oracle_d), ("t100", t100_d))}
    out = {"dev": d, "label": "C1: oracle teacher (100M words; out of track); C1-self teacher = 2x exposure"}
    for q in ("npw_raw", "npw_ovr"):
        G = d["self"][q] - d["t100"][q]
        out[f"G_{q}"] = G
        out[f"f1_{q}"] = (d["self"][q] - d["oracle"][q]) / G if G > 0 else float("nan")
    return out


BIAS_BANDS = (("c10=0", 0, 0), ("c10=1-63", 1, 63), ("c10=64-8k", 64, 8000), ("c10=8k+", 8001, 10**12))
BIAS_DEFAULT_DIFFS = ("member-ens8", "ens8-c1oracle", "c1oracle-t100", "member-f1stu")


def word_bands(files: list[str], dev_dir: Path, data_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Per DEV word, in the dumps' word order (keep_mask's order): its train_10M whitespace-word count (c10) and whether
    its line (>= 8 words) appears verbatim in train_10M (seen)."""
    from babylm.analysis import p2_slices
    dd = data_dir or C.DEFAULT_DATA_DIR
    counts = p2_slices.word_counts("train_10M", dd)
    hashes = p2_slices.line_hashes("train_10M", dd)
    c10, keys = [], []
    for f in files:
        for line in (dev_dir / f).read_text(encoding="utf-8").split("\n"):
            lw = line.split()
            key = p2_slices.line_key(lw) if len(lw) >= p2_slices.OVERLAP_MIN_WORDS else 0
            c10.extend(counts.get(w, 0) for w in lw)
            keys.extend([key] * len(lw))
    k = np.array(keys, dtype=np.uint64)
    return {"c10": np.array(c10, dtype=np.int64), "seen": (k != 0) & np.isin(k, hashes)}


def bias_table(cols: dict[str, np.ndarray], c10: np.ndarray, seen: np.ndarray, keep: np.ndarray,
               diffs: list[tuple[str, str]], leg: str = "raw", drop: np.ndarray | None = None) -> list[dict]:
    """Rows of nats/word per (leg, text slice, band) for every column (per-word nats arrays) and A - B diff. With drop
    (the guarded leg: words holding a never-seen character, their tokens' nats already zeroed), each cell is the
    registered (N - NU) / (W - WU): the sum of the band's per-word nats over the band's non-dropped words, and `words`
    counts the non-dropped words."""
    rows = []
    for text, tmask in (("all", np.ones(len(c10), dtype=bool)), ("seen", seen), ("new", ~seen)):
        for band, lo, hi in (("all", 0, 10**12), *BIAS_BANDS):
            m = tmask & (c10 >= lo) & (c10 <= hi)
            if leg == "ovr":
                m = m & keep
            n = int((m & ~drop).sum()) if drop is not None else int(m.sum())
            row = {"leg": leg, "text": text, "band": band, "words": n, "word_share": n / len(c10)}
            for name, pw in cols.items():
                row[name] = float(pw[m].sum() / n) if n else float("nan")
            for a, b in diffs:
                row[f"{a}-{b}"] = row[a] - row[b]
            rows.append(row)
    return rows


def parse_col(spec: str) -> tuple[str, str, list[str]]:
    name, rest = spec.split("=", 1)
    kind, _, paths = rest.partition(":")
    if _ == "":
        return name, "single", [rest]
    if kind not in ("mean", "mix"):
        raise SystemExit(f"--col {spec}: kind must be mean: or mix:")
    return name, kind, paths.split(",")


def bias_cols(specs: list[str], mask: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], Dump, dict[str, Dump]]:
    """Per-word nats per column; with mask, the masked tokens' nats are zeroed first (the guarded leg). Also returns
    the reference dump and, per single-dump column, its Dump (for the registered-value check)."""
    cols, singles, ref = {}, {}, None
    for spec in specs:
        name, kind, paths = parse_col(spec)
        ds = [Dump(x) for x in paths]
        ref = ref or ds[0]
        for d in ds:
            if d.files != ref.files or not np.array_equal(d.word, ref.word):
                raise SystemExit(f"{d.path}: token stream differs from {ref.path} (one tokenizer, one protocol)")
        nats = [np.where(mask, 0.0, d.nats) if mask is not None else d.nats for d in ds]
        if kind == "mix":
            cols[name] = ref.per_word(ensemble_tokens(nats))
        elif kind == "mean":
            cols[name] = np.mean([ref.per_word(x) for x in nats], axis=0)
        else:
            cols[name] = ref.per_word(nats[0])
            singles[name] = ds[0]
    return cols, ref, singles


def bias_main(a: argparse.Namespace, dd: Path, dev_dir: Path, out: Path) -> dict:
    cols, ref, singles = bias_cols(a.col)
    names = list(cols)
    diffs = [tuple(x.split("-", 1)) for x in (a.diff or [d for d in BIAS_DEFAULT_DIFFS
                                                          if all(n in names for n in d.split("-"))])]
    for x, y in diffs:
        if x not in names or y not in names:
            raise SystemExit(f"--diff {x}-{y}: unknown column")
    keep = keep_mask(ref.files, dev_dir, dd)
    wb = word_bands(ref.files, dev_dir, dd)
    rows = bias_table(cols, wb["c10"], wb["seen"], keep, diffs, "raw") + bias_table(cols, wb["c10"], wb["seen"],
                                                                                   keep, diffs, "ovr")
    checks = []
    if a.guard:
        mask, wu = guard_tokens(ref, dev_dir, a.guard)
        gw = np.bincount(ref.word, weights=mask.astype(np.float64), minlength=ref.n_words) > 0
        if int(gw.sum()) != wu:
            raise SystemExit(f"guard: {int(gw.sum())} words hold a masked token but the rule counted WU = {wu}")
        gcols, _, _ = bias_cols(a.col, mask)
        grows = bias_table(gcols, wb["c10"], wb["seen"], keep, diffs, "guarded", drop=gw)
        top = next(r for r in grows if r["text"] == "all" and r["band"] == "all")
        for name, d in singles.items():
            reg = twin_json_guard(d.path)
            if reg is not None:
                ok_ = abs(top[name] - reg) <= 1e-6 * reg
                checks.append({"col": name, "dump": d.path, "guarded_all": top[name], "twin_json": reg, "ok": ok_})
                if not ok_:
                    raise SystemExit(f"{name} ({d.path}): guarded (all, all) {top[name]:.6f} vs registered {reg:.6f}")
        rows += grows
    with (out / "bias.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"label": "D1 bias map, diagnostic: ens8 is K x exposure; c1oracle/t100 carry 100M information, so "
                    "ens8 - c1oracle is an upper bound on learnable bias; cross-env columns are void until v2e",
           "cols": {n: a.col[i] for i, n in enumerate(names)}, "diffs": [f"{x}-{y}" for x, y in diffs],
           "seen_word_share": float(wb["seen"].mean()), "kept_word_share": float(keep.mean()),
           "twin_json_checks": checks, "rows": rows}
    (out / "bias.json").write_text(json.dumps(res, indent=1))
    print(json.dumps([r for r in rows if r["text"] == "all"], indent=1))
    return res


def plot_c3(res: dict, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [r["K"] for r in res["rows"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for q, ls in (("npw_raw", "-"), ("npw_ovr", "--")):
        ax.plot(ks, [r[f"g_{q}"] for r in res["rows"]], ls, marker="o", label=f"g(K), {q}")
    if "g_inf_npw_raw" in res:
        kk = np.linspace(2, max(ks), 50)
        ax.plot(kk, res["g_inf_npw_raw"] - res["b_npw_raw"] / kk, ":", label=f"fit, g_inf {res['g_inf_npw_raw']:.3f}")
    if "blank_excluded" in res:
        ax.plot(ks, [r["g_npw_raw"] for r in res["blank_excluded"]["rows"]], "-.", marker="x",
                label="g(K), npw_raw, blank-line events zeroed")
    if "never_seen_char_guard" in res:
        ax.plot(ks, [r["g_npw_raw"] for r in res["never_seen_char_guard"]["rows"]], linestyle=(0, (1, 3)), marker="+",
                label="g(K), npw_raw, never-seen-character words zeroed")
    ax.set_xlabel("K seeds averaged")
    ax.set_ylabel("mean member DEV - ensemble DEV (nats/word)")
    ax.set_title(f"C3 posterior channel ({C3_LABEL})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a3 = sub.add_parser("c3")
    a3.add_argument("dumps", nargs="+")
    a3.add_argument("--ref", nargs="+", default=[], help="reference runs' dumps (adds vs_ref)")
    a3.add_argument("--calib", action="store_true", help="add the calibrated-gain control (v2 twin dumps)")
    a3.add_argument("--calib_subsets", type=int, default=5)
    a3.add_argument("--ensemble", default=None, help="a jointly trained ensemble's own dump (card PE): the K = n "
                    "mixture of the member dumps must reproduce its registered DEV within 1e-4 nats/word (critic ref-0417)")
    a1 = sub.add_parser("c1")
    a1.add_argument("--self", dest="self_dump", required=True)
    a1.add_argument("--oracle", required=True)
    a1.add_argument("--t100", required=True)
    ab = sub.add_parser("bias")
    ab.add_argument("--col", action="append", required=True, help="NAME=DUMP | NAME=mean:D1,D2 | NAME=mix:D1,D2")
    ab.add_argument("--diff", action="append", default=None, help="A-B (column names); default: the standard four")
    for p in (a3, a1, ab):
        p.add_argument("--out", required=True)
        p.add_argument("--dev_dir", default=None, help="the raw DEV files the dumps were scored on")
        p.add_argument("--data_dir", default=None)
        p.add_argument("--blank_tok", default=None, help="an HF export dir: also report with blank-line events zeroed")
        p.add_argument("--guard", nargs="?", const=DEFAULT_TOK, default=None, metavar="TOK_DIR",
                       help="also report on dev_bpw_twin's ex_unseen_chars rule (TOK_DIR: any export's tokenizer)")
    a = ap.parse_args(argv)
    dd = Path(a.data_dir) if a.data_dir else C.DEFAULT_DATA_DIR
    dev_dir = Path(a.dev_dir) if a.dev_dir else dd / "raw" / "dev"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "bias":
        bias_main(a, dd, dev_dir, out)
        return
    n = len(a.dumps) if a.cmd == "c3" else 3
    ds = [Dump(x) for x in ([*a.dumps, *a.ref] if a.cmd == "c3" else (a.self_dump, a.oracle, a.t100))]
    keep = keep_mask(ds[0].files, dev_dir, dd)
    run = (lambda xs: c3(xs[:n], keep, xs[n:])) if a.cmd == "c3" else (lambda xs: c1(*xs, keep))
    res = run(ds)
    if a.blank_tok:
        ev = blank_events(ds[0], dev_dir, a.blank_tok)
        res["blank_excluded"] = {"label": "approximate blank-line twin, diagnostic", "events": int(ev.sum()),
                                 "nats_share": [float(d.nats[ev].sum() / d.nats.sum()) for d in ds],
                                 **run([d.zeroed(ev) for d in ds])}
    if a.guard:
        mask, wu = guard_tokens(ds[0], dev_dir, a.guard)
        g = {"label": "never-seen-character guard, dev_bpw_twin ex_unseen_chars rule (ref-0026): (N - NU) / (W - WU)",
             "words": wu, "tokens": int(mask.sum()), **run([d.zeroed(mask, d.n_words - wu) for d in ds])}
        checks = []
        for d, x in zip(ds, g["single"] if a.cmd == "c3" else []):
            ref = twin_json_guard(d.path)
            if ref is not None:
                ok_ = abs(x["npw_raw"] - ref) <= 1e-5 * ref
                checks.append({"dump": d.path, "npw_raw": x["npw_raw"], "twin_json": ref, "ok": ok_})
                if not ok_:
                    raise SystemExit(f"{d.path}: guarded {x['npw_raw']:.6f} vs dev_bpw_twin.json {ref:.6f}")
        g["twin_json_checks"] = checks
        res["never_seen_char_guard"] = g
    if a.cmd == "c3" and a.calib:
        res["calibrated"] = calib_c3(ds[:n], ds[n:], a.calib_subsets)
    if a.cmd == "c3":
        rows = (res["rows"] + [{**r, "K": f"{r['K']}nb"} for r in res.get("blank_excluded", {}).get("rows", [])]
                + [{**r, "K": f"{r['K']}g"} for r in res.get("never_seen_char_guard", {}).get("rows", [])])
        with (out / "c3.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        plot_c3(res, out / "c3.png")
    if a.cmd == "c3" and a.ensemble:
        e = dev(Dump(a.ensemble).per_word(), keep)
        mix = res["rows"][-1]
        diff = abs(mix["ens_npw_raw"] - e["npw_raw"])
        res["ensemble_check"] = {"dump": a.ensemble, "npw_raw": e["npw_raw"], "member_mixture_npw_raw":
                                 mix["ens_npw_raw"], "abs_diff": diff, "ok": diff <= 1e-4}
        if diff > 1e-4:
            (out / "c3.json").write_text(json.dumps(res, indent=1))
            raise SystemExit(f"member mixture {mix['ens_npw_raw']:.6f} vs ensemble {e['npw_raw']:.6f}: |diff| {diff:.2e}")
    res["kept_word_share"] = float(keep.mean())
    (out / f"{a.cmd}.json").write_text(json.dumps(res, indent=1))
    sides = ("blank_excluded", "never_seen_char_guard", "calibrated")
    brief = {k: v for k, v in res.items() if k not in ("rows", "single", *sides)}
    for side in sides:
        if side in res:
            brief[side] = {k: v for k, v in res[side].items() if k not in ("rows", "single")}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
