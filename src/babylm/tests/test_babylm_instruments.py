# Moved from tests/ (o3stxz) to babylm/tests/ in the clean tree.
"""Math checks for the BabyLM instruments (no checkpoints or data needed)."""
import numpy as np

from babylm.instruments.anatomy import fit_weights, mix_nll
from babylm.instruments.bottleneck import effective_words
from babylm.instruments.cells import mean_ci
from babylm.instruments.ngram import KN3
from babylm.instruments.prize import GROUPS, band_shares, new_text_parts, strat_weights
from babylm.instruments.score import format_text
from babylm.instruments.signature import ratio_ci
from babylm.instruments.units import doc_starts


def test_kneser_ney_normalises():
    rng = np.random.default_rng(0)
    nv = 30
    kn = KN3(rng.integers(0, nv, 5000).astype(np.int32), nv)
    for u, v in [(1, 2), (3, 3), (-1, 4), (-1, -1), (29, 0)]:
        w = np.arange(nv)
        p = kn.prob(np.full(nv, u), np.full(nv, v), w)
        assert abs(p.sum() - 1) < 1e-9


def test_em_recovers_mixture_weights():
    rng = np.random.default_rng(1)
    p1, p2 = rng.dirichlet(np.ones(20)), rng.dirichlet(np.ones(20))
    x = np.where(rng.uniform(size=50000) < 0.3, rng.choice(20, 50000, p=p2), rng.choice(20, 50000, p=p1))
    lp = np.log(np.vstack([p1[x], p2[x]]))
    w = fit_weights(lp)
    assert abs(w[1] - 0.3) < 0.03 and abs(w.sum() - 1) < 1e-9
    assert mix_nll(lp, w).mean() <= min(-lp[0].mean(), -lp[1].mean())


def test_effective_words_interpolates_in_log_words():
    ws, loss = np.array([10e6, 100e6]), np.array([6.0, 5.0])
    w, flag = effective_words(ws, loss, 5.5)
    assert flag == "=" and abs(w - np.sqrt(10e6 * 100e6)) < 1
    assert effective_words(ws, loss, 6.5)[1] == "<="
    assert effective_words(ws, loss, 4.5)[1] == ">="


def test_ratio_ci_point_estimate():
    chunk = np.repeat(np.arange(50), 20)
    a = np.ones(1000)
    d = np.where(np.arange(1000) % 2 == 0, 0.4, 0.2)
    num, den = np.arange(1000) % 2 == 0, np.arange(1000) % 2 == 1
    point, lo, hi = ratio_ci(d, a, num, den, chunk)
    assert abs(point - 2.0) < 1e-12 and lo <= point <= hi


def test_doc_starts_blank_lines_and_switchboard_pieces():
    lines = ["a b", "c", "", "", "d", "", "e f"]
    assert doc_starts(lines, "childes").tolist() == [0, 4, 6]
    assert doc_starts(["x"] * 4500, "switchboard").tolist() == [0, 2000, 4000]


def test_format_text_keeps_words():
    r = {"text": "*CHI:\thi  there\n\n  bye ", "source": "childes"}
    assert format_text(r, "newline_space", ("childes",)) == "*CHI: hi  there\n \n   bye "
    assert format_text(r, "newline", (), collapse=True) == "*CHI: hi there\nbye"
    for args in [("newline", ()), ("newline_space", ("childes",)), ("newline", ("childes",), True)]:
        assert format_text(r, *args).split() == r["text"].split()


def test_band_shares_add_up():
    rng = np.random.default_rng(2)
    n = 4000
    d, grp = rng.normal(0.8, 1.0, n), rng.integers(0, len(GROUPS), n)
    unit, src = np.repeat(np.arange(40), n // 40), np.repeat(np.arange(40) % 2, n // 40)
    keep = rng.uniform(size=n) < 0.9
    bs = band_shares(d, grp, src, unit, None, keep, reps=50)
    assert abs(bs["word_share"][:, 0].sum() - 1) < 1e-9 and abs(bs["gap_share"][:, 0].sum() - 1) < 1e-9
    assert abs(bs["contrib"][:, 0].sum() - d[keep].mean()) < 1e-9
    share = np.array([0.25, 0.75])
    bs = band_shares(d, grp, src, unit, share, keep, reps=50)
    want = sum(w * d[keep & (src == i)].mean() for i, w in enumerate(share))
    assert abs(bs["contrib"][:, 0].sum() - want) < 1e-9


def test_strat_weights_keep_every_source():
    stratum = np.array([0, 0, 0, 1, 2, 2])
    W = strat_weights(stratum, 200)
    assert W.shape == (200, 6)
    for st in (0, 1, 2):
        assert (W[:, stratum == st].sum(1) == (stratum == st).sum()).all()


def test_new_text_parts_subtract_seen_text_gap():
    rng = np.random.default_rng(3)
    n = 4000
    grp, grpt = rng.integers(0, len(GROUPS), n), rng.integers(0, len(GROUPS), n)
    unit = np.repeat(np.arange(40), n // 40)
    per_band = np.array([6.0, 2.5, 1.0, 0.4])
    d = per_band[grp]
    p = new_text_parts(d, grp, unit, per_band[grpt], grpt, unit, [[0, 1], [2, 3]], reps=20)
    assert np.allclose(p, 0)
    p = new_text_parts(d, grp, unit, np.zeros(n), grpt, unit, [[0, 1], [2, 3]], reps=20)
    assert abs(p[0, 0] - d[grp < 2].sum() / n) < 1e-9 and abs(p[1, 0] - d[grp >= 2].sum() / n) < 1e-9


def test_mean_ci_clusters_by_document():
    rng = np.random.default_rng(0)
    unit = np.repeat(np.arange(40), 50)
    d = rng.normal(0, 1, 40)[unit] + 0.3  # all variance between documents
    point, lo, hi = mean_ci(d, unit)
    assert abs(point - d.mean()) < 1e-12 and lo < point < hi
    naive = 1.96 * d.std() / np.sqrt(len(d))  # pretends the 2,000 words are independent
    assert (hi - lo) / 2 > 3 * naive


def test_bands_partition_words_and_drop_markup_from_never_seen():
    from babylm.instruments.bands import BANDS, masks
    from babylm.instruments.slices import CHAT_MARKUP

    fl = ["0", "1", "2-3", "4-7", "8-15", "16-63", "64-255", "256-1023", "1k-8k", "8k+"]
    words = ["the", "⌈yeah⌉", "zyx", "cat", "↗"]
    f10 = np.array([fl.index("8k+"), fl.index("0"), fl.index("0"), fl.index("1k-8k"), fl.index("0")])
    F = {"markup": np.array([not CHAT_MARKUP.isdisjoint(w) for w in words])}
    m = masks(F, {"freq10": (f10, fl)})
    assert F["markup"].tolist() == [False, True, False, False, True]
    assert sum(m[b].astype(int) for b, _ in BANDS).tolist() == [1] * len(words)
    assert m["never seen"].tolist() == [False, True, True, False, True]
    assert m["never seen, no markup"].tolist() == [False, False, True, False, False]
    assert m["all, no markup"].sum() == 3


def test_scorer_windows_score_every_token_once():
    from babylm.instruments.scorers import CTX, STRIDE, T, block_jobs, sliding_jobs

    for n in (5, CTX - 1, CTX, CTX + 1, 3 * CTX + 17, 2 * T + 5):
        ids = list(range(1, n + 1))
        for jobs, first in ((sliding_jobs(ids, 0), 0), (sliding_jobs(ids, None), 1), (block_jobs(ids), 1)):
            seen = np.zeros(n, int)
            for win, s, p0 in jobs:
                seen[p0:p0 + s] += 1
                # the scored targets are the last s tokens of the window, and they are ids[p0:p0+s]
                assert win[len(win) - s:] == ids[p0:p0 + s]
            assert seen[first:].tolist() == [1] * (n - first) and seen[:first].sum() == 0
        # registered windows (dev_bpw): BOS + up to CTX-1 tokens; after the first, >= STRIDE-1 tokens of context
        for win, s, p0 in sliding_jobs(ids, 0)[1:]:
            assert win[0] == 0 and len(win) <= CTX and len(win) - 1 - s >= STRIDE - 1


def test_token_words_attach_whitespace_to_the_next_word():
    from babylm.instruments.devfull_docs import token_words

    text = "ab cd\nef"
    offsets = [(0, 1), (1, 2), (2, 5), (5, 6), (6, 8)]  # 'a' 'b' ' cd' '\n' 'ef'
    assert token_words(text, offsets).tolist() == [0, 0, 1, 2, 2]
