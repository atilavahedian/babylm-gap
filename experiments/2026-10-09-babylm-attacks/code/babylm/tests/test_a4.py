"""CPU tests for attack A4 (closed-system re-expression): babylm.a4.{ngram,kvcache,data,generate,audit,ledger},
babylm.harness.cycle and its train.py wiring (--init_from / --sources), babylm.configs.make_a4_jobs."""
from __future__ import annotations

import csv
import gzip
import json
import re
import subprocess

import numpy as np
import pytest
import torch

from babylm.a4 import audit as AU
from babylm.a4 import data as D
from babylm.a4 import generate as GEN
from babylm.a4 import kvcache as KV
from babylm.a4 import ledger as LG
from babylm.a4 import ngram as NG
from babylm.a4 import t2 as T2
from babylm.a4 import unseen as UN
from babylm.a4 import verdict as VD
from babylm.configs import make_a4_jobs as MJ
from babylm.configs import make_a4_t2_jobs as MT
from babylm.data import corpus as C
from babylm.harness import cycle as CY
from babylm.harness import levers as LV
from babylm.harness import train as TR
from babylm.models import baseline_reg
from babylm.moonshot import kd as KD
from babylm.tests.conftest import build_data_dir, tiny_args
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

COMMON = ("--no_pt", "--milestone_scale", "0.002", "--seed", "1")


def read(run, name):
    return list(csv.DictReader((run / name).open()))


@pytest.fixture(scope="module")
def a4_dir(tmp_path_factory):
    """Its own synthetic data dir (the halves and synthetic splits are written into it)."""
    return build_data_dir(tmp_path_factory.mktemp("a4_data"))


@pytest.fixture(scope="module")
def g_run(a4_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("a4_runs") / "G"
    TR.main(tiny_args(a4_dir, out, "--epochs", "2", "--save", "--no_full_dev", *COMMON))
    return out


@pytest.fixture(scope="module")
def tok():
    from babylm.data import prepare as P
    return P.load_tokenizer()


# ------------------------------------------------------------------------------------------------
# n-gram hashing and the copy block
# ------------------------------------------------------------------------------------------------
def test_hashes_agree_across_numpy_rows_and_torch():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 16384, 500)
    h = NG.ngram_hashes(s, 8)
    rows = np.stack([s[i:i + 8] for i in range(len(s) - 7)])
    assert np.array_equal(h, NG.hash_rows(rows))
    assert np.array_equal(h, NG.hash_rows_torch(torch.from_numpy(rows)).numpy())
    assert len(np.unique(h)) == len({tuple(r) for r in rows.tolist()})  # no collision on distinct rows here


def test_block_index_is_exact():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 6, 4000)  # tiny alphabet: contexts repeat, so runs have many tokens
    n = 4
    succ: dict[tuple, set] = {}
    for i in range(len(s) - n + 1):
        succ.setdefault(tuple(s[i:i + n - 1]), set()).add(int(s[i + n - 1]))
    bi = NG.BlockIndex.build(s, n)
    ctx = torch.from_numpy(rng.integers(0, 7, (64, n - 1)))
    for row, got in zip(ctx.tolist(), bi.blocked_sets(ctx)):
        assert got == succ.get(tuple(row), set())
    lg = torch.zeros(64, 8)
    lg[:, 7] = float("-inf")  # an already-masked id stays masked and is not counted
    removed = bi.mask(lg, ctx)
    for r, row in enumerate(ctx.tolist()):
        want = succ.get(tuple(row), set())
        assert set(torch.nonzero(torch.isinf(lg[r])).flatten().tolist()) == want | {7}
        assert int(removed[r]) == len(want)
    full = torch.tensor([[0, 0, 0]])  # every allowed token blocked: the row is left unblocked and counted
    if succ.get((0, 0, 0)) == set(range(6)):
        lg = torch.zeros(1, 6)
        assert int(bi.mask(lg, full)[0]) == 0 and bi.fallbacks == 1 and torch.isfinite(lg).all()


# ------------------------------------------------------------------------------------------------
# KV cache
# ------------------------------------------------------------------------------------------------
def rand_model(extra=None, tie=False, reg=False, L=4):
    cfg = ModelConfig(vocab_size=96, seq_len=64, n_layer=L, n_head=2, n_embd=32, tie_embeddings=tie, extra=extra or {})
    torch.manual_seed(0)
    m = (baseline_reg.build(cfg) if reg else baseline.build(cfg)).eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3 + (1.0 if p.ndim == 1 else 0.0))
    return m


@pytest.mark.parametrize("kw", [{}, {"extra": {"n_value_embeds": 2}}, {"tie": True, "extra": {"n_value_embeds": 0}},
                                {"extra": {"unet": False, "x0_inject": False, "embnorm": False}},
                                {"reg": True, "extra": {"dropout": 0.1}}, {"L": 1}])
def test_cached_decoding_matches_forward(kw):
    m = rand_model(**kw)
    prompts = torch.randint(1, 96, (3, 7), generator=torch.Generator().manual_seed(2))
    assert KV.self_check(m, prompts, steps=10) < 1e-4


def test_sample_stop_ban_and_determinism():
    m = rand_model()
    prompts = torch.randint(1, 96, (16, 5), generator=torch.Generator().manual_seed(3))
    kw = dict(stop_id=0, ban_stop_steps=4, allowed_vocab=3)  # ids 0-2 only: the stop id comes up often
    a = KV.sample(m, prompts, 20, generator=torch.Generator().manual_seed(4), **kw)
    b = KV.sample(m, prompts, 20, generator=torch.Generator().manual_seed(4), **kw)
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.lengths, b.lengths)
    assert a.stopped.any() and (a.lengths[a.stopped] >= 4).all()
    for r in range(16):
        L = int(a.lengths[r])
        assert (a.tokens[r, :L] != 0).all() and (a.tokens[r, :L] < 3).all()
        if a.stopped[r]:
            assert int(a.tokens[r, L]) == 0
    assert a.stats["steps"] <= 20 and a.stats["flops_per_row"] > 0


def test_sampling_never_completes_an_indexed_ngram():
    m = rand_model()
    rng = np.random.default_rng(5)
    stream = rng.integers(1, 12, 3000)  # dense index over ids 1..11: most contexts block several tokens
    n = 3
    bi = NG.BlockIndex.build(stream, n)
    have = {tuple(stream[i:i + n]) for i in range(len(stream) - n + 1)}
    prompts = torch.from_numpy(rng.integers(1, 12, (32, 6)))
    for block_from in (0, 3):
        out = KV.sample(m, prompts, 24, allowed_vocab=12, block=bi, block_from=block_from, stop_id=None,
                        generator=torch.Generator().manual_seed(6))
        seq = torch.cat([prompts, out.tokens], 1).tolist()
        for row in seq:
            for j in range(6 + block_from, len(row)):
                assert tuple(row[j - n + 1:j + 1]) not in have or bi.fallbacks > 0
        assert (out.blocked_steps > 0).any() and (out.blocked_mass > 0).any()


# ------------------------------------------------------------------------------------------------
# data: halves, seeds, synthetic splits, N2
# ------------------------------------------------------------------------------------------------
def test_halves_partition_train_10M(a4_dir):
    metas = D.make_halves(a4_dir)
    tr = C.load_split("train_10M", a4_dir)
    hA, hB = (C.load_split(n, a4_dir) for n in D.HALF_NAMES)
    assert sorted(metas[0]["doc_indices"] + metas[1]["doc_indices"]) == list(range(len(tr.doc_starts())))
    assert hA.n_words + hB.n_words == tr.n_words and hA.n_tokens + hB.n_tokens == tr.n_tokens
    b = D.doc_bounds(tr)
    toks = np.asarray(tr.tokens)
    want = np.concatenate([toks[b[k]:b[k + 1]] for k in metas[0]["doc_indices"]])
    assert np.array_equal(np.asarray(hA.tokens), want)
    assert int(C.word_initial(np.asarray(hA.words)).sum()) == hA.n_words  # word ids re-based and contiguous
    for s in C.SOURCES:  # every source split (switchboard is one document in the fixture)
        a, bb = hA.meta["per_source"].get(s, {}).get("words", 0), hB.meta["per_source"].get(s, {}).get("words", 0)
        assert a + bb == tr.meta["per_source"][s]["words"]


def test_seed_starts(a4_dir, tok):
    data = C.load_split("train_10M", a4_dir)
    t = np.asarray(data.tokens)
    nl = LV.newline_ids(tok)
    s = D.seed_starts(t, nl, 16)
    starts = set(LV.line_starts(t, nl, C.BOS_ID).tolist())
    assert len(s) > 50 and (np.diff(s) >= 16).all()
    for x in s.tolist():
        assert (t[x] == C.BOS_ID and x + 1 in starts) or x in starts
        assert not (t[x + 1:x + 16] == C.BOS_ID).any()
    o = D.seed_order(s, 0)
    assert sorted(o.tolist()) == s.tolist() and not np.array_equal(o, s)


def test_synth_split_roundtrip_and_n2(a4_dir, tmp_path):
    texts = ["*CHI: the dog ran home.\n\nA: I think you'd better go . yes", "  ", "Hello there, big world!  Fine (laughs)",
             "one two three four five six seven. eight nine ten?"]
    meta = D.write_synth_split(texts, [1, 2, 6, 0], tmp_path / "s", {"x": 1}, seed=0)
    data = C.load_split("synth", tmp_path / "s")
    kept = [D.clean_doc(t) for t in texts if D.clean_doc(t)]
    assert meta["docs"] == 3 and data.n_words == sum(len(t.split()) for t in kept)
    assert int(C.word_initial(np.asarray(data.words)).sum()) == data.n_words
    got, srcs = D.read_synth_docs(tmp_path / "s")
    assert sorted(got) == sorted(kept) and sorted(srcs) == [0, 1, 6] and "self" in meta["per_source"]
    D.make_shuffled(tmp_path / "s", tmp_path / "n2", seed=0)
    sh, srcs2 = D.read_synth_docs(tmp_path / "n2")
    assert srcs2 == srcs
    strip = lambda w: w.rstrip(".!?,;:\"')]")  # noqa: E731
    for a, b in zip(got, sh):
        assert len(a.split()) == len(b.split())
        for la, lb in zip(a.split("\n"), b.split("\n")):
            wa, wb = la.split(), lb.split()
            assert sorted(map(strip, wa)) == sorted(map(strip, wb)) and len(wa) == len(wb)
            if wa and D.SPEAKER.match(wa[0]):
                assert wb[0] == wa[0]
            if wa and D.PUNCT_ONLY.match(wa[-1]):
                assert wb[-1] == wa[-1]


def test_shuffle_keeps_sentence_ends():
    w = D.shuffle_text("*MOT: where is the big red ball ? look at that one.", np.random.default_rng(0)).split()
    assert len(w) == 12 and w[0] == "*MOT:" and w[7] == "?" and w[-1].endswith(".")
    assert sorted(w[1:7]) == sorted("where is the big red ball".split())
    assert sorted(x.rstrip(".") for x in w[8:]) == sorted("look at that one".split())


# ------------------------------------------------------------------------------------------------
# the cycle sampler and the train.py wiring
# ------------------------------------------------------------------------------------------------
def make_syn(a4_dir, out, n_docs=40, seed=0):
    rng = np.random.default_rng(seed)
    words = "the cat sat on a mat dog ran home I think you'd better go. Yeah. no!".split()
    texts = ["\n".join(" ".join(rng.choice(words, rng.integers(3, 12))) for _ in range(rng.integers(2, 6)))
             for _ in range(n_docs)]
    D.write_synth_split(texts, [6] * n_docs, out, {}, seed=seed)
    return out


def test_cycle_sampler_exposure_interleave_and_prefix(a4_dir, tmp_path):
    syn = make_syn(a4_dir, tmp_path / "syn")
    real = CY.load_source("real:train_10M:2", a4_dir)
    s = CY.load_source(f"synthetic:{syn}:1", a4_dir)
    spec = LV.EpochSpec("stream", 32)
    cs = CY.CycleSampler([real, s], spec, 32 * 8, seed=3)
    tot = {k: 0 for k in CY.KINDS}
    for i in range(cs.total_steps):  # exact exposure: word-initial flags at each window's target positions
        x, y, _, _, words = cs.batch_at(i)
        for r, w in enumerate(range(i * cs.batch_seqs, (i + 1) * cs.batch_seqs)):
            so = cs.sources[int(cs.w_src[w])]
            st = int(cs.w_start[w])
            wi = C.word_initial(np.asarray(so.data.words))
            assert int(wi[st + 1:st + 33].sum()) == int(cs.w_words[w])
            assert torch.equal(y[r], torch.from_numpy(np.asarray(so.data.tokens[st + 1:st + 33], dtype=np.int64)))
        for k in CY.KINDS:
            tot[k] += words[k]
    assert tot == cs.planned_words and tot["synthetic"] > 0
    n = np.asarray(cs.windows_per_source, dtype=float)  # proportional interleave: every prefix tracks the shares
    cum = np.cumsum(cs.w_src == 1)
    assert np.abs(cum - np.arange(1, len(cum) + 1) * n[1] / n.sum()).max() <= 1.0
    long = CY.CycleSampler([CY.load_source("real:train_10M:4", a4_dir)], spec, 32 * 8, seed=3)
    mine = cs.w_start[cs.w_src == 0]  # A4a's real epochs = the first epochs of N0' in the same order
    assert np.array_equal(mine, long.w_start[:len(mine)])
    sd = cs.state_dict()
    cs2 = CY.CycleSampler([real, s], spec, 32 * 8, seed=3)
    cs2.load_state_dict(sd)
    with pytest.raises(ValueError):
        CY.CycleSampler([real], spec, 32 * 8, seed=3).load_state_dict(sd)


def test_inert_init_from_reproduces_the_port(a4_dir, tmp_path):
    """--init_from a step-0 checkpoint of the same run (same initial weights, 0 words) changes nothing: the wiring
    (loading, lineage, warmup, logging) perturbs no number."""
    args = ("--epochs", "2", "--steps", "8", *COMMON)
    TR.main(tiny_args(a4_dir, tmp_path / "z", "--epochs", "2", "--steps", "0", "--save", *COMMON))
    a = TR.main(tiny_args(a4_dir, tmp_path / "a", *args))
    b = TR.main(tiny_args(a4_dir, tmp_path / "b", *args, "--init_from", str(tmp_path / "z" / "resume.pt")))
    assert "cycle" not in a["config"] and "exposure" not in a and "words_real" not in read(tmp_path / "a", "train_log.csv")[0]
    assert [r["loss"] for r in read(tmp_path / "a", "train_log.csv")] == [r["loss"] for r in read(tmp_path / "b", "train_log.csv")]
    assert a["full_eval"]["nats"] == b["full_eval"]["nats"] and b["exposure"]["total"] == a["words_seen"]


def test_cycle_run_lineage_resume_and_refusals(a4_dir, g_run, tmp_path):
    syn = make_syn(a4_dir, tmp_path / "syn", n_docs=60)
    g = json.loads((g_run / "result.json").read_text())
    cyc = ("--init_from", str(g_run / "resume.pt"), "--warmup_frac", "0.1", "--sources", "real:train_10M:1",
           f"synthetic:{syn}:1", "--exposure_extra", f"G_A=@{g_run}/result.json", "gen_x=1000")
    r = TR.main(tiny_args(a4_dir, tmp_path / "c", *cyc, "--no_full_dev", *COMMON))
    ex = r["exposure"]
    assert ex["prior_words"] == g["words_seen"] and ex["extra_words"] == {"G_A": g["words_seen"], "gen_x": 1000}
    assert ex["cycle_words"] == ex["planned_cycle_words"] and ex["cycle_words"]["synthetic"] > 0
    assert r["words_seen"] == g["words_seen"] + sum(ex["cycle_words"].values())
    assert ex["total"] == r["words_seen"] + g["words_seen"] + 1000 and ex["within_cap"]
    rows = read(tmp_path / "c", "train_log.csv")
    assert int(rows[-1]["words_real"]) + int(rows[-1]["words_synthetic"]) == sum(ex["cycle_words"].values())
    assert int(rows[0]["words_seen"]) > g["words_seen"] and ex["warmup_steps"] >= 1
    assert all(int(x["words_seen"]) > g["words_seen"] for x in read(tmp_path / "c", "curves.csv"))  # pre-fired
    # preemption: a paused cycle resumes bit-exactly
    TR.main(tiny_args(a4_dir, tmp_path / "d", *cyc, "--no_full_dev", *COMMON, "--save", "--max_session_steps", "3"))
    d = TR.main(tiny_args(a4_dir, tmp_path / "d", *cyc, "--no_full_dev", *COMMON, "--save", "--resume", "auto"))
    assert d["sessions"] == 2 and d["exposure"]["cycle_words"] == ex["cycle_words"]
    for p in (tmp_path / "d").glob("checkpoints/*.pt"):
        assert torch.load(p, weights_only=False)["lineage"]["prior_words"] == g["words_seen"]
    assert [x["loss"] for x in read(tmp_path / "d", "train_log.csv")] == [x["loss"] for x in rows]
    with pytest.raises(SystemExit, match="R0"):
        TR.main(tiny_args(a4_dir, tmp_path / "e", *cyc, "--exposure_cap", str(ex["total"] - 1), *COMMON))
    with pytest.raises(SystemExit, match="cfg differs"):
        TR.main(tiny_args(a4_dir, tmp_path / "f", *cyc, "--n_layer", "3", *COMMON))
    with pytest.raises(SystemExit, match="exclusive"):
        TR.main(tiny_args(a4_dir, tmp_path / "g", *cyc, "--window_plan", "line:32", *COMMON))
    led = LG.main([str(tmp_path / "c"), "--base_epochs", "10"])["arms"][0]
    assert led["words_total"] == ex["total"] and led["within_R0"]
    assert led["flops"]["G_A_train"] == pytest.approx(led["flops"]["G_train"]) and led["flops"]["gen_x_train"] is None
    assert led["flops_total"] is None  # gen_x has no run to read FLOPs from


def test_cycle_line_windows_with_mntp(a4_dir, tmp_path):
    TR.main(tiny_args(a4_dir, tmp_path / "g", "--epochs", "1", "--steps", "2", "--save", "--mntp_frac", "0.25",
                      "--no_full_dev", *COMMON))
    syn = make_syn(a4_dir, tmp_path / "syn")
    r = TR.main(tiny_args(a4_dir, tmp_path / "c", "--init_from", str(tmp_path / "g" / "resume.pt"), "--mntp_frac",
                          "0.25", "--sources", "real:train_10M:1", f"synthetic:{syn}:1", "--cycle_window", "line:32",
                          "--steps", "4", "--no_full_dev", *COMMON))
    assert r["status"] == "ok" and r["exposure"]["window"] == "line:32" and r["config"]["model_cfg"]["seq_len"] == 32
    assert sum(r["exposure"]["cycle_words"].values()) == r["words_seen"] - r["exposure"]["prior_words"] > 0


def test_round1_all_on_levers_through_generation_and_cycle(a4_dir, tmp_path):
    """The round-1 all-on lever set (baseline_reg + dropout, no value embeds, tied, head wd, MNTP) as G and student."""
    lev = ("--model", "baseline_reg", "--extra", "dropout=0.1", "--extra", "n_value_embeds=0", "--tie_embeddings",
           "--head_wd", "0.1", "--mntp_frac", "0.25", "--no_full_dev", *COMMON)
    TR.main(tiny_args(a4_dir, tmp_path / "g", "--epochs", "1", "--steps", "6", "--save", *lev))
    st = GEN.main(["--ckpt", str(tmp_path / "g" / "resume.pt"), "--data_dir", str(a4_dir), "--batch", "16",
                   "--prompt_len", "16", "--n_new", "32", "--device", "cpu", "--out", str(tmp_path / "gen" / "a"),
                   "--seed_split", "train_10M", "--block_split", "train_10M", "--rows", "32"])
    assert st["self_check_max_logit_diff"] < 1e-4
    assert st["cfg"]["vocab_size"] > 16384  # MNTP pads the vocab with its mask id; sampling masks ids >= 16384
    D.build_from_shards(tmp_path / "syn", [tmp_path / "gen" / "a.jsonl.gz"], words=200)
    r = TR.main(tiny_args(a4_dir, tmp_path / "c", "--init_from", str(tmp_path / "g" / "resume.pt"), "--sources",
                          "real:train_10M:1", f"synthetic:{tmp_path / 'syn'}:1", "--steps", "4", *lev))
    assert r["status"] == "ok" and r["exposure"]["cycle_words"]["synthetic"] > 0


# ------------------------------------------------------------------------------------------------
# the T2 arm: KD on the synthetic rows, plain CE on the real rows; corpus words capped, synthetic uncapped
# ------------------------------------------------------------------------------------------------
def _kd_brute(zs, zt, y, rows, alpha, tau):
    """Per row: real rows CE, synthetic rows (1 - alpha) CE + alpha tau^2 KL; summed over scored targets / count."""
    tot, n, k = 0.0, int((y >= 0).sum()), 0
    for b in range(zs.shape[0]):
        m = y[b] >= 0
        ce = torch.nn.functional.cross_entropy(zs[b][m], y[b][m], reduction="sum")
        if rows[b]:
            lq = torch.log_softmax(zs[b][m] / tau, -1)[:, :zt.shape[-1]]
            lp = torch.log_softmax(zt[k][m] / tau, -1)
            tot = tot + (1 - alpha) * ce + alpha * tau ** 2 * (lp.exp() * (lp - lq)).sum()
            k += 1
        else:
            tot = tot + ce
    return tot / n


@pytest.mark.parametrize("alpha,tau", [(0.5, 1.0), (1.0, 1.0), (0.3, 2.0), (0.0, 1.0)])
def test_t2_kd_rows_terms(alpha, tau):
    g = torch.Generator().manual_seed(7)
    B, T, Vs, Vt = 5, 6, 11, 9
    zs = torch.randn(B, T, Vs, generator=g, requires_grad=True)
    zt = torch.randn(B, T, Vt, generator=g)
    y = torch.randint(0, Vt, (B, T), generator=g)
    y[0, :2] = -1
    y[3, 4:] = -1
    every = torch.ones(B, dtype=torch.bool)  # all rows synthetic: exactly the moonshot's kd_terms
    a, sa = T2.kd_rows_terms(zs, zt, y, every, alpha, tau)
    b, sb = KD.kd_terms(zs, zt, y, alpha, tau)
    assert float(a) == pytest.approx(float(b), rel=1e-6, abs=1e-7)
    assert sa["kd_ce"] == pytest.approx(sb["kd_ce"], rel=1e-6) and sa["syn_tok"] == 1.0
    ga, = torch.autograd.grad(a, zs)
    gb, = torch.autograd.grad(b, zs)
    assert torch.allclose(ga, gb, atol=1e-7)
    none = torch.zeros(B, dtype=torch.bool)  # no synthetic rows: plain CE, the teacher is not needed
    c, sc = T2.kd_rows_terms(zs, None, y, none, alpha, tau)
    ce = torch.nn.functional.cross_entropy(zs.view(B * T, Vs), y.view(-1), ignore_index=-1)
    assert float(c) == pytest.approx(float(ce), rel=1e-6) and sc["syn_tok"] == 0.0 and sc["kd_kl"] == 0.0
    rows = torch.tensor([True, False, True, True, False])  # mixed: per-row brute force, loss and gradient
    d, sd = T2.kd_rows_terms(zs, zt[rows], y, rows, alpha, tau)
    e = _kd_brute(zs, zt[rows], y, rows, alpha, tau)
    assert float(d) == pytest.approx(float(e), rel=1e-5)
    gd, = torch.autograd.grad(d, zs)
    ge, = torch.autograd.grad(e, zs)
    assert torch.allclose(gd, ge, atol=1e-6)
    n_syn = int((y[rows] >= 0).sum())
    assert sd["syn_tok"] == pytest.approx(n_syn / int((y >= 0).sum()))
    assert sd["kd_ce"] == pytest.approx(float(ce), rel=1e-6)  # the logged loss is the plain path's CE
    if alpha > 0:
        with pytest.raises(ValueError, match="exactly the synthetic rows"):
            T2.kd_rows_terms(zs, zt, y, rows, alpha, tau)


def test_t2_cycle_sampler_marks_synthetic_rows(a4_dir, tmp_path):
    syn = make_syn(a4_dir, tmp_path / "syn")
    cs = CY.CycleSampler([CY.load_source("real:train_10M:1", a4_dir), CY.load_source(f"synthetic:{syn}:2", a4_dir)],
                         LV.EpochSpec("stream", 32), 32 * 8, seed=5)
    seen = 0
    for i in range(cs.total_steps):
        cs.next()
        want = np.array([cs.sources[int(s)].kind == "synthetic"
                         for s in cs.w_src[i * cs.batch_seqs:(i + 1) * cs.batch_seqs]])
        assert np.array_equal(cs.last_syn, want)
        seen += int(want.sum())
    assert 0 < seen < cs.total_steps * cs.batch_seqs


def test_capped_words_rules():
    kinds = {"real": 30, "synthetic": 500}
    assert CY.capped_words("r0", 10, {"G": 7}, kinds) == 547
    assert CY.capped_words("t2", 10, {"G": 7}, kinds) == 47
    with pytest.raises(ValueError):
        CY.capped_words("t3", 0, {}, kinds)


def test_t2_cycle_run(a4_dir, g_run, tmp_path):
    syn = make_syn(a4_dir, tmp_path / "syn", n_docs=60)
    G = max((g_run / "checkpoints").glob("*.pt"), key=lambda p: torch.load(p, weights_only=False)["words_seen"])
    g_words = torch.load(G, weights_only=False)["words_seen"]
    cyc = ("--init_from", str(G), "--sources", "real:train_10M:1", f"synthetic:{syn}:2", "--steps", "6",
           "--no_full_dev", *COMMON)
    t2 = ("--kd_teacher", str(G), "--kd_rows", "synthetic", "--exposure_rule", "t2")
    plain = TR.main(tiny_args(a4_dir, tmp_path / "p", *cyc))
    ex = plain["exposure"]
    capped = g_words + ex["cycle_words"]["real"]
    assert ex["total"] > capped and "capped_words" not in ex and plain["config"]["levers"].get("kd") is None
    r = TR.main(tiny_args(a4_dir, tmp_path / "t", *cyc, *t2, "--exposure_cap", str(capped)))  # r0 would refuse
    e = r["exposure"]
    assert e["rule"] == "t2" and e["tier"] == T2.TIER == CY.RULES["t2"]
    assert e["capped_words"] == e["planned_capped"] == capped and e["within_cap"] and e["total"] > capped
    assert e["cycle_words"] == ex["cycle_words"]
    kd = r["config"]["levers"]["kd"]
    assert kd["alpha"] == 0.5 and kd["temp"] == 1.0 and kd["teacher"]["words_seen"] == g_words
    steps = [json.loads(x) for x in (tmp_path / "t" / "log.jsonl").read_text().splitlines()]
    steps = [x for x in steps if x["type"] == "train"]
    assert len(steps) == 6 and all(0.0 < x["syn_tok"] < 1.0 for x in steps)
    assert all(np.isfinite([x["kd_kl"], x["ce_real"], x["ce_syn"], x["kd_t_label"], x["kd_t_ent"]]).all()
               for x in steps)
    assert steps[0]["kd_kl"] < 1e-5 < steps[-1]["kd_kl"]  # the student starts as the teacher, then moves
    assert steps[0]["kd_ce"] == pytest.approx(float(read(tmp_path / "p", "train_log.csv")[0]["loss"]), rel=1e-5)
    # alpha 0: the KD-rows path is the plain cycle (real and synthetic rows on CE, same mean over targets)
    z = TR.main(tiny_args(a4_dir, tmp_path / "z", *cyc, *t2, "--kd_alpha", "0", "--exposure_cap", str(capped)))
    assert z["status"] == "ok"
    for u, v in zip(read(tmp_path / "z", "train_log.csv"), read(tmp_path / "p", "train_log.csv")):
        assert float(u["loss"]) == pytest.approx(float(v["loss"]), rel=1e-4)
    with pytest.raises(SystemExit, match="T2: planned corpus exposure"):
        TR.main(tiny_args(a4_dir, tmp_path / "x", *cyc, *t2, "--exposure_cap", str(capped - 1)))
    with pytest.raises(SystemExit, match="R0"):
        TR.main(tiny_args(a4_dir, tmp_path / "x", *cyc, "--exposure_cap", str(capped)))
    for bad in (("--kd_rows", "synthetic"),  # no teacher
                ("--kd_rows", "synthetic", "--kd_teacher", str(G), "--mntp_frac", "0.25")):
        with pytest.raises(SystemExit, match="--kd_rows synthetic needs"):
            TR.main(tiny_args(a4_dir, tmp_path / "x", *cyc, *bad))
    with pytest.raises(SystemExit, match="--kd_rows synthetic needs"):
        TR.main(tiny_args(a4_dir, tmp_path / "x", "--kd_rows", "synthetic", "--kd_teacher", str(G), *COMMON))
    with pytest.raises(SystemExit, match="applies to a cycle"):
        TR.main(tiny_args(a4_dir, tmp_path / "x", "--exposure_rule", "t2", *COMMON))
    led = LG.main([str(tmp_path / "t"), "--base_epochs", "10"])["arms"][0]
    assert led["words_total"] == e["total"] and not led["within_R0"]  # every-word count: outside R0.2
    assert led["rule"] == "t2" and led["words_capped"] == capped and led["within_cap"] and led["tier"] == T2.TIER
    teach = sum(x["syn_tok"] * (x["tokens_seen"] - p) for x, p in zip(steps, [0] + [y["tokens_seen"] for y in steps]))
    assert led["flops"]["kd_teacher"] == pytest.approx(r["flops_per_token"] / 3 * teach)
    g_res = json.loads((g_run / "result.json").read_text())
    assert led["flops"]["G_train"] == pytest.approx(LG.train_flops(g_res) * g_words / g_res["words_seen"])
    assert "T2 ok: corpus" in LG.table([led])

# ------------------------------------------------------------------------------------------------
# generation, audits, ledger and jobs end to end
# ------------------------------------------------------------------------------------------------
def test_generate_build_audit_ledger(a4_dir, g_run, tmp_path, tok):
    common = ["--ckpt", str(g_run / "resume.pt"), "--data_dir", str(a4_dir), "--batch", "16", "--prompt_len", "16",
              "--n_new", "32", "--device", "cpu"]
    st = GEN.main([*common, "--out", str(tmp_path / "g" / "a"), "--seed_split", "train_10M", "--block_split",
                   "train_10M", "--words", "600"])
    assert st["words"] >= 600 and st["self_check_max_logit_diff"] < 1e-4 and st["block"]["keys"] > 0
    assert st["flops"] > 0 and st["seeds"] > 0 and st["blocked_step_rate"] is not None
    rows = D.read_shard(tmp_path / "g" / "a.jsonl.gz")
    assert sum(r["words"] for r in rows) == st["words"] and all(r["src"] < 6 for r in rows)
    n1 = GEN.main([*common, "--out", str(tmp_path / "g" / "n1"), "--selfseed", "--rows", "16"])
    assert n1["rows"] == 16 and all(r["src"] == D.SELF_SRC for r in D.read_shard(tmp_path / "g" / "n1.jsonl.gz"))
    probe = GEN.main([*common, "--out", str(tmp_path / "g" / "p"), "--seed_split", "train_10M", "--rows", "20"])
    assert probe["rows"] == 20 and probe["block"] is None
    meta = D.build_from_shards(tmp_path / "syn", [tmp_path / "g" / "a.jsonl.gz"], words=500)
    assert 500 <= meta["words"] < st["words"] + 1 and meta["shards"][0]["gen"]["flops"] == st["flops"]
    # the block holds on the decoded, re-tokenized text up to a handful of boundary 8-grams
    cp = AU.copy_audit(str(tmp_path / "syn"), str(tmp_path / "g" / "p.jsonl.gz"), str(a4_dir), tmp_path / "au" / "c.json",
                       n_docs=20, min_len=8)
    assert cp["overlap"]["synthetic"]["8gram"] < 0.05 < cp["overlap"]["dev"]["8gram"]
    assert (tmp_path / "au" / "c_nn.csv").exists() and cp["nearest_neighbour"]["synthetic"]["docs"] > 0
    ev = tmp_path / "eval" / "fast"
    ev.mkdir(parents=True)
    syn_doc = max(D.read_synth_docs(tmp_path / "syn")[0], key=lambda t: len(t.split()))
    tr_doc = json.loads((a4_dir / "docs.json").read_text())["train_10M"]["childes"]
    tr_line = next(" ".join(x.split()) for x in tr_doc if len(x.split()) >= 6)
    items = [{"good": " ".join(syn_doc.split()[:20]), "bad": tr_line}, {"a": "nothing to see here at all ok", "b": 1}]
    (ev / "x.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    (ev / "img.png").write_bytes(b"\x89PNG")
    lk = AU.leak([str(tmp_path / "syn")], str(tmp_path / "eval"), str(a4_dir), tmp_path / "au" / "leak.json")
    s = lk["synthetic"][str(tmp_path / "syn")]
    assert lk["skipped_files"] == ["fast/img.png"] and lk["eval_items"] == 3
    assert lk["train_10M"]["items_hit"] >= 1 and s["items_hit_synthetic_only"] >= 1
    assert any(i["text"] == items[0]["good"] for i in s["synthetic_only_items"])


def test_myers_distance_is_semi_global_edit_distance():
    def brute(p, t):
        m = len(p)
        prev = list(range(m + 1))
        best = m
        for c in t:
            cur = [0]
            for i in range(1, m + 1):
                cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (p[i - 1] != c)))
            best = min(best, cur[m])
            prev = cur
        return min(best, m)

    rng = np.random.default_rng(0)
    for _ in range(300):
        p = rng.integers(0, 4, rng.integers(1, 70)).tolist()
        t = rng.integers(0, 4, rng.integers(0, 90)).tolist()
        assert AU.myers_distance(p, t) == brute(p, t)
    p = list(range(100))
    t = [7, 7] + p[:40] + [999] + p[41:] + [5]
    assert AU.myers_distance(p, t) == 1


def test_train_index_finds_near_copies():
    rng = np.random.default_rng(1)
    train = rng.integers(0, 500, 20000)
    idx = AU.TrainIndex(train)
    q = train[5000:5100].copy()
    q[[10, 50, 90]] = 499 - q[[10, 50, 90]]  # 3 substitutions
    d, nc = idx.nn(q)
    assert d == 3 and nc >= 1
    assert idx.nn(rng.integers(1000, 2000, 100))[0] == 100  # nothing shared: the length


def test_make_a4_jobs_plan_and_files(tmp_path):
    r = MJ.main(["--out_dir", str(tmp_path), "--commit", "a" * 40])
    p = r["trainer"]["plan"]
    assert (p["R_a"], p["R_b"], p["syn_words_a"]) == (3, 2, 3 * MJ.TRAIN_WORDS)
    assert max(p["words_a"], p["words_b"]) <= MJ.CAP
    jobs = {j["name"]: j["args"] for j in r["trainer"]["jobs"]}
    assert jobs["a4_G"][jobs["a4_G"].index("--epochs") + 1] == "4"
    assert jobs["a4_GB"][jobs["a4_GB"].index("--split") + 1] == "train_10M_hB"
    for n in ("a4_n0a", "a4_a", "a4_n1a", "a4_n2a", "a4_n0b", "a4_b", "a4_n1b", "a4_n2b"):
        a = jobs[n]
        assert "--epochs" not in a and "--init_from" in a and "--sources" in a and "--save" in a
        assert ("--exposure_extra" in a) == (n in ("a4_b", "a4_n1b", "a4_n2b"))
        TR.build_parser().parse_args(a)  # every job parses
    for f in ("a4_screen.json", "a4_screen_b2.json"):
        j = json.loads((tmp_path / "gcp" / f).read_text())
        assert j["commit"] == "a" * 40 and j["keep_checkpoints"] and all(x["cmd"] for x in j["jobs"])
    with pytest.raises(SystemExit):
        MJ.main(["--out_dir", str(tmp_path), "--g_uri", "${BUCKET}/y", "--g_epochs", "4"])
    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=MJ.HERE, capture_output=True).returncode == 0:
        r = MJ.main(["--out_dir", str(tmp_path), "--commit", "a" * 40, "--provisional",
                     "--base_job", "HEAD:babylm/configs/jobs_a4_screen.json:a4_G"])  # a base read from git, E* = 4
        assert r["trainer"]["plan"]["E_star"] == 4 and "PROVISIONAL" in r["trainer"]["description"]
        head = subprocess.run(["git", "show", "HEAD:babylm/configs/jobs_a4_screen.json"], cwd=MJ.HERE,
                              capture_output=True, text=True).stdout
        MJ.check_pinned("HEAD", head)  # the pinned commit carries this trainer file
        with pytest.raises(SystemExit):
            MJ.check_pinned("HEAD", head + " ")  # it carries another: the executor would run the wrong jobs
    with pytest.raises(SystemExit):
        MJ.check_pinned("a" * 40, "{}")  # no such commit



def test_make_a4_jobs_seed_replicate(tmp_path):
    """Seed k > 0: the _sk twins of every run and data directory, as make_stack_jobs --own_g reads them."""
    for sub in ("s0", "s1", "s2"):
        (tmp_path / sub).mkdir()
    c = ["--commit", "a" * 40]
    j0 = {j["name"]: j["args"] for j in MJ.main(["--out_dir", str(tmp_path / "s0"), *c])["trainer"]["jobs"]}
    r1 = MJ.main(["--out_dir", str(tmp_path / "s1"), *c, "--seed", "1", "--arm", "ab"])
    j1 = {j["name"]: j["args"] for j in r1["trainer"]["jobs"]}
    assert set(j1) == {"a4_G_s1", "a4_GA_s1", "a4_GB_s1", "a4_n0a_s1", "a4_a_s1", "a4_n0b_s1", "a4_b_s1"}

    def twin(v):  # --own_g: babylm/runs/X/... -> babylm/runs/X_s1/...
        return re.sub(r"(babylm/runs/[^/]+)/", r"\1_s1/", v)
    for n in ("a4_G", "a4_GA", "a4_GB", "a4_n0a", "a4_a", "a4_n0b", "a4_b"):
        want = MJ.set_flag(MJ.set_flag([twin(x) for x in j0[n]], "--seed", "1"), "--tag", n + "_s1")
        assert j1[n + "_s1"] == want, n
        TR.build_parser().parse_args(j1[n + "_s1"])
    ex = r1["batch"]
    blob = json.dumps(ex)
    assert ex["name"] == "a4_s1" and ex["commit"] == "a" * 40 and "jobs_a4_s1.json" in blob
    assert "jobs_a4_screen" not in blob and all(m.endswith("_s1") for m in re.findall(r"babylm/runs/(a4_\w+)", blob))
    gens = next(x["cmd"] for x in ex["jobs"] if x["name"] == "a4_generate")
    assert gens.count(" --seed 1") == 4 and "--selfseed" not in gens  # a4a, probe, a4b_A, a4b_B; no nulls
    r2 = MJ.main(["--out_dir", str(tmp_path / "s2"), *c, "--seed", "2", "--arm", "a", "--data_only"])
    assert [j["name"] for j in r2["trainer"]["jobs"]] == ["a4_G_s2"]
    assert [x["name"] for x in r2["batch"]["jobs"]] == ["a4_G", "a4_generate", "a4_build", "a4_audit"]
    for bad in (["--seed", "1"], ["--arm", "a"], ["--data_only"]):
        with pytest.raises(SystemExit):
            MJ.main(["--out_dir", str(tmp_path / "s2"), *c, *bad])

BUILD = "2.7.0+cu126"  # the reference build (critic ref-0037): p2venv torch 2.7.0


def _cfg(run, torch=BUILD, gpu="NVIDIA A100-SXM4-40GB", **args):
    """A run's build record (ENV_RULE): config.json torch and its args (the recipe), result.json gpu."""
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"torch": torch, "total_steps": 2435,
                                                 "args": {"seed": 0, "epochs": 10.0, "out": str(run), **args}}))
    rp = run / "result.json"
    rp.write_text(json.dumps({**(json.loads(rp.read_text()) if rp.exists() else {}), "gpu": gpu}))


def _arm(root, name, dev, blimp, flops, syn=None):
    d = root / name
    (d / "score").mkdir(parents=True)
    src = ["real:train_10M:3"] + ([f"synthetic:{syn[0]}:1"] if syn else [])
    sha = ["r" * 64] + ([syn[1]] if syn else [])
    (d / "result.json").write_text(json.dumps({"exposure": {"sources": src, "source_sha256": sha}}))
    _cfg(d, init_from="a4_G/chck_40M.pt")
    (d / "score" / "score.json").write_text(json.dumps({"dev_nats_per_word": dev, "fast": {VD.BLIMP: blimp},
                                                        VD.GUARDED["key"]: dev}))
    _pw(d, dev)
    return {"run": name, "flops_total": flops, "within_R0": True}


def _gx(score_json, v):
    """Set (None: drop) a score's never-seen-character number, the key the executor merges at copy-back."""
    s = json.loads(score_json.read_text())
    s.pop(VD.GUARDED["key"], None)
    score_json.write_text(json.dumps(s if v is None else {**s, VD.GUARDED["key"]: v}))


def _steps(run, ckpt, rows, final):
    """Set a run dir's scored checkpoint (score.json's "checkpoint"), curves.csv rows [(label, step)] and result.json
    steps (its score.json sits in score/ below it, or in it)."""
    sj = run / "score" / "score.json" if (run / "score" / "score.json").exists() else run / "score.json"
    rp = run / "result.json"
    rp.write_text(json.dumps({**(json.loads(rp.read_text()) if rp.exists() else {}), "steps": final}))
    (run / "curves.csv").write_text("step,words_seen,label\n" + "".join(f"{st},0,{lb}\n" for lb, st in rows))
    sj.write_text(json.dumps({**json.loads(sj.read_text()), "checkpoint": ckpt}))


def _pw(d, rest):
    """A stand-in full-DEV per-word dump (read by the fake cross-check below as the run's harness rest)."""
    (d / "pw").mkdir(parents=True, exist_ok=True)
    np.save(d / "pw" / UN.PW, np.array([rest]))


@pytest.fixture
def fake_leg(monkeypatch):
    """The harness cross-check without the real DEV: each dump's single value is the run's rest."""
    monkeypatch.setattr(VD, "unseen_rests", lambda dumps, data_dir=None: {
        **{k: None if p is None else float(np.load(p)[0]) for k, p in dumps.items()}, "_info": {"instrument": "test"}})


def _calib(path, scores, shift=None, edge=(), fmt="v2"):
    """A calib_control --json stand-in: raw = each run's twin numbers (score_file's v2 / v2x), calibrated_t = raw -
    shift (default 0.03, so calibrated gains equal raw ones), lam-only calibrated = raw - 0.01; t_edge on the names in
    edge."""
    runs = {n: {leg: {"raw": s[k], "calibrated": s[k] - 0.01, "calibrated_t": s[k] - (shift or {}).get(n, 0.03),
                      "t_edge": n in edge} for leg, k in VD.LEGS} for n, s in scores.items()}
    path.write_text(json.dumps({"format": fmt, "grid": [0.0], "runs": runs}))


def _evidence(path, runs, instrument="solo", gpu="NVIDIA A100-SXM4-40GB", **row):
    """Executor 3's instrument table (critic ref-0035) for runs; row overrides columns of every row."""
    cells = {"vm": "bl-test-1010-000000", "machine_type": "a2-highgpu-1g", "gpu": gpu, "instrument": instrument,
             "evidence": "startup.log: one job at a time", **row}
    path.write_text("# instrument by run (test)\n" + "\t".join(VD.EVIDENCE) + "\n"
                    + "".join("\t".join([r] + [cells[c] for c in VD.EVIDENCE[1:]]) + "\n" for r in runs))
    return path


def test_verdict_promotion_gates(tmp_path, fake_leg):
    """Critic ref-0011 / ref-0029 (calibrated_t over every reference) and ref-0030 / ref-0031 (the same-seed base at
    E*_raw and E*_cal): a screen pass below the kill line on either is not promoted; the screen verdict stays."""
    runs, data = tmp_path / "runs", tmp_path / "runs" / "a4_data"
    syn = {"a4_a": ("A", "a" * 64), "a4_n1a": ("N1", "b" * 64), "a4_n2a": ("N2", "c" * 64)}
    rows = [_arm(runs, n, d, 63.1, 10.0, syn.get(n)) for n, d in
            (("a4_n0a", 5.30), ("a4_a", 5.20), ("a4_n1a", 5.28), ("a4_n2a", 5.25))]
    for n in ("a4_n0a", "a4_a", "a4_n1a", "a4_n2a"):  # v2 twins 0.02 below v1; the leg (score.json) = v1
        d = json.loads((runs / n / "score" / "score.json").read_text())["dev_nats_per_word"]
        (runs / n / "score" / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": d - 0.02}))
    (data / "audit").mkdir(parents=True)
    (data / "ledger.json").write_text(json.dumps({"arms": rows}))
    _leak(data, syn.values())
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"dev_nats_per_word": 5.31, "fast": {VD.BLIMP: 63.0}, VD.GUARDED["key"]: 5.31,
                                VD.METRICS["v2"]: 5.29}))
    _pw(tmp_path, 5.31)
    _cfg(tmp_path)
    es = tmp_path / "e5" / "score.json"  # the same-seed base at E*_raw (= E*_cal): 0.06 / 0.07 over the arm
    es.parent.mkdir()
    es.write_text(json.dumps({VD.METRICS["v2"]: 5.24, VD.GUARDED["key"]: 5.27, "run": "e5_s0"}))
    _cfg(es.parent, epochs=5.0)
    sc = {"base": VD.score_file(base), "estar_cal": {"v2": 5.24, "v2x": 5.27},
          **{n: VD.score(runs / n) for n in ("a4_n0a", "a4_a", "a4_n1a", "a4_n2a")}}
    cj = tmp_path / "calib.json"
    _calib(cj, sc)
    compared = ["a4_a", "a4_n0a", "a4_n1a", "a4_n2a", tmp_path.name, "e5_s0"]  # base: its dir's name (no "run")
    listed = compared + ["e5"]  # and the E*_raw base by its dir's name, for its rewrites below without "run"
    ev = _evidence(tmp_path / "instruments.tsv", listed)
    args = ["--base_score", str(base), "--runs_root", str(runs), "--ref_instrument", "base=solo"]
    gx = ["--calib", str(cj), "--estar_raw", str(es), "--ref_instrument", "estar_raw=solo", "--vm_evidence", str(ev)]
    r = VD.main(args)
    assert r["verdict"] == "PASS" and r["promote"] is None
    assert r["promotion"] == ("pending: no calibrated control (--calib: calib_control @ ref-0033 over the temps "
                              "dumps); no same-seed base at E*_raw (--estar_raw); the instrument is not corroborated "
                              f"from VM evidence for a4_a, a4_n0a, a4_n1a, a4_n2a, {tmp_path.name} (critic ref-0035 / "
                              "ref-0036: --vm_evidence or a complete runner sidecar)")
    # critic ref-0032: a bar read on calibrated_t is the raw bar x the leg's calibrated_t/raw SD ratio
    assert [VD.cal_t_bar(b, "solo", leg) for b in (0.010, 0.0074) for leg in ("full", "guarded")] == \
        [0.0128, 0.0127, 0.0095, 0.0094]
    assert [VD.cal_t_bar(b, "packed", leg) for b in (0.0144, 0.0102) for leg in ("full", "guarded")] == \
        [0.0184, 0.0184, 0.0131, 0.0131]
    r = VD.main(args + gx + ["--estar", "E*_raw = E*_cal = 5 (test)", "--out", str(data / "v.json")])
    g = r["families"]["a"]["gates"]
    assert r["verdict"] == "PASS" and r["promotion"] == "promote" and r["promote"] == "a4_a"
    assert g["calibrated"]["N2"]["full"] == pytest.approx(0.05)
    assert g["calibrated"]["N2"]["guarded_lam"] == pytest.approx(0.05)
    assert g["estar_raw"]["full"] == pytest.approx(0.06) and g["estar_raw"]["guarded"] == pytest.approx(0.07)
    assert g["estar_cal"]["full"] == pytest.approx(0.06) and g["estar_cal"]["guarded"] == pytest.approx(0.07)
    assert g["estar_cal"]["build_via"] == "--estar_raw" and r["env_check"]["home"] == BUILD
    assert g["kill_lines"] == {"raw": {"full": 0.0074, "guarded": 0.0074},
                               "calibrated_t": {"full": 0.0095, "guarded": 0.0094}}
    md = (data / "v.md").read_text()
    assert "E*_raw = E*_cal = 5" in md and "calibrated_t over N2" in md and "promotion: promote" in md
    assert "scope: a screen only (critic ref-0039)" in md
    assert "instrument evidence (critic ref-0035): 6 of 6 compared runs corroborated solo from VM evidence" in md
    # critic ref-0035: promotion needs every compared run's instrument corroborated from VM evidence; the screen stays
    _evidence(ev, [n for n in listed if n != "a4_n1a"])
    r = VD.main(args + gx)
    assert r["verdict"] == "PASS" and r["promote"] is None and r["promotion"] == (
        "pending: the instrument is not corroborated from VM evidence for a4_n1a (critic ref-0035 / ref-0036: "
        "--vm_evidence or a complete runner sidecar)")
    _evidence(ev, listed, machine_type="")  # a row without its machine type corroborates nothing
    assert "a4_a (incomplete row)" in VD.main(args + gx)["promotion"]
    _evidence(ev, listed, gpu="a100-SXM4-40GB")  # nvidia-smi's spelling: the same GPU model
    assert VD.main(args + gx)["promotion"] == "promote"
    for bad, msg in (({"instrument": "packed"}, "a4_a: its ticket says solo, the VM evidence says packed"),
                     ({"gpu": "NVIDIA A100-SXM4-80GB"}, "a4_a: result.json records NVIDIA A100-SXM4-40GB, the VM "
                                                        "evidence names NVIDIA A100-SXM4-80GB")):
        _evidence(ev, listed, **bad)  # the ticket (or the run's GPU record) and the VM disagree: VOID
        with pytest.raises(SystemExit, match="VOID read: " + re.escape(msg)):
            VD.main(args + gx)
    _evidence(ev, listed + ["a4_a"])
    with pytest.raises(SystemExit, match="two rows for a4_a"):
        VD.main(args + gx)
    _evidence(ev, listed, instrument="duo")
    with pytest.raises(SystemExit, match=r"instrument 'duo', expected solo\|packed"):
        VD.main(args + gx)
    ev.write_text("run\tvm\tgpu\tinstrument\tevidence\n")
    with pytest.raises(SystemExit, match="no machine_type column"):
        VD.main(args + gx)
    _evidence(ev, listed)
    side = runs / "a4_a" / "instrument.json"  # the runner's sidecar (executor 3 @ ref-0040), checked when present
    rec = {"run": "a4_a", "job": "a4_a", "instrument": "solo", "pack_partners": [], "vm": "bl-test-1010-000000",
           "machine_type": "a2-highgpu-1g", "gpu": ["NVIDIA A100-SXM4-40GB"], "gpu_count": 1, "commit": "ref-0048",
           "source": VD.SIDECAR_SOURCE}  # ref-0040's record
    side.write_text(json.dumps(rec))
    r = VD.main(args + gx)
    assert r["promotion"] == "promote" and r["instrument_check"]["sidecars"] == ["a4_a"]
    assert r["instrument_check"]["by_sidecar"] == []
    _evidence(ev, [n for n in listed if n != "a4_a"])  # ref-0040's record does not stand in for the row (ref-0036)
    assert VD.main(args + gx)["promotion"].endswith(
        "for a4_a (no row; its instrument.json lacks startup.sh's commit, PACK recorded as unset or 1, one training "
        "process at launch, a GPU-process maximum of 1 over the job (no training window)) (critic ref-0035 / ref-0036: "
        "--vm_evidence or a complete runner sidecar)")
    full = {**rec, "startup_commit": "ref-0047", "pack_env": None, "train_procs_at_launch": 1, "gpu_procs_max": 1,
            "gpu_proc_samples": 96}  # executor 3 @ ref-0047
    side.write_text(json.dumps(full))  # critic ref-0036: a complete sidecar stands in for a solo run's row
    r = VD.main(args + gx + ["--out", str(data / "v.json")])
    assert r["promotion"] == "promote" and r["instrument_check"]["by_sidecar"] == ["a4_a"]
    assert ("6 of 6 compared runs corroborated solo from VM evidence (1 by a complete instrument.json sidecar, critic "
            "ref-0036)") in (data / "v.md").read_text()
    side.write_text(json.dumps({**full, "pack_env": "1"}))  # PACK explicitly 1
    assert VD.main(args + gx)["promotion"] == "promote"
    for gap, lack in (("source", "startup.sh as its writer"), ("startup_commit", "startup.sh's commit"),
                      ("pack_env", "PACK recorded as unset or 1"), ("pack_partners", "an empty pack-partner list"),
                      ("train_procs_at_launch", "one training process at launch"), ("vm", "its VM"),
                      ("machine_type", "its machine type")):  # (c): no pack lines alone is not solo
        side.write_text(json.dumps({k: v for k, v in full.items() if k != gap}))
        assert f"a4_a (no row; its instrument.json lacks {lack})" in VD.main(args + gx)["promotion"]
    for gap, lack in (({"gpu": []}, "its nvidia-smi GPUs"), ({"train_procs_at_launch": 0}, "one training process"),
                      ({"pack_env": ""}, "PACK recorded as unset or 1"),
                      ({"startup_commit": "ref-0047+dirty"}, "a committed startup.sh (it ran +dirty)"),
                      ({"gpu_procs_max": 2}, "a GPU-process maximum of 1 over the job (no training window)"),
                      ({"train_window_unix": [100, 110], "gpu_proc_log": [[90, 1], [130, 2]], "gpu_procs_max": 2},
                       "a GPU-process maximum of 1 over the job (no training window)"),  # no sample inside it
                      ({"train_window_unix": [100, 200], "gpu_proc_log": [[120, 1], [150, 0], [180, 1]]},
                       "exactly 1 GPU process in every sample of the training window")):  # a gap, not a second job
        side.write_text(json.dumps({**full, **gap}))
        assert f"lacks {lack}" in VD.main(args + gx)["promotion"]
    log = [[90, 0], [120, 1], [150, 1], [180, 1], [230, 2]]  # critic ref-0041: scoring after the final checkpoint
    side.write_text(json.dumps({**full, "gpu_procs_max": 2, "train_window_unix": [100, 200], "gpu_proc_log": log}))
    assert VD.main(args + gx)["instrument_check"]["by_sidecar"] == ["a4_a"]  # write is excluded by time
    side.write_text(json.dumps(full))
    ev.write_text(ev.read_text() + "a4_a\tbl-test-1010-000000\ta2-highgpu-1g\tNVIDIA A100-SXM4-40GB\tsolo\t\n")
    assert VD.main(args + gx)["instrument_check"]["by_sidecar"] == ["a4_a"]  # an incomplete row, a complete sidecar
    _evidence(ev, listed)
    for bad, msg in (({"instrument": "packed", "pack_partners": ["a4_b"]}, "a4_a: its ticket says solo, the runner's "
                      "instrument.json says packed"),
                     ({"pack_env": "2"}, "a4_a: its ticket says solo, the runner's instrument.json records PACK=2"),
                     ({"pack_partners": ["a4_b"]}, "a4_a: its ticket says solo, the runner's instrument.json "
                      "records pack partners a4_b"),
                     ({"train_procs_at_launch": 2}, "a4_a: its ticket says solo, the runner's instrument.json "
                      "records 2 training processes at launch"),
                     ({"train_window_unix": [100, 200], "gpu_proc_log": [[120, 1], [150, 2], [180, 1]]}, "a4_a: its "
                      "ticket says solo, the runner's instrument.json records 2 GPU processes in the training window"),
                     ({"run": "a4_n0a"}, "a4_a: the scored copy's instrument.json was written for a4_n0a"),
                     ({"gpu": ["NVIDIA A100-SXM4-80GB"]}, "a4_a: result.json records NVIDIA A100-SXM4-40GB, the "
                      "runner's instrument.json names"),
                     ({"vm": "bl-other-1010-000000"}, "a4_a: the VM evidence row names bl-test-1010-000000, the "
                      "scored copy's instrument.json bl-other-1010-000000"),
                     ({"machine_type": "a2-ultragpu-1g"}, "a4_a: the VM evidence row names machine type "
                      "a2-highgpu-1g, the scored copy's instrument.json a2-ultragpu-1g")):
        side.write_text(json.dumps({**full, **bad}))  # sidecar, row, ticket and GPU record disagree: VOID
        with pytest.raises(SystemExit, match="VOID read: " + re.escape(msg)):
            VD.main(args + gx)
    side.unlink()
    _calib(cj, sc, {"a4_n2a": 0.0755})  # calibrated_t gain over N2 0.0045 on both legs: calibration-equivalent
    r = VD.main(args + gx)
    assert r["verdict"] == "PASS" and r["promote"] is None
    assert r["promotion"] == "not promoted: calibration-equivalent"
    assert len(r["families"]["a"]["gates"]["blocked"]) == 2  # both legs
    _calib(cj, sc, {"a4_n2a": 0.0715})  # 0.0085: over the raw kill line (0.0074), under calibrated_t's on both legs
    assert len(VD.main(args + gx)["families"]["a"]["gates"]["blocked"]) == 2
    _calib(cj, sc, {"a4_n2a": 0.0706})  # 0.0094: under the full leg's line (0.0095), on the guarded leg's (0.0094)
    r = VD.main(args + gx)
    assert r["promotion"] == "not promoted: calibration-equivalent"
    assert r["families"]["a"]["gates"]["blocked"] == ["calibration-equivalent: calibrated_t gain over N2 (full) "
                                                      "+0.0094 < 0.0095"]
    _calib(cj, sc, {"a4_n2a": 0.0675})  # 0.0125 clears the solo line; a packed read (packed base) needs 0.0131
    assert VD.main(args + gx)["promotion"] == "promote"
    px = ["--packed", "--ref_instrument", "base=packed", "estar_raw=packed"]
    with pytest.raises(SystemExit, match="its ticket says packed, the VM evidence says solo"):
        VD.main(args + gx + px)
    pk = VD.main(args + gx + px + ["--vm_evidence", str(_evidence(tmp_path / "packed.tsv", listed, "packed"))])
    assert pk["promotion"] == "not promoted: calibration-equivalent" and pk["env_check"]["instrument"] == "packed"
    assert pk["families"]["a"]["gates"]["kill_lines"]["calibrated_t"] == {"full": 0.0131, "guarded": 0.0131}
    side.write_text(json.dumps({**full, "instrument": "packed", "pack_env": "2", "pack_partners": ["a4_b"],
                                "train_procs_at_launch": 2}))  # ref-0036 rules on solo runs: a packed one needs its row
    pk = VD.main(args + gx + px + ["--vm_evidence", str(_evidence(tmp_path / "packed.tsv", listed[1:], "packed"))])
    assert pk["instrument_check"]["missing"] == ["a4_a (no row; a sidecar stands in for a solo run only)"]
    side.unlink()
    _calib(cj, sc, {"a4_n2a": 0.0755})
    es.write_text(json.dumps({VD.METRICS["v2"]: 5.185, VD.GUARDED["key"]: 5.27}))  # raw gain over E*_raw 0.005
    r = VD.main(args + gx)
    assert r["promotion"] == "not promoted: calibration-equivalent and epoch-equivalent"
    _calib(cj, sc)
    r = VD.main(args + gx)
    assert r["promotion"] == "not promoted: epoch-equivalent" and r["verdict"] == "PASS"
    es.write_text(json.dumps({VD.METRICS["v2"]: 5.18, VD.GUARDED["key"]: 5.26}))  # one run: E*_raw = E*_cal
    _calib(cj, {**sc, "estar_cal": {"v2": 5.18, "v2x": 5.26}})  # calibrated_t gain over E*_cal -0.0 (full)
    r = VD.main(args + gx)
    assert r["promotion"] == "not promoted: epoch-equivalent"
    assert any("E*_cal (full)" in b for b in r["families"]["a"]["gates"]["blocked"])
    es.write_text(json.dumps({VD.METRICS["v2"]: 5.24, VD.GUARDED["key"]: 5.27}))
    r = VD.main(args + gx)  # the calib json's estar_cal is not the --estar_raw run: no build record (ENV_RULE)
    assert r["promotion"] == "pending: the same-seed base at E*_cal: no build record (the calib json's estar_cal is " \
                             "not the --estar_raw run; ENV_RULE)"
    assert r["families"]["a"]["gates"]["estar_cal"]["build_via"] is None
    _calib(cj, sc, edge=("a4_a",))  # a T pick at the grid edge: extend the grid first (ref-0011 amendment 1)
    r = VD.main(args + gx)
    assert r["promotion"].startswith("pending") and "grid edge" in r["promotion"] and r["promote"] is None
    _calib(cj, {k: v for k, v in sc.items() if k != "estar_cal"})
    assert "no same-seed base at E*_cal" in VD.main(args + gx)["promotion"]
    _calib(cj, {k: v for k, v in sc.items() if k != "a4_n1a"})  # a reference without a temps dump
    assert VD.main(args + gx)["promotion"] == "pending: calibrated_t gain over N1: no number (full, guarded)"
    _calib(cj, {**sc, "base": {"v2": 5.30, "v2x": 5.31}})  # base's raw is not base's twin: another dump
    with pytest.raises(SystemExit):
        VD.main(args + gx)
    _calib(cj, sc, fmt="harness")  # the binding control reads v2 temps dumps only
    with pytest.raises(SystemExit):
        VD.main(args + gx)
    _calib(cj, sc)
    (runs / "a4_n2a" / "score" / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": 5.189}))
    _calib(cj, {**sc, "a4_n2a": VD.score(runs / "a4_n2a")})  # an inconclusive screen: numbers, no promotion
    r = VD.main(args + gx)
    assert r["verdict"] == "INCONCLUSIVE" and r["promotion"] is None and r["families"]["a"]["gates"]["calibrated"]
    assert all(x in r["next"] for x in ("pools of 3 runs or more in the arms' environment and instrument",
                                        "v2e, v2f and one more", "ref-0030", "claim floor of its own metric "
                                        "and pool", "the torch 2.9 pool's SDs are not borrowed", "ref-0043",
                                        "the claim prints both comparisons (experimenter 19:55Z)", "E6 s0-s4",
                                        "clears the E10 pool but not the E* pool is epoch-equivalent and not promoted",
                                        "corroborated from VM evidence (critic ref-0035)",
                                        "better (higher-BLiMP) of the two pools' means", "read as a pool on the same "
                                        "rule (critic ref-0044)"))
    assert r["scope"] == VD.SCOPE and "0.0074 / 0.0076" not in r["next"]


def test_step_rule_voids_mid_anneal_scores(tmp_path, fake_leg):
    """Critic (experimenter 14:48Z, the K3 bug): a score counts only if its scored checkpoint's step is the run's final step;
    the readout refuses while any score it reads (runs, base, E*_raw base) is VOID."""
    runs, data = tmp_path / "runs", tmp_path / "runs" / "a4_data"
    syn = {"a4_a": ("A", "a" * 64), "a4_n1a": ("N1", "b" * 64), "a4_n2a": ("N2", "c" * 64)}
    rows = [_arm(runs, n, d, 63.1, 10.0, syn.get(n)) for n, d in
            (("a4_n0a", 5.30), ("a4_a", 5.20), ("a4_n1a", 5.28), ("a4_n2a", 5.25))]
    (data / "audit").mkdir(parents=True)
    (data / "ledger.json").write_text(json.dumps({"arms": rows}))
    _leak(data, syn.values())
    base = tmp_path / "b" / "score.json"
    base.parent.mkdir()
    base.write_text(json.dumps({"dev_nats_per_word": 5.31, "fast": {VD.BLIMP: 63.0}, VD.GUARDED["key"]: 5.31}))
    _cfg(base.parent)
    args = ["--base_score", str(base), "--runs_root", str(runs), "--ref_instrument", "base=solo"]
    a = runs / "a4_a"
    _steps(a, "chck_100M.pt", [("90M", 1228), ("100M", 1460)], 1460)  # the end milestone at the final step
    chk = VD.score(a)["steps"]
    assert chk["ok"] and chk["ckpt_step"] == 1460 and chk["last_row"] == ["100M", 1460]
    r = VD.main(args)
    assert r["step_check"]["a4_a"]["ok"] and r["step_check"]["a4_n0a"] is None  # names no checkpoint: unchecked
    _steps(a, "chck_20M.pt", [("10M", 245), ("20M", 489), ("final", 490)], 490)  # scored one step early
    assert not VD.score(a)["steps"]["ok"]
    with pytest.raises(SystemExit, match="VOID score.*a4_a: scored chck_20M.pt at step 489, the run ends at step 490"):
        VD.main(args)
    _steps(a, "chck_final.pt", [("20M", 489), ("final", 490)], 490)  # ref-0045: the end weights are scored
    assert VD.score(a)["steps"]["ok"] and VD.main(args)["step_check"]["a4_a"]["ckpt_step"] == 490
    _steps(base.parent, "chck_100M.pt", [("100M", 2434), ("final", 2435)], 2435)  # the base is read the same way
    with pytest.raises(SystemExit, match="base: scored chck_100M.pt at step 2434, the run ends at step 2435"):
        VD.main(args)
    (base.parent / "curves.csv").unlink()
    with pytest.raises(SystemExit, match="base: no curves.csv in"):
        VD.main(args)
    (base.parent / "result.json").unlink()
    with pytest.raises(SystemExit, match="base: no result.json beside the score"):
        VD.main(args)
    _steps(base.parent, "chck_100M.pt", [("100M", 2435)], 2435)
    _cfg(base.parent)  # result.json's GPU model again (ENV_RULE)
    es = tmp_path / "e6" / "score" / "score.json"  # E*_raw: score/ below the run dir, as on the drive
    es.parent.mkdir(parents=True)
    es.write_text(json.dumps({VD.METRICS["v2"]: 5.24, VD.GUARDED["key"]: 5.27, "checkpoint": "chck_60M.pt"}))
    (tmp_path / "e6" / "result.json").write_text(json.dumps({"steps": 1461}))
    _cfg(tmp_path / "e6", epochs=6.0)
    (tmp_path / "e6" / "curves.csv").write_text("step,label\n1461,60M\n")
    ex = ["--estar_raw", str(es), "--ref_instrument", "estar_raw=solo"]
    assert VD.main(args + ex)["step_check"]["estar_raw"]["ok"]
    (tmp_path / "e6" / "curves.csv").write_text("step,label\n1460,60M\n1461,final\n")
    with pytest.raises(SystemExit, match="estar_raw: scored chck_60M.pt at step 1460"):
        VD.main(args + ex)


def test_env_rule_voids_cross_build_reads(tmp_path, fake_leg):
    """Critic ref-0037, 15:34Z, ref-0038 and ref-0039 (experimenter 15:07Z, 19:19Z): the torch/CUDA build, the GPU model and
    the packing instrument are part of the recipe. The batch (runs and generators) shares one environment; a reference
    in another one (base_r1 s0 is torch 2.9.1) or under another (declared) instrument is refused, with no pair
    admission (the 0.002 line is retracted): the E10 base is v2e, solo, by flag."""
    runs, data = tmp_path / "runs", tmp_path / "runs" / "a4_data"
    syn = {"a4_a": ("A", "a" * 64), "a4_n1a": ("N1", "b" * 64), "a4_n2a": ("N2", "c" * 64)}
    rows = [_arm(runs, n, d, 63.1, 10.0, syn.get(n)) for n, d in
            (("a4_n0a", 5.30), ("a4_a", 5.20), ("a4_n1a", 5.28), ("a4_n2a", 5.25))]
    for r in rows:  # v2 twins 0.02 below v1
        (runs / r["run"] / "score" / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": VD.score(
            runs / r["run"])["v1"] - 0.02}))
    (data / "audit").mkdir(parents=True)
    (data / "ledger.json").write_text(json.dumps({"arms": rows}))
    _leak(data, syn.values())

    def ref(name, v2, v2x, torch=BUILD, **args):
        p = tmp_path / name / "score.json"
        _cfg(p.parent, torch, **args)
        p.write_text(json.dumps({"dev_nats_per_word": v2 + 0.02, VD.METRICS["v2"]: v2, VD.GUARDED["key"]: v2x,
                                 "fast": {VD.BLIMP: 63.0}, "run": name}))
        return p

    solo = ["--ref_instrument", "base=solo"]
    base = ref("base_r1", 5.29, 5.31, "2.9.1+cu129")
    args = ["--base_score", str(base), "--runs_root", str(runs), *solo]
    with pytest.raises(SystemExit, match=r"VOID cross-environment read: base ran on torch 2.9.1\+cu129 / NVIDIA "
                                         r"A100-SXM4-40GB, the batch on torch 2.7.0\+cu126 / NVIDIA A100-SXM4-40GB\. "
                                         r".*single-pair admission is retracted \(critic ref-0038\)"):
        VD.main(args)
    with pytest.raises(SystemExit):  # no pair admits it any more (argparse: unrecognized --env_pair)
        VD.main(args + ["--env_pair", str(ref("v2e_pair", 5.2905, 5.3092))])
    v2e = ref("v2e", 5.30, 5.33, gpu="NVIDIA A100-SXM4-80GB")  # GPU model binding (critic 15:34Z): another model
    rv = ["--base_score", str(v2e), "--runs_root", str(runs)]
    with pytest.raises(SystemExit, match=r"base ran on torch 2.7.0\+cu126 / NVIDIA A100-SXM4-80GB, the batch on "
                                         r"torch 2.7.0\+cu126 / NVIDIA A100-SXM4-40GB\."):
        VD.main(rv + solo)
    v2e = ref("v2e", 5.30, 5.33)  # the E10 base in the batch's environment (any A100-SXM4-40GB)
    r = VD.main(rv + solo + ["--out", str(data / "v.json")])
    assert r["families"]["a"]["refs"]["base"]["dev"] == 5.30 and r["env_check"]["home"] == BUILD
    assert r["env_check"]["gpu"] == r["env_check"]["gpus"][0] == "NVIDIA A100-SXM4-40GB"
    assert r["env_check"]["instrument"] == "solo" and r["env_check"]["ref_instruments"] == {"base": "solo"}
    assert "environment: torch 2.7.0+cu126 / NVIDIA A100-SXM4-40GB / solo" in (data / "v.md").read_text()
    # the packing instrument (critic 48436034 / ref-0039): run files do not record it, so each reference's is declared
    with pytest.raises(SystemExit, match=r"VOID read: no packing instrument declared for base \(run files do not "
                                         r"record packing"):
        VD.main(rv)
    with pytest.raises(SystemExit, match=r"VOID cross-instrument read: base ran packed, the batch solo\."):
        VD.main(rv + ["--ref_instrument", "base=packed"])
    with pytest.raises(SystemExit, match=r"VOID cross-instrument read: base ran solo, the batch packed\."):
        VD.main(rv + solo + ["--packed"])
    r = VD.main(rv + ["--packed", "--ref_instrument", "base=packed"])  # packed vs packed reads
    assert r["env_check"]["instrument"] == "packed" and r["thresholds"]["condition"] == "packed"
    assert VD.main(rv + ["--ref_instrument", "base=packed", "--ref_instrument", "base=solo"])["env_check"][
        "ref_instruments"] == {"base": "solo"}  # the last declaration counts
    for bad in ("base", "base=duo", "oracle=solo"):
        with pytest.raises(SystemExit, match=r"expected NAME=solo\|packed, NAME one of base, estar_raw"):
            VD.main(rv + ["--ref_instrument", bad])
    es = ref("e6", 5.24, 5.27, epochs=6.0)
    with pytest.raises(SystemExit, match=r"no packing instrument declared for estar_raw"):
        VD.main(rv + solo + ["--estar_raw", str(es)])
    assert VD.main(rv + solo + ["--estar_raw", str(es), "--ref_instrument", "estar_raw=solo"])["env_check"][
        "ref_instruments"] == {"base": "solo", "estar_raw": "solo"}
    es = ref("e6", 5.24, 5.27, "2.9.1+cu129", epochs=6.0)
    with pytest.raises(SystemExit, match=r"estar_raw ran on torch 2.9.1\+cu129"):
        VD.main(rv + solo + ["--estar_raw", str(es), "--ref_instrument", "estar_raw=solo"])
    _cfg(runs / "a4_G", "2.9.1+cu129", epochs=4.0)  # a G from another build: the whole batch is cross-build
    with pytest.raises(SystemExit, match=r"the batch spans environments.*a4_G torch 2.9.1\+cu129"):
        VD.main(rv + solo)
    _cfg(runs / "a4_G", gpu="NVIDIA H100 80GB HBM3", epochs=4.0)  # a G on another GPU model
    with pytest.raises(SystemExit, match=r"the batch spans environments.*a4_G torch 2.7.0\+cu126 / NVIDIA H100"):
        VD.main(rv + solo)
    _cfg(runs / "a4_G", epochs=4.0)
    rp = runs / "a4_n1a" / "result.json"  # a run that records no GPU model
    gpu = json.loads(rp.read_text()).pop("gpu")
    rp.write_text(json.dumps({k: v for k, v in json.loads(rp.read_text()).items() if k != "gpu"}))
    with pytest.raises(SystemExit, match=r"VOID read: a4_n1a: \S*a4_n1a/result.json records no GPU model"):
        VD.main(rv + solo)
    rp.write_text(json.dumps({**json.loads(rp.read_text()), "gpu": gpu}))
    (runs / "a4_n1a" / "config.json").unlink()
    with pytest.raises(SystemExit, match=r"VOID read: a4_n1a: no config.json"):
        VD.main(rv + solo)


def test_verdict_applies_the_registered_kill_rule(tmp_path, fake_leg):
    runs, data = tmp_path / "runs", tmp_path / "runs" / "a4_data"
    rows = [_arm(runs, "a4_n0a", 5.30, 63.2, 10.0), _arm(runs, "a4_a", 5.20, 63.1, 10.1, ("A", "a" * 64)),
            _arm(runs, "a4_n1a", 5.28, 63.0, 10.1, ("N1", "b" * 64)),
            _arm(runs, "a4_n2a", 5.25, 63.0, 10.1, ("N2", "c" * 64)),
            _arm(runs, "a4_n0b", 5.40, 62.0, 8.0), _arm(runs, "a4_b", 5.35, 62.0, 10.2, ("B", "d" * 64))]
    (data / "audit").mkdir(parents=True)
    (data / "ledger.json").write_text(json.dumps({"arms": rows}))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"dev_nats_per_word": 5.31, "fast": {VD.BLIMP: 63.0}, "run": "r1",
                                VD.GUARDED["key"]: 5.31}))
    _pw(tmp_path, 5.31)  # the base's dump beside its score.json (the base_r1 layout)
    _cfg(tmp_path)
    twin = {"a4_n0a": 5.28, "a4_a": 5.18, "a4_n1a": 5.26, "a4_n2a": 5.23}  # blank-line twins: every gain as under v1
    for n, d in twin.items():
        (runs / n / "score" / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": d}))
    (tmp_path / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": 5.29}))  # beside base.json
    args = ["--base_score", str(base), "--ref_instrument", "base=solo", "--runs_root", str(runs), "--metric", "v1"]
    with pytest.raises(SystemExit):  # amendment 4: no number before the leak audit
        VD.main(args)
    leak = {k: {"sha256": h, "words": 1, "items_hit": 0, "items_hit_synthetic_only": 0, "dev_13gram_share": 0.0,
                "dev_13gram_share_synthetic_only": 0.0}
            for k, h in (("A", "a" * 64), ("N1", "b" * 64), ("N2", "c" * 64))}
    (data / "audit" / "leak.json").write_text(json.dumps({"synthetic": leak}))
    with pytest.raises(SystemExit):  # A4b's split is not audited yet
        VD.main(args)
    leak["B"] = {**leak["A"], "sha256": "d" * 64}
    (data / "audit" / "leak.json").write_text(json.dumps({"synthetic": leak}))
    r = VD.main(args)  # the registered floor v1: pass >= 0.016, inconclusive from 0.012, BLiMP-fast drop > 2.0 kills
    a = r["families"]["a"]
    assert r["judged"] == "a" and r["thresholds"]["ref"].startswith("critic ref-0022")
    assert a["comparator"] == "N0'" and a["refs"]["N2"]["gain"] == pytest.approx(0.05)
    assert r["verdict"] == "PASS" and r["next"] is None
    assert r["promote"] is None and r["promotion"].startswith("pending: no calibrated control")  # gates: no inputs
    r = VD.main(args + ["--pass_gain", "0.08", "--kill_gain", "0.06"])  # the gain over N2 (0.05) is below 0.06
    assert r["verdict"] == "KILL" and r["promote"] is None and any("N2" in k for k in r["families"]["a"]["kill"])
    assert r["thresholds"]["ref"].startswith("custom")
    r = VD.main(args + ["--pass_gain", "0.06", "--kill_gain", "0.04"])  # 0.05 sits in the band
    assert r["verdict"] == "INCONCLUSIVE" and r["promote"] is None and r["next"] is None  # seeds follow v2 reads only
    assert r["families"]["a"]["refs"]["N2"]["status"] == "inconclusive" and not r["families"]["a"]["kill"]
    r = VD.main(args + ["--pass_gain", "0.05", "--kill_gain", "0.04"])  # a gain exactly at the pass line passes
    assert r["verdict"] == "PASS"
    r = VD.main(args + ["--blimp_kill", "0.05"])  # vs the comparator, N0' (63.2): a drop of 0.1 > 0.05
    assert r["verdict"] == "KILL" and any("BLiMP" in k for k in r["families"]["a"]["kill"])
    assert VD.main(args + ["--blimp_kill", "0.1"])["verdict"] == "PASS"  # a drop of exactly the line does not kill
    b = r["families"]["b"]  # 10.2 / 8.0 > 1.25: N0'_a joins as the FLOP-matched reference; N1b, N2b not run
    assert "FLOP-matched a4_n0a" in b["refs"] and set(b["missing"]) == {"N1", "N2"} and b["verdict"].startswith("NO")
    v2 = args[:-2]  # the primary metric (v2, the blank-line twin) on the interim v2 floor (ref-0019)
    r = VD.main(v2)
    assert r["thresholds"]["ref"].startswith("critic ref-0019") and r["thresholds"]["pass_gain"] == 0.010
    assert r["verdict"] == "PASS" and r["promote"] is None and r["families"]["a"]["dev"] == 5.18
    assert r["families"]["a"]["refs"]["base"]["gain"] == pytest.approx(0.11) and r["families"]["a"]["refs"]["N2"]["ok"]
    tw2 = lambda v: (runs / "a4_n2a" / "score" / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": v}))  # noqa: E731
    tw2(5.21)  # v2 gain over N2 0.030: passes under v2 (the guard holds by construction) ...
    assert VD.main(v2)["verdict"] == "PASS"
    r = VD.main(args)  # ... while a v1 read (information) flags it: v2 0.03 vs v1 0.05
    a = r["families"]["a"]
    assert r["verdict"] == "PROTOCOL-ARTIFACT" and a["refs"]["N2"]["status"] == "protocol-artifact"
    assert a["refs"]["N2"]["gain_v1"] == pytest.approx(0.05) and r["promote"] is None
    tw2(5.19)  # v2 gain over N2 0.010 passes over a base, but against a null the bar is 0.020 (critic ref-0028)
    r = VD.main(v2)
    a = r["families"]["a"]
    assert r["verdict"] == "INCONCLUSIVE" and a["refs"]["N2"]["null"] and not a["refs"]["base"]["null"]
    assert any("null bar" in x for x in a["inconclusive"]) and r["next"].endswith("in the band: N2 (critic ref-0028)")
    assert VD.null_bar(VD.thresholds(), True)["pass_gain"] == 0.020 and VD.null_bar(VD.thresholds(), False) == \
        VD.thresholds()
    tw2(5.189)  # v2 gain 0.009: inconclusive band 0.0074-0.010
    r = VD.main(v2)
    assert r["verdict"] == "INCONCLUSIVE" and "3-seed" in r["next"] and "claim floor" in r["next"]
    tw2(5.187)  # v2 gain 0.007 < 0.0074: kill
    assert VD.main(v2)["verdict"] == "KILL"
    (runs / "a4_n2a" / "score" / "dev_bpw_twin.json").unlink()  # a v1 read cannot pass before the guard has the twin
    r = VD.main(args)
    assert r["verdict"].startswith("PENDING") and r["families"]["a"]["guard_pending"] == ["N2"]
    assert VD.main(v2)["verdict"].startswith("NO VERDICT")  # under v2 the reference has no number at all
    tw2(5.23)  # back to a v2 pass on every reference (N2 gain 0.05); the never-seen-character leg (ref-0001) next
    r = VD.main(v2)
    a = r["families"]["a"]
    assert r["verdict"] == "PASS" and a["refs"]["N2"]["gain_ex_unseen"] == pytest.approx(0.05)
    assert a["refs"]["base"]["gain_ex_unseen"] == pytest.approx(0.11) and a["unseen_leg"]["arm"] == 5.20
    assert a["refs"]["N2"]["gain_ex_unseen_harness"] == pytest.approx(0.05)  # the harness cross-check, reported
    arm_sc = runs / "a4_a" / "score" / "score.json"
    _gx(arm_sc, 5.241)  # leg gain over N2 0.009: in the band, so the reading is inconclusive
    r = VD.main(v2)
    assert r["verdict"] == "INCONCLUSIVE" and r["families"]["a"]["refs"]["N2"]["status"] == "inconclusive"
    assert any("never-seen" in x for x in r["families"]["a"]["inconclusive"])
    _gx(arm_sc, 5.245)  # leg gain 0.005 < 0.0074: a kill although the full-DEV gain passes
    r = VD.main(v2)
    assert r["verdict"] == "KILL" and any("never-seen" in x for x in r["families"]["a"]["kill"])
    _gx(arm_sc, None)  # no number for the leg: no pass (critic ref-0025: a missing leg is pending)
    r = VD.main(v2)
    assert r["verdict"].startswith("PENDING") and "never-seen" in r["verdict"]
    twin_a = runs / "a4_a" / "score" / "dev_bpw_twin.json"  # the number from the twin json beside score.json
    ex = {"nats_per_word": 5.18, "ex_unseen_chars_nats_per_word": 5.20, "unseen_chars_count": 85}
    twin_a.write_text(json.dumps({**ex, "unseen_chars_words": 10_277}))
    r = VD.main(v2)
    assert r["verdict"] == "PASS" and r["families"]["a"]["unseen_leg"]["arm"] == 5.20
    twin_a.write_text(json.dumps({**ex, "unseen_chars_words": 10_275}))  # other counts: not the canonical number
    r = VD.main(v2)
    assert r["verdict"].startswith("PENDING") and r["families"]["a"]["unseen_leg"]["other_counts"] == ["arm"]
    twin_a.write_text(json.dumps({**ex, "unseen_chars_words": 10_277}))
    _pw(runs / "a4_a", 5.245)  # the harness cross-check alone (0.005 over N2) never changes a status
    r = VD.main(v2)
    assert r["verdict"] == "PASS" and r["families"]["a"]["refs"]["N2"]["gain_ex_unseen_harness"] == pytest.approx(0.005)
    _pw(runs / "a4_a", 5.20)
    with pytest.raises(SystemExit, match="VOID cross-instrument read: base ran solo, the batch packed"):
        VD.main(v2 + ["--packed"])  # a packed screen needs a packed base (ENV_RULE, critic ref-0039)
    pk = ["--packed", "--ref_instrument", "base=packed"]
    r = VD.main(v2 + pk)  # a packed screen: the packed floor (0.0144 / 0.0102)
    assert r["thresholds"]["ref"].startswith("critic ref-0001") and r["thresholds"]["condition"] == "packed"
    assert r["verdict"] == "PASS" and VD.thresholds()["condition"] == "solo"
    tw2(5.19)  # v2 gain over N2 0.010: solo inconclusive (null bar 0.020), a packed kill (below 0.0102)
    assert VD.main(v2)["verdict"] == "INCONCLUSIVE" and VD.main(v2 + pk)["verdict"] == "KILL"
    with pytest.raises(SystemExit):  # no packed floor on v1
        VD.main(args + ["--packed"])


def test_unseen_leg_follows_the_pinned_tool(tmp_path):
    """babylm.a4.unseen = unseen_chars.py's rules: never-seen = DEV chars - train_10M chars; a word is flagged when it
    has a non-ASCII token and its tokens decode to a never-seen character (an ASCII one, '>', alone does not flag);
    rest = (sum - flagged sum) / W, NaN counted 0; a per-token dump is summed per word."""
    from babylm.data import prepare as P
    train = ["the cat sat on the café mat", "", "a dog ran home", "", "a big vexing box", ""]
    dev = ["the naïve cat sat", "", "a>b café Ğx dog", "", "the cat ran ÿes", ""]
    for split, lines in (("train_10M", train), ("dev", dev)):
        for s in C.SOURCES:
            q = C.raw_path(tmp_path, split, s)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text("\n".join(lines), encoding="utf-8")
    docs = P.load_docs(tmp_path, "dev")
    toks = P.tokenize_split(P.load_tokenizer(), docs)
    tokens, words, src, per = P.assemble([(i, toks[s], list(range(len(docs[s].ranges)))) for i, s in enumerate(C.SOURCES)])
    P.write_split("dev", tokens, words, src, per, {}, tmp_path, C.sha256_file(C.TOKENIZER_PATH))
    f = UN.flagged(str(tmp_path))
    assert f["never"] == {"ï", ">", "Ğ", "ÿ"}
    want = np.array([any(c in "ïĞÿ" for c in w) for s in C.SOURCES for t in docs[s].texts for w in t.split()])
    assert f["W"] == want.size and (f["bad"][:want.size] == want).all() and int(f["bad"].sum()) == 18
    rng = np.random.default_rng(0)
    arm, ref = rng.uniform(1, 9, want.size), rng.uniform(1, 9, want.size)
    arm[0] = np.nan  # a word with the unscored first token
    for n, x in (("arm", arm), ("ref", ref)):
        (tmp_path / n / "pw").mkdir(parents=True)
        np.save(tmp_path / n / "pw" / UN.PW, x.astype(np.float16))
    r = UN.rests({"arm": UN.find(tmp_path / "arm"), "ref": UN.find(tmp_path / "ref" / "score.json"), "x": None},
                 tmp_path, check=False)
    for n, x in (("arm", arm), ("ref", ref)):
        x = np.nan_to_num(x.astype(np.float16).astype(np.float64))
        assert r[n] == pytest.approx((x.sum() - x[want].sum()) / want.size)
    assert r["x"] is None and r["_info"]["words"] == 18 and r["_info"]["chars"] == 4
    pt = rng.uniform(0, 3, f["n_tokens"])
    np.save(tmp_path / "pt.npy", pt)
    per_word = np.bincount(f["wid"], weights=pt)
    assert UN.split(tmp_path / "pt.npy", f)["rest"] == pytest.approx(per_word[~want].sum() / want.size)
    with pytest.raises(SystemExit):  # not the pinned derivation (85 characters, 10,275 words)
        UN.rests({"arm": UN.find(tmp_path / "arm")}, tmp_path)
    assert UN.rests({"arm": None})["arm"] is None  # no dump: no data needed
    r = VD.unseen_rests({"arm": UN.find(tmp_path / "arm")}, str(tmp_path / "nowhere"))  # the verdicts' cross-check
    assert r["arm"] is None and "no raw text" in r["_info"]["error"]


# ------------------------------------------------------------------------------------------------
# T2: job files, gate readout, verdict
# ------------------------------------------------------------------------------------------------
def test_make_a4_t2_jobs(tmp_path):
    r = MT.main(["--out_dir", str(tmp_path), "--commit", "a" * 40])
    jobs = {j["name"]: j["args"] for j in r["trainer"]["jobs"]}
    assert set(jobs) == {*MT.GATE_RUNS, MT.G_RUN, "a4_t2", "a4_t2_n0", "a4_t2_n1", "a4_t2_n2"}
    val = lambda a, f: a[a.index(f) + 1] if f in a else None  # noqa: E731
    for n, a in jobs.items():
        TR.build_parser().parse_args(a)  # every job parses
        gate = n.startswith("a4_t2g")
        assert val(a, "--split") == ("train_1M" if gate else "train_10M") and val(a, "--seed") == "0"
        if n in MT.GATE_REFS + (MT.G_RUN,):  # G (5 epochs, annealed by the base schedule), base@1M, size M @1M
            g = n in (MT.G_RUN, MT.G1_RUN)
            assert val(a, "--epochs") == ("5" if g else "10") and "--init_from" not in a and "--sources" not in a
            assert val(a, "--size") == ("M" if n[-1] == "M" else "S") and val(a, "--cooldown_frac") == "0.5"
            assert "--save" in a and "--ema_decay" not in a
            continue
        src = a[a.index("--sources") + 1:a.index("--tag")]
        src = src[:next((i for i, x in enumerate(src) if x.startswith("--")), len(src))]
        g = MT.G1_CKPT if gate else MT.G_CKPT
        assert g.endswith(f"{MT.G1_RUN if gate else MT.G_RUN}/resume.pt")
        assert "--epochs" not in a and val(a, "--init_from") == g and val(a, "--exposure_rule") == "t2"
        assert val(a, "--exposure_cap") == (str(MT.CAP_1M) if gate else None)
        assert src[0] == f"real:train_{'1' if gate else '10'}M:5"
        null0 = n.endswith("_n0")
        assert len(src) == (1 if null0 else 2) and (val(a, "--kd_teacher") == (None if null0 else g))
        if not null0:
            assert val(a, "--kd_rows") == "synthetic" and val(a, "--kd_alpha") == "0.5" and val(a, "--kd_temp") == "1.0"
    # corpus words: G@1M (123 steps of train_1M = base@1M's 5M milestone) + N0' real; G (1218 steps of train_10M, the
    # round-1 base's words at that step) + 5 real epochs
    assert MT.CAP_1M == 10_060_938 and 5_039_418 + 4_998_319 <= MT.CAP_1M
    assert 49_791_035 + 49_763_942 <= MJ.CAP
    gate, b1, b2 = r["gate"], r["batch1"], r["batch2"]
    assert all(f["commit"] == "a" * 40 and f["keep_checkpoints"] for f in (gate, b1, b2))
    assert MT.TRAIN_1M["sha256"] in gate["setup"]
    for f in (gate, b1, b2):
        blob = json.dumps(f)
        assert "G_URI" not in blob and "chck_50M" not in blob and "chck_5M" not in blob
        names = [x["name"] for x in f["jobs"]]
        assert len(names) == len(set(names)) and "pool: moonshot" in f["note"]
        for x in f["jobs"]:  # every score gets its blank-line twin, merged into score.json
            assert x["cmd"].count("score_trial.sh") == x["cmd"].count("dev_bpw_twin.py") == x["cmd"].count(
                "babylm.a4.t2 twin")
    assert [x["name"] for x in gate["jobs"]][:2] == [MT.G1_RUN, "a4_t2g_generate"]
    assert [x["name"] for x in b1["jobs"]][:2] == [MT.G_RUN, "a4_t2_generate"]
    assert f"--only {MT.G_RUN}" not in json.dumps(b2) and MT.G_CKPT in json.dumps(b2)  # batch 2 restores G
    blob = json.dumps(gate)
    assert "--seed_split train_1M --block_split train_1M --words 10000000" in blob and "train_10M --" not in blob
    assert f"--oracle_dev {MT.ORACLE_DEV} --out" in blob and "a4_t2g_score_refs" in blob and "--metric" not in blob
    assert "--words 100000000" in json.dumps(b1) and "--selfseed" in json.dumps(b2) and "--selfseed" not in json.dumps(b1)
    assert "--base_epochs 10" in json.dumps(b1) and "babylm.a4.t2 verdict" not in json.dumps([b1, b2])
    # critic ref-0001 / ref-0027: the 10M screen runs solo, and each student writes its per-word dumps (the
    # never-seen-character leg); the gate (labels only) may run packed and needs no dumps
    assert b1["solo"] is True and b2["solo"] is True and "solo" not in gate and "SOLO" in b1["note"] + b2["note"]
    for f, want in ((b1, {"a4_t2", "a4_t2_n0"}), (b2, {"a4_t2_n1", "a4_t2_n2"}), (gate, set())):
        has = {x["name"] for x in f["jobs"] if "babylm.analysis.perword" in x["cmd"]}
        assert has == want and all(f"perword babylm/runs/{n} --epochs_only" in json.dumps(f) for n in want)
    assert "pw/" in b1["note"] and "pw/" not in gate["note"] and "may run packed" in gate["note"]
    for f in (gate, b1, b2):  # the twin @ ref-0026 (the canonical second leg) replaces the vendored copy, sha checked
        assert f"gsutil -q cp {MT.TWIN['uri']} babylm/eval/dev_bpw_twin.py; echo '{MT.TWIN['sha256']}  " \
               "babylm/eval/dev_bpw_twin.py' | sha256sum -c -" in f["setup"]
    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=MJ.HERE, capture_output=True).returncode == 0:  # VMs: no .git
        with pytest.raises(SystemExit, match="10-epoch"):  # G = 5 of the 10-epoch base's epochs
            MT.main(["--out_dir", str(tmp_path), "--commit", "a" * 40, "--base_job",
                     "HEAD:babylm/configs/jobs_a4_screen.json:a4_G"])


def test_vendored_twin_is_the_registered_one_and_merges(tmp_path):
    import hashlib
    sha = "9d6d8ad2ad77a0dd99c3532158b1dbf68ed02812194601cf79766942731cc775"  # dev_bpw_twin.py @ ref-0049
    assert hashlib.sha256(T2.TWIN_SCRIPT.read_bytes()).hexdigest() == sha
    _t2_run(tmp_path, "r", 5.03, 68.0)
    assert T2.main(["twin", str(tmp_path / "r")]) == {str(tmp_path / "r"): None}  # the twin did not run: no change
    (tmp_path / "r" / "score" / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": 5.01, "separators": 7}))
    T2.main(["twin", str(tmp_path / "r")])
    s = json.loads((tmp_path / "r" / "score" / "score.json").read_text())
    assert s["dev_nats_per_word_twin"] == 5.01 and s["dev_nats_per_word"] == 5.03 and s["dev_twin"]["script_sha256"] == sha
    assert VD.score(tmp_path / "r")["dev"] == 5.01 and VD.score(tmp_path / "r", "v1")["dev"] == 5.03
    assert VD.score(tmp_path / "r")["v2x"] == 5.03  # _t2_run's number; the twin @ ref-0049 has none
    ex = {"nats_per_word": 5.01, "ex_unseen_chars_nats_per_word": 4.98, "unseen_chars_count": 85,
          "unseen_chars_words": 10_277, "unseen_chars_sha256": "u" * 64}  # the twin @ ref-0026 (critic ref-0023)
    (tmp_path / "r" / "score" / "dev_bpw_twin.json").write_text(json.dumps(ex))
    T2.main(["twin", str(tmp_path / "r")])
    s = json.loads((tmp_path / "r" / "score" / "score.json").read_text())
    assert s[VD.GUARDED["key"]] == 4.98 and s["dev_twin"]["unseen_chars_words"] == 10_277
    assert VD.score(tmp_path / "r")["v2x"] == 4.98 and VD.score(tmp_path / "r")["v2x_counts"] == [85, 10_277]


def _t2_run(root, name, dev, blimp, syn=None):
    d = root / name
    (d / "score").mkdir(parents=True)
    src = ["real:train_10M:5"] + ([f"synthetic:{syn[0]}:1"] if syn else [])
    sha = ["r" * 64] + ([syn[1]] if syn else [])
    (d / "result.json").write_text(json.dumps({"exposure": {"sources": src, "source_sha256": sha}}))
    _cfg(d)
    (d / "score" / "score.json").write_text(json.dumps({"dev_nats_per_word": dev, "fast": {VD.BLIMP: blimp},
                                                        VD.GUARDED["key"]: dev}))
    _pw(d, dev)


def _leak(data, splits):
    (data / "audit").mkdir(parents=True, exist_ok=True)
    leak = {k: {"sha256": h, "words": 1, "items_hit": 0, "items_hit_synthetic_only": 0, "dev_13gram_share": 0.0,
                "dev_13gram_share_synthetic_only": 0.0} for k, h in splits}
    (data / "audit" / "leak.json").write_text(json.dumps({"synthetic": leak}))


def test_t2_gate_readout(tmp_path):
    runs, data = tmp_path / "runs", tmp_path / "runs" / "a4_t2g_data"
    devs = {"a4_t2g_base": 6.10, "a4_t2g_M": 6.05, "a4_t2g": 5.85, "a4_t2g_n0": 5.95, "a4_t2g_n1": 5.97,
            "a4_t2g_n2": 6.00}
    syn = {"a4_t2g": ("T", "a" * 64), "a4_t2g_n1": ("N1", "b" * 64), "a4_t2g_n2": ("N2", "c" * 64)}
    for n, d in devs.items():
        _t2_run(runs, n, d, 60.0, syn.get(n))
    _t2_run(runs, "a4_t2g_G", 6.30, 59.0)  # the teacher: reported, never a comparator
    oracle = tmp_path / "o" / "score.json"  # the E10 base at 10M, in the gate's build
    _cfg(oracle.parent)
    oracle.write_text(json.dumps({"dev_nats_per_word": 5.0367, "fast": {VD.BLIMP: 70.5}, VD.GUARDED["key"]: 5.0367}))
    (oracle.parent / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": 5.0167}))
    gv1 = ["gate", "--runs_root", str(runs), "--oracle_score", str(oracle), "--ref_instrument", "oracle=solo",
           "--metric", "v1"]
    with pytest.raises(SystemExit):  # audits first
        T2.main(gv1)
    _leak(data, syn.values())
    r = T2.main(gv1 + ["--out", str(data / "gate.json")])
    assert r["G1"] == pytest.approx(1.0633) and r["phi"]["arm"] == pytest.approx(0.25 / 1.0633)
    assert r["binding_comparator"] == "N0'" and r["lead"] == pytest.approx(0.10) and r["rule"]["lead_min"] == 0.016
    assert r["decision"].startswith("NO READOUT") and r["guard"]["ok"] is None  # the guard needs the twins
    assert (data / "gate.md").exists() and "labels only" in r["label"] and r["teacher"]["dev"] == 6.30
    for n, d in devs.items():  # twins: every DEV 0.02 lower, gains unchanged
        (runs / n / "score" / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": d - 0.02}))
    r = T2.main(gv1)
    assert r["decision"].startswith("proceed") and r["guard"]["ok"]
    # the step rule (experimenter 14:48Z): train_1M has 1,001,443 words, so each gate run ends just past a milestone
    _steps(runs / "a4_t2g_n2", "chck_20M.pt", [("20M", 488), ("final", 489)], 489)
    with pytest.raises(SystemExit, match="VOID score.*N2: scored chck_20M.pt at step 488"):
        T2.main(gv1)
    _steps(runs / "a4_t2g_n2", "chck_20M.pt", [("20M", 489)], 489)
    _steps(runs / "a4_t2g_G", "chck_5M.pt", [("5M", 122), ("final", 123)], 123)  # the teacher is read too
    with pytest.raises(SystemExit, match=r"G@1M \(teacher\): scored chck_5M.pt at step 122"):
        T2.main(gv1)
    _steps(runs / "a4_t2g_G", "chck_5M.pt", [("5M", 123)], 123)
    r = T2.main(gv1)
    assert r["step_check"]["N2"]["ok"] and r["step_check"]["G@1M (teacher)"]["ok"] and r["step_check"]["arm"] is None
    gv2 = ["gate", "--runs_root", str(runs), "--oracle_score", str(oracle), "--ref_instrument", "oracle=solo"]  # v2
    r = T2.main(gv2)
    assert r["decision"].startswith("proceed") and r["dev"]["arm"] == pytest.approx(5.83) and r["rule"]["lead_min"] == 0.016
    assert r["phi"]["arm"] == pytest.approx(0.25 / 1.0633) and r["dev_v1"]["arm"] == 5.85
    assert T2.main(gv2 + ["--lead_min", "0.2"])["decision"] == "killed (proxy)"
    with pytest.raises(SystemExit, match="oracle: a bare number"):  # ENV_RULE: a number carries no build record
        T2.main(["gate", "--runs_root", str(runs), "--oracle_dev", "5.0167"])
    _cfg(oracle.parent, "2.9.1+cu129")  # base_r1 s0's build: refused, no pair admits it (critic ref-0038)
    with pytest.raises(SystemExit, match=r"VOID cross-environment read: oracle ran on torch 2.9.1\+cu129"):
        T2.main(gv2)
    _cfg(oracle.parent)  # v2e: the oracle in the gate's environment
    assert T2.main(gv2)["env_check"]["home"] == BUILD
    with pytest.raises(SystemExit, match="VOID read: no packing instrument declared for oracle"):  # ref-0039
        T2.main(gv2[:-2])
    with pytest.raises(SystemExit, match="VOID cross-instrument read: oracle ran solo, the batch packed"):
        T2.main(gv2 + ["--packed"])  # a packed gate (labels only) needs a packed oracle
    assert T2.main(gv2 + ["--packed", "--ref_instrument", "oracle=packed"])["env_check"]["instrument"] == "packed"
    import shutil
    shutil.rmtree(runs / "a4_t2g")
    _t2_run(runs, "a4_t2g", 5.90, 60.0, syn["a4_t2g"])  # phi 0.188 < 0.20, a experimenter of 0.05 over N0'
    r = T2.main(gv1)
    assert r["decision"] == "killed (proxy)" and any("phi_stu" in k for k in r["kill"])
    shutil.rmtree(runs / "a4_t2g_M")
    _t2_run(runs, "a4_t2g_M", 5.89, 60.0)  # the size-M reference binds: the arm leads it by 0.01 only
    r = T2.main(gv1)
    assert r["binding_comparator"] == "FLOP ref (size M @1M)" and len(r["kill"]) == 2
    shutil.rmtree(runs / "a4_t2g_n1")
    r = T2.main(gv1)
    assert r["decision"] == "killed (proxy)" and r["missing"] == ["N1"]  # a kill shows before every run is in


def test_t2_verdict(tmp_path, fake_leg):
    import shutil
    runs, data = tmp_path / "runs", tmp_path / "runs" / "a4_t2_data"
    tw = lambda d, v: (d / "dev_bpw_twin.json").write_text(json.dumps({"nats_per_word": v}))  # noqa: E731

    def run(n, dev, blimp, syn=None):  # every score with its blank-line twin, 0.02 lower (gains unchanged)
        _t2_run(runs, n, dev, blimp, syn)
        tw(runs / n / "score", dev - 0.02)

    (tmp_path / "b").mkdir()
    base = tmp_path / "b" / "score.json"
    base.write_text(json.dumps({"dev_nats_per_word": 5.0367, "fast": {VD.BLIMP: 70.5}, "run": "r1",
                                VD.GUARDED["key"]: 5.0367}))
    tw(tmp_path / "b", 5.0167)
    _pw(tmp_path / "b", 5.0367)
    _cfg(tmp_path / "b")
    syn = {"a4_t2": ("T", "a" * 64), "a4_t2_n1": ("N1", "b" * 64), "a4_t2_n2": ("N2", "c" * 64)}
    run("a4_t2", 5.000, 70.0, syn["a4_t2"])
    run("a4_t2_n0", 5.030, 70.2)
    _leak(data, [syn["a4_t2"]])
    (data / "ledger.json").write_text(json.dumps({"arms": [
        {"run": "a4_t2", "within_cap": True, "words_capped": 99_554_977, "flops_vs_base": 2.8},
        {"run": "a4_t2_n0", "within_cap": True}]}))
    args = ["verdict", "--runs_root", str(runs), "--base_score", str(base), "--ref_instrument", "base=solo",
            "flop_ref=solo", "--metric", "v1"]
    r = T2.main(args)  # batch 1: the arm clears the base (0.0367) and N0' (0.030); N1, N2, FLOP ref not in yet
    assert r["clears_base_and_n0"] and r["verdict"].startswith("PENDING") and "batch 2" in r["verdict"]
    assert set(r["missing"]) == {"N1", "N2", T2.FLOP_REF} and r["labels"][0] == T2.TIER
    run("a4_t2_n1", 5.020, 70.1, syn["a4_t2_n1"])
    run("a4_t2_n2", 5.050, 69.0, syn["a4_t2_n2"])
    with pytest.raises(SystemExit):  # batch 2's splits are not audited
        T2.main(args)
    _leak(data, syn.values())
    (tmp_path / "m").mkdir()
    fref = tmp_path / "m" / "score.json"
    fref.write_text(json.dumps({"dev_nats_per_word": 5.025, "fast": {VD.BLIMP: 71.0}, VD.GUARDED["key"]: 5.025}))
    tw(tmp_path / "m", 5.005)
    _pw(tmp_path / "m", 5.025)
    _cfg(tmp_path / "m", size="M")
    r = T2.main(args + ["--flop_ref_score", str(fref), "--out", str(data / "v.json")])
    assert r["binding_comparator"] == "N1" and r["gain_over_binding"] == pytest.approx(0.02)
    assert r["verdict"] == "PASS" and not r["promote"] and "reproduction at strict-small" in r["next"]
    assert r["blimp_ref"] == "N1" and r["blimp_drop"] == pytest.approx(0.1) and (data / "v.md").exists()
    r = T2.main(args + ["--flop_ref_score", str(fref), "--blimp_kill", "0.05"])
    assert r["verdict"] == "KILL" and any("BLiMP" in k for k in r["kill"])
    shutil.rmtree(runs / "a4_t2")
    run("a4_t2", 5.025, 70.0, syn["a4_t2"])  # 0.005 over N0', 0.0 over the FLOP ref
    r = T2.main(args)
    assert r["verdict"] == "KILL" and r["missing"] == [T2.FLOP_REF]  # a kill is final with a reference missing
    shutil.rmtree(runs / "a4_t2")
    run("a4_t2", 4.70, 70.0, syn["a4_t2"])
    r = T2.main(args + ["--flop_ref_score", str(fref)])
    assert r["verdict"] == "PASS" and not r["promote"] and r["next"].startswith("promotion gates: pending")
    (data / "ledger.json").write_text(json.dumps({"arms": [{"run": "a4_t2", "within_cap": False}]}))
    assert T2.main(args + ["--flop_ref_score", str(fref)])["verdict"] == "KILL"
    (data / "ledger.json").write_text(json.dumps({"arms": [{"run": "a4_t2", "within_cap": True}]}))
    v2 = args[:-2] + ["--flop_ref_score", str(fref)]  # the primary metric on the interim v2 floor (ref-0019)
    r = T2.main(v2)
    assert r["verdict"] == "PASS" and r["promotion"].startswith("pending")
    assert r["refs"]["N1"]["gain"] == pytest.approx(0.32)
    sc = {"base": VD.score_file(base), "flop_ref": VD.score_file(fref), "estar_cal": {"v2": 4.99, "v2x": 5.01},
          **{n: VD.score(runs / n) for n in ("a4_t2", "a4_t2_n0", "a4_t2_n1", "a4_t2_n2")}}
    es = tmp_path / "es" / "score.json"
    es.parent.mkdir()
    es.write_text(json.dumps({VD.METRICS["v2"]: 4.99, VD.GUARDED["key"]: 5.01}))
    _cfg(es.parent, epochs=5.0)
    _calib(tmp_path / "calib.json", sc)
    ev = _evidence(tmp_path / "instruments.tsv", ["a4_t2", "a4_t2_n0", "a4_t2_n1", "a4_t2_n2", "r1", "m", "es"])
    gx = ["--calib", str(tmp_path / "calib.json"), "--estar_raw", str(es), "--estar", "E*_raw = E*_cal = 5 (test)",
          "--ref_instrument", "estar_raw=solo"]
    r = T2.main(v2 + gx)  # critic ref-0035: no VM evidence, no promotion
    assert r["verdict"] == "PASS" and not r["promote"] and r["promotion"] == (
        "pending: the instrument is not corroborated from VM evidence for a4_t2, a4_t2_n0, a4_t2_n1, a4_t2_n2, r1, m, "
        "es (critic ref-0035 / ref-0036: --vm_evidence or a complete runner sidecar)")
    gx += ["--vm_evidence", str(ev)]
    r = T2.main(v2 + gx + ["--out", str(data / "g.json")])  # every gate clears: the 0.32 gain is promoted
    assert r["promote"] and r["promotion"] == "promote" and "3 seeds" in r["next"]
    assert "7 of 7 compared runs corroborated solo" in (data / "g.md").read_text()
    _evidence(ev, ["a4_t2", "a4_t2_n0", "a4_t2_n1", "a4_t2_n2", "r1", "m", "es"], gpu="NVIDIA H100 80GB HBM3")
    with pytest.raises(SystemExit, match="VOID read: a4_t2: result.json records NVIDIA A100-SXM4-40GB"):
        T2.main(v2 + gx)
    _evidence(ev, ["a4_t2", "a4_t2_n0", "a4_t2_n1", "a4_t2_n2", "r1", "m", "es"])
    assert r["gates"]["kill_lines"]["calibrated_t"] == {"full": 0.0095, "guarded": 0.0094}  # critic ref-0032
    assert r["env_check"]["ref_instruments"] == {"base": "solo", T2.FLOP_REF: "solo", "estar_raw": "solo"}
    assert f"scope: {VD.SCOPE}" in (data / "g.md").read_text()
    assert r["gates"]["estar_raw"]["full"] == pytest.approx(0.31) and r["estar"]["note"].startswith("E*_raw")
    assert "calibrated_t over N1" in (data / "g.md").read_text()
    _calib(tmp_path / "calib.json", sc, {"a4_t2_n1": 0.346})  # calibrated_t gain over N1: 0.32 - 0.346 + 0.03
    r = T2.main(v2 + gx)
    assert r["verdict"] == "PASS" and not r["promote"]
    assert r["next"] == "promotion gates: not promoted: calibration-equivalent"
    _calib(tmp_path / "calib.json", sc)
    assert r["dev"] == pytest.approx(4.68) and r["refs"]["base"]["dev"] == 5.0167 and r["refs"]["N1"]["ok"]
    tw(runs / "a4_t2_n1" / "score", 4.695)  # v2 gain over N1 0.015: below the null bar 0.020 (critic ref-0028)
    r = T2.main(v2)
    assert r["verdict"] == "INCONCLUSIVE" and any("null bar" in x for x in r["inconclusive"])
    tw(runs / "a4_t2_n1" / "score", 4.70)  # v2 gain over N1 0.02 (v1 0.32): a pass under v2, not a promotion
    r = T2.main(v2)
    assert r["verdict"] == "PASS" and not r["promote"] and r["binding_comparator"] == "N1"
    r = T2.main(args + ["--flop_ref_score", str(fref)])  # a v1 read (information) flags the gap
    assert r["verdict"] == "PROTOCOL-ARTIFACT" and not r["promote"] and r["artifact"]
    tw(runs / "a4_t2_n1" / "score", 4.6855)  # v2 gain 0.0055 < 0.0074 over N1: a kill
    assert T2.main(v2)["verdict"] == "KILL"
    (runs / "a4_t2_n1" / "score" / "dev_bpw_twin.json").unlink()
    r = T2.main(args + ["--flop_ref_score", str(fref)])  # v1 read without N1's twin: the guard cannot pass it yet
    assert r["verdict"].startswith("PENDING") and r["guard_pending"] == ["N1"]
    assert T2.main(v2)["missing"] == ["N1"]
    tw(runs / "a4_t2_n1" / "score", 4.70)  # a v2 pass (0.02 over N1); the never-seen-character leg (ref-0001) next
    r = T2.main(v2)
    assert r["verdict"] == "PASS" and r["refs"]["N1"]["gain_ex_unseen"] == pytest.approx(0.32)
    assert r["refs"]["N1"]["gain_ex_unseen_harness"] == pytest.approx(0.32)  # the cross-check, reported
    n1 = runs / "a4_t2_n1" / "score" / "score.json"
    _gx(n1, 4.709)  # leg gain over N1 0.009: inconclusive
    assert T2.main(v2)["verdict"] == "INCONCLUSIVE"
    _gx(n1, 4.705)  # leg gain 0.005: a kill
    r = T2.main(v2)
    assert r["verdict"] == "KILL" and any("never-seen" in x for x in r["kill"])
    _pw(runs / "a4_t2_n1", 4.705)
    _gx(n1, 5.02)  # the harness cross-check alone (0.005) never changes a status
    r = T2.main(v2)
    assert r["verdict"] == "PASS" and r["refs"]["N1"]["gain_ex_unseen_harness"] == pytest.approx(0.005)
    _gx(n1, None)
    r = T2.main(v2)
    assert r["verdict"].startswith("PENDING") and r["leg_pending"] == ["N1"]
    with pytest.raises(SystemExit, match=r"VOID cross-instrument read: base ran solo; FLOP ref \(size M\) ran solo, "
                                         r"the batch packed"):
        T2.main(v2 + ["--packed"])  # a packed screen needs packed references (ENV_RULE, critic ref-0039)
    r = T2.main(v2 + ["--packed", "--ref_instrument", "base=packed", "flop_ref=packed"])  # N1's leg still missing
    assert r["thresholds"]["condition"] == "packed" and r["verdict"].startswith("PENDING")
    _steps(runs / "a4_t2", "chck_200M.pt", [("200M", 3650), ("final", 3651)], 3651)  # the step rule (experimenter 14:48Z)
    with pytest.raises(SystemExit, match="VOID score.*arm: scored chck_200M.pt at step 3650"):
        T2.main(v2)
    _steps(runs / "a4_t2", "chck_200M.pt", [("200M", 3651)], 3651)
    assert T2.main(v2)["step_check"]["arm"]["ok"]
    _cfg(tmp_path / "m", "2.9.1+cu129", size="M")  # ENV_RULE: the size-M reference in another build
    with pytest.raises(SystemExit, match=r"FLOP ref \(size M\) ran on torch 2.9.1\+cu129 / NVIDIA A100-SXM4-40GB, "
                                         r"the batch on torch 2.7.0\+cu126"):
        T2.main(v2)
