"""CPU tests for temporal self-distillation (babylm.harness.self_distill) and its harness wiring."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from babylm.data import corpus as C
from babylm.harness import levers as L
from babylm.harness import self_distill as SD
from babylm.harness import stream as S
from babylm.harness import train as TR
from babylm.tests.conftest import tiny_args
from babylm.tests.test_levers import synth_stream


def _train_recs(out):
    return [r for r in (json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()) if r.get("type") == "train"]


# ------------------------------------------------------------------------------------------------
# positions: the cache key of every target is its stream position, under the samplers' random offsets
# ------------------------------------------------------------------------------------------------
def test_stream_positions_exact_under_random_offsets():
    tok = synth_stream(n_lines=300)
    T, bs, E = 32, 4, 4
    smp = S.EpochSampler(len(tok), T, data_seed=11)
    n_steps = E * ((len(tok) - 1) // T) // bs
    visits = np.zeros(len(tok), dtype=np.int64)
    offs = set()
    for _ in range(n_steps):
        st = smp.next_starts(bs)
        offs.add(int(st.min() % T))
        x, y = S.gather_windows(tok, st, T)
        pos = SD.stream_positions(st, T)
        assert np.array_equal(tok[pos].astype(np.int64), y.numpy())       # key -> the target token itself
        assert np.array_equal(tok[pos - 1].astype(np.int64), x.numpy())   # and its input is the previous token
        np.add.at(visits, pos.ravel(), 1)
    assert len(offs) > 1  # the epochs really used different offsets
    assert visits[0] == 0 and visits.max() <= E + 1


@pytest.mark.parametrize("plan", ["line:32,line:64,line:32", "stream:32,line:48,stream:32"])
def test_plan_positions_exact(plan):
    tok = synth_stream(n_lines=300)
    starts = L.line_starts(tok, np.array([7]), 0)
    smp = L.PlanSampler(tok, L.parse_plan(plan, 3, 64), 4 * 64, seed=3, bos_id=0, pad_id=0, starts=starts)
    for i in range(smp.total_steps):
        x, y, _, Tw, tpos = smp.batch_at(i)
        pos = SD.plan_positions(y, tpos)
        v = y.numpy() >= 0
        assert np.array_equal(pos >= 0, v)
        assert np.array_equal(tok[pos[v]].astype(np.int64), y.numpy()[v])


def test_cache_tracks_positions_over_epochs():
    """A fake model whose prediction at stream position p peaks on token f(p): after any number of shuffled epochs
    the cache's top entry at every visited p is f(p), and the visit count equals the sampler's visits."""
    tok = synth_stream(n_lines=200)
    N, T, bs, V, k = len(tok), 16, 8, 50, 4
    f = lambda p: (p * 7 + 3) % V  # noqa: E731
    cache = SD.TargetCache(N, k, decay=0.6, vocab=V)
    smp = S.EpochSampler(N, T, data_seed=2)
    visits = np.zeros(N, dtype=np.int64)
    for _ in range(3 * ((N - 1) // T) // bs):
        pos = SD.stream_positions(smp.next_starts(bs), T).ravel()
        z = torch.full((len(pos), V), -4.0)
        z[torch.arange(len(pos)), torch.from_numpy(f(pos))] = 4.0
        pt = torch.from_numpy(pos)
        cache.update(pt, F.log_softmax(z, -1))
        np.add.at(visits, pos, 1)
    seen = np.flatnonzero(visits)
    idx, p, has = cache.read(torch.from_numpy(seen))
    assert has.all() and not cache.read(torch.tensor([0]))[2].any()
    assert np.array_equal(idx[:, 0].numpy(), f(seen))
    assert np.array_equal(cache.cnt[torch.from_numpy(seen)].numpy().astype(np.int64), visits[seen])
    assert (p.sum(-1) <= 1 + 1e-3).all()


# ------------------------------------------------------------------------------------------------
# EMA merge and KL math
# ------------------------------------------------------------------------------------------------
def test_merge_is_exact_ema_without_truncation():
    V, M, d = 8, 5, 0.6
    torch.manual_seed(0)
    cache = SD.TargetCache(M, V, decay=d, vocab=V)  # k = V: no truncation
    pos = torch.arange(M)
    qs = [torch.softmax(torch.randn(M, V) * 2, -1) for _ in range(3)]
    for q in qs:
        cache.update(pos, q.log())
    want = qs[0]
    for q in qs[1:]:
        want = d * want + (1 - d) * q
    idx, p, _ = cache.read(pos)
    dense = torch.zeros(M, V).scatter(1, idx, p)
    assert torch.allclose(dense, want, atol=2e-3)  # fp16 storage


def test_merge_truncated_matches_dense_reference():
    V, M, k, d = 20, 6, 3, 0.5
    torch.manual_seed(1)
    cache = SD.TargetCache(M, k, decay=d, vocab=V)
    pos = torch.arange(M)
    q1, q2 = (torch.softmax(torch.randn(M, V) * 3, -1) for _ in range(2))
    cache.update(pos, q1.log())
    cache.update(pos, q2.log())
    # dense reference: t1 = top-k(q1) (zero elsewhere); w = d t1 + (1-d) q2 on supp(t1) U topk(q2); keep top-k
    t1 = torch.zeros(M, V).scatter(1, q1.topk(k, -1).indices, q1.topk(k, -1).values)
    w = d * t1 + (1 - d) * q2
    cand = torch.zeros(M, V, dtype=torch.bool).scatter(1, q1.topk(k, -1).indices, True)
    cand.scatter_(1, q2.topk(k, -1).indices, True)
    w = torch.where(cand, w, torch.zeros(()))
    ref = torch.zeros(M, V).scatter(1, w.topk(k, -1).indices, w.topk(k, -1).values)
    idx, p, _ = cache.read(pos)
    got = torch.zeros(M, V).scatter(1, idx, p)
    assert torch.allclose(got, ref, atol=2e-3)


def test_decay_one_freezes_first_visit():
    V, M, k = 10, 4, 3
    cache = SD.TargetCache(M, k, decay=1.0, vocab=V)
    pos = torch.arange(M)
    q1 = torch.softmax(torch.randn(M, V), -1)
    cache.update(pos, q1.log())
    a = cache.read(pos)
    cache.update(pos, torch.softmax(torch.randn(M, V) * 5, -1).log())
    b = cache.read(pos)
    ga = torch.zeros(M, V).scatter(1, a[0], a[1])
    gb = torch.zeros(M, V).scatter(1, b[0], b[1])
    assert torch.equal(ga, gb)


def _dense_binned_kl(t_dense_bins, q, idx):
    qb = torch.cat([q.gather(1, idx), 1 - q.gather(1, idx).sum(-1, keepdim=True)], 1)
    return (torch.special.xlogy(t_dense_bins, t_dense_bins) - torch.special.xlogy(t_dense_bins, qb)).sum(-1)


def test_binned_kl_toy_targets():
    V, M = 12, 7
    torch.manual_seed(2)
    # 1) k = V: equals the full KL
    t = torch.softmax(torch.randn(M, V), -1)
    z = torch.randn(M, V, requires_grad=True)
    lq = F.log_softmax(z, -1)
    idx = torch.arange(V).expand(M, V)
    full = (t * (t.log() - lq)).sum(-1)
    assert torch.allclose(SD.binned_kl(lq, idx, t), full, atol=1e-5)
    # 2) k < V: equals the KL of the coarsened distributions (k tokens + one tail bin), gradients included
    k = 4
    tk = t.topk(k, -1)
    got = SD.binned_kl(lq, tk.indices, tk.values)
    bins = torch.cat([tk.values, 1 - tk.values.sum(-1, keepdim=True)], 1)
    ref = _dense_binned_kl(bins, lq.exp(), tk.indices)
    assert torch.allclose(got, ref, atol=1e-5)
    g1 = torch.autograd.grad(got.sum(), z, retain_graph=True)[0]
    g2 = torch.autograd.grad(ref.sum(), z)[0]
    assert torch.allclose(g1, g2, atol=1e-5)
    # 3) target = the model's own coarsened prediction -> KL 0 and zero gradient
    lq = F.log_softmax(z, -1)
    q = lq.exp().detach()
    qk = q.topk(k, -1)
    kl0 = SD.binned_kl(lq, qk.indices, qk.values)
    assert kl0.abs().max() < 1e-5
    assert torch.autograd.grad(kl0.sum(), z)[0].abs().max() < 1e-5
    # 4) peaked toy target: KL pushes mass toward the target token
    tt = torch.zeros(M, 1) + 0.9
    ti = torch.full((M, 1), 3, dtype=torch.long)
    gz = torch.autograd.grad(SD.binned_kl(F.log_softmax(z, -1), ti, tt).sum(), z)[0]
    assert (gz[:, 3] < 0).all()


def test_mixed_entropy_matches_dense():
    V, M, k, a = 15, 9, 4, 0.7
    torch.manual_seed(3)
    q = torch.softmax(torch.randn(M, V) * 2, -1)
    tk = q.topk(k, -1)
    y = torch.randint(0, V, (M,))
    y[:3] = tk.indices[:3, 0]  # some labels inside the support
    tail = 1 - tk.values.sum(-1, keepdim=True)
    t_dense = (tail / (V - k)).expand(M, V).clone().scatter(1, tk.indices, tk.values)
    m = (F.one_hot(y, V).float() + a * t_dense) / (1 + a)
    want = -(torch.special.xlogy(m, m)).sum(-1)
    got = SD.mixed_entropy(y, tk.indices, tk.values, a, V)
    assert torch.allclose(got.float(), want, atol=1e-5)


def test_label_smoothing_entropy_inverse():
    V = 16384
    for eps in (0.0, 0.01, 0.1, 0.4):
        H = SD.ls_entropy(eps, V)
        assert SD.ls_eps_for_entropy(H, V) == pytest.approx(eps, abs=1e-9)
    # the closed form equals the entropy of F.cross_entropy's smoothing target
    eps, V2 = 0.2, 10
    t = torch.full((V2,), eps / V2)
    t[0] += 1 - eps
    assert SD.ls_entropy(eps, V2) == pytest.approx(float(-(t * t.log()).sum()), rel=1e-6)


def test_unigram_smoothing_entropy():
    rng = np.random.default_rng(0)
    toks = rng.zipf(1.3, 5000) % 30
    u = SD.unigram(toks, 32)
    assert u.sum() == pytest.approx(1.0) and u[30:].sum() == 0
    eps = 0.3
    want = 0.0
    for y in range(32):
        m = eps * u.copy()
        m[y] += 1 - eps
        want += u[y] * -(m[m > 0] * np.log(m[m > 0])).sum()
    assert SD.unigram_ls_entropy(eps, u) == pytest.approx(want, rel=1e-9)
    assert SD.unigram_eps_for_entropy(want, u) == pytest.approx(eps, abs=1e-6)


# ------------------------------------------------------------------------------------------------
# harness wiring
# ------------------------------------------------------------------------------------------------
COMMON = ("--epochs", "3", "--no_full_dev", "--no_pt", "--seed", "1")


def test_off_and_alpha_zero_equal_port(synth_dir, tmp_path):
    """Off: the port's loop (the existing harness tests also run that path). temporal with alpha 0 and smooth with
    alpha 0, eps 0 go through the SD forward (hidden + logits + the same F.cross_entropy) and must reproduce the
    port's losses exactly: the plumbing adds nothing to the gradient."""
    base = ("--steps", "10", *COMMON[2:], "--epochs", "3")
    a = TR.main(tiny_args(synth_dir, tmp_path / "port", *base))
    b = TR.main(tiny_args(synth_dir, tmp_path / "off", *base, "--sd_mode", "none"))
    c = TR.main(tiny_args(synth_dir, tmp_path / "t0", *base, "--sd_mode", "temporal", "--sd_alpha", "0",
                          "--sd_start_epoch", "0"))
    d = TR.main(tiny_args(synth_dir, tmp_path / "s0", *base, "--sd_mode", "smooth", "--sd_alpha", "0",
                          "--sd_smooth_eps", "0", "--sd_start_epoch", "0"))
    la = [r["loss"] for r in _train_recs(tmp_path / "port")]
    assert la == [r["loss"] for r in _train_recs(tmp_path / "off")]
    assert la == [r["loss"] for r in _train_recs(tmp_path / "t0")]
    assert la == [r["loss"] for r in _train_recs(tmp_path / "s0")]
    TR.main(tiny_args(synth_dir, tmp_path / "u0", *base, "--sd_mode", "smooth_unigram", "--sd_alpha", "0",
                      "--sd_smooth_eps", "0", "--sd_start_epoch", "0"))
    assert la == [r["loss"] for r in _train_recs(tmp_path / "u0")]
    assert "self_distill" not in a["config"]["levers"] and "self_distill" not in b["config"]["levers"]
    assert c["config"]["levers"]["self_distill"]["mode"] == "temporal"
    assert d["final_train_loss"] == a["final_train_loss"]


def test_temporal_runs_switches_on_and_resumes_exactly(synth_dir, tmp_path):
    args = (*COMMON, "--sd_mode", "temporal", "--sd_alpha", "0.5", "--sd_k", "8", "--sd_start_epoch", "1",
            "--milestone_scale", "0.002", "--save")
    r = TR.main(tiny_args(synth_dir, tmp_path / "a", *args))
    assert r["status"] == "ok"
    recs = _train_recs(tmp_path / "a")
    ep = np.array([x["epoch"] for x in recs])
    on = np.array([x["sd_on"] for x in recs])
    assert on[0] == 0 and on[-1] == 1 and set(on) == {0.0, 1.0}
    # first epoch: nothing cached yet at the start; after one epoch almost every target has a cached target
    assert recs[0]["sd_cov"] == 0
    later = [x for x, e in zip(recs, ep) if e > 1.2]
    assert later and min(x["sd_cov"] for x in later) > 0.95
    assert all(math.isfinite(x["sd_kl"]) and x["sd_kl"] >= 0 for x in recs if x["sd_on"] == 1)
    assert all(0 < x["sd_H"] < math.log(16384) for x in later)
    assert all(0 < x["sd_mass"] <= 1.001 for x in later)
    sd_cfg = r["config"]["levers"]["self_distill"]
    assert sd_cfg["cache_bytes"] == C.load_split("train_10M", synth_dir).n_tokens * (8 * 4 + 1)
    # pause + resume reproduces the uninterrupted run bit for bit (the cache travels in sd_cache.pt)
    TR.main(tiny_args(synth_dir, tmp_path / "b", *args, "--max_session_steps", "17"))
    assert (tmp_path / "b" / "sd_cache.pt").exists()
    TR.main(tiny_args(synth_dir, tmp_path / "b", *args, "--resume", "auto"))
    assert [x["loss"] for x in recs] == [x["loss"] for x in _train_recs(tmp_path / "b")]
    assert [x.get("sd_kl") for x in recs] == [x.get("sd_kl") for x in _train_recs(tmp_path / "b")]


def test_temporal_changes_training_and_null_twin_calibrates(synth_dir, tmp_path):
    args = (*COMMON, "--sd_alpha", "1.0", "--sd_start_epoch", "1")
    TR.main(tiny_args(synth_dir, tmp_path / "port", *COMMON))
    TR.main(tiny_args(synth_dir, tmp_path / "t", *args, "--sd_mode", "temporal"))
    lp = [x["loss"] for x in _train_recs(tmp_path / "port")]
    lt = _train_recs(tmp_path / "t")
    first_on = next(i for i, x in enumerate(lt) if x["sd_on"] == 1)
    assert [x["loss"] for x in lt[:first_on]] == lp[:first_on]  # identical until the KL switches on
    assert [x["loss"] for x in lt[first_on + 1:]] != lp[first_on + 1:]
    cal = SD.null_eps_from_run(tmp_path / "t")
    assert 0 < cal["eps"] < 1 and cal["vocab"] == 16384
    assert SD.ls_entropy(cal["eps"], 16384) == pytest.approx(cal["mean_H"], rel=1e-6)
    r = TR.main(tiny_args(synth_dir, tmp_path / "s", *args, "--sd_mode", "smooth", "--sd_smooth_eps",
                          f"{cal['eps']:.6f}"))
    assert r["status"] == "ok" and r["config"]["levers"]["self_distill"]["smooth_eps"] == pytest.approx(cal["eps"])
    co = SD.null_eps_from_run(tmp_path / "t", match="off")
    assert co["eps"] * (1 - 1 / 16384) == pytest.approx(co["mean_off"], rel=1e-9)
    assert all(x["sd_off"] == pytest.approx(1.0 * (1 - x["sd_t_label"]) / 2.0) for x in lt if "sd_off" in x)
    cu = SD.null_eps_from_run(tmp_path / "t", null="unigram")
    u = SD.unigram(C.load_split("train_10M", synth_dir).tokens, 16384)
    assert cu["mode"] == "smooth_unigram" and SD.unigram_ls_entropy(cu["eps"], u) == pytest.approx(cu["mean_H"], rel=1e-5)
    r = TR.main(tiny_args(synth_dir, tmp_path / "u", *args, "--sd_mode", "smooth_unigram", "--sd_smooth_eps",
                          f"{cu['eps']:.6f}"))
    lu = [x["loss"] for x in _train_recs(tmp_path / "u")]
    assert r["status"] == "ok" and lu[:first_on] == lp[:first_on] and lu[first_on + 1:] != lp[first_on + 1:]


def test_temporal_with_mntp_and_line_windows(synth_dir, tmp_path):
    r = TR.main(tiny_args(synth_dir, tmp_path / "m", "--epochs", "2", "--no_full_dev", "--no_pt",
                          "--window_plan", "line:32,line:64", "--mntp_frac", "0.25", "--sd_mode", "temporal",
                          "--sd_start_epoch", "1"))
    assert r["status"] == "ok"
    recs = _train_recs(tmp_path / "m")
    assert all(math.isfinite(x["mntp_loss"]) for x in recs)
    on = [x for x in recs if x["sd_on"] == 1]
    assert on and all(math.isfinite(x["sd_kl"]) for x in on if "sd_kl" in x)
    # MNTP rows are never cached: coverage in epoch 2 stays below 1 (their epoch-1 positions have no target)
    assert np.mean([x["sd_cov"] for x in on]) < 0.95



def test_write_until_freezes_the_cache(synth_dir, tmp_path):
    r = TR.main(tiny_args(synth_dir, tmp_path / "w", *COMMON, "--sd_mode", "temporal", "--sd_start_epoch", "1",
                          "--sd_write_until", "2"))
    assert r["config"]["levers"]["self_distill"]["write_until"] == 2.0
    recs = _train_recs(tmp_path / "w")
    ep_start = np.array([0.0] + [x["epoch"] for x in recs[:-1]])  # the epoch at the start of a step decides
    w = np.array([x["sd_write"] for x in recs])
    assert np.array_equal(w, (ep_start < 2).astype(float))
    assert w[0] == 1 and w[-1] == 0


def test_a1_job_generator_keeps_the_base_argv(tmp_path):
    from babylm.configs import make_a1_jobs as MJ
    base = {"name": "b", "module": "babylm.harness.train", "timeout_s": 100,
            "args": ["--size", "S", "--seed", "0", "--muon_wd", "2.0", "--compile", "--tag", "b", "--save"]}
    jobs = MJ.stage1(base, "p0_")
    assert [j["name"] for j in jobs] == ["p0_sd_base", "p0_A1a", "p0_A1b", "p0_A1c", "p0_A1d", "p0_A1e"]
    for j in jobs:
        core = [a for a in j["args"] if not a.startswith("--sd_")]
        assert j["args"][:7] == ["--size", "S", "--seed", "0", "--muon_wd", "2.0", "--compile"]
        assert j["args"][-2:] == ["--tag", j["name"]] and j["args"].count("--tag") == 1 and "--save" in core
    with pytest.raises(SystemExit):
        MJ.derive({**base, "args": base["args"] + ["--sd_mode", "temporal"]}, "x", [])
