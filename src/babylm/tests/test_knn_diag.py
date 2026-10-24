"""kNN-LM diagnostic (babylm.analysis.knn_diag): window protocol, exact kNN against brute force, p_knn and the
interpolation (lam = 0 is the LM bit for bit), hooked keys = head inputs, and end-to-end runs on the synthetic data
dir with a random tiny model (CPU, seconds)."""
from __future__ import annotations

import contextlib
import csv
import math

import numpy as np
import pytest
import torch

from babylm.analysis import knn_diag as K
from babylm.data import corpus as C
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

CPU = torch.device("cpu")


@pytest.mark.parametrize("n_pos,ctx", [(1, 8), (5, 8), (8, 8), (9, 8), (12, 8), (13, 8), (100, 8), (1000, 64),
                                       (1001, 63), (7, 2), (50, 3)])
def test_windows_score_each_position_once(n_pos, ctx):
    plan = K.window_plan(n_pos, ctx)
    s = ctx // 2
    seen = np.zeros(n_pos, dtype=np.int64)
    for j, (lo, hi, s0) in enumerate(plan):
        assert lo == j * s and hi == min(lo + ctx, n_pos) and lo <= s0 < hi
        assert s0 == (lo if j == 0 else lo + ctx - s)  # later windows: >= ctx - s tokens of context
        seen[s0:hi] += 1
    assert (seen == 1).all() and plan[-1][1] == n_pos


def test_doc_prefix_and_target_docs():
    toks = np.array([0, 5, 6, 0, 7, 0, 8, 9, 9, 0, 4], dtype=np.uint16)  # documents open at 0, 3, 5, 9
    assert K.doc_prefix(toks, None) == 11 and K.doc_prefix(toks, 100) == 11
    assert K.doc_prefix(toks, 6) == 5 and K.doc_prefix(toks, 9) == 9 and K.doc_prefix(toks, 2) == 3
    assert K.doc_prefix(toks, 2, min_docs=2) == 5
    assert K.target_docs(toks, 11).tolist() == [0, 0, 1, 1, 2, 2, 2, 2, 3, 3]  # targets 1..10; a BOS opens its doc
    assert K.target_docs(toks, 5).tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("q_chunk,k_chunk,dtype", [(16, 128, torch.float16), (7, 40, torch.float16),
                                                   (64, 5000, torch.bfloat16), (10, 300, torch.float32)])
def test_exact_knn_matches_brute_force(q_chunk, k_chunk, dtype):
    g = torch.Generator().manual_seed(0)
    d, Nk, Nq, k, tol = 24, 1000, 37, 50, 1e-3
    keys = torch.randn(Nk, d, generator=g).to(dtype)
    qs = torch.randn(Nq, d, generator=g).to(dtype)
    ref = torch.cdist(qs.double(), keys.double()) ** 2
    rv, ri = ref.topk(k, dim=1, largest=False)
    got = list(K.knn_search(qs, keys, k, q_chunk, k_chunk))
    assert [(a, b) for a, b, _, _ in got] == [(i, min(Nq, i + q_chunk)) for i in range(0, Nq, q_chunk)]
    dist, idx = torch.cat([x[2] for x in got]), torch.cat([x[3] for x in got])
    assert dist.shape == idx.shape == (Nq, k) and (dist[:, 1:] >= dist[:, :-1]).all()
    assert torch.allclose(dist.double(), rv, atol=tol, rtol=1e-6)
    assert torch.allclose(ref.gather(1, idx), dist.double(), atol=tol, rtol=1e-6)  # each distance is its key's
    assert all(len(set(r.tolist())) == k for r in idx)
    clear = (rv[:, 1:] - rv[:, :-1]).min(1).values > tol  # rows without near-ties: identical index lists
    assert clear.sum() > Nq // 3 and torch.equal(idx[clear], ri[clear])


def test_knn_logp_and_interpolation():
    dist = torch.tensor([[0.0, 1.0, 2.0, 5.0]])
    vals = torch.tensor([[5, 7, 5, 9]], dtype=torch.int32)
    lp = K.knn_logp(dist, vals, torch.tensor([5], dtype=torch.int32), [1.0, 2.0], [2, 4])
    for a, T in enumerate([1.0, 2.0]):
        for b, kp in enumerate([2, 4]):
            w = torch.exp(-dist[0, :kp] / T)
            want = float(w[vals[0, :kp] == 5].sum() / w.sum())
            assert math.isclose(float(lp[a, b, 0].exp()), want, rel_tol=1e-6)
    assert torch.isneginf(K.knn_logp(dist, vals, torch.tensor([3], dtype=torch.int32), [1.0], [4])).all()
    g = torch.Generator().manual_seed(1)
    lm = -torch.rand(1000, generator=g) * 12
    lpk = torch.log(torch.rand(1000, generator=g))
    lpk[::7] = float("-inf")  # no neighbour carries the target
    assert torch.equal(K.interp_nll(lpk, lm, 0.0), -lm)  # lam = 0 reproduces the LM exactly
    for lam in (0.05, 0.3, 0.5):
        want = -torch.log(lam * lpk.double().exp() + (1 - lam) * lm.double().exp())
        assert torch.allclose(K.interp_nll(lpk, lm, lam).double(), want, rtol=1e-5, atol=1e-6)


def tiny_model(seed: int = 0, n_embd: int = 32, seq_len: int = 32) -> torch.nn.Module:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=C.VOCAB_SIZE, seq_len=seq_len, n_layer=2, n_head=2, n_embd=n_embd,
                      extra={"softcap": 15.0, "unet": True})
    m = baseline.build(cfg)
    with torch.no_grad():  # the zero-initialised output projections would make every hidden state context-free
        for p in m.parameters():
            if p.ndim == 2 and not p.abs().any():
                p.normal_(0.0, 0.05)
    return m.eval()


def test_encode_stream_keys_are_head_inputs():
    m = tiny_model()
    tap = K.HeadTap(m.lm_head)
    rng = np.random.default_rng(0)
    toks = rng.integers(1, 500, size=100).astype(np.uint16)
    toks[[0, 40]] = C.BOS_ID
    ctx, taus = 16, [0.5, 1.0, 2.0]
    keys, lm, lmt, check = K.encode_stream(m, tap, toks, ctx, 3, CPU, contextlib.nullcontext(), torch.float32, CPU,
                                           "t", lm_temps=taus)
    tap.handle.remove()
    assert keys.shape == (99, 32) and lm.shape == (99,) and lmt.shape == (3, 99) and check < 1e-4
    assert np.allclose(lmt[1], lm, atol=1e-5)  # tau = 1 is the model's own log-prob
    for lo, hi, s0 in K.window_plan(99, ctx):
        x = torch.from_numpy(toks[lo:hi].astype(np.int64))[None]
        y = torch.from_numpy(toks[lo + 1:hi + 1].astype(np.int64))[None]
        with torch.no_grad():
            h = m.hidden(x)[0]
            _, aux = m(x, y)
            z = m.logits(h)
        assert torch.allclose(keys[s0:hi], h[s0 - lo:hi - lo], atol=1e-5)
        assert np.allclose(lm[s0:hi], -aux["per_token_loss"][0, s0 - lo:hi - lo].numpy(), atol=1e-5)
        for ti, tau in enumerate(taus):
            ref = torch.log_softmax(z / tau, -1).gather(1, y[0][:, None])[:, 0]
            assert np.allclose(lmt[ti, s0:hi], ref[s0 - lo:hi - lo].numpy(), atol=1e-5)
    # the datastore pass needs neither LM column
    tap = K.HeadTap(m.lm_head)
    keys2, lm2, lmt2, check2 = K.encode_stream(m, tap, toks, ctx, 4, CPU, contextlib.nullcontext(), torch.float16,
                                               CPU, "t", want_lm=False)
    assert lm2 is None and lmt2 is None and check2 is None and keys2.dtype == torch.float16
    assert torch.allclose(keys2.float(), keys, atol=1e-2)


def save_ckpt(m: torch.nn.Module, p) -> None:
    """The harness's milestone checkpoint format (babylm.harness.train)."""
    torch.save({"model": {k: v.detach().clone() for k, v in m.state_dict().items()}, "cfg": m.cfg.to_dict(),
                "model_name": "baseline", "words_seen": 0, "words_nominal": None, "tokens_seen": 0, "step": 0,
                "label": "chck_0M", "tokenizer_sha256": C.sha256_file(C.TOKENIZER_PATH), "git_commit": "test"}, p)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    p = tmp_path_factory.mktemp("knn_ckpt") / "chck_0M.pt"
    save_ckpt(tiny_model(), p)
    return p


def run(ckpt, data_dir, out, *extra):
    return K.main(["--ckpt", str(ckpt), "--out", str(out), "--data_dir", str(data_dir), "--ctx", "32", "--k", "16",
                   "--device", "cpu", "--batch", "8", *extra])


def test_end_to_end(ckpt, synth_dir, tmp_path):
    out = tmp_path / "o"
    s = run(ckpt, synth_dir, out)
    ev, tr = C.load_split("dev_sub", synth_dir), C.load_split("train_10M", synth_dir)
    assert s["datastore"]["keys"] == tr.n_tokens - 1 and s["eval"]["queries"] == ev.n_tokens - 1
    assert s["grid"]["ks"] == [1, 4, 16] and len(s["grid"]["lambdas"]) == 11
    labels = ("lm", "knn", "lmtemp", "knntemp")
    pts = {x: np.load(out / "pt" / f"pt_dev_{x}.npy") for x in labels}
    for pt in pts.values():
        assert pt.dtype == np.float16 and pt.shape == (ev.n_tokens,) and np.isnan(pt[0])
        assert not np.isnan(pt[1:]).any()
    pt_lm = pts["lm"]
    # LM column = the model's own forward on the first window
    m = tiny_model()
    x = torch.from_numpy(np.asarray(ev.tokens[:32], dtype=np.int64))[None]
    y = torch.from_numpy(np.asarray(ev.tokens[1:33], dtype=np.int64))[None]
    with torch.no_grad():
        ref = m(x, y)[1]["per_token_loss"][0].numpy()
    assert np.allclose(pt_lm[1:33].astype(np.float32), ref, atol=1e-2)
    for lab in labels:
        for suffix in ("", "_odd"):
            assert np.load(out / "pw" / f"pw_dev_{lab}{suffix}.npy").shape == (ev.n_words,)
    with open(out / "grid.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 * 3 * 3 * 11 and {r["base"] for r in rows} == {"lm", f"lm_tau{s['chosen']['tau']:g}"}
    held = s["held_out_odd_docs"]
    for r in rows:
        if float(r["lam"]) == 0.0 and r["base"] == "lm":  # lam = 0 reproduces the LM exactly
            assert float(r["odd_nats_per_token"]) == round(held["lm_nats_per_token"], 6)
    with open(out / "lm_temp.csv") as fh:
        trows = {float(r["tau"]): r for r in csv.DictReader(fh)}
    assert len(trows) == 15 and abs(float(trows[1.0]["odd_nats_per_token"]) - held["lm_nats_per_token"]) < 1e-5
    assert s["protocol"]["temp_check_max_abs_logp_diff"] < 1e-4
    hl = s["headline_held_out_nats_per_token"]
    assert math.isclose(hl["gain_knn"], hl["lm"] - hl["knn"])
    assert math.isclose(hl["gain_knn"], held["gain_nats_per_token"])
    assert math.isclose(hl["gain_temp_only"], hl["lm"] - hl["temp_only"])
    assert math.isclose(hl["gain_knn_beyond_temp"], hl["temp_only"] - hl["knn_over_temp"], abs_tol=1e-12)
    assert hl["gain_temp_only_oracle_same_tokens"] >= hl["gain_temp_only"] - 1e-12
    assert hl["gain_knn_oracle_same_tokens"] >= hl["gain_knn"] - 1e-12
    assert s["eval"]["targets"]["even"] + s["eval"]["targets"]["odd"] == s["eval"]["queries"]
    assert s["eval"]["words"]["even"] + s["eval"]["words"]["odd"] == ev.n_words
    assert set(s["slicer_cmd"]) == {"all_docs", "drop_overlap", "held_out_only"}


def test_self_retrieval_finds_itself(ckpt, synth_dir, tmp_path):
    """Datastore = the eval split: each query's own key (distance 0) carries its target, so keys, values and
    queries are aligned iff the top-1 neighbour is the target almost always and the kNN-LM gain is large."""
    s = run(ckpt, synth_dir, tmp_path / "o", "--train_split", "dev_sub", "--lm_temps", "1")
    nb = s["neighbours"]
    assert nb["top1_value_is_target"] > 0.99 and nb["frac_nn1_sqdist_lt_1e-3"] > 0.99
    assert s["held_out_odd_docs"]["gain_nats_per_token"] > 1.0


def test_temperature_control_catches_overconfidence(synth_dir, tmp_path):
    """An overconfident LM (head scaled x50: confident noise) gains from flattening alone; the control must pick
    tau > 1 on the even documents and show the held-out gain, so a kNN gain that is only recalibration shows up."""
    m = tiny_model()
    with torch.no_grad():
        m.lm_head.weight.mul_(50.0)
    save_ckpt(m, tmp_path / "sharp.pt")
    s = run(tmp_path / "sharp.pt", synth_dir, tmp_path / "o", "--lm_temps", "1,2,4,8", "--max_train_tokens", "3000")
    t, hl = s["held_out_odd_docs"]["temp_only"], s["headline_held_out_nats_per_token"]
    assert t["tau"] > 1 and t["gain_nats_per_token"] > 0.1 and hl["gain_temp_only"] == t["gain_nats_per_token"]
    assert hl["gain_knn_beyond_temp"] == s["held_out_odd_docs"]["knn_over_temp"]["gain_over_temp_nats_per_token"]


def test_truncation_keeps_whole_documents(ckpt, synth_dir, tmp_path):
    out = tmp_path / "o"
    ev = C.load_split("dev_sub", synth_dir)
    cut = int(ev.doc_starts()[3])
    s = run(ckpt, synth_dir, out, "--max_eval_tokens", str(cut + 5), "--max_train_tokens", "3000")
    assert s["eval"]["tokens_used"] == cut and s["eval"]["docs"] == 3 and s["eval"]["truncated"]
    assert s["datastore"]["tokens_used"] <= 3000 and s["datastore"]["truncated"]
    pt = np.load(out / "pt" / "pt_dev_knn.npy")
    assert np.isnan(pt[0]) and not np.isnan(pt[1:cut]).any() and np.isnan(pt[cut:]).all()
    w = np.asarray(ev.words)
    pw = np.load(out / "pw" / "pw_dev_knn.npy")
    assert np.isnan(pw[0]) and not np.isnan(pw[1:w[cut]]).any() and np.isnan(pw[w[cut]:]).all()
    odd = np.load(out / "pw" / "pw_dev_knn_odd.npy")  # held-out variant: only document 1's words
    assert np.array_equal(~np.isnan(odd), (np.arange(ev.n_words) >= w[ev.doc_starts()[1]])
                          & (np.arange(ev.n_words) < w[ev.doc_starts()[2]]))


def test_refuses_the_test_split(ckpt, synth_dir, tmp_path):
    with pytest.raises(SystemExit, match="protected FINAL"):
        run(ckpt, synth_dir, tmp_path / "o", "--eval_split", "test")
