"""CPU tests for attack A2, input-noise augmentation (babylm.harness.input_noise) and its train.py wiring."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from babylm.data import corpus as C
from babylm.harness import input_noise as IN
from babylm.harness import train as TR
from babylm.tests.conftest import tiny_args
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

V = 64  # toy vocab for the unit tests; id 0 = BOS


def rows(B=8, T=32, seed=0, pad_rows=()):
    """Stream-style rows (x[j+1] == y[j]) with some BOS inside, optionally line-style rows padded with -1."""
    g = torch.Generator().manual_seed(seed)
    seg = torch.randint(1, V, (B, T + 1), generator=g)
    seg[seg % 13 == 0] = 0
    x, y = seg[:, :-1].clone(), seg[:, 1:].clone()
    for r in pad_rows:  # line window: [BOS, t..], y padded after n targets
        n = T // 2
        x[r, 0] = 0
        x[r, 1:n] = y[r, :n - 1]
        x[r, n:] = 0
        y[r, n:] = -1
    return x, y


def test_replace_keeps_targets_and_protects_bos_and_pad():
    x, y = rows(B=64, T=64, pad_rows=(0, 3))
    nz = IN.InputNoise("replace", 0.3, vocab_real=V, seed=1)
    xn, yn = nz(x, y)
    assert torch.equal(yn, y)  # targets are clean
    changed = xn != x
    assert not changed[x == 0].any()          # BOS never corrupted
    assert not changed[y < 0].any()           # pad never touched
    assert not (xn[changed] == 0).any()       # BOS never inserted
    assert int(xn.max()) < V
    st = nz.pop_stats()
    assert st["noise_frac"] == pytest.approx(0.3, abs=0.03)
    assert changed.float().sum() / ((y >= 0) & (x != 0)).sum() <= st["noise_frac"] + 1e-9  # same-id draws
    assert nz.pop_stats()["noise_frac"] == 0.0


def test_unigram_replacement_follows_counts():
    counts = np.zeros(V)
    counts[[5, 6, 7]] = [1, 2, 7]
    counts[0] = 1000  # BOS is excluded whatever its count
    nz = IN.InputNoise("replace", 0.5, "unigram", vocab_real=V, counts=counts, seed=2)
    toks = nz.sample_tokens(20000)
    assert set(toks.tolist()) <= {5, 6, 7}
    assert (toks == 7).float().mean() == pytest.approx(0.7, abs=0.02)
    with pytest.raises(ValueError):
        IN.InputNoise("replace", 0.1, "unigram", vocab_real=V)


def test_mask_mode_uses_mask_id_only():
    x, y = rows(B=32)
    nz = IN.InputNoise("mask", 0.2, vocab_real=V, mask_id=99, r2l_id=100, seed=3)
    xn, yn = nz(x, y)
    assert torch.equal(yn, y) and set(xn[xn != x].tolist()) == {99}


def test_r2l_reverses_with_same_targets():
    x, y = rows(B=200, T=16, pad_rows=(1, 2, 5))
    nz = IN.InputNoise("none", 0.0, r2l=0.5, vocab_real=V, r2l_id=100, seed=4)
    xn, yn = nz(x, y)
    rev = xn[:, 0] == 100
    assert 70 < int(rev.sum()) < 130 and nz.pop_stats()["noise_r2l_frac"] == pytest.approx(rev.float().mean())
    for r in range(len(x)):
        n = int((y[r] >= 0).sum())
        assert sorted(yn[r, :n].tolist()) == sorted(y[r, :n].tolist())  # same target tokens
        assert torch.equal(yn[r, n:], y[r, n:]) and torch.equal(xn[r, n:], x[r, n:])
        if rev[r]:
            assert torch.equal(yn[r, :n], y[r, :n].flip(0))
            assert torch.equal(xn[r, 1:n], yn[r, :n - 1])  # x'[j+1] == y'[j]: a causal R2L row
        else:
            assert torch.equal(xn[r], x[r]) and torch.equal(yn[r], y[r])


def test_r2l_then_replace_protects_marker():
    x, y = rows(B=64)
    nz = IN.InputNoise("replace", 0.5, r2l=1.0, vocab_real=V, r2l_id=100, seed=5)
    xn, yn = nz(x, y)
    assert (xn[:, 0] == 100).all() and int(xn[:, 1:].max()) < V


def test_generator_is_private_and_resumable():
    x, y = rows()
    torch.manual_seed(0)
    before = torch.get_rng_state()
    a = IN.InputNoise("replace", 0.2, r2l=0.3, vocab_real=V, r2l_id=100, seed=6)
    a(x, y)
    assert torch.equal(torch.get_rng_state(), before)  # global RNG untouched
    sd = a.state_dict()
    out1 = a(x, y)
    b = IN.InputNoise("replace", 0.2, r2l=0.3, vocab_real=V, r2l_id=100, seed=999)
    b.load_state_dict(sd)
    out2 = b(x, y)
    assert torch.equal(out1[0], out2[0]) and torch.equal(out1[1], out2[1])
    c = IN.InputNoise("replace", 0.1, vocab_real=V, seed=6)
    with pytest.raises(ValueError):
        c.load_state_dict(sd)


def test_gauss_energy_matches_replacement():
    """E||delta e||^2 of the gauss null equals that of uniform replacement at the same rate (raw embedding)."""
    torch.manual_seed(0)
    emb = torch.nn.Embedding(V + 4, 16)
    nz = IN.InputNoise("gauss", 0.15, vocab_real=V, seed=7)
    nz.attach_gauss(emb)
    ids = torch.randint(1, V, (400, 50))
    ids[:, 0] = 0
    with torch.no_grad():
        clean = emb.weight[ids]
        d_g = ((emb(ids) - clean) ** 2).sum(-1)
        rep = IN.InputNoise("replace", 0.15, vocab_real=V, seed=8)
        xr, _ = rep(ids, torch.zeros_like(ids))
        d_r = ((emb.weight[xr] - clean) ** 2).sum(-1)
        emb.eval()
        assert torch.equal(emb(ids), clean)  # no noise in eval
        emb.train()
        nz.hook_on = False
        assert torch.equal(emb(ids), clean)  # no noise on MNTP rows
    assert (d_g[:, 0] == 0).all()  # BOS untouched
    assert float(d_g[:, 1:].mean()) == pytest.approx(float(d_r[:, 1:].mean()), rel=0.05)


def test_flag_validation():
    with pytest.raises(ValueError):
        IN.InputNoise("none", 0.1)
    with pytest.raises(ValueError):
        IN.InputNoise("replace", 1.0)
    assert IN.needs_pad("mask", 0) and IN.needs_pad("replace", 0.5) and not IN.needs_pad("replace", 0)
    assert not IN.needs_pad("gauss", 0)


# ------------------------------------------------------------------------------------------------ harness wiring
def read(out, name):
    return list(csv.DictReader((out / name).open()))


COMMON = ("--epochs", "2", "--steps", "10", "--no_pt", "--milestone_scale", "0.002", "--seed", "1")


def test_off_and_p0_reproduce_the_port_bit_for_bit(synth_dir, tmp_path):
    """Default flags build no noise object; replace at p = 0 (object built, wired, never fires) and gauss at p = 0
    reproduce the port's losses and DEV numbers exactly, so the wiring itself perturbs nothing."""
    a = TR.main(tiny_args(synth_dir, tmp_path / "a", *COMMON))
    assert "noise" not in a["config"]
    for name, extra in (("b", ("--noise_mode", "replace", "--noise_p", "0")),
                        ("c", ("--noise_mode", "gauss", "--noise_p", "0"))):
        b = TR.main(tiny_args(synth_dir, tmp_path / name, *COMMON, *extra))
        assert b["config"]["noise"]["mode"] == extra[1] and b["config"]["model_cfg"]["vocab_size"] == C.VOCAB_SIZE
        assert [r["loss"] for r in read(tmp_path / "a", "train_log.csv")] == \
               [r["loss"] for r in read(tmp_path / name, "train_log.csv")]
        assert a["full_eval"]["nats"] == b["full_eval"]["nats"]


@pytest.mark.parametrize("extra,vocab", [
    (("--noise_mode", "replace", "--noise_p", "0.15"), 16384),
    (("--noise_mode", "replace", "--noise_p", "0.15", "--noise_dist", "unigram"), 16384),
    (("--noise_mode", "mask", "--noise_p", "0.15"), 16512),
    (("--noise_mode", "gauss", "--noise_p", "0.15"), 16384),
    (("--noise_mode", "replace", "--noise_p", "0.05", "--noise_r2l", "0.5"), 16512),
])
def test_modes_run_change_training_and_keep_exposure(synth_dir, tmp_path, extra, vocab):
    a = TR.main(tiny_args(synth_dir, tmp_path / "a", *COMMON))
    b = TR.main(tiny_args(synth_dir, tmp_path / "b", *COMMON, *extra))
    assert b["status"] == "ok" and b["config"]["model_cfg"]["vocab_size"] == vocab
    assert b["words_seen"] == a["words_seen"] and b["tokens_seen"] == a["tokens_seen"]  # exposure unchanged
    la = [float(r["loss"]) for r in read(tmp_path / "a", "train_log.csv")]
    lb = [float(r["loss"]) for r in read(tmp_path / "b", "train_log.csv")]
    assert all(np.isfinite(lb)) and la != lb
    tr = [json.loads(x) for x in (tmp_path / "b" / "log.jsonl").read_text().splitlines()]
    tr = [x for x in tr if x.get("type") == "train"]
    if extra[1] in ("replace", "mask"):
        assert np.mean([x["noise_frac"] for x in tr]) == pytest.approx(float(extra[3]), abs=0.03)
    if "--noise_r2l" in extra:
        assert 0.2 < np.mean([x["noise_r2l_frac"] for x in tr]) < 0.8


def test_noise_resume_is_bit_exact(synth_dir, tmp_path):
    base = ["--steps", "16", "--milestone_scale", "0.002", "--no_full_dev", "--no_pt", "--seed", "1",
            "--noise_mode", "replace", "--noise_p", "0.2", "--noise_r2l", "0.3"]
    a = TR.main(tiny_args(synth_dir, tmp_path / "a", *base))
    TR.main(tiny_args(synth_dir, tmp_path / "b", *base, "--max_session_steps", "5", "--save"))
    b = TR.main(tiny_args(synth_dir, tmp_path / "b", *base, "--resume", "auto", "--save"))
    assert b["sessions"] == 2
    assert [r["loss"] for r in read(tmp_path / "a", "train_log.csv")] == \
           [r["loss"] for r in read(tmp_path / "b", "train_log.csv")]


def test_noise_with_mntp_and_line_windows(synth_dir, tmp_path):
    r = TR.main(tiny_args(synth_dir, tmp_path / "m", "--epochs", "2", "--steps", "6", "--no_pt", "--no_full_dev",
                          "--milestone_scale", "0.002", "--window_plan", "line:32,line:64", "--mntp_frac", "0.25",
                          "--noise_mode", "gauss", "--noise_p", "0.15", "--noise_r2l", "0.25"))
    assert r["status"] == "ok" and r["config"]["model_cfg"]["vocab_size"] == 16512


def test_gauss_hook_on_real_model_only_in_training():
    cfg = ModelConfig(vocab_size=16384, n_layer=2, n_head=2, n_embd=64, seq_len=32)
    torch.manual_seed(0)
    m = baseline.build(cfg)
    nz = IN.InputNoise("gauss", 0.15, vocab_real=16384, seed=0)
    nz.attach_gauss(m.embed)
    x = torch.randint(1, 16384, (2, 32))
    m.eval()
    with torch.no_grad():
        h0 = m.hidden(x)
        m.train()
        h1 = m.hidden(x)
    assert not torch.equal(h0, h1)


def test_gauss_hook_survives_compile_cache(synth_dir, tmp_path):
    """Dynamo skips hook guards: a compiled base run in the same process must not hide the gauss hook."""
    common = ("--epochs", "1", "--steps", "3", "--no_pt", "--no_full_dev", "--milestone_scale", "0.002", "--compile")
    a = TR.main(tiny_args(synth_dir, tmp_path / "a", *common))
    b = TR.main(tiny_args(synth_dir, tmp_path / "b", *common, "--noise_mode", "gauss", "--noise_p", "0.15"))
    assert a["final_train_loss"] != b["final_train_loss"]


def test_dev_bpw_vocab_mask():
    """dev_bpw --vocab_mask N renormalises over ids < N; off (0) is the registered protocol, unchanged."""
    import importlib.util
    from pathlib import Path as _P
    pytest.importorskip("transformers")
    spec = importlib.util.spec_from_file_location("dev_bpw", _P(C.REPO_ROOT) / "babylm" / "eval" / "dev_bpw.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    g = torch.Generator().manual_seed(0)
    W = torch.randn(12, 12, generator=g)

    def model(input_ids, attention_mask):
        return {"logits": W[input_ids]}
    ids = torch.randint(1, 8, (40,), generator=g).tolist()
    a = D.nll_of(model, ids, 0, "cpu", CTX=16, STRIDE=8)
    b = D.nll_of(model, ids, 0, "cpu", CTX=16, STRIDE=8, vocab_mask=8)
    lp = torch.log_softmax(W, -1)
    mass = torch.logsumexp(lp[:, 8:], -1).exp()
    assert b < a and a - b == pytest.approx(-float(torch.log1p(-mass[[0] + ids[:-1]]).sum()), rel=1e-4)
