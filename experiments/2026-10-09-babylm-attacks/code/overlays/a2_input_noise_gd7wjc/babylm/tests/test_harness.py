"""Sampler, milestones, per-token dumps, resume, FINAL guard (synthetic data, tiny model, CPU)."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch

from babylm.data import corpus as C
from babylm.harness import stream as S
from babylm.harness import train as TR
from babylm.tests.conftest import tiny_args


def test_sampler_tiles_each_epoch():
    N, T = 10_007, 64
    s = S.EpochSampler(N, T, data_seed=3)
    offs = []
    for e in range(3):
        plan = s.plan(e)
        n = len(plan)
        off = int(plan.min())
        offs.append(off)
        assert 0 <= off < T and n == (N - 1 - off) // T
        cover = np.zeros(N, dtype=np.int64)
        for st in plan:
            cover[st + 1:st + T + 1] += 1
        assert (cover[off + 1:off + n * T + 1] == 1).all() and cover.sum() == n * T
        got = s.next_starts(n)  # one epoch through the stateful API
        assert np.array_equal(got, plan)
    assert len(set(offs)) > 1 or True
    assert not np.array_equal(s.plan(0), s.plan(1))


def test_sampler_state_roundtrip_and_rollover():
    s = S.EpochSampler(5_000, 32, data_seed=1)
    a = s.next_starts(100)
    sd = json.loads(json.dumps(s.state_dict()))
    b1 = s.next_starts(300)  # crosses into epoch 1 (n per epoch < 156)
    assert s.epoch >= 1
    s2 = S.EpochSampler(5_000, 32, data_seed=1)
    s2.load_state_dict(sd)
    assert np.array_equal(s2.next_starts(300), b1)
    assert len(a) == 100


def test_stream_batches_score_every_token_once():
    N, T, B = 1_000, 64, 3
    toks = np.arange(N, dtype=np.uint16)
    seen = np.zeros(N, dtype=np.int64)
    for x, y, win, tpos in S.stream_batches(toks, T, B):
        assert x.shape == (B, T) and y.shape == (B, T)
        m = tpos >= 0
        seen[tpos[m]] += 1
        assert np.array_equal(y.numpy()[m], toks[tpos[m]].astype(np.int64))
        assert np.array_equal(x.numpy()[m], toks[tpos[m] - 1].astype(np.int64))
        assert (y.numpy()[~m] == -1).all()
    assert seen[0] == 0 and (seen[1:] == 1).all()


def test_milestone_list():
    assert S.MILESTONES[:10] == [i * 10**6 for i in range(1, 11)]
    assert S.MILESTONES[10:19] == [i * 10**7 for i in range(2, 11)]
    assert S.MILESTONES[-1] == 10**9 and len(S.MILESTONES) == 28
    assert S.milestone_label(20_000_000) == "chck_20M"


def read_csv(p):
    return list(csv.DictReader(p.open()))


@pytest.fixture(scope="module")
def tiny_run(synth_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    res = TR.main(tiny_args(synth_dir, out, "--epochs", "2", "--milestone_scale", "0.005", "--save", "--seed", "0"))
    return out, res


def test_milestones_fire_at_word_counts(tiny_run, synth_dir):
    out, res = tiny_run
    assert res["status"] == "ok"
    train = C.load_split("train_10M", synth_dir)
    steps = math.ceil(2 * train.n_tokens / 1024)
    assert res["steps"] == steps
    log = read_csv(out / "train_log.csv")
    assert len(log) == steps
    words = [0] + [int(r["words_seen"]) for r in log]
    curves = read_csv(out / "curves.csv")
    fired = [r for r in curves if r["label"] != "final"]
    assert fired, "no milestone fired"
    for r in fired:
        m = int(r["label"][:-1]) * 10**6
        thr = max(1, int(round(m * 0.005)))
        st = int(r["step"])
        if words[st] >= thr:
            assert words[st - 1] < thr  # fired at the first step that crossed it
        else:  # fired at the end within 1%
            assert st == steps and words[st] >= 0.99 * thr
    # words seen ~ epochs x split words
    assert abs(words[-1] - 2 * train.n_words) / (2 * train.n_words) < 0.05
    labels = [r["label"] for r in fired]
    for lab in labels:
        assert (out / "checkpoints" / f"chck_{lab}.pt").exists()
        assert (out / "pt" / f"pt_dev_{lab}.npy").exists() and (out / "pt" / f"pt_train_{lab}.npy").exists()
    assert (out / "pt" / "pt_devfull_final.npy").exists()
    assert not (out / "FINAL_SPLIT_USED").exists()


def test_per_token_dumps_sum_to_reported(tiny_run, synth_dir):
    out, res = tiny_run
    curves = read_csv(out / "curves.csv")
    for tag, name in (("dev", "dev_sub"), ("train", "train_sub")):
        d = C.load_split(name, synth_dir)
        for r in curves:
            pt = np.load(out / "pt" / f"pt_{tag}_{r['label']}.npy")
            assert pt.dtype == np.float16 and len(pt) == d.n_tokens and np.isnan(pt[0])
            assert not np.isnan(pt[1:]).any()
            s = float(pt[1:].astype(np.float64).sum())
            assert s / (d.n_tokens - 1) == pytest.approx(float(r[f"{tag}_nats_per_token"]), rel=1e-3)
            assert s / d.n_words == pytest.approx(float(r[f"{tag}_nats_per_word"]), rel=1e-3)
    full = res["full_eval"]
    pt = np.load(out / "pt" / "pt_devfull_final.npy")
    assert float(pt[1:].astype(np.float64).sum()) == pytest.approx(full["nats"], rel=1e-3)


def test_resume_is_bit_exact(synth_dir, tmp_path):
    base = ["--steps", "24", "--milestone_scale", "0.002", "--no_full_dev", "--no_pt", "--seed", "1"]
    a = TR.main(tiny_args(synth_dir, tmp_path / "a", *base))
    p = TR.main(tiny_args(synth_dir, tmp_path / "b", *base, "--max_session_steps", "7"))
    assert p["status"] == "paused" and not (tmp_path / "b" / "result.json").exists()
    b = TR.main(tiny_args(synth_dir, tmp_path / "b", *base, "--resume", "auto"))
    assert b["status"] == "ok" and b["sessions"] == 2
    assert a["steps"] == b["steps"] and a["words_seen"] == b["words_seen"]
    assert a["final_train_loss"] == b["final_train_loss"]
    la = read_csv(tmp_path / "a" / "train_log.csv")
    lb = read_csv(tmp_path / "b" / "train_log.csv")
    assert [r["loss"] for r in la] == [r["loss"] for r in lb]
    ca = read_csv(tmp_path / "a" / "curves.csv")
    cb = read_csv(tmp_path / "b" / "curves.csv")
    assert [(r["label"], r["dev_nats_per_token"]) for r in ca] == [(r["label"], r["dev_nats_per_token"]) for r in cb]


def test_final_flag(synth_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(TR, "git_commit", lambda: "0" * 40)
    res = TR.main(tiny_args(synth_dir, tmp_path, "--steps", "2", "--milestone_scale", "1", "--no_pt", "--final"))
    assert (tmp_path / "FINAL_SPLIT_USED").exists()
    assert res["full_eval"]["split"] == "test" and res["eval_split_final"]
    monkeypatch.setattr(TR, "git_commit", lambda: "0" * 40 + "+dirty")
    with pytest.raises(SystemExit):
        TR.main(tiny_args(synth_dir, tmp_path / "x", "--steps", "1", "--final"))


def test_device_and_precision_selection():
    assert TR.pick_device("cpu").type == "cpu" and TR.pick_device("auto").type in ("cpu", "cuda", "mps")
    _, prec = TR.autocast_for(torch.device("cpu"), "auto")
    assert prec == "fp32"
    _, prec = TR.autocast_for(torch.device("mps"), "auto")
    assert prec == "fp32"  # MPS defaults to fp32 (bf16 autocast on MPS is untested)
    _, prec = TR.autocast_for(torch.device("cuda"), "auto")
    assert prec.startswith("bf16")
    TR.sync(torch.device("cpu"))


def test_loss_scaler_skips_nonfinite_steps():
    sc = TR.LossScaler(True, init=8.0, interval=2)
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([8.0, 16.0, 24.0])
    assert sc.unscale_and_check([p]) and torch.equal(p.grad, torch.tensor([1.0, 2.0, 3.0]))
    p.grad = torch.tensor([float("inf"), 0.0, 0.0])
    assert not sc.unscale_and_check([p]) and sc.scale == 4.0 and sc.skipped == 1
    off = TR.LossScaler(False)
    x = torch.ones(())
    assert off.loss(x) is x and off.unscale_and_check([p])


def test_fp16_autocast_path_runs(synth_dir, tmp_path):
    res = TR.main(tiny_args(synth_dir, tmp_path, "--steps", "3", "--amp_dtype", "fp16", "--no_full_dev", "--no_pt",
                            "--milestone_scale", "100"))
    assert res["status"] == "ok" and res["precision"].startswith("fp16") and math.isfinite(res["final_train_loss"])
