"""CPU composition tests for the stacked attack flags (babylm.harness.stack, docs/babylm/attacks/STACKING.md):
off = the base bit for bit, alone and together; every pair and the full stack run; R0 exposure is unchanged by
A1 + A2; the defined interactions (A1 on the noised forward, R2L rows outside the A1 cache) hold."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from babylm.data import corpus as C
from babylm.harness import input_noise as IN
from babylm.harness import self_distill as SD
from babylm.harness import stack as ST
from babylm.harness import stream as S
from babylm.harness import train as TR
from babylm.tests.conftest import tiny_args
from babylm.tests.test_levers import synth_stream
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

COMMON = ("--epochs", "3", "--no_pt", "--milestone_scale", "0.002", "--seed", "1")
A1 = ("--sd_mode", "temporal", "--sd_alpha", "0.5", "--sd_k", "8", "--sd_start_epoch", "1")


def losses(out):
    return [r["loss"] for r in csv.DictReader((out / "train_log.csv").open())]


def train_recs(out):
    return [r for r in (json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()) if r.get("type") == "train"]


@pytest.fixture(scope="module")
def base(synth_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("stack_base") / "base"
    return out, TR.main(tiny_args(synth_dir, out, *COMMON))


# ------------------------------------------------------------------------------------------------ off = base
@pytest.mark.parametrize("name,extra", [
    ("explicit_off", ("--sd_mode", "none", "--noise_mode", "none", "--noise_p", "0", "--noise_r2l", "0")),
    ("a1_zero", ("--sd_mode", "temporal", "--sd_alpha", "0", "--sd_start_epoch", "0")),
    ("a2_zero", ("--noise_mode", "replace", "--noise_p", "0")),
    ("a1_a2_zero", ("--sd_mode", "temporal", "--sd_alpha", "0", "--sd_start_epoch", "0",
                    "--noise_mode", "replace", "--noise_p", "0")),
    ("a1_gauss_zero", ("--sd_mode", "temporal", "--sd_alpha", "0", "--sd_start_epoch", "0",
                       "--noise_mode", "gauss", "--noise_p", "0")),
])
def test_off_is_the_base_bit_for_bit(synth_dir, tmp_path, base, name, extra):
    """Every attack built and wired but inert, alone and together, reproduces the base's step losses and full-DEV
    nats exactly; explicit off flags build nothing and add no stack record."""
    b_out, b = base
    r = TR.main(tiny_args(synth_dir, tmp_path / name, *COMMON, *extra))
    assert losses(tmp_path / name) == losses(b_out)
    assert r["full_eval"]["nats"] == b["full_eval"]["nats"]
    assert r["words_seen"] == b["words_seen"] and r["tokens_seen"] == b["tokens_seen"]
    assert ("stack" in r["config"]) == (name != "explicit_off")
    if name == "explicit_off":
        assert "exposure_r0" not in r and "noise" not in r["config"] and "self_distill" not in r["config"]["levers"]


# ------------------------------------------------------------------------------------------------ pairs and stack
PAIRS = {
    "a1t+replace": (*A1, "--noise_mode", "replace", "--noise_p", "0.15"),
    "a1t+mask": (*A1, "--noise_mode", "mask", "--noise_p", "0.15"),
    "a1t+gauss": (*A1, "--noise_mode", "gauss", "--noise_p", "0.15"),
    "a1t+r2l": (*A1, "--noise_r2l", "0.5"),
    "a1smooth+replace": ("--sd_mode", "smooth", "--sd_alpha", "0.5", "--sd_smooth_eps", "0.1", "--sd_start_epoch", "1",
                         "--noise_mode", "replace", "--noise_p", "0.15"),
    "full": (*A1, "--noise_mode", "replace", "--noise_p", "0.1", "--noise_dist", "unigram", "--noise_r2l", "0.3"),
}


@pytest.mark.parametrize("name", list(PAIRS))
def test_pairs_and_full_stack_run_with_unchanged_exposure(synth_dir, tmp_path, base, name):
    b_out, b = base
    r = TR.main(tiny_args(synth_dir, tmp_path / name, *COMMON, *PAIRS[name]))
    assert r["status"] == "ok"
    ls = [float(x) for x in losses(tmp_path / name)]
    assert all(math.isfinite(x) for x in ls) and ls != [float(x) for x in losses(b_out)]
    # R0: one view per window, words counted on the canonical targets before any transform
    assert r["words_seen"] == b["words_seen"] and r["tokens_seen"] == b["tokens_seen"]
    e = r["exposure_r0"]
    n_words = C.load_split("train_10M", synth_dir).n_words
    assert e["total_words"] == e["input_words"] == r["words_seen"] and e["views_per_window"] == 1
    assert e["cap_words"] == 10 * n_words and e["within_cap"]
    st = r["config"]["stack"]
    assert len(st["attacks"]) == 2 and st["order"] == list(ST.ORDER)
    recs = train_recs(tmp_path / name)
    on = [x for x in recs if x["sd_on"] == 1]
    assert on and all(math.isfinite(x.get("sd_kl", 0.0)) for x in on)
    if "--noise_r2l" in PAIRS[name]:
        assert st["sd_skips_r2l_rows"]
        r2l = float(PAIRS[name][PAIRS[name].index("--noise_r2l") + 1])
        # reversed rows carry no cache key: coverage after epoch 1 sits near 1 - r2l, not near 1
        late = [x["sd_cov"] for x in recs if x["epoch"] > 1.3]
        assert abs(np.mean(late) - (1 - r2l)) < 0.2


def test_full_stack_with_mntp_ema_and_line_windows(synth_dir, tmp_path):
    """The base round-1 levers (MNTP, EMA, line windows) under A1 + A2 replace + R2L: A2 mask shares the MNTP MASK
    id 16384 by design; MNTP rows get neither noise nor distillation."""
    plan = ("--epochs", "2", "--steps", "8", "--no_pt", "--no_full_dev", "--milestone_scale", "0.002", "--seed", "1",
            "--window_plan", "line:32,line:64", "--mntp_frac", "0.25", "--ema_decay", "0.9")
    b = TR.main(tiny_args(synth_dir, tmp_path / "b", *plan))
    for name, extra in (("r", ("--noise_mode", "replace", "--noise_p", "0.1", "--noise_r2l", "0.3")),
                        ("m", ("--noise_mode", "mask", "--noise_p", "0.1"))):
        r = TR.main(tiny_args(synth_dir, tmp_path / name, *plan, *A1, *extra))
        assert r["status"] == "ok" and r["config"]["model_cfg"]["vocab_size"] == 16512
        assert r["words_seen"] == b["words_seen"] and r["tokens_seen"] == b["tokens_seen"]
        recs = train_recs(tmp_path / name)
        assert all(math.isfinite(x["mntp_loss"]) for x in recs)
        assert all(0 <= x["noise_frac"] <= 0.2 for x in recs)


def test_full_stack_resume_is_bit_exact(synth_dir, tmp_path):
    """A1's cache (sd_cache.pt) and A2's private generator (resume.pt) both travel: pause + resume = one session."""
    args = (*COMMON, *PAIRS["full"], "--no_full_dev", "--save")
    TR.main(tiny_args(synth_dir, tmp_path / "a", *args))
    TR.main(tiny_args(synth_dir, tmp_path / "b", *args, "--max_session_steps", "17"))
    r = TR.main(tiny_args(synth_dir, tmp_path / "b", *args, "--resume", "auto"))
    assert r["sessions"] == 2
    assert losses(tmp_path / "a") == losses(tmp_path / "b")
    assert [x.get("sd_kl") for x in train_recs(tmp_path / "a")] == [x.get("sd_kl") for x in train_recs(tmp_path / "b")]


# ------------------------------------------------------------------------------------------------ interactions
def _tiny_model(V=16512):
    torch.manual_seed(0)
    return baseline.build(ModelConfig(vocab_size=V, n_layer=2, n_head=2, n_embd=64, seq_len=32))


def _sd(V, k=4):
    class A:  # the parsed flags SelfDistill reads
        sd_mode, sd_alpha, sd_start_epoch, sd_write_until, sd_smooth_eps, sd_k, sd_decay = \
            "temporal", 0.5, 0.0, math.inf, 0.0, k, 0.0
        sd_cache_device = "same"
    return A


def _batch(B=6, T=32, seed=4):
    tok = synth_stream(n_lines=300)
    smp = S.EpochSampler(len(tok), T, data_seed=seed)
    starts = smp.next_starts(B)
    x, y = S.gather_windows(tok, starts, T)
    return tok, x, y, SD.stream_positions(starts, T)


def test_a1_caches_the_noised_forward():
    """A1 + A2 replace: the cached top-1 at every position is the model's argmax on the NOISED input (no hidden
    clean forward), and it differs from the clean-input argmax somewhere."""
    tok, x, y, pos = _batch()
    m = _tiny_model()
    nz = IN.InputNoise("replace", 0.5, vocab_real=C.VOCAB_SIZE, seed=1)
    xn, yn = nz(x, y)
    assert torch.equal(yn, y) and not torch.equal(xn, x)
    sd = SD.SelfDistill(_sd(16512), len(tok), 16512, torch.device("cpu"))
    with torch.no_grad():
        sd.loss(m, xn, yn, ST.sd_positions(pos, nz), 0.0, torch.autocast("cpu", enabled=False))
        zn = m.logits(m.hidden(xn)).argmax(-1)
        zc = m.logits(m.hidden(x)).argmax(-1)
    pt = torch.from_numpy(pos.reshape(-1))
    idx, _, has = sd.cache.read(pt)
    assert has.all()
    assert torch.equal(idx[:, 0], zn.reshape(-1))
    assert not torch.equal(idx[:, 0], zc.reshape(-1))


def test_r2l_rows_neither_read_nor_write_the_a1_cache():
    tok, x, y, pos = _batch(B=8)
    m = _tiny_model()
    nz = IN.InputNoise("none", 0.0, r2l=0.5, vocab_real=C.VOCAB_SIZE, seed=3)
    xn, yn = nz(x, y)
    rows = nz.last_r2l
    assert rows is not None and 0 < int(rows.sum()) < len(rows)
    pk = ST.sd_positions(pos, nz)
    assert (pk[rows.numpy()] == -1).all() and np.array_equal(pk[~rows.numpy()], pos[~rows.numpy()])
    sd = SD.SelfDistill(_sd(16512), len(tok), 16512, torch.device("cpu"))
    amp = torch.autocast("cpu", enabled=False)
    with torch.no_grad():
        sd.loss(m, xn, yn, pk, 0.0, amp)  # first visit: writes only the L2R rows
    assert sd.pop_stats()["sd_cov"] == 0.0
    with torch.no_grad():
        for p in m.parameters():  # the model moves on before its next visit, so the cached target differs from q
            p.add_(0.05 * torch.randn_like(p))
    written = sd.cache.cnt.numpy()[:len(tok)] > 0  # the last row is the cache's spare row
    expect = np.zeros(len(tok), dtype=bool)
    expect[pos[~rows.numpy()].ravel()] = True
    only_r2l = np.setdiff1d(pos[rows.numpy()].ravel(), pos[~rows.numpy()].ravel())
    assert written[expect].all() and not written[only_r2l].any()
    # second visit with the KL on: the loss = CE + alpha * KL over the L2R rows only, averaged over ALL targets
    z = m.logits(m.hidden(xn))
    loss, ce = sd.loss(m, xn, yn, pk, 0.0, amp)
    assert float(ce) == pytest.approx(float(F.cross_entropy(z.reshape(-1, z.shape[-1]), yn.reshape(-1))), rel=1e-6)
    assert float(loss) > float(ce)
    st = sd.pop_stats()
    assert st["sd_cov"] == pytest.approx(float((~rows).sum()) / len(rows))


def test_all_rows_without_a_key_fall_back_to_ce():
    tok, x, y, pos = _batch(B=4)
    m = _tiny_model()
    sd = SD.SelfDistill(_sd(16512), len(tok), 16512, torch.device("cpu"))
    loss, ce = sd.loss(m, x, y, np.full_like(pos, -1), 0.0, torch.autocast("cpu", enabled=False))
    assert float(loss) == float(ce) and int(sd.cache.cnt[:len(tok)].sum()) == 0
    with pytest.raises(RuntimeError):  # a key on a padded target is still an error (checked whenever a batch has -1s)
        y2, p2 = y.clone(), pos.copy()
        y2[0, -1] = -1
        p2[1, 0] = -1
        sd.loss(m, x, y2, p2, 0.0, torch.autocast("cpu", enabled=False))


def test_exposure_ledger_counts_every_source():
    e = ST.exposure_r0(90_000_000, 10_000_000, synthetic_words=8_000_000, generator_words=3_000_000)
    assert e["total_words"] == 101_000_000 and e["cap_words"] == 100_000_000 and not e["within_cap"]
    assert ST.exposure_r0(50, 10, views_per_window=2)["total_words"] == 100


# ------------------------------------------------------------------------------------------------ A3 (BPE-dropout)
A3 = ("--bpe_dropout", "0.2x2,0", "--bpe_dropout_workers", "1")


def test_canonical_keys_on_a_dropout_stream(synth_dir):
    """Every key points at a canonical token with the same id; keys of intact tokens increase; re-segmented
    pieces get -1; the canonical stream maps to itself."""
    from babylm.harness import bpe_dropout as BD
    tr = C.load_split("train_10M", synth_dir)
    meta = BD.build_streams(tr, synth_dir, [0.2], "all", 63, 1, workers=1, log=lambda m: None)
    tok, _ = BD.load_stream(synth_dir, meta[0]["name"])
    ck = ST.CanonicalKeys(tr.tokens, BD.BPE().tok_bytes)
    k = ck.keys(tok)
    hit = k >= 0
    assert len(k) == len(tok) > tr.n_tokens and 0.3 < hit.mean() < 1.0
    assert np.array_equal(np.asarray(tok)[hit], np.asarray(tr.tokens)[k[hit]])
    assert (np.diff(k[hit]) > 0).all()
    # intact share = canonical tokens that survived; every canonical BOS survives
    assert set(np.flatnonzero(np.asarray(tr.tokens) == C.BOS_ID)) <= set(k[hit].tolist())
    assert np.array_equal(ck.keys(tr.tokens), np.arange(tr.n_tokens))
    smp = BD.MultiStreamSampler([(tok, _)], (tr.tokens, tr.words), 64, 3)
    x, y, _nw = smp.next_batch(5)
    pos = ck.positions(smp, 64)
    v = pos >= 0
    assert pos.shape == y.shape and np.array_equal(np.asarray(tr.tokens)[pos[v]], y.numpy()[v])


def test_a3_zero_with_every_inert_attack_is_the_base(synth_dir, tmp_path, base):
    b_out, b = base
    r = TR.main(tiny_args(synth_dir, tmp_path / "z", *COMMON, "--bpe_dropout", "0", "--sd_mode", "temporal",
                          "--sd_alpha", "0", "--sd_start_epoch", "0", "--noise_mode", "replace", "--noise_p", "0"))
    assert losses(tmp_path / "z") == losses(b_out) and r["full_eval"]["nats"] == b["full_eval"]["nats"]
    assert r["words_seen"] == b["words_seen"]


@pytest.fixture(scope="module")
def a3_alone(synth_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("stack_a3") / "a3"
    return out, TR.main(tiny_args(synth_dir, out, *COMMON, "--no_full_dev", *A3))


A3_PAIRS = {
    "a3+a1t": A1,
    "a3+replace": ("--noise_mode", "replace", "--noise_p", "0.15"),
    "a3+r2l": ("--noise_r2l", "0.3"),
    "a3+a1t+replace+r2l": (*A1, "--noise_mode", "replace", "--noise_p", "0.1", "--noise_r2l", "0.3"),
}


@pytest.mark.parametrize("name", list(A3_PAIRS))
def test_a3_pairs_and_full_stack(synth_dir, tmp_path, a3_alone, name):
    """With A3 on, A1/A2 change nothing about exposure: the same steps, tokens and canonical words as A3 alone.
    A1 keys only intact canonical tokens, so its coverage in the second dropout epoch is lower than in the canonical
    last epoch (where every token has a key and is cached if it was intact in either earlier epoch)."""
    a_out, a = a3_alone
    extra = A3_PAIRS[name]
    r = TR.main(tiny_args(synth_dir, tmp_path / name, *COMMON, "--no_full_dev", *A3, *extra))
    assert r["status"] == "ok"
    assert (r["steps"], r["tokens_seen"], r["words_seen"]) == (a["steps"], a["tokens_seen"], a["words_seen"])
    ls = [float(x) for x in losses(tmp_path / name)]
    assert all(math.isfinite(x) for x in ls) and ls != [float(x) for x in losses(a_out)]
    e = r["exposure_r0"]
    assert e["input_words"] == r["words_seen"] and e["within_cap"]
    assert r["config"]["stack"]["attacks"][-1].startswith("A3:")
    if "--sd_mode" in extra:
        assert r["config"]["stack"]["sd_keys_canonical_tokens_only"]
        recs = train_recs(tmp_path / name)
        r2l = 0.3 if "--noise_r2l" in extra else 0.0
        mid = [x["sd_cov"] for x in recs if 1.2 < x["epoch"] < 1.9]   # epoch 2: dropout stream, epoch 1 cached
        last = [x["sd_cov"] for x in recs if x["epoch"] > 2.3]        # epoch 3: canonical stream
        assert mid and last and 0 < np.mean(mid) < np.mean(last) <= 1 - r2l + 0.15


def test_a3_with_a1_resume_is_bit_exact(synth_dir, tmp_path, a3_alone):
    args = (*COMMON, "--no_full_dev", *A3, *A1, "--noise_mode", "replace", "--noise_p", "0.1", "--save")
    full = TR.main(tiny_args(synth_dir, tmp_path / "a", *args))
    TR.main(tiny_args(synth_dir, tmp_path / "b", *args, "--max_session_steps", str(full["steps"] // 2)))
    r = TR.main(tiny_args(synth_dir, tmp_path / "b", *args, "--resume", "auto"))
    assert r["sessions"] == 2 and losses(tmp_path / "a") == losses(tmp_path / "b")


# ------------------------------------------------------------------------------------------------ A4 (second cycle)
@pytest.fixture(scope="module")
def a4_g(synth_dir, tmp_path_factory):
    from babylm.tests.test_a4 import make_syn
    root = tmp_path_factory.mktemp("stack_a4")
    TR.main(tiny_args(synth_dir, root / "G", "--epochs", "2", "--save", "--no_full_dev", "--no_pt",
                      "--milestone_scale", "0.002", "--seed", "1"))
    syn = make_syn(synth_dir, root / "syn", n_docs=60)
    cyc = ("--init_from", str(root / "G" / "resume.pt"), "--warmup_frac", "0.1", "--sources", "real:train_10M:2",
           f"synthetic:{syn}:1", "--exposure_extra", f"G=@{root / 'G'}/result.json")
    alone = TR.main(tiny_args(synth_dir, root / "alone", *cyc, "--no_full_dev", "--no_pt", "--milestone_scale",
                              "0.002", "--seed", "1"))
    return root, cyc, alone


def test_a4_inert_with_inert_a1_a2_is_the_base(synth_dir, tmp_path):
    """--init_from a step-0 checkpoint (same initial weights, 0 words) plus inert A1 and A2 = the base."""
    args = ("--epochs", "2", "--steps", "8", "--no_pt", "--milestone_scale", "0.002", "--seed", "1")
    TR.main(tiny_args(synth_dir, tmp_path / "z", "--epochs", "2", "--steps", "0", "--save", *args[4:]))
    a = TR.main(tiny_args(synth_dir, tmp_path / "a", *args))
    b = TR.main(tiny_args(synth_dir, tmp_path / "b", *args, "--init_from", str(tmp_path / "z" / "resume.pt"),
                          "--sd_mode", "temporal", "--sd_alpha", "0", "--sd_start_epoch", "0",
                          "--noise_mode", "replace", "--noise_p", "0"))
    assert losses(tmp_path / "a") == losses(tmp_path / "b") and a["full_eval"]["nats"] == b["full_eval"]["nats"]


@pytest.mark.parametrize("name,extra", [
    ("a4+a1t", A1),
    ("a4+replace", ("--noise_mode", "replace", "--noise_p", "0.15")),
    ("a4+gauss", ("--noise_mode", "gauss", "--noise_p", "0.15")),
    ("a4+a1t+replace", (*A1, "--noise_mode", "replace", "--noise_p", "0.1")),
])
def test_a4_cycle_pairs(synth_dir, tmp_path, a4_g, name, extra):
    """A1/A2 inside the cycle: the cycle's ledger is unchanged (same words per kind, same lineage); exposure_r0
    restates it (real input incl. prior lineage + synthetic + generator words). A1 keys only real training-split
    rows, so its coverage stays below the real share."""
    root, cyc, alone = a4_g
    r = TR.main(tiny_args(synth_dir, tmp_path / name, *cyc, *extra, "--no_full_dev", "--no_pt", "--milestone_scale",
                          "0.002", "--seed", "1"))
    assert r["status"] == "ok"
    assert r["exposure"]["cycle_words"] == alone["exposure"]["cycle_words"] and r["words_seen"] == alone["words_seen"]
    e = r["exposure_r0"]
    assert e["input_words"] + e["synthetic_words"] == r["words_seen"]
    assert e["synthetic_words"] == r["exposure"]["cycle_words"]["synthetic"] > 0
    assert e["generator_words"] == sum(r["exposure"]["extra_words"].values())
    assert e["total_words"] == r["exposure"]["total"] and e["within_cap"] == r["exposure"]["within_cap"]
    assert any(a.startswith("A4:") for a in r["config"]["stack"]["attacks"])
    ls = [float(x) for x in losses(tmp_path / name)]
    assert all(math.isfinite(x) for x in ls) and ls != [float(x) for x in losses(root / "alone")]
    if "--sd_mode" in extra:
        recs = train_recs(tmp_path / name)
        cw = r["exposure"]["cycle_words"]
        real_share = cw["real"] / (cw["real"] + cw["synthetic"])
        assert all(x["sd_cov"] <= real_share + 0.1 for x in recs)
        assert max(x["sd_cov"] for x in recs) > 0.2  # the second real pass reads what the first wrote


def test_cycle_keys_point_at_training_tokens(synth_dir, a4_g):
    from babylm.harness import cycle as CY
    from babylm.harness import levers as LV
    root, cyc, _ = a4_g
    syn = cyc[cyc.index("--sources") + 2].split(":", 1)[1].rsplit(":", 1)[0]
    tr = C.load_split("train_10M", synth_dir)
    for spec in (LV.EpochSpec("stream", 32),):
        cs = CY.CycleSampler([CY.load_source("real:train_10M:1", synth_dir),
                              CY.load_source(f"synthetic:{syn}:1", synth_dir)], spec, 32 * 8, seed=3)
        n_real = n_syn = 0
        for _ in range(cs.total_steps):
            x, y, _, _, _ = cs.next()
            pos = ST.cycle_positions(cs, tr, y)
            v = pos >= 0
            assert np.array_equal(np.asarray(tr.tokens)[pos[v]], y.numpy()[v])
            rows = range((cs.ptr - 1) * cs.batch_seqs, cs.ptr * cs.batch_seqs)
            src = np.array([int(cs.w_src[w]) for w in rows])
            assert (pos[src == 1] == -1).all() and (pos[src == 0] >= 0).all()
            n_real, n_syn = n_real + int((src == 0).sum()), n_syn + int((src == 1).sum())
        assert n_real and n_syn


def test_a4_refusals(synth_dir, tmp_path, a4_g):
    root, cyc, _ = a4_g
    with pytest.raises(SystemExit, match="do not compose"):
        TR.main(tiny_args(synth_dir, tmp_path / "x", *cyc, "--epochs", "1", *A3, "--no_pt"))
    with pytest.raises(SystemExit, match="cfg differs"):  # R2L pads the vocab; G was trained unpadded
        TR.main(tiny_args(synth_dir, tmp_path / "y", *cyc, "--noise_r2l", "0.3", "--no_pt"))
