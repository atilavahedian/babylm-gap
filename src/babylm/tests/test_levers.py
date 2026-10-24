"""CPU tests for the attack-base levers (babylm.harness.levers) and their harness wiring."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from babylm.data import corpus as C
from babylm.harness import levers as L
from babylm.harness import stream as S
from babylm.harness import train as TR
from babylm.tests.conftest import tiny_args
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

NL, EOT, BOS, PAD, MASK = 7, 0, 0, 0, 99


def synth_stream(n_lines=400, seed=0, max_line=40):
    """Token stream with lines (ending in NL) grouped in documents (EOT first), some lines longer than T."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_lines):
        if i % 25 == 0:
            out.append(EOT)
        ln = int(rng.integers(1, max_line))
        out += list(rng.integers(10, 90, ln)) + [NL]
    return np.asarray(out, dtype=np.uint16)


def test_parse_plan():
    assert L.parse_plan(None, 10, 1024) == [L.EpochSpec("stream", 1024)] * 10
    assert L.parse_plan("line:512", 3, 1024) == [L.EpochSpec("line", 512)] * 3
    p = L.parse_plan("line:128x6,line:256x2,line:512x2", 10, 1024)
    assert [s.T for s in p] == [128] * 6 + [256] * 2 + [512] * 2
    with pytest.raises(ValueError):
        L.parse_plan("line:128x3", 10, 1024)
    with pytest.raises(ValueError):
        L.parse_plan("words:12", 1, 1024)


def test_line_starts():
    tok = np.array([EOT, 5, 6, NL, 8, NL, EOT, 9, NL, NL, 3], dtype=np.uint16)
    s = L.line_starts(tok, np.array([NL]), EOT)
    assert s.tolist() == [0, 1, 4, 7, 9, 10]  # 6 (EOT) is not a start; 9 (empty line NL) is


def test_stream_plan_reproduces_port_sampler():
    tok = synth_stream()
    T, E, bs = 32, 3, 4
    a = S.EpochSampler(len(tok), T, data_seed=5)
    b = L.PlanSampler(tok, L.parse_plan(None, E, T), bs * T, seed=5)
    assert b.dropped_windows < bs
    for _ in range(b.total_steps):
        xa, ya = S.gather_windows(tok, a.next_starts(bs), T)
        xb, yb, _, Tb, tpos = b.next()
        assert Tb == T and torch.equal(xa, xb) and torch.equal(ya, yb)
        assert np.array_equal(tok[tpos].astype(np.int64), yb.numpy().ravel())


@pytest.mark.parametrize("plan,E", [("line:32", 3), ("line:16x2,line:32", 3), ("line:32,stream:32x2", 3)])
def test_line_plan_exposure_and_alignment(plan, E):
    tok = synth_stream()
    starts = L.line_starts(tok, np.array([NL]), EOT)
    s = L.PlanSampler(tok, L.parse_plan(plan, E, 32), 128, seed=1, bos_id=BOS, pad_id=PAD, starts=starts)
    c = s.target_counts()
    assert c.max() <= E
    if s.dropped_windows == 0 and "stream" not in plan:
        assert (c == E).all()  # every token a target exactly once per epoch
    for i in range(s.total_steps):
        x, y, ep, T, tpos = s.batch_at(i)
        assert np.array_equal(tok[tpos].astype(np.int64), y.numpy()[y.numpy() >= 0])
        assert x.shape == y.shape == (128 // T, T)
        w0, n, _ = s.steps[i]
        for r in range(n):
            w = w0 + r
            if s.w_bos[w]:
                assert x[r, 0] == BOS
                ln = int(s.w_len[w])
                assert 1 <= ln <= T and (y[r, ln:] == -1).all() and (y[r, :ln] >= 0).all()
            valid = y[r, 1:] >= 0
            assert torch.equal(y[r, :-1][valid], x[r, 1:][valid])  # y[j-1] == x[j] on real tokens


def test_line_windows_start_on_lines():
    tok = synth_stream(max_line=20)  # all lines shorter than T: every window starts at a line start
    starts = L.line_starts(tok, np.array([NL]), EOT)
    s = L.PlanSampler(tok, L.parse_plan("line:64", 2, 64), 256, seed=3, bos_id=BOS, pad_id=PAD, starts=starts)
    assert set(s.w_start.tolist()) <= set(starts.tolist())
    fill = s.w_len.mean() / 64
    assert fill > 0.6


def test_mntp_corrupt():
    g = torch.Generator().manual_seed(0)
    T = 64
    seq = torch.randint(10, 90, (8, T + 1), generator=g)
    x, y = seq[:, :-1].clone(), seq[:, 1:].clone()
    y[0, 40:] = -1  # a padded row
    x[0, 41:] = PAD
    xc, ym = L.mntp_corrupt(x, y, 0.3, MASK, 90, g)
    chosen_j = (ym[:, :-1] >= 0)  # ym[j-1] scored <-> position j chosen
    assert (ym[:, :-1][chosen_j] == x[:, 1:][chosen_j]).all()
    assert (ym[:, -1] == -1).all() and (xc[:, 0] == x[:, 0]).all()
    assert not chosen_j[0, 40:].any()  # never past the real tokens
    assert chosen_j.any(1).all()
    unchosen = ~torch.cat([torch.zeros(8, 1, dtype=torch.bool), chosen_j], 1)
    assert torch.equal(xc[unchosen], x[unchosen])
    frac = chosen_j.float().sum() / (y[:, 1:] >= 0).sum()
    assert 0.2 < frac < 0.4
    big = torch.randint(10, 90, (64, 513), generator=g)
    xc, ym = L.mntp_corrupt(big[:, :-1], big[:, 1:], 0.3, MASK, 90, g)
    ch = torch.cat([torch.zeros(64, 1, dtype=torch.bool), ym[:, :-1] >= 0], 1)
    m = (xc[ch] == MASK).float().mean()
    assert 0.75 < m < 0.85
    assert L.mntp_rows(16, 1 / 16) == 1 and L.mntp_rows(16, 0.25) == 4 and L.mntp_rows(16, 0) == 0
    assert L.mntp_rows(4, 1 / 16) == 1
    assert abs(L.mntp_p(0, 100, 0.3, 0.15) - 0.3) < 1e-9 and abs(L.mntp_p(99, 100, 0.3, 0.15) - 0.15) < 1e-9


def tiny_cfg():
    return ModelConfig(vocab_size=600, n_layer=2, n_head=2, n_embd=64, seq_len=64, extra={})


def test_hybrid_causal_identical_and_bidirectional_sees_future():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    a = baseline.build(cfg).eval()
    torch.manual_seed(0)
    b = baseline.build(cfg).eval()
    assert L.make_switchable(b) == cfg.n_layer
    assert a.state_dict().keys() == b.state_dict().keys()
    for (k, va), vb in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(va, vb), k
    with torch.no_grad():  # make the zero-init projections live so attention matters
        for m in (a, b):
            torch.manual_seed(1)
            for blk in m.blocks:
                blk.attn.c_proj.weight.normal_(0, 0.05)
                blk.mlp.c_proj.weight.normal_(0, 0.05)
    x = torch.randint(0, 500, (2, 32))
    y = torch.randint(0, 500, (2, 32))
    la, aa = a(x, y)
    lb, ab = b(x, y)
    assert torch.equal(aa["per_token_loss"], ab["per_token_loss"])
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % 500
    with L.BidirectionalAttention(b):
        _, f1 = b(x, y)
        _, f2 = b(x2, y)
    assert not torch.allclose(f1["per_token_loss"][:, 0], f2["per_token_loss"][:, 0])  # future changes the past
    _, c1 = b(x, y)
    _, c2 = b(x2, y)
    assert torch.allclose(c1["per_token_loss"][:, :-1], c2["per_token_loss"][:, :-1])  # causal restored
    assert all(blk.attn.is_causal for blk in b.blocks)


def test_ema():
    m = torch.nn.Linear(3, 3)
    e = L.EMA(m, 0.9)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    d = e.update(m)
    assert abs(d - 0.1) < 1e-9  # warmup: (1 + 0) / (10 + 0)
    assert torch.allclose(e.shadow.weight, w0 * 0.1 + (w0 + 1) * 0.9)
    for _ in range(200):
        d = e.update(m)
    assert d == 0.9 and torch.allclose(e.shadow.weight, m.weight, atol=1e-6)
    sd = e.state_dict()
    e2 = L.EMA(m, 0.5)
    e2.load_state_dict(sd)
    assert e2.decay == 0.9 and e2.n == e.n


# ------------------------------------------------------------------------------------------------
# harness wiring (synthetic data from conftest, real 16k tokenizer, tiny model, CPU)
# ------------------------------------------------------------------------------------------------
def _losses(out):
    return [float(r["loss"]) for r in (json.loads(x) for x in (out / "log.jsonl").read_text().splitlines())
            if r.get("type") == "train"]


def test_harness_stream_plan_equals_port_path(synth_dir, tmp_path):
    common = ("--epochs", "2", "--steps", "6", "--no_full_dev", "--no_pt")
    TR.main(tiny_args(synth_dir, tmp_path / "port", *common))
    TR.main(tiny_args(synth_dir, tmp_path / "plan", *common, "--window_plan", "stream:64"))
    assert _losses(tmp_path / "port") == _losses(tmp_path / "plan")


def test_harness_line_windows_tokenizer_newlines(synth_dir):
    from tokenizers import Tokenizer
    tr = C.load_split("train_10M", synth_dir)
    nl = L.newline_ids(Tokenizer.from_file(str(C.TOKENIZER_PATH)))
    assert len(nl) > 0
    starts = L.line_starts(np.asarray(tr.tokens), nl, C.BOS_ID)
    docs = json.loads((synth_dir / "docs.json").read_text())["train_10M"]
    n_lines = sum(len(t.split("\n")) for s in docs.values() for t in s)
    assert abs(len(starts) - n_lines) <= 0.02 * n_lines  # one start per document line


def test_harness_all_levers_run_and_checkpoint(synth_dir, tmp_path):
    out = tmp_path / "all"
    r = TR.main(tiny_args(synth_dir, out, "--epochs", "2", "--no_pt", "--save", "--milestone_scale", "0.001",
                          "--window_plan", "line:32,line:64", "--mntp_frac", "0.25", "--ema_decay", "0.9",
                          "--lr_scale", "0.5"))
    assert r["status"] == "ok" and r["config"]["model_cfg"]["vocab_size"] == 16512
    lv = r["config"]["levers"]
    assert lv["window_plan"] == ["line:32", "line:64"] and lv["mask_id"] == C.VOCAB_SIZE
    assert r["config"]["base_lrs"]["muon"] == pytest.approx(0.0325)
    recs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    tr = [x for x in recs if x.get("type") == "train"]
    assert all(np.isfinite(x["mntp_loss"]) for x in tr)
    assert tr[-1]["mntp_loss"] < tr[0]["mntp_loss"]
    ck = torch.load(sorted((out / "checkpoints").glob("*.pt"))[-1], weights_only=False)
    assert ck["model_raw"] is not None and ck["ema_decay"] == 0.9
    assert not torch.equal(ck["model"]["lm_head.weight"], ck["model_raw"]["lm_head.weight"])
    assert ck["mntp_frac"] == 0.25
    from babylm.harness import export_hf as E
    ck_path = sorted((out / "checkpoints").glob("*.pt"))[-1]
    assert E.export_checkpoint(ck_path, out / "hf")["automodel_attention"] == "bidirectional"
    assert E.verify_export(out / "hf", ck_path) < 1e-4  # the exported model is the EMA weights
    words = C.load_split("train_10M", synth_dir).n_words
    assert r["words_seen"] <= 2 * words and r["words_seen"] >= 1.9 * words


def test_harness_group_wd_and_dev_minimum(synth_dir, tmp_path):
    """--head_wd / --embed_wd set only their AdamW group's decay (others keep --adam_wd); the head actually shrinks
    relative to a no-decay run; result.json carries the DEV-sub minimum over all milestones."""
    common = ("--epochs", "2", "--steps", "8", "--no_pt", "--save", "--milestone_scale", "0.001")
    r0 = TR.main(tiny_args(synth_dir, tmp_path / "a", *common))
    r1 = TR.main(tiny_args(synth_dir, tmp_path / "b", *common, "--head_wd", "5.0", "--embed_wd", "0.001"))
    gw = r1["config"]["levers"]["adam_group_wd"]
    assert gw["head"] == 5.0 and gw["embed"] == 0.001 and gw.get("scalar", 0.0) == 0.0
    assert set(r0["config"]["levers"]["adam_group_wd"].values()) == {0.0}
    h0 = torch.load(tmp_path / "a" / "resume.pt", weights_only=False)["model"]["lm_head.weight"]
    h1 = torch.load(tmp_path / "b" / "resume.pt", weights_only=False)["model"]["lm_head.weight"]
    assert h1.norm() < h0.norm()
    m = r1["dev_sub_min"]
    rows = [float(x) for x in __import__("csv").DictReader((tmp_path / "b" / "curves.csv").open())
            for x in [x["dev_nats_per_word"]]]
    assert len(rows) >= 2 and m["dev_nats_per_word"] == pytest.approx(min(rows))
    assert m["final_minus_min"] == pytest.approx(rows[-1] - min(rows)) and m["final_minus_min"] >= 0


def test_harness_tied_embeddings_and_no_value_embeds(synth_dir, tmp_path):
    """--tie_embeddings: one weight (head init scale) in the head AdamW group only, trains, exports exactly;
    --extra n_value_embeds=0 removes the value-embedding tables and exports exactly."""
    from babylm.harness import export_hf as E
    out = tmp_path / "tie"
    r = TR.main(tiny_args(synth_dir, out, "--epochs", "2", "--no_pt", "--save", "--milestone_scale", "0.001",
                          "--tie_embeddings", "--extra", "n_value_embeds=0"))
    assert r["status"] == "ok" and r["config"]["levers"]["tie_embeddings"] is True
    assert r["config"]["model_cfg"]["tie_embeddings"] is True
    ck_path = sorted((out / "checkpoints").glob("*.pt"))[-1]
    ck = torch.load(ck_path, weights_only=False)
    assert torch.equal(ck["model"]["embed.weight"], ck["model"]["lm_head.weight"])
    assert not any(k.startswith("value_embeds") for k in ck["model"])
    assert E.export_checkpoint(ck_path, out / "hf")["cfg"]["tie_embeddings"] is True
    assert E.verify_export(out / "hf", ck_path) < 1e-4
    # optimizer layout: the tied weight is the head group's only param, at the head LR; no embed group remains
    groups = [g for o in torch.load(out / "resume.pt", weights_only=False)["optimizer"]["optimizers"]
              for g in o["param_groups"] if g.get("kind") == "adam"]
    names = {g["name"]: g for g in groups}
    assert "embed" not in names and len(names["head"]["params"]) == 1
    assert names["head"]["initial_lr"] == pytest.approx(r["config"]["base_lrs"]["adam_head"])


def test_fast_summary_parses_reports(tmp_path):
    from babylm.harness.fast_summary import summarize
    base = tmp_path / "m" / "fast" / "zero_shot" / "causal"
    for task, ds, v in (("blimp", "blimp_fast", "71.25"), ("ewok", "ewok_fast", "52.5")):
        d = base / task / ds
        d.mkdir(parents=True)
        (d / "best_temperature_report.txt").write_text(f"TEMPERATURE: 1.0\n\n### AVERAGE ACCURACY\n{v}\n")
        (d / "predictions.json").write_text("{}")
    (base / "reading").mkdir(parents=True)
    (base / "reading" / "report.txt").write_text("EYE TRACKING SCORE: 9.50\nSELF-PACED READING SCORE: 3.40\n")
    s = summarize(str(tmp_path))
    assert s == {"blimp/blimp_fast": 71.25, "ewok/ewok_fast": 52.5, "reading/eye_tracking": 9.5,
                 "reading/self_paced_reading": 3.4}


def test_round1_cells_start(synth_dir, tmp_path):
    """Every round-1 cell's lever flags (model, extras, switches, group WDs, MNTP) start and train at tiny scale:
    the all-on cell once combined --embed_wd with --tie_embeddings and died at startup on the GPU."""
    from babylm.configs import make_base_jobs as MB
    keep = ("--model", "--embed_wd", "--head_wd", "--mntp_frac", "--extra")
    for j in MB.stage_round1("S", "2.0", "0.5")["jobs"][1:]:
        a, lever = j["args"], []
        for i, x in enumerate(a):
            if x in keep and not (x == "--extra" and a[i + 1] in ("softcap=15", "unet=true")):
                lever += [x, a[i + 1]]
            elif x == "--tie_embeddings":
                lever.append(x)
        r = TR.main(tiny_args(synth_dir, tmp_path / j["name"], "--steps", "2", "--no_pt", "--milestone_scale", "0.001",
                              *lever))
        assert r["status"] == "ok", j["name"]


def test_size_stage_matches_s_winner():
    """The size stage is the S round-1 winner's job with only --size, --muon_wd, --lr_scale and the tag changed."""
    from babylm.configs.make_base_jobs import override, stage_round1, stage_size
    win = next(j for j in stage_round1("S", "2.0", "0.5")["jobs"] if j["name"] == "base_S_wd0p4_lr0p5_r1_ve0_tied")
    for j in stage_size("M", ("0.8",), ("0.25",))["jobs"]:
        want = override(override(override(override(win["args"], "--size", "M"), "--muon_wd", "0.8"),
                                 "--lr_scale", "0.25"), "--tag", j["name"])
        assert j["args"] == want


def test_round2_one_change_each(tmp_path):
    """Each round-2 cell changes at most one knob besides the epoch count (plus the tag)."""
    from babylm.configs.make_base_jobs import stage_round1, stage_round2
    src = stage_round1("S", "2.0", "0.5")
    f = tmp_path / "r1.json"
    f.write_text(json.dumps(src))
    name = "base_S_wd0p4_lr0p5_r1_ve0_tied"
    a = next(j for j in src["jobs"] if j["name"] == name)["args"]
    flags = {"--epochs", "--muon_wd", "--cooldown_frac", "--lr_scale"}
    for j in stage_round2(str(f), name)["jobs"]:
        b = j["args"]
        assert len(a) == len(b) and b[b.index("--tag") + 1] == j["name"]
        changed = {a[i - 1] for i, (x, y) in enumerate(zip(a, b)) if x != y} - {"--tag"}
        assert changed and changed <= flags and len(changed - {"--epochs"}) <= 1
