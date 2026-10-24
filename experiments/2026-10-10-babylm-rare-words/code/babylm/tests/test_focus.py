"""CPU tests for the A5 rare-word loss-focusing lever: the mask builder (babylm.data.focus_mask), the pure keep
rule (babylm.harness.levers) and the harness wiring (synthetic data from conftest, tiny model)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from babylm.data import corpus as C
from babylm.data import focus_mask as FM
from babylm.harness import levers as L
from babylm.harness import train as TR
from babylm.tests.conftest import tiny_args


def test_normal_form():
    assert FM.normal_form("Hello,") == "hello" and FM.normal_form("“No!”") == "no"
    assert FM.normal_form("don't") == "don't" and FM.normal_form("--") == "--" and FM.normal_form("A.B.") == "a.b"


def test_build_focus_words_rare_and_aftermath_within_document():
    docs = [["the", "cat", "Zyx", "sat", "on", "the", "mat"], ["The", "cat", "sat", "on", "the", "mat.", "qux."]]
    counts = FM.word_counts(docs)
    assert counts["the"] == 4 and counts["cat"] == 2 and counts["mat"] == 2 and counts["zyx"] == 1 and counts["qux"] == 1
    focus, rare = FM.build_focus_words(docs, counts, count_max=1, after=2)
    assert rare.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # Zyx + the 2 words after it; qux. is the last word of its document: the aftermath never crosses into the next
    assert focus.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    f0, _ = FM.build_focus_words(docs, counts, count_max=1, after=0)
    assert np.array_equal(f0, rare)
    f_all, _ = FM.build_focus_words(docs, counts, count_max=4, after=0)
    assert f_all.all()


def test_focus_keep_rule():
    g = torch.Generator().manual_seed(0)
    fo = torch.zeros(64, 256, dtype=torch.bool)
    fo[:, ::4] = True  # 25% focus
    real = torch.ones_like(fo)
    real[0, 200:] = False
    k1 = L.focus_keep(fo, real, 1.0, g)
    assert torch.equal(k1, real)
    k0 = L.focus_keep(fo, real, 0.0, g)
    assert torch.equal(k0, fo & real)
    k = L.focus_keep(fo, real, 0.25, g)
    assert (k | ~fo | ~real).all() and not k[0, 200:].any()  # real focus targets always kept, padding never
    share = k.float().sum() / real.float().sum()
    assert abs(share - L.focus_expected_keep(0.25, 0.25)) < 0.02
    kn = L.focus_keep(fo, real, 0.25, g, null_share=L.focus_expected_keep(0.25, 0.25))
    assert abs(kn.float().sum() / real.float().sum() - share) < 0.02
    assert not (kn | ~fo | ~real).all()  # the twin drops focus targets too
    assert L.parse_focus("7:0.25", 10) == (7, 0.25) and L.parse_focus("1:1", 2.0) == (1, 1.0)
    for bad in ("0:0.5", "11:0.5", "7:1.5", "7", "a:b"):
        with pytest.raises(ValueError):
            L.parse_focus(bad, 10)


def test_focus_gather_matches_targets():
    mask = (np.arange(1000) % 7 == 0).astype(np.uint8)
    starts = np.array([3, 500, 900])
    fo = L.focus_gather(mask, starts, 64)
    for r, s in enumerate(starts):
        assert np.array_equal(fo[r].numpy(), mask[s + 1:s + 65].astype(bool))


def _synth_doc_words(synth_dir: Path, split: str) -> list[list[str]]:
    docs = json.loads((synth_dir / "docs.json").read_text())[split]
    return [t.split() for s in C.SOURCES for t in docs[s]]


def _count_max(doc_words) -> int:
    counts = FM.word_counts(doc_words)
    occ = sorted(counts[FM.normal_form(w)] for ws in doc_words for w in ws)
    return occ[len(occ) // 3]  # about a third of the words are "rare"


def test_focus_mask_build_and_load(synth_dir):
    dw = _synth_doc_words(synth_dir, "train_10M")
    cm = _count_max(dw)
    meta = FM.build(synth_dir, "train_10M", count_max=cm, after=3, tag="focus_test", doc_words=dw)
    mask, meta2 = FM.load_focus(synth_dir, "train_10M", "focus_test")
    assert meta2 == meta and meta["count_max"] == cm and meta["after"] == 3
    d = C.load_split("train_10M", synth_dir)
    assert len(mask) == d.n_tokens
    assert 0.05 < meta["rare_token_share"] < meta["focus_token_share"] < 0.95
    assert mask[np.asarray(d.tokens) == C.BOS_ID].max() == 0
    # every token of a rare word is flagged; a token of a frequent word far from any rare word is not
    words = [w for ws in dw for w in ws]
    counts = FM.word_counts(dw)
    wid = np.asarray(d.words)
    rare_w = np.array([counts[FM.normal_form(w)] <= cm for w in words])
    tok_rare = np.zeros(len(wid), bool)
    ok = wid < len(words)
    tok_rare[ok] = rare_w[wid[ok]]
    tok_rare &= np.asarray(d.tokens) != C.BOS_ID
    assert (mask[tok_rare] == 1).all()
    far = np.ones(len(words), bool)
    for dd in range(0, 4):
        far[dd:] &= ~rare_w[:len(rare_w) - dd] if dd else ~rare_w
    tok_far = np.zeros(len(wid), bool)
    tok_far[ok] = far[wid[ok]]
    assert (mask[tok_far] == 0).all()
    # raw-file path: write the documents back as prepare.py lays them out and rebuild from the files
    docs = json.loads((synth_dir / "docs.json").read_text())["train_10M"]
    for s in C.SOURCES:
        p = C.raw_path(synth_dir, "train_10M", s)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n\n".join(docs[s]) + "\n", encoding="utf-8")
    assert FM.split_doc_words(synth_dir, "train_10M") == dw
    meta3 = FM.build(synth_dir, "train_10M", count_max=cm, after=3, tag="focus_test2")
    assert meta3["sha256"] == meta["sha256"]
    with pytest.raises(FileNotFoundError):
        FM.load_focus(synth_dir, "train_10M", "no_such_mask")


def _train_recs(out: Path) -> list[dict]:
    return [r for r in (json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()) if r.get("type") == "train"]


def test_harness_focus_plan(synth_dir, tmp_path):
    """Epoch 1 identical to the base path; from epoch 2 the kept share is the expected one; exposure counters are
    unchanged; the null twin keeps the same share without the mask; MNTP rows keep their targets; resume restores
    the focus draws."""
    dw = _synth_doc_words(synth_dir, "train_10M")
    cm = _count_max(dw)
    FM.build(synth_dir, "train_10M", count_max=cm, after=3, tag="focus", doc_words=dw)
    common = ("--epochs", "2", "--no_full_dev", "--no_pt")
    r0 = TR.main(tiny_args(synth_dir, tmp_path / "base", *common))
    r1 = TR.main(tiny_args(synth_dir, tmp_path / "focus", *common, "--focus_plan", "2:0.25"))
    r2 = TR.main(tiny_args(synth_dir, tmp_path / "null", *common, "--focus_plan", "2:0.25", "--focus_null"))
    assert r0["config"]["levers"].get("focus") is None
    lv = r1["config"]["levers"]["focus"]
    assert lv["start_epoch"] == 2 and lv["beta"] == 0.25 and lv["null"] is False and lv["count_max"] == cm
    q = L.focus_expected_keep(lv["focus_token_share"], 0.25)
    assert lv["expected_kept_share_arm"] == pytest.approx(q)
    assert r2["config"]["levers"]["focus"]["null"] is True
    assert (r1["tokens_seen"], r1["words_seen"]) == (r0["tokens_seen"], r0["words_seen"])  # exposure unchanged
    assert (r2["tokens_seen"], r2["words_seen"]) == (r0["tokens_seen"], r0["words_seen"])
    t0, t1, t2 = _train_recs(tmp_path / "base"), _train_recs(tmp_path / "focus"), _train_recs(tmp_path / "null")
    assert len(t0) == len(t1) == len(t2) >= 6
    N = C.load_split("train_10M", synth_dir).n_tokens
    e1 = [i for i, r in enumerate(t1) if r["tokens_seen"] - 1024 < N]  # steps that started inside epoch 1
    e2 = [i for i, r in enumerate(t1) if r["tokens_seen"] - 1024 >= N]
    assert len(e1) >= 2 and len(e2) >= 2
    assert all(t1[i]["focus_kept"] == 1.0 for i in e1) and all(t1[i]["loss"] == t0[i]["loss"] for i in e1)
    k1 = np.array([t1[i]["focus_kept"] for i in e2])
    k2 = np.array([t2[i]["focus_kept"] for i in e2])
    assert abs(k1.mean() - q) < 0.03 and abs(k2.mean() - q) < 0.03
    assert (k1 < 1.0).all() and any(t1[i]["loss"] != t0[i]["loss"] for i in e2)
    # resume: a run paused inside the focused epochs continues with the same draws and the same losses
    out = tmp_path / "resumed"
    rp = TR.main(tiny_args(synth_dir, out, *common, "--focus_plan", "2:0.25", "--max_session_steps", str(e2[0] + 1)))
    assert rp["status"] == "paused"
    rr = TR.main(tiny_args(synth_dir, out, *common, "--focus_plan", "2:0.25", "--resume", "auto"))
    assert rr["status"] == "ok"
    tr = _train_recs(out)
    assert [r["focus_kept"] for r in tr] == [r["focus_kept"] for r in t1]
    assert [r["loss"] for r in tr] == pytest.approx([r["loss"] for r in t1], rel=1e-4)
    # MNTP hybrid: causal rows are focused, masked rows keep their targets, everything stays finite
    rm = TR.main(tiny_args(synth_dir, tmp_path / "mntp", *common, "--focus_plan", "1:0.25", "--mntp_frac", "0.25"))
    tm = _train_recs(tmp_path / "mntp")
    assert rm["status"] == "ok" and all(np.isfinite(r["mntp_loss"]) and 0 < r["focus_kept"] < 1 for r in tm)
    # a plan path (line windows) gathers the mask through target positions
    rl = TR.main(tiny_args(synth_dir, tmp_path / "line", *common, "--focus_plan", "1:0.25", "--window_plan", "line:64"))
    tl = _train_recs(tmp_path / "line")
    assert rl["status"] == "ok" and abs(np.mean([r["focus_kept"] for r in tl]) - q) < 0.05


def test_focus_plan_refuses_without_mask(synth_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        TR.main(tiny_args(synth_dir, tmp_path / "x", "--epochs", "1", "--steps", "1", "--no_full_dev", "--no_pt",
                          "--focus_plan", "1:0.5", "--focus_mask", "missing_mask"))
