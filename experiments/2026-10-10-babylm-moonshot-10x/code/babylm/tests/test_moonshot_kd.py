"""CPU tests for the moonshot's logit-distillation lever (babylm.moonshot.kd) and the mini-ruler splits."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from babylm.moonshot import kd as KD


def test_kd_terms_alpha0_is_ce():
    torch.manual_seed(0)
    zs, zt = torch.randn(2, 5, 11), torch.randn(2, 5, 11)
    y = torch.randint(0, 11, (2, 5))
    y[0, 0] = -1
    loss, st = KD.kd_terms(zs, zt, y, alpha=0.0, tau=1.0)
    ref = F.cross_entropy(zs.view(-1, 11), y.view(-1), ignore_index=-1)
    assert torch.allclose(loss, ref, atol=1e-6)
    assert st["kd_kl"] == 0.0


def test_kd_terms_kl_zero_when_equal_and_positive_otherwise():
    torch.manual_seed(1)
    z = torch.randn(3, 4, 9)
    y = torch.randint(0, 9, (3, 4))
    _, st = KD.kd_terms(z, z.clone(), y, alpha=0.5, tau=2.0)
    assert abs(st["kd_kl"]) < 1e-6
    _, st2 = KD.kd_terms(z, torch.randn(3, 4, 9), y, alpha=0.5, tau=2.0)
    assert st2["kd_kl"] > 0


def test_kd_terms_teacher_vocab_prefix_and_exact_kl():
    torch.manual_seed(2)
    zs, zt = torch.randn(1, 3, 12), torch.randn(1, 3, 10)  # student padded vocab (MNTP pad ids)
    y = torch.randint(0, 10, (1, 3))
    loss, st = KD.kd_terms(zs, zt, y, alpha=1.0, tau=1.0)
    p = F.softmax(zt, -1)
    lq = F.log_softmax(zs, -1)[..., :10]
    kl = (p * (p.log() - lq)).sum(-1).mean()
    assert torch.allclose(loss, kl, atol=1e-5)
    with pytest.raises(ValueError):
        KD.kd_terms(torch.randn(1, 2, 5), torch.randn(1, 2, 6), torch.zeros(1, 2, dtype=torch.long), 0.5, 1.0)


def test_kd_gradient_pulls_student_toward_teacher():
    torch.manual_seed(3)
    zt = torch.randn(1, 6, 7)
    zs = torch.zeros(1, 6, 7, requires_grad=True)
    y = torch.full((1, 6), -1)
    y[0, 0] = 0  # one scored target keeps n >= 1
    opt = torch.optim.SGD([zs], lr=1.0)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss, st = KD.kd_terms(zs, zt, y, alpha=1.0, tau=1.0)
        loss.backward()
        opt.step()
        first = st["kd_kl"] if first is None else first
    assert st["kd_kl"] < 0.1 * first


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2] / "data/babylm/bin/train_10M.meta.json").exists(),
                    reason="needs prepared train_10M")
def test_mini_ruler_nested(tmp_path):
    from babylm.data import corpus as C
    from babylm.moonshot import mini_ruler as MR
    tr = C.load_split("train_10M")
    pcs = MR.pieces_of(np.asarray(tr.tokens)[:200_000], np.asarray(tr.words)[:200_000], 512)
    assert all(e - s <= 512 + 0 for s, e in pcs)
    assert pcs[0][0] == 0 and all(pcs[i][1] == pcs[i + 1][0] for i in range(len(pcs) - 1))
    a, b = C.load_split("train_1M"), C.load_split("train_2M")
    assert abs(a.n_words - 1_000_000) < 20_000 and abs(b.n_words - 2_000_000) < 20_000
    assert a.meta["parent_sha256"] == tr.meta["sha256"]


def _tiny_teacher(tmp_path, vocab, seed=0):
    from sprint1.harness.model_api import ModelConfig
    from sprint1.models import baseline
    cfg = ModelConfig(vocab_size=vocab, n_layer=2, n_head=2, n_embd=64, seq_len=64)
    torch.manual_seed(seed)
    ck = tmp_path / f"teacher{seed or ''}.pt"
    torch.save({"model": baseline.build(cfg).state_dict(), "cfg": cfg.to_dict(), "model_name": "baseline",
                "words_seen": 0, "label": "test"}, ck)
    return ck


@pytest.mark.parametrize("extra", [(), ("--mntp_frac", "0.25")])
def test_harness_kd_end_to_end_cpu(synth_dir, tmp_path, extra):
    """CPU steps of the tiny model with a tiny teacher (synthetic data): KD stats logged, config records the teacher;
    with MNTP the student pads its vocabulary and the teacher's is a prefix."""
    from babylm.data import corpus as C
    from babylm.harness import train as TR
    from babylm.tests.conftest import tiny_args
    ck = _tiny_teacher(tmp_path, C.VOCAB_SIZE)
    out = tmp_path / "run"
    r = TR.main(tiny_args(synth_dir, out, "--epochs", "1", "--steps", "3", "--no_full_dev", "--no_pt",
                          "--kd_teacher", str(ck), "--kd_alpha", "0.5", *extra))
    assert r["status"] == "ok"
    conf = json.loads((out / "config.json").read_text())
    assert conf["levers"]["kd"]["alpha"] == 0.5
    rows = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    tr = [x for x in rows if x.get("type") == "train"]
    assert tr and all(np.isfinite(x["kd_kl"]) and x["kd_kl"] >= 0 for x in tr)


def test_harness_init_from_branches(synth_dir, tmp_path):
    """--init_from: a child starts from the trunk's resume.pt (weights and optimizer moments) with its own steps,
    schedule and data seed, and records the trunk's words for the exposure ledger; a mismatched cfg is refused."""
    import torch
    from babylm.harness import train as TR
    from babylm.tests.conftest import tiny_args
    base = ("--epochs", "1", "--no_full_dev", "--no_pt", "--extra", "n_value_embeds=0", "--tie_embeddings")
    trunk = TR.main(tiny_args(synth_dir, tmp_path / "trunk", *base, "--steps", "3", "--cooldown_frac", "0",
                              "--save"))
    rp = tmp_path / "trunk" / "resume.pt"
    ck = torch.load(rp, weights_only=False)
    frozen = TR.main(tiny_args(synth_dir, tmp_path / "frozen", *base, "--steps", "1", "--lr_scale", "0", "--save",
                               "--init_from", str(rp), "--data_seed", "7"))
    info = frozen["config"]["levers"]["init_from"]
    assert info["words_seen"] == trunk["words_seen"] and info["optimizer_state"] and info["path"] == str(rp)
    got = torch.load(tmp_path / "frozen" / "resume.pt", weights_only=False)["model"]
    assert all(torch.equal(got[k], v) for k, v in ck["model"].items())  # lr 0: the child is the trunk
    child = TR.main(tiny_args(synth_dir, tmp_path / "child", *base, "--steps", "2", "--init_from", str(rp),
                              "--data_seed", "7"))
    fresh = TR.main(tiny_args(synth_dir, tmp_path / "fresh", *base, "--steps", "2", "--data_seed", "7"))
    assert child["status"] == "ok" and child["final_train_loss"] < fresh["final_train_loss"]
    assert child["words_seen"] < 2 * trunk["words_seen"] and fresh["config"]["levers"]["init_from"] is None
    with pytest.raises(SystemExit):
        TR.main(tiny_args(synth_dir, tmp_path / "bad", "--epochs", "1", "--steps", "1", "--no_full_dev", "--no_pt",
                          "--init_from", str(rp)))


def test_harness_kd_with_provisional_base_flags(synth_dir, tmp_path):
    """The provisional base's flag set (muon_wd 2, lr_scale 0.5, embed_wd, softcap, U-net) with KD from a teacher."""
    from babylm.data import corpus as C
    from babylm.harness import train as TR
    from babylm.tests.conftest import tiny_args
    ck = _tiny_teacher(tmp_path, C.VOCAB_SIZE)
    r = TR.main(tiny_args(synth_dir, tmp_path / "p0", "--epochs", "1", "--steps", "3", "--no_full_dev", "--no_pt",
                          "--muon_wd", "2.0", "--lr_scale", "0.5", "--embed_wd", "0.005", "--extra", "softcap=15",
                          "--extra", "unet=true", "--kd_teacher", str(ck), "--kd_alpha", "1.0"))
    assert r["status"] == "ok" and r["config"]["levers"]["kd"]["alpha"] == 1.0
    r = TR.main(tiny_args(synth_dir, tmp_path / "ve0_tied", "--epochs", "1", "--steps", "3", "--no_full_dev",
                          "--no_pt", "--muon_wd", "2.0", "--lr_scale", "0.5", "--extra", "n_value_embeds=0",
                          "--tie_embeddings", "--kd_teacher", str(ck), "--kd_alpha", "1.0"))
    assert r["status"] == "ok" and r["config"]["levers"]["kd"]["alpha"] == 1.0


def test_make_ceiling_jobs(tmp_path):
    from babylm.moonshot import make_ceiling_jobs as MJ
    base = {"name": "b", "module": "babylm.harness.train", "timeout_s": 100,
            "args": ["--size", "S", "--seed", "0", "--save", "--tag", "b"]}
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps({"jobs": [base]}))
    d = MJ.build(f"{f}:b", "floor_s1")
    names = [j["name"] for j in d["jobs"]]
    assert names[:5] == [f"ms_c3_b_s{s}" for s in MJ.C3_SEEDS] and len(names) == 7
    for j in d["jobs"]:
        a = j["args"]
        assert a.count("--seed") == 1 and a.count("--tag") == 1 and a[a.index("--tag") + 1] == j["name"]
    c1 = d["jobs"][-1]["args"]
    assert c1[c1.index("--kd_teacher") + 1] == MJ.T100_PATH and c1[c1.index("--kd_alpha") + 1] == "1.0"
    assert d["jobs"][5]["args"][d["jobs"][5]["args"].index("--kd_teacher") + 1].startswith("babylm/runs/floor_s1/")
    assert d["jobs"][5]["after"] == "floor_s1" and MJ.build(f"{f}:b")["jobs"][5]["after"] == "b_s1"


def _tiny_hf(tmp_path, seed):
    """A tiny baseline checkpoint exported to HF (the scorer's input), on the real 16k tokenizer."""
    from babylm.data import corpus as C
    from babylm.harness import export_hf as E
    from sprint1.harness.model_api import ModelConfig
    from sprint1.models import baseline
    cfg = ModelConfig(vocab_size=C.VOCAB_SIZE, n_layer=2, n_head=2, n_embd=32, seq_len=512)
    torch.manual_seed(seed)
    ck = tmp_path / f"m{seed}.pt"
    torch.save({"model": baseline.build(cfg).state_dict(), "cfg": cfg.to_dict(), "model_name": "baseline",
                "words_seen": 0, "label": "test"}, ck)
    out = tmp_path / f"hf{seed}"
    E.export_checkpoint(ck, out)
    return out


def _tiny_dev(tmp_path):
    d = tmp_path / "dev"
    d.mkdir()
    rng = np.random.default_rng(0)
    vocab = ["the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "home", "Mary", "said", "hello,", "world."]
    shared = "the cat sat on the mat and the dog ran home"   # >= 8 words: a line we mark as 'in train_100M'
    for name in ("b.dev", "a.dev"):
        lines = [" ".join(rng.choice(vocab, rng.integers(3, 15))) for _ in range(60)]
        lines[5] = shared
        lines[7] = "  " + lines[7].replace(" ", "\t", 1)  # whitespace the protocol collapses
        (d / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d, shared


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_dev_bpw_dump_matches_registered_scorer_and_c3(tmp_path, monkeypatch):
    import subprocess
    import sys
    from babylm.analysis import p2_slices
    from babylm.moonshot import ceiling_report as R
    from babylm.moonshot import dev_bpw_dump as DD
    dev, shared = _tiny_dev(tmp_path)
    dumps = []
    for seed in (0, 1, 2):
        hf = _tiny_hf(tmp_path, seed)
        ref = tmp_path / f"ref{seed}.json"
        subprocess.run([sys.executable, "babylm/eval/dev_bpw.py", str(hf), str(dev), "--slice", "full", "--ws",
                        "collapse", "--line_join", "newline", "--out", str(ref)], check=True, capture_output=True,
                       cwd=Path(__file__).resolve().parents[2])
        out = tmp_path / f"d{seed}.npz"
        DD.main([str(hf), str(dev), "--out", str(out), "--check", str(ref)])  # fails on any drift from the scorer
        d = R.Dump(out)
        assert d.files == ["a.dev", "b.dev"] and d.n_words == json.loads(ref.read_text())["words"]
        assert d.per_word().sum() == pytest.approx(d.nats.sum())
        dumps.append(d)
    hashes = np.array(sorted([p2_slices.line_key(shared.split())]), dtype=np.uint64)
    monkeypatch.setattr(p2_slices, "line_hashes", lambda split, data_dir: hashes)
    keep = R.keep_mask(dumps[0].files, dev, None)
    assert len(keep) == dumps[0].n_words and (~keep).sum() == 2 * len(shared.split())
    res = R.c3(dumps, keep)
    assert res["rows"][0]["g_npw_raw"] == pytest.approx(0.0, abs=1e-12)
    assert all(r["g_npw_raw"] > 0 for r in res["rows"][1:])  # Jensen: a mixture never loses to its mean member
    raw0 = json.loads((tmp_path / "ref0.json").read_text())["nats_per_word"]
    assert res["single"][0]["npw_raw"] == pytest.approx(raw0, rel=1e-5)
    c1 = R.c1(dumps[0], dumps[1], dumps[2], keep)
    assert set(c1["dev"]) == {"self", "oracle", "t100"} and "f1_npw_ovr" in c1


def test_ceiling_report_math():
    from babylm.moonshot import ceiling_report as R
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 3, 50)
    assert np.allclose(R.ensemble_tokens([x, x]), x)
    g_inf, b = R.fit_ginf([2, 4, 8], [0.1, 0.15, 0.175])
    assert g_inf == pytest.approx(0.2) and b == pytest.approx(0.2)
    assert len(R.subsets(8, 4)) == R.MAX_SUBSETS and len(R.subsets(4, 2)) == 6


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_ceiling_report_blank_line_events(tmp_path, monkeypatch):
    """--blank_tok zeroes exactly the blank-line newlines and the token after each blank line (critic ref-0020)."""
    from babylm.analysis import p2_slices
    from babylm.moonshot import ceiling_report as R
    from babylm.moonshot import dev_bpw_dump as DD
    d = tmp_path / "dev"
    d.mkdir()
    (d / "a.dev").write_text("the cat sat\n\nthe dog ran home\nhello world.\n\n\nMary said hello\n", encoding="utf-8")
    (d / "b.dev").write_text("a mat\nthe cat\n\nsat on a mat\n", encoding="utf-8")
    dumps = []
    for seed in (0, 1):
        hf = _tiny_hf(tmp_path, seed)
        DD.main([str(hf), str(d), "--out", str(tmp_path / f"d{seed}.npz")])
        dumps.append(R.Dump(tmp_path / f"d{seed}.npz"))
    ev = R.blank_events(dumps[0], d, str(tmp_path / "hf0"))
    # 4 blank newlines + 3 tokens after a blank line (the double blank line has one 'after' token)
    assert len(ev) == len(dumps[0].nats) and int(ev.sum()) == 4 + 3
    monkeypatch.setattr(p2_slices, "line_hashes", lambda split, data_dir: np.zeros(0, dtype=np.uint64))
    R.main(["c3", str(tmp_path / "d0.npz"), str(tmp_path / "d1.npz"), "--out", str(tmp_path / "rep"), "--dev_dir",
            str(d), "--blank_tok", str(tmp_path / "hf0")])
    res = json.loads((tmp_path / "rep" / "c3.json").read_text())
    be = res["blank_excluded"]
    assert be["events"] == 7 and be["single"][0]["npw_raw"] < res["single"][0]["npw_raw"]
    assert (tmp_path / "rep" / "c3.csv").read_text().count("nb,") == 2


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_ceiling_report_never_seen_char_guard(tmp_path, monkeypatch):
    """--guard is dev_bpw_twin's ex_unseen_chars (ref-0026) on twin dumps: equal to the scorer's json for a single run,
    (N - NU) / (W - WU) for mixtures and references alike."""
    import subprocess
    import sys
    from babylm.analysis import p2_slices
    from babylm.moonshot import ceiling_report as R
    from babylm.moonshot import dev_bpw_dump as DD
    d = tmp_path / "dev"
    d.mkdir()
    (d / "a.dev").write_text("the cat sat\n\nthe \u00abdog\u00bb ran home\nhello --> world.\n", encoding="utf-8")
    (d / "b.dev").write_text("a mat on\n\n\nthe cat \u2605 sat\n", encoding="utf-8")
    runs = []
    for seed in (0, 1, 2):
        hf, run = _tiny_hf(tmp_path, seed), tmp_path / f"run{seed}"
        run.mkdir()
        subprocess.run([sys.executable, "babylm/eval/dev_bpw_twin.py", str(hf), str(d), "--slice", "full", "--ws",
                        "collapse", "--line_join", "newline", "--out", str(run / "dev_bpw_twin.json")], check=True,
                       capture_output=True, cwd=Path(__file__).resolve().parents[2])
        DD.main([str(hf), str(d), "--out", str(run / "dev_v2.npz"), "--twin", "--check", str(run / "dev_bpw_twin.json")])
        runs.append(run)
    tw = json.loads((runs[0] / "dev_bpw_twin.json").read_text())
    assert tw["unseen_chars_words"] == 3  # '\u00abdog\u00bb', '-->' and '\u2605'
    monkeypatch.setattr(p2_slices, "line_hashes", lambda split, data_dir: np.zeros(0, dtype=np.uint64))
    R.main(["c3", str(runs[0] / "dev_v2.npz"), str(runs[1] / "dev_v2.npz"), "--out", str(tmp_path / "rep"),
            "--dev_dir", str(d), "--guard", str(tmp_path / "hf0"), "--ref", str(runs[2] / "dev_v2.npz")])
    res = json.loads((tmp_path / "rep" / "c3.json").read_text())
    g = res["never_seen_char_guard"]
    assert g["words"] == 3 and g["tokens"] == tw["unseen_chars_tokens"]
    assert len(g["twin_json_checks"]) == 2 and all(c["ok"] for c in g["twin_json_checks"])
    assert g["single"][0]["npw_raw"] == pytest.approx(tw["ex_unseen_chars_nats_per_word"], rel=1e-5)
    mask, wu = R.guard_tokens(R.Dump(runs[0] / "dev_v2.npz"), d, str(tmp_path / "hf0"))
    nats = [R.Dump(r / "dev_v2.npz").nats for r in runs]
    mix = R.ensemble_tokens(nats[:2])
    assert g["rows"][-1]["ens_npw_raw"] == pytest.approx(mix[~mask].sum() / (tw["words"] - wu))
    assert (tmp_path / "rep" / "c3.csv").read_text().count("g,") == 2
    for side in (res, g):  # --ref: the reference mean minus the 2-member mixture, on both legs
        v = side["vs_ref"]
        assert v["mix_npw_raw"] == side["rows"][-1]["ens_npw_raw"]
        assert v["gain_npw_raw"] == pytest.approx(v["ref_npw_raw"] - v["mix_npw_raw"])
    assert g["vs_ref"]["ref_npw_raw"] == pytest.approx(nats[2][~mask].sum() / (tw["words"] - wu))


def test_teacher_ensemble_and_perm(tmp_path):
    """Several teachers = their probability average (log p_bar); --kd_perm permutes positions, rows intact."""
    from babylm.harness.train import load_model_module
    from babylm.moonshot import kd as KD
    cks = [_tiny_teacher(tmp_path, 97, seed=i) for i in range(3)]
    ts = [KD.load_teacher(c, torch.device("cpu"), load_model_module)[0] for c in cks]
    ens, meta = KD.load_teacher(",".join(map(str, cks)), torch.device("cpu"), load_model_module)
    assert meta["k"] == 3 and len(meta["members"]) == 3
    x = torch.randint(0, 97, (2, 8))
    want = torch.log(torch.stack([F.softmax(t.logits(t.hidden(x)).float(), -1) for t in ts]).mean(0))
    got = KD.teacher_logits(ens, x)
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.equal(KD.teacher_logits(ts[0], x), ts[0].logits(ts[0].hidden(x)))  # single teacher unchanged
    pz = KD.teacher_logits(ens, x, torch.Generator().manual_seed(0))
    assert not torch.allclose(pz, got)
    a, b = got.reshape(16, -1), pz.reshape(16, -1)
    assert sorted(a.sum(-1).tolist()) == pytest.approx(sorted(b.sum(-1).tolist()))


def test_harness_multi_teacher_kd_and_perm(synth_dir, tmp_path):
    from babylm.data import corpus as C
    from babylm.harness import train as TR
    from babylm.tests.conftest import tiny_args
    t = [str(_tiny_teacher(tmp_path, C.VOCAB_SIZE, seed=i)) for i in range(2)]
    for name, extra in (("ens", []), ("perm", ["--kd_perm"])):
        r = TR.main(tiny_args(synth_dir, tmp_path / name, "--epochs", "1", "--steps", "3", "--no_full_dev", "--no_pt",
                              "--kd_teacher", ",".join(t), "--kd_alpha", "0.5", *extra))
        assert r["status"] == "ok" and r["config"]["levers"]["kd"]["teacher"]["k"] == 2
        assert r["config"]["levers"]["kd"]["perm"] == (name == "perm")


def test_make_f1_jobs(tmp_path):
    from babylm.moonshot import make_f1_jobs as MF
    base = {"name": "b", "module": "babylm.harness.train", "timeout_s": 100,
            "args": ["--model", "baseline", "--size", "S", "--split", "train_10M", "--seed", "0", "--tag", "b"]}
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps({"jobs": [base]}))
    d = MF.build(f"{f}:b")
    names = [j["name"] for j in d["jobs"]]
    assert names == [f"ms_f1_m{s}" for s in range(8)] + ["ms_f1_stu", "ms_f1_nsingle", "ms_f1_nperm", "ms_f1_FM"]
    by = {j["name"]: j for j in d["jobs"]}
    for j in d["jobs"]:
        a = j["args"]
        assert a[a.index("--split") + 1] == "train_1M" and a.count("--seed") == 1 and a.count("--tag") == 1
    stu = by["ms_f1_stu"]["args"]
    assert stu[stu.index("--kd_teacher") + 1].count(",") == 7 and by["ms_f1_stu"]["after"] == names[:8]
    assert "--kd_perm" in by["ms_f1_nperm"]["args"] and "--kd_perm" not in stu
    ns = by["ms_f1_nsingle"]["args"]
    assert ns[ns.index("--kd_teacher") + 1] == "babylm/runs/ms_f1_m1/checkpoints/chck_10M.pt"
    fm = by["ms_f1_FM"]["args"]
    assert fm[fm.index("--size") + 1] == "M" and "--kd_teacher" not in fm


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_dev_bpw_dump_twin_matches_registered_twin(tmp_path):
    """--twin reproduces metric v2 (babylm/eval/dev_bpw_twin.py) per token, words unchanged."""
    import subprocess
    import sys
    from babylm.moonshot import ceiling_report as R
    from babylm.moonshot import dev_bpw_dump as DD
    d = tmp_path / "dev"
    d.mkdir()
    (d / "a.dev").write_text("\nthe cat sat\n\nthe dog ran home\nhello world.\n\n\nMary said hello\n\n", encoding="utf-8")
    (d / "b.dev").write_text("a mat\nthe cat\n  \nsat on a mat\n", encoding="utf-8")
    hf = _tiny_hf(tmp_path, 0)
    ref = tmp_path / "twin.json"
    subprocess.run([sys.executable, "babylm/eval/dev_bpw_twin.py", str(hf), str(d), "--slice", "full", "--ws",
                    "collapse", "--line_join", "newline", "--out", str(ref)], check=True, capture_output=True,
                   cwd=Path(__file__).resolve().parents[2])
    DD.main([str(hf), str(d), "--out", str(tmp_path / "t.npz"), "--twin", "--check", str(ref)])
    dump = R.Dump(tmp_path / "t.npz")
    assert dump.n_words == json.loads(ref.read_text())["words"] == 20 and dump.protocol["twin"]
    with pytest.raises(SystemExit):
        R.blank_events(dump, d, str(hf))


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_dev_bpw_dump_temps(tmp_path):
    """--temps adds nats_temps / temps from the same forwards; nats (T = 1) is bit-identical to a run without it."""
    from babylm.moonshot import dev_bpw_dump as DD
    d = tmp_path / "dev"
    d.mkdir()
    (d / "a.dev").write_text("the cat sat\n\nthe dog ran home\nhello world.\n", encoding="utf-8")
    (d / "b.dev").write_text("a mat on\n\n\nthe cat sat\n", encoding="utf-8")
    hf = _tiny_hf(tmp_path, 0)
    DD.main([str(hf), str(d), "--out", str(tmp_path / "p.npz"), "--twin"])
    DD.main([str(hf), str(d), "--out", str(tmp_path / "t.npz"), "--twin", "--temps", "1,1.1,2"])
    p, t = np.load(tmp_path / "p.npz"), np.load(tmp_path / "t.npz")
    assert np.array_equal(p["nats"], t["nats"]) and "nats_temps" not in p.files
    nt = t["nats_temps"].astype(np.float64)
    assert t["nats_temps"].dtype == np.float16 and nt.shape == (len(t["nats"]), 3) and list(t["temps"]) == [1, 1.1, 2]
    assert np.allclose(nt[:, 0], t["nats"], rtol=2e-3, atol=2e-3)  # T = 1 through the temperature formula
    from babylm.data import corpus as C
    logv = np.log(C.VOCAB_SIZE)  # a hotter softmax sits nearer the uniform
    assert np.abs(t["nats"] - logv).mean() > np.abs(nt[:, 2] - logv).mean() > 0
    with pytest.raises(SystemExit):
        DD.main([str(hf), str(d), "--out", str(tmp_path / "x.npz"), "--temps", "0"])


def test_ceiling_report_calib_plumbing():
    """--calib: g(K) and vs_ref, raw and calibrated, from the pinned instrument's calibrate() on mixed nats."""
    import hashlib
    from types import SimpleNamespace
    from babylm.moonshot import ceiling_report as R
    assert hashlib.sha256(R.CALIB_TOOL.read_bytes()).hexdigest() == R.CALIB_SHA256
    rng = np.random.default_rng(1)
    mk = lambda i: SimpleNamespace(nats=rng.uniform(0.1, 4, 40), protocol={"twin": True}, path=f"d{i}")  # noqa: E731

    def calibrate(x, ids, words, ok, drop, S):
        leg = lambda v: {"raw": float(v), "calibrated": float(0.9 * v), "lam_even": 0.0, "lam_odd": 0.0}  # noqa: E731
        return {"full": leg(x.sum()), "guarded": leg(x[1:].sum())}
    cc = SimpleNamespace(shared=lambda: {}, tokenizer=lambda: None, calibrate=calibrate,
                         load_v2=lambda path, S, tok: (None, None, None, None, None, None))
    ds, ref = [mk(i) for i in range(3)], mk(9)
    out = R.calib_c3(ds, [ref], 5, cc=cc)
    mix2 = [R.ensemble_tokens([ds[i].nats, ds[j].nats]).sum() for i, j in R.subsets(3, 2)]
    mem2 = [(ds[i].nats.sum() + ds[j].nats.sum()) / 2 for i, j in R.subsets(3, 2)]
    r2 = out["rows"][1]
    assert out["rows"][0]["g_full_raw"] == pytest.approx(0.0)
    assert r2["g_full_raw"] == pytest.approx(np.mean(mem2) - np.mean(mix2))
    assert r2["g_full_calibrated"] == pytest.approx(0.9 * r2["g_full_raw"])
    full = R.ensemble_tokens([d.nats for d in ds])
    assert out["vs_ref"]["gain_full_raw"] == pytest.approx(ref.nats.sum() - full.sum())
    assert out["vs_ref"]["gain_guarded_raw"] == pytest.approx(ref.nats[1:].sum() - full[1:].sum())
    with pytest.raises(SystemExit):
        R.calib_c3([SimpleNamespace(nats=ds[0].nats, protocol={}, path="v1")], [], 5, cc=cc)


def test_make_split_jobs(tmp_path):
    """Legal-split jobs: summed member epochs <= 10; branch children read the trunk's resume.pt with new data seeds."""
    from babylm.moonshot import make_split_jobs as MS
    base = {"name": "b", "module": "babylm.harness.train", "timeout_s": 100,
            "args": ["--model", "baseline", "--size", "S", "--split", "train_10M", "--epochs", "10", "--seed", "0",
                     "--cooldown_frac", "0.5", "--save", "--resume", "auto", "--tag", "b"]}
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps({"jobs": [base]}))
    arg = lambda j, flag: j["args"][j["args"].index(flag) + 1]  # noqa: E731
    s3 = MS.build(f"{f}:b", "split", 3)
    assert [j["name"] for j in s3["jobs"]] == ["ms_ls_k3_s0", "ms_ls_k3_s1", "ms_ls_k3_s2"]
    assert all(arg(j, "--epochs") == "3.333333" and arg(j, "--cooldown_frac") == "0.5" for j in s3["jobs"])
    assert [arg(j, "--seed") for j in s3["jobs"]] == ["0", "1", "2"] and s3["summed_epochs"] <= 10
    br = MS.build(f"{f}:b", "branch", 2, trunk=4)
    t, c1, c2 = br["jobs"]
    assert arg(t, "--epochs") == "4" and arg(t, "--cooldown_frac") == "0" and t["args"].count("--save") == 1
    for c, seed in ((c1, "101"), (c2, "102")):
        assert arg(c, "--epochs") == "3" and arg(c, "--data_seed") == seed and arg(c, "--cooldown_frac") == "1"
        assert arg(c, "--init_from") == f"babylm/runs/{t['name']}/resume.pt" and c["after"] == [t["name"]]
        assert arg(c, "--muon_momentum_warmup_steps") == "0" and arg(c, "--tag") == c["name"]
    assert br["summed_epochs"] == 10
    assert MS.build(f"{f}:b", "branch", 3, trunk=4)["summed_epochs"] == 10
    for bad in (("split", 1), ("branch", 2, 10.0)):
        with pytest.raises(SystemExit):
            MS.build(f"{f}:b", *bad)


def test_ceiling_report_calib_shared_temperature():
    """With nats_temps, a mixture side is calibrate_t on members tempered at one shared T and then mixed."""
    from types import SimpleNamespace
    from babylm.moonshot import ceiling_report as R
    rng = np.random.default_rng(2)
    temps = [0.9, 0.95, 1.05, 1.1, 1.15, 1.2]

    def mk(i):
        cols = [rng.uniform(0.1, 4, 30) for _ in range(len(temps) + 1)]
        return SimpleNamespace(nats=cols[0], protocol={"twin": True}, path=f"d{i}", temps=temps,
                               nats_at=lambda c=cols: c)
    seen = []

    def calibrate(x, ids, words, ok, drop, S):
        leg = {"raw": float(x.sum()), "calibrated": float(x.sum()), "lam_even": 0.0, "lam_odd": 0.0}
        return {"full": dict(leg), "guarded": dict(leg)}

    def calibrate_t(x, xt, ts, ids, words, ok, drop, S):
        seen.append((x, xt))
        v = {"calibrated_t": float(xt[:, 3].sum()), "t_even": 1.1, "lam_t_even": 0.0, "t_odd": 1.1,
             "lam_t_odd": 0.0, "t_edge": False}
        return {"full": dict(v), "guarded": dict(v)}
    cc = SimpleNamespace(shared=lambda: {}, tokenizer=lambda: None, calibrate=calibrate, calibrate_t=calibrate_t,
                         load_v2=lambda path, S, tok: (None,) * 8)
    ds = [mk(i) for i in range(2)]
    out = R.calib_c3(ds, [], 5, cc=cc)
    x, xt = seen[-1]  # the K = 2 mixture is the last side calibrated
    assert np.allclose(x, R.ensemble_tokens([d.nats for d in ds]))
    for j in range(len(temps)):
        assert np.allclose(xt[:, j], R.ensemble_tokens([d.nats_at()[j + 1] for d in ds]))
    r2 = out["rows"][1]
    assert r2["g_full_calibrated_t"] == pytest.approx(np.mean([d.nats_at()[4].sum() for d in ds]) - xt[:, 3].sum())
    assert out["temps"] == temps and r2["t_full"] == [[1.1, 1.1, False]]


def test_pe_gate2_read(tmp_path, monkeypatch):
    """Card PE Gate 2: J = mean member - p_bar, rho = J(PE3) / J(I3), floor = SD of the I3 singles, (a) and (b)."""
    from babylm.moonshot import ceiling_report as R
    from babylm.moonshot import pe_gate2_read as G
    rng = np.random.default_rng(0)
    n, steps = 400, (rng.random(400) < 0.6).astype(np.int64)
    steps[0] = 0
    words = int(steps.sum()) + 1

    def dump(name, nats):
        p = tmp_path / f"{name}.npz"
        np.savez(p, nats=nats.astype(np.float32), word_step=steps, files=np.array(["a.dev"]),
                 file_words=np.array([words]), file_tokens=np.array([n]))
        return p

    base = rng.gamma(2.0, 1.5, n)
    pe = [dump(f"pe{k}", base + rng.normal(0, 0.8, n).clip(-1, 1)) for k in range(3)]
    i3 = [dump(f"i{k}", base + 0.05 * k + rng.normal(0, 1.0, n).clip(-1, 1)) for k in range(3)]
    mix = R.ensemble_tokens([R.Dump(p).nats for p in pe])
    own = dump("pe_own", mix)
    f = [dump("fdeep", base + 0.3), dump("fm", base + 0.2)]
    keep = np.ones(words, dtype=bool)
    r = G.read([R.Dump(p) for p in pe], R.Dump(own), [R.Dump(p) for p in i3], [R.Dump(p) for p in f], None, keep)
    one = lambda p: R.Dump(p).per_word().mean()  # noqa: E731
    assert r["J_pe3"] == pytest.approx(np.mean([one(p) for p in pe]) - one(own), abs=1e-5)
    j_i3 = np.mean([one(p) for p in i3]) - R.Dump(i3[0]).per_word(R.ensemble_tokens([R.Dump(p).nats for p in i3])).mean()
    assert r["J_i3"] == pytest.approx(j_i3) and r["rho"] == pytest.approx(r["J_pe3"] / j_i3)
    assert r["floor_1M"] == pytest.approx(np.std([one(p) for p in i3], ddof=1))
    assert r["pe3_mixture_check"] and r["better_F"] == pytest.approx(one(f[1]))
    assert r["a_killed"] == (one(own) > one(f[1]) - 2 * r["floor_1M"]) and r["b_killed"] == (r["rho"] < 0.5)
    from babylm.analysis import p2_slices
    monkeypatch.setattr(p2_slices, "line_hashes", lambda split, data_dir: np.zeros(0, dtype=np.uint64))
    d = tmp_path / "dev"
    d.mkdir()
    (d / "a.dev").write_text(" ".join(["w"] * words) + "\n")
    fj = tmp_path / "fm_twin.json"  # an F run known only by its twin json joins the F set on the full leg
    fj.write_text(json.dumps({"nats_per_word": one(own) - 1.0, "ex_unseen_chars_nats_per_word": 0.0}))
    r2 = G.read([R.Dump(p) for p in pe], R.Dump(own), [R.Dump(p) for p in i3], [R.Dump(f[0])], None, keep,
                f_values={str(fj): one(own) - 1.0})
    assert r2["better_F"] == pytest.approx(one(own) - 1.0) and r2["a_killed"]
    with pytest.raises(SystemExit, match="1e-4"):  # an own dump that is not the member mixture is refused
        G.main(["--pe3", *map(str, pe), "--pe3_ens", str(f[0]), "--i3", *map(str, i3), "--f", *map(str, f),
                "--f_json", str(fj), "--out", str(tmp_path / "o"), "--dev_dir", str(d)])


def test_f1_read(tmp_path):
    """F1 stage 1: G1 = DEV(m0) - DEV(base@10M), phi = (DEV(m0) - DEV(x)) / G1, kill and build rules."""
    from babylm.moonshot import ceiling_report as R
    from babylm.moonshot import f1_read as F
    rng = np.random.default_rng(1)
    n, steps = 300, (rng.random(300) < 0.6).astype(np.int64)
    steps[0] = 0
    words = int(steps.sum()) + 1

    def dump(name, nats):
        p = tmp_path / f"{name}.npz"
        np.savez(p, nats=nats.astype(np.float32), word_step=steps, files=np.array(["a.dev"]),
                 file_words=np.array([words]), file_tokens=np.array([n]))
        return R.Dump(p)

    base = rng.gamma(2.0, 1.5, n) + 1.0
    mem = [dump(f"m{k}", base + rng.normal(0, 0.5, n).clip(-0.9, 0.9)) for k in range(8)]
    one = lambda d: d.per_word().mean()  # noqa: E731
    b10 = [dump("b10", base - 1.0)]
    g1 = one(mem[0]) - one(b10[0])
    comp = {"nsingle": dump("ns", base - 0.10), "nperm": dump("np", base - 0.05), "fm": dump("fm", base)}
    keep = np.ones(words, dtype=bool)
    for shift, killed, build in ((0.45, False, True), (0.25, False, False), (0.15, True, False)):
        stu = dump("stu", R.Dump(mem[0].path).nats - shift * g1 * words / n)
        r = F.read(mem, stu, comp, b10, keep)
        assert r["G1"] == pytest.approx(g1) and r["phi_stu"] == pytest.approx(shift, abs=1e-4)
        assert r["floor_1M"] == pytest.approx(np.std([one(d) for d in mem], ddof=1))
        assert r["g8"] > 0 and r["phi_ens8"] == pytest.approx((one(mem[0]) - r["ens8"]) / g1)
        ahead_ok = all(one(c) - one(stu) >= 2 * r["floor_1M"] for c in comp.values())
        assert r["killed"] == (killed or not ahead_ok) and r["build_10M_student"] == (build and ahead_ok)


def test_make_menu_jobs(tmp_path):
    """R4.4 menu: the card's FLOP ratios, inheritance from larger F, the registered E grid, buildable shapes."""
    from babylm.moonshot import make_menu_jobs as M
    from sprint1.harness.model_api import ModelConfig
    from sprint1.models import baseline
    x = {(L, d): round(M.flops(L, d) / M.BASE_FLOPS, 3) for g, es in M.MENU.items() for _, L, d, _ in es}
    assert x == {(28, 384): 2.073, (12, 592): 2.092, (35, 384): 2.543, (12, 672): 2.614, (42, 384): 3.012,
                 (12, 736): 3.074, (60, 384): 4.22, (12, 880): 4.241, (36, 512): 4.358} \
        and M.flops(12, 384) == M.BASE_FLOPS
    assert round(2 * M.flops(28, 384) / M.BASE_FLOPS, 3) == 4.146  # the 28L legal split's F
    assert [e[1:3] for e in M.entries(3.0)] == [(42, 384), (12, 736), (60, 384), (12, 880), (36, 512), (12, 768)]
    assert [e[1:3] for e in M.entries(4.146)] == [(60, 384), (12, 880), (36, 512)]  # M (3.317x) is below F
    assert len(M.entries(2.0)) == 10 and len(M.entries(2.463)) == 8
    for _, L, d, H in M.entries(2.0):  # every shape builds (head_dim even, d divisible by heads)
        baseline.build(ModelConfig(vocab_size=64, n_layer=1, n_head=H, n_embd=d, seq_len=16))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"jobs": [{"name": "b", "module": "babylm.harness.train", "timeout_s": 10,
                                          "args": ["--size", "S", "--split", "train_10M", "--epochs", "10",
                                                   "--seed", "3", "--muon_wd", "0.4", "--tag", "b"]}]}))
    d = M.build(f"{base}:b", 2.0, [4, 5], only={"28L384d", "12L592d"})
    assert [j["name"] for j in d["jobs"]] == ["ms_menu_28L384d_e4_wd0p4", "ms_menu_28L384d_e5_wd0p4",
                                             "ms_menu_12L592d_e4_wd0p4", "ms_menu_12L592d_e5_wd0p4"]
    a = d["jobs"][2]["args"]
    for flag, v in (("--n_layer", "12"), ("--n_embd", "592"), ("--n_head", "8"), ("--epochs", "4"), ("--seed", "0"),
                    ("--muon_wd", "0.4"), ("--tag", "ms_menu_12L592d_e4_wd0p4")):
        assert a[a.index(flag) + 1] == v
    assert M.build(f"{base}:b", 2.0, [5], wd=0.2)["jobs"][0]["name"].endswith("_wd0p2")
    r = M.build(f"{base}:b", 2.0, [5], only={"28L384d"}, seeds=[2, 3])["jobs"]  # claim replicates: _s<seed>, own seed
    assert [j["name"] for j in r] == ["ms_menu_28L384d_e5_wd0p4_s2", "ms_menu_28L384d_e5_wd0p4_s3"]
    assert [j["args"][j["args"].index("--seed") + 1] for j in r] == ["2", "3"]
    assert [j["args"][j["args"].index("--tag") + 1] for j in r] == [j["name"] for j in r]
    with pytest.raises(SystemExit, match="grid"):
        M.build(f"{base}:b", 2.0, [3])



def test_a5_yardstick_report():
    """A5 (critic ref-0354): cal_t Y and gains on both views, df-2 gain interval through 10 ** (g / Y), headline."""
    from scipy import stats
    from babylm.moonshot import a5_yardstick as A

    def rd(ct, raw=None, d_ovr=0.01):
        return {v: {l: {"raw": (raw if raw is not None else ct) + (d_ovr if v == "ovr" else 0),
                        "cal_t": None if ct is None else ct + (d_ovr if v == "ovr" else 0)} for l in A.LEGS}
                for v in A.VIEWS}
    db = {("p0",): rd(4.96), ("p1",): rd(4.98), ("p2",): rd(4.97), ("t100",): rd(4.27),
          ("a", "b"): rd(4.86), ("c", "d"): rd(4.88), ("e", "f"): rd(4.87), ("m",): rd(4.94)}
    cfg = {"y": {"pool": ["p0", "p1", "p2"], "t100": "t100"},
           "gains": {"same": {"arm": [["a", "b"], ["c", "d"], ["e", "f"]], "ref": ["p0", "p1", "p2"]},
                     "menu": {"arm": [["a", "b"], ["c", "d"], ["e", "f"]], "ref": ["m"], "ref_sd": ["p0", "p1", "p2"],
                              "labels": ["Y from S only"]}}}
    out = A.report(cfg, lambda members: db[tuple(members)])
    y = out["Y"]["all/full"]
    assert y["Y"] == pytest.approx([0.7]) and not y["provisional"]
    assert y["Y_ci95_halfwidth"] == pytest.approx(stats.t.ppf(0.975, 2) * 0.01 * (4 / 3) ** 0.5)
    same = out["gains"]["same"]["rows"]["all/full"]
    assert same["g"] == pytest.approx(0.1) and same["se"] == pytest.approx((0.01 ** 2 / 3 * 2) ** 0.5)
    h = stats.t.ppf(0.975, 2) * same["se"]
    assert same["x"] == pytest.approx([10 ** (0.1 / 0.7)])
    assert same["x_ci95"] == pytest.approx([10 ** ((0.1 - h) / 0.7), 10 ** ((0.1 + h) / 0.7)])
    menu = out["gains"]["menu"]
    assert menu["rows"]["all/full"]["g"] == pytest.approx(0.07)
    assert menu["rows"]["all/full"]["se"] == pytest.approx((0.01 ** 2 / 3 + 0.01 ** 2) ** 0.5)  # sd_r from ref_sd
    db.update({("q0",): rd(4.940), ("q1",): rd(4.941), ("q2",): rd(4.942)})  # a tighter size pool: S E6 SD floors it
    floored = A.report(dict(cfg, gains={"m": dict(cfg["gains"]["menu"], ref_sd=["q0", "q1", "q2"])}),
                       lambda members: db[tuple(members)])["gains"]["m"]["rows"]["all/full"]
    assert floored["ref_sd"] == pytest.approx(0.01)
    assert menu["headline_x_lower"]["full"] == pytest.approx(min(menu["rows"][f"{v}/full"]["x_ci95"][0] for v in A.VIEWS))
    assert menu["labels"][:2] == [*A.LABELS, A.T100_UNTUNED] and menu["labels"][-1] == "Y from S only"
    # T100 without temps: Y bracketed by its tempering gain in [0, the pool's (0.02)], labelled provisional
    db[("t100",)] = rd(None, raw=4.29)
    b = A.report({"y": cfg["y"]}, lambda members: db[tuple(members)])
    assert b["Y"]["all/guarded"]["Y"] == pytest.approx([0.68, 0.68]) and b["labels"]
    db[("p0",)], db[("p1",)], db[("p2",)] = rd(4.96, 4.98), rd(4.98, 5.0), rd(4.97, 4.99)
    b = A.report({"y": cfg["y"]}, lambda members: db[tuple(members)])
    assert b["Y"]["all/full"]["Y"] == pytest.approx([0.68, 0.7]) and b["Y"]["all/full"]["provisional"]
    # three T100 seeds: Y over their mean, Welch-style SE at df n_p + n_t - 2
    db.update({("u0",): rd(4.26), ("u1",): rd(4.28), ("u2",): rd(4.27)})
    y3 = A.report({"y": {"pool": cfg["y"]["pool"], "t100": ["u0", "u1", "u2"]}},
                  lambda members: db[tuple(members)])["Y"]["all/full"]
    assert y3["Y"] == pytest.approx([0.7]) and y3["Y_ci95_halfwidth"] == pytest.approx(  # equal SDs, n: Welch df 4
        stats.t.ppf(0.975, 4) * (0.01 ** 2 / 3 * 2) ** 0.5)


def test_a5_reader_mixture_and_overlap_view():
    """Reader: a mixture is tempered at one shared T then mixed; the ovr view masks words and shrinks W on both legs."""
    from types import SimpleNamespace
    from babylm.moonshot import a5_yardstick as A
    from babylm.moonshot import ceiling_report as R
    rng = np.random.default_rng(5)
    temps = [0.9, 0.95, 1.05, 1.1, 1.15, 1.2]
    words = np.array([0, 0, 1, 2, 2, 3, 4, 5])
    ids, drop = np.arange(8), np.array([0, 0, 0, 1, 1, 0, 0, 0], bool)
    dumps = {p: (rng.uniform(0.1, 4, 8), rng.uniform(0.1, 4, (8, 6))) for p in ("a", "b")}
    S = {"W": 6, "bad": np.array([0, 0, 1, 0, 0, 0], bool)}
    load = lambda p, S_, tok: (dumps[p][0], ids, words, np.ones(8, bool), drop, None, dumps[p][1], temps)  # noqa: E731

    def calibrate(x, ids_, w, ok, dr, s):
        return {"full": {"raw": x[ok].sum() / s["W"]}, "guarded": {"raw": x[ok & ~dr].sum() / (s["W"] - s["bad"].sum())}}

    def calibrate_t(x, xt, ts, ids_, w, ok, dr, s):
        leg = lambda m, den: {"calibrated_t": xt[m, 3].sum() / den, "t_even": 1.1, "t_odd": 1.1, "t_edge": False}  # noqa: E731
        return {"full": leg(ok, s["W"]), "guarded": leg(ok & ~dr, s["W"] - s["bad"].sum())}
    cc = SimpleNamespace(shared=lambda: S, tokenizer=lambda: None, load_v2=load, calibrate=calibrate,
                         calibrate_t=calibrate_t)
    keep = np.array([1, 1, 1, 0, 1, 1], bool)  # word 3 dropped by the overlap rule
    r = A.Reader(cc, keep)(["a", "b"])
    mt = R.ensemble_tokens([dumps["a"][1][:, 3], dumps["b"][1][:, 3]])
    mx = R.ensemble_tokens([dumps["a"][0], dumps["b"][0]])
    assert r["all"]["full"]["cal_t"] == pytest.approx(mt.sum() / 6)
    kt = keep[words]
    assert r["ovr"]["full"]["cal_t"] == pytest.approx(mt[kt].sum() / 5)
    assert r["ovr"]["guarded"]["raw"] == pytest.approx(mx[kt & ~drop].sum() / 4)
    assert r["ovr"]["guarded"]["t"] == [1.1, 1.1, False]
