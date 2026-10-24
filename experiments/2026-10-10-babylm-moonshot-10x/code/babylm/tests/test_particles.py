"""Shared-trunk particle ensemble (babylm/models/particles.py; moonshot card PE): CPU tests from the card's list."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from babylm.models import particles as PM
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

HAS_HF = importlib.util.find_spec("transformers") is not None


def _cfg(n_layer=8, d=32, V=97, T=16, **extra):
    return ModelConfig(vocab_size=V, n_layer=n_layer, n_head=2, n_embd=d, seq_len=T, tie_embeddings=True,
                       extra={"n_value_embeds": 0, **extra})


def _build(cfg, seed=0, mod=PM, jitter=False):
    """jitter: move every parameter (the zero-initialised output projections included) by a fixed small draw, so
    particles are different functions, as they are after the first optimizer step."""
    torch.manual_seed(seed)
    m = mod.build(cfg)
    if jitter:
        g = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


def _batch(V=97, B=2, T=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (B, T), generator=g)
    y = torch.randint(0, V, (B, T), generator=g)
    y[0, :3] = -1
    return x, y


@pytest.mark.parametrize("ls", [4, 8])
@pytest.mark.parametrize("ve", [0, 2])
def test_k1_is_the_port_gpt_bit_for_bit(ls, ve):
    cfg = _cfg(pe_k=1, pe_shared=ls)
    cfg.extra["n_value_embeds"] = ve
    ref, pe = _build(cfg, mod=baseline), _build(cfg)
    sa, sb = ref.state_dict(), pe.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    x, y = _batch()
    for train in (True, False):
        ref.train(train), pe.train(train)
        (la, aa), (lb, ab) = ref(x, y), pe(x, y)
        assert torch.equal(la, lb)
        if not train:
            assert torch.equal(aa["per_token_loss"], ab["per_token_loss"])
    assert pe.flops_per_token() == ref.flops_per_token()
    assert torch.equal(pe.hidden(x), ref.hidden(x))
    assert not hasattr(pe, "make_optimizer")


def test_flops_ledger_size_s():
    """The card's FLOP numbers at size S (12L/384d, T 1024, V 16384): base 193,462,272; every K = 3 / Ls = 4 variant
    (PE3, NJ, PE3w and the tied twin) 476,577,792 = 2.463x, counted from executed matmuls, not parameters."""
    with torch.device("meta"):
        base = baseline.build(_cfg(12, 384, 16384, 1024))
        assert base.flops_per_token() == 193_462_272
        for extra in ({}, {"pe_objective": "mixture"}, {"pe_private_wd_scale": 0.5}, {"pe_tied": True}):
            m = PM.build(_cfg(12, 384, 16384, 1024, pe_k=3, pe_shared=4, **extra))
            assert m.flops_per_token() == 476_577_792
        assert round(476_577_792 / 193_462_272, 3) == 2.463
        tied = PM.build(_cfg(12, 384, 16384, 1024, pe_k=3, pe_shared=4, pe_tied=True))
        assert sum(p.numel() for p in tied.parameters()) == sum(p.numel() for p in base.parameters())


def test_shared_block_bounds():
    for bad in (0, 3, 5, 8):  # Ls < 4 needs a critic entry; Ls > L - n_enc would share a decoder block
        with pytest.raises(ValueError):
            PM.build(_cfg(8, pe_k=3, pe_shared=bad))
    with pytest.raises(ValueError):
        PM.build(_cfg(8, pe_k=3, pe_objective="bootstrap"))
    PM.build(_cfg(8, pe_k=1, pe_shared=0))  # K = 1: the port model, Ls unused


def test_mixture_loss_and_objectives():
    cfg = _cfg(pe_k=3, pe_shared=4)
    m = _build(cfg, jitter=True)
    x, y = _batch()
    m.eval()
    loss, aux = m(x, y)
    pt = aux["per_token_loss"]
    lp = m.mix_logprobs(x)  # log p_bar, normalised
    assert torch.allclose(lp.exp().sum(-1), torch.ones(()), atol=1e-5)
    valid = y >= 0
    want = -lp.gather(-1, y.clamp_min(0)[..., None]).squeeze(-1)
    assert torch.allclose(pt[valid], want[valid], atol=1e-5) and bool((pt[~valid] == 0).all())
    assert torch.allclose(loss, pt[valid].mean(), atol=1e-6)
    zs = [m.logits(h) for h in m.particle_hiddens(x)]
    members = torch.stack([F.cross_entropy(z.view(-1, z.size(-1)), y.view(-1), ignore_index=-1) for z in zs])
    assert loss < members.mean()  # Jensen: the mixture never loses to its mean member
    m.train()
    lpe, apx = m(x, y)
    assert torch.allclose(lpe, members.mean(), atol=1e-6) and torch.allclose(apx["pe_mix_ce"], loss, atol=1e-6)
    nj = _build(_cfg(pe_k=3, pe_shared=4, pe_objective="mixture"), jitter=True).train()
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), nj.state_dict().values()))
    lnj, _ = nj(x, y)
    assert torch.allclose(lnj, loss, atol=1e-6)  # NJ trains on the mixture CE: same net, only the loss differs


def test_particles_share_the_trunk_and_differ_after_it():
    assert len(set(map(float, _build(_cfg(pe_k=3, pe_shared=4)).particle_hiddens(_batch()[0]).std((1, 2, 3))))) == 1, \
        "at init (zero output projections) every particle is the same function"
    m = _build(_cfg(pe_k=3, pe_shared=4), jitter=True)
    x, _ = _batch()
    hs = m.particle_hiddens(x)
    assert hs.shape == (3, 2, 16, 32)
    assert not torch.allclose(hs[0], hs[1]) and not torch.allclose(hs[1], hs[2])
    # particle 0 is the port GPT built from the same draws: same function as baseline with those weights
    ref = baseline.build(_cfg())
    ref.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith(("pblocks", "pskip"))})
    assert torch.allclose(hs[0], ref.hidden(x), atol=1e-6)
    # gradients: a shared block gets a gradient from every particle; a private block only from its own
    m.train()
    torch.stack([m.logits(h).sum() for h in m.particle_hiddens(x)])[1].backward()
    assert m.blocks[0].attn.c_q.weight.grad.abs().sum() > 0
    assert m.pblocks[0][0].mlp.c_fc.weight.grad.abs().sum() > 0
    for other in (m.blocks[4], m.pblocks[1][0]):  # particles 0 and 2
        assert other.mlp.c_fc.weight.grad is None or not other.mlp.c_fc.weight.grad.any()


def test_tied_twin_is_one_function():
    t = _build(_cfg(pe_k=3, pe_shared=4, pe_tied=True), jitter=True)
    base = _build(_cfg(), mod=baseline)
    base.load_state_dict(t.state_dict())
    x, y = _batch()
    hs = t.particle_hiddens(x)
    assert torch.equal(hs[0], hs[1]) and torch.equal(hs[0], hs[2])
    assert torch.allclose(t.mix_logprobs(x), F.log_softmax(base.logits(base.hidden(x)), -1), atol=1e-5)
    t.eval(), base.eval()
    assert torch.allclose(t(x, y)[0], base(x, y)[0], atol=1e-5)


def test_param_groups_and_private_wd():
    from sprint1.harness.optim import Muon, build_optimizer, split_adam
    m = _build(_cfg(pe_k=3, pe_shared=4))
    g = m.param_groups()
    allp = {id(p) for p in m.parameters()}
    assert {id(p) for p in g["muon"]} | {id(p) for p in g["adam"]} == allp
    assert all(p.ndim == 2 for p in g["muon"]) and len(g["muon"]) == 6 * (4 + 3 * 4)
    buckets = split_adam(m, g["adam"])
    assert {id(p) for p in m.pskip} <= {id(p) for p in buckets["scalar"]}
    lrs = {"muon": 0.05, "adam_embed": 0.6, "adam_head": 0.02, "adam_scalar": 0.04}
    opt = build_optimizer(m, lrs, muon_wd=0.4)
    assert sum(len(gr["params"]) for gr in opt.param_groups) == len(allp)
    w = _build(_cfg(pe_k=3, pe_shared=4, pe_private_wd_scale=0.5))
    cfg = w.cfg
    cfg.extra.update(muon_momentum=0.95, muon_wd=0.4, adam_betas=[0.8, 0.95], adam_eps=1e-10, adam_wd=0.0)
    opt = w.make_optimizer(w, cfg, lrs)
    mu = [o for o in opt.optimizers if isinstance(o, Muon)][0]
    shared, priv = mu.param_groups
    assert shared["weight_decay"] == 0.4 and priv["weight_decay"] == pytest.approx(0.2)
    assert len(shared["params"]) == 6 * 4 and len(priv["params"]) == 6 * 3 * 4
    assert {id(p) for p in priv["params"]} == {id(p) for p in w.private_matrices()}
    assert id(w.blocks[4].mlp.c_fc.weight) in {id(p) for p in priv["params"]}
    assert sum(len(gr["params"]) for gr in opt.param_groups) == len({id(p) for p in w.parameters()})


def _run(synth_dir, out, *extra):
    from babylm.harness import train as TR
    from babylm.tests.conftest import tiny_args
    return TR.main(tiny_args(synth_dir, out, "--epochs", "1", "--steps", "3", "--no_full_dev", "--no_pt",
                             "--n_layer", "8", "--extra", "n_value_embeds=0", "--tie_embeddings", *extra))


def test_harness_end_to_end_cpu(synth_dir, tmp_path):
    """Harness steps for B (K = 1 through the attack code), PE3, NJ and PE3w; same words fed (exposure counter
    unchanged); flops from the model; KD / MNTP refused for K > 1."""
    b = _run(synth_dir, tmp_path / "b", "--model", "baseline")
    b1 = _run(synth_dir, tmp_path / "b1", "--model", "particles")
    assert b1["final_train_loss"] == b["final_train_loss"] and b1["flops_per_token"] == b["flops_per_token"]
    pe = ["--model", "particles", "--extra", "pe_k=3", "--extra", "pe_shared=4"]
    for name, extra in (("pe3", []), ("nj", ["--extra", "pe_objective=mixture"]),
                        ("pe3w", ["--extra", "pe_private_wd_scale=0.5"])):
        r = _run(synth_dir, tmp_path / name, *pe, *extra)
        assert r["status"] == "ok" and r["words_seen"] == b["words_seen"] and r["tokens_seen"] == b["tokens_seen"]
        assert r["flops_source"] == "model" and r["flops_per_token"] > 2 * b["flops_per_token"]
        rows = [json.loads(l) for l in (tmp_path / name / "log.jsonl").read_text().splitlines()]
        tr = [x for x in rows if x.get("type") == "train"]
        assert tr and all(np.isfinite(x["pe_member_ce"]) and x["pe_mix_ce"] <= x["pe_member_ce"] + 1e-6 for x in tr)
    with pytest.raises(SystemExit):
        _run(synth_dir, tmp_path / "mntp", *pe, "--mntp_frac", "0.25", "--vocab_pad", "16512")


@pytest.mark.skipif(not HAS_HF, reason="needs transformers")
@pytest.mark.parametrize("extra", [{}, {"pe_tied": True}])
def test_hf_export_matches_harness(tmp_path, extra):
    """HF log p_bar == harness mix_logprobs (fp32, < 1e-5), with right padding; AutoModel = mean particle state;
    the registered scorer (dev_bpw.py) and its per-token dump agree on the exported ensemble."""
    import subprocess
    import sys
    from babylm.data import corpus as C
    from babylm.harness import export_hf as E
    from babylm.moonshot import dev_bpw_dump as DD
    from babylm.tests.test_moonshot_kd import _tiny_dev
    cfg = ModelConfig(vocab_size=C.VOCAB_SIZE, n_layer=8, n_head=2, n_embd=32, seq_len=512, tie_embeddings=True,
                      extra={"n_value_embeds": 0, "pe_k": 3, "pe_shared": 4, **extra})
    m = _build(cfg, jitter=True)
    ck = tmp_path / "pe.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg.to_dict(), "model_name": "particles", "words_seen": 0,
                "label": "test"}, ck)
    out = tmp_path / "hf"
    E.export_checkpoint(ck, out)
    hcfg = json.loads((out / "config.json").read_text())
    assert hcfg["auto_map"]["AutoModelForCausalLM"] == "modeling_hinton_pe.HintonPEForCausalLM" and hcfg["pe_k"] == 3
    assert E.verify_export(out, ck) < 1e-5
    dev, _ = _tiny_dev(tmp_path)
    ref = tmp_path / "ref.json"
    subprocess.run([sys.executable, "babylm/eval/dev_bpw.py", str(out), str(dev), "--slice", "full", "--ws",
                    "collapse", "--line_join", "newline", "--out", str(ref)], check=True, capture_output=True,
                   cwd=Path(__file__).resolve().parents[2])
    DD.main([str(out), str(dev), "--out", str(tmp_path / "d.npz"), "--check", str(ref)])
    assert math.isfinite(json.loads(ref.read_text())["nats_per_word"])


def test_make_particle_jobs(tmp_path):
    from babylm.moonshot import make_particle_jobs as MP
    base = {"name": "b", "module": "babylm.harness.train", "timeout_s": 100,
            "args": ["--model", "baseline", "--size", "S", "--split", "train_10M", "--seed", "0", "--save", "--tag",
                     "b"]}
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps({"jobs": [base]}))
    for gate, n in ((2, 7), (3, 6)):
        d = MP.build(f"{f}:b", gate)
        assert len(d["jobs"]) == n
        for j in d["jobs"]:
            a = j["args"]
            for flag in ("--model", "--split", "--seed", "--tag", "--size"):
                assert a.count(flag) == 1
            assert a[a.index("--tag") + 1] == j["name"]
            assert a[a.index("--split") + 1] == ("train_1M" if gate == 2 else "train_10M")
        by = {j["name"]: j["args"] for j in d["jobs"]}
        pe3 = by[f"ms_pe_g{gate}_PE3_s0"]
        assert pe3[pe3.index("--model") + 1] == "particles" and "pe_k=3" in pe3 and "pe_shared=4" in pe3
        assert "pe_objective=mixture" in by[f"ms_pe_g{gate}_NJ_s0"]
        assert by[f"ms_pe_g{gate}_FM_s0"][by[f"ms_pe_g{gate}_FM_s0"].index("--size") + 1] == "M"
        assert "35" in by[f"ms_pe_g{gate}_Fdeep_s0"]
    assert [j["name"] for j in MP.build(f"{f}:b", 2)["jobs"]][:3] == [f"ms_pe_g2_B_s{s}" for s in (0, 1, 2)]
    from babylm.moonshot import make_f1_jobs as F1
    full, r = {j["name"]: j for j in MP.build(f"{f}:b", 2)["jobs"]}, MP.build(f"{f}:b", 2, reuse_f1=True)
    f1 = {j["name"]: j["args"] for j in F1.build(f"{f}:b")["jobs"]}
    assert [j["name"] for j in r["jobs"]] == ["ms_pe_g2_PE3_s0", "ms_pe_g2_NJ_s0", "ms_pe_g2_Fdeep_s0"]
    assert set(r["reused_from_f1"]) == set(full) - {j["name"] for j in r["jobs"]}

    def bare(a):  # the arguments with --tag and --model removed (K = 1 particles == baseline, tested above)
        a = list(a)
        for flag in ("--tag", "--model"):
            if flag in a:
                del a[a.index(flag):a.index(flag) + 2]
        return a
    for pe_name, f1_name in r["reused_from_f1"].items():
        assert bare(full[pe_name]["args"]) == bare(f1[f1_name])
    with pytest.raises(SystemExit):
        MP.build(f"{f}:b", 3, reuse_f1=True)
    e5 = MP.build(f"{f}:b", 3, epochs=5)["jobs"]
    assert [j["name"] for j in e5][:2] == ["ms_pe_g3_B_s0_e5", "ms_pe_g3_PE3_s0_e5"]
    assert all(j["args"][j["args"].index("--epochs") + 1] == "5" and j["args"].count("--epochs") == 1 for j in e5)
    with pytest.raises(SystemExit):
        MP.build(f"{f}:b", 2, epochs=5)


@pytest.mark.parametrize("tied", [False, True])
def test_particle_members_are_the_particles(tmp_path, tied):
    """Each split member (a plain baseline checkpoint) is particle k's function; their mixture is the ensemble."""
    from babylm.moonshot import particle_members as PMB
    cfg = _cfg(pe_k=3, pe_shared=4, pe_tied=tied)
    m = _build(cfg, jitter=True).eval()
    ck = tmp_path / "pe.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg.to_dict(), "model_name": "particles", "words_seen": 7,
                "label": "x"}, ck)
    x, _ = _batch()
    lps = []
    for k, p in enumerate(PMB.split_checkpoint(ck, tmp_path / "members")):
        mk = torch.load(p, weights_only=False)
        assert mk["model_name"] == "baseline" and mk["words_seen"] == 7 and mk["particle_member"] == k
        assert not any(n.startswith("pe_") for n in mk["cfg"]["extra"])
        b = baseline.build(ModelConfig.from_dict(mk["cfg"]))
        b.load_state_dict(mk["model"])
        lp = F.log_softmax(b.logits(b.hidden(x)), -1)
        assert torch.allclose(lp, F.log_softmax(m.logits(m.particle_hiddens(x)[k]), -1), atol=1e-5)
        lps.append(lp)
    assert torch.allclose(torch.logsumexp(torch.stack(lps), 0) - math.log(3), m.mix_logprobs(x), atol=1e-5)


@pytest.mark.skipif(not HAS_HF, reason="needs transformers")
def test_member_mixture_reproduces_pe_score(tmp_path, monkeypatch):
    """Critic ref-0417 (2): the split members' mixture, scored by the registered protocol, reproduces the PE
    export's p_bar DEV within 1e-4 nats/word (ceiling_report c3 --ensemble enforces it)."""
    from babylm.analysis import p2_slices
    from babylm.data import corpus as C
    from babylm.harness import export_hf as E
    from babylm.moonshot import ceiling_report as R
    from babylm.moonshot import dev_bpw_dump as DD
    from babylm.moonshot import particle_members as PMB
    from babylm.tests.test_moonshot_kd import _tiny_dev
    cfg = ModelConfig(vocab_size=C.VOCAB_SIZE, n_layer=8, n_head=2, n_embd=32, seq_len=512, tie_embeddings=True,
                      extra={"n_value_embeds": 0, "pe_k": 3, "pe_shared": 4})
    ck = tmp_path / "pe.pt"
    torch.save({"model": _build(cfg, jitter=True).state_dict(), "cfg": cfg.to_dict(), "model_name": "particles",
                "words_seen": 0, "label": "test"}, ck)
    dev, _ = _tiny_dev(tmp_path)
    dumps = []
    for name, c in [("pe", ck)] + [(f"m{k}", p) for k, p in enumerate(PMB.split_checkpoint(ck, tmp_path / "mem"))]:
        E.export_checkpoint(c, tmp_path / f"hf_{name}")
        DD.main([str(tmp_path / f"hf_{name}"), str(dev), "--out", str(tmp_path / f"{name}.npz")])
        dumps.append(str(tmp_path / f"{name}.npz"))
    monkeypatch.setattr(p2_slices, "line_hashes", lambda split, data_dir: np.zeros(0, dtype=np.uint64))
    R.main(["c3", *dumps[1:], "--ensemble", dumps[0], "--out", str(tmp_path / "rep"), "--dev_dir", str(dev)])
    chk = json.loads((tmp_path / "rep" / "c3.json").read_text())["ensemble_check"]
    assert chk["ok"] and chk["abs_diff"] < 1e-5
