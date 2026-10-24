"""Looped depth (babylm/models/looped.py): k=1 is the baseline, k=2 is a weight-shared unroll of the plain 2L-deep
model, FLOPs count the shared blocks per pass, and the HF export reproduces the harness logits."""
from __future__ import annotations

import copy
import math

import pytest
import torch

from babylm.data import corpus as C
from babylm.models import looped
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

CFG = dict(vocab_size=C.VOCAB_SIZE, n_layer=4, n_head=2, n_embd=64, seq_len=64)


def build(mod, n_layer=4, **extra):
    torch.manual_seed(0)
    return mod.build(ModelConfig(extra={"softcap": 15, "unet": True, **extra}, **dict(CFG, n_layer=n_layer)))


def perturb(m):
    with torch.no_grad():  # non-trivial weights everywhere (zero-init projections, lambdas, skip weights)
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


@pytest.mark.parametrize("nve", [0, 1])
def test_one_loop_equals_baseline(nve):
    a, b = build(baseline, n_value_embeds=nve), build(looped, n_value_embeds=nve, n_loops=1)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert a.flops_per_token() == b.flops_per_token()
    x = torch.randint(0, C.VOCAB_SIZE, (2, 32))
    y = torch.randint(0, C.VOCAB_SIZE, (2, 32))
    la, _ = a(x, y)
    lb, _ = b(x, y)
    assert torch.equal(la, lb)


@pytest.mark.parametrize("nve", [0, 1])
def test_two_loops_is_a_weight_shared_unroll(nve):
    """2 blocks looped 2x == the plain 4-block model whose blocks 2, 3 are copies of blocks 0, 1."""
    lo = perturb(build(looped, n_layer=2, n_value_embeds=nve, n_loops=2))
    plain = build(baseline, n_layer=4, n_value_embeds=nve)
    sd = {k: v for k, v in lo.state_dict().items() if not k.startswith("blocks.")}
    for i in range(4):
        for k, v in lo.blocks[i % 2].state_dict().items():
            sd[f"blocks.{i}.{k}"] = v
    plain.load_state_dict(sd)
    assert lo.skip_weights.numel() == plain.skip_weights.numel() == 2
    lo.eval()
    plain.eval()
    x = torch.randint(0, C.VOCAB_SIZE, (2, 32))
    assert torch.allclose(lo.logits(lo.hidden(x)), plain.logits(plain.hidden(x)), atol=1e-5, rtol=0)
    # shared weights: one parameter tensor per block matrix, gradient from both passes
    n_lo = sum(p.numel() for p in lo.blocks.parameters())
    assert n_lo * 2 == sum(p.numel() for p in plain.blocks.parameters())


def test_flops_count_shared_blocks_per_pass():
    lo = build(looped, n_layer=2, n_value_embeds=0, n_loops=2)
    plain = build(baseline, n_layer=4, n_value_embeds=0)
    assert lo.flops_per_token() == pytest.approx(plain.flops_per_token())


def test_bad_loop_count():
    with pytest.raises(ValueError):
        build(looped, n_loops=0)


def test_hf_export_looped_logits(tmp_path):
    from transformers import AutoModelForCausalLM

    from babylm.harness import export_hf as E

    m = perturb(build(looped, n_layer=2, n_value_embeds=0, n_loops=2))
    ck = tmp_path / "chck_1M.pt"
    torch.save({"model": {k: v.float() for k, v in m.state_dict().items()}, "cfg": m.cfg.to_dict(),
                "model_name": "looped", "words_seen": 1, "words_nominal": 1, "label": "chck_1M", "step": 1}, ck)
    out = tmp_path / "hf"
    E.export_checkpoint(ck, out)
    assert E.verify_export(out, ck) < 1e-4
    hf = AutoModelForCausalLM.from_pretrained(str(out), trust_remote_code=True, torch_dtype=torch.float32).eval()
    assert hf.config.n_loops == 2
    ref = copy.deepcopy(m).eval()
    ids = torch.randint(1, C.VOCAB_SIZE, (2, 24))
    with torch.no_grad():
        z = hf(input_ids=ids).logits
        zr = ref.logits(ref.hidden(ids))
    assert torch.allclose(z, zr, atol=1e-4, rtol=0)


def test_harness_trains_looped(synth_dir, tmp_path):
    from babylm.harness import train as TR
    from babylm.tests.conftest import tiny_args

    args = ["--steps", "6", "--milestone_scale", "0.002", "--no_full_dev", "--no_pt", "--save",
            "--model", "looped", "--extra", "n_loops=2"]
    r = TR.main(tiny_args(synth_dir, tmp_path / "run", *args))
    assert r["status"] == "ok" and math.isfinite(r["final_train_loss"])
    cks = sorted((tmp_path / "run" / "checkpoints").glob("chck_*.pt"))
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    assert ck["model_name"] == "looped" and ck["cfg"]["extra"]["n_loops"] == 2
    from babylm.harness import export_hf as E
    E.export_checkpoint(cks[-1], tmp_path / "hf")
    assert E.verify_export(tmp_path / "hf", cks[-1]) < 1e-4
