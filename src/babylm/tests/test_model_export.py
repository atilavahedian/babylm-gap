"""dropout=0 equals the frozen model; HF export matches the harness model and loads by revision offline."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from babylm.data import corpus as C
from babylm.models import baseline_reg
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

CFG = dict(vocab_size=C.VOCAB_SIZE, n_layer=4, n_head=2, n_embd=64, seq_len=64)


def build(mod, **extra):
    torch.manual_seed(0)
    return mod.build(ModelConfig(extra={"softcap": 15, "unet": True, **extra}, **CFG))


def test_dropout_zero_equals_frozen():
    a, b = build(baseline), build(baseline_reg, dropout=0.0)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    x = torch.randint(0, C.VOCAB_SIZE, (2, 32))
    y = torch.randint(0, C.VOCAB_SIZE, (2, 32))
    for train in (True, False):
        a.train(train)
        b.train(train)
        la, _ = a(x, y)
        lb, _ = b(x, y)
        assert torch.equal(la, lb)
        la.backward()
        lb.backward()
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            assert torch.equal(p.grad, q.grad), n


def test_dropout_positive_is_identity_in_eval_and_active_in_train():
    a, b = build(baseline), build(baseline_reg, dropout=0.3)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    x = torch.randint(0, C.VOCAB_SIZE, (2, 32))
    a.eval()
    b.eval()
    assert torch.allclose(a.logits(a.hidden(x)), b.logits(b.hidden(x)), atol=0, rtol=0)
    b.train()
    with torch.no_grad():  # make the zero-init projections non-zero so residual dropout matters
        for blk in b.blocks:
            blk.mlp.c_proj.weight.normal_(0, 0.02)
            blk.attn.c_proj.weight.normal_(0, 0.02)
    torch.manual_seed(1)
    h1 = b.hidden(x)
    torch.manual_seed(2)
    h2 = b.hidden(x)
    assert not torch.equal(h1, h2)


def save_ckpt(path: Path, model, label="chck_1M", words=1_000_000):
    torch.save({"model": {k: v.float() for k, v in model.state_dict().items()}, "cfg": model.cfg.to_dict(),
                "model_name": "baseline", "words_seen": words, "words_nominal": words, "label": label, "step": 1},
               path)


def test_hf_export_logits_and_revisions(tmp_path):
    from babylm.harness import export_hf as E

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    m = build(baseline)
    with torch.no_grad():  # non-trivial weights everywhere (zero-init projections, lambdas)
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    save_ckpt(run / "checkpoints" / "chck_1M.pt", m)
    with torch.no_grad():
        m.lm_head.weight.mul_(2.0)
    save_ckpt(run / "checkpoints" / "chck_2M.pt", m, "chck_2M", 2_000_000)
    rep = E.export_run(run, tmp_path / "exp", "tinyrun", "hinton", verify=True)
    for rev, v in rep["revisions"].items():
        assert v["max_abs_logit_diff"] < 1e-4, (rev, v)
    assert rep["main"] == "chck_2M"
    # the pipeline's loading call, unmodified: repo id + revision, offline, from the exported hub cache
    code = (
        "import torch;from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "outs=[]\n"
        "for r in ('chck_1M','chck_2M','main'):\n"
        "  m=AutoModelForCausalLM.from_pretrained('hinton/tinyrun',trust_remote_code=True,revision=r)\n"
        "  t=AutoTokenizer.from_pretrained('hinton/tinyrun',revision=r)\n"
        "  ids=t('Hello there.',return_tensors='pt')['input_ids']\n"
        "  assert ids[0,0].item()==0\n"
        "  outs.append(m(input_ids=ids).logits)\n"
        "assert not torch.allclose(outs[0],outs[1]) and torch.equal(outs[1],outs[2])\n"
        "print('OK')\n")
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_CACHE=str(tmp_path / "exp" / "hub"),
               HF_MODULES_CACHE=str(tmp_path / "modules"), TRANSFORMERS_OFFLINE="1")
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0 and "OK" in p.stdout, p.stderr[-3000:]


def test_hf_export_mntp_bidirectional_automodel(tmp_path):
    """MNTP-trained checkpoint -> AutoModel attends bidirectionally (position 0 sees later tokens, matches the harness
    model under BidirectionalAttention, right padding hidden); AutoModelForCausalLM stays causal and exact."""
    import json

    from transformers import AutoModel, AutoModelForCausalLM

    from babylm.harness import export_hf as E

    cfg = dict(CFG, vocab_size=16512)
    torch.manual_seed(0)
    m = baseline.build(ModelConfig(extra={"softcap": 15, "unet": True}, **cfg))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ck = tmp_path / "chck_1M.pt"
    torch.save({"model": {k: v.float() for k, v in m.state_dict().items()}, "cfg": m.cfg.to_dict(),
                "model_name": "baseline", "words_seen": 1, "words_nominal": 1, "label": "chck_1M", "step": 1,
                "mntp_frac": 0.25}, ck)
    out = tmp_path / "hf"
    prov = E.export_checkpoint(ck, out)
    assert prov["automodel_attention"] == "bidirectional"
    assert json.loads((out / "hinton_export.json").read_text())["automodel_attention"] == "bidirectional"
    assert E.verify_export(out, ck) < 1e-4  # causal logits exact; bidirectional trunk == harness bidirectional
    base = AutoModel.from_pretrained(str(out), trust_remote_code=True, torch_dtype=torch.float32).eval()
    lm = AutoModelForCausalLM.from_pretrained(str(out), trust_remote_code=True, torch_dtype=torch.float32).eval()
    ids = torch.randint(1, C.VOCAB_SIZE, (2, 24))
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % C.VOCAB_SIZE
    with torch.no_grad():
        h1, h2 = base(input_ids=ids).last_hidden_state, base(input_ids=ids2).last_hidden_state
        z1, z2 = lm(input_ids=ids).logits, lm(input_ids=ids2).logits
    assert not torch.allclose(h1[:, 0], h2[:, 0])  # BOS position sees the last token
    assert torch.equal(z1[:, :-1], z2[:, :-1])  # causal LM unaffected by the future
    # causal-trained checkpoints keep a causal AutoModel
    ck2 = tmp_path / "c.pt"
    torch.save({**torch.load(ck, weights_only=False), "mntp_frac": 0.0}, ck2)
    assert E.export_checkpoint(ck2, tmp_path / "hf2")["automodel_attention"] == "causal"
    assert E.verify_export(tmp_path / "hf2", ck2) < 1e-4


def test_hf_export_last_pool_automodel(tmp_path):
    """--automodel_pool last (critic ref-0058 / ref-0059): on a right-padded batch, AutoModel position 0 equals the
    unpadded run's last hidden state; other positions are unchanged; CausalLM logits unchanged; refused with a
    bidirectional AutoModel; default stays first (BOS) pooling."""
    import json

    import pytest
    from transformers import AutoModel, AutoModelForCausalLM

    from babylm.harness import export_hf as E

    torch.manual_seed(0)
    m = baseline.build(ModelConfig(extra={"softcap": 15, "unet": True}, **CFG))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ck = tmp_path / "chck_1M.pt"
    torch.save({"model": {k: v.float() for k, v in m.state_dict().items()}, "cfg": m.cfg.to_dict(),
                "model_name": "baseline", "words_seen": 1, "words_nominal": 1, "label": "chck_1M", "step": 1}, ck)
    first, last = tmp_path / "first", tmp_path / "last"
    assert E.export_checkpoint(ck, first)["automodel_pool"] == "first"
    prov = E.export_checkpoint(ck, last, automodel_pool="last")
    assert prov["automodel_pool"] == "last" and prov["automodel_attention"] == "causal"
    assert json.loads((last / "config.json").read_text())["automodel_pool"] == "last"
    assert json.loads((last / "hinton_export.json").read_text())["automodel_pool"] == "last"
    assert E.verify_export(last, ck) < 1e-4 and E.verify_export(first, ck) < 1e-4

    bf = AutoModel.from_pretrained(str(first), trust_remote_code=True, torch_dtype=torch.float32).eval()
    bl = AutoModel.from_pretrained(str(last), trust_remote_code=True, torch_dtype=torch.float32).eval()
    lf = AutoModelForCausalLM.from_pretrained(str(first), trust_remote_code=True, torch_dtype=torch.float32).eval()
    ll = AutoModelForCausalLM.from_pretrained(str(last), trust_remote_code=True, torch_dtype=torch.float32).eval()
    lens = [20, 13, 7]
    ids = torch.randint(1, C.VOCAB_SIZE, (3, 20))
    ids[:, 0] = C.BOS_ID
    am = torch.zeros_like(ids)
    for i, n in enumerate(lens):
        am[i, :n] = 1
        ids[i, n:] = C.BOS_ID  # right padding with pad = BOS
    am3 = am.unsqueeze(1).repeat(1, 20, 1).tril(diagonal=0)
    with torch.no_grad():
        hf_, hl = bf(input_ids=ids, attention_mask=am).last_hidden_state, bl(input_ids=ids, attention_mask=am).last_hidden_state
        hl3 = bl(input_ids=ids, attention_mask=am3).last_hidden_state
        for i, n in enumerate(lens):
            solo = bf(input_ids=ids[i:i + 1, :n]).last_hidden_state[0]
            assert torch.allclose(hl[i, 0], solo[n - 1], atol=1e-5)
            assert torch.allclose(hl3[i, 0], solo[n - 1], atol=1e-5)
            assert torch.allclose(hf_[i, 0], solo[0], atol=1e-5)  # default: BOS state
        assert torch.equal(hl[:, 1:], hf_[:, 1:])
        assert torch.equal(ll(input_ids=ids, attention_mask=am).logits, lf(input_ids=ids, attention_mask=am).logits)
        with pytest.raises(AssertionError, match="right padding"):
            bl(input_ids=ids, attention_mask=am.flip(-1))
    torch.save({**torch.load(ck, weights_only=False), "mntp_frac": 0.25}, tmp_path / "mntp.pt")
    with pytest.raises(SystemExit, match="causal AutoModels only"):
        E.export_checkpoint(tmp_path / "mntp.pt", tmp_path / "bad", automodel_pool="last")
