"""CPU unit test of make_hinton_mix.py.   python test_hinton_mix.py HF_TEMPLATE_DIR

Two tiny random Hinton members (different seeds): the mixture export's logits must equal
logsumexp_k(log_softmax(member_k logits)) - log 2 computed from the members loaded on their own, on a right-padded
batch; its log_softmax must be itself; a mixture of two identical members must equal that member's log_softmax;
AutoModel must refuse; members with different tokenizers must be refused.
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hinton_mix import make  # noqa: E402

tpl = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
CFG = {"model_type": "hinton_gpt", "architectures": ["HintonForCausalLM"],
       "auto_map": {"AutoConfig": "configuration_hinton.HintonConfig", "AutoModel": "modeling_hinton.HintonModel",
                    "AutoModelForCausalLM": "modeling_hinton.HintonForCausalLM"},
       "vocab_size": 64, "n_layer": 2, "n_head": 2, "n_embd": 32, "seq_len": 64}


def member(name, seed):
    d = tmp / name; d.mkdir()
    for f in ("configuration_hinton.py", "modeling_hinton.py"):
        shutil.copy(tpl / f, d / f)
    (d / "config.json").write_text(json.dumps(CFG))
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(d, trust_remote_code=True), trust_remote_code=True)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
    m.save_pretrained(d)
    for f in ("configuration_hinton.py", "modeling_hinton.py"):
        shutil.copy(tpl / f, d / f)
    (d / "config.json").write_text(json.dumps(CFG))
    (d / "tokenizer.json").write_text('{"fake": 1}')
    return d


a, b = member("a", 1), member("b", 2)
mix = make(tmp / "mix", [a, b])
ids = torch.tensor([[0, 5, 9, 3, 7, 2], [0, 8, 4, 0, 0, 0]]); am = torch.tensor([[1] * 6, [1, 1, 1, 0, 0, 0]])
with torch.no_grad():
    M = AutoModelForCausalLM.from_pretrained(mix, trust_remote_code=True).eval()
    assert type(M).__name__ == "HintonMixForCausalLM"
    z = M(input_ids=ids, attention_mask=am).logits
    lps = [F.log_softmax(AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True).eval()(
        input_ids=ids, attention_mask=am).logits.float(), -1) for d in (a, b)]
    want = torch.logsumexp(torch.stack(lps), 0) - math.log(2)
    assert torch.allclose(z, want, atol=1e-6), (z - want).abs().max()
    assert torch.allclose(F.log_softmax(z, -1), z, atol=1e-5)
    assert not torch.allclose(lps[0], lps[1], atol=1e-3)   # the members really differ
    same = make(tmp / "same", [a, a])
    zs = AutoModelForCausalLM.from_pretrained(same, trust_remote_code=True).eval()(input_ids=ids, attention_mask=am).logits
    assert torch.allclose(zs, lps[0], atol=1e-5)
try:
    AutoModel.from_pretrained(mix, trust_remote_code=True)(input_ids=ids); raise SystemExit("AutoModel did not refuse")
except NotImplementedError:
    pass
(b / "tokenizer.json").write_text('{"fake": 2}')
try:
    make(tmp / "bad", [a, b]); raise SystemExit("different tokenizers not refused")
except AssertionError:
    pass
print("hinton mixture OK: logits = log mean of member softmaxes, normalised, identical members = member, "
      "AutoModel refuses, tokenizer mismatch refused")
