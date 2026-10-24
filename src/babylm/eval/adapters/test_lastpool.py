"""python test_lastpool.py PUBLISHED_GPT2_DIR : unit test of the pooling-only adapter."""
import sys, tempfile, os, torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GPT2Model, GPT2LMHeadModel
src = sys.argv[1]; d = tempfile.mkdtemp()
os.system(f'{sys.executable} {os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_lastpool.py")} {src} {d}')
tok = AutoTokenizer.from_pretrained(d); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = 'right'
enc = tok(['The cat sat on the mat.', 'Hi'], return_tensors='pt', padding=True)
ada, ref = AutoModel.from_pretrained(d, trust_remote_code=True).eval(), GPT2Model.from_pretrained(src).eval()
assert type(ada).__name__ == 'GPT2LastTokenPoolModel'
with torch.no_grad():
    a, r = ada(input_ids=enc.input_ids, attention_mask=enc.attention_mask)[0], ref(input_ids=enc.input_ids, attention_mask=enc.attention_mask)[0]
    n = enc.attention_mask.sum(-1) - 1
    for b in range(2):
        assert torch.allclose(a[b, 0], r[b, n[b]], atol=1e-5), (b, (a[b, 0] - r[b, n[b]]).abs().max())
    assert torch.allclose(a[:, 1:], r[:, 1:], atol=1e-5)
    m3 = enc.attention_mask[:, None, :].repeat(1, enc.input_ids.size(1), 1).tril()
    a3 = ada(input_ids=enc.input_ids, attention_mask=m3)[0]
    assert torch.allclose(a3[:, 0], a[:, 0], atol=1e-5)
    lm = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True).eval()
    assert type(lm) is GPT2LMHeadModel
    assert torch.equal(lm(enc.input_ids).logits, GPT2LMHeadModel.from_pretrained(src).eval()(enc.input_ids).logits)
    tok.padding_side = 'left'; encl = tok(['The cat sat on the mat.', 'Hi'], return_tensors='pt', padding=True)
    try:
        ada(input_ids=encl.input_ids, attention_mask=encl.attention_mask); raise SystemExit('left padding not refused')
    except AssertionError:
        pass
print('lastpool adapter OK: position 0 = last real token, other positions and the causal LM unchanged')
