"""Usage: python test_rot.py TEMPLATE_DIR OUT_DIR [build]  -> prints loss and cos[0,0] after AutoModelForCausalLM load."""
import sys, os, shutil, json, torch
tpl, out = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:  # build a tiny random model with this template
    sys.path.insert(0, os.path.dirname(tpl)); import importlib
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    for f in ('configuration_hinton.py', 'modeling_hinton.py'): shutil.copyfile(os.path.join(tpl, f), os.path.join(out, f))
    open(os.path.join(out, '__init__.py'), 'w').close()
    sys.path.insert(0, os.path.dirname(out)); pkg = os.path.basename(out)
    cfgm = importlib.import_module(pkg + '.configuration_hinton'); mm = importlib.import_module(pkg + '.modeling_hinton')
    c = cfgm.HintonConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, seq_len=128)
    c.auto_map = {"AutoConfig": "configuration_hinton.HintonConfig", "AutoModel": "modeling_hinton.HintonModel",
                  "AutoModelForCausalLM": "modeling_hinton.HintonForCausalLM"}
    torch.manual_seed(0); m = mm.HintonForCausalLM(c)
    with torch.no_grad():
        for p in m.parameters(): p.normal_(0, 0.05)
    m.save_pretrained(out, safe_serialization=True)
    print('built', out); sys.exit()
import transformers
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True).float().eval()
rot = m.blocks[0].attn.rotary if hasattr(m.blocks[0], 'attn') else [x for x in m.modules() if type(x).__name__ == 'Rotary'][0]
cos_before = rot.cos.flatten()[0].item() if rot.cos.device.type != 'meta' else 'meta'
torch.manual_seed(1); ids = torch.randint(0, 256, (2, 100))
with torch.no_grad():
    lg = m(input_ids=ids).logits.float()
loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1)).item()
print(json.dumps({'transformers': transformers.__version__, 'cos00_before_forward': cos_before,
                  'cos00_after': rot.cos.flatten()[0].item(), 'loss': loss, 'logit_sum': lg.double().sum().item()}))
