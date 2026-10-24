"""Build control checkpoints for pipeline certification (critic Step 0).
Usage: python make_controls.py HF_DIR OUT_DIR
  untrained_gpt2     GPT-2 baseline config, random init (seed 0), published tokenizer
  untrained_gptbert  GPT-BERT causal-focus config/code, random init (seed 0)
  nan_gptbert        GPT-BERT published weights with every parameter set to NaN (any score from it is void)"""
import os, shutil, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM
hf, out = sys.argv[1], sys.argv[2]
g2, gb = os.path.join(hf, 'babylm-baseline-10m-gpt2'), os.path.join(hf, 'babylm-baseline-10m-gpt-bert-causal-focus')

def copy_aux(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(('.json', '.py', '.txt')) and not f.startswith('.'):
            shutil.copy(os.path.join(src, f), dst)

torch.manual_seed(0)
m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(g2))
copy_aux(g2, f'{out}/untrained_gpt2'); m.save_pretrained(f'{out}/untrained_gpt2')

for name, fill in (('untrained_gptbert', None), ('nan_gptbert', float('nan'))):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(gb, trust_remote_code=True)
    sd = model.state_dict()
    if fill is None:   # re-initialise every tensor the way the model's own init does
        fresh = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(gb, trust_remote_code=True), trust_remote_code=True)
        sd = fresh.state_dict()
    else:
        sd = {k: torch.full_like(v, fill) if v.is_floating_point() else v for k, v in sd.items()}
    d = f'{out}/{name}'; copy_aux(gb, d)
    torch.save(sd, os.path.join(d, 'pytorch_model.bin'))
    for f in ('model.safetensors',):
        if os.path.exists(os.path.join(d, f)): os.remove(os.path.join(d, f))
    # sanity: reload through both heads
    for cls in (AutoModelForCausalLM, AutoModelForMaskedLM):
        r = cls.from_pretrained(d, trust_remote_code=True)
        print(name, cls.__name__, 'nan params:', sum(torch.isnan(p).any().item() for p in r.parameters()))
print('controls written to', out)
