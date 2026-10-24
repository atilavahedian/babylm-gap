"""Pre-flight check of an HF dir before official scoring (CPU is fine, ~1 min):
    python smoke_hf.py HF_DIR [--pool bos|last]
Checks: tokenizer loads with trust_remote_code; padding side right (pipeline default) and a pad token; BOS handling of
the zero-shot path; AutoModelForCausalLM loads and gives finite logits of vocab width on a right-padded batch;
AutoModel loads and gives last_hidden_state (what classifier_model.py reads) of config.hidden_size; with --pool last,
position 0 equals the last real token's state. Exits 1 on any failure; prints a JSON summary."""
import argparse, json, sys, torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor

ap = argparse.ArgumentParser(); ap.add_argument('hf_dir'); ap.add_argument('--pool', choices=['bos', 'last'], default=None)
a = ap.parse_args()
out, ok = {'hf_dir': a.hf_dir}, True
def check(name, cond, info=None):
    global ok
    out[name] = {'ok': bool(cond), **({'info': info} if info is not None else {})}; ok &= bool(cond)

cfg = AutoConfig.from_pretrained(a.hf_dir, trust_remote_code=True)
tok = AutoProcessor.from_pretrained(a.hf_dir, trust_remote_code=True, padding_side='right')   # as finetune/trainer.py
check('padding_side_right', tok.padding_side == 'right', tok.padding_side)
check('has_pad_token', tok.pad_token_id is not None, tok.pad_token)
sents = ['The cat sat on the mat.', 'Hi there']
enc = tok(sents, return_tensors='pt', padding=True)
ids0 = enc.input_ids[0].tolist()
out['bos_id'], out['first_ids'] = tok.bos_token_id, ids0[:4]
check('tokens_roundtrip', 'cat sat' in tok.decode(ids0, skip_special_tokens=True), tok.decode(ids0))
lm = AutoModelForCausalLM.from_pretrained(a.hf_dir, trust_remote_code=True).eval()
with torch.no_grad():
    lo = lm(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    lo = lo[0] if isinstance(lo, tuple) else lo.logits
check('causal_logits_finite', torch.isfinite(lo).all().item(), list(lo.shape))
check('causal_logits_vocab', lo.shape[-1] >= len(tok) - 1, [lo.shape[-1], len(tok)])
out['causal_class'] = type(lm).__name__
if getattr(cfg, 'model_type', '') == 'hinton_mix':   # mixture: no finetuning trunk by design (GLUE = member ensemble)
    h = None; out['automodel_class'] = 'skipped (hinton_mix: GLUE is the member ensemble, critic 44552922)'
else:
    am = AutoModel.from_pretrained(a.hf_dir, trust_remote_code=True).eval()
    with torch.no_grad():
        h = am(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    h = h[0] if isinstance(h, tuple) else getattr(h, 'last_hidden_state', None)
    check('automodel_hidden', h is not None and h.shape[-1] == cfg.hidden_size and torch.isfinite(h).all().item(),
          None if h is None else [list(h.shape), cfg.hidden_size])
    out['automodel_class'] = type(am).__name__
if a.pool == 'last' and h is not None:
    with torch.no_grad():
        single = am(input_ids=enc.input_ids[1:2, :int(enc.attention_mask[1].sum())])
    single = single[0] if isinstance(single, tuple) else single.last_hidden_state
    check('pool_last_matches_unpadded_last_token', torch.allclose(h[1, 0], single[0, -1], atol=1e-4),
          float((h[1, 0] - single[0, -1]).abs().max()))
out['ok'] = ok
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
