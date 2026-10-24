"""Unit test of make_hinton_lastpool.py (critic 44552922; experimenter 11 Oct 01:50Z). CPU, a few seconds.

    python test_hinton_lastpool.py SRC_HF_DIR [--pipeline EVAL2025_DIR]
    python test_hinton_lastpool.py --tiny HF_TEMPLATE_DIR [--pipeline EVAL2025_DIR]   (random 2-layer export)

Checks, on a batch of different-length inputs padded on the right AND on the left:
  1. the pooled index moves with padding: position 0 of the adapter output = the trunk's state at each row's last
     real token (right: n-1; left: T-1), and positions 1.. are the trunk's own;
  2. right- vs left-padded give identical pooled states per example, and so identical classifier logits through the
     official ClassifierHead (and ModelForSequenceClassification) when --pipeline is given;
  3. each example alone (no padding) gives the same pooled state as inside the padded batches;
  4. the same holds for the 3-D (B, T, T) causal mask the official collate builds with --causal;
  5. AutoModelForCausalLM logits are bit-identical to the source export (zero-shot untouched).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_hinton_lastpool import make  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="?")
ap.add_argument("--tiny", help="hf_template dir (configuration_hinton.py, modeling_hinton.py): build a random export")
ap.add_argument("--pipeline", help="evaluation-pipeline-2025 checkout: also test the official classifier")
a = ap.parse_args()
torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp())

if a.tiny:
    src = tmp / "src"; src.mkdir()
    for f in ("configuration_hinton.py", "modeling_hinton.py"):
        shutil.copy(Path(a.tiny) / f, src / f)
    (src / "__init__.py").write_text("")
    cfg = {"model_type": "hinton_gpt", "architectures": ["HintonForCausalLM"],
           "auto_map": {"AutoConfig": "configuration_hinton.HintonConfig", "AutoModel": "modeling_hinton.HintonModel",
                        "AutoModelForCausalLM": "modeling_hinton.HintonForCausalLM"},
           "vocab_size": 64, "n_layer": 2, "n_head": 2, "n_embd": 32, "seq_len": 64, "automodel_pool": "first"}
    (src / "config.json").write_text(json.dumps(cfg))
    m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(src, trust_remote_code=True), trust_remote_code=True)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    m.save_pretrained(src)
    for f in ("configuration_hinton.py", "modeling_hinton.py"):   # save_pretrained may not copy remote code
        shutil.copy(Path(a.tiny) / f, src / f)
else:
    src = Path(a.src)

dst = make(src, tmp / "lastpool")
cfg = json.loads((dst / "config.json").read_text())
V = cfg["vocab_size"]
ada = AutoModel.from_pretrained(dst, trust_remote_code=True).eval()
ref_cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
assert getattr(ref_cfg, "automodel_pool", "first") == "first" and not getattr(ref_cfg, "automodel_bidirectional", False)
ref = AutoModel.from_pretrained(src, trust_remote_code=True).eval()   # trunk with no pooling = the per-position states
assert getattr(ada.config, "automodel_pool") == "last"

# three inputs of lengths 7, 3, 5; real tokens never use the pad id 0 except the leading BOS (as our tokenizer does)
g = torch.Generator().manual_seed(1)
seqs = [torch.cat([torch.zeros(1, dtype=torch.long), torch.randint(1, V, (n - 1,), generator=g)]) for n in (7, 3, 5)]
T = max(len(s) for s in seqs)


def batch(side):
    ids = torch.zeros(len(seqs), T, dtype=torch.long); am = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        sl = slice(0, len(s)) if side == "right" else slice(T - len(s), T)
        ids[i, sl] = s; am[i, sl] = 1
    return ids, am


def tril3(am):   # what the official collate builds with --causal
    return am.unsqueeze(1).repeat(1, am.size(-1), 1).tril(diagonal=0)


atol = 1e-5
pooled = {}
with torch.no_grad():
    for side in ("right", "left"):
        ids, am = batch(side)
        h = ada(input_ids=ids, attention_mask=am).last_hidden_state
        r = ref(input_ids=ids, attention_mask=am).last_hidden_state
        last = torch.tensor([len(s) - 1 if side == "right" else T - 1 for s in seqs])
        for i in range(len(seqs)):   # 1. index moves with padding
            assert torch.allclose(h[i, 0], r[i, last[i]], atol=atol), (side, i)
        assert torch.allclose(h[:, 1:], r[:, 1:], atol=atol), side
        h3 = ada(input_ids=ids, attention_mask=tril3(am)).last_hidden_state   # 4. 3-D mask
        assert torch.allclose(h3[:, 0], h[:, 0], atol=atol), side
        pooled[side] = h[:, 0]
    assert torch.allclose(pooled["right"], pooled["left"], atol=atol), (pooled["right"] - pooled["left"]).abs().max()  # 2.
    for i, s in enumerate(seqs):   # 3. alone = in batch
        alone = ada(input_ids=s[None], attention_mask=torch.ones(1, len(s), dtype=torch.long)).last_hidden_state[0, 0]
        assert torch.allclose(alone, pooled["right"][i], atol=atol), i
    lm_a = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True).eval()   # 5. zero-shot untouched
    lm_r = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True).eval()
    ids, am = batch("right")
    assert torch.equal(lm_a(input_ids=ids, attention_mask=am).logits, lm_r(input_ids=ids, attention_mask=am).logits)

    if a.pipeline:   # 2. identical logits through the official classifier (the path the GLUE finetune runs)
        sys.path.insert(0, a.pipeline)
        from argparse import Namespace
        from evaluation_pipeline.finetune.classifier_model import ModelForSequenceClassification
        ns = Namespace(model_name_or_path=str(dst), revision_name=None, enc_dec=False, causal=False, take_final=False,
                       classifier_layer_norm_eps=1e-5, classifier_dropout=0.1, num_labels=3)
        clf = ModelForSequenceClassification(ns).eval()
        logits = {}
        for side in ("right", "left"):
            ids, am = batch(side)
            logits[side] = clf(ids, am)
            ns3 = clf(ids, tril3(am))   # --causal collate
            assert torch.allclose(ns3, logits[side], atol=atol), side
        assert torch.allclose(logits["right"], logits["left"], atol=atol), (logits["right"] - logits["left"]).abs().max()
        print("official classifier: right- and left-padded logits identical per example")

print(f"hinton lastpool OK ({src}): pooled index moves with padding, right == left == unpadded, 3-D mask same, "
      "causal LM unchanged")
