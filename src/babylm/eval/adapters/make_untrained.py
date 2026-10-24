"""python make_untrained.py SRC_HF_DIR DST_DIR [SEED] : same export, weights replaced by a fresh random init (chance floor).

GLUE sanity pair (critic 44552922): an untrained copy of the architecture must sit near the majority class while the
trained model is off it on >= 5 of 7 tasks. Config, tokenizer and code files are copied unchanged; the weights come
from AutoModelForCausalLM.from_config under torch.manual_seed(SEED) (default 0), i.e. the modules' own default
initialisation (the Hinton template's _init_weights is a no-op, so this is PyTorch's default init).
"""
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
shutil.copytree(src, dst, dirs_exist_ok=True)
for w in list(dst.glob("*.safetensors")) + list(dst.glob("*.bin")):
    w.unlink()
torch.manual_seed(seed)
m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(src, trust_remote_code=True), trust_remote_code=True)
m.save_pretrained(dst, safe_serialization=True)
for f in list(src.glob("*.py")) + [src / "config.json"]:   # keep the exact code and config (save_pretrained rewrites them)
    shutil.copy(f, dst / f.name)
print(f"untrained copy (seed {seed}) -> {dst}")
