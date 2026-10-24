"""Write tokenizer.json (DebertaV2TokenizerFast, converted from spm.model) into AMLM checkpoint dirs.

AMLM's trainer saves only the slow sentencepiece tokenizer; the eval venv (pipeline ref-0243) has no sentencepiece, so
AutoProcessor cannot load it. The authors' released checkpoints ship this same conversion: on 300 DEV lines the
converted leukas/amlm_hard_decay tokenizer.json encodes identically to the released one (byte fallback is lost in both).
usage: python add_fast_tok.py CKPT_DIR [CKPT_DIR ...]   (needs transformers + sentencepiece + protobuf)"""
import sys, os
from transformers import DebertaV2TokenizerFast
for d in sys.argv[1:]:
    if not os.path.exists(os.path.join(d, "tokenizer.json")):
        DebertaV2TokenizerFast.from_pretrained(d).save_pretrained(d)
    print("tokenizer.json ok:", d)
