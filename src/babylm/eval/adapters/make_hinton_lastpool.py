"""python make_hinton_lastpool.py SRC_HF_DIR DST_DIR : last-token GLUE pooling for a causal Hinton export (hinton_gpt).

Experimenter ruling 11 Oct 01:50Z + critic 44552922: the official finetune classifier reads encoding[:, 0]
(classifier_model.py, no --take_final). Our exports start every input with <|endoftext|> (BOS = pad = id 0), so for a
causal trunk position 0 carries no context and GLUE collapses to the majority class (base_S_e6: mcc 0 on every binary
task, 11 Oct). This copies the export, sets config.automodel_pool = "last" (HintonModel then writes each row's last
NON-PAD hidden state into position 0, like GPT2ForSequenceClassification), and replaces the exported last_pool() with a
padding-side-agnostic version: last real index = the highest position whose attention_mask is 1, for right- AND
left-padded batches (the template's version asserted right padding). Weights, tokenizer, every other position and
HintonForCausalLM (zero-shot) are unchanged. GLUE scored this way is labelled "lastpool" and reported beside the
official BOS-pooled number, never instead of it. Unit test: test_hinton_lastpool.py.
"""
import json
import re
import shutil
import sys
from pathlib import Path

LAST_POOL = '''def last_pool(h: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Copy each row's last NON-PAD token state into position 0 (babylm/eval/adapters/make_hinton_lastpool.py):
    last = highest index whose attention_mask is 1, so right- and left-padded batches pool the same token. A 3-D
    (B, T, T) query x key mask uses its last row (the key padding). Real tokens must be contiguous."""
    if attention_mask is not None and attention_mask.dim() == 3:
        attention_mask = attention_mask[:, -1, :]
    if attention_mask is None:
        last = torch.full((h.size(0),), h.size(1) - 1, device=h.device, dtype=torch.long)
    else:
        m = (attention_mask != 0).long().to(h.device)
        pos = torch.arange(m.size(1), device=h.device)
        last = (m * pos).argmax(-1)
        first = torch.where(m.bool(), pos, m.size(1)).amin(-1)
        assert bool((m.sum(-1) == last - first + 1).all()), "lastpool: real tokens must be contiguous"
    pooled = h[torch.arange(h.size(0), device=h.device), last]
    return torch.cat([pooled[:, None], h[:, 1:]], 1)
'''


def make(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, dirs_exist_ok=True)
    cfg_p = dst / "config.json"
    c = json.loads(cfg_p.read_text())
    assert c.get("model_type") == "hinton_gpt", c.get("model_type")
    assert not c.get("automodel_bidirectional"), "bidirectional AutoModel: BOS pooling already sees the whole input"
    c["automodel_pool"] = "last"
    cfg_p.write_text(json.dumps(c, indent=2))
    mp = dst / "modeling_hinton.py"
    s = mp.read_text()
    s2, n = re.subn(r"^def last_pool\(.*?(?=^class |^def |\Z)", LAST_POOL + "\n\n", s, count=1, flags=re.S | re.M)
    assert n == 1, "last_pool() not found in modeling_hinton.py"
    mp.write_text(s2)
    return dst


if __name__ == "__main__":
    print("hinton lastpool adapter ->", make(Path(sys.argv[1]), Path(sys.argv[2])))
