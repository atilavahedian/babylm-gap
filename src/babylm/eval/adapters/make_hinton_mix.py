"""python make_hinton_mix.py DST_DIR MEMBER_HF_DIR MEMBER_HF_DIR [...] : one causal HF model = per-token mixture.

Critic 44552922 (official K=2 scoring): zero-shot = the per-token mixture (mean of member softmaxes) exported as ONE
causal model and scored by the unchanged official pipeline. Members are independent causal Hinton exports (hinton_gpt)
with identical tokenizer and code files (checked by sha256); their weights are stored unchanged under members.<k>.
The tokenizer, configuration_hinton.py and modeling_hinton.py come from member 0; mix/configuration_hinton_mix.py and
mix/modeling_hinton_mix.py are added. hinton_mix.json records each member's source dir and file hashes.
GLUE is never run on this export (HintonMixModel refuses); see glue_ensemble.py. Unit test: test_hinton_mix.py.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SHARED = ("configuration_hinton.py", "modeling_hinton.py", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "vocab.json", "merges.txt")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make(dst: Path, members: list[Path]) -> Path:
    assert len(members) >= 2, "need >= 2 members"
    cfgs = [json.loads((m / "config.json").read_text()) for m in members]
    for m, c in zip(members, cfgs):
        assert c.get("model_type") == "hinton_gpt", (m, c.get("model_type"))
        assert not c.get("automodel_bidirectional"), f"{m}: MNTP/bidirectional export, not a causal member"
        assert c["vocab_size"] == cfgs[0]["vocab_size"], "members differ in vocab_size"
    for f in SHARED:
        hs = {sha(m / f) if (m / f).exists() else None for m in members}
        assert len(hs) == 1, f"members differ in {f}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in SHARED:
        if (members[0] / f).exists():
            shutil.copy(members[0] / f, dst / f)
    for f in ("configuration_hinton_mix.py", "modeling_hinton_mix.py"):
        shutil.copy(HERE / "mix" / f, dst / f)
    sd = {}
    for k, m in enumerate(members):
        w = list(m.glob("*.safetensors"))
        assert len(w) == 1, f"{m}: expected one safetensors file, found {len(w)}"
        sd.update({f"members.{k}.{n}": t.contiguous() for n, t in load_file(str(w[0])).items()})
    save_file(sd, str(dst / "model.safetensors"), metadata={"format": "pt"})
    c0 = cfgs[0]
    cfg = {"model_type": "hinton_mix", "architectures": ["HintonMixForCausalLM"],
           "auto_map": {"AutoConfig": "configuration_hinton_mix.HintonMixConfig",
                        "AutoModel": "modeling_hinton_mix.HintonMixModel",
                        "AutoModelForCausalLM": "modeling_hinton_mix.HintonMixForCausalLM"},
           "member_configs": cfgs, "vocab_size": c0["vocab_size"], "hidden_size": c0.get("hidden_size", c0.get("n_embd")),
           "bos_token_id": c0.get("bos_token_id", 0), "eos_token_id": c0.get("eos_token_id", 0),
           "pad_token_id": c0.get("pad_token_id", 0)}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    (dst / "hinton_mix.json").write_text(json.dumps({"members": [
        {"dir": str(m), "files": {p.name: sha(p) for p in sorted(m.iterdir()) if p.is_file()}} for m in members]}, indent=1))
    return dst


if __name__ == "__main__":
    print("hinton mixture ->", make(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]))
