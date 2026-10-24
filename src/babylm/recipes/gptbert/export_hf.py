"""Export train-side GPT-BERT weights (official/model_extra.py `Bert`) to a HuggingFace directory that loads with the
PUBLISHED modeling_gpt_bert.py (hf.co/BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus) through
AutoModelForCausalLM / AutoModelForMaskedLM(trust_remote_code=True) - i.e. exactly how eval2025 (causal, mntp) loads models.

Name map (train -> HF):  embedding.*                       -> model.embedding.*
                          transformer.attention_layers.i.*  -> model.attention_layers.i.*   (position_indices buffer dropped)
                          transformer.mlp_layers.i.*        -> model.mlp_layers.i.*
                          transformer.dwa_modules.alphas.j  -> model.dwa_modules.alphas.j
                          classifier.nonlinearity.k.*       -> lm_head.nonlinearity.k.*     (k=5 weight tied to word embedding)

NOTE on configuration_gpt_bert.py: the published ModelConfig IGNORES config.json (it overwrites every field with the
small-model defaults). For the published size we copy it verbatim (bit-identical eval code). For any other size we write
a patched copy that only fills defaults for missing keys (flagged in export_info.json).

CLI:
  python export_hf.py --state model.pt --out_dir hf/final          # export a saved train-side state_dict
  python export_hf.py --verify                                      # published weights -> train model -> export -> reload
"""
import argparse, functools, json, os, shutil, sys, tempfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HF_FILES = os.path.join(HERE, "hf_files")
PUBLISHED = "~/babylm/models/babylm-baseline-10m-gpt-bert-causal-focus"
sys.path.insert(0, os.path.join(HERE, "official"))

# the published model's effective config (configuration_gpt_bert.ModelConfig defaults)
PUBLISHED_MODEL_CONFIG = dict(attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1, hidden_size=384, intermediate_size=1280,
                              max_position_embeddings=512, position_bucket_size=32, num_attention_heads=6, num_hidden_layers=12,
                              vocab_size=16384, layer_norm_eps=1e-7)
MODEL_KEYS = list(PUBLISHED_MODEL_CONFIG)
COPY_FILES = ["modeling_gpt_bert.py", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "__init__.py"]


class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def build_train_model(model_config):
    from model_extra import Bert
    return Bert(Cfg(model_config))


def train_to_hf(sd):
    out = {}
    for k, v in sd.items():
        if k.endswith("position_indices"):
            continue
        if k.startswith("embedding."):
            nk = "model." + k
        elif k.startswith("transformer."):
            nk = "model." + k[len("transformer."):]
        elif k.startswith("classifier."):
            nk = "lm_head." + k[len("classifier."):]
        else:
            raise KeyError(k)
        out[nk] = v.detach().to("cpu", torch.float32)
    # keep the tie explicit (same storage), as in the published pytorch_model.bin
    out["lm_head.nonlinearity.5.weight"] = out["model.embedding.word_embedding.weight"]
    return out


def hf_to_train(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("model.embedding."):
            nk = k[len("model."):]
        elif k.startswith("model."):
            nk = "transformer." + k[len("model."):]
        elif k.startswith("lm_head."):
            nk = "classifier." + k[len("lm_head."):]
        else:
            raise KeyError(k)
        out[nk] = v
    return out


PATCHED_CONFIG_SRC = '''from __future__ import annotations
# PATCHED copy of the published configuration_gpt_bert.py (hinton babylm recipe): the published file overwrites every field
# with small-model defaults, ignoring config.json. This copy fills defaults ONLY for missing fields, so non-default model
# sizes load correctly. For the published size the original file is used unchanged.
import json, copy
from typing import Any
from transformers.configuration_utils import PretrainedConfig

_DEFAULTS = dict(attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1, hidden_size=384, intermediate_size=1280,
                 max_sequence_length=512, position_bucket_size=32, num_attention_heads=6, num_layers=12,
                 vocab_size=16384, layer_norm_eps=1e-7)


class ModelConfig(PretrainedConfig):
    def __init__(self, config_file=None, **kwargs):
        super().__init__(**kwargs)
        for k, v in _DEFAULTS.items():
            if k not in kwargs:
                setattr(self, k, v)
        if "num_hidden_layers" in kwargs and "num_layers" not in kwargs:
            self.num_layers = kwargs["num_hidden_layers"]
        if not hasattr(self, "num_hidden_layers") or "num_hidden_layers" not in kwargs:
            self.num_hidden_layers = self.num_layers

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\\n"
'''


def export(train_state_dict, out_dir, model_config=None, info=None):
    """Write an HF directory loadable by the published modeling code. Returns out_dir."""
    model_config = dict(PUBLISHED_MODEL_CONFIG if model_config is None else {k: model_config[k] for k in MODEL_KEYS})
    os.makedirs(out_dir, exist_ok=True)
    for f in COPY_FILES:
        shutil.copy2(os.path.join(HF_FILES, f), os.path.join(out_dir, f))
    is_published_size = all(model_config[k] == PUBLISHED_MODEL_CONFIG[k] for k in MODEL_KEYS)
    if is_published_size:
        shutil.copy2(os.path.join(HF_FILES, "config.json"), os.path.join(out_dir, "config.json"))
        shutil.copy2(os.path.join(HF_FILES, "configuration_gpt_bert.py"), os.path.join(out_dir, "configuration_gpt_bert.py"))
    else:
        cfg = json.load(open(os.path.join(HF_FILES, "config.json")))
        cfg.update(model_config)
        cfg["num_layers"] = model_config["num_hidden_layers"]
        cfg["max_sequence_length"] = model_config["max_position_embeddings"]
        json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
        with open(os.path.join(out_dir, "configuration_gpt_bert.py"), "w") as f:
            f.write(PATCHED_CONFIG_SRC)
    torch.save(train_to_hf(train_state_dict), os.path.join(out_dir, "pytorch_model.bin"))
    meta = {"published_size_verbatim_code": is_published_size, "model_config": model_config}
    meta.update(info or {})
    json.dump(meta, open(os.path.join(out_dir, "export_info.json"), "w"), indent=2)
    return out_dir


# ----------------------------------------------------------------------------------------------------------- verify
def _train_logits(model, ids, causal):
    T = ids.size(1)
    mask = torch.zeros(ids.size(0), T, T, dtype=torch.bool)
    if causal:
        mask |= torch.ones(T, T, dtype=torch.bool).triu(1)
    h = model.get_contextualized(ids.t(), mask)          # [T, B, D]
    return model.classifier.nonlinearity(h).transpose(0, 1)  # [B, T, V]


@functools.lru_cache(maxsize=8)
def _hf_model(path, causal):
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
    cls = AutoModelForCausalLM if causal else AutoModelForMaskedLM
    return cls.from_pretrained(path, trust_remote_code=True).eval()


def _hf_logits(path, ids, causal):
    m = _hf_model(path, causal)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids))
        return out.logits if hasattr(out, 'logits') else out[0]


def verify(published=PUBLISHED):
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    torch.manual_seed(0)
    sents = ["The cat sat on the mat.", "Where did you put my new blanket?", "She has been reading that book since Monday.",
             "*CHI:\tthere a napkin.", "Who do you think the teacher said that the boy saw yesterday at the park?"]
    tok = AutoTokenizer.from_pretrained(published)
    res = {}
    sd = torch.load(os.path.join(published, "pytorch_model.bin"), map_location="cpu")
    model = build_train_model(PUBLISHED_MODEL_CONFIG).eval()
    missing, unexpected = model.load_state_dict(hf_to_train(sd), strict=False)
    assert not unexpected and all(k.endswith("position_indices") for k in missing), (missing, unexpected)
    with tempfile.TemporaryDirectory() as tmp:
        export(model.state_dict(), tmp, PUBLISHED_MODEL_CONFIG, {"source": "published weights roundtrip"})
        exported = torch.load(os.path.join(tmp, "pytorch_model.bin"), map_location="cpu")
        assert set(exported) == set(sd), set(exported) ^ set(sd)
        res["state_dict_roundtrip_max_abs_diff"] = max((exported[k] - sd[k]).abs().max().item() for k in sd)
        for causal in (True, False):
            tag = "causal" if causal else "masked"
            d_train, d_export = 0.0, 0.0
            for s in sents:
                ids = torch.tensor([tok(s)["input_ids"]])
                with torch.no_grad():
                    ref = _hf_logits(published, ids, causal)
                    d_train = max(d_train, (_train_logits(model, ids, causal) - ref).abs().max().item())
                d_export = max(d_export, (_hf_logits(tmp, ids, causal) - ref).abs().max().item())
            res[f"{tag}: train-side model vs published HF, max|dlogit|"] = d_train
            res[f"{tag}: exported HF vs published HF, max|dlogit|"] = d_export
    # non-default size: patched config path
    tiny = dict(PUBLISHED_MODEL_CONFIG, hidden_size=64, intermediate_size=128, num_attention_heads=2, num_hidden_layers=2)
    tm = build_train_model(tiny).eval()
    with tempfile.TemporaryDirectory() as tmp:
        export(tm.state_dict(), tmp, tiny)
        ids = torch.tensor([tok(sents[0])["input_ids"]])
        with torch.no_grad():
            res["tiny non-default size (patched config): exported vs train-side, max|dlogit| causal"] = \
                (_hf_logits(tmp, ids, True) - _train_logits(tm, ids, True)).abs().max().item()
            res["tiny non-default size (patched config): exported vs train-side, max|dlogit| masked"] = \
                (_hf_logits(tmp, ids, False) - _train_logits(tm, ids, False)).abs().max().item()
    for k, v in res.items():
        print(f"{k}: {v:.3e}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--published", default=PUBLISHED)
    ap.add_argument("--state", help="train-side state_dict .pt (model.pt / ema.pt written by train.py)")
    ap.add_argument("--model_config", default=None, help="json with model fields (default: published small config)")
    ap.add_argument("--out_dir")
    a = ap.parse_args()
    if a.verify:
        r = verify(a.published)
        json.dump(r, open(os.path.join(HERE, "verify_export_result.json"), "w"), indent=2)
        return
    cfg = json.load(open(a.model_config)) if a.model_config else None
    sd = torch.load(a.state, map_location="cpu")
    print(export(sd, a.out_dir, cfg, {"source_state": os.path.abspath(a.state)}))


if __name__ == "__main__":
    main()
