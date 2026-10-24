"""Export sprint1_port checkpoints to HF directories loadable by the official BabyLM 2025 pipeline (backend causal).

    python babylm/recipes/sprint1_port/export_hf.py --run_dir <run> --out <run>/hf          # every ckpt/*.pt
    python babylm/recipes/sprint1_port/export_hf.py --ckpt <run>/ckpt/final.pt --out <dir>/final

Each export dir: config.json (auto_map -> modeling_sprint1.py, trust_remote_code), modeling_sprint1.py,
model.safetensors (float32, keys "model.<sprint-1 state_dict key>"), tokenizer.json / tokenizer_config.json /
special_tokens_map.json / vocab.json / merges.txt, export_info.json.

Tokenizer: openai-community/gpt2 byte-level BPE (unchanged vocab/merges/pre-tokenizer) with one change: the
post-processor prepends <|endoftext|> (id 50256) to every encoded text, i.e. bos = eos = pad = unk = <|endoftext|>.
Why: the official causal evaluation (sentence_zero_shot/dataset.py) tokenises each sentence with the tokenizer's own
special tokens, feeds tokens[:-1] and scores tokens[1:], so without a BOS the first word of every sentence is never
scored (the official GPT-2 baseline's tokenizer prepends <s> the same way). <|endoftext|> is what precedes every
document in our training stream, so position 0 = EOT matches training. Padding is on the right with id 50256; the
model is causal so right padding cannot change real-token logits (checked by verify_export.py).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "hf_template" / "modeling_sprint1.py"
TOK_DIR = HERE / "gpt2_tokenizer"
EOT_STR = "<|endoftext|>"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tokenizer(out: Path) -> None:
    from tokenizers import Tokenizer, processors

    tok = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))
    tok.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=False),  # the original gpt2 post-processor (offsets unchanged)
        processors.TemplateProcessing(single=f"{EOT_STR} $A", pair=f"{EOT_STR} $A {EOT_STR} $B",
                                      special_tokens=[(EOT_STR, 50256)]),
    ])
    tok.save(str(out / "tokenizer.json"))
    for f in ("vocab.json", "merges.txt"):
        shutil.copy(TOK_DIR / f, out / f)
    sp = {"bos_token": EOT_STR, "eos_token": EOT_STR, "pad_token": EOT_STR, "unk_token": EOT_STR}
    (out / "special_tokens_map.json").write_text(json.dumps(sp, indent=1))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "GPT2TokenizerFast", **sp, "add_prefix_space": False, "add_bos_token": False,
        "model_max_length": 1024, "padding_side": "right", "clean_up_tokenization_spaces": False,
        "errors": "replace"}, indent=1))


def export(ckpt_path: Path, out: Path, autocast_bf16: bool = False) -> Path:
    from safetensors.torch import save_file

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mm = ck.get("model_module", "sprint1.models.baseline")
    if mm != "sprint1.models.baseline":
        raise SystemExit(f"{ckpt_path}: model {mm!r} is not the sprint-1 baseline; this exporter only wraps baseline.py")
    cfg = ck["cfg"]
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("modeling_sprint1_tmpl", TEMPLATE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    hc = m.Sprint1Config(vocab_size=cfg["vocab_size"], seq_len=cfg["seq_len"], n_layer=cfg["n_layer"],
                         n_head=cfg["n_head"], n_embd=cfg["n_embd"], mlp_ratio=cfg["mlp_ratio"],
                         tie_embeddings=cfg["tie_embeddings"], extra=cfg.get("extra", {}), autocast_bf16=autocast_bf16)
    hc.architectures = ["Sprint1ForCausalLM"]
    hc.auto_map = {"AutoConfig": "modeling_sprint1.Sprint1Config",
                   "AutoModel": "modeling_sprint1.Sprint1Model",
                   "AutoModelForCausalLM": "modeling_sprint1.Sprint1ForCausalLM"}
    hc.torch_dtype = "float32"
    hc.save_pretrained(out)
    shutil.copy(TEMPLATE, out / "modeling_sprint1.py")
    if cfg["tie_embeddings"]:
        raise SystemExit("tied embeddings not supported by this exporter (the frozen recipe is untied)")
    sd = {f"model.{k}": v.detach().to(torch.float32).contiguous() for k, v in ck["model"].items()}
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    write_tokenizer(out)
    info = {"source_ckpt": str(ckpt_path), "source_ckpt_sha256": sha256_file(ckpt_path), "name": ck.get("name"),
            "step": ck.get("step"), "tokens_seen": ck.get("tokens_seen"), "words_seen_est": ck.get("words_seen_est"),
            "epoch": ck.get("epoch"), "model_cfg": cfg,
            "files_sha256": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "export_info.json"}}
    (out / "export_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[export] {ckpt_path.name} -> {out} (step {ck.get('step')}, ~{(ck.get('words_seen_est') or 0) / 1e6:.2f}M words)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--run_dir", default=None, help="export every <run_dir>/ckpt/*.pt to <out>/<name>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--final_as", default=None, help="also export final.pt under this name (e.g. chck_100M)")
    ap.add_argument("--autocast_bf16", action="store_true", help="CUDA forward under bf16 autocast (training numerics)")
    a = ap.parse_args()
    if a.ckpt:
        export(Path(a.ckpt), Path(a.out), a.autocast_bf16)
    elif a.run_dir:
        for p in sorted((Path(a.run_dir) / "ckpt").glob("*.pt")):
            export(p, Path(a.out) / p.stem, a.autocast_bf16)
            if p.stem == "final" and a.final_as:
                export(p, Path(a.out) / a.final_as, a.autocast_bf16)
    else:
        raise SystemExit("give --ckpt or --run_dir")


if __name__ == "__main__":
    main()
