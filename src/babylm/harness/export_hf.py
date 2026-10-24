"""Export harness checkpoints to Hugging Face directories the BabyLM 2025 evaluation pipeline can load.

One checkpoint -> one self-contained directory:
    python -m babylm.harness.export_hf --ckpt RUN/checkpoints/chck_100M.pt --out EXPORT_DIR [--verify]
A whole run (every milestone checkpoint) -> per-revision directories plus an offline Hugging Face hub cache:
    python -m babylm.harness.export_hf --run_dir RUN --out EXPORT_ROOT --name NAME [--org hinton] [--verify]
  EXPORT_ROOT/NAME/chck_1M/ ... EXPORT_ROOT/NAME/chck_100M/   self-contained dirs (main = the last milestone)
  EXPORT_ROOT/hub/models--ORG--NAME/refs/{chck_1M,...,main}   files holding a 40-hex snapshot id
  EXPORT_ROOT/hub/models--ORG--NAME/snapshots/<id> -> ../../../NAME/<rev>   (relative symlinks)
The pipeline loads checkpoints with from_pretrained(model, revision="chck_XM"), which ignores `revision` for a local
directory; with the hub cache it works unmodified, offline:
    HF_HUB_OFFLINE=1 HF_HUB_CACHE=EXPORT_ROOT/hub bash eval_zero_shot_fast_all_revisions.sh ORG/NAME causal strict-small
    HF_HUB_OFFLINE=1 HF_HUB_CACHE=EXPORT_ROOT/hub bash eval_aoa.sh ORG/NAME causal strict-small
    HF_HUB_OFFLINE=1 HF_HUB_CACHE=EXPORT_ROOT/hub bash eval_zero_shot.sh ORG/NAME causal      (main = final model)
Each directory: config.json (auto_map: AutoConfig, AutoModel = HintonModel (final normalised hidden states, for the
finetuning classifier; hidden_size = n_embd), AutoModelForCausalLM = HintonForCausalLM; modeling_hinton.py is a
standalone copy of the sprint-1 baseline; tie_word_embeddings false), model.safetensors (fp32), tokenizer.json + tokenizer_config.json + special_tokens_map.json
(PreTrainedTokenizerFast; its post-processor PREPENDS <|endoftext|> as BOS to every text, eos = pad = BOS, padding
right), hinton_export.json (provenance). A checkpoint trained with the MNTP hybrid (checkpoint field mntp_frac > 0, or
--automodel_attention bidirectional) gets config.automodel_bidirectional = true: AutoModel attends bidirectionally,
as GPT-BERT's does, because the official finetuning classifier pools encoding[:, 0] (BOS) for every model;
AutoModelForCausalLM stays causal. Recorded as "automodel_attention" in hinton_export.json. --automodel_pool last
(causal AutoModels only; critic ref-0058 / ref-0059) writes each row's last real token state into position 0 of the
AutoModel output, so the official classifier's encoding[:, 0] pools the last token; default first = official BOS
pooling. Recorded as config.automodel_pool and "automodel_pool" in hinton_export.json; GLUE scored with last is
reported next to the BOS-pooled number, never instead of it. --verify: max |logit| and |hidden| diff vs the harness model < 1e-4 (CPU
fp32), also under right padding with 2-D and 3-D (finetuning --causal) masks; tokenizer prepends one BOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from babylm.data import corpus as C

TEMPLATE = Path(__file__).resolve().parent / "hf_template"
ARCH_KEYS = {"n_value_embeds", "softcap", "attn_scale", "rope_base", "rope_frac", "unet", "x0_inject", "embnorm",
             "n_loops"}
TRAIN_ONLY_KEYS = {"dropout", "head_init_std", "muon_momentum", "muon_wd", "adam_betas", "adam_eps", "adam_wd"}
SUPPORTED_MODELS = {"baseline", "baseline_reg", "looped"}


def hf_config(cfg: dict[str, Any], automodel_bidirectional: bool = False, automodel_pool: str = "first") -> dict[str, Any]:
    extra = dict(cfg.get("extra") or {})
    unknown = set(extra) - ARCH_KEYS - TRAIN_ONLY_KEYS
    if unknown:
        raise SystemExit(f"cfg.extra has keys the HF port does not implement: {sorted(unknown)}")
    c = {"model_type": "hinton_gpt", "architectures": ["HintonForCausalLM"],
         "auto_map": {"AutoConfig": "configuration_hinton.HintonConfig",
                      "AutoModel": "modeling_hinton.HintonModel",
                      "AutoModelForCausalLM": "modeling_hinton.HintonForCausalLM"},
         "hidden_size": cfg["n_embd"],
         "vocab_size": cfg["vocab_size"], "n_layer": cfg["n_layer"], "n_head": cfg["n_head"], "n_embd": cfg["n_embd"],
         "mlp_ratio": cfg["mlp_ratio"], "seq_len": cfg["seq_len"], "n_value_embeds": extra.get("n_value_embeds"),
         "softcap": float(extra.get("softcap", 15.0)), "attn_scale": extra.get("attn_scale"),
         "rope_base": float(extra.get("rope_base", 1024.0)), "rope_frac": float(extra.get("rope_frac", 0.25)),
         "unet": bool(extra.get("unet", True)), "x0_inject": bool(extra.get("x0_inject", True)),
         "embnorm": bool(extra.get("embnorm", True)), "n_loops": int(extra.get("n_loops", 1)),
         "tie_word_embeddings": False,
         "automodel_bidirectional": bool(automodel_bidirectional), "automodel_pool": automodel_pool,
         "bos_token_id": C.BOS_ID, "eos_token_id": C.BOS_ID, "pad_token_id": C.BOS_ID,
         "torch_dtype": "float32", "transformers_version": __import__("transformers").__version__}
    # tied embeddings (--tie_embeddings) export as two equal untied matrices: same outputs, plain HF loading
    return c


def save_tokenizer(out: Path, seq_len: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer.from_file(str(C.TOKENIZER_PATH))
    tok.post_processor = TemplateProcessing(single=f"{C.BOS_TOKEN} $A", pair=f"{C.BOS_TOKEN} $A {C.BOS_TOKEN} $B:1",
                                            special_tokens=[(C.BOS_TOKEN, C.BOS_ID)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=C.BOS_TOKEN, eos_token=C.BOS_TOKEN,
                                   pad_token=C.BOS_TOKEN, padding_side="right", model_max_length=seq_len,
                                   clean_up_tokenization_spaces=False)
    fast.save_pretrained(str(out))


def automodel_mode(ck: dict[str, Any], override: str = "auto") -> bool:
    """True = bidirectional AutoModel: 'auto' follows the checkpoint's mntp_frac."""
    if override != "auto":
        return override == "bidirectional"
    return float(ck.get("mntp_frac") or 0.0) > 0


def export_checkpoint(ckpt_path: Path, out: Path, automodel_attention: str = "auto",
                      automodel_pool: str = "first") -> dict[str, Any]:
    from safetensors.torch import save_file

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("model_name", "baseline") not in SUPPORTED_MODELS:
        raise SystemExit(f"{ckpt_path}: model {ck.get('model_name')!r} has no HF port")
    out.mkdir(parents=True, exist_ok=True)
    cfg = ck["cfg"]
    bidir = automodel_mode(ck, automodel_attention)
    if automodel_pool not in ("first", "last"):
        raise SystemExit(f"automodel_pool must be first or last, got {automodel_pool!r}")
    if automodel_pool == "last" and bidir:
        raise SystemExit(f"{ckpt_path}: --automodel_pool last is for causal AutoModels only (this export is bidirectional)")
    (out / "config.json").write_text(json.dumps(hf_config(cfg, bidir, automodel_pool), indent=2))
    for f in ("configuration_hinton.py", "modeling_hinton.py"):
        shutil.copyfile(TEMPLATE / f, out / f)
    sd = {k: v.detach().float().contiguous().clone() for k, v in ck["model"].items()}
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    save_tokenizer(out, cfg["seq_len"])
    prov = {"source_checkpoint": str(ckpt_path), "words_seen": ck.get("words_seen"),
            "words_nominal": ck.get("words_nominal"), "tokens_seen": ck.get("tokens_seen"), "step": ck.get("step"),
            "label": ck.get("label"), "model_name": ck.get("model_name"), "git_commit": ck.get("git_commit"),
            "tokenizer_sha256": C.sha256_file(C.TOKENIZER_PATH), "cfg": cfg,
            "mntp_frac": ck.get("mntp_frac"), "ema_decay": ck.get("ema_decay"),
            "automodel_attention": "bidirectional" if bidir else "causal", "automodel_pool": automodel_pool}
    (out / "hinton_export.json").write_text(json.dumps(prov, indent=2, default=str))
    return prov


def harness_model(ckpt_path: Path):
    from sprint1.harness.model_api import ModelConfig
    from sprint1.models import baseline

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"])
    if ck.get("model_name") == "looped":
        from babylm.models import looped
        m = looped.build(cfg)
    else:
        m = baseline.build(cfg)
    m.load_state_dict(ck["model"])
    from babylm.harness.levers import make_switchable
    make_switchable(m)  # causal by default: identical outputs; BidirectionalAttention flips it
    return m.eval(), cfg


@torch.no_grad()
def verify_export(out: Path, ckpt_path: Path, n: int = 2, T: int = 48, seed: int = 0) -> float:
    """max |logit diff| between the exported HF model and the harness model on random ids (CPU fp32)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ref, cfg = harness_model(ckpt_path)
    hf = AutoModelForCausalLM.from_pretrained(str(out), trust_remote_code=True, torch_dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, cfg.vocab_size, (n, T), generator=g)
    a = ref.logits(ref.hidden(ids))
    b = hf(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    diff = float((a - b).abs().max())
    # right padding must not change real positions
    am = torch.ones_like(ids)
    am[:, T // 2:] = 0
    c = hf(input_ids=ids, attention_mask=am).logits
    diff = max(diff, float((a[:, :T // 2] - c[:, :T // 2]).abs().max()))
    # AutoModel (finetuning trunk): final normalised hidden == harness hidden(); right padding (2-D key mask or the
    # finetuning collate's 3-D causal mask) leaves real positions unchanged
    from transformers import AutoConfig, AutoModel

    base = AutoModel.from_pretrained(str(out), trust_remote_code=True, torch_dtype=torch.float32).eval()
    hcfg = AutoConfig.from_pretrained(str(out), trust_remote_code=True)
    if hcfg.hidden_size != cfg.n_embd:
        raise AssertionError("config.hidden_size != n_embd")
    h = base(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    if h.shape != (n, T, cfg.n_embd):
        raise AssertionError(f"last_hidden_state shape {tuple(h.shape)}")
    h2 = base(input_ids=ids, attention_mask=am).last_hidden_state
    if getattr(hcfg, "automodel_bidirectional", False):
        # bidirectional trunk == harness model with bidirectional attention; key padding hides the pads, so the
        # real half equals the harness on the unpadded half alone
        from babylm.harness.levers import BidirectionalAttention
        with BidirectionalAttention(ref):
            href = ref.hidden(ids)
            href_half = ref.hidden(ids[:, :T // 2])
        diff = max(diff, float((h - href).abs().max()), float((h2[:, :T // 2] - href_half).abs().max()))
    else:
        href = ref.hidden(ids)
        am3 = am.unsqueeze(1).repeat(1, T, 1).tril(diagonal=0)
        h3 = base(input_ids=ids, attention_mask=am3).last_hidden_state
        if getattr(hcfg, "automodel_pool", "first") == "last":
            # position 0 = last real token (T-1 unpadded, T//2-1 padded); positions 1.. unchanged
            diff = max(diff, float((h[:, 0] - href[:, -1]).abs().max()),
                       float((h2[:, 0] - href[:, T // 2 - 1]).abs().max()),
                       float((h3[:, 0] - href[:, T // 2 - 1]).abs().max()))
            h, h2, h3 = h[:, 1:], h2[:, 1:], h3[:, 1:]
            href = href[:, 1:]
        diff = max(diff, float((h - href).abs().max()))
        r = T // 2 - (T - href.size(1))  # real positions left after dropping pooled position 0
        diff = max(diff, float((h2 - href)[:, :r].abs().max()), float((h3 - href)[:, :r].abs().max()))
    tok = AutoTokenizer.from_pretrained(str(out))
    enc = tok("The cat sat.")
    if enc["input_ids"][0] != C.BOS_ID or enc["input_ids"].count(C.BOS_ID) != 1:
        raise AssertionError(f"tokenizer does not prepend exactly one BOS: {enc['input_ids']}")
    if tok.padding_side != "right" or tok.pad_token_id != C.BOS_ID or tok.eos_token_id != C.BOS_ID:
        raise AssertionError("tokenizer pad/eos/padding side wrong")
    return diff


def snapshot_id(rev_dir: Path, rev: str) -> str:
    h = hashlib.sha1(rev.encode())
    h.update(C.sha256_file(rev_dir / "model.safetensors").encode())
    return h.hexdigest()


def export_run(run_dir: Path, root: Path, name: str, org: str, verify: bool,
               automodel_attention: str = "auto", automodel_pool: str = "first") -> dict[str, Any]:
    cks = sorted((run_dir / "checkpoints").glob("chck_*M.pt"), key=lambda p: int(p.stem[5:-1]))
    if not cks:
        raise SystemExit(f"no checkpoints in {run_dir / 'checkpoints'} (train with --save)")
    model_dir = root / name
    repo = root / "hub" / f"models--{org}--{name}"
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "snapshots").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"run_dir": str(run_dir), "repo_id": f"{org}/{name}", "revisions": {}}
    for ck in cks:
        rev = ck.stem
        d = model_dir / rev
        prov = export_checkpoint(ck, d, automodel_attention, automodel_pool)
        sid = snapshot_id(d, rev)
        link = repo / "snapshots" / sid
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(d, link.parent), link)
        (repo / "refs" / rev).write_text(sid)
        rec = {"dir": str(d), "snapshot": sid, "words_seen": prov["words_seen"]}
        if verify:
            rec["max_abs_logit_diff"] = verify_export(d, ck)
        report["revisions"][rev] = rec
        print(f"[export] {rev}: {d} words_seen={prov['words_seen']}"
              + (f" diff={rec['max_abs_logit_diff']:.2e}" if verify else ""), flush=True)
    last = cks[-1].stem
    (repo / "refs" / "main").write_text(report["revisions"][last]["snapshot"])
    report["main"] = last
    (model_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="one checkpoint (checkpoints/chck_XM.pt)")
    ap.add_argument("--run_dir", default=None, help="export every milestone checkpoint of a run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="model name for --run_dir (default: the run dir name)")
    ap.add_argument("--org", default="hinton")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--automodel_attention", default="auto", choices=["auto", "causal", "bidirectional"],
                    help="AutoModel (finetuning trunk) attention; auto = bidirectional iff trained with MNTP")
    ap.add_argument("--automodel_pool", default="first", choices=["first", "last"],
                    help="AutoModel position 0: first = official BOS pooling; last = last real token (causal only)")
    a = ap.parse_args(argv)
    if bool(a.ckpt) == bool(a.run_dir):
        raise SystemExit("give exactly one of --ckpt / --run_dir")
    if a.ckpt:
        out = Path(a.out)
        export_checkpoint(Path(a.ckpt), out, a.automodel_attention, a.automodel_pool)
        if a.verify:
            d = verify_export(out, Path(a.ckpt))
            print(f"[export] max |logit diff| {d:.3e}", flush=True)
            if d >= 1e-4:
                sys.exit(1)
        print(f"[export] {out}", flush=True)
    else:
        rep = export_run(Path(a.run_dir), Path(a.out), a.name or Path(a.run_dir).name, a.org, a.verify,
                         a.automodel_attention, a.automodel_pool)
        bad = [r for r, v in rep["revisions"].items() if v.get("max_abs_logit_diff", 0) >= 1e-4]
        if bad:
            print(f"[export] logit check FAILED for {bad}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
