"""Score a causal checkpoint on the frozen slicing samples, per token and per word.

For every sample chunk the text is tokenized with the checkpoint's own
tokenizer, a BOS token is prepended, and every token is scored causally
(512-token windows; longer chunks continue with stride 256 so each token
keeps at least 256 tokens of context). Token losses are summed into
whitespace words (a token belongs to the word it starts in, or to the next
word if it is whitespace), so results are comparable across tokenizers.

Outputs data/babylm_scores/<run>/<revision>/<split>.npz with
  word_nll (nats), word_ntok, word_top1 (every token of the word is argmax),
  tok_nll, tok_top1, tok_pmax, tok_ent, tok_word, tok_pos, tok_chunk
and meta.json (weights hash, dtype, timings).

Usage:
  python -m babylm.instruments.score --run gpt2_10m --line-join newline_space \
      --repo BabyLM-community/babylm-baseline-10m-gpt2 --revisions chck_10M,chck_100M
  python -m babylm.instruments.score --run my_model --local-path path/to/hf_dir --line-join newline
  python -m babylm.instruments.score --run port_S_10M_e10_s0 --local-path <harness HF export> --line-join newline --collapse
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "data" / "babylm_instruments"
VERSION = "v1"
GPTBERT_AMP = False


def load_rows(split: str) -> list[dict]:
    from babylm.instruments.corpus import sname

    with open(SAMPLES / f"{sname(split)}_sample_{VERSION}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fetch(repo: str, revision: str, dest: Path) -> tuple[Path, dict]:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    path = Path(snapshot_download(repo, revision=revision, local_dir=dest,
                                  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"]))
    return path, {"repo": repo, "revision": revision, "commit": info.sha, "weights_sha256": {k: v for k, v in files.items() if v}}


# sha256 of babylm/harness/hf_template/{modeling,configuration}_hinton.py versions we have read (our own harness,
# exported next to each checkpoint). Unknown code is refused unless --trust-model-code; it is never fetched remotely.
HINTON_CODE = {
    "0bd7842bd79b8bd58b8013016a53b4fedc8123667dbac8237d92ec2c10c73ae5",  # modeling, causal only
    "076a5e583052cea14d8e63c1e42c28dd88f0eca9b170f89fef3f331bf3776936",  # modeling, + bidirectional / last-pool AutoModel
    "709e8e82af219e07e294c399a2cf0cb0b311bd5ed2e12c8d23c1d2d6530e98ff",  # configuration
    "d9968dc4f28ef5febfb68e621c9899f80ee1602d8e8d5725ded02e01ac00fb0c",  # configuration
}
TRUST_MODEL_CODE = False


def load_hinton(path: Path):
    """A babylm.harness export (model_type hinton_gpt): import its two code files as a private package after
    checking their hashes, build HintonForCausalLM and load model.safetensors strictly."""
    import hashlib
    import importlib
    import sys

    from safetensors.torch import load_file

    files = ("configuration_hinton.py", "modeling_hinton.py")
    digests = [hashlib.sha256((path / f).read_bytes()).hexdigest() for f in files]
    unknown = [f for f, d in zip(files, digests) if d not in HINTON_CODE]
    if unknown and not TRUST_MODEL_CODE:
        raise RuntimeError(f"unreviewed model code {unknown} in {path} ({digests}); read it, then add its sha256 or pass --trust-model-code")
    pkg = REPO / "data" / "babylm_model_code" / f"hinton_{digests[1][:12]}_{digests[0][:12]}"
    pkg.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(path / f, pkg / f)
    (pkg / "__init__.py").touch()
    sys.path.insert(0, str(pkg.parent))
    mod = importlib.import_module(f"{pkg.name}.modeling_hinton")
    conf_mod = importlib.import_module(f"{pkg.name}.configuration_hinton")
    cfg = json.loads((path / "config.json").read_text())
    config = conf_mod.HintonConfig(**{k: v for k, v in cfg.items() if k not in ("auto_map", "architectures", "model_type")})
    model = mod.HintonForCausalLM(config)
    missing, unexpected = model.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"hinton state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model, digests


def load_model(path: Path, dtype: torch.dtype):
    from transformers import PreTrainedTokenizerFast

    cfg = json.loads((path / "config.json").read_text())
    # tokenizer.json directly: never goes through auto_map / remote code
    special = {}
    if (path / "special_tokens_map.json").exists():
        for k, v in json.loads((path / "special_tokens_map.json").read_text()).items():
            if isinstance(v, (str, dict)):
                special[k] = v["content"] if isinstance(v, dict) else v
    tok = PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"), **special)
    if cfg.get("model_type") == "gpt2":
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained(path)
        family = "gpt2"
    elif cfg.get("model_type") == "hinton_gpt":
        model, _ = load_hinton(path)
        family = "hinton"
    elif "GPTBERTFoCausalLM" in cfg.get("architectures", []):
        from babylm.instruments.vendor_gptbert.configuration_gpt_bert import ModelConfig
        from babylm.instruments.vendor_gptbert.modeling_gpt_bert import GPTBERTForCausalLM

        conf = ModelConfig(**{k: v for k, v in cfg.items() if k not in ("auto_map", "architectures")})
        model = GPTBERTForCausalLM(conf)
        sd = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith("position_indices")]
        if missing or [k for k in unexpected if not k.startswith("head")]:
            raise RuntimeError(f"GPT-BERT state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        family = "gpt_bert"
        from babylm.instruments import gptbert_fast

        gptbert_fast.check(model.eval())  # exact match with the published attention, then keep the fast path
    else:
        raise ValueError(f"unsupported config: {cfg.get('model_type')} {cfg.get('architectures')}")
    # GPT-BERT's DenseFormer accumulator cannot hold bf16 weights; it runs fp32 weights,
    # under bf16 autocast only if --gptbert-amp is given (no faster for the 31M model).
    amp = family == "gpt_bert" and dtype == torch.bfloat16 and GPTBERT_AMP
    model = model.eval().to(torch.float32 if family == "gpt_bert" else dtype)
    model.amp = amp
    n_params = sum(p.numel() for p in model.parameters())
    return model, tok, family, n_params


def windows_for(ids: list[int], max_len: int, stride: int):
    """Yield (start, end, first_scored) over the BOS-prefixed id list; targets are ids[1:]."""
    n = len(ids)
    if n <= max_len:
        yield 0, n, 1
        return
    end = max_len
    yield 0, end, 1
    while end < n:
        new_end = min(end + stride, n)
        start = new_end - max_len
        yield start, new_end, end
        end = new_end


LINE_JOINS = {"newline": "\n", "newline_space": "\n "}


def format_text(r: dict, line_join: str = "newline", tab_space: tuple = (), collapse: bool = False) -> str:
    """A sample chunk's text as the model is scored on it (see tokenize_chunks)."""
    text = r["text"]
    if collapse:
        text = "\n".join(c for c in (" ".join(line.split()) for line in text.split("\n")) if c)
    sep = LINE_JOINS[line_join]
    text = text.replace("\n", sep) if sep != "\n" else text
    if r["source"] in tab_space:  # speaker-label tabs the model never saw in training
        text = text.replace("\t", " ")
    return text


def tokenize_chunks(tok, rows: list[dict], line_join: str = "newline", tab_space: tuple = (), collapse: bool = False):
    """Per chunk: BOS-prefixed ids and, for each target token, its word index in the chunk.

    line_join must match how the model saw line breaks in training: the GPT-2 baselines
    were trained on ' '.join(f.readlines()), i.e. every newline followed by a space
    (momergul/babylm-gpt2-baseline data_utils.py). collapse rewrites every line as
    ' '.join(line.split()), as babylm.data.corpus does for the harness and GPT-BERT ports
    (lines left empty are dropped). Words are unchanged either way."""
    bos = tok.bos_token_id
    out = []
    for r in rows:
        text = format_text(r, line_join, tab_space, collapse)
        ends = np.array([m.end() for m in re.finditer(r"\S+", text)])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        starts = np.array([s for s, _ in enc["offset_mapping"]], dtype=np.int64)
        widx = np.searchsorted(ends, starts, side="right")
        widx = np.minimum(widx, len(ends) - 1)
        if len(widx) and np.any(np.diff(widx) < 0):
            raise RuntimeError(f"non-monotone token->word map in {r['id']}")
        out.append(([bos] + list(enc["input_ids"]), widx, len(ends)))
    return out


@torch.inference_mode()
def score_split(model, tok, rows, family: str, max_len=512, stride=256, batch_tokens=8192, temps=(), line_join="newline", tab_space=(),
                collapse=False):
    chunks = tokenize_chunks(tok, rows, line_join, tab_space, collapse)
    jobs = []  # (chunk, start, end, first_scored)
    for ci, (ids, _, _) in enumerate(chunks):
        for s, e, f in windows_for(ids, max_len, stride):
            jobs.append((ci, s, e, f))
    jobs.sort(key=lambda j: j[2] - j[1])
    n_tok = [len(ids) - 1 for ids, _, _ in chunks]
    res = {k: [np.zeros(n, dtype=dt) for n in n_tok] for k, dt in
           (("nll", np.float32), ("top1", bool), ("pmax", np.float16), ("ent", np.float16), ("pos", np.int16))}
    res_t = [[np.zeros(n, dtype=np.float32) for n in n_tok] for _ in temps]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    i = 0
    while i < len(jobs):
        L = jobs[i][2] - jobs[i][1]
        j = i
        while j < len(jobs) and (j - i + 1) * (jobs[j][2] - jobs[j][1]) <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        batch = jobs[i:j]
        L = max(e - s for _, s, e, _ in batch)
        x = torch.full((len(batch), L), pad, dtype=torch.long)
        for b, (ci, s, e, _) in enumerate(batch):
            x[b, : e - s] = torch.tensor(chunks[ci][0][s:e])
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=getattr(model, "amp", False)):
            out = model(input_ids=x)
        logits = out.logits if hasattr(out, "logits") else out[0]  # [B, T, V] for both families
        for b, (ci, s, e, f) in enumerate(batch):
            ids = chunks[ci][0]
            # position p in the window predicts ids[s + p + 1]; score targets with absolute index in [f, e)
            p0, p1 = f - s - 1, e - s - 1
            lg = logits[b, p0:p1].float()
            tgt = torch.tensor(ids[f:e])
            logp = F.log_softmax(lg, dim=-1)
            nll = -logp.gather(1, tgt[:, None])[:, 0]
            pm, am = logp.max(dim=-1)
            ent = -(logp.exp() * logp).sum(-1)
            sl = slice(f - 1, e - 1)  # target index t -> slot t-1
            res["nll"][ci][sl] = nll.numpy()
            res["top1"][ci][sl] = (am == tgt).numpy()
            res["pmax"][ci][sl] = pm.exp().numpy()
            res["ent"][ci][sl] = ent.numpy()
            res["pos"][ci][sl] = np.arange(p0 + 1, p1 + 1)  # context length incl. BOS
            for ti, T in enumerate(temps):  # same logits, softmax temperature T (calibration check)
                lt = F.log_softmax(lg / T, dim=-1)
                res_t[ti][ci][sl] = (-lt.gather(1, tgt[:, None])[:, 0]).numpy()
        i = j
    word_offsets = np.cumsum([0] + [nw for _, _, nw in chunks])
    tok_word = np.concatenate([w + word_offsets[ci] for ci, (_, w, _) in enumerate(chunks)])
    tok_chunk = np.concatenate([np.full(len(w), ci, dtype=np.int32) for ci, (_, w, _) in enumerate(chunks)])
    tok = {k: np.concatenate(v) for k, v in res.items()}
    n_words = int(word_offsets[-1])
    word_nll = np.bincount(tok_word, weights=tok["nll"], minlength=n_words).astype(np.float32)
    word_ntok = np.bincount(tok_word, minlength=n_words).astype(np.int16)
    word_wrong = np.bincount(tok_word, weights=(~tok["top1"]).astype(np.float64), minlength=n_words)
    extra = {}
    if temps:
        tn = np.stack([np.concatenate(r) for r in res_t])
        extra = {"temps": np.asarray(temps, dtype=np.float32), "tok_nll_temps": tn,
                 "word_nll_temps": np.stack([np.bincount(tok_word, weights=t, minlength=n_words) for t in tn]).astype(np.float32)}
    return {
        **extra,
        "word_nll": word_nll, "word_ntok": word_ntok, "word_top1": word_wrong == 0,
        "tok_nll": tok["nll"], "tok_top1": tok["top1"], "tok_pmax": tok["pmax"], "tok_ent": tok["ent"],
        "tok_pos": tok["pos"], "tok_word": tok_word.astype(np.int32), "tok_chunk": tok_chunk,
    }


def merge_rescored(base: dict, part: dict | None, changed: np.ndarray, n_words: list[int]) -> dict:
    """Token arrays of every chunk, from part (the changed chunks only, scored in order) where changed, else from base,
    then the word arrays as in score_split. Chunks are scored independently, so the unchanged ones are reused as is."""
    keys = ("tok_nll", "tok_top1", "tok_pmax", "tok_ent", "tok_pos")
    offs = np.cumsum([0] + list(n_words))
    sub = np.flatnonzero(changed)
    sub_offs = np.cumsum([0] + [n_words[c] for c in sub])
    bnd = {"base": np.searchsorted(base["tok_chunk"], np.arange(len(changed) + 1))}
    if part is not None:
        bnd["part"] = np.searchsorted(part["tok_chunk"], np.arange(len(sub) + 1))
    out = {k: [] for k in keys + ("tok_word", "tok_chunk")}
    for ci in range(len(changed)):
        if changed[ci]:
            j = int(np.searchsorted(sub, ci))
            src, a, b = part, bnd["part"][j], bnd["part"][j + 1]
            w = src["tok_word"][a:b] - sub_offs[j] + offs[ci]
        else:
            src, a, b = base, bnd["base"][ci], bnd["base"][ci + 1]
            w = src["tok_word"][a:b]
        for k in keys:
            out[k].append(src[k][a:b])
        out["tok_word"].append(w.astype(np.int32))
        out["tok_chunk"].append(np.full(b - a, ci, dtype=np.int32))
    tok = {k: np.concatenate(v) for k, v in out.items()}
    n = int(offs[-1])
    wrong = np.bincount(tok["tok_word"], weights=(~tok["tok_top1"]).astype(np.float64), minlength=n)
    return {"word_nll": np.bincount(tok["tok_word"], weights=tok["tok_nll"], minlength=n).astype(np.float32),
            "word_ntok": np.bincount(tok["tok_word"], minlength=n).astype(np.int16), "word_top1": wrong == 0, **tok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="name for this model series, e.g. gpt2_10m")
    ap.add_argument("--repo")
    ap.add_argument("--revisions", default="main")
    ap.add_argument("--local-path", type=Path)
    ap.add_argument("--splits", default="dev,train")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", type=Path, default=REPO / "data" / "babylm_scores")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "babylm_ckpt_tmp")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep downloaded weights")
    ap.add_argument("--gptbert-amp", action="store_true")
    ap.add_argument("--line-join", required=True, choices=sorted(LINE_JOINS),
                    help="how training text joined lines: newline_space for the BabyLM GPT-2 baselines, newline for GPT-BERT")
    ap.add_argument("--tab-space", default="", help="sources whose tabs become spaces, e.g. childes,switchboard (see docs/babylm/instruments.md)")
    ap.add_argument("--collapse", action="store_true", help="each line becomes ' '.join(line.split()) (harness and GPT-BERT port format)")
    ap.add_argument("--revision-name", help="label for a --local-path checkpoint (default: the export's label, else the directory name)")
    ap.add_argument("--trust-model-code", action="store_true", help="load a harness export whose model code hash is not in HINTON_CODE")
    ap.add_argument("--reuse", help="run scored on the same revision in another format: copy its chunks whose text is "
                    "unchanged in this format and score only the rest")
    ap.add_argument("--temps", default="", help="e.g. 0.8,0.9,1.1,1.2: also score at these softmax temperatures, to <split>_temps.npz")
    args = ap.parse_args()
    from babylm.instruments.corpus import SAMPLE, sname

    global GPTBERT_AMP, TRUST_MODEL_CODE
    GPTBERT_AMP = args.gptbert_amp
    TRUST_MODEL_CODE = args.trust_model_code
    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    splits = args.splits.split(",")
    rows = {s: load_rows(s) for s in splits}
    tab_space = tuple(x for x in args.tab_space.split(",") if x)
    if args.local_path:
        exp = args.local_path / "hinton_export.json"
        label = json.loads(exp.read_text()).get("label") if exp.exists() else None
        revs = [args.revision_name or label or args.local_path.name]
    else:
        revs = args.revisions.split(",")
    for rev in revs:
        out_dir = args.out / args.run / rev
        temps = tuple(float(t) for t in args.temps.split(",") if t)
        suffix = "_temps" if temps else ""
        if all((out_dir / f"{sname(s)}{suffix}.npz").exists() for s in splits):
            print(f"skip {args.run}/{rev} (exists)", flush=True)
            continue
        t0 = time.time()
        if args.local_path:
            path, meta = args.local_path, {"local_path": str(args.local_path)}
        else:
            path, meta = fetch(args.repo, rev, args.cache / args.run / rev)
        t_dl = time.time() - t0
        model, tok, family, n_params = load_model(path, dtype)
        out_dir.mkdir(parents=True, exist_ok=True)
        eff = "fp32" if family == "gpt_bert" and not GPTBERT_AMP else args.dtype
        prev = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {}
        meta.update({"run": args.run, "family": family, "n_params": n_params, "dtype": eff, "line_join": args.line_join, "tab_space": sorted(tab_space),
                     "collapse": args.collapse,
                     "torch": torch.__version__, "threads": torch.get_num_threads(), "download_s": round(t_dl, 1),
                     "sample_version": VERSION, "sample": SAMPLE, "splits": prev.get("splits", {})})
        if args.reuse:
            bdir = args.out / args.reuse / rev
            bm = json.loads((bdir / "meta.json").read_text())
            if temps or bm.get("commit") != meta.get("commit") or bm.get("dtype") != eff:
                raise SystemExit(f"--reuse {args.reuse}/{rev}: needs no --temps and the same commit and dtype")
            bfmt = (bm["line_join"], tuple(bm["tab_space"]), bm.get("collapse", False))
            meta["reused_from"] = {"run": args.reuse, "format": list(bfmt)}
        for s in splits:
            t1 = time.time()
            if args.reuse:
                fmt = (args.line_join, tab_space, args.collapse)
                changed = np.array([format_text(x, *bfmt) != format_text(x, *fmt) for x in rows[s]])
                part = (score_split(model, tok, [x for x, c in zip(rows[s], changed) if c], family, line_join=args.line_join,
                                    tab_space=tab_space, collapse=args.collapse) if changed.any() else None)
                r = merge_rescored(dict(np.load(bdir / f"{sname(s)}.npz")), part, changed, [len(x["text"].split()) for x in rows[s]])
                meta["reused_from"][s] = {"chunks_rescored": int(changed.sum()), "chunks": len(changed)}
            else:
                r = score_split(model, tok, rows[s], family, temps=temps, line_join=args.line_join, tab_space=tab_space,
                                collapse=args.collapse)
            np.savez_compressed(out_dir / f"{sname(s)}{suffix}.npz", **r)
            nw, nt = len(r["word_nll"]), len(r["tok_nll"])
            if temps:
                meta.setdefault("temps", {})[sname(s)] = {str(T): float(r["word_nll_temps"][i].sum() / nw) for i, T in enumerate(temps)}
            meta["splits"][sname(s) + suffix] = {"words": nw, "tokens": nt, "seconds": round(time.time() - t1, 1),
                                 "nats_per_word": float(r["word_nll"].sum() / nw),
                                 "bits_per_word": float(r["word_nll"].sum() / nw / np.log(2)),
                                 "nats_per_token": float(r["tok_nll"].mean())}
            print(f"{args.run}/{rev} {sname(s)}{suffix}: {meta['splits'][sname(s) + suffix]}", flush=True)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        del model
        if not args.local_path and not args.keep:
            shutil.rmtree(args.cache / args.run / rev, ignore_errors=True)


if __name__ == "__main__":
    main()
