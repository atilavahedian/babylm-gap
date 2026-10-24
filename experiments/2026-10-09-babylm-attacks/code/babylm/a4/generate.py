"""Generation for attack A4 (card docs/babylm/attacks/A4_reexpression.md, "Generation"): one shard of synthetic text.

    python -m babylm.a4.generate --ckpt G/resume.pt --out OUT/a4a --seed_split train_10M --block_split train_10M \
        --words 29861319                                                       # A4a: real seeds, exact 8-gram block
    python -m babylm.a4.generate --ckpt GA/resume.pt --out OUT/a4b_A --seed_split train_10M_hB --words 9953773
    python -m babylm.a4.generate --ckpt G/resume.pt --out OUT/n1 --selfseed --block_split train_10M --words ...   # N1
    python -m babylm.a4.generate --ckpt G/resume.pt --out OUT/probe --seed_split train_10M --rows 2000  # audit probe

Rows: seeded = a prompt_len-token seed of --seed_split (data.seed_starts, in data.seed_order(--seed), used in that
order and again in the same order if the target needs more rows); --selfseed = BOS, then prompt_len tokens sampled
with BOS masked and no block (discarded), so only the seed's origin differs from the arm. Then n_new tokens at
--temperature from the full softmax (ids >= 16384 masked; a sampled BOS ends the row); with --block_split, every
token completing a --block_n-gram of that split's token stream is removed (ngram.BlockIndex: an exact draw from the
renormalised rest). Kept text = the continuation decoded and cut to whole words (a first word continuing the seed's
last word is dropped; if the row did not stop, so is a last word the length limit may have cut), lines normalised,
blank lines dropped. Batches run until the kept words reach --words (or --rows rows are written). First, cached
decoding is checked against the plain forward in fp32 (--check_tol) and under the sampling autocast (--check_tol_amp).
Output: OUT.jsonl.gz (per row: text, src, seed_pos, pass, n_tok, stopped, blocked_steps, blocked_mass) and OUT.json
(settings, checkpoint sha256, rows, kept words, stop rate, block statistics, generation FLOPs, wall time).
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from babylm.a4 import data as D
from babylm.a4 import kvcache as KV
from babylm.a4 import ngram as NG
from babylm.data import corpus as C
from babylm.harness import cycle as CY
from babylm.harness import levers as LV


def log(msg: str) -> None:
    print(f"[a4.generate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_generator(ckpt: str | Path, which: str, device: torch.device):
    """The checkpoint's evaluated weights (EMA if any) in sprint1.models.baseline: every harness model has its
    parameters and its eval-mode outputs (export_hf.harness_model builds the same)."""
    from sprint1.harness.model_api import ModelConfig
    from sprint1.models import baseline

    sd, cfg_d, words = CY.load_init(ckpt, which)
    cfg = ModelConfig.from_dict(cfg_d)
    m = baseline.build(cfg)
    m.load_state_dict(sd)
    return m.to(device).eval(), cfg, words


def cut_words(prompt_text: str, cont: str, stopped: bool) -> str:
    """The continuation cut to whole words (see the module docstring)."""
    if cont and not cont[0].isspace() and prompt_text and not prompt_text[-1].isspace():
        i = next((k for k, ch in enumerate(cont) if ch.isspace()), len(cont))
        cont = cont[i:]
    if not stopped and cont and not cont[-1].isspace():
        i = next((k for k in range(len(cont) - 1, -1, -1) if cont[k].isspace()), -1)
        cont = cont[:i + 1] if i >= 0 else ""
    return D.clean_doc(cont)


def decode_rows(tok, prompts: np.ndarray, conts: list[np.ndarray], stopped: np.ndarray) -> list[str]:
    """Kept text per row. The continuation is decoded in the context of its prompt (byte-level BPE: a character can
    straddle the boundary), falling back to decoding it alone."""
    pr = [p[p != C.BOS_ID].tolist() for p in prompts]
    p_txt = tok.decode_batch(pr, skip_special_tokens=False)
    full = tok.decode_batch([a + c.tolist() for a, c in zip(pr, conts)], skip_special_tokens=False)
    alone = tok.decode_batch([c.tolist() for c in conts], skip_special_tokens=False)
    out = []
    for pt, ft, al, st in zip(p_txt, full, alone, stopped):
        cont = ft[len(pt):] if ft.startswith(pt) else al
        out.append(cut_words(pt, cont, bool(st)))
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="resume.pt or chck_*.pt of the generator run")
    ap.add_argument("--which", choices=("eval", "raw"), default="eval")
    ap.add_argument("--out", required=True, help="output prefix: OUT.jsonl.gz and OUT.json")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--seed_split", default=None, help="seeded rows: the split the seeds come from")
    ap.add_argument("--selfseed", action="store_true", help="N1: BOS + prompt_len self-generated tokens (discarded)")
    ap.add_argument("--words", type=float, default=None, help="stop once the kept words reach this")
    ap.add_argument("--rows", type=int, default=None, help="stop after this many rows (the audit probe)")
    ap.add_argument("--block_split", default=None, help="exact n-gram block against this split's token stream")
    ap.add_argument("--block_n", type=int, default=8)
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--n_new", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--allowed_vocab", type=int, default=C.VOCAB_SIZE)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0, help="seed order and sampling")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_batches", type=int, default=None, help="safety stop")
    ap.add_argument("--check_tol", type=float, default=0.01,
                    help="refuse if cached decoding differs from the plain forward by more in fp32 (softcapped logits)")
    ap.add_argument("--check_tol_amp", type=float, default=1.0,
                    help="the same under the bf16 autocast of sampling (rounding only; a cache bug moves whole nats)")
    a = ap.parse_args(argv)
    if (a.words is None) == (a.rows is None):
        raise SystemExit("give exactly one of --words and --rows")
    if a.selfseed == (a.seed_split is not None):
        raise SystemExit("give exactly one of --seed_split and --selfseed")
    if a.seed_split == "test" or a.block_split == "test":
        raise SystemExit("the FINAL split is never read here")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    t0 = time.time()
    model, cfg, g_words = load_generator(a.ckpt, a.which, device)
    from babylm.data import prepare as P
    tok = P.load_tokenizer()
    nl = LV.newline_ids(tok)

    seeds = src_of = toks = None
    if a.seed_split:
        sd = C.load_split(a.seed_split, a.data_dir)
        toks = np.asarray(sd.tokens)
        seeds = D.seed_order(D.seed_starts(toks, nl, a.prompt_len), a.seed)
        src_of = np.asarray(sd.src)
        if len(seeds) == 0:
            raise SystemExit(f"{a.seed_split}: no seed of {a.prompt_len} tokens")
        log(f"{len(seeds):,} seeds of {a.prompt_len} tokens from {a.seed_split} ({sd.n_tokens:,} tokens)")
    block, block_info = None, None
    if a.block_split:
        bd = C.load_split(a.block_split, a.data_dir)
        block = NG.BlockIndex.build(np.asarray(bd.tokens), a.block_n, device)
        block_info = {"split": a.block_split, "sha256": bd.meta.get("sha256"), "n": a.block_n, "keys": len(block)}
        log(f"block: {len(block):,} {a.block_n}-gram keys of {a.block_split}")
    gen = torch.Generator(device=device).manual_seed(a.seed)
    probe = torch.from_numpy(np.stack([toks[q:q + a.prompt_len] for q in seeds[:4]]).astype(np.int64)) \
        if seeds is not None else torch.zeros(4, 1, dtype=torch.long)
    check = KV.self_check(model, probe.to(device))  # fp32: the cached path's correctness
    check_amp = KV.self_check(model, probe.to(device), amp=amp) if device.type == "cuda" else check
    log(f"self-check: cached vs plain forward max |logit diff| {check:.2e} fp32, {check_amp:.2e} bf16 autocast")
    if check > a.check_tol or check_amp > a.check_tol_amp:
        raise SystemExit(f"KV-cache decoding differs from the model's forward by {check:.3g} fp32 (tol {a.check_tol}), "
                         f"{check_amp:.3g} bf16 (tol {a.check_tol_amp})")

    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    jl = Path(str(out_p) + ".jsonl.gz")
    tot = {"rows": 0, "words": 0, "tokens_kept": 0, "stopped": 0, "empty": 0, "blocked_steps": 0,
           "live_steps": 0, "blocked_mass": 0.0, "flops": 0.0, "decode_steps": 0, "batches": 0}
    target_rows = a.rows
    pre = a.prompt_len if a.selfseed else 0
    n_new = pre + a.n_new
    with gzip.open(jl, "wt", encoding="utf-8") as f:
        while True:
            if a.words is not None and tot["words"] >= a.words:
                break
            if target_rows is not None and tot["rows"] >= target_rows:
                break
            if a.max_batches is not None and tot["batches"] >= a.max_batches:
                raise SystemExit(f"--max_batches {a.max_batches} reached before the target")
            B = a.batch if target_rows is None else min(a.batch, target_rows - tot["rows"])
            i0 = tot["rows"]
            if seeds is not None:
                k = (i0 + np.arange(B)) % len(seeds)
                pos = seeds[k]
                prompts = np.stack([toks[p:p + a.prompt_len] for p in pos]).astype(np.int64)
                srcs = src_of[pos].astype(int).tolist()
                passes = ((i0 + np.arange(B)) // len(seeds)).tolist()
            else:
                pos = np.full(B, -1)
                prompts = np.full((B, 1), C.BOS_ID, dtype=np.int64)
                srcs, passes = [D.SELF_SRC] * B, [0] * B
            pt = torch.from_numpy(prompts).to(device)
            so = KV.sample(model, pt, n_new, temperature=a.temperature, allowed_vocab=a.allowed_vocab, block=block,
                           block_from=pre, stop_id=C.BOS_ID, ban_stop_steps=pre, generator=gen, amp=amp)
            out = so.tokens.cpu().numpy()
            lengths = so.lengths.cpu().numpy()
            stopped = so.stopped.cpu().numpy()
            if a.selfseed:  # the discarded self-generated seed becomes the decoding context
                prompts = np.concatenate([prompts, out[:, :pre]], 1)
            conts = [out[r, pre:lengths[r]] for r in range(B)]
            texts = decode_rows(tok, prompts, conts, stopped)
            bsteps = so.blocked_steps.cpu().numpy()
            bmass = so.blocked_mass.cpu().numpy()
            for r in range(B):
                w = len(texts[r].split())
                f.write(json.dumps({"i": i0 + r, "seed_pos": int(pos[r]), "src": int(srcs[r]), "pass": int(passes[r]),
                                    "n_tok": int(len(conts[r])), "stopped": bool(stopped[r]), "words": w,
                                    "blocked_steps": int(bsteps[r]), "blocked_mass": round(float(bmass[r]), 6),
                                    "text": texts[r]}, ensure_ascii=False) + "\n")
                tot["words"] += w
                tot["empty"] += int(w == 0)
                tot["tokens_kept"] += int(len(conts[r]))
                tot["live_steps"] += len(conts[r]) + int(stopped[r])  # live steps (the stop's included)
            tot["rows"] += B
            tot["stopped"] += int(stopped.sum())
            tot["blocked_steps"] += int(bsteps.sum())
            tot["blocked_mass"] += float(bmass.sum())
            tot["flops"] += so.stats["flops_per_row"] * B
            tot["decode_steps"] += so.stats["steps"]
            tot["batches"] += 1
            if tot["batches"] % 20 == 1:
                log(f"batch {tot['batches']}: {tot['rows']:,} rows, {tot['words']:,} words, "
                    f"stop {tot['stopped'] / tot['rows']:.3f}, {time.time() - t0:,.0f}s")
    rows = max(tot["rows"], 1)
    stats = {
        "ckpt": str(a.ckpt), "ckpt_sha256": C.sha256_file(a.ckpt), "weights": a.which, "generator_words_seen": g_words,
        "cfg": cfg.to_dict(), "mode": "selfseed" if a.selfseed else "seeded", "seed_split": a.seed_split,
        "seeds": None if seeds is None else int(len(seeds)),
        "seed_passes": None if seeds is None else -(-tot["rows"] // len(seeds)),
        "block": block_info, "block_fallbacks": None if block is None else block.fallbacks,
        "prompt_len": a.prompt_len, "n_new": a.n_new, "temperature": a.temperature, "allowed_vocab": a.allowed_vocab,
        "batch": a.batch, "seed": a.seed, "words_target": a.words, "rows_target": a.rows,
        **tot, "stop_rate": tot["stopped"] / rows, "words_per_row": tot["words"] / rows,
        "blocked_step_rate": tot["blocked_steps"] / max(tot["live_steps"], 1) if block is not None else None,
        "blocked_mass_per_token": tot["blocked_mass"] / max(tot["live_steps"], 1) if block is not None else None,
        "self_check_max_logit_diff": check, "self_check_max_logit_diff_amp": check_amp,
        "seconds": time.time() - t0, "device": str(device), "jsonl": str(jl),
    }
    D.shard_stats_path(jl).write_text(json.dumps(stats, indent=1))
    log(f"done: {tot['rows']:,} rows, {tot['words']:,} words, stop rate {stats['stop_rate']:.3f}, "
        f"{tot['flops']:.3e} FLOPs, {stats['seconds']:,.0f}s -> {jl}")
    return stats


if __name__ == "__main__":
    sys.exit(main() and None)
