"""Never-seen-character leg of a verdict (critic ref-0001; instrument pinned ref-0015): the arm's gain over a reference
on DEV without the words that hold a character never present in the train_10M raw text. Gains on those words come
from training untrained rows (byte tokens), not from learning the language, so a pass must clear the bar on both legs.
Interim instrument (ref-0015): the harness per-word dumps (RUN/pw/pw_devfull_final.npy, babylm.analysis.perword),
compared harness with harness only, never differenced against v2; the full-DEV leg stays v2.

This is babylm/tools/unseen_chars.py (sha256 e03fddc9314ceead...aab267, commit ref-0016, the critic's pinned tool)
step for step, as functions:
  never   = set(DEV raw text) - set(train_10M raw text) (85 characters; RAW = DATA/raw/{dev,train_10M}/*)
  flagged = DEV words (dev.words.bin ids) with a token holding a byte >= 128 whose tokens' bytes, joined and decoded
            as UTF-8 (errors replaced), hold a never-seen character (10,275 words; a word whose only such character
            is ASCII, '>', is not flagged, as in the tool)
  per run = per-word nats (a per-token dump is summed per word; NaN counts 0): total = sum / W, unseen = sum over the
            flagged words / W, rest = total - unseen, W = all DEV words
Guarded gain over a reference = rest(ref) - rest(arm).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from babylm.data import corpus as C

PW = "pw_devfull_final.npy"
TOOL = "babylm/tools/unseen_chars.py sha256 ref-0017 (ref-0016), critic ref-0001 / ref-0015"
N_CHARS, N_WORDS = 85, 10_275  # what the pinned tool derives from the real DEV and train_10M


def token_bytes(tokenizer_json: str | Path) -> list[bytes]:
    """Each vocabulary entry's bytes under the GPT-2 byte-to-unicode map; added tokens are empty."""
    tj = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    tb = [b""] * (max(vocab.values()) + 1)
    for s, i in vocab.items():
        try:
            tb[i] = bytes(u2b[ch] for ch in s)
        except KeyError:
            tb[i] = s.encode()
    for a in tj.get("added_tokens", []):
        tb[a["id"]] = b""
    return tb


@lru_cache(maxsize=2)
def flagged(data_dir: str | None = None, tokenizer_json: str | None = None) -> dict:
    """The never-seen characters, the flagged-word mask (by DEV word id), W and the DEV stream's word ids."""
    root = Path(data_dir or C.DEFAULT_DATA_DIR)
    raw = root / "raw"
    if not (raw / "dev").is_dir() or not (raw / "train_10M").is_dir():
        raise SystemExit(f"never-seen-character leg: no raw text under {raw} (python -m babylm.data.prepare --download "
                         "--splits train_10M,dev, or --data_dir)")
    chars: dict[str, set[str]] = {}
    for split in ("train_10M", "dev"):
        chars[split] = set()
        for f in os.listdir(raw / split):
            chars[split] |= set((raw / split / f).read_text(encoding="utf-8"))
    never = chars["dev"] - chars["train_10M"]
    tb = token_bytes(tokenizer_json or C.TOKENIZER_PATH)
    nonascii = np.array([any(x >= 128 for x in t) for t in tb])
    dev = np.fromfile(root / "bin" / "dev.bin", dtype=np.uint16)
    wid = np.fromfile(root / "bin" / "dev.words.bin", dtype=np.uint32)
    words = json.loads((root / "bin" / "dev.meta.json").read_text())["words"]
    cand = np.unique(wid[nonascii[dev]])
    starts, ends = np.searchsorted(wid, cand, side="left"), np.searchsorted(wid, cand, side="right")
    bad = np.zeros(int(wid.max()) + 1, bool)
    for w, a, b in zip(cand, starts, ends):
        s = b"".join(tb[t] for t in dev[a:b]).decode("utf-8", errors="replace")
        if any(ch in never for ch in s):
            bad[w] = True
    return {"never": never, "bad": bad, "W": words, "wid": wid, "n_tokens": dev.size}


def split(dump: str | Path, f: dict) -> dict:
    """total, unseen and rest nats/word (all over W) of one per-word (or per-token) DEV dump."""
    x = np.load(dump).astype(np.float64)
    x[np.isnan(x)] = 0
    pw = np.bincount(f["wid"], weights=x, minlength=f["bad"].size) if x.size == f["n_tokens"] else x
    tot = pw.sum() / f["W"]
    ub = pw[f["bad"][:pw.size]].sum() / f["W"]
    return {"total": tot, "unseen": ub, "rest": tot - ub}


def find(p: str | Path | None) -> Path | None:
    """The full-DEV per-word dump beside a run dir or a score.json (RUN/pw/, RUN/score/../pw/)."""
    if p is None:
        return None
    p = Path(p)
    d = p if p.is_dir() else p.parent
    for q in (d / "pw" / PW, d.parent / "pw" / PW):
        if q.exists():
            return q
    return None


def rests(dumps: dict[str, str | Path | None], data_dir: str | None = None, tokenizer_json: str | None = None,
          check: bool = True) -> dict:
    """{name: rest or None (no dump)} plus "_info" (the instrument, flagged words, characters). The flagged set must be
    the pinned tool's (85 characters, 10,275 words) unless check is off (tests)."""
    have = {k: Path(p) for k, p in dumps.items() if p is not None and Path(p).exists()}
    out: dict = {k: None for k in dumps}
    if not have:
        out["_info"] = {"instrument": TOOL, "words": None, "chars": None}
        return out
    f = flagged(None if data_dir is None else str(data_dir), None if tokenizer_json is None else str(tokenizer_json))
    n_words, n_chars = int(f["bad"].sum()), len(f["never"])
    if check and (n_words, n_chars) != (N_WORDS, N_CHARS):
        raise SystemExit(f"never-seen-character leg: {n_chars} characters / {n_words} words, not the pinned tool's "
                         f"{N_CHARS} / {N_WORDS}: the data differ from the critic's derivation")
    for k, p in have.items():
        out[k] = split(p, f)["rest"]
    out["_info"] = {"instrument": TOOL, "words": n_words, "chars": n_chars, "W": f["W"]}
    return out
