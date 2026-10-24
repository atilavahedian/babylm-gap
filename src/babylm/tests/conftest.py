"""Fixtures: a small synthetic BabyLM data dir built with the real tokenizer and the real prepare.py code, and the
real prepared data dir (tests that need it skip when it is absent)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from babylm.data import corpus as C
from babylm.data import prepare as P

WORDS = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "home", "I", "think", "you'd", "better", "go.",
         "Yeah.", "no!", "\u00e9t\u00e9", "\u2014", "caf\u00e9", "2024", "U.S.", "x\u00a0y", "\u65e5\u672c", "(laughs)"]
SPEAKERS = ["*CHI:\t", "*MOT:\t", "A:\t", "B:\t", ""]


def synth_lines(rng: np.random.Generator, n_docs: int, source: str) -> list[str]:
    lines: list[str] = []
    for d in range(n_docs):
        if source == "simple_wiki":
            lines.append(f"= = = Title {d} = = =")
        for _ in range(int(rng.integers(3, 25))):
            ws = [WORDS[i] for i in rng.integers(0, len(WORDS), size=int(rng.integers(1, 14)))]
            sp = SPEAKERS[int(rng.integers(0, len(SPEAKERS)))] if source in ("childes", "switchboard") else ""
            tail = " " if rng.random() < 0.1 else ""
            lines.append(sp + " ".join(ws) + tail)
        if source != "switchboard":
            lines.append("")
    lines.append("")
    return lines


def build_data_dir(root: Path, scale: int = 1) -> Path:
    rng = np.random.Generator(np.random.PCG64(7))
    tok = P.load_tokenizer()
    tok_sha = C.sha256_file(C.TOKENIZER_PATH)
    docs_by_split = {}
    for split, n_docs in (("train_10M", 16 * scale), ("dev", 8), ("test", 4)):
        d = {}
        for s in C.SOURCES:
            d[s] = P.SourceDocs(split, s, synth_lines(rng, n_docs, s))
        docs_by_split[split] = d
        toks = P.tokenize_split(tok, d)
        parts = [(sid, toks[s], list(range(len(d[s].ranges)))) for sid, s in enumerate(C.SOURCES)]
        tokens, words, src, per = P.assemble(parts)
        P.write_split(split, tokens, words, src, per, {}, root, tok_sha)
        if split in ("train_10M", "dev"):
            name = "train_sub" if split == "train_10M" else "dev_sub"
            chosen = P.select_subset({s: d[s].words for s in C.SOURCES}, target=1500)
            t2, w2, s2, p2 = P.assemble([(sid, toks[s], chosen[s]) for sid, s in enumerate(C.SOURCES)])
            P.write_split(name, t2, w2, s2, p2, {"fingerprint": C.sha256_array(t2.astype("<u2"))}, root, tok_sha)
    (root / "docs.json").write_text(json.dumps({sp: {s: d[s].texts for s in C.SOURCES}
                                                for sp, d in docs_by_split.items()}))
    return root


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    return build_data_dir(tmp_path_factory.mktemp("babylm_synth"))


@pytest.fixture(scope="session")
def real_dir() -> Path:
    d = C.DEFAULT_DATA_DIR
    if not (d / "bin" / "train_10M.meta.json").exists() or not (d / "bin" / "train_50M.meta.json").exists():
        pytest.skip("prepared BabyLM data not present (python -m babylm.data.prepare)")
    return d


def tiny_args(data_dir: Path, out: Path, *extra: str) -> list[str]:
    return ["--model", "baseline", "--size", "tiny", "--seq_len", "64", "--batch_tokens", "1024",
            "--micro_batch_size", "8", "--data_dir", str(data_dir), "--out", str(out), "--device", "cpu",
            "--threads", "4", "--eval_batch_size", "16", "--log_every", "1000", *extra]
