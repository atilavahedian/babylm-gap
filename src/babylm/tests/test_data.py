"""Words, documents, nesting, subsets, FINAL protection."""
from __future__ import annotations

import json

import numpy as np
import pytest

from babylm.data import corpus as C
from babylm.data import prepare as P


@pytest.mark.parametrize("text", [
    "hello world", "  lead and trail  ", "a\tb\nc", "x\u00a0y \u2003z", "caf\u00e9 \u2014 na\u00efve",
    "*CHI:\tno.", "\x1cA\x1dB\x1eC\x1fD", "line one\n\nline two \n", "\u65e5\u672c\u8a9e \U0001F600 ok", "",
    "a\u200bb c",  # zero-width space is not whitespace for str.split
])
def test_token_word_ids_match_str_split(text):
    tok = P.load_tokenizer()
    enc = tok.encode(text, add_special_tokens=False)
    starts = np.array([o[0] for o in enc.offsets], dtype=np.int64)
    w, n = C.token_word_ids(text, starts)
    assert n == len(text.split())
    # every word owns >= 1 token, ids are monotone, and decoding the tokens of word k gives that word
    if len(w):
        assert (np.diff(w) >= 0).all()
    owned = set(int(x) for x in w if x < n)
    assert owned == set(range(n))
    words = text.split()
    for k in range(n):
        ids = [i for i, wk in zip(enc.ids, w) if wk == k]
        assert tok.decode(ids).strip() == words[k]


def test_synth_counts_equal_str_split(synth_dir):
    docs = json.loads((synth_dir / "docs.json").read_text())
    for split in ("train_10M", "dev"):
        d = C.load_split(split, synth_dir)
        texts = [t for s in C.SOURCES for t in docs[split][s]]
        assert d.n_words == sum(len(t.split()) for t in texts)
        starts = d.doc_starts()
        assert len(starts) == len(texts) == d.meta["docs"]
        wi = C.word_initial(np.asarray(d.words))
        # word-initial targets (positions >= 1) = every word except word 0 (whose first token is the BOS at 0)
        assert int(wi[1:].sum()) == d.n_words - 1 + int(np.asarray(d.words)[-1] == d.n_words)
        ends = list(starts[1:]) + [d.n_tokens]
        w = np.asarray(d.words)
        for i, (a, b) in enumerate(zip(starts, ends)):  # per document: word ids span exactly its words
            n = len(texts[i].split())
            first = int(w[a])
            assert int(w[b - 1]) - first in (n - 1, n)


def test_normalize_line_keeps_words_and_boundaries():
    lines = ["*CHI:\tno  more .", "   ", "A:\t\tyes\u00a0 ok ", "", "x"]
    assert C.normalize_line(lines[0]) == "*CHI: no more ."
    assert C.normalize_line(lines[2]) == "A: yes ok" and C.normalize_line("  ") == ""
    for l in lines:
        assert C.normalize_line(l).split() == l.split()
    assert C.doc_ranges([C.normalize_line(l) for l in lines], "childes") == C.doc_ranges(lines, "childes")
    assert C.doc_text(lines, 0, 1) == "*CHI: no more ."


def test_doc_ranges_switchboard_pieces():
    lines = [f"A:\tline {i}" for i in range(4500)] + [""]
    r = C.doc_ranges(lines, "switchboard")
    assert r == [(0, 2000), (2000, 4000), (4000, 4500)]
    r2 = C.doc_ranges(["a", "", "", "b", "c", " ", "d"], "childes")
    assert r2 == [(0, 1), (3, 5), (6, 7)]


def test_final_split_untouchable(synth_dir):
    with pytest.raises(C.FinalSplitError):
        C.load_split("test", synth_dir)
    assert C.load_split("test", synth_dir, final=True).n_tokens > 0


def test_real_word_counts_and_manifest(real_dir):
    man = json.loads(C.MANIFEST_PATH.read_text())
    assert man["raw"]["train_10M"]["total"]["wc_posix"] == 9_946_560
    assert man["raw"]["train_100M"]["total"]["wc_posix"] == 98_941_494
    assert man["raw"]["dev"]["total"]["wc_posix"] == 10_402_202
    assert man["tokenizer"]["sha256"] == C.sha256_file(C.TOKENIZER_PATH)
    assert "normalize_line" in man["normalisation"]
    for split in ("train_10M", "dev"):
        meta = json.loads((C.bin_dir(real_dir) / f"{split}.meta.json").read_text())
        assert meta["words"] == man["raw"][split]["total"]["words"]
        w = np.memmap(C.bin_dir(real_dir) / f"{split}.words.bin", dtype="<u4", mode="r")
        assert int(w[-1]) in (meta["words"] - 1, meta["words"])
    # spot check: the first documents of dev decode to the raw text and carry its str.split() count
    tok = P.load_tokenizer()
    d = C.load_split("dev", real_dir)
    lines = C.read_lines(C.raw_path(real_dir, "dev", "bnc_spoken"))
    rngs = C.doc_ranges(lines, "bnc_spoken")[:3]
    starts = d.doc_starts()[:4]
    for k, (a, b) in enumerate(rngs):
        text = C.doc_text(lines, a, b)
        assert text == "\n".join(" ".join(l.split()) for l in lines[a:b])
        ids = np.asarray(d.tokens[starts[k] + 1:starts[k + 1]], dtype=np.int64).tolist()
        assert tok.decode(ids) == text
        ww = np.asarray(d.words[starts[k]:starts[k + 1]])
        assert len(np.unique(ww[ww < ww[0] + len(text.split())])) == len(text.split())


def test_real_nested_splits(real_dir):
    man = json.loads(C.MANIFEST_PATH.read_text())
    prev = {s: set() for s in C.SOURCES}
    prev_words = man["splits"]["train_10M"]["words"]
    t10 = C.load_split("train_10M", real_dir)
    for name in ("train_20M", "train_30M", "train_50M"):
        dl = json.loads((real_dir / "ruler" / f"{name}.docs.json").read_text())
        for s in C.SOURCES:
            cur = set(dl["extra_docs_100M"][s])
            assert prev[s] <= cur
            prev[s] = cur
        meta = json.loads((C.bin_dir(real_dir) / f"{name}.meta.json").read_text())
        assert prev_words < meta["words"] < man["splits"]["train_100M"]["words"]
        assert abs(meta["words"] - dl["target_words"]) / dl["target_words"] < 0.02
        prev_words = meta["words"]
        # every train_10M token stream (per source) is the head of the same source's stream in this split
        sp = C.load_split(name, real_dir)
        for sid, s in enumerate(C.SOURCES):
            a = np.flatnonzero(np.asarray(t10.src) == sid)
            b = np.flatnonzero(np.asarray(sp.src) == sid)
            assert np.array_equal(np.asarray(t10.tokens)[a], np.asarray(sp.tokens)[b[:len(a)]])
    for s in C.SOURCES:  # extra documents never touch a train_10M line
        info = man["ruler"]["alignment"][s]
        assert info["docs_100M_free"] + info["docs_100M_touching_10M"] == info["docs_100M"]


def test_real_subset_fingerprints_stable(real_dir):
    for name, split in C.SUBSETS.items():
        lst = json.loads((C.SUBSET_DIR / f"{name}.json").read_text())
        d = C.load_split(name, real_dir)
        assert C.sha256_array(np.asarray(d.tokens).astype("<u2")) == lst["sha256_tokens"] == d.meta["fingerprint"]
        assert d.n_words == lst["words"] and 0.9e6 < d.n_words < 1.1e6
        # re-select from the raw documents: same document list
        dw = {}
        for s in C.SOURCES:
            lines = C.read_lines(C.raw_path(real_dir, split, s))
            dw[s] = np.array([sum(len(lines[i].split()) for i in range(a, b)) for a, b in C.doc_ranges(lines, s)])
        assert P.select_subset(dw) == {s: lst["docs"][s] for s in C.SOURCES}
