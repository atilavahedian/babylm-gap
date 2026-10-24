"""CPU tests for attack A3 (babylm.harness.bpe_dropout) and its harness wiring (synthetic data, tiny model)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from babylm.data import corpus as C
from babylm.harness import bpe_dropout as B
from babylm.harness import stream as S
from babylm.harness import train as TR
from babylm.tests.conftest import tiny_args


@pytest.fixture(scope="module")
def bpe():
    return B.BPE()


@pytest.fixture(scope="module")
def synth_tab(synth_dir, bpe):
    return B.build_table(C.load_split("train_10M", synth_dir), bpe, log=lambda m: None)


def _docs(tokens, words):
    tokens, words = np.asarray(tokens).astype(np.int64), np.asarray(words).astype(np.int64)
    bos = np.flatnonzero(tokens == C.BOS_ID)
    return [(tokens[s + 1:e], words[s:e]) for s, e in zip(bos, np.append(bos[1:], len(tokens)))]


def _text(bpe, ids):
    return b"".join(bytes(bpe.char_to_byte[c] for c in bpe.id_to_str[i]) for i in ids.tolist()).decode("utf-8")


def test_parse_plan():
    assert B.parse_plan("0.1x8,0x2", 10) == [0.1] * 8 + [0.0] * 2
    assert B.parse_plan("0.05", 3) == [0.05] * 3
    with pytest.raises(ValueError):
        B.parse_plan("0.1x4", 5)
    with pytest.raises(ValueError):
        B.parse_plan("1.0", 1)


def test_p0_is_the_committed_tokenizer(bpe):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(C.TOKENIZER_PATH))
    texts = ["aaaa aaaaa", "the cat's hat.\n  Yes!!!", "été café 日本日本",
             "x y . ( ---------- ==== 2024 U.S.", "unbelievably antidisestablishmentarianism", "\t*CHI:\tno"]
    for t in texts:
        for piece, _ in tok.pre_tokenizer.pre_tokenize_str(t):
            want = tuple(tok.encode(bytes(bpe.char_to_byte[c] for c in piece).decode("utf-8"),
                                    add_special_tokens=False).ids)
            assert bpe.segment(bpe.init_ids(piece), 0.0)[0] == want, piece


def test_table_rebuilds_the_stream_exactly(synth_dir, synth_tab):
    tr = C.load_split("train_10M", synth_dir)
    tk, wd = B.render(synth_tab, 0, len(synth_tab), [0.0], None, B.BPE())
    assert np.array_equal(tk, np.asarray(tr.tokens)) and np.array_equal(wd, np.asarray(tr.words))


def test_dropout_stream_keeps_text_and_words(synth_dir, synth_tab, bpe):
    """Same text per document, more tokens, and word ids equal prepare's rule applied to the new tokens."""
    tr = C.load_split("train_10M", synth_dir)
    el = B.eligible(synth_tab, "all", 63)
    tk, wd = B.render(synth_tab, 0, len(synth_tab), [0.3, 0, 0], el, bpe)
    assert len(tk) > tr.n_tokens and tk.max() < C.VOCAB_SIZE
    assert int(C.word_initial(wd)[1:].sum()) == int(C.word_initial(np.asarray(tr.words))[1:].sum())
    canon, drop = _docs(tr.tokens, tr.words), _docs(tk, wd)
    assert len(canon) == len(drop)
    for (ci, cw), (di, dw) in zip(canon, drop):
        text = _text(bpe, ci)
        assert _text(bpe, di) == text and dw[0] == cw[0]  # BOS word id kept
        # char offset of every new token (a token starting inside a multi-byte char gets that char)
        byte_char = np.repeat(np.arange(len(text)), [len(ch.encode("utf-8")) for ch in text])
        starts = np.concatenate([[0], np.cumsum(bpe.tok_bytes[di])[:-1]])
        want, _ = C.token_word_ids(text, byte_char[starts])
        assert np.array_equal(dw[1:], want + cw[0])


def test_dropout_is_seeded_and_scoped(synth_dir, synth_tab, bpe):
    tr = C.load_split("train_10M", synth_dir)
    el = B.eligible(synth_tab, "all", 63)
    a = B.render(synth_tab, 0, len(synth_tab), [0.2, 0, 1], el, bpe)[0]
    assert np.array_equal(a, B.render(synth_tab, 0, len(synth_tab), [0.2, 0, 1], el, bpe)[0])
    assert not np.array_equal(a, B.render(synth_tab, 0, len(synth_tab), [0.2, 1, 1], el, bpe)[0])
    assert not np.array_equal(a, B.render(synth_tab, 0, len(synth_tab), [0.2, 0, 2], el, bpe)[0])
    # chunk-local randomness: a chunk renders the same alone as inside the whole stream
    n = min(len(synth_tab), B.CHUNK)
    assert np.array_equal(B.render(synth_tab, 0, n, [0.2, 0, 1], el, bpe)[0], a[:len(B.render(
        synth_tab, 0, n, [0.2, 0, 1], el, bpe)[0])])
    # nothing eligible -> canonical
    none = B.eligible(synth_tab, "rare", 0)
    assert not none.any()
    assert np.array_equal(B.render(synth_tab, 0, len(synth_tab), [0.5, 0, 0], none, bpe)[0], np.asarray(tr.tokens))
    # rare scope only re-segments words with count <= rare_max
    cnt = np.unique(synth_tab.word_count[synth_tab.word_count > 0])
    thr = int(np.median(cnt))
    tk, wd = B.render(synth_tab, 0, len(synth_tab), [0.5, 0, 0], B.eligible(synth_tab, "rare", thr), bpe)
    seg_c, seg_d = {}, {}
    for toks, wds, seg in ((np.asarray(tr.tokens), np.asarray(tr.words), seg_c), (tk, wd, seg_d)):
        for t, w in zip(toks.tolist(), wds.tolist()):
            seg.setdefault(w, []).append(t)
    wc = np.full(tr.n_words + 1, -1)
    wc[synth_tab.anchor] = synth_tab.word_count
    changed = [w for w in seg_c if seg_c[w] != seg_d[w]]
    assert changed and all(wc[w] <= thr for w in changed)


def test_inflation_and_null_matching(synth_tab):
    i_all = B.inflation(synth_tab, 0.1, "all", 63)
    assert 1.0 < i_all < 1.5
    assert B.inflation(synth_tab, 0.0, "all", 63) == 1.0


def test_build_streams_cached_and_worker_independent(synth_dir, tmp_path):
    import shutil
    d1, d2 = tmp_path / "w1", tmp_path / "w2"
    for d in (d1, d2):
        shutil.copytree(synth_dir / "bin", d / "bin")
    tr1, tr2 = C.load_split("train_10M", d1), C.load_split("train_10M", d2)
    m1 = B.build_streams(tr1, d1, [0.2, 0.0], "all", 63, 3, workers=1, log=lambda m: None)
    m2 = B.build_streams(tr2, d2, [0.2, 0.0], "all", 63, 3, workers=2, log=lambda m: None)
    assert m1[1] is None and m1[0]["sha256"] == m2[0]["sha256"] and m1[0]["inflation"] > 1
    again = B.build_streams(tr1, d1, [0.2, 0.0], "all", 63, 3, workers=1, log=lambda m: None)
    assert again[0] == m1[0]  # read from the cache


def test_zero_plan_sampler_equals_port_sampler(synth_dir):
    tr = C.load_split("train_10M", synth_dir)
    T = 64
    port = S.EpochSampler(tr.n_tokens, T, 5)
    ms = B.MultiStreamSampler([(tr.tokens, tr.words)] * 3, (tr.tokens, tr.words), T, 5)
    winit = C.word_initial(np.asarray(tr.words))
    for _ in range(40):
        st = port.next_starts(7)
        x0, y0 = S.gather_windows(tr.tokens, st, T)
        x1, y1, nw = ms.next_batch(7)
        assert (x0 == x1).all() and (y0 == y1).all() and nw == S.words_in_targets(winit, st, T)
    sd = ms.state_dict()
    ms2 = B.MultiStreamSampler([(tr.tokens, tr.words)] * 3, (tr.tokens, tr.words), T, 5)
    ms2.load_state_dict(sd)
    assert (ms.next_batch(5)[0] == ms2.next_batch(5)[0]).all()


# ------------------------------------------------------------------------------------------------
# harness
# ------------------------------------------------------------------------------------------------
def _losses(out):
    return [float(r["loss"]) for r in (json.loads(x) for x in (out / "log.jsonl").read_text().splitlines())
            if r.get("type") == "train"]


def test_harness_zero_dropout_equals_port(synth_dir, tmp_path):
    common = ("--epochs", "2", "--no_full_dev", "--no_pt")
    a = TR.main(tiny_args(synth_dir, tmp_path / "port", *common))
    b = TR.main(tiny_args(synth_dir, tmp_path / "zero", *common, "--bpe_dropout", "0"))
    assert _losses(tmp_path / "port") == _losses(tmp_path / "zero")
    assert a["words_seen"] == b["words_seen"] and a["steps"] == b["steps"]
    assert "bpe_dropout" not in json.loads((tmp_path / "port" / "config.json").read_text())


def test_harness_dropout_same_words_more_tokens_and_resume(synth_dir, tmp_path):
    common = ("--epochs", "3", "--no_full_dev", "--no_pt", "--milestone_scale", "0.002")
    a = TR.main(tiny_args(synth_dir, tmp_path / "port", *common))
    b = TR.main(tiny_args(synth_dir, tmp_path / "bpd", *common, "--bpe_dropout", "0.2x2,0", "--bpe_dropout_workers",
                          "1"))
    cfg = json.loads((tmp_path / "bpd" / "config.json").read_text())["bpe_dropout"]
    assert cfg["token_ratio"] > 1.0 and cfg["streams"][2] is None and b["steps"] > a["steps"]
    words = C.load_split("train_10M", synth_dir).n_words
    assert abs(b["words_seen"] - a["words_seen"]) < 0.03 * 3 * words  # same exposure up to the tiling tails
    p = TR.main(tiny_args(synth_dir, tmp_path / "res", *common, "--bpe_dropout", "0.2x2,0", "--max_session_steps",
                          str(b["steps"] // 2)))
    assert p["status"] == "paused"
    c = TR.main(tiny_args(synth_dir, tmp_path / "res", *common, "--bpe_dropout", "0.2x2,0", "--resume", "auto"))
    assert c["words_seen"] == b["words_seen"] and _losses(tmp_path / "res") == _losses(tmp_path / "bpd")
