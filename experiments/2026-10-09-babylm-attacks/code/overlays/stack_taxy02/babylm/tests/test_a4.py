"""CPU tests for attack A4 (closed-system re-expression): babylm.a4.{ngram,kvcache,data,generate,audit,ledger},
babylm.harness.cycle and its train.py wiring (--init_from / --sources), babylm.configs.make_a4_jobs."""
from __future__ import annotations

import csv
import gzip
import json
import subprocess

import numpy as np
import pytest
import torch

from babylm.a4 import audit as AU
from babylm.a4 import data as D
from babylm.a4 import generate as GEN
from babylm.a4 import kvcache as KV
from babylm.a4 import ledger as LG
from babylm.a4 import ngram as NG
from babylm.a4 import verdict as VD
from babylm.configs import make_a4_jobs as MJ
from babylm.data import corpus as C
from babylm.harness import cycle as CY
from babylm.harness import levers as LV
from babylm.harness import train as TR
from babylm.models import baseline_reg
from babylm.tests.conftest import build_data_dir, tiny_args
from sprint1.harness.model_api import ModelConfig
from sprint1.models import baseline

COMMON = ("--no_pt", "--milestone_scale", "0.002", "--seed", "1")


def read(run, name):
    return list(csv.DictReader((run / name).open()))


@pytest.fixture(scope="module")
def a4_dir(tmp_path_factory):
    """Its own synthetic data dir (the halves and synthetic splits are written into it)."""
    return build_data_dir(tmp_path_factory.mktemp("a4_data"))


@pytest.fixture(scope="module")
def g_run(a4_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("a4_runs") / "G"
    TR.main(tiny_args(a4_dir, out, "--epochs", "2", "--save", "--no_full_dev", *COMMON))
    return out


@pytest.fixture(scope="module")
def tok():
    from babylm.data import prepare as P
    return P.load_tokenizer()


# ------------------------------------------------------------------------------------------------
# n-gram hashing and the copy block
# ------------------------------------------------------------------------------------------------
def test_hashes_agree_across_numpy_rows_and_torch():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 16384, 500)
    h = NG.ngram_hashes(s, 8)
    rows = np.stack([s[i:i + 8] for i in range(len(s) - 7)])
    assert np.array_equal(h, NG.hash_rows(rows))
    assert np.array_equal(h, NG.hash_rows_torch(torch.from_numpy(rows)).numpy())
    assert len(np.unique(h)) == len({tuple(r) for r in rows.tolist()})  # no collision on distinct rows here


def test_block_index_is_exact():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 6, 4000)  # tiny alphabet: contexts repeat, so runs have many tokens
    n = 4
    succ: dict[tuple, set] = {}
    for i in range(len(s) - n + 1):
        succ.setdefault(tuple(s[i:i + n - 1]), set()).add(int(s[i + n - 1]))
    bi = NG.BlockIndex.build(s, n)
    ctx = torch.from_numpy(rng.integers(0, 7, (64, n - 1)))
    for row, got in zip(ctx.tolist(), bi.blocked_sets(ctx)):
        assert got == succ.get(tuple(row), set())
    lg = torch.zeros(64, 8)
    lg[:, 7] = float("-inf")  # an already-masked id stays masked and is not counted
    removed = bi.mask(lg, ctx)
    for r, row in enumerate(ctx.tolist()):
        want = succ.get(tuple(row), set())
        assert set(torch.nonzero(torch.isinf(lg[r])).flatten().tolist()) == want | {7}
        assert int(removed[r]) == len(want)
    full = torch.tensor([[0, 0, 0]])  # every allowed token blocked: the row is left unblocked and counted
    if succ.get((0, 0, 0)) == set(range(6)):
        lg = torch.zeros(1, 6)
        assert int(bi.mask(lg, full)[0]) == 0 and bi.fallbacks == 1 and torch.isfinite(lg).all()


# ------------------------------------------------------------------------------------------------
# KV cache
# ------------------------------------------------------------------------------------------------
def rand_model(extra=None, tie=False, reg=False, L=4):
    cfg = ModelConfig(vocab_size=96, seq_len=64, n_layer=L, n_head=2, n_embd=32, tie_embeddings=tie, extra=extra or {})
    torch.manual_seed(0)
    m = (baseline_reg.build(cfg) if reg else baseline.build(cfg)).eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3 + (1.0 if p.ndim == 1 else 0.0))
    return m


@pytest.mark.parametrize("kw", [{}, {"extra": {"n_value_embeds": 2}}, {"tie": True, "extra": {"n_value_embeds": 0}},
                                {"extra": {"unet": False, "x0_inject": False, "embnorm": False}},
                                {"reg": True, "extra": {"dropout": 0.1}}, {"L": 1}])
def test_cached_decoding_matches_forward(kw):
    m = rand_model(**kw)
    prompts = torch.randint(1, 96, (3, 7), generator=torch.Generator().manual_seed(2))
    assert KV.self_check(m, prompts, steps=10) < 1e-4


def test_sample_stop_ban_and_determinism():
    m = rand_model()
    prompts = torch.randint(1, 96, (16, 5), generator=torch.Generator().manual_seed(3))
    kw = dict(stop_id=0, ban_stop_steps=4, allowed_vocab=3)  # ids 0-2 only: the stop id comes up often
    a = KV.sample(m, prompts, 20, generator=torch.Generator().manual_seed(4), **kw)
    b = KV.sample(m, prompts, 20, generator=torch.Generator().manual_seed(4), **kw)
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.lengths, b.lengths)
    assert a.stopped.any() and (a.lengths[a.stopped] >= 4).all()
    for r in range(16):
        L = int(a.lengths[r])
        assert (a.tokens[r, :L] != 0).all() and (a.tokens[r, :L] < 3).all()
        if a.stopped[r]:
            assert int(a.tokens[r, L]) == 0
    assert a.stats["steps"] <= 20 and a.stats["flops_per_row"] > 0


def test_sampling_never_completes_an_indexed_ngram():
    m = rand_model()
    rng = np.random.default_rng(5)
    stream = rng.integers(1, 12, 3000)  # dense index over ids 1..11: most contexts block several tokens
    n = 3
    bi = NG.BlockIndex.build(stream, n)
    have = {tuple(stream[i:i + n]) for i in range(len(stream) - n + 1)}
    prompts = torch.from_numpy(rng.integers(1, 12, (32, 6)))
    for block_from in (0, 3):
        out = KV.sample(m, prompts, 24, allowed_vocab=12, block=bi, block_from=block_from, stop_id=None,
                        generator=torch.Generator().manual_seed(6))
        seq = torch.cat([prompts, out.tokens], 1).tolist()
        for row in seq:
            for j in range(6 + block_from, len(row)):
                assert tuple(row[j - n + 1:j + 1]) not in have or bi.fallbacks > 0
        assert (out.blocked_steps > 0).any() and (out.blocked_mass > 0).any()


# ------------------------------------------------------------------------------------------------
# data: halves, seeds, synthetic splits, N2
# ------------------------------------------------------------------------------------------------
def test_halves_partition_train_10M(a4_dir):
    metas = D.make_halves(a4_dir)
    tr = C.load_split("train_10M", a4_dir)
    hA, hB = (C.load_split(n, a4_dir) for n in D.HALF_NAMES)
    assert sorted(metas[0]["doc_indices"] + metas[1]["doc_indices"]) == list(range(len(tr.doc_starts())))
    assert hA.n_words + hB.n_words == tr.n_words and hA.n_tokens + hB.n_tokens == tr.n_tokens
    b = D.doc_bounds(tr)
    toks = np.asarray(tr.tokens)
    want = np.concatenate([toks[b[k]:b[k + 1]] for k in metas[0]["doc_indices"]])
    assert np.array_equal(np.asarray(hA.tokens), want)
    assert int(C.word_initial(np.asarray(hA.words)).sum()) == hA.n_words  # word ids re-based and contiguous
    for s in C.SOURCES:  # every source split (switchboard is one document in the fixture)
        a, bb = hA.meta["per_source"].get(s, {}).get("words", 0), hB.meta["per_source"].get(s, {}).get("words", 0)
        assert a + bb == tr.meta["per_source"][s]["words"]


def test_seed_starts(a4_dir, tok):
    data = C.load_split("train_10M", a4_dir)
    t = np.asarray(data.tokens)
    nl = LV.newline_ids(tok)
    s = D.seed_starts(t, nl, 16)
    starts = set(LV.line_starts(t, nl, C.BOS_ID).tolist())
    assert len(s) > 50 and (np.diff(s) >= 16).all()
    for x in s.tolist():
        assert (t[x] == C.BOS_ID and x + 1 in starts) or x in starts
        assert not (t[x + 1:x + 16] == C.BOS_ID).any()
    o = D.seed_order(s, 0)
    assert sorted(o.tolist()) == s.tolist() and not np.array_equal(o, s)


def test_synth_split_roundtrip_and_n2(a4_dir, tmp_path):
    texts = ["*CHI: the dog ran home.\n\nA: I think you'd better go . yes", "  ", "Hello there, big world!  Fine (laughs)",
             "one two three four five six seven. eight nine ten?"]
    meta = D.write_synth_split(texts, [1, 2, 6, 0], tmp_path / "s", {"x": 1}, seed=0)
    data = C.load_split("synth", tmp_path / "s")
    kept = [D.clean_doc(t) for t in texts if D.clean_doc(t)]
    assert meta["docs"] == 3 and data.n_words == sum(len(t.split()) for t in kept)
    assert int(C.word_initial(np.asarray(data.words)).sum()) == data.n_words
    got, srcs = D.read_synth_docs(tmp_path / "s")
    assert sorted(got) == sorted(kept) and sorted(srcs) == [0, 1, 6] and "self" in meta["per_source"]
    D.make_shuffled(tmp_path / "s", tmp_path / "n2", seed=0)
    sh, srcs2 = D.read_synth_docs(tmp_path / "n2")
    assert srcs2 == srcs
    strip = lambda w: w.rstrip(".!?,;:\"')]")  # noqa: E731
    for a, b in zip(got, sh):
        assert len(a.split()) == len(b.split())
        for la, lb in zip(a.split("\n"), b.split("\n")):
            wa, wb = la.split(), lb.split()
            assert sorted(map(strip, wa)) == sorted(map(strip, wb)) and len(wa) == len(wb)
            if wa and D.SPEAKER.match(wa[0]):
                assert wb[0] == wa[0]
            if wa and D.PUNCT_ONLY.match(wa[-1]):
                assert wb[-1] == wa[-1]


def test_shuffle_keeps_sentence_ends():
    w = D.shuffle_text("*MOT: where is the big red ball ? look at that one.", np.random.default_rng(0)).split()
    assert len(w) == 12 and w[0] == "*MOT:" and w[7] == "?" and w[-1].endswith(".")
    assert sorted(w[1:7]) == sorted("where is the big red ball".split())
    assert sorted(x.rstrip(".") for x in w[8:]) == sorted("look at that one".split())


# ------------------------------------------------------------------------------------------------
# the cycle sampler and the train.py wiring
# ------------------------------------------------------------------------------------------------
def make_syn(a4_dir, out, n_docs=40, seed=0):
    rng = np.random.default_rng(seed)
    words = "the cat sat on a mat dog ran home I think you'd better go. Yeah. no!".split()
    texts = ["\n".join(" ".join(rng.choice(words, rng.integers(3, 12))) for _ in range(rng.integers(2, 6)))
             for _ in range(n_docs)]
    D.write_synth_split(texts, [6] * n_docs, out, {}, seed=seed)
    return out


def test_cycle_sampler_exposure_interleave_and_prefix(a4_dir, tmp_path):
    syn = make_syn(a4_dir, tmp_path / "syn")
    real = CY.load_source("real:train_10M:2", a4_dir)
    s = CY.load_source(f"synthetic:{syn}:1", a4_dir)
    spec = LV.EpochSpec("stream", 32)
    cs = CY.CycleSampler([real, s], spec, 32 * 8, seed=3)
    tot = {k: 0 for k in CY.KINDS}
    for i in range(cs.total_steps):  # exact exposure: word-initial flags at each window's target positions
        x, y, _, _, words = cs.batch_at(i)
        for r, w in enumerate(range(i * cs.batch_seqs, (i + 1) * cs.batch_seqs)):
            so = cs.sources[int(cs.w_src[w])]
            st = int(cs.w_start[w])
            wi = C.word_initial(np.asarray(so.data.words))
            assert int(wi[st + 1:st + 33].sum()) == int(cs.w_words[w])
            assert torch.equal(y[r], torch.from_numpy(np.asarray(so.data.tokens[st + 1:st + 33], dtype=np.int64)))
        for k in CY.KINDS:
            tot[k] += words[k]
    assert tot == cs.planned_words and tot["synthetic"] > 0
    n = np.asarray(cs.windows_per_source, dtype=float)  # proportional interleave: every prefix tracks the shares
    cum = np.cumsum(cs.w_src == 1)
    assert np.abs(cum - np.arange(1, len(cum) + 1) * n[1] / n.sum()).max() <= 1.0
    long = CY.CycleSampler([CY.load_source("real:train_10M:4", a4_dir)], spec, 32 * 8, seed=3)
    mine = cs.w_start[cs.w_src == 0]  # A4a's real epochs = the first epochs of N0' in the same order
    assert np.array_equal(mine, long.w_start[:len(mine)])
    sd = cs.state_dict()
    cs2 = CY.CycleSampler([real, s], spec, 32 * 8, seed=3)
    cs2.load_state_dict(sd)
    with pytest.raises(ValueError):
        CY.CycleSampler([real], spec, 32 * 8, seed=3).load_state_dict(sd)


def test_inert_init_from_reproduces_the_port(a4_dir, tmp_path):
    """--init_from a step-0 checkpoint of the same run (same initial weights, 0 words) changes nothing: the wiring
    (loading, lineage, warmup, logging) perturbs no number."""
    args = ("--epochs", "2", "--steps", "8", *COMMON)
    TR.main(tiny_args(a4_dir, tmp_path / "z", "--epochs", "2", "--steps", "0", "--save", *COMMON))
    a = TR.main(tiny_args(a4_dir, tmp_path / "a", *args))
    b = TR.main(tiny_args(a4_dir, tmp_path / "b", *args, "--init_from", str(tmp_path / "z" / "resume.pt")))
    assert "cycle" not in a["config"] and "exposure" not in a and "words_real" not in read(tmp_path / "a", "train_log.csv")[0]
    assert [r["loss"] for r in read(tmp_path / "a", "train_log.csv")] == [r["loss"] for r in read(tmp_path / "b", "train_log.csv")]
    assert a["full_eval"]["nats"] == b["full_eval"]["nats"] and b["exposure"]["total"] == a["words_seen"]


def test_cycle_run_lineage_resume_and_refusals(a4_dir, g_run, tmp_path):
    syn = make_syn(a4_dir, tmp_path / "syn", n_docs=60)
    g = json.loads((g_run / "result.json").read_text())
    cyc = ("--init_from", str(g_run / "resume.pt"), "--warmup_frac", "0.1", "--sources", "real:train_10M:1",
           f"synthetic:{syn}:1", "--exposure_extra", f"G_A=@{g_run}/result.json", "gen_x=1000")
    r = TR.main(tiny_args(a4_dir, tmp_path / "c", *cyc, "--no_full_dev", *COMMON))
    ex = r["exposure"]
    assert ex["prior_words"] == g["words_seen"] and ex["extra_words"] == {"G_A": g["words_seen"], "gen_x": 1000}
    assert ex["cycle_words"] == ex["planned_cycle_words"] and ex["cycle_words"]["synthetic"] > 0
    assert r["words_seen"] == g["words_seen"] + sum(ex["cycle_words"].values())
    assert ex["total"] == r["words_seen"] + g["words_seen"] + 1000 and ex["within_cap"]
    rows = read(tmp_path / "c", "train_log.csv")
    assert int(rows[-1]["words_real"]) + int(rows[-1]["words_synthetic"]) == sum(ex["cycle_words"].values())
    assert int(rows[0]["words_seen"]) > g["words_seen"] and ex["warmup_steps"] >= 1
    assert all(int(x["words_seen"]) > g["words_seen"] for x in read(tmp_path / "c", "curves.csv"))  # pre-fired
    # preemption: a paused cycle resumes bit-exactly
    TR.main(tiny_args(a4_dir, tmp_path / "d", *cyc, "--no_full_dev", *COMMON, "--save", "--max_session_steps", "3"))
    d = TR.main(tiny_args(a4_dir, tmp_path / "d", *cyc, "--no_full_dev", *COMMON, "--save", "--resume", "auto"))
    assert d["sessions"] == 2 and d["exposure"]["cycle_words"] == ex["cycle_words"]
    for p in (tmp_path / "d").glob("checkpoints/*.pt"):
        assert torch.load(p, weights_only=False)["lineage"]["prior_words"] == g["words_seen"]
    assert [x["loss"] for x in read(tmp_path / "d", "train_log.csv")] == [x["loss"] for x in rows]
    with pytest.raises(SystemExit, match="R0"):
        TR.main(tiny_args(a4_dir, tmp_path / "e", *cyc, "--exposure_cap", str(ex["total"] - 1), *COMMON))
    with pytest.raises(SystemExit, match="cfg differs"):
        TR.main(tiny_args(a4_dir, tmp_path / "f", *cyc, "--n_layer", "3", *COMMON))
    with pytest.raises(SystemExit, match="exclusive"):
        TR.main(tiny_args(a4_dir, tmp_path / "g", *cyc, "--window_plan", "line:32", *COMMON))
    led = LG.main([str(tmp_path / "c"), "--base_epochs", "10"])["arms"][0]
    assert led["words_total"] == ex["total"] and led["within_R0"]
    assert led["flops"]["G_A_train"] == pytest.approx(led["flops"]["G_train"]) and led["flops"]["gen_x_train"] is None
    assert led["flops_total"] is None  # gen_x has no run to read FLOPs from


def test_cycle_line_windows_with_mntp(a4_dir, tmp_path):
    TR.main(tiny_args(a4_dir, tmp_path / "g", "--epochs", "1", "--steps", "2", "--save", "--mntp_frac", "0.25",
                      "--no_full_dev", *COMMON))
    syn = make_syn(a4_dir, tmp_path / "syn")
    r = TR.main(tiny_args(a4_dir, tmp_path / "c", "--init_from", str(tmp_path / "g" / "resume.pt"), "--mntp_frac",
                          "0.25", "--sources", "real:train_10M:1", f"synthetic:{syn}:1", "--cycle_window", "line:32",
                          "--steps", "4", "--no_full_dev", *COMMON))
    assert r["status"] == "ok" and r["exposure"]["window"] == "line:32" and r["config"]["model_cfg"]["seq_len"] == 32
    assert sum(r["exposure"]["cycle_words"].values()) == r["words_seen"] - r["exposure"]["prior_words"] > 0


def test_round1_all_on_levers_through_generation_and_cycle(a4_dir, tmp_path):
    """The round-1 all-on lever set (baseline_reg + dropout, no value embeds, tied, head wd, MNTP) as G and student."""
    lev = ("--model", "baseline_reg", "--extra", "dropout=0.1", "--extra", "n_value_embeds=0", "--tie_embeddings",
           "--head_wd", "0.1", "--mntp_frac", "0.25", "--no_full_dev", *COMMON)
    TR.main(tiny_args(a4_dir, tmp_path / "g", "--epochs", "1", "--steps", "6", "--save", *lev))
    st = GEN.main(["--ckpt", str(tmp_path / "g" / "resume.pt"), "--data_dir", str(a4_dir), "--batch", "16",
                   "--prompt_len", "16", "--n_new", "32", "--device", "cpu", "--out", str(tmp_path / "gen" / "a"),
                   "--seed_split", "train_10M", "--block_split", "train_10M", "--rows", "32"])
    assert st["self_check_max_logit_diff"] < 1e-4
    assert st["cfg"]["vocab_size"] > 16384  # MNTP pads the vocab with its mask id; sampling masks ids >= 16384
    D.build_from_shards(tmp_path / "syn", [tmp_path / "gen" / "a.jsonl.gz"], words=200)
    r = TR.main(tiny_args(a4_dir, tmp_path / "c", "--init_from", str(tmp_path / "g" / "resume.pt"), "--sources",
                          "real:train_10M:1", f"synthetic:{tmp_path / 'syn'}:1", "--steps", "4", *lev))
    assert r["status"] == "ok" and r["exposure"]["cycle_words"]["synthetic"] > 0

# ------------------------------------------------------------------------------------------------
# generation, audits, ledger and jobs end to end
# ------------------------------------------------------------------------------------------------
def test_generate_build_audit_ledger(a4_dir, g_run, tmp_path, tok):
    common = ["--ckpt", str(g_run / "resume.pt"), "--data_dir", str(a4_dir), "--batch", "16", "--prompt_len", "16",
              "--n_new", "32", "--device", "cpu"]
    st = GEN.main([*common, "--out", str(tmp_path / "g" / "a"), "--seed_split", "train_10M", "--block_split",
                   "train_10M", "--words", "600"])
    assert st["words"] >= 600 and st["self_check_max_logit_diff"] < 1e-4 and st["block"]["keys"] > 0
    assert st["flops"] > 0 and st["seeds"] > 0 and st["blocked_step_rate"] is not None
    rows = D.read_shard(tmp_path / "g" / "a.jsonl.gz")
    assert sum(r["words"] for r in rows) == st["words"] and all(r["src"] < 6 for r in rows)
    n1 = GEN.main([*common, "--out", str(tmp_path / "g" / "n1"), "--selfseed", "--rows", "16"])
    assert n1["rows"] == 16 and all(r["src"] == D.SELF_SRC for r in D.read_shard(tmp_path / "g" / "n1.jsonl.gz"))
    probe = GEN.main([*common, "--out", str(tmp_path / "g" / "p"), "--seed_split", "train_10M", "--rows", "20"])
    assert probe["rows"] == 20 and probe["block"] is None
    meta = D.build_from_shards(tmp_path / "syn", [tmp_path / "g" / "a.jsonl.gz"], words=500)
    assert 500 <= meta["words"] < st["words"] + 1 and meta["shards"][0]["gen"]["flops"] == st["flops"]
    # the block holds on the decoded, re-tokenized text up to a handful of boundary 8-grams
    cp = AU.copy_audit(str(tmp_path / "syn"), str(tmp_path / "g" / "p.jsonl.gz"), str(a4_dir), tmp_path / "au" / "c.json",
                       n_docs=20, min_len=8)
    assert cp["overlap"]["synthetic"]["8gram"] < 0.05 < cp["overlap"]["dev"]["8gram"]
    assert (tmp_path / "au" / "c_nn.csv").exists() and cp["nearest_neighbour"]["synthetic"]["docs"] > 0
    ev = tmp_path / "eval" / "fast"
    ev.mkdir(parents=True)
    syn_doc = max(D.read_synth_docs(tmp_path / "syn")[0], key=lambda t: len(t.split()))
    tr_doc = json.loads((a4_dir / "docs.json").read_text())["train_10M"]["childes"]
    tr_line = next(" ".join(x.split()) for x in tr_doc if len(x.split()) >= 6)
    items = [{"good": " ".join(syn_doc.split()[:20]), "bad": tr_line}, {"a": "nothing to see here at all ok", "b": 1}]
    (ev / "x.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    (ev / "img.png").write_bytes(b"\x89PNG")
    lk = AU.leak([str(tmp_path / "syn")], str(tmp_path / "eval"), str(a4_dir), tmp_path / "au" / "leak.json")
    s = lk["synthetic"][str(tmp_path / "syn")]
    assert lk["skipped_files"] == ["fast/img.png"] and lk["eval_items"] == 3
    assert lk["train_10M"]["items_hit"] >= 1 and s["items_hit_synthetic_only"] >= 1
    assert any(i["text"] == items[0]["good"] for i in s["synthetic_only_items"])


def test_myers_distance_is_semi_global_edit_distance():
    def brute(p, t):
        m = len(p)
        prev = list(range(m + 1))
        best = m
        for c in t:
            cur = [0]
            for i in range(1, m + 1):
                cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (p[i - 1] != c)))
            best = min(best, cur[m])
            prev = cur
        return min(best, m)

    rng = np.random.default_rng(0)
    for _ in range(300):
        p = rng.integers(0, 4, rng.integers(1, 70)).tolist()
        t = rng.integers(0, 4, rng.integers(0, 90)).tolist()
        assert AU.myers_distance(p, t) == brute(p, t)
    p = list(range(100))
    t = [7, 7] + p[:40] + [999] + p[41:] + [5]
    assert AU.myers_distance(p, t) == 1


def test_train_index_finds_near_copies():
    rng = np.random.default_rng(1)
    train = rng.integers(0, 500, 20000)
    idx = AU.TrainIndex(train)
    q = train[5000:5100].copy()
    q[[10, 50, 90]] = 499 - q[[10, 50, 90]]  # 3 substitutions
    d, nc = idx.nn(q)
    assert d == 3 and nc >= 1
    assert idx.nn(rng.integers(1000, 2000, 100))[0] == 100  # nothing shared: the length


def test_make_a4_jobs_plan_and_files(tmp_path):
    r = MJ.main(["--out_dir", str(tmp_path), "--commit", "a" * 40])
    p = r["trainer"]["plan"]
    assert (p["R_a"], p["R_b"], p["syn_words_a"]) == (3, 2, 3 * MJ.TRAIN_WORDS)
    assert max(p["words_a"], p["words_b"]) <= MJ.CAP
    jobs = {j["name"]: j["args"] for j in r["trainer"]["jobs"]}
    assert jobs["a4_G"][jobs["a4_G"].index("--epochs") + 1] == "4"
    assert jobs["a4_GB"][jobs["a4_GB"].index("--split") + 1] == "train_10M_hB"
    for n in ("a4_n0a", "a4_a", "a4_n1a", "a4_n2a", "a4_n0b", "a4_b", "a4_n1b", "a4_n2b"):
        a = jobs[n]
        assert "--epochs" not in a and "--init_from" in a and "--sources" in a and "--save" in a
        assert ("--exposure_extra" in a) == (n in ("a4_b", "a4_n1b", "a4_n2b"))
        TR.build_parser().parse_args(a)  # every job parses
    for f in ("a4_screen.json", "a4_screen_b2.json"):
        j = json.loads((tmp_path / "gcp" / f).read_text())
        assert j["commit"] == "a" * 40 and j["keep_checkpoints"] and all(x["cmd"] for x in j["jobs"])
    with pytest.raises(SystemExit):
        MJ.main(["--out_dir", str(tmp_path), "--g_uri", "${BUCKET}/y", "--g_epochs", "4"])
    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=MJ.HERE, capture_output=True).returncode == 0:
        r = MJ.main(["--out_dir", str(tmp_path), "--commit", "a" * 40, "--provisional",
                     "--base_job", "HEAD:babylm/configs/jobs_a4_screen.json:a4_G"])  # a base read from git, E* = 4
        assert r["trainer"]["plan"]["E_star"] == 4 and "PROVISIONAL" in r["trainer"]["description"]
        head = subprocess.run(["git", "show", "HEAD:babylm/configs/jobs_a4_screen.json"], cwd=MJ.HERE,
                              capture_output=True, text=True).stdout
        MJ.check_pinned("HEAD", head)  # the pinned commit carries this trainer file
        with pytest.raises(SystemExit):
            MJ.check_pinned("HEAD", head + " ")  # it carries another: the executor would run the wrong jobs
    with pytest.raises(SystemExit):
        MJ.check_pinned("a" * 40, "{}")  # no such commit


def _arm(root, name, dev, blimp, flops, syn=None):
    d = root / name
    (d / "score").mkdir(parents=True)
    src = ["real:train_10M:3"] + ([f"synthetic:{syn[0]}:1"] if syn else [])
    sha = ["r" * 64] + ([syn[1]] if syn else [])
    (d / "result.json").write_text(json.dumps({"exposure": {"sources": src, "source_sha256": sha}}))
    (d / "score" / "score.json").write_text(json.dumps({"dev_nats_per_word": dev, "fast": {VD.BLIMP: blimp}}))
    return {"run": name, "flops_total": flops, "within_R0": True}


def test_verdict_applies_the_registered_kill_rule(tmp_path):
    runs, data = tmp_path / "runs", tmp_path / "runs" / "a4_data"
    rows = [_arm(runs, "a4_n0a", 5.30, 63.2, 10.0), _arm(runs, "a4_a", 5.20, 63.1, 10.1, ("A", "a" * 64)),
            _arm(runs, "a4_n1a", 5.28, 63.0, 10.1, ("N1", "b" * 64)),
            _arm(runs, "a4_n2a", 5.25, 63.0, 10.1, ("N2", "c" * 64)),
            _arm(runs, "a4_n0b", 5.40, 62.0, 8.0), _arm(runs, "a4_b", 5.35, 62.0, 10.2, ("B", "d" * 64))]
    (data / "audit").mkdir(parents=True)
    (data / "ledger.json").write_text(json.dumps({"arms": rows}))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"dev_nats_per_word": 5.31, "fast": {VD.BLIMP: 63.0}, "run": "r1"}))
    args = ["--base_score", str(base), "--runs_root", str(runs)]
    with pytest.raises(SystemExit):  # amendment 4: no number before the leak audit
        VD.main(args)
    leak = {k: {"sha256": h, "words": 1, "items_hit": 0, "items_hit_synthetic_only": 0, "dev_13gram_share": 0.0,
                "dev_13gram_share_synthetic_only": 0.0}
            for k, h in (("A", "a" * 64), ("N1", "b" * 64), ("N2", "c" * 64))}
    (data / "audit" / "leak.json").write_text(json.dumps({"synthetic": leak}))
    with pytest.raises(SystemExit):  # A4b's split is not audited yet
        VD.main(args)
    leak["B"] = {**leak["A"], "sha256": "d" * 64}
    (data / "audit" / "leak.json").write_text(json.dumps({"synthetic": leak}))
    r = VD.main(args)
    assert r["judged"] == "a" and r["verdict"].startswith("NO VERDICT") and r["promote"] is None
    r = VD.main(args + ["--floor", "0.01", "--blimp_sigma", "0.5"])
    a = r["families"]["a"]
    assert a["comparator"] == "N0'" and a["refs"]["N2"]["gain"] == pytest.approx(0.05)
    assert r["verdict"] == "PASS" and r["promote"] == "a4_a"
    r = VD.main(args + ["--floor", "0.03", "--blimp_sigma", "0.5"])  # 2F = 0.06 > the gain over N2 (0.05)
    assert r["verdict"] == "KILL" and r["promote"] is None and any("N2" in k for k in r["families"]["a"]["kill"])
    r = VD.main(args + ["--floor", "0.01", "--blimp_sigma", "0.04"])  # vs the comparator, N0' (63.2): drop 0.1 >= 2S
    assert r["verdict"] == "KILL" and any("BLiMP" in k for k in r["families"]["a"]["kill"])
    b = r["families"]["b"]  # 10.2 / 8.0 > 1.25: N0'_a joins as the FLOP-matched reference; N1b, N2b not run
    assert "FLOP-matched a4_n0a" in b["refs"] and set(b["missing"]) == {"N1", "N2"} and b["verdict"].startswith("NO")
