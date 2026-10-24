"""Second LR cycle on mixed sources (attack A4; babylm.harness.train --init_from / --sources; off by default).

A cycle continues a trained checkpoint (--init_from: its weights, fresh optimizer state, a new LR schedule over this
run's steps) on one or more token streams ("sources"), each fed for a whole number of epochs:
    --sources real:train_10M:3 synthetic:/path/to/A4a_syn:1
KIND is the ledger label (real | synthetic), WHERE a split name under --data_dir or a directory holding
bin/synth.{bin,words.bin,src.bin,meta.json} (babylm.a4.data.write_synth_split), EPOCHS a positive integer.

CycleSampler: every source s and epoch e is tiled exactly as levers.PlanSampler tiles one epoch with the cycle's window
spec (stream: the port's random offset + permutation; line: line-aligned windows with BOS), with per-source seed
SeedSequence([data_seed, 0xA4, s]), so a source listed first sees the same windows in the same order in every run with
the same seed whatever else is mixed in (A4a's 3 real epochs = the first 3 of N0''s 6). Each source's windows (epoch
after epoch) are then interleaved in proportion: window i of the n_s windows of source s gets key (i + 0.5) / n_s, and
a stable sort over the keys (ties by source order) gives the cycle's window order, so every stretch of the cycle carries
the sources in their planned shares. Batches are consecutive runs of batch_seqs windows; the tail (< 1 batch) is
dropped and counted. Exposure is exact: per window, the word-initial target tokens of its own source
(prefix sums of corpus.word_initial), summed per KIND for every step and for the whole plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from babylm.data import corpus as C
from babylm.harness import levers as LV

KINDS = ("real", "synthetic")
SYNTH_SPLIT = "synth"


@dataclass
class Source:
    kind: str
    where: str
    epochs: int
    data: C.SplitData

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.where}:{self.epochs}"


def parse_source(spec: str) -> tuple[str, str, int]:
    m = re.fullmatch(r"(real|synthetic):(.+):(\d+)", spec)
    if not m or int(m.group(3)) < 1:
        raise ValueError(f"bad --sources item {spec!r} (want KIND:WHERE:EPOCHS, KIND in {KINDS}, EPOCHS >= 1)")
    return m.group(1), m.group(2), int(m.group(3))


def load_source(spec: str, data_dir: str | Path | None) -> Source:
    kind, where, ep = parse_source(spec)
    p = Path(where)
    if p.is_dir() and (p / "bin" / f"{SYNTH_SPLIT}.meta.json").exists():
        data = C.load_split(SYNTH_SPLIT, p)
    else:
        data = C.load_split(where, data_dir)
    return Source(kind, where, ep, data)


def parse_window(spec: str | None, default_T: int) -> LV.EpochSpec:
    if not spec:
        return LV.EpochSpec("stream", default_T)
    return LV.parse_plan(spec, 1, default_T)[0]


def source_seed(data_seed: int, s: int) -> int:
    return int(np.random.SeedSequence([int(data_seed), 0xA4, int(s)]).generate_state(1)[0])


class CycleSampler:
    def __init__(self, sources: list[Source], spec: LV.EpochSpec, batch_tokens: int, seed: int, *,
                 bos_id: int = C.BOS_ID, pad_id: int = C.BOS_ID, nl_ids: np.ndarray | None = None):
        if not sources:
            raise ValueError("a cycle needs at least one source")
        self.sources, self.spec, self.seed = sources, spec, int(seed)
        self.bos_id, self.pad_id = bos_id, pad_id
        T = spec.T
        self.batch_seqs = max(1, batch_tokens // T)
        S_l, L_l, B_l, src_l, key_l, wrd_l = [], [], [], [], [], []
        self.windows_per_source = []
        self._prefix = []
        for s, so in enumerate(sources):
            tok = so.data.tokens
            starts = None
            if spec.fmt == "line":
                if nl_ids is None:
                    raise ValueError("line windows need the tokenizer's newline ids")
                starts = LV.line_starts(np.asarray(tok), nl_ids, bos_id)
            ps = LV.PlanSampler(tok, [spec] * so.epochs, batch_tokens, source_seed(seed, s), bos_id=bos_id,
                                pad_id=pad_id, starts=starts)
            n = len(ps.w_start)
            winit = C.word_initial(np.asarray(so.data.words)).astype(np.int64)
            pre = np.concatenate([[0], np.cumsum(winit)])
            self._prefix.append(pre)
            # targets: stream windows s+1..s+T, line windows s..s+len-1 (levers.PlanSampler.batch_at)
            a = np.where(ps.w_bos, ps.w_start, ps.w_start + 1)
            b = np.where(ps.w_bos, ps.w_start + ps.w_len, ps.w_start + T + 1)
            S_l.append(ps.w_start); L_l.append(ps.w_len); B_l.append(ps.w_bos)
            src_l.append(np.full(n, s, dtype=np.int32))
            key_l.append((np.arange(n) + 0.5) / n)
            wrd_l.append(pre[b] - pre[a])
            self.windows_per_source.append(n)
        order = np.argsort(np.concatenate(key_l), kind="stable")
        self.w_start = np.concatenate(S_l)[order]
        self.w_len = np.concatenate(L_l)[order]
        self.w_bos = np.concatenate(B_l)[order]
        self.w_src = np.concatenate(src_l)[order]
        self.w_words = np.concatenate(wrd_l)[order]
        n_all = len(order)
        self.total_steps = n_all // self.batch_seqs
        self.dropped_windows = n_all - self.total_steps * self.batch_seqs
        used = slice(0, self.total_steps * self.batch_seqs)
        self.planned_words = {k: 0 for k in KINDS}
        for s, so in enumerate(sources):
            self.planned_words[so.kind] += int(self.w_words[used][self.w_src[used] == s].sum())
        self.ptr = 0
        self.last_words: dict[str, int] = {}

    @property
    def max_T(self) -> int:
        return self.spec.T

    def batch_at(self, i: int):
        """(x, y, cycle progress after the step, T, words per KIND)."""
        T, n = self.spec.T, self.batch_seqs
        x = np.full((n, T), self.pad_id, dtype=np.int64)
        y = np.full((n, T), -1, dtype=np.int64)
        words = {k: 0 for k in KINDS}
        for r, w in enumerate(range(i * n, (i + 1) * n)):
            so = self.sources[int(self.w_src[w])]
            tok = so.data.tokens
            s, ln = int(self.w_start[w]), int(self.w_len[w])
            if self.w_bos[w]:
                x[r, 0] = self.bos_id
                x[r, 1:ln] = tok[s:s + ln - 1]
                y[r, :ln] = tok[s:s + ln]
            else:
                seg = np.asarray(tok[s:s + T + 1], dtype=np.int64)
                x[r] = seg[:-1]
                y[r] = seg[1:]
            words[so.kind] += int(self.w_words[w])
        return torch.from_numpy(x), torch.from_numpy(y), (i + 1) / self.total_steps, T, words

    def next(self):
        """Same 5-tuple shape as levers.PlanSampler.next(); the per-KIND words of the batch go to self.last_words."""
        assert self.ptr < self.total_steps, "sampler exhausted"
        x, y, prog, T, words = self.batch_at(self.ptr)
        self.ptr += 1
        self.last_words = words
        return x, y, prog, T, None

    def signature(self) -> dict:
        return {"sources": [so.label for so in self.sources], "sha256": [so.data.meta.get("sha256") for so in self.sources],
                "window": f"{self.spec.fmt}:{self.spec.T}", "seed": self.seed, "total_steps": self.total_steps}

    def state_dict(self) -> dict:
        return {"ptr": self.ptr, **self.signature()}

    def load_state_dict(self, d: dict) -> None:
        if {k: d.get(k) for k in self.signature()} != self.signature():
            raise ValueError("cycle sampler state is for different sources, window, seed or plan")
        self.ptr = int(d["ptr"])


def load_init(path: str | Path, which: str = "eval") -> tuple[dict, dict, int | None]:
    """(state_dict, model cfg dict, words_seen) from a milestone checkpoint (chck_*.pt) or a resume.pt.
    which = 'eval': the weights the run evaluated and exported (the EMA shadow when the run had EMA, else the raw
    weights); 'raw': the optimiser's weights."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state" in ck and "optimizer" in ck:  # resume.pt: model = raw weights, ema = {"shadow": ...} or None
        ema = ck.get("ema")
        sd = ema["shadow"] if which == "eval" and ema else ck["model"]
        words = ck["state"].get("words_seen")
    else:  # milestone checkpoint: model = evaluated weights, model_raw = raw weights if EMA else None
        sd = ck["model_raw"] if which == "raw" and ck.get("model_raw") is not None else ck["model"]
        words = ck.get("words_seen")
    return sd, ck["cfg"], None if words is None else int(words)


CFG_FREE = ("seq_len",)  # the window length: no weight depends on it (RoPE tables grow on demand)


def cfg_diff(a: dict, b: dict, ignore: tuple[str, ...] = CFG_FREE) -> dict:
    keys = sorted((set(a) | set(b)) - set(ignore))
    return {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
