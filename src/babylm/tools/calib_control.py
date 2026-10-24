"""Calibrated-gain control (critic ref-0292): a comparator, never an entry.

Each model is interpolated with a train_10M unigram, p = (1 - lam) p_model + lam u, where u = add-0.5 token counts
of data/babylm/bin/train_10M.bin. lam comes from a fixed grid and is cross-fitted on DEV document parity: picked on
even documents and read on odd ones, and vice versa, using the same split for every model. The calibrated
nats/word is the sum of the two read halves over the DEV words. Both verdict legs are reported: full DEV, and DEV
minus words holding a character never seen in train_10M (babylm/tools/unseen_chars_list.json, pinned at ref-0016),
by dev_bpw_twin.py's ex_unseen_chars rule (ref-0026): a token is dropped if any character of its span lies in such a
word (whitespace-only tokens and BOS separators stay), and the leg's value is (N - NU) / (W - WU). The harness format
has no offsets, so there a token is dropped if its word holds such a character and the token is not whitespace-only
or BOS. When a dev_bpw_twin.json sits beside a v2 dump (same dir, its parent, or score/), raw values are checked
against its nats_per_word and ex_unseen_chars_nats_per_word. Gains are first model minus each later one (positive = later model better). Known
idea: classic unigram interpolation; Xie et al. 2017, "Data noising as smoothing".

    python3 -I calib_control.py [--format harness|v2] BASE=dump ARM=dump ...
      harness: pt_devfull_final.npy (one loss per token of data/babylm/bin/dev.bin)
      v2:      dev_v2.npz from babylm.moonshot.dev_bpw_dump --twin; token ids are rebuilt with the twin protocol
               (dev_bpw.load_text + dev_bpw_twin.documents, mirrored below) and checked against the dump
    python3 -I calib_control.py --selftest_twin dev_bpw_twin.json   (rebuilt per-file token counts vs a twin json)

Documents (for the parity split) are the BOS-separated documents of dev.bin, mapped to whitespace words, so both
formats use the identical split.

Temperature axis (critic ref-0011 / ref-0499; binding only once pinned with dev_bpw_dump --temps): a v2 dump that
also carries 'temps' (exactly 0.90, 0.95, 1.05, 1.10, 1.15, 1.20) and 'nats_temps' ([tokens, 6], per-token nats of
softmax(z / T)) gets 'calibrated_t': the same cross-fit over {1.0} + temps x lam, jointly, on both legs, with
't_edge' set when either half picks T = 0.90 or 1.20 (then the grid is extended for every model). 'calibrated'
(lam only, T = 1) is reported unchanged beside it.
"""
import argparse
import bisect
import json
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, 'data', 'babylm')
GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.05)
TEMPS = (0.90, 0.95, 1.05, 1.10, 1.15, 1.20)  # critic ref-0011: symmetric T grid around 1.0 (T = 1 is the dump's 'nats')

# --- mirrored from babylm/eval/dev_bpw.py (load_text) and babylm/eval/dev_bpw_twin.py (documents); any drift fails
# the token-count checks below.
LEAD = re.compile(r'\A(?:[^\S\n]*\n)+')
TRAIL = re.compile(r'\n(?:[^\S\n]*\n)*[^\S\n]*\Z')
SEP = re.compile(r'\n(?:[^\S\n]*\n)+')


def load_text(path, ws='collapse', line_join='newline', tab_space=()):
    lines = open(path, encoding='utf-8').readlines()
    if any(os.path.basename(path).startswith(s) for s in tab_space):
        lines = [l.replace('\t', ' ') for l in lines]
    if ws == 'collapse':
        lines = [' '.join(l.split()) + ('\n' if l.endswith('\n') else '') for l in lines]
    return ('' if line_join == 'newline' else ' ').join(lines)


def documents(text):
    return SEP.split(TRAIL.sub('\n', LEAD.sub('', text)))


def token_words(text, offsets):
    starts = [m.start() for m in re.finditer(r'\S+', text)]
    w = np.empty(len(offsets), dtype=np.int64)
    for i, (a, b) in enumerate(offsets):
        w[i] = max(0, bisect.bisect_right(starts, max(a, b - 1)) - 1)
    return w


def unseen_mask(text, offsets, chars):
    word_of = np.full(len(text), -1, dtype=np.int64)
    bad = []
    for k, m in enumerate(re.finditer(r'\S+', text)):
        word_of[m.start():m.end()] = k
        bad.append(any(c in chars for c in m.group()))
    bad = np.array(bad + [False], dtype=bool)
    return np.array([bool(bad[word_of[a:b]].any()) if b > a else False for a, b in offsets], dtype=bool)


def twin_encode(tok, text, sep=0, chars=None):
    ids, words, drop, base = [], [], [], 0
    for k, d in enumerate(documents(text)):
        if k:
            ids.append(sep)
            words.append(base - 1)
            drop.append(False)
        enc = tok.encode(d, add_special_tokens=False)
        nw = len(d.split())
        w = token_words(d, enc.offsets) + base if nw else np.full(len(enc.ids), base - 1)
        ids += enc.ids
        words += w.tolist()
        if chars is not None:
            drop += unseen_mask(d, enc.offsets, chars).tolist()
        base += nw
    return np.asarray(ids, dtype=np.int64), np.asarray(words, dtype=np.int64), np.asarray(drop, dtype=bool)
# --- end mirror


def tokenizer():
    from tokenizers import Tokenizer
    p = os.environ.get('CALIB_TOKENIZER') or os.path.join(DATA, 'tokenizer', 'tokenizer.json')
    if not os.path.exists(p):
        p = 'shared/babylm/base_r1/base_S_wd0p4_lr0p5_r1_ve0_tied/hf/tokenizer.json'
    return Tokenizer.from_file(p)


def dev_files():
    d = os.path.join(DATA, 'raw', 'dev')
    return d, sorted(os.listdir(d))


def shared():
    """train_10M unigram, DEV word -> document parity, DEV word -> never-seen-character flag."""
    b = os.path.join(DATA, 'bin')
    dev = np.fromfile(os.path.join(b, 'dev.bin'), dtype=np.uint16)
    wid = np.fromfile(os.path.join(b, 'dev.words.bin'), dtype=np.uint32).astype(np.int64)
    tr = np.fromfile(os.path.join(b, 'train_10M.bin'), dtype=np.uint16)
    V = int(max(dev.max(), tr.max())) + 1
    c = np.bincount(tr, minlength=V).astype(np.float64)
    u = (c + 0.5) / (c.sum() + 0.5 * V)
    W = json.load(open(os.path.join(b, 'dev.meta.json')))['words']
    doc_tok = np.cumsum(dev == 0) - 1
    word_doc = np.full(W, -1, dtype=np.int64)
    nb = dev != 0
    word_doc[wid[nb]] = doc_tok[nb]
    assert (word_doc >= 0).all(), 'a DEV word has no non-BOS token'
    lst = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unseen_chars_list.json')))
    never = {c['char'] for c in lst['chars']}
    assert len(never) == lst['count'] == 85
    d, files = dev_files()
    words = []
    for f in files:
        words += open(os.path.join(d, f), encoding='utf-8').read().split()
    assert len(words) == W
    bad = np.fromiter((any(ch in never for ch in w) for w in words), dtype=bool, count=W)
    return dict(dev=dev, wid=wid, u=u, W=W, odd=(word_doc % 2).astype(bool), bad=bad, never=never)


def ws_only_table(tok):
    """Per vocab id: True if the token's bytes are all whitespace (GPT-2 byte-level vocabulary)."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    vocab = tok.get_vocab(with_added_tokens=True)
    t = np.zeros(max(vocab.values()) + 1, dtype=bool)
    for piece, i in vocab.items():
        try:
            by = bytes(u2b[ch] for ch in piece)
        except KeyError:
            continue
        t[i] = len(by) > 0 and all(x in b' \t\n\r\x0b\x0c' for x in by)
    return t


def load_harness(path, S, ws_only):
    x = np.load(path).astype(np.float64)
    assert x.size == S['dev'].size, f'{path}: {x.size} losses vs {S["dev"].size} dev.bin tokens'
    ok = ~np.isnan(x)
    x[~ok] = 0
    ids = S['dev'].astype(np.int64)
    drop = S['bad'][S['wid']] & ~ws_only[ids] & (ids != 0)
    return x, ids, S['wid'], ok, drop, None, None, None


def load_v2(path, S, tok, cache={}):
    z = np.load(path, allow_pickle=False)
    proto = json.loads(str(z['protocol']))
    assert proto.get('twin'), f'{path}: not a --twin dump'
    key = (proto['ws'], proto['line_join'], proto['tab_space'])
    if key not in cache:
        d, files = dev_files()
        assert list(z['files']) == files, f'{path}: file list differs from data/babylm/raw/dev'
        ids, words, drops, off = [], [], [], 0
        for f, nt, nw in zip(files, z['file_tokens'], z['file_words']):
            text = load_text(os.path.join(d, f), proto['ws'], proto['line_join'],
                             tuple(x for x in proto['tab_space'].split(',') if x))
            i, w, dr = twin_encode(tok, text, chars=S['never'])
            assert len(i) == nt, f'{f}: rebuilt {len(i)} tokens vs dump {nt}'
            assert len(text.split()) == nw
            ids.append(i)
            words.append(w + off)
            drops.append(dr)
            off += nw
        cache[key] = (np.concatenate(ids), np.concatenate(words), np.concatenate(drops))
    ids, words, drop = cache[key]
    steps = np.concatenate([np.diff(w - w[0], prepend=0) for w in np.split(words, np.cumsum(z['file_tokens'])[:-1])])
    assert np.array_equal(steps.astype(np.uint8), z['word_step']), f'{path}: rebuilt word map differs from the dump'
    x = z['nats'].astype(np.float64)
    ref = None
    for d in (os.path.dirname(path), os.path.dirname(os.path.dirname(path)), os.path.join(os.path.dirname(os.path.dirname(path)), 'score')):
        if os.path.exists(os.path.join(d, 'dev_bpw_twin.json')):
            ref = json.load(open(os.path.join(d, 'dev_bpw_twin.json')))
            break
    xt = temps = None
    if 'nats_temps' in z.files:
        temps = [float(t) for t in z['temps']]
        xt = z['nats_temps']
        assert xt.shape == (x.size, len(temps)), f'{path}: nats_temps {xt.shape} vs {x.size} tokens x {len(temps)} temps'
        assert len(temps) == len(TEMPS) and all(abs(t - r) < 1e-3 for t, r in zip(temps, TEMPS)), \
            f'{path}: temps {temps} != registered {TEMPS} (critic ref-0499)'
    return x, ids, words, np.ones(x.size, bool), drop, ref, xt, temps


def calibrate(x, ids, words, ok, drop, S):
    W, WU = S['W'], int(S['bad'].sum())
    p = np.exp(-x)
    ut = S['u'][ids]
    odd = S['odd'][words]
    out = {}
    for leg, m, den in (('full', ok, W), ('guarded', ok & ~drop, W - WU)):
        sums = np.empty((len(GRID), 2))
        for i, lam in enumerate(GRID):
            l = x if lam == 0 else -np.log((1 - lam) * p + lam * ut)
            sums[i] = (l[m & ~odd].sum(), l[m & odd].sum())
        le, lo = int(np.argmin(sums[:, 0])), int(np.argmin(sums[:, 1]))
        out[leg] = dict(raw=float(x[m].sum() / den), calibrated=float((sums[le, 1] + sums[lo, 0]) / den),
                        lam_even=GRID[le], lam_odd=GRID[lo], words=den)
    return out


def calibrate_t(x, xt, temps, ids, words, ok, drop, S):
    """calibrate() over the grid {1.0} + temps x lam: one (T, lam) picked on each document-parity half, read on the
    other. Returns per leg calibrated_t and the picks."""
    W, WU = S['W'], int(S['bad'].sum())
    ut = S['u'][ids]
    odd = S['odd'][words]
    tg = [1.0] + list(temps)
    cols = [x] + [xt[:, j].astype(np.float64) for j in range(len(temps))]
    out = {}
    for leg, m, den in (('full', ok, W), ('guarded', ok & ~drop, W - WU)):
        sums = np.empty((len(tg) * len(GRID), 2))
        for a, xc in enumerate(cols):
            p = np.exp(-xc)
            for i, lam in enumerate(GRID):
                l = xc if lam == 0 else -np.log((1 - lam) * p + lam * ut)
                sums[a * len(GRID) + i] = (l[m & ~odd].sum(), l[m & odd].sum())
        le, lo = int(np.argmin(sums[:, 0])), int(np.argmin(sums[:, 1]))
        te, to = tg[le // len(GRID)], tg[lo // len(GRID)]
        out[leg] = dict(calibrated_t=float((sums[le, 1] + sums[lo, 0]) / den),
                        t_even=te, lam_t_even=GRID[le % len(GRID)], t_odd=to, lam_t_odd=GRID[lo % len(GRID)],
                        t_edge=bool({te, to} & {min(tg), max(tg)}))  # critic ref-0499: an edge pick extends the grid
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='*', help='NAME=dump, first = base')
    ap.add_argument('--format', choices=['harness', 'v2'], default='harness')
    ap.add_argument('--selftest_twin', help='dev_bpw_twin.json whose per-file token counts the rebuild must match')
    ap.add_argument('--json')
    a = ap.parse_args()
    if a.selftest_twin:
        ref = json.load(open(a.selftest_twin))
        tok = tokenizer()
        d, files = dev_files()
        for f in files:
            i, w, _ = twin_encode(tok, load_text(os.path.join(d, f), ref.get('ws', 'collapse'), ref.get('line_join', 'newline'),
                                              tuple(x for x in ref.get('tab_space', '').split(',') if x)))
            r = ref['per_domain'][f]
            print(f'{f:22s} rebuilt {len(i):9d} tokens, twin json {r["tokens"]:9d}  {"OK" if len(i) == r["tokens"] else "MISMATCH"}')
            assert len(i) == r['tokens']
        return
    S = shared()
    tok = tokenizer()
    ws_only = ws_only_table(tok)
    res = {}
    for spec in a.runs:
        name, path = spec.split('=', 1)
        x, ids, words, ok, drop, ref, xt, temps = (load_v2(path, S, tok) if a.format == 'v2'
                                                   else load_harness(path, S, ws_only))
        res[name] = calibrate(x, ids, words, ok, drop, S)
        if xt is not None:
            for leg, v in calibrate_t(x, xt, temps, ids, words, ok, drop, S).items():
                res[name][leg].update(v)
        r = res[name]
        if ref is not None:
            for leg, k in (('full', 'nats_per_word'), ('guarded', 'ex_unseen_chars_nats_per_word')):
                if ref.get(k) is not None:
                    good = abs(r[leg]['raw'] - ref[k]) <= 1e-5 * ref[k]
                    print(f'{name}: {leg} raw {r[leg]["raw"]:.6f} vs dev_bpw_twin.json {k} {ref[k]:.6f} '
                          f'{"OK" if good else "MISMATCH"}')
                    assert good, 'dump and twin json disagree'
        print(f"{name:12s} full raw {r['full']['raw']:.4f} cal {r['full']['calibrated']:.4f} "
              f"(lam {r['full']['lam_even']}/{r['full']['lam_odd']})  guarded raw {r['guarded']['raw']:.4f} "
              f"cal {r['guarded']['calibrated']:.4f} (lam {r['guarded']['lam_even']}/{r['guarded']['lam_odd']})")
        if 'calibrated_t' in r['full']:
            print(f"{'':12s} full cal_t {r['full']['calibrated_t']:.4f} (T,lam {r['full']['t_even']},{r['full']['lam_t_even']}/"
                  f"{r['full']['t_odd']},{r['full']['lam_t_odd']})  guarded cal_t {r['guarded']['calibrated_t']:.4f} "
                  f"(T,lam {r['guarded']['t_even']},{r['guarded']['lam_t_even']}/{r['guarded']['t_odd']},"
                  f"{r['guarded']['lam_t_odd']})" + ('  T GRID EDGE: extend the grid for every model (critic ref-0011)'
                                                     if r['full']['t_edge'] or r['guarded']['t_edge'] else ''))
    names = list(res)
    for n in names[1:]:
        b, r = res[names[0]], res[n]
        print(f'{names[0]} - {n}: ' + '  '.join(f'{leg} raw {b[leg]["raw"] - r[leg]["raw"]:+.4f} '
                                                   f'cal {b[leg]["calibrated"] - r[leg]["calibrated"]:+.4f}'
                                                   + (f' cal_t {b[leg]["calibrated_t"] - r[leg]["calibrated_t"]:+.4f}'
                                                      if 'calibrated_t' in b[leg] and 'calibrated_t' in r[leg] else '')
                                                   for leg in ('full', 'guarded')))
    if a.json:
        json.dump({'format': a.format, 'grid': GRID, 'runs': res}, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
