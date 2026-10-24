"""Full-DEV per-word losses split by the word's train_10M frequency (exact whitespace word), with words holding a
character never present in train_10M split out first (see unseen_chars.py). Diagnostic only, never a verdict.
usage: python3 -I freq_bands.py BASE=file.npy ARM=file.npy ...   (pt_devfull_final.npy per token or pw_*.npy per word)
Prints, per band, each run's contribution to nats/word and each later run's gain over the first (base - arm)."""
import sys, json, os, collections
import numpy as np

B = 'data/babylm/'
R = B + 'raw/'
tr_text = {f: open(R + 'train_10M/' + f, encoding='utf-8').read() for f in sorted(os.listdir(R + 'train_10M'))}
cnt = collections.Counter()
for t in tr_text.values():
    cnt.update(t.split())
never = set()
dv_words = []
for f in sorted(os.listdir(R + 'dev')):
    t = open(R + 'dev/' + f, encoding='utf-8').read()
    never |= set(t)
    dv_words += t.split()
never -= set(''.join(tr_text.values()))
meta = json.load(open(B + 'bin/dev.meta.json'))
W = meta['words']
assert len(dv_words) == W, (len(dv_words), W)
wid = np.fromfile(B + 'bin/dev.words.bin', dtype=np.uint32)
src = np.fromfile(B + 'bin/dev.src.bin', dtype=np.uint8)
wsrc = np.zeros(W, np.uint8)
wsrc[wid] = src
freq = np.fromiter((cnt.get(w, 0) for w in dv_words), dtype=np.int64, count=W)
bad = np.fromiter((any(c in never for c in w) for w in dv_words), dtype=bool, count=W)
edges = [('never-seen char', None), ('0 (unseen word)', (0, 0)), ('1-7', (1, 7)), ('8-63', (8, 63)),
         ('64-1023', (64, 1023)), ('1k-8k', (1024, 8191)), ('8k+', (8192, 1 << 62))]
masks = {}
for name, e in edges:
    masks[name] = bad.copy() if e is None else (~bad) & (freq >= e[0]) & (freq <= e[1])
print(f'DEV words {W}; never-seen chars {len(never)}; words: ' +
      ', '.join(f'{k} {m.mean():.2%}' for k, m in masks.items()))
res = {}
for arg in sys.argv[1:]:
    name, p = arg.split('=', 1)
    x = np.load(p).astype(np.float64)
    x[np.isnan(x)] = 0
    pw = np.bincount(wid, weights=x, minlength=W)[:W] if x.size == wid.size else x[:W]
    res[name] = pw
names = list(res)
hdr = f"{'band':16s} {'share':>7s} " + ' '.join(f'{n[:12]:>12s}' for n in names) + '   ' + \
      ' '.join(f'{"gain " + n[:7]:>12s}' for n in names[1:]) + '   (contribution to nats/word; gain = first - run)'
print(hdr)
for k, m in masks.items():
    c = [res[n][m].sum() / W for n in names]
    print(f'{k:16s} {m.mean():7.2%} ' + ' '.join(f'{v:12.4f}' for v in c) + '   ' +
          ' '.join(f'{c[0] - v:+12.4f}' for v in c[1:]))
c = [res[n].sum() / W for n in names]
print(f"{'total':16s} {1:7.2%} " + ' '.join(f'{v:12.4f}' for v in c) + '   ' + ' '.join(f'{c[0] - v:+12.4f}' for v in c[1:]))
print('\nper source, words without never-seen chars (gain = first - run, nats/word contribution):')
for i, s in enumerate(meta['per_source']):
    m = (wsrc == i) & ~bad
    c = [res[n][m].sum() / W for n in names]
    print(f'{s:16s} {m.mean():7.2%} ' + ' '.join(f'{v:12.4f}' for v in c) + '   ' + ' '.join(f'{c[0] - v:+12.4f}' for v in c[1:]))
