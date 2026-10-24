"""Merge temp_check.py outputs scored on the same windows into one file (models renumbered in argument order)."""
import sys
import numpy as np
out, ins = sys.argv[1], sys.argv[2:]
rs = [np.load(p) for p in ins]
for r in rs[1:]:
    assert np.array_equal(r['starts'], rs[0]['starts']) and np.array_equal(r['y'], rs[0]['y']) and np.array_equal(r['Ts'], rs[0]['Ts'])
res = {k: rs[0][k] for k in ('starts', 'Ts', 'y')}
k = 0
for r in rs:
    for j in range(sum(1 for f in r.files if f.startswith('zy'))):
        res[f'zy{k}'], res[f'lse{k}'] = r[f'zy{j}'], r[f'lse{j}']
        k += 1
np.savez(out, **res)
print(out, k, 'models')
