"""Official readout B (critic ref-0185): zero-shot composite + GLUE7 per model, mean over replicates per arm, beats-test.

    python official_readout.py --arm S_split=mixS_p01,mixS_p23,mixS_p45 --arm S_E6=S_e6_s0,S_e6_s1,S_e6_s2 \
        [--beats S_split:S_E6 ...] [--json out.json] OUT_DIR [OUT_DIR ...]

OUT_DIRs are vm_job.sh output dirs (OUT/eval/<model>/main/...). The task list and score lines are fixed by
official_zs_tasks_pipeline2025.json (committed before the first claim-grade official number):
  composite = unweighted mean of the 8 composite tasks (BLiMP, Supplement, EWoK full, COMPS, WUG adj x100, WUG past x100,
              Reading eye tracking, Reading self-paced); a model missing any of them has no composite.
  Entity    = reported only.
  GLUE7     = mean of the 7 official GLUE metrics x100 from <model>_lastpool_ens2 (single causal model: final-token
              pooling, seeds 42+43) or, for a K=2 mixture export (MIX_GUARD_<model>.txt present), <model>_ens2; 7/7 or none.
Validity: VOID_<model> -> no scores; a mixture needs MIX_GUARD PASS (else no scores: DEV-equality ref-0347);
GLUE_INCOMPLETE_<g> or GLUE_ENS_FAILED_<g> -> no GLUE7. Each arm: mean, sample SD and n over its replicates, per column,
using only valid replicates; an arm with fewer than 3 valid replicates is labelled "no claim (n<3)".
Beats (A:R): A - R >= 2 sqrt(sd_A^2/n_A + sd_R^2/n_R), on composite and GLUE7 separately; per-task rows are descriptive.
Amended Reading (critic 81702719, "amended after data"): when eval/<model>_ps/... (vm_job.sh READING_PS=1: the same
weights, tokenizer add_prefix_space=True) is among the OUT_DIRs, a second table gives Reading_rt/spr from it and the
composite with those two swapped in. Amended arm diffs are printed with their bars but labelled "amended, no claim":
they never change a verdict; they only put absolute numbers on GPT-BERT's / the leaderboard's tokenization convention.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, 'official_zs_tasks_pipeline2025.json')))
GLUE = {'boolq': 'accuracy', 'mnli': 'accuracy', 'mrpc': 'f1', 'multirc': 'accuracy', 'qqp': 'f1', 'rte': 'accuracy',
        'wsc': 'accuracy'}   # = collect.py and the README: accuracies except MRPC and QQP (F1)
READ = {'Reading_rt': 'EYE TRACKING SCORE', 'Reading_spr': 'SELF-PACED READING SCORE'}


def report_avg(path):
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith('### AVERAGE'):
            return float(lines[i + 1])
    return None


def find(outs, rel):
    hits = [p for o in outs for p in glob.glob(os.path.join(o, rel))]
    if len(hits) > 1:
        raise SystemExit(f'ambiguous: {rel} found in {len(hits)} places: {hits}')
    return hits[0] if hits else None


def marker(outs, name):
    return any(os.path.exists(os.path.join(o, name)) for o in outs)


def model_scores(outs, m):
    r = {'model': m, 'valid': True, 'notes': []}
    if marker(outs, f'VOID_{m}'):
        r['valid'] = False; r['notes'].append('VOID')
    guard = find(outs, f'MIX_GUARD_{m}.txt')
    is_mix = guard is not None
    if is_mix:
        g = open(guard).read().strip()
        r['guard'] = g
        if not g.startswith('PASS'):
            r['valid'] = False; r['notes'].append('mixture guard not PASS')
    for t in SPEC['tasks']:
        p = find(outs, f"eval/{m}/main/{t['file']}")
        v = None
        if p is not None:
            if t['key'] in READ:
                mm = re.search(READ[t['key']] + r': ([-\d.]+)', open(p).read())
                v = float(mm.group(1)) if mm else None
            else:
                v = report_avg(p)
                if t['key'].startswith('WUG') and v is not None:
                    v *= 100   # leaderboard scale (rho x 100)
        r[t['key']] = v
    comp = [r[t['key']] for t in SPEC['tasks'] if t['composite']]
    r['composite'] = sum(comp) / len(comp) if all(v is not None for v in comp) else None
    if r['composite'] is None:
        r['notes'].append('zero-shot incomplete: ' + ','.join(t['key'] for t in SPEC['tasks'] if t['composite'] and r[t['key']] is None))
    gname = f'{m}_ens2' if is_mix else f'{m}_lastpool_ens2'
    r['glue_source'] = gname
    vals = []
    for task, met in GLUE.items():
        p = find(outs, f'eval/{gname}/main/finetune/{task}/results.txt')
        kv = dict(ln.split(': ', 1) for ln in open(p).read().splitlines() if ': ' in ln) if p else {}
        vals.append(float(kv[met]) * 100 if met in kv else None)
    bad = marker(outs, f'GLUE_INCOMPLETE_{gname}') or marker(outs, f'GLUE_ENS_FAILED_{gname}')
    r['GLUE7'] = sum(vals) / 7 if all(v is not None for v in vals) and not bad else None
    if r['GLUE7'] is None:
        r['notes'].append(f'no GLUE7 ({gname}: {sum(v is not None for v in vals)}/7{", failed marker" if bad else ""})')
    for t in SPEC['tasks']:
        if t['key'] in READ:
            p = find(outs, f"eval/{m}_ps/main/{t['file']}")
            mm = re.search(READ[t['key']] + r': ([-\d.]+)', open(p).read()) if p else None
            r[t['key'] + '_amended'] = float(mm.group(1)) if mm else None
    comp = [r.get(t['key'] + '_amended', r[t['key']]) for t in SPEC['tasks'] if t['composite']]
    r['composite_amended'] = (sum(comp) / len(comp) if all(v is not None for v in comp)
                              and all(r[k + '_amended'] is not None for k in READ) else None)
    if not r['valid']:
        for k in [t['key'] for t in SPEC['tasks']] + ['composite', 'GLUE7', 'composite_amended'] + [k + '_amended' for k in READ]:
            r[k] = None
    return r


def stats(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {'mean': None, 'sd': None, 'n': 0}
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else None
    return {'mean': mu, 'sd': sd, 'n': n}


def beats(a, r):
    if a['n'] < 3 or r['n'] < 3 or a['sd'] is None or r['sd'] is None:
        return {'diff': None if a['mean'] is None or r['mean'] is None else a['mean'] - r['mean'], 'bar': None,
                'beats': None, 'note': 'no claim (n<3)'}
    d, bar = a['mean'] - r['mean'], 2 * math.sqrt(a['sd'] ** 2 / a['n'] + r['sd'] ** 2 / r['n'])
    return {'diff': d, 'bar': bar, 'beats': d >= bar}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', default=[], help='NAME=model1,model2,...')
    ap.add_argument('--beats', action='append', default=[], help='ARM:REF')
    ap.add_argument('--json')
    ap.add_argument('outs', nargs='+')
    a = ap.parse_args()
    cols = [t['key'] for t in SPEC['tasks']] + ['composite', 'GLUE7']
    acols = [k + '_amended' for k in READ] + ['composite_amended']
    arms, out = {}, {'spec': SPEC['rule'], 'models': {}, 'arms': {}, 'beats': {}}
    for spec in a.arm:
        name, ms = spec.split('=', 1)
        rows = [model_scores(a.outs, m) for m in ms.split(',')]
        for r in rows:
            out['models'][r['model']] = r
        arms[name] = {c: stats([r[c] for r in rows]) for c in cols + acols}
        arms[name]['claim'] = 'ok' if min(arms[name]['composite']['n'], arms[name]['GLUE7']['n']) >= 3 else 'no claim (n<3)'
    out['arms'] = arms
    fmt = lambda v: '' if v is None else f'{v:.2f}'
    print('| model | ' + ' | '.join(cols) + ' | notes |')
    print('|' + '---|' * (len(cols) + 2))
    for m, r in out['models'].items():
        print(f'| {m} | ' + ' | '.join(fmt(r[c]) for c in cols) + f" | {'; '.join(r['notes'])} |")
    print('\n| arm | n (comp/GLUE) | composite mean (sd) | GLUE7 mean (sd) | Entity mean | claim |')
    print('|---|---|---|---|---|---|')
    for n, s in arms.items():
        c, g = s['composite'], s['GLUE7']
        print(f"| {n} | {c['n']}/{g['n']} | {fmt(c['mean'])} ({fmt(c['sd'])}) | {fmt(g['mean'])} ({fmt(g['sd'])}) | "
              f"{fmt(s['Entity']['mean'])} | {s['claim']} |")
    for b in a.beats:
        x, y = b.split(':')
        out['beats'][b] = {c: beats(arms[x][c], arms[y][c]) for c in ('composite', 'GLUE7')}
        for c, v in out['beats'][b].items():
            print(f"- {x} vs {y}, {c}: diff {fmt(v['diff'])}, bar {fmt(v['bar'])} -> "
                  f"{v.get('note') or ('BEATS' if v['beats'] else 'does not beat')}")
    if any(r['composite_amended'] is not None for r in out['models'].values()):
        print('\nAmended after data (critic 81702719): Reading with add_prefix_space=True, same weights; never changes a verdict')
        print('\n| model | Reading_rt (amended) | Reading_spr (amended) | composite (amended) |')
        print('|---|---|---|---|')
        for m, r in out['models'].items():
            print(f'| {m} | ' + ' | '.join(fmt(r[c]) for c in acols) + ' |')
        print('\n| arm | n | composite amended mean (sd) |')
        print('|---|---|---|')
        for n, s in arms.items():
            c = s['composite_amended']
            print(f"| {n} | {c['n']} | {fmt(c['mean'])} ({fmt(c['sd'])}) |")
        for b in a.beats:
            x, y = b.split(':')
            v = beats(arms[x]['composite_amended'], arms[y]['composite_amended'])
            out['beats'][b]['composite_amended'] = dict(v, label='amended, no claim')
            print(f"- {x} vs {y}, composite (amended): diff {fmt(v['diff'])}, bar {fmt(v['bar'])} -> amended, no claim")
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
