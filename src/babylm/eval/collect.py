"""Collect official-pipeline results into one table (JSON + markdown).

Usage: python collect.py JOB_OUT_DIR [JOB_OUT_DIR ...] > table.md   (also writes JOB_OUT_DIR/scores.json)

Reads OUT/eval/<model>/<revision>/zero_shot/<backend>/<task>/<dataset>/best_temperature_report.txt,
OUT/eval/<model>/<revision>/finetune/<task>/results.txt, reading-score files, and OUT/ties_*.jsonl.
Aggregates (experimenter, docs/babylm/facts_2025.md; reproduce Findings Table 3):
  NLP   = mean(BLiMP, Supplement, EWoK, Entity, COMPS, GLUE7)    GLUE7 = mean(BoolQ, MNLI, MRPC-F1, MultiRC, QQP-F1, RTE, WSC)
  Human = mean(WUG-adj, WUG-past, Reading, AoA)                  (AoA not run yet -> reported as partial)
"""
import json, os, re, sys, glob
from collections import defaultdict

ZS = {('blimp', 'blimp_filtered'): 'BLiMP', ('blimp', 'supplement_filtered'): 'Supplement', ('blimp', 'blimp_fast'): 'BLiMP',
      ('blimp', 'supplement_fast'): 'Supplement', ('entity_tracking', None): 'Entity', ('wug_adj', None): 'WUG_adj',
      ('wug_past', None): 'WUG_past', ('comps', None): 'COMPS', ('ewok', None): 'EWoK'}
GLUE = {'boolq': 'accuracy', 'mnli': 'accuracy', 'mrpc': 'f1', 'multirc': 'accuracy', 'qqp': 'f1', 'rte': 'accuracy', 'wsc': 'accuracy'}


def avg_from_report(path):
    lines = open(path).read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith('### AVERAGE'):
            return float(lines[i + 1])
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def collect(out):
    rows = {}
    for rep in glob.glob(f'{out}/eval/*/*/zero_shot/*/*/*/best_temperature_report.txt'):
        parts = rep.split('/')
        model, rev, backend, task, ds = parts[-7], parts[-6], parts[-4], parts[-3], parts[-2]
        key = ZS.get((task, ds)) or ZS.get((task, None))
        v = avg_from_report(rep)
        if key in ('WUG_adj', 'WUG_past') and v is not None and abs(v) <= 1:
            v *= 100  # leaderboard reports rho x 100
        rows.setdefault((model, rev, backend), {})[key] = v
    for res in glob.glob(f'{out}/eval/*/*/finetune/*/results.txt'):
        parts = res.split('/')
        model, rev, task = parts[-5], parts[-4], parts[-2]
        kv = dict(l.split(': ', 1) for l in open(res).read().splitlines() if ': ' in l)
        v = float(kv[GLUE[task]]) if GLUE.get(task) in kv else None
        if v is not None and v <= 1: v *= 100
        for (m, r, b), d in rows.items():
            if m == model and r == rev: d[f'GLUE_{task}'] = v
        rows.setdefault((model, rev, 'finetune'), {})[f'GLUE_{task}'] = v
    for f in glob.glob(f'{out}/eval/*/**/*', recursive=True):
        if not os.path.isfile(f) or os.path.getsize(f) > 10_000_000: continue
        try: txt = open(f, errors='ignore').read()
        except Exception: continue
        for lab, key in (('EYE TRACKING SCORE', 'Reading_rt'), ('SELF-PACED READING SCORE', 'Reading_spr')):
            m = re.search(lab + r': ([-\d.]+)', txt)
            if m:
                parts = os.path.relpath(f, f'{out}/eval').split('/')
                for (mm, r, b), d in rows.items():
                    if mm == parts[0] and r == parts[1] and b in parts: d[key] = float(m.group(1))
    ties = defaultdict(lambda: [0, 0, 0])
    for tf in glob.glob(f'{out}/ties_*.jsonl'):
        tag = os.path.basename(tf)[5:-6]
        for l in open(tf):
            j = json.loads(l)
            t = ties[(tag, j['task'], os.path.basename(j['data_path'].rstrip('/')))]
            t[0] += j['n']; t[1] += j['ties']; t[2] += j['nonfinite']
    table = []
    for (model, rev, backend), d in sorted(rows.items()):
        if backend == 'finetune': continue
        g = [d.get(f'GLUE_{t}') for t in GLUE]
        d['GLUE7'] = mean(g) if all(x is not None for x in g) else None
        d['Reading'] = mean([d.get('Reading_rt'), d.get('Reading_spr')])
        d['NLP'] = mean([d.get(k) for k in ('BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GLUE7')]) \
            if all(d.get(k) is not None for k in ('BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GLUE7')) else None
        d['Human_partial_noAoA'] = mean([d.get(k) for k in ('WUG_adj', 'WUG_past', 'Reading')])
        table.append(dict(model=model, revision=rev, backend=backend, **d))
    json.dump({'rows': table, 'ties': {'|'.join(k): v for k, v in ties.items()}}, open(f'{out}/scores.json', 'w'), indent=1)
    return table, ties


COLS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GLUE7', 'NLP', 'WUG_adj', 'WUG_past', 'Reading_rt', 'Reading_spr']
if __name__ == '__main__':
    print('| model | rev | backend | ' + ' | '.join(COLS) + ' |')
    print('|' + '---|' * (3 + len(COLS)))
    all_ties = {}
    for out in sys.argv[1:]:
        table, ties = collect(out)
        all_ties.update(ties)
        for r in table:
            print(f"| {r['model']} | {r['revision']} | {r['backend']} | " + ' | '.join('' if r.get(c) is None else f'{r[c]:.2f}' for c in COLS) + ' |')
    print('\nHeld-out DEV bits/word (dev_bpw.py; ex-markup = without words carrying CHILDES CA symbols, experimenter 10 Oct):')
    print('| file | ws | line_join | tab_space | bits/word | ex-markup bits/word | childes | childes ex-markup | markup words |')
    print('|---|---|---|---|---|---|---|---|---|')
    for out in sys.argv[1:]:
        for f in sorted(glob.glob(f'{out}/dev_bpw_*.json')):
            j = json.load(open(f)); c = j['per_domain'].get('childes.dev', {})
            fmt = lambda v: '' if v is None else f'{v:.4f}'
            print(f"| {os.path.basename(f)} | {j.get('ws')} | {j.get('line_join')} | {j.get('tab_space')} | {fmt(j['bits_per_word'])} | "
                  f"{fmt(j.get('ex_markup_bits_per_word'))} | {fmt(c.get('bits_per_word'))} | {fmt(c.get('ex_markup_bits_per_word'))} | "
                  f"{j.get('markup_words', '')} |")
    print('\nArgmax ties (n, ties, rows with non-finite log-probs) per run tag / task / dataset:')
    for k, (n, t, nf) in sorted(all_ties.items()):
        print(f'- {" / ".join(k)}: n={n} ties={t} nonfinite={nf}')
