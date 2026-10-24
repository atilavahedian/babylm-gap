"""Check every number printed in main.tex's tables (and Figure 1's bars) against data/*.csv.

Each CSV row holds one printed number, keyed by (table label, data row, column, position in cell), with its
full-precision value, how it was derived and the record it comes from. A printed number passes when the value
rounds to it at the printed precision. Integers without a % sign (counts, layer numbers, epochs) are not results
and are skipped, as are the row-label columns listed in LABEL_COLS.

Usage: python3 check_tables.py   (from this directory). Exit status 1 on any mismatch.
"""
import csv
import glob
import re
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))

LABEL_COLS = {'tab:map': {0}, 'tab:failed': {0, 1}, 'tab:ensemble': {0, 1}, 'tab:official': {0}, 'tab:rung0': {0}}
NUM = re.compile(r'[-+]?\d+\.\d+%?|[-+]?\d+%')


def clean(cell):
    cell = re.sub(r'\\cite[pt]?\{[^}]*\}', '', cell)
    cell = cell.replace('$-$', '-').replace('$+$', '+').replace('\\%', '%').replace('--', ' to ')
    cell = re.sub(r'\$\^\\?[A-Za-z]+\$|\$\\[A-Za-z]+\$', ' ', cell)
    return cell


def numbers(cell):
    return NUM.findall(clean(cell))


def table_rows(tex, label):
    i = tex.index('\\label{%s}' % label)
    start = tex.rfind('\\begin{table', 0, i)
    body = tex[start:i]
    m = re.search(r'\\begin\{tabularx?\}(\{[^}]*\})?\{.*?\}\n', body)
    body = body[m.end():body.index('\\end{tabular')]
    body = body.split('\\midrule', 1)[1]  # drop the header row
    body = re.sub(r'\\(top|mid|bottom)rule', '', body)
    rows = [r.strip() for r in body.split('\\\\') if r.strip()]
    return [[c.strip() for c in r.split('&')] for r in rows]


def figure_rows(tex):
    i = tex.index('\\label{fig:bands}')
    body = tex[tex.rfind('\\begin{figure', 0, i):i]
    ticks = re.search(r'yticklabels=\{(.*)\},\s*\n', body).group(1)
    rows = [[ticks]]
    rows += [[c] for c in re.findall(r'coordinates \{(.*?)\};', body)]
    return rows


def main():
    tex = open(os.path.join(HERE, '..', 'paper', 'main.tex')).read()
    expected = {}
    for path in sorted(glob.glob(os.path.join(HERE, '..', 'data', '*.csv'))):
        for r in csv.DictReader(open(path)):
            expected[(r['table'], int(r['row']), int(r['col']), int(r['k']))] = (float(r['value']), r['source'], path)
    bad, checked, seen = [], 0, set()
    tables = sorted({k[0] for k in expected})
    for t in tables:
        rows = figure_rows(tex) if t.startswith('fig:') else table_rows(tex, t)
        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                if ci in LABEL_COLS.get(t, set()):
                    continue
                nums = numbers(cell) if not t.startswith('fig:') else re.findall(r'\(([-\d.]+),|\((\d+)\\%\)', cell)
                if t.startswith('fig:'):
                    nums = [a or b + '%' for a, b in nums]
                for k, s in enumerate(nums):
                    key = (t, ri, ci, k)
                    seen.add(key)
                    if key not in expected:
                        bad.append(f'{t} row {ri} col {ci}: printed {s} has no CSV entry')
                        continue
                    value, src, path = expected[key]
                    pct = s.endswith('%')
                    p = s.rstrip('%')
                    d = len(p.split('.')[1]) if '.' in p else 0
                    checked += 1
                    if abs(value - float(p)) > 0.5 * 10 ** -d + 1e-9:
                        bad.append(f'{t} row {ri} col {ci}: printed {s}, source {value:.6g} ({src})')
    for key in sorted(set(expected) - seen):
        bad.append(f'{key[0]} row {key[1]} col {key[2]} #{key[3]}: CSV value {expected[key][0]:.6g} not printed')
    print(f'{checked} printed numbers checked against {len(expected)} CSV entries in {len(tables)} tables/figures')
    for b in bad:
        print('MISMATCH', b)
    print('all match' if not bad else f'{len(bad)} mismatch(es)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
