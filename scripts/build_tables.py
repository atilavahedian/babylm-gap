"""Regenerate data/*.csv (every number printed in the paper's tables and Figure 1) from the experiment records.

Usage, from the repository root: python3 scripts/build_tables.py [--table tab:map|fig:bands|tab:failed|tab:ensemble|tab:official|tab:rung0]
Without --table it writes all six CSVs. Then python3 scripts/check_tables.py checks them against paper/main.tex."""
import csv, json, statistics as st, sys
import os
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + '/'
OUT = S + 'data/'
def J(p): return json.load(open(S + p))
BY = IJ = ''
CL = 'experiments/2026-10-09-babylm-official-ruler/governance/critic_log.md @ '
rows_out = {}
def put(table, r, c, vals):
    for k, (v, how, src) in enumerate(vals):
        rows_out.setdefault(table, []).append(dict(table=table, row=r, col=c, k=k, value=repr(float(v)), derivation=how, source=src))

# Table 1 and Figure 1: gap map
bias_p = 'experiments/2026-10-09-babylm-gap-map/results/d1_bias/bias.json'
b = {(r['leg'], r['text'], r['band']): r for r in J(BY + bias_p)['rows']}
a = b[('raw', 'all', 'all')]
gap = a['member'] - a['t100']
for i, key in enumerate(['member-ens8', 'ens8-c1oracle', 'c1oracle-t100']):
    put('tab:map', i, 1, [(a[key], f'raw/all/all {key}', bias_p)])
    put('tab:map', i, 2, [(100 * a[key] / gap, f'100 * {key} / (member - t100)', bias_p)])
put('tab:map', 3, 1, [(gap, 'raw/all/all member - t100', bias_p)])
bands = ['c10=8k+', 'c10=64-8k', 'c10=1-63', 'c10=0']
put('fig:bands', 0, 0, [(100 * b[('raw', 'all', x)]['word_share'], f'100 * word_share {x}', bias_p) for x in bands])
for i, key in enumerate(['member-ens8', 'ens8-c1oracle', 'c1oracle-t100']):
    put('fig:bands', i + 1, 0, [(b[('raw', 'all', x)]['word_share'] * b[('raw', 'all', x)][key], f'word_share * {key}, {x}', bias_p) for x in bands])

# Table 2: failed attacks
a4a = 'experiments/2026-10-09-babylm-attacks/results/gcp_reads/a4_b1/a4_a/score/dev_bpw_twin.json'; a4n = 'experiments/2026-10-09-babylm-attacks/results/gcp_reads/a4_b1/a4_n0a/score/dev_bpw_twin.json'
A, N = J(BY + a4a), J(BY + a4n)
sm = 'experiments/2026-10-09-babylm-gap-map/results/base_size_m/SHORT_E.md'
f4 = 'experiments/2026-10-10-babylm-moonshot-10x/results/f4_menu/README.md'
s1a_p = 'experiments/2026-10-12-babylm-rung0-data-wall/results/calib_s1a.json'; lx_p = 'experiments/2026-10-12-babylm-rung0-data-wall/results/calib_lx.json'
s1a = J(BY + s1a_p)['single']; lx = J(BY + lx_p)['single']
ct = lambda d, m, leg: d[m][leg]['calibrated_t']
T = 'tab:failed'
put(T, 0, 2, [(0.0220, 'base minus A2a, full raw', CL + '2026-10-10T11:24Z'), (0.0128, 'guarded raw', CL + '2026-10-10T11:24Z'),
              (0.0047, 'full calibrated_t', CL + '2026-10-10T11:24Z'), (-0.0025, 'guarded calibrated_t', CL + '2026-10-10T11:24Z')])
put(T, 0, 3, [(0.0102, 'packed kill line', CL + '2026-10-10T11:24Z')])
put(T, 1, 2, [(0.0040, 'a3a minus null, full raw', CL + '2026-10-10T11:56Z')])
put(T, 1, 3, [(0.0074, 'screen kill line', CL + '2026-10-10T10:51Z')])
put(T, 2, 2, [(0.0431, 'A3g E10 vs E6 pool, full cal_t', CL + '2026-10-10T22:27Z'), (0.0458, 'guarded cal_t', CL + '2026-10-10T22:27Z'),
              (-0.0128, 'arm minus 28L e6 reference, full cal_t', CL + '2026-10-10T22:27Z'), (-0.0072, 'guarded cal_t', CL + '2026-10-10T22:27Z'),
              (0.0138, 'plain K=2 minus A3g K=2, 3-rep mean, full cal_t', CL + '2026-10-11T00:55Z'), (0.0129, 'guarded cal_t', CL + '2026-10-11T00:55Z')])
put(T, 2, 3, [(0.0112, 'E6-pool claim floor, full', CL + '2026-10-10T22:27Z'), (0.020, 'stack bar', CL + '2026-10-10T21:14Z')])
put(T, 3, 2, [(N['nats_per_word'] - A['nats_per_word'], 'n0a minus a4_a nats_per_word (negative = arm worse)', a4a + ' ; ' + a4n),
              (N['ex_unseen_chars_nats_per_word'] - A['ex_unseen_chars_nats_per_word'], 'same, ex_unseen_chars', a4a + ' ; ' + a4n)])
put(T, 3, 3, [(0.0074, 'solo screen kill line', CL + '2026-10-10T19:30Z')])
put(T, 4, 2, [(0.0502, 'student margin over N-single, full', CL + '2026-10-10T13:23Z'), (0.0506, 'guarded', CL + '2026-10-10T13:23Z')])
put(T, 4, 3, [(0.107, '2 x 8-member floor', CL + '2026-10-10T13:23Z')])
put(T, 4, 4, [(a['member-ens8'], 'noise slice, raw/all/all member-ens8', bias_p)])
put(T, 5, 2, [(36.4, '100 * rho, full leg 0.3289 / 0.9030', CL + '2026-10-10T16:11Z'), (-0.037, 'K=2 beats trunk+branches, full', CL + '2026-10-10T21:28Z'),
              (-0.037, 'guarded', CL + '2026-10-10T21:28Z')])
put(T, 5, 3, [(50, 'kill if rho < 0.5', CL + '2026-10-10T16:11Z')])
put(T, 6, 2, [(0.003, 'hetero diversity, low end', CL + '2026-10-10T21:13Z'), (0.007, 'high end', CL + '2026-10-10T21:13Z')])
put(T, 6, 3, [(0.03, 'kill line', CL + '2026-10-10T21:13Z')])
put(T, 7, 2, [(5.1059 - 4.9635, 'best size-M cell (E10 wd0.8) minus S E6, raw twin', sm), (5.3577 - 4.9635, 'worst (wd3.2 e5) minus S E6', sm),
              (5.1745 - 4.8443, 'best short-epoch size-M cell (wd1.6 e3) minus K=2 e5 mix (P-F2e)', sm + ' ; ' + CL + '2026-10-10T21:51Z')])
put(T, 7, 3, [(-0.06, 'P-F2e registered band, low', CL + '2026-10-10T21:51Z'), (0.02, 'high', CL + '2026-10-10T21:51Z')])
put(T, 7, 4, [(4.8758 - 4.8728, '12L/880d e4 minus 36L/512d e4, cal_t full', f4), (4.8524 - 4.8461, 'guarded', f4)])
put(T, 8, 2, [(ct(s1a, 'C04', 'full') - ct(s1a, 'W08', 'full'), 'C minus W0.8, cal_t full', s1a_p),
              (ct(s1a, 'C04', 'guarded') - ct(s1a, 'W08', 'guarded'), 'cal_t guarded', s1a_p),
              (0.4, 'design: control weight decay', CL + '2026-10-12T05:37Z')])
put(T, 8, 3, [(0.0074, 'kill line', CL + '2026-10-12T11:13Z')])
put(T, 9, 2, [(ct(lx, 'P24', 'full') - ct(lx, 'L12x2', 'full'), 'P24 minus L12x2, cal_t full', lx_p),
              (ct(lx, 'P24', 'guarded') - ct(lx, 'L12x2', 'guarded'), 'cal_t guarded', lx_p),
              (ct(lx, 'C04', 'full') - ct(lx, 'L12x2', 'full'), 'C minus L12x2, cal_t full', lx_p),
              (ct(lx, 'C04', 'guarded') - ct(lx, 'L12x2', 'guarded'), 'cal_t guarded', lx_p)])
put(T, 9, 3, [(0.0074, 'kill line', CL + '2026-10-12T12:59Z')])

# Table 3: ensemble yardstick
a5p = 'experiments/2026-10-09-babylm-official-ruler/results/readouts/a5_xfactor_final_a5_out.json'; a2p = 'experiments/2026-10-09-babylm-official-ruler/results/readouts/a5_xfactor_final_a2_registered.json'
g5 = J(IJ + a5p)['gains']; a2 = J(IJ + a2p)
def row5(key):
    r = g5[key]['rows']; return [(r['all/full']['g'], f'{key} all/full g', a5p), (r['all/guarded']['g'], 'all/guarded g', a5p)], \
        [(r['all/full']['x_ci95'][0], 'all/full x_ci95 low', a5p)], [(r['all/full']['x'][0], 'all/full x', a5p)]
def put3(r, trip):
    for c, v in zip((2, 3, 4), trip): put('tab:ensemble', r, c, v)
put3(0, row5('S_same_size_vs_E6_pool'))
put3(1, ([(a2['all/full']['g'], 'registered single-seed comparator, all/full g', a2p), (a2['all/guarded']['g'], 'all/guarded g', a2p)],
         [(a2['all/full']['x_ci95'][0], 'all/full x_ci95 low', a2p)], [(a2['all/full']['x'], 'all/full x', a2p)]))
put3(2, row5('A2_S_matched_vs_28L_e6_F2'))
put3(3, row5('28L_same_size_vs_e6_mean'))
put3(4, row5('A4_28L_matched_vs_best_F4'))

# Table 4: official scores
zsS = 'experiments/2026-10-09-babylm-official-ruler/results/readouts/zs_S_split_vs_E6.json'; zs28 = 'experiments/2026-10-09-babylm-official-ruler/results/readouts/zs_28L_mix_vs_single.json'
gl28 = 'experiments/2026-10-09-babylm-official-ruler/results/readouts/glue_28L_mix_vs_single.json'
D = {p: J(IJ + p) for p in (zsS, zs28, gl28)}
def ms(p, prefix, field):
    v = [m[field] for k, m in D[p]['models'].items() if k.startswith(prefix)]
    assert len(v) == 3, (p, prefix, field)
    return [(st.mean(v), f'mean of {prefix}* {field} (n=3)', p), (st.stdev(v), f'sample SD of {prefix}* {field}', p)]
def db(p, key, field):
    x = D[p]['beats'][key][field]; return [(x['diff'], f'beats {key} {field} diff', p), (x['bar'], f'beats {key} {field} bar', p)]
O = 'tab:official'
put(O, 0, 1, ms(zsS, 'mixS_', 'composite')); put(O, 0, 2, ms(zs28, 'mix28L_', 'composite'))
put(O, 1, 1, ms(zsS, 'S_e6_', 'composite')); put(O, 1, 2, ms(zs28, 'L28_e6_', 'composite'))
put(O, 2, 1, db(zsS, 'S_split:S_E6', 'composite')); put(O, 2, 2, db(zs28, 'mix28L:single28L', 'composite'))
put(O, 3, 1, ms(zsS, 'mixS_', 'GLUE7')); put(O, 3, 2, ms(gl28, 'mix28L_', 'GLUE7'))
put(O, 4, 1, ms(zsS, 'S_e6_', 'GLUE7')); put(O, 4, 2, ms(gl28, 'L28_e6_', 'GLUE7'))
put(O, 5, 1, db(zsS, 'S_split:S_E6', 'GLUE7')); put(O, 5, 2, db(gl28, 'mix28L:single28L', 'GLUE7'))

# Table 5: rung 0
cells = [('experiments/2026-10-12-babylm-rung0-data-wall/results/calib_s1a.json', 'C04'), ('experiments/2026-10-12-babylm-rung0-data-wall/results/calib_s1a.json', 'W02'), ('experiments/2026-10-12-babylm-rung0-data-wall/results/calib_s1a.json', 'W08'),
         ('experiments/2026-10-12-babylm-rung0-data-wall/results/calib_s1b.json', 'W16'), ('experiments/2026-10-12-babylm-rung0-data-wall/results/calib_s1b.json', 'W32'), ('experiments/2026-10-12-babylm-rung0-data-wall/results/calib_lr.json', 'LR08'),
         ('experiments/2026-10-12-babylm-rung0-data-wall/results/calib_lx.json', 'L12x2'), ('experiments/2026-10-12-babylm-rung0-data-wall/results/calib_lx.json', 'P24')]
for i, (p, m) in enumerate(cells):
    d = J(BY + p)['single']
    put('tab:rung0', i, 1, [(d[m]['full']['calibrated_t'], f'{m} full calibrated_t', p)])
    put('tab:rung0', i, 2, [(d[m]['guarded']['calibrated_t'], f'{m} guarded calibrated_t', p)])
Y = J(IJ + a5p)['Y']
put('tab:rung0', 8, 1, [(Y['all/full']['pool_mean'], 'E6 pool mean, all/full', a5p)])
put('tab:rung0', 8, 2, [(Y['all/guarded']['pool_mean'], 'E6 pool mean, all/guarded', a5p)])

names = {'tab:map': 'table1_gap_map.csv', 'fig:bands': 'figure1_bands.csv', 'tab:failed': 'table2_failed_attacks.csv',
         'tab:ensemble': 'table3_ensemble.csv', 'tab:official': 'table4_official.csv', 'tab:rung0': 'table5_rung0.csv'}
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--table'); only = ap.parse_args().table
for t, rows in rows_out.items():
    if only and t != only:
        continue
    with open(OUT + names[t], 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['table', 'row', 'col', 'k', 'value', 'derivation', 'source'])
        w.writeheader(); w.writerows(rows)
    print(names[t], len(rows))
