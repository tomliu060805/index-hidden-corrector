# -*- coding: utf-8 -*-
"""两个悬而未决的诊断 (140 指数, H=6, 基线 B8+B9):

诊断①  train 段 QLIKE 为负 (-17.5%, t=-0.43) 而 val 为正 (+3.6%, t=2.97) —— 方向反常。
        查法: 分年看 QLIKE 改善, 并看 train 段的负值是不是集中在少数年份/少数极端点。
        若集中在早年(2014-15 股灾等), 则是 regime 问题而非模型问题。

诊断②  基线自身 train/val R² 差 +0.17 —— "增量比 0.99"有两种读法:
        (a) 无过拟合; (b) **基线在 train 上吃掉更多, val 增量是基线自己退化腾出的空间**。
        查法: 把 val 按"基线相对其 train 水平退化的程度"分组, 看增量是否集中在退化最厉害处。
        ★退化程度用**逐指数**口径: 该指数 val 的基线 R² 相对它 train 的基线 R² 掉了多少。
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB, har_1m_features as H1, extract_universe as U
from run_universe_eval import build_panel_u, load_hidden
from run_layerscan import ridge_path, LAMS
from dm_test_v2 import dm_stat, qlike_loss
OUT = f'{_R}/out'

ap = argparse.ArgumentParser(); ap.add_argument('--H', type=int, default=6)
a = ap.parse_args()
cur = json.load(open(f'{OUT}/index_universe_curated.json'))
cu0 = pd.read_csv(f'{OUT}/universe_curated_list.csv')
bs = set(cu0[cu0['名称'].astype(str).str.contains('B股|B指')]['代码'])
want = set(cur['codes']) - bs
_, _, allc = U.load_grid()
jsel = np.array([i for i, c in enumerate(allc) if c in want])
Hm, key = load_hidden()
d, y, F, codes = build_panel_u(a.H, jsel, 'rv')
m1 = json.load(open(f'{OUT}/index1m_grid_140_meta.json'))
arr1m = np.load(f'{OUT}/index1m_grid_140.npy', mmap_mode='r')
loc = {c: i for i, c in enumerate(m1['codes'])}
order = np.array([loc[allc[j]] for j in jsel])
for k, v in H1.build_1m_features(np.asarray(arr1m[:, order, :])).items():
    F[k] = v[d['t'].values, d['jloc'].values]
del arr1m
k = d['t'].values * 1000 + d['j'].values
p = np.clip(np.searchsorted(key, k), 0, len(key) - 1); good = key[p] == k
d, y = d[good].reset_index(drop=True), y[good]; F = {kk: v[good] for kk, v in F.items()}
Z = Hm[p[good], 1, :].astype(np.float32)
tr = (d.seg == 'train').values; va = (d.seg == 'val').values
dts = d['date'].values
Xb = np.column_stack([HB.baseline_design(d, F, len(jsel), 'B8')] + [F[c] for c in H1.B9_EXTRA])
Xb = np.nan_to_num(Xb, nan=0., posinf=0., neginf=0.)
base = HB.ridge_fit(Xb, y, tr)
pr = ridge_path(Z, y - base, tr, LAMS)
lam = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va)); full = base + pr[lam]
q0, q1 = qlike_loss(y, base, tr), qlike_loss(y, full, tr)
e0, e1 = (y - base) ** 2, (y - full) ** 2
yr = pd.Series(dts).str[:4].values

print('\n=== 诊断① train 段 QLIKE 为负: 分年看 ===')
print(f"  {'年':<6s}{'段':<7s}{'n':>10s}{'ΔR²':>9s}{'QLIKE改善%':>12s}{'基线QLIKE均值':>14s}")
for Y in sorted(set(yr)):
    m = (yr == Y) & (tr | va)
    if m.sum() < 2000: continue
    seg = 'train' if (yr == Y)[tr].any() and not (yr == Y)[va].any() else ('val' if not (yr==Y)[tr].any() else 'mix')
    print(f'  {Y:<6s}{seg:<7s}{m.sum():>10,}{HB.r2(y,full,m)-HB.r2(y,base,m):>+9.4f}'
          f'{(1-q1[m].mean()/q0[m].mean())*100:>11.1f}%{q0[m].mean():>14.4f}')
# train 的负值是否集中在少数极端点
dq = (q0 - q1)[tr]
print(f'\n  train 段 QLIKE 差(q0-q1): 均值 {dq.mean():+.5f}  中位 {np.median(dq):+.5f}  '
      f'为正占比 {(dq>0).mean()*100:.1f}%')
for qq in [0.999, 0.9999]:
    thr = np.quantile(np.abs(dq), qq); mm = np.abs(dq) > thr
    print(f'    |差| 前 {(1-qq)*100:.2f}% 的点 (n={mm.sum():,}) 贡献了均值的 '
          f'{dq[mm].sum()/dq.sum()*100 if dq.sum()!=0 else float("nan"):.0f}%')
w = np.quantile(dq, [0.005, 0.995]); dqw = np.clip(dq, w[0], w[1])
print(f'    winsor 0.5% 后 train QLIKE 改善 = {dqw.mean()/q0[tr].mean()*100:+.1f}% '
      f'(原 {(1-q1[tr].mean()/q0[tr].mean())*100:+.1f}%)')

print('\n=== 诊断② val 增量是否集中在"基线退化最厉害"的指数 ===')
rows = []
for jg in np.unique(d['j'].values):
    mt = tr & (d['j'].values == jg); mv = va & (d['j'].values == jg)
    if mv.sum() < 500 or mt.sum() < 2000: continue
    b_tr, b_va = HB.r2(y, base, mt), HB.r2(y, base, mv)
    rows.append(dict(code=codes[jg], base_tr=b_tr, base_va=b_va, decay=b_tr - b_va,
                     d_val=HB.r2(y, full, mv) - HB.r2(y, base, mv)))
r = pd.DataFrame(rows)
r['bucket'] = pd.qcut(r.decay, 4, labels=['退化最小Q1', 'Q2', 'Q3', '退化最大Q4'])
print(f"  {'桶':<12s}{'n':>5s}{'基线train':>11s}{'基线val':>10s}{'退化':>9s}{'val增量':>10s}")
for b, g in r.groupby('bucket', observed=True):
    print(f'  {str(b):<12s}{len(g):>5d}{g.base_tr.mean():>11.4f}{g.base_va.mean():>10.4f}'
          f'{g.decay.mean():>9.4f}{g.d_val.mean():>+10.4f}')
rho = r[['decay', 'd_val']].corr(method='spearman').iloc[0, 1]
print(f'\n  ★corr(基线退化, val增量) = {rho:+.3f} (Spearman, n={len(r)})')
print('   若显著为正 => 增量主要是"基线自己退化腾出的空间"; 若≈0 => 增量与基线退化无关')
