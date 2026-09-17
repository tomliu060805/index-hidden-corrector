"""正交性对拍: M2隐层增量 (p2) vs 广度熵/广度NLL/clv/uw (000905, 2020+, ti重叠点)

问题: p2 的波动预报增量有多少被已有广度信号解释?
口径: log空间, 系数只在train拟合冻结; F3−F1 = p2在广度之后的增量, F3−F2 = 广度在p2之后的增量
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import numpy as np, pandas as pd
from scipy.stats import spearmanr

B = f'{_R}'
z = np.load(f'{B}/out/paper1_preds.npz')
d = pd.DataFrame({'date': z['date'], 'bar': z['bar'], 'j': z['j'], 'y': z['y'],
                  'yh0': z['yh0'], 'p2': z['p2'], 'seg': z['seg']})
d = d[d.j == 1].copy()                      # 000905
d['ti'] = d['bar'] + 1

# 外部面板(本仓之外的广度信号), 路径经 IHC_BREADTH_PANEL 指定
cb = pd.read_parquet(CFG.BREADTH_PANEL)
BR = ['cs_ent_tod', 'cs_nll_tod', 'cs_clv', 'cs_uw']
m = d.merge(cb[['date', 'ti'] + BR], on=['date', 'ti'], how='inner').dropna(subset=BR)
print(f'join后 {len(m):,} 行, {m.date.nunique()} 天, ' +
      '/'.join(f'{k}={int((m.seg==k).sum()):,}' for k in ['train','val','test','holdout']))

tr = (m.seg == 'train').values
e = m.y.values - m.yh0.values               # M0残差

print('\n相关性 corr(p2增量, 广度信号) 分段:')
for s in ['val', 'test', 'holdout']:
    mm = (m.seg == s).values
    cs = {b: np.corrcoef(m.p2.values[mm], m[b].values[mm])[0, 1] for b in BR}
    print(f'  {s}: ' + '  '.join(f'{b}={v:+.3f}' for b, v in cs.items()))

def ridge(X, y, tr, lam=1e-4):
    X = np.column_stack([X, np.ones(len(X))])
    A = X[tr].T @ X[tr] + lam * len(X[tr]) * np.eye(X.shape[1])
    return X @ np.linalg.solve(A, X[tr].T @ y[tr])

y = m.y.values
Xb = m[BR].values.astype(float)
sets = {'F0 仅M0': m[['yh0']].values,
        'F1 M0+广度4信号': np.column_stack([m.yh0, Xb]),
        'F2 M0+p2': m[['yh0', 'p2']].values,
        'F3 M0+广度+p2': np.column_stack([m.yh0, Xb, m.p2])}
R = {}
def r2(yh, mm): return 1 - ((y[mm]-yh[mm])**2).sum()/((y[mm]-y[mm].mean())**2).sum()
print('\nlog R² (系数冻结train):')
print(f"{'':>16s}" + ''.join(f'{s:>10s}' for s in ['val','test','holdout']))
for nm, X in sets.items():
    yh = ridge(X, y, tr)
    R[nm[:2]] = {s: r2(yh, (m.seg == s).values) for s in ['val','test','holdout']}
    print(f'{nm:>16s}' + ''.join(f"{R[nm[:2]][s]:>10.4f}" for s in ['val','test','holdout']))
print('p2 在广度之后的增量 (F3−F1): ' + '  '.join(f"{s}={R['F3'][s]-R['F1'][s]:+.4f}" for s in ['val','test','holdout']))
print('广度在 p2 之后的增量 (F3−F2): ' + '  '.join(f"{s}={R['F3'][s]-R['F2'][s]:+.4f}" for s in ['val','test','holdout']))

# 控广度后的残差日IC
beta_e = None
Xbr = np.column_stack([Xb, np.ones(len(m))])
be = np.linalg.lstsq(Xbr[tr], e[tr], rcond=None)[0]
e2 = e - Xbr @ be
print('\np2 vs 控广度后残差 e2 的日IC:')
for s in ['val', 'test', 'holdout']:
    mm = (m.seg == s).values
    dd = pd.DataFrame({'date': m.date.values[mm], 'x': m.p2.values[mm], 'e': e2[mm]})
    ic = dd.groupby('date').apply(lambda g: spearmanr(g.x, g.e)[0] if len(g) > 4 else np.nan,
                                  include_groups=False).dropna()
    print(f'  {s}: IC={ic.mean():+.4f} ICIR={ic.mean()/ic.std()*np.sqrt(len(ic)):+.2f} (n={len(ic)})')
