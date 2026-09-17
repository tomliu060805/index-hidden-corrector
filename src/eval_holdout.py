"""holdout 段 (>2025-07-17, 从未参与训练/早停/选择) 一次性开封
预注册: 最终模型 = M2 (GatedLinear, val最优); 其余模型只做参照
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr

B = f'{_R}'
z = np.load(f'{B}/out/paper1_preds.npz')
y, seg, date, j = z['y'], z['seg'], z['date'], z['j']
models = {'M0 基线': z['yh0'], 'M1 +LGBM(含隐层)': z['yh0'] + z['p1'],
          'M4 +LGBM(无隐层)': z['yh0'] + z['p4'], 'M2 +GatedLinear': z['yh0'] + z['p2'],
          'M3 M2+LGBM': z['yh0'] + z['p2'] + z['p3']}
m = seg == 'holdout'
dh = np.sort(pd.unique(date[m]))
print(f'holdout: {m.sum():,} 样本, {dh[0]} ~ {dh[-1]}, {len(dh)} 天')

def r2(yh): return 1 - ((y[m]-yh[m])**2).sum()/((y[m]-y[m].mean())**2).sum()
base = r2(z['yh0'])
resid0 = y - z['yh0']
for nm, yh in models.items():
    delta = yh - z['yh0']
    dd = pd.DataFrame({'date': date[m], 'x': delta[m], 'e': resid0[m]})
    ic = dd.groupby('date').apply(lambda g: spearmanr(g.x, g.e)[0] if len(g) > 5 else np.nan,
                                  include_groups=False).dropna()
    icir = ic.mean()/ic.std()*np.sqrt(len(ic)) if len(ic) > 1 else np.nan
    dd2 = pd.DataFrame({'date': date[m], 'y': y[m], 'yh': yh[m]})
    per = dd2.groupby('date').apply(lambda g: g.y.corr(g.yh), include_groups=False).dropna().mean()
    print(f'{nm:>18s}: R²={r2(yh):.4f} Δ={r2(yh)-base:+.4f}  残差日IC={ic.mean():+.4f} ICIR={icir:+.2f}  per-day corr={per:.4f}')

print('\n分指数 holdout R² (M0 → M2):')
for jj, c in enumerate(['000300', '000905', '000852']):
    mm = m & (j == jj)
    r = lambda yh: 1 - ((y[mm]-yh[mm])**2).sum()/((y[mm]-y[mm].mean())**2).sum()
    print(f'  {c}: {r(z["yh0"]):.4f} → {r(z["yh0"]+z["p2"]):.4f} (Δ{r(z["yh0"]+z["p2"])-r(z["yh0"]):+.4f})')
