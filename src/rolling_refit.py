"""滚动重训 vs 冻结 (1h目标)

年度 walk-forward: 每年1月1日用截至上年末的全部数据重训 (岭+GatedLinear, 早停val=训练窗最后15%日期),
预测当年; 与冻结版(系数2023-04, 早停val=2023-04~2024-06)在相同评估行上比 R²。
注: holdout 已开封, 2026年重训窗含2025H2数据, 属开封后的事后稳健性实验, 不影响主结论的预注册性。
"""
import os
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_index_ridge import load_all, ridge_fit, r2
from run_paper1_stack import train_gated

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d, Hm = load_all()
seg = d.seg.values
y = np.log(d['fvol'].values)
dates = d.date.values
Xb = np.column_stack([
    np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
Xfull = np.column_stack([Xb, d.nll.values, d.ent.values, Hm]).astype(np.float32)

# 冻结版 (生产口径)
z = np.load(f'{B}/out/paper1_preds.npz')
yh0_f, yh2_f = z['yh0'], z['yh0'] + z['p2']

print(f"{'评估窗':>16s}{'n':>8s}{'冻结M0':>9s}{'冻结M2':>9s}{'重训M0':>9s}{'重训M2':>9s}{'M2差(重训-冻结)':>16s}")
for y0, y1 in [('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-12-31')]:
    ev = (dates >= y0) & (dates < y1)
    trm = dates < y0
    trd = np.sort(np.unique(dates[trm]))
    cut = trd[int(len(trd) * 0.85)]
    tr2 = trm & (dates < cut); va2 = trm & (dates >= cut)
    yh0_r = ridge_fit(Xb, y, tr2)
    torch.manual_seed(0); np.random.seed(0)
    p, ep = train_gated(Xfull, y - yh0_r, tr2, va2)
    yh2_r = yh0_r + p
    r = {k: r2(y, v, ev) for k, v in [('f0', yh0_f), ('f2', yh2_f), ('r0', yh0_r), ('r2', yh2_r)]}
    print(f"{y0[:4]+'年':>16s}{ev.sum():>8,d}{r['f0']:>9.4f}{r['f2']:>9.4f}{r['r0']:>9.4f}{r['r2']:>9.4f}"
          f"{r['r2'] - r['f2']:>+16.4f}")
