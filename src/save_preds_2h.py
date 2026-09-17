"""重训 2h 目标 (bar≤23) 的 M0+GatedLinear 并保存预测 -> out/preds_2h.npz (供期货覆盖层用)"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import glob, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, f'{_R}/src')
from eval_index_ridge import ridge_fit, r2
from run_paper1_stack import train_gated
from task3_horizon import load_close, seg_of

B = f'{_R}'
np.random.seed(0); torch.manual_seed(0)
H24 = 24

lr, dates = load_close()
NG = lr.shape[0]
alr = np.abs(lr)
fvol24 = np.full(lr.shape, np.nan)
for i in range(NG - H24):
    if i % 48 > 47 - H24: continue
    fvol24[i] = np.nanmean(alr[i + 1:i + 1 + H24], 0)
pv = pd.DataFrame(alr).shift(1)
v12 = pv.rolling(12, min_periods=6).mean().values
v48 = pv.rolling(48, min_periods=24).mean().values
v240 = pv.rolling(240, min_periods=120).mean().values

fs = sorted(glob.glob(f'{B}/out/hidden/chunk_*.npz'))
Hs, Ms = [], []
for f in fs:
    z = np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
Hi = np.concatenate(Hs).astype(np.float32); Mi = np.concatenate(Ms)
ti = Mi[:, 0].astype(np.int64); ji = Mi[:, 1].astype(np.int64)
keep = (ti % 48) <= 47 - H24
d = pd.DataFrame({'t': ti[keep], 'j': ji[keep], 'nll': Mi[keep, 2], 'ent': Mi[keep, 3]})
d['y'] = np.log(np.clip(fvol24[d.t, d.j], 1e-8, None))
for nm, a in [('v12', v12), ('v48', v48), ('v240', v240)]:
    d[nm] = a[d.t, d.j]
d['aret'] = alr[d.t, d.j]
d['date'] = dates[d.t // 48]; d['bar'] = d.t % 48
ok = d[['y', 'v12', 'v48', 'v240', 'aret']].notna().all(1).values & np.isfinite(d.y.values)
d = d[ok].reset_index(drop=True); Hk = Hi[keep][ok]
seg = seg_of(d.date.values)
y = d.y.values
RVlog = np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']])
Xb = np.column_stack([RVlog, pd.get_dummies(d.bar).values, pd.get_dummies(d.j).values]).astype(np.float32)
tr = seg == 'train'; va = seg == 'val'
yh0 = ridge_fit(Xb, y, tr)
pg, ep = train_gated(np.column_stack([Xb, d.nll.values, d.ent.values, Hk]).astype(np.float32),
                     y - yh0, tr, va)
for s in ['val', 'test', 'holdout']:
    m = seg == s
    print(f'{s}: M0 {r2(y, yh0, m):.4f} -> M2 {r2(y, yh0 + pg, m):.4f}')
np.savez(f'{B}/out/preds_2h.npz', yh0=yh0, p2=pg, y=y, seg=seg.astype('U8'),
         date=d.date.values.astype('U10'), j=d.j.values, bar=d.bar.values)
print('saved preds_2h.npz')
