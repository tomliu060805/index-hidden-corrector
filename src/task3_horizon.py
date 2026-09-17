"""任务3: 期限结构外推 —— ①2h(24根) 日内目标  ②次日RV (EOD隐层 vs Log-HAR(Q))

②是主戏: 决策点=每日15:00收盘 (bar47隐层, out/hidden_eod), 目标=次日全天RV(log)
   基线 Log-HAR: logRV_d / logRV_w(5d) / logRV_m(22d) + 指数独热; HARQ 加 RQ 修正项
   模型: H2=+512隐层(岭) H3=+GatedLinear
①用既有日内隐层, 决策bar≤23, 目标=未来24根|r|均值, 复用M0/GatedLinear结构
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import glob, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, f'{_R}/src')
from eval_index_ridge import ridge_fit, r2
from run_paper1_stack import train_gated

B = f'{_R}'
CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']
SEGS = ['train', 'val', 'test', 'holdout']
np.random.seed(0); torch.manual_seed(0)

def seg_of(dates):
    return np.where(dates <= '2023-04-27', 'train', np.where(dates <= '2024-06-07', 'val',
           np.where(dates <= '2025-07-17', 'test', 'holdout')))

def load_close():
    df = pd.read_parquet(f'{B}/out/index5m.parquet')
    dates = np.array(sorted(df['date'].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    ci = pd.Series(np.arange(3), index=CODES)
    NG = len(dates) * 48
    close = np.full((NG, 3), np.nan, np.float32)
    close[di.reindex(df['date']).values * 48 + df['bar'].values,
          ci.reindex(df['code']).values] = df['close'].values
    lr = np.full_like(close, np.nan); lr[1:] = np.log(close[1:] / close[:-1])
    lr[np.arange(NG) % 48 == 0] = np.nan
    return lr, dates

def report(name, y, yh, seg, base=None):
    rr = {s: r2(y, yh, seg == s) for s in SEGS}
    inc = '' if base is None else '  Δ ' + ' '.join(f"{s}={rr[s]-base[s]:+.4f}" for s in ['val','test','holdout'])
    print(f"{name:>26s}: " + ' '.join(f"{s}={rr[s]:.4f}" for s in SEGS) + inc, flush=True)
    return rr

# ========== ② 次日 RV ==========
print('=' * 100); print('### ② 次日全天RV (决策点=15:00收盘, EOD隐层)'); print('=' * 100)
lr, dates = load_close()
Nd = len(dates)
alr2 = (lr ** 2).reshape(Nd, 48, 3)
RV = np.sqrt(np.nansum(alr2, 1))                    # 日RV
RV[np.isnan(lr[:, :].reshape(Nd, 48, 3)).all(1)] = np.nan
RQ = np.nansum((lr ** 4).reshape(Nd, 48, 3), 1) * 48 / 3

fs = sorted(glob.glob(f'{B}/out/hidden_eod/chunk_*.npz'))
Hs, Ms = [], []
for f in fs:
    zz = np.load(f); Hs.append(zz['h']); Ms.append(zz['meta'])
Hm = np.concatenate(Hs).astype(np.float32); M = np.concatenate(Ms)
t = M[:, 0].astype(np.int64); j = M[:, 1].astype(np.int64)
d_i = t // 48
rows = pd.DataFrame({'di': d_i, 'j': j, 'nll': M[:, 2], 'ent': M[:, 3]})
rows['date'] = dates[rows.di]
rv = pd.DataFrame(RV, columns=range(3))
lrv = np.log(np.clip(RV, 1e-8, None))
def roll_feat(win):
    return pd.DataFrame(lrv).rolling(win, min_periods=max(2, win // 2)).mean().values
lrv_w, lrv_m = roll_feat(5), roll_feat(22)
rows['y'] = lrv[np.clip(rows.di + 1, 0, Nd - 1), rows.j]        # 次日
rows['rv_d'] = lrv[rows.di, rows.j]
rows['rv_w'] = lrv_w[rows.di, rows.j]
rows['rv_m'] = lrv_m[rows.di, rows.j]
rows['rq'] = np.sqrt(np.clip(RQ[rows.di, rows.j], 0, None)) * np.exp(rows.rv_d)  # HARQ 项
ok = (rows.di + 1 < Nd) & rows[['y', 'rv_d', 'rv_w', 'rv_m']].notna().all(1) \
     & np.isfinite(rows.y) & np.isfinite(lrv[np.clip(rows.di + 1, 0, Nd - 1), rows.j])
rows = rows[ok.values].reset_index(drop=True); Hm2 = Hm[ok.values]
seg = seg_of(rows.date.values)
print(f'{len(rows):,} 天×指数样本, ' + '/'.join(f'{s}={int((seg==s).sum()):,}' for s in SEGS))

y = rows.y.values
oi = pd.get_dummies(rows.j).values.astype(np.float32)
Xhar = np.column_stack([rows.rv_d, rows.rv_w, rows.rv_m, oi]).astype(np.float32)
Xharq = np.column_stack([Xhar, rows.rq]).astype(np.float32)
tr = seg == 'train'; va = seg == 'val'

h0 = report('H0 Log-HAR', y, ridge_fit(Xhar, y, tr), seg)
report('H0q Log-HARQ', y, ridge_fit(Xharq, y, tr), seg, h0)
report('H1 +NLL+熵', y, ridge_fit(np.column_stack([Xhar, rows.nll, rows.ent]), y, tr), seg, h0)
yh2 = ridge_fit(np.column_stack([Xhar, Hm2]), y, tr)
report('H2 +512隐层(岭)', y, yh2, seg, h0)
yh_har = ridge_fit(Xhar, y, tr)
p3, ep = train_gated(np.column_stack([Xhar, rows.nll.values, rows.ent.values, Hm2]).astype(np.float32),
                     y - yh_har, tr, va)
h3 = report(f'H3 +GatedLinear({ep}ep)', y, yh_har + p3, seg, h0)
print('分指数 H0→H3 (test/holdout Δ):')
for jj, c in enumerate(CODES):
    for s in ['test', 'holdout']:
        m = (seg == s) & (rows.j == jj).values
        print(f'  {c} {s}: {r2(y, yh_har, m):.4f} → {r2(y, yh_har + p3, m):.4f} (Δ{r2(y, yh_har+p3, m)-r2(y, yh_har, m):+.4f})')

# ========== ① 2h 日内目标 ==========
print(); print('=' * 100); print('### ① 未来2h波动 (决策bar≤23, 既有日内隐层)'); print('=' * 100)
z = np.load(f'{B}/out/paper1_preds.npz')
fs = sorted(glob.glob(f'{B}/out/hidden/chunk_*.npz'))
Hs, Ms = [], []
for f in fs:
    zz = np.load(f); Hs.append(zz['h']); Ms.append(zz['meta'])
Hi = np.concatenate(Hs).astype(np.float32); Mi = np.concatenate(Ms)
ti = Mi[:, 0].astype(np.int64); ji = Mi[:, 1].astype(np.int64)
H24 = 24
bar = ti % 48
keep = bar <= 47 - H24
alr = np.abs(lr)
fvol24 = np.full(lr.shape, np.nan)
NG = lr.shape[0]
for i in range(NG - H24):
    if i % 48 > 47 - H24: continue
    fvol24[i] = np.nanmean(alr[i + 1:i + 1 + H24], 0)
pv = pd.DataFrame(alr).shift(1)
v12 = pv.rolling(12, min_periods=6).mean().values
v48 = pv.rolling(48, min_periods=24).mean().values
v240 = pv.rolling(240, min_periods=120).mean().values
d2 = pd.DataFrame({'t': ti[keep], 'j': ji[keep], 'nll': Mi[keep, 2], 'ent': Mi[keep, 3]})
d2['y'] = np.log(np.clip(fvol24[d2.t, d2.j], 1e-8, None))
for nm, arrf in [('v12', v12), ('v48', v48), ('v240', v240)]:
    d2[nm] = arrf[d2.t, d2.j]
d2['aret'] = alr[d2.t, d2.j]
d2['date'] = dates[d2.t // 48]; d2['bar'] = d2.t % 48
okm = d2[['y', 'v12', 'v48', 'v240', 'aret']].notna().all(1).values & np.isfinite(d2.y.values)
d2 = d2[okm].reset_index(drop=True); Hk = Hi[keep][okm]
seg2 = seg_of(d2.date.values)
print(f'{len(d2):,} 样本')
y2 = d2.y.values
RVlog = np.column_stack([np.log(np.clip(d2[c].values, 1e-8, None)) for c in ['v12','v48','v240','aret']])
Xb = np.column_stack([RVlog, pd.get_dummies(d2.bar).values, pd.get_dummies(d2.j).values]).astype(np.float32)
tr2 = seg2 == 'train'; va2 = seg2 == 'val'
m0 = report('M0 基线', y2, ridge_fit(Xb, y2, tr2), seg2)
yh0 = ridge_fit(Xb, y2, tr2)
pg, ep2 = train_gated(np.column_stack([Xb, d2.nll.values, d2.ent.values, Hk]).astype(np.float32),
                      y2 - yh0, tr2, va2)
report(f'M2 +GatedLinear({ep2}ep)', y2, yh0 + pg, seg2, m0)
