"""通用修正器: 无指数身份(去idx独热)训练GL于原三指数 -> 冻结迁移到三个新指数"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import glob, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, f'{_R}/src')
from eval_index_ridge import load_all, ridge_fit, r2
from run_paper1_stack import GatedLinear

B = f'{_R}'
SEGS = ['val', 'test', 'holdout']
H = 12
seg_of = lambda dt: np.where(dt <= '2023-04-27', 'train', np.where(dt <= '2024-06-07', 'val',
                    np.where(dt <= '2025-07-17', 'test', 'holdout')))

d0, H0 = load_all()
seg0 = d0.seg.values; tr0 = seg0 == 'train'; va0 = seg0 == 'val'
y0 = np.log(d0['fvol'].values)
RV0 = np.column_stack([np.log(np.clip(d0[c].values, 1e-8, None)) for c in ['v12','v48','v240','aret']])
bar0 = pd.get_dummies(d0['bar']).values.astype(np.float32)
Xb0_full = np.column_stack([RV0, bar0, pd.get_dummies(d0['j']).values]).astype(np.float32)
yh00 = ridge_fit(Xb0_full, y0, tr0)
resid0 = (y0 - yh00).astype(np.float32)
Xu = np.column_stack([RV0, bar0, d0.nll.values, d0.ent.values, H0]).astype(np.float32)

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(100)
mu, sd = Xu[tr0].mean(0), Xu[tr0].std(0) + 1e-8
Xn = ((Xu - mu) / sd).astype(np.float32)
net = GatedLinear(Xu.shape[1], 32)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
Xt = torch.from_numpy(Xn); yt = torch.from_numpy(resid0)
idx = np.where(tr0)[0]; best, bs, pat = np.inf, None, 0
for ep in range(60):
    net.train(); perm = np.random.permutation(idx)
    for i in range(0, len(perm), 8192):
        b = perm[i:i+8192]; opt.zero_grad()
        ((net(Xt[b]) - yt[b])**2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        v = ((net(Xt[va0]).numpy() - resid0[va0])**2).mean()
    if v < best - 1e-6: best, bs, pat = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
    else:
        pat += 1
        if pat >= 8: break
net.load_state_dict(bs); net.eval()
with torch.no_grad(): pu = net(Xt).numpy()
print('通用GL(无idx) 原数据: ' + ' '.join(
    f"{s}D={r2(y0, yh00+pu, seg0==s)-r2(y0, yh00, seg0==s):+.4f}" for s in SEGS), flush=True)

def build_d(panel, hidden_dir):
    df = pd.read_parquet(panel)
    dates = np.array(sorted(df['date'].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    codes = sorted(df['code'].unique())
    ci = pd.Series(np.arange(len(codes)), index=codes)
    NG = len(dates) * 48
    close = np.full((NG, len(codes)), np.nan, np.float32)
    close[di.reindex(df['date']).values*48 + df['bar'].values, ci.reindex(df['code']).values] = df['close'].values
    lr = np.full_like(close, np.nan); lr[1:] = np.log(close[1:]/close[:-1])
    lr[np.arange(NG) % 48 == 0] = np.nan
    alr = np.abs(lr)
    fvol = np.full_like(close, np.nan)
    bar = np.arange(NG) % 48
    for i in range(NG - H):
        if bar[i] > 47 - H: continue
        fvol[i] = np.nanmean(alr[i+1:i+1+H], 0)
    pv = pd.DataFrame(alr).shift(1)
    v12 = pv.rolling(12, min_periods=6).mean().values
    v48 = pv.rolling(48, min_periods=24).mean().values
    v240 = pv.rolling(240, min_periods=120).mean().values
    Hs, Ms = [], []
    for f in sorted(glob.glob(f'{hidden_dir}/chunk_*.npz')):
        z = np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
    Hm = np.concatenate(Hs).astype(np.float32); M = np.concatenate(Ms)
    t = M[:,0].astype(np.int64); j = M[:,1].astype(np.int64)
    d = pd.DataFrame({'t': t, 'j': j, 'nll': M[:,2], 'ent': M[:,3],
                      'fvol': fvol[t,j], 'v12': v12[t,j], 'v48': v48[t,j],
                      'v240': v240[t,j], 'aret': alr[t,j]})
    d['date'] = dates[t//48]; d['bar'] = t % 48
    ok = d[['fvol','v12','v48','v240','aret']].notna().all(1).values & (d.fvol > 0).values
    return d[ok].reset_index(drop=True), Hm[ok], codes

dt, Ht, codes = build_d(f'{B}/out/index5m_transfer.parquet', f'{B}/out/hidden_transfer')
segt = seg_of(dt.date.values); trt = segt == 'train'
yt_ = np.log(dt['fvol'].values)
RVt = np.column_stack([np.log(np.clip(dt[c].values, 1e-8, None)) for c in ['v12','v48','v240','aret']])
bart = np.zeros((len(dt), 35), np.float32)
bb = dt.bar.values
bart[np.arange(len(dt))[bb >= 1], bb[bb >= 1] - 1] = 1
Xbt = np.column_stack([RVt, bart]).astype(np.float32)
yh0t = ridge_fit(Xbt, yt_, trt)
Xut = np.column_stack([RVt, bart, dt.nll.values, dt.ent.values, Ht]).astype(np.float32)
Xnt = ((Xut - mu) / sd).astype(np.float32)
with torch.no_grad(): pt = net(torch.from_numpy(Xnt)).numpy()
NAMES = {'000016.XSHG': '上证50', '000688.XSHG': '科创50', '399006.XSHE': '创业板指'}
print('通用GL 零样本迁移 (冻结, 未见过的指数):')
for jj, c in enumerate(codes):
    for s in SEGS:
        m = (segt == s) & (dt.j == jj).values
        if m.sum() < 500: continue
        print(f'  {NAMES[c]:>6s} {s}: D={r2(yt_, yh0t+pt, m)-r2(yt_, yh0t, m):+.4f}')
for s in SEGS:
    m = segt == s
    print(f'  {"汇总":>6s} {s}: D={r2(yt_, yh0t+pt, m)-r2(yt_, yh0t, m):+.4f}')
