"""零样本迁移评估: 冻结的生产 GatedLinear 应用于从未见过的指数 (上证50/创业板指/科创50)
协议: 本地拟合 M0 基线(新指数train段) + 冻结GL修正(idx独热=0, 用原mu/sd标准化)
对照: 本地训练GL(skyline) | 附: LOIO(300+1000训练→500评估)
"""
import os
import glob, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_index_ridge import ridge_fit, r2
from run_paper1_stack import GatedLinear, train_gated

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODES_T = ['000016.XSHG', '399006.XSHE', '000688.XSHG']
NAMES = ['上证50', '创业板指', '科创50']
SEGS = ['val', 'test', 'holdout']
H = 12
seg_of = lambda dt: np.where(dt <= '2023-04-27', 'train', np.where(dt <= '2024-06-07', 'val',
                    np.where(dt <= '2025-07-17', 'test', 'holdout')))


def build_d(panel, hidden_dir, codes):
    df = pd.read_parquet(panel)
    dates = np.array(sorted(df['date'].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    ci = pd.Series(np.arange(len(codes)), index=codes)
    NG = len(dates) * 48
    close = np.full((NG, len(codes)), np.nan, np.float32)
    close[di.reindex(df['date']).values * 48 + df['bar'].values,
          ci.reindex(df['code']).values] = df['close'].values
    lr = np.full_like(close, np.nan); lr[1:] = np.log(close[1:] / close[:-1])
    lr[np.arange(NG) % 48 == 0] = np.nan
    alr = np.abs(lr)
    fvol = np.full_like(close, np.nan)
    bar = np.arange(NG) % 48
    for i in range(NG - H):
        if bar[i] > 47 - H: continue
        fvol[i] = np.nanmean(alr[i + 1:i + 1 + H], 0)
    pv = pd.DataFrame(alr).shift(1)
    v12 = pv.rolling(12, min_periods=6).mean().values
    v48 = pv.rolling(48, min_periods=24).mean().values
    v240 = pv.rolling(240, min_periods=120).mean().values
    fs = sorted(glob.glob(f'{hidden_dir}/chunk_*.npz'))
    Hs, Ms = [], []
    for f in fs:
        z = np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
    Hm = np.concatenate(Hs).astype(np.float32); M = np.concatenate(Ms)
    t = M[:, 0].astype(np.int64); j = M[:, 1].astype(np.int64)
    d = pd.DataFrame({'t': t, 'j': j, 'nll': M[:, 2], 'ent': M[:, 3],
                      'fvol': fvol[t, j], 'v12': v12[t, j], 'v48': v48[t, j],
                      'v240': v240[t, j], 'aret': alr[t, j]})
    d['date'] = dates[t // 48]; d['bar'] = t % 48
    ok = d[['fvol', 'v12', 'v48', 'v240', 'aret']].notna().all(1).values & (d.fvol > 0).values
    return d[ok].reset_index(drop=True), Hm[ok]


d, Hm = build_d(f'{B}/out/index5m_transfer.parquet', f'{B}/out/hidden_transfer', CODES_T)
seg = seg_of(d.date.values); tr = seg == 'train'; va = seg == 'val'
y = np.log(d['fvol'].values)
bar_oh = np.zeros((len(d), 35), np.float32)          # 生产为bar1..35(bar0被aret过滤)
bb = d.bar.values
bar_oh[np.arange(len(d))[bb >= 1], bb[bb >= 1] - 1] = 1
Xb = np.column_stack([
    np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    bar_oh, np.zeros((len(d), 3), np.float32)]).astype(np.float32)   # idx独热=0
X = np.column_stack([Xb, d.nll.values, d.ent.values, Hm]).astype(np.float32)
print(f'{len(d):,} 迁移样本, ' + '/'.join(f'{s}={(seg==s).sum():,}' for s in ['train'] + SEGS))

yh0 = ridge_fit(Xb, y, tr)                       # 本地基线
resid0 = y - yh0
ck = torch.load(f'{B}/out/m2_model.pt', weights_only=False)
net = GatedLinear(ck['dim'], 32); net.load_state_dict(ck['state']); net.eval()
Xn = ((X - ck['mu']) / ck['sd']).astype(np.float32)
with torch.no_grad():
    p_frozen = net(torch.from_numpy(Xn)).numpy()
torch.manual_seed(0); np.random.seed(0)
p_local, ep = train_gated(X, resid0, tr, va)

print('\n===== 零样本迁移 (冻结生产GL, 未见过这三个指数) =====')
print(f"{'指数':>8s}{'段':>9s}{'本地M0':>9s}{'+冻结GL':>9s}{'Δ冻结':>9s}{'+本地GL':>9s}{'Δ本地':>9s}")
for jj, nm in enumerate(NAMES + ['汇总']):
    for s in SEGS:
        m = (seg == s) & ((d.j == jj).values if jj < 3 else np.ones(len(d), bool))
        if m.sum() < 500: continue
        r0 = r2(y, yh0, m); rf = r2(y, yh0 + p_frozen, m); rl = r2(y, yh0 + p_local, m)
        print(f'{nm:>8s}{s:>9s}{r0:>9.4f}{rf:>9.4f}{rf-r0:>+9.4f}{rl:>9.4f}{rl-r0:>+9.4f}')

# ---- LOIO: 300+1000 训练 -> 500 ----
print('\n===== LOIO: GL只用300+1000训练, 迁移到500 =====')
from eval_index_ridge import load_all
d0, H0 = load_all()
seg0 = d0.seg.values; y0 = np.log(d0['fvol'].values)
Xb0 = np.column_stack([
    np.column_stack([np.log(np.clip(d0[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    pd.get_dummies(d0['bar']).values, pd.get_dummies(d0['j']).values]).astype(np.float32)
X0 = np.column_stack([Xb0, d0.nll.values, d0.ent.values, H0]).astype(np.float32)
yh00 = ridge_fit(Xb0, y0, seg0 == 'train')
r00 = y0 - yh00
not5 = (d0.j != 1).values
torch.manual_seed(0); np.random.seed(0)
p_loio, _ = train_gated(X0, r00, (seg0 == 'train') & not5, (seg0 == 'val') & not5)
z = np.load(f'{B}/out/paper1_preds.npz')
for s in SEGS:
    m = (seg0 == s) & (d0.j == 1).values
    dl = r2(y0, yh00 + p_loio, m) - r2(y0, yh00, m)
    dp = r2(y0, z['yh0'] + z['p2'], m) - r2(y0, z['yh0'], m)
    print(f'  500 {s}: LOIO Δ={dl:+.4f}  vs 生产(含500训练) Δ={dp:+.4f}')
