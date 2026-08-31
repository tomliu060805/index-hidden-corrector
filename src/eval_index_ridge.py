"""指数版岭回归基线: 512维隐层对指数未来1h波动的增量 (复刻 eval_hidden.py 口径)

B1  = 仅已实现波动 (v12/v48/v240/aret, log) —— 与个股版完全同口径
B1b = B1 + bar独热36 + 指数独热3 (日内节律基线, 更强更诚实)
B2  = B1b + nll + ent (前九轮的标量提取)
B3  = B1b + 512维隐层
B4  = 仅512维隐层
指标: 分段R² / 对B1b的增量 / 残差日IC (δ=yhat_B3−yhat_B1b vs e=y−yhat_B1b 的日内Spearman)
     / 论文一三相关性 (pooled, per-day, cross-day)
切分: train≤2023-04-27, val≤2024-06-07, test≤2025-07-17, holdout>(不打印,锁定)
"""
import os
import glob, numpy as np, pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = 12
CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']


def load_all():
    fs = sorted(glob.glob(f'{BASE}/out/hidden/chunk_*.npz'))
    Hs, Ms = [], []
    for f in fs:
        z = np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
    Hm = np.concatenate(Hs).astype(np.float32); M = np.concatenate(Ms)
    print(f'{len(M):,} 样本 (chunks={len(fs)})')

    df = pd.read_parquet(f'{BASE}/out/index5m.parquet')
    dates = np.array(sorted(df['date'].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    ci = pd.Series(np.arange(len(CODES)), index=CODES)
    g = di.reindex(df['date']).values * 48 + df['bar'].values
    s = ci.reindex(df['code']).values
    NG = len(dates) * 48
    close = np.full((NG, 3), np.nan, np.float32)
    close[g, s] = df['close'].values
    lr = np.full_like(close, np.nan); lr[1:] = np.log(close[1:] / close[:-1])
    lr[np.arange(NG) % 48 == 0] = np.nan
    alr = np.abs(lr)
    fvol = np.full_like(close, np.nan); fret = np.full_like(close, np.nan)
    bar = np.arange(NG) % 48
    for i in range(NG - H):
        if bar[i] > 47 - H: continue
        seg = lr[i + 1:i + 1 + H]
        fvol[i] = np.nanmean(np.abs(seg), 0); fret[i] = np.nansum(seg, 0)
    pv = pd.DataFrame(alr).shift(1)
    v12 = pv.rolling(12, min_periods=6).mean().values
    v48 = pv.rolling(48, min_periods=24).mean().values
    v240 = pv.rolling(240, min_periods=120).mean().values

    t = M[:, 0].astype(np.int64); j = M[:, 1].astype(np.int64)
    d = pd.DataFrame({'t': t, 'j': j, 'nll': M[:, 2], 'ent': M[:, 3],
                      'fvol': fvol[t, j], 'fret': fret[t, j],
                      'v12': v12[t, j], 'v48': v48[t, j], 'v240': v240[t, j],
                      'aret': alr[t, j]})
    d['date'] = dates[t // 48]; d['bar'] = t % 48
    d['seg'] = np.where(d.date <= '2023-04-27', 'train',
                np.where(d.date <= '2024-06-07', 'val',
                np.where(d.date <= '2025-07-17', 'test', 'holdout')))
    ok = d[['fvol', 'v12', 'v48', 'v240', 'aret']].notna().all(1).values & (d.fvol > 0).values
    d = d[ok].reset_index(drop=True); Hm = Hm[ok]
    print('/'.join(f'{k}={int((d.seg == k).sum()):,}' for k in ['train', 'val', 'test', 'holdout']))
    return d, Hm


def ridge_fit(X, y, tr, lam_scale=1.0):
    X = np.column_stack([X, np.ones(len(X))])
    lam = lam_scale * tr.sum() / 1e4
    A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
    b = np.linalg.solve(A, X[tr].T @ y[tr])
    return X @ b


def r2(y, yh, m):
    return 1 - ((y[m] - yh[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


def three_corr(d, y, yh, m):
    """论文一口径: pooled / per-day / cross-day"""
    dd = pd.DataFrame({'date': d['date'].values[m], 'y': y[m], 'yh': yh[m]})
    pooled = np.corrcoef(dd.y, dd.yh)[0, 1]
    per = dd.groupby('date').apply(lambda g: g.y.corr(g.yh) if len(g) > 5 else np.nan).dropna()
    dm = dd.groupby('date').mean()
    return pooled, per.mean(), np.corrcoef(dm.y, dm.yh)[0, 1]


def daily_ic(d, delta, e, m):
    dd = pd.DataFrame({'date': d['date'].values[m], 'x': delta[m], 'e': e[m]})
    ic = dd.groupby('date').apply(lambda g: spearmanr(g.x, g.e)[0] if len(g) > 5 else np.nan).dropna()
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)), len(ic)


def main():
    d, Hm = load_all()
    tr = (d.seg == 'train').values
    SEGS = ['train', 'val', 'test']              # holdout 锁定
    onehot_bar = pd.get_dummies(d['bar']).values.astype(np.float32)
    onehot_idx = pd.get_dummies(d['j']).values.astype(np.float32)
    RVlog = np.column_stack([np.log(np.clip(d[c].values, 1e-8, None))
                             for c in ['v12', 'v48', 'v240', 'aret']])

    for tgt, name in [('fvol', '指数未来1h波动(log)'), ('fret', '指数未来1h收益(bp)')]:
        y = np.log(d[tgt].values) if tgt == 'fvol' else d[tgt].values * 1e4
        print(f'\n=== 目标: {name} ===')
        print(f"{'特征集':>28s}{'维度':>6s}{'train':>9s}{'val':>9s}{'test':>9s}{'val增量':>9s}{'test增量':>9s}")
        sets = [('B1 仅已实现波动', RVlog),
                ('B1b +bar/idx独热', np.column_stack([RVlog, onehot_bar, onehot_idx])),
                ('B2 B1b+NLL+熵', np.column_stack([RVlog, onehot_bar, onehot_idx, d.nll, d.ent])),
                ('B3 B1b+512维隐层', np.column_stack([RVlog, onehot_bar, onehot_idx, Hm])),
                ('B4 仅512维隐层', Hm)]
        yhs, base = {}, None
        for nm, X in sets:
            yh = ridge_fit(X, y, tr); yhs[nm[:3].strip()] = yh
            rr = {k: r2(y, yh, (d.seg == k).values) for k in SEGS}
            if base is None and nm.startswith('B1b'): pass
            if nm.startswith('B1b'): base = rr
            ref = base if base else rr
            print(f'{nm:>28s}{X.shape[1]:>6d}' + ''.join(f'{rr[k]:>9.4f}' for k in SEGS)
                  + (f"{rr['val']-base['val']:>+9.4f}{rr['test']-base['test']:>+9.4f}" if base else ' ' * 18))
        if tgt == 'fvol':
            e = y - yhs['B1b']; delta = yhs['B3'] - yhs['B1b']
            print('残差日IC (B3增量 vs B1b残差):')
            for k in ['val', 'test']:
                m = (d.seg == k).values
                ic, icir, n = daily_ic(d, delta, e, m)
                print(f'  {k}: IC={ic:+.4f}  ICIR={icir:+.2f}  n={n}天')
            print('论文一三相关性 (pooled/per-day/cross-day):')
            for nm in ['B1b', 'B3']:
                for k in ['val', 'test']:
                    m = (d.seg == k).values
                    p, pd_, cd = three_corr(d, y, yhs[nm if nm in yhs else nm], m)
                    print(f'  {nm} {k}: {p:.4f} / {pd_:.4f} / {cd:.4f}')
        # 分指数
        if tgt == 'fvol':
            print('分指数 test R² (B1b → B3):')
            for jj, c in enumerate(CODES):
                m = (d.seg == 'test').values & (d.j == jj).values
                print(f'  {c}: {r2(y, yhs["B1b"], m):.4f} → {r2(y, yhs["B3"], m):.4f}'
                      f'  (Δ{r2(y, yhs["B3"], m)-r2(y, yhs["B1b"], m):+.4f})')


if __name__ == '__main__':
    main()
