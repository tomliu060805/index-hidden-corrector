"""判据评估: 秒级 HAR 基线 vs 基线+L2 盘口组。只用 train 段, 内部 70/30 按日期切。"""
import numpy as np, pandas as pd

P = '{OUTDIR}/panel_train.parquet'
BASE = ['rv30','rv120','rv600','bp30','bp120','bp600','rvday']
L2G = {
 '价差 spr':   ['spr30','spr120','spr600','spr_now'],
 '深度 dep':   ['dep30','dep120','dep600','dep_now'],
 '队列不平衡 qi':['qi30','qi120','qi600','qi_now'],
 '盘口斜率 slp':['slp30','slp120','slp600'],
 '成交量 lv':  ['lv30','lv120','lv600'],
 'OFI':        ['ofi30','ofi120','ofi600'],
 '中价-成交价': ['mlp'],
}
L2ALL = [c for v in L2G.values() for c in v]


def design(d, cols, nod=24):
    X = [d[cols].values.astype(np.float64)]
    b = np.clip(((d['tod'].values - 9.5) * 12).astype(int), 0, 71)   # 5min 节律桶
    B = np.zeros((len(d), 72)); B[np.arange(len(d)), b] = 1
    X.append(B[:, 1:])
    pr = pd.get_dummies(d['prod']).values.astype(float)
    X.append(pr[:, 1:])
    X.append(np.ones((len(d), 1)))
    return np.column_stack(X)


def ridge(X, y, tr, lam=10.0):
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-12] = 1
    Z = (X - mu) / sd; Z[:, -1] = 1
    A = Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1]); A[-1, -1] -= lam
    return Z @ np.linalg.solve(A, Z[tr].T @ y[tr])


def r2(y, yh, m):
    return 1 - ((y[m] - yh[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


d = pd.read_parquet(P)
d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=BASE + L2ALL + ['y30', 'y60'])
days = np.sort(d.date.unique()); cut = days[int(len(days) * .7)]
tr, ev = (d.date < cut).values, (d.date >= cut).values
print(f'面板 {len(d):,} 行 / {len(days)} 天 (全部 train 段)')
print(f'  拟合 {tr.sum():,} 行 ({days[0]}~{cut})   评估 {ev.sum():,} 行 ({cut}~{days[-1]})\n')

rng = np.random.default_rng(0)
for H in ['y30', 'y60']:
    y = d[H].values
    Xb = design(d, BASE); Xf = design(d, BASE + L2ALL)
    pb, pf = ridge(Xb, y, tr), ridge(Xf, y, tr)
    r0, r1 = r2(y, pb, ev), r2(y, pf, ev)
    # 安慰剂: L2 特征在同一 (品种, 5min节律桶) 内跨日打乱
    dd = d.copy(); key = dd['prod'].astype(str) + '_' + ((dd['tod'] - 9.5) * 12).astype(int).astype(str)
    sh = dd.groupby(key, sort=False)[L2ALL].transform(lambda s: s.sample(frac=1, random_state=0).values)
    dd[L2ALL] = sh.values
    pp = ridge(design(dd, BASE + L2ALL), y, tr)
    rp = r2(y, pp, ev)
    print(f'=== 目标: 未来 {H[1:]} 秒 RV (log) ===')
    print(f'  基线(秒级HAR+节律+品种)        R² = {r0:.4f}')
    print(f'  基线 + L2 盘口组               R² = {r1:.4f}      ΔR² = {r1-r0:+.4f}')
    print(f'  安慰剂(L2 同桶跨日打乱)        R² = {rp:.4f}      Δ   = {rp-r0:+.4f}')
    print('  逐组拆分(单独加在基线上):')
    for nm, cs in L2G.items():
        pg = ridge(design(d, BASE + cs), y, tr)
        print(f'    {nm:<14s} ΔR² = {r2(y,pg,ev)-r0:+.4f}')
    print()
