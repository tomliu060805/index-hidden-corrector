"""★L2 线的 test 段开封(确认性)。预注册判据写在看数之前, 见下。

背景: 本线已按 PREREG_probe.md 在 train 段判负(纯订单簿净增量 +0.0032, ≤+0.005 的判负线)。
开封目的是**确认**, 不是翻案 —— 已判负的线不存在"在 test 上调参"的风险。

判据(写在跑之前):
  test 段【订单簿在成交流之后的净增量】≤ +0.005  ⇒ 确认判负, 线永久关闭
  > +0.005 且 CW t > 3                          ⇒ 与 train 段矛盾, 需重开研究(不得直接翻案)
同时必报(不作判据): 功效 —— test 天数与点数、能检出多大效应。

拟合只用 train(2020-01-02~2023-04-27), test = 2024-06-08~2025-07-17。val 段一并报告。
holdout(2025-07-18 之后)不碰。
"""
import os, sys, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W = (30, 120, 600)
BASE = ['rv30','rv120','rv600','bp30','bp120','bp600','rvday']
FLOW = [f'{k}{w}' for k in ('lv','trd','sz','upd') for w in W]
BOOK = ([f'{k}{w}' for k in ('spr','dep','qi','slp','sprsd','qisd','depsd','tofi','atofi') for w in W]
        + ['spr_now','dep_now','qi_now','mlp'])
Ys = ['y30','y60','y300','y900','y1800']
LAB = dict(zip(Ys, ['30秒','1分钟','5分钟','15分钟','30分钟']))
TR_END, VAL_END, TEST_END = '20230427', '20240607', '20250717'


def design(d, cols):
    X = [d[cols].values.astype(np.float64)]
    b = np.clip(((d['tod'].values - 9.5) * 12).astype(int), 0, 71)
    B = np.zeros((len(d), 72)); B[np.arange(len(d)), b] = 1; X.append(B[:, 1:])
    X.append(pd.get_dummies(d['prod']).values.astype(float)[:, 1:])
    X.append(np.ones((len(d), 1)))
    return np.column_stack(X)


def ridge(X, y, tr, lam=10.):
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-12] = 1
    Z = (X - mu) / sd; Z[:, -1] = 1
    A = Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1]); A[-1, -1] -= lam
    return Z @ np.linalg.solve(A, Z[tr].T @ y[tr])


def r2(y, yh, m):
    return 1 - ((y[m] - yh[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


def nw_var(x, lag=10):
    n = len(x); e = x - x.mean(); v = (e @ e) / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * ((e[l:] @ e[:-l]) / n)
    return max(v, 1e-30)


def cw(y, f1, f2, dates, lag=10):
    a = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    dd = pd.DataFrame({'d': a, 'date': dates}).groupby('date')['d'].mean().values
    return dd.mean() / np.sqrt(nw_var(dd, lag) / len(dd)), len(dd)


def main():
    print('=' * 88)
    print('★ L2 线 test 段开封(确认性) —— 判据: 订单簿净增量 ≤ +0.005 即确认判负')
    print('=' * 88, flush=True)
    d = pd.read_parquet('out/panel_full.parquet').replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=BASE + FLOW + BOOK + Ys)
    seg = np.where(d.date <= TR_END, 'train',
          np.where(d.date <= VAL_END, 'val',
          np.where(d.date <= TEST_END, 'TEST', 'SEALED')))
    d = d.assign(seg=seg)
    M = {s: (d.seg == s).values for s in ['train', 'val', 'TEST']}
    for s in ['train', 'val', 'TEST']:
        m = M[s]
        print(f'  {s:>6s}: {m.sum():>9,d} 点  {pd.Series(d.date[m]).nunique():>4d} 天  '
              f'{d.date[m].min()}~{d.date[m].max()}')
    print(f'  封存(holdout): {(d.seg=="SEALED").sum():,} 点, 不参与\n', flush=True)
    tr = M['train']
    XF, XA = design(d, BASE + FLOW), design(d, BASE + FLOW + BOOK)
    dts = d['date'].values
    print(f"{'视界':>8s}{'段':>7s}{'基线R²':>10s}{'+订单簿':>10s}{'★净增量':>10s}{'CW t':>8s}{'n天':>6s}")
    print('-' * 88)
    rows = []
    for H in Ys:
        y = d[H].values
        pF, pA = ridge(XF, y, tr), ridge(XA, y, tr)
        for s in ['val', 'TEST']:
            m = M[s]
            rF, rA = r2(y, pF, m), r2(y, pA, m)
            t, nd = cw(y[m], pF[m], pA[m], dts[m])
            print(f'{LAB[H]:>8s}{s:>7s}{rF:>10.4f}{rA:>10.4f}{rA-rF:>+10.4f}{t:>8.2f}{nd:>6d}')
            rows.append((H, s, rA - rF, t))
        print('-' * 88)
    te = {h: (v, t) for h, s, v, t in rows if s == 'TEST'}
    mx = max(v for v, _ in te.values())
    print(f'\n★判读: test 段订单簿净增量最大值 = {mx:+.4f} (判负线 +0.005)')
    print('  ' + ('=> 确认判负, 线永久关闭' if mx <= 0.005 else '=> 与 train 矛盾, 需重开研究'))
    print(f'\n功效参照: test {M["TEST"].sum():,} 点 / {pd.Series(d.date[M["TEST"]]).nunique()} 天; '
          f'val 上同一量级的效应 CW t 在 {min(t for h,s,v,t in rows if s=="val"):.1f}~'
          f'{max(t for h,s,v,t in rows if s=="val"):.1f}')
    print('★ holdout 未读。')


if __name__ == '__main__':
    main()
