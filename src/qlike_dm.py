"""QLIKE 损失 + Diebold-Mariano 检验 + 洗牌安慰剂

QLIKE(方差口径): L = RV/F − log(RV/F) − 1, RV=fvol², F=(c·exp(ŷ))², c=train段乘性偏差校正(各模型自己的)
DM: 日均损失差 d̄ 的 t 检验, Newey-West lag5 (日度聚合已消掉大部分日内重叠)
安慰剂: 把隐层在同bar内跨日随机重排后重训 GatedLinear -> 增量应塌到≈0 (证明增量非结构伪影)
"""
import os
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_index_ridge import load_all, ridge_fit, r2
from run_paper1_stack import train_gated

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGS = ['val', 'test', 'holdout']


def qlike_dm(y, yh0, yh2, dates, seg):
    tr = seg == 'train'
    out = []
    for nm, yh in [('M0', yh0), ('M2', yh2)]:
        c = np.exp((y - yh)[tr].mean())              # 乘性偏差校正 (train)
        F = (c * np.exp(yh)) ** 2
        RV = np.exp(y) ** 2
        ratio = RV / F
        L = ratio - np.log(ratio) - 1
        out.append(L)
    L0, L2 = out
    print(f"{'段':>8s}{'QLIKE M0':>12s}{'QLIKE M2':>12s}{'改善%':>8s}{'DM t':>8s}{'n天':>6s}")
    for s in SEGS:
        m = seg == s
        dd = pd.DataFrame({'d': (L0 - L2)[m], 'date': dates[m]}).groupby('date')['d'].mean()
        x = dd.values
        # Newey-West lag 5
        n = len(x); mu = x.mean()
        e = x - mu
        g0 = (e @ e) / n
        var = g0
        for l in range(1, 6):
            gl = (e[l:] @ e[:-l]) / n
            var += 2 * (1 - l / 6) * gl
        t = mu / np.sqrt(var / n)
        print(f'{s:>8s}{L0[m].mean():>12.4f}{L2[m].mean():>12.4f}'
              f'{(1 - L2[m].mean() / L0[m].mean()) * 100:>7.1f}%{t:>8.2f}{n:>6d}')


def main():
    d, Hm = load_all()
    seg = d.seg.values
    tr, va = seg == 'train', seg == 'val'
    y = np.log(d['fvol'].values)
    Xb = np.column_stack([
        np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
        pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
    Xfull = np.column_stack([Xb, d.nll.values, d.ent.values, Hm]).astype(np.float32)
    yh0 = ridge_fit(Xb, y, tr)
    z = np.load(f'{B}/out/paper1_preds.npz')
    yh2 = z['yh0'] + z['p2']                        # seed0 生产版
    print('===== 1h 目标: QLIKE 与 DM (M0 vs M2) =====')
    qlike_dm(y, yh0, yh2, d.date.values, seg)

    # ---- 安慰剂: 隐层同bar跨日重排 ----
    print('\n===== 安慰剂: 隐层在同bar内跨日随机重排后重训 =====')
    rng = np.random.default_rng(0)
    Hp = Hm.copy()
    bars = d.bar.values; js = d.j.values
    for b in np.unique(bars):
        for j in range(3):
            idx = np.where((bars == b) & (js == j))[0]
            Hp[idx] = Hp[rng.permutation(idx)]
    Xp = np.column_stack([Xb, d.nll.values, d.ent.values, Hp]).astype(np.float32)
    torch.manual_seed(0); np.random.seed(0)
    pp, ep = train_gated(Xp, y - yh0, tr, va)
    for s in SEGS:
        m = seg == s
        print(f'  {s}: Δ={r2(y, yh0 + pp, m) - r2(y, yh0, m):+.4f}  (真实隐层对应 +0.045~0.096)')


if __name__ == '__main__':
    main()
