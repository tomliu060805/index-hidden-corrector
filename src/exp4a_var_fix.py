"""实验④A 修复版: 日内VaR分位回归 — y标准化+800epoch+train截距校准
对照: 无条件VaR(train段按指数×bar的分位, 最朴素) / 线性基线Xb / +隐层
"""
import os
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_index_ridge import load_all

torch.set_num_threads(60)
SEGS = ['val', 'test', 'holdout']
d, Hm = load_all()
seg = d.seg.values; tr = seg == 'train'
fret = (d.fret.values * 1e4).astype(np.float32)
ys = fret / fret[tr].std()
Xb = np.column_stack([
    np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
Xh = np.column_stack([Xb, d.nll.values, d.ent.values, Hm]).astype(np.float32)

def std_fit(X):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    return ((X - mu) / sd).astype(np.float32)
Xb_n, Xh_n = std_fit(Xb), std_fit(Xh)

def qreg(Xn, q, epochs=800, lr=3e-2):
    torch.manual_seed(0)
    Xt = torch.from_numpy(Xn[tr]); yt = torch.from_numpy(ys[tr])
    w = torch.zeros(Xn.shape[1], requires_grad=True)
    b = torch.full((1,), float(np.quantile(ys[tr], q)), requires_grad=True)   # 从无条件分位起步
    opt = torch.optim.Adam([w, b], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        e = yt - (Xt @ w + b)
        torch.maximum(q * e, (q - 1) * e).mean().backward()
        opt.step()
    with torch.no_grad():
        p = (torch.from_numpy(Xn) @ w + b).numpy()
    p = p + np.quantile(ys[tr] - p[tr], q)          # train截距校准
    return p * fret[tr].std()

# 无条件对照: train段 (指数,bar) 分位
def uncond(q):
    key = pd.Series(list(zip(d.j.values, d.bar.values)))
    qs = pd.DataFrame({'k': key[tr].values, 'y': fret[tr]}).groupby('k')['y'].quantile(q)
    return key.map(qs).values.astype(np.float32)

for q in [0.05, 0.01]:
    preds = {'无条件(指数×bar)': uncond(q), '线性基线': qreg(Xb_n, q), '+隐层': qreg(Xh_n, q)}
    print(f'-- q={q} --')
    print(f"{'模型':>14s}" + ''.join(f"{s+'损失':>10s}{s+'覆盖':>10s}" for s in SEGS))
    for nm, p in preds.items():
        row = f'{nm:>14s}'
        for s in SEGS:
            m = seg == s
            e = fret[m] - p[m]
            pl = np.maximum(q * e, (q - 1) * e).mean()
            row += f'{pl:>10.3f}{(fret[m] < p[m]).mean():>10.4f}'
        print(row, flush=True)
