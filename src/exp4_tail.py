"""实验④ 尾部目标: 隐层对 日内VaR / 大跌概率 / 尾部波动 的增量 (现有隐层, 无需重提)

A. 未来1h收益 q05/q01 分位回归 (线性pinball, torch): 基线Xb vs Xb+隐层
   指标 = pinball损失改善% + VaR覆盖率校准 (命中率 vs 名义, 近似Kupiec z; 样本重叠, z为近似)
B. 大跌概率: label = fret < train段该指数5%分位, 线性logistic, AUC
C. 尾部波动: log q95(|r|, 未来12根), 岭回归 R²
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, f'{_R}/src')
from eval_index_ridge import load_all, ridge_fit, r2

torch.set_num_threads(60)
SEGS = ['val', 'test', 'holdout']
B = f'{_R}'

d, Hm = load_all()
seg = d.seg.values
tr = seg == 'train'
fret = d.fret.values * 1e4                      # bp
Xb = np.column_stack([
    np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
Xh = np.column_stack([Xb, d.nll.values, d.ent.values, Hm]).astype(np.float32)


def std_fit(X):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


Xb_n, Xh_n = std_fit(Xb), std_fit(Xh)


def train_linear(Xn, y, loss_fn, epochs=400, lr=5e-2, wd=1e-5):
    torch.manual_seed(0)
    Xt = torch.from_numpy(Xn[tr]); yt = torch.from_numpy(y[tr].astype(np.float32))
    w = torch.zeros(Xn.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for ep in range(epochs):
        opt.zero_grad()
        pred = Xt @ w + b
        loss = loss_fn(pred, yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        return (torch.from_numpy(Xn) @ w + b).numpy()


# ---------- A. 分位回归 ----------
print('===== A. 日内VaR (未来1h收益分位, 线性pinball) =====')
for q in [0.05, 0.01]:
    def pinball(pred, yt, q=q):
        e = yt - pred
        return torch.maximum(q * e, (q - 1) * e).mean()
    res = {}
    for nm, Xn in [('基线', Xb_n), ('+隐层', Xh_n)]:
        p = train_linear(Xn, fret, pinball)
        res[nm] = p
    print(f'-- q={q} --')
    print(f"{'段':>8s}{'pinball基线':>12s}{'pinball+隐层':>13s}{'改善':>8s}{'覆盖基线':>10s}{'覆盖+隐层':>10s}{'名义':>7s}{'z基线':>8s}{'z+隐层':>8s}")
    for s in SEGS:
        m = seg == s
        pl = {}
        cov = {}
        for nm in ['基线', '+隐层']:
            e = fret[m] - res[nm][m]
            pl[nm] = np.maximum(q * e, (q - 1) * e).mean()
            cov[nm] = (fret[m] < res[nm][m]).mean()
        n = m.sum()
        z = {nm: (cov[nm] - q) / np.sqrt(q * (1 - q) / n) for nm in cov}
        print(f'{s:>8s}{pl["基线"]:>12.3f}{pl["+隐层"]:>13.3f}'
              f'{(1 - pl["+隐层"] / pl["基线"]) * 100:>7.2f}%'
              f'{cov["基线"]:>10.4f}{cov["+隐层"]:>10.4f}{q:>7.2f}{z["基线"]:>+8.1f}{z["+隐层"]:>+8.1f}')

# ---------- B. 大跌概率 ----------
print('\n===== B. 大跌概率 (fret < train段该指数5%分位, 线性logistic, AUC) =====')
thr = {j: np.quantile(fret[tr & (d.j == j).values], 0.05) for j in range(3)}
lab = np.array([fret[i] < thr[d.j.values[i]] for i in range(len(d))]).astype(np.float32)
print(f'阈值(bp): ' + ' '.join(f'{k}:{v:.0f}' for k, v in thr.items()) + f'  正例率 train={lab[tr].mean():.3f}')
bce = torch.nn.BCEWithLogitsLoss()
from sklearn.metrics import roc_auc_score
res = {}
for nm, Xn in [('基线', Xb_n), ('+隐层', Xh_n)]:
    res[nm] = train_linear(Xn, lab, lambda p, y: bce(p, y))
print(f"{'段':>8s}{'AUC基线':>10s}{'AUC+隐层':>10s}{'Δ':>8s}")
for s in SEGS:
    m = seg == s
    a0 = roc_auc_score(lab[m], res['基线'][m])
    a1 = roc_auc_score(lab[m], res['+隐层'][m])
    print(f'{s:>8s}{a0:>10.4f}{a1:>10.4f}{a1 - a0:>+8.4f}')

# ---------- C. 尾部波动 q95 ----------
print('\n===== C. 尾部波动 log q95(|r|, 未来12根) 岭回归 =====')
from task3_horizon import load_close
lr_, dates = load_close()
from numpy.lib.stride_tricks import sliding_window_view
NG = lr_.shape[0]
sw = sliding_window_view(np.abs(lr_), 12, axis=0)      # (NG-11, 3, 12)
q95 = np.full(lr_.shape, np.nan)
vals = np.nanquantile(sw, 0.95, axis=2)                # (NG-11, 3)
q95[:NG - 11] = vals
t_i = d.t.values; j_i = d.j.values
yq = np.log(np.clip(q95[t_i + 1, j_i], 1e-8, None))    # 未来12根: 窗口起点 t+1
okq = np.isfinite(yq)
yh0 = ridge_fit(Xb[okq], yq[okq], tr[okq])
yh1 = ridge_fit(Xh[okq], yq[okq], tr[okq])
sq = seg[okq]
print(f"{'段':>8s}{'基线R²':>10s}{'+隐层R²':>10s}{'Δ':>8s}")
for s in SEGS:
    m = sq == s
    print(f'{s:>8s}{r2(yq[okq], yh0, m):>10.4f}{r2(yq[okq], yh1, m):>10.4f}'
          f'{r2(yq[okq], yh1, m) - r2(yq[okq], yh0, m):>+8.4f}')
