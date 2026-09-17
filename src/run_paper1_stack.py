"""论文一栈 (arXiv 2608.08825): 冻结Kronos + GatedLinear修正器 + LGBM残差 —— 指数未来1h波动

M0 = B1b 岭回归 (已实现波动+节律独热)             —— 基线
M1 = M0 + LGBM残差(全特征含512维隐层)             —— 仅经典 (论文的RF组件)
M2 = M0 + GatedLinear修正器(低秩双线性+门控)       —— 仅神经
M3 = M2 + LGBM残差                                —— 完整混合栈 (论文最优 GatedLinear+RF)
M4 = M0 + LGBM残差(无隐层特征)                     —— 隐层归因: M1−M4 = 隐层通过树的贡献
早停/调参只用 val; test 只做最终报告; holdout 锁定不碰
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, f'{_R}/src')
from eval_index_ridge import load_all, ridge_fit, r2, three_corr, daily_ic
import lightgbm as lgb

torch.manual_seed(0); np.random.seed(0)
SEGS = ['train', 'val', 'test']


class GatedLinear(nn.Module):
    """论文一 GatedLinear: 低秩投影 + 门控逐元素控制修正幅度"""
    def __init__(self, dim, rank=32):
        super().__init__()
        self.proj_r = nn.Linear(dim, rank)
        self.proj_g = nn.Linear(dim, rank)
        self.head = nn.Linear(rank, 1)

    def forward(self, x):
        r = self.proj_r(x)
        g = torch.sigmoid(self.proj_g(x))
        return self.head(g * r).squeeze(-1)


def train_gated(X, resid, tr, va, rank=32, lr=1e-3, epochs=60, bs=8192, wd=1e-4):
    torch.set_num_threads(30)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xn = (X - mu) / sd
    Xt = torch.from_numpy(Xn.astype(np.float32))
    yt = torch.from_numpy(resid.astype(np.float32))
    net = GatedLinear(X.shape[1], rank)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    idx = np.where(tr)[0]
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xt[va]).numpy()
        vloss = ((pv - resid[va]) ** 2).mean()
        if vloss < best - 1e-6:
            best, best_state, patience = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return net(Xt).numpy(), ep


def lgb_resid(X, resid, tr, va, cols):
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                          min_child_samples=100, subsample=0.8, colsample_bytree=0.7,
                          n_jobs=30, random_state=0, verbose=-1)
    m.fit(X[tr], resid[tr], eval_set=[(X[va], resid[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(X), m.best_iteration_


def report(name, d, y, yh, base_rr=None):
    rr = {k: r2(y, yh, (d.seg == k).values) for k in SEGS}
    inc = ''
    if base_rr:
        inc = f"  Δval={rr['val']-base_rr['val']:+.4f} Δtest={rr['test']-base_rr['test']:+.4f}"
    print(f"{name:>24s}: train={rr['train']:.4f} val={rr['val']:.4f} test={rr['test']:.4f}{inc}", flush=True)
    return rr


def main():
    d, Hm = load_all()
    tr = (d.seg == 'train').values; va = (d.seg == 'val').values
    y = np.log(d['fvol'].values)
    onehot_bar = pd.get_dummies(d['bar']).values.astype(np.float32)
    onehot_idx = pd.get_dummies(d['j']).values.astype(np.float32)
    RVlog = np.column_stack([np.log(np.clip(d[c].values, 1e-8, None))
                             for c in ['v12', 'v48', 'v240', 'aret']]).astype(np.float32)
    Xbase = np.column_stack([RVlog, onehot_bar, onehot_idx])
    Xfull = np.column_stack([Xbase, d.nll.values, d.ent.values, Hm]).astype(np.float32)
    Xnoh = np.column_stack([Xbase, d.nll.values, d.ent.values]).astype(np.float32)

    yh0 = ridge_fit(Xbase, y, tr)
    rr0 = report('M0 岭回归基线', d, y, yh0)
    resid0 = y - yh0

    # M1: 仅经典 —— LGBM 学基线残差 (含隐层)
    p1, it1 = lgb_resid(Xfull, resid0, tr, va, None)
    rr1 = report(f'M1 M0+LGBM残差(含隐层,{it1}轮)', d, y, yh0 + p1, rr0)

    # M4: LGBM 残差但无隐层 —— 隐层通过树的贡献 = M1−M4
    p4, it4 = lgb_resid(Xnoh, resid0, tr, va, None)
    rr4 = report(f'M4 M0+LGBM残差(无隐层,{it4}轮)', d, y, yh0 + p4, rr0)

    # M2: 仅神经 —— GatedLinear 修正器
    p2, ep2 = train_gated(Xfull, resid0, tr, va)
    rr2 = report(f'M2 M0+GatedLinear({ep2}ep)', d, y, yh0 + p2, rr0)

    # M3: 完整混合 —— GatedLinear 之后 LGBM 再学一层残差 (论文最优结构)
    resid2 = y - (yh0 + p2)
    p3, it3 = lgb_resid(np.column_stack([Xfull, p2]).astype(np.float32), resid2, tr, va, None)
    rr3 = report(f'M3 M2+LGBM残差({it3}轮)', d, y, yh0 + p2 + p3, rr0)

    print('\n残差日IC (各模型增量 vs M0残差, val/test):')
    for nm, p in [('M1', p1), ('M4', p4), ('M2', p2), ('M3', p2 + p3)]:
        for k in ['val', 'test']:
            m = (d.seg == k).values
            ic, icir, n = daily_ic(d, p, resid0, m)
            print(f'  {nm} {k}: IC={ic:+.4f} ICIR={icir:+.2f} (n={n})')

    print('\n论文一三相关性 pooled/per-day/cross-day (test):')
    mt = (d.seg == 'test').values
    for nm, yh in [('M0', yh0), ('M1', yh0 + p1), ('M3', yh0 + p2 + p3)]:
        p_, pd_, cd_ = three_corr(d, y, yh, mt)
        print(f'  {nm}: {p_:.4f} / {pd_:.4f} / {cd_:.4f}')

    print('\n分指数 test R² (M0 → M3):')
    CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']
    yh3 = yh0 + p2 + p3
    for jj, c in enumerate(CODES):
        m = mt & (d.j == jj).values
        print(f'  {c}: {r2(y, yh0, m):.4f} → {r2(y, yh3, m):.4f} (Δ{r2(y, yh3, m)-r2(y, yh0, m):+.4f})')

    np.savez(f'{_R}/out/paper1_preds.npz',
             yh0=yh0, p1=p1, p2=p2, p3=p3, p4=p4, y=y,
             seg=d.seg.values.astype('U8'), date=d.date.values.astype('U10'),
             j=d.j.values, bar=d.bar.values)


if __name__ == '__main__':
    main()
