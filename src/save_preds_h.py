"""按指定视界 H 重训 M0(岭) + GatedLinear 修正器, 保存全样本预测 -> out/preds_H{H}.npz

生产覆盖层用。与 save_preds_2h.py 完全同构, 只把 H 参数化, 并修两处:
  ★1 原脚本会打印 test/holdout 的 R² —— 那是封存段。本脚本只打印 train/val。
     (预测本身在全样本上算没问题: 模型只用 train 拟合、val 早停; 封存的是**评估**。)
  ★2 线程数改为读 OMP_NUM_THREADS, 原来硬编码 30, 与其他作业并行时会超核数预算。

目标口径保持 MAD(mean|r|) 而非 RV —— 因为 VT 规则 w=clip(σ*/σ̂,1,3) 要的是 **σ 量纲**。
MAD ≈ σ; 若换成 log-RV 则 exp() 出来是 σ², 仓位规则会变成 σ*²/σ̂², 是另一套更激进的规则。
(v2 已证两者在对数空间相关 0.99、尺度恰好差 2 倍, 故换与不换对**预报精度**无影响。)

用法: python save_preds_h.py --H 6
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, f'{_R}/src')
# ★不要 import task3_horizon —— 它没有 __main__ 保护, import 即跑整个实验并打印封存段。
#   需要的两个纯函数已搬到无副作用的 _paneldata。
from _paneldata import load_close, seg_of, ridge_fit, r2
from run_paper1_stack import GatedLinear

B = f'{_R}'


def train_gated_nt(X, resid, tr, va, rank=32, lr_=1e-3, epochs=60, bs=8192, wd=1e-4, seed=0):
    nt = int(os.environ.get('OMP_NUM_THREADS', '8'))
    torch.set_num_threads(nt)
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(resid.astype(np.float32))
    net = GatedLinear(X.shape[1], rank)
    opt = torch.optim.Adam(net.parameters(), lr=lr_, weight_decay=wd)
    idx = np.where(tr)[0]
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            ((net(Xt[b]) - yt[b]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(Xt[va]).numpy() - resid[va]) ** 2).mean()
        if vl < best - 1e-7:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return net(Xt).numpy(), ep, (mu, sd, best_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=int, required=True)
    a = ap.parse_args()
    H = a.H
    lr, dates = load_close()
    NG = lr.shape[0]
    alr = np.abs(lr)
    fvol = np.full(lr.shape, np.nan)
    for i in range(NG - H):
        if i % 48 > 47 - H:
            continue
        fvol[i] = np.nanmean(alr[i + 1:i + 1 + H], 0)
    pv = pd.DataFrame(alr).shift(1)
    v12 = pv.rolling(12, min_periods=6).mean().values
    v48 = pv.rolling(48, min_periods=24).mean().values
    v240 = pv.rolling(240, min_periods=120).mean().values

    Hs, Ms = [], []
    for f in sorted(glob.glob(f'{B}/out/hidden/chunk_*.npz')):
        z = np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
    Hi = np.concatenate(Hs).astype(np.float32); Mi = np.concatenate(Ms)
    ti = Mi[:, 0].astype(np.int64); ji = Mi[:, 1].astype(np.int64)
    keep = (ti % 48) <= 47 - H
    d = pd.DataFrame({'t': ti[keep], 'j': ji[keep], 'nll': Mi[keep, 2], 'ent': Mi[keep, 3]})
    d['y'] = np.log(np.clip(fvol[d.t, d.j], 1e-8, None))
    for nm, arr in [('v12', v12), ('v48', v48), ('v240', v240)]:
        d[nm] = arr[d.t, d.j]
    d['aret'] = alr[d.t, d.j]
    d['date'] = dates[d.t // 48]; d['bar'] = d.t % 48
    ok = d[['y', 'v12', 'v48', 'v240', 'aret']].notna().all(1).values & np.isfinite(d.y.values)
    d = d[ok].reset_index(drop=True); Hk = Hi[keep][ok]
    seg = seg_of(d.date.values)
    y = d.y.values
    RVlog = np.column_stack([np.log(np.clip(d[c].values, 1e-8, None))
                             for c in ['v12', 'v48', 'v240', 'aret']])
    Xb = np.column_stack([RVlog, pd.get_dummies(d.bar).values,
                          pd.get_dummies(d.j).values]).astype(np.float32)
    tr = seg == 'train'; va = seg == 'val'
    print(f'H={H} ({H*5}min)  n={len(d):,}  train {tr.sum():,} / val {va.sum():,}', flush=True)
    yh0 = ridge_fit(Xb, y, tr)
    X = np.column_stack([Xb, d.nll.values, d.ent.values, Hk]).astype(np.float32)
    pg, ep, _ = train_gated_nt(X, (y - yh0).astype(np.float32), tr, va)
    # ★只报 train/val —— test/holdout 封存
    for s in ['train', 'val']:
        m = seg == s
        print(f'  {s}: M0 {r2(y, yh0, m):.4f} -> M2 {r2(y, yh0 + pg, m):.4f} '
              f'(Δ{r2(y, yh0+pg, m)-r2(y, yh0, m):+.4f})', flush=True)
    fo = f'{B}/out/preds_H{H}.npz'
    np.savez(fo, yh0=yh0, p2=pg, y=y, seg=seg.astype('U8'),
             date=d.date.values.astype('U10'), j=d.j.values, bar=d.bar.values)
    print(f'saved {fo}  (早停于 epoch {ep})')


if __name__ == '__main__':
    main()
