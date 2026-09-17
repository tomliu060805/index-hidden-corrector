"""实验①②③统一评估: 各表示变体对 1h 波动的岭回归增量 (vs B1b强基线)
变体: small final_last(原) / mid_last / mean128 / mean12 / 组合 | base final_last | chronos EOS/mean
最后对 val 最优变体加训 GatedLinear 与生产 M2 对比。
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import glob, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, f'{_R}/src')
from eval_index_ridge import load_all, ridge_fit, r2
from run_paper1_stack import train_gated

B = f'{_R}'
SEGS = ['val', 'test', 'holdout']

d, Hm = load_all()
seg = d.seg.values; tr = seg == 'train'; va = seg == 'val'
y = np.log(d['fvol'].values)
Xb = np.column_stack([
    np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
key = pd.MultiIndex.from_arrays([d.t.values, d.j.values])
pos = pd.Series(np.arange(len(d)), index=key)
base = {s: r2(y, ridge_fit(Xb, y, tr), seg == s) for s in SEGS}
yh_b = ridge_fit(Xb, y, tr)
print('B1b基线: ' + ' '.join(f'{s}={base[s]:.4f}' for s in SEGS), flush=True)


def load_dir(dirname):
    fs = sorted(glob.glob(f'{B}/out/{dirname}/chunk_*.npz'))
    if not fs:
        return None, None
    Hs, Ms = [], []
    for f in fs:
        z = np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
    return np.concatenate(Hs), np.concatenate(Ms)


def align(Hraw, meta):
    """按 (t,j) 对齐到 d 的行序; 未覆盖行=NaN"""
    mk = pd.MultiIndex.from_arrays([meta[:, 0], meta[:, 1]])
    idx = pos.reindex(mk)
    ok = idx.notna().values
    out = np.full((len(d),) + Hraw.shape[1:], np.nan, np.float32)
    out[idx.values[ok].astype(int)] = Hraw[ok].astype(np.float32)
    return out


def evaluate(name, Hv):
    okm = np.isfinite(Hv).all(tuple(range(1, Hv.ndim)))
    X = np.column_stack([Xb[okm], Hv[okm].reshape(okm.sum(), -1)])
    yy = y[okm]; ss = seg[okm]; trm = ss == 'train'
    yh = ridge_fit(X, yy, trm)
    yb = ridge_fit(Xb[okm], yy, trm)
    row = f'{name:>28s} ({Hv.reshape(len(Hv), -1).shape[1]:>5d}维)'
    for s in SEGS:
        m = ss == s
        row += f'  {s}Δ={r2(yy, yh, m) - r2(yy, yb, m):+.4f}'
    print(row + f'  n={okm.sum():,}', flush=True)
    return name


print('\n===== 参考: small final_last (生产) =====')
evaluate('small_final_last', Hm)

Hmulti, Mmulti = load_dir('hidden_multi')
if Hmulti is not None:
    print('\n===== 实验① 多层/多位置 (Kronos-small) =====')
    Ha = align(Hmulti, Mmulti)
    names = ['final_last(sanity)', 'mid_last(第4层)', 'mean128(全窗池化)', 'mean12(近1h池化)']
    for k, nm in enumerate(names):
        evaluate(nm, Ha[:, k, :])
    evaluate('final_last+mean128', Ha[:, [0, 2], :])
    evaluate('final_last+mid', Ha[:, [0, 1], :])
    evaluate('全部4路', Ha)

Hb, Mb = load_dir('hidden_base')
if Hb is not None:
    print('\n===== 实验② Kronos-base final_last =====')
    evaluate('base_final_last', align(Hb, Mb))

Hc, Mc = load_dir('hidden_chronos')
if Hc is not None:
    print('\n===== 实验③ Chronos-t5-small (单变量收盘) =====')
    Hca = align(Hc, Mc)
    evaluate('chronos_EOS', Hca[:, 0, :])
    evaluate('chronos_mean', Hca[:, 1, :])
    evaluate('chronos_两路', Hca)
    evaluate('small_final_last + chronos_EOS',
             np.concatenate([Hm[:, None, :], Hca[:, :1, :]], 1))
