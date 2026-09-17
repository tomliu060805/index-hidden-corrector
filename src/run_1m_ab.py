"""1min 输入 vs 5min 输入的四格对照 —— 3 指数, train/val, test/holdout 封存。

四格(★干净的输入频率对比是 C vs D: 同一基线, 只变 Kronos 的输入粒度):

  格  Kronos 输入        基线          回答什么
  A   5min × L=128      B8            现状
  B   1min × L=512      B8            仅作诊断 —— 信息不对等, 会虚高
  C   1min × L=512      B8 + B9       1min 隐层在**对等**基线上的增量
  D   5min × L=128      B8 + B9       5min 隐层在**对等**基线上的增量

  C − D  = 输入频率的真实价值
  A − D  = B9(1min 派生的基线特征)本身值多少 —— 基线侧的红利
  B − C  = 基线没看见的 1min 信息值多少 —— `baseline-sees-fewer-inputs` 的量化版

用法: python run_1m_ab.py --H 6 --pos L8n
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB
import har_1m_features as H1
from run_layerscan import (build_panel, load_layers, align, segmask,
                           ridge_path, LAMS, POS_NAMES, CODES, SEAL_OK)
from dm_test_v2 import dm_stat, cw_stat, qlike_loss

OUT = f'{_R}/out'


def load_hidden_1m():
    fs = sorted(glob.glob(f'{OUT}/hidden_1m/chunk_*.npz'))
    Hs, Ms = [], []
    for f in fs:
        z = np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
    Hm = np.concatenate(Hs); M = np.concatenate(Ms).astype(np.int64)
    key = M[:, 0] * 100 + M[:, 1]
    o = np.argsort(key)
    print(f'  1min 隐层库 {Hm.shape}  chunks={len(fs)}', flush=True)
    return Hm[o], key[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=int, default=6)
    ap.add_argument('--pos', default='L8n')
    ap.add_argument('--kind', default='rv')
    a = ap.parse_args()
    assert not SEAL_OK
    pi5 = POS_NAMES.index(a.pos)          # 5min 库: 10 位置
    pi1 = ['L6', 'L8n'].index(a.pos)      # 1min 库: 只存了 2 位置

    d, y, F = build_panel(a.H, a.kind)

    # ---- 接上 B9: 1min 派生的基线特征 ----
    import extract_1m as X1
    arr1m, dates1, codes1 = X1.load_grid()
    f1 = H1.build_1m_features(np.asarray(arr1m))
    for k, v in f1.items():
        F[k] = v[d['t'].values, d['j'].values]

    # ---- 对齐两个隐层库 ----
    Hm5, key5 = load_layers()
    Hm1, key1 = load_hidden_1m()
    k = d['t'].values.astype(np.int64) * 100 + d['j'].values.astype(np.int64)
    p5 = np.clip(np.searchsorted(key5, k), 0, len(key5) - 1); g5 = key5[p5] == k
    p1 = np.clip(np.searchsorted(key1, k), 0, len(key1) - 1); g1 = key1[p1] == k
    good = g5 & g1
    for nm, gg in [('5min隐层', g5), ('1min隐层', g1), ('两者都有', good)]:
        print(f'  {nm} 覆盖 {gg.sum():,}/{len(d):,} ({gg.mean()*100:.1f}%)', flush=True)
    d, y = d[good].reset_index(drop=True), y[good]
    F = {kk: v[good] for kk, v in F.items()}
    Z5 = Hm5[p5[good], pi5, 0, :].astype(np.float32)
    Z1 = Hm1[p1[good], pi1, :].astype(np.float32)
    tr, va = segmask(d, 'train'), segmask(d, 'val')
    dts = d['date'].values
    ok9 = np.isfinite(np.column_stack([F[c] for c in H1.B9_EXTRA])).all(1)
    print(f'  B9 特征可用 {ok9.mean()*100:.1f}%  |  对齐后 {len(d):,} 点 '
          f'(train {tr.sum():,} / val {va.sum():,})', flush=True)

    def design(base_id, with_b9):
        Xb = HB.baseline_design(d, F, len(CODES), base_id)
        if with_b9:
            Xb = np.column_stack([Xb] + [F[c] for c in H1.B9_EXTRA])
        return np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n{'格':<4s}{'Kronos输入':<12s}{'基线':<10s}{'基线R²':>9s}{'+隐层R²':>10s}"
          f"{'ΔR²':>9s}{'CW t':>8s}{'QLIKE%':>9s}{'Q t':>7s}")
    res = {}
    for tag, Z, base_id, b9, lab_in, lab_b in [
            ('A', Z5, 'B8', False, '5min×128', 'B8'),
            ('B', Z1, 'B8', False, '1min×512', 'B8'),
            ('C', Z1, 'B8', True, '1min×512', 'B8+B9'),
            ('D', Z5, 'B8', True, '5min×128', 'B8+B9')]:
        Xb = design(base_id, b9)
        base = HB.ridge_fit(Xb, y, tr)
        pr = ridge_path(Z, y - base, tr, LAMS)
        lam = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va))
        full = base + pr[lam]
        r0, r1 = HB.r2(y, base, va), HB.r2(y, full, va)
        t_cw, nd = cw_stat(y[va], base[va], full[va], dts[va])
        q0 = qlike_loss(y, base, tr)[va]; q1 = qlike_loss(y, full, tr)[va]
        t_q, _ = dm_stat(q0, q1, dts[va])
        res[tag] = dict(base=r0, full=r1, d=r1 - r0, cw=t_cw,
                        q=(1 - q1.mean() / q0.mean()) * 100, qt=t_q, lam=lam)
        print(f'{tag:<4s}{lab_in:<12s}{lab_b:<10s}{r0:>9.4f}{r1:>10.4f}{r1-r0:>+9.4f}'
              f'{t_cw:>8.2f}{(1-q1.mean()/q0.mean())*100:>8.1f}%{t_q:>7.2f}', flush=True)

    print(f'\n★ C − D = {res["C"]["d"] - res["D"]["d"]:+.4f}   '
          f'(同基线 B8+B9, 只变输入粒度 => **输入频率的真实价值**)')
    print(f'  A − D = {res["A"]["d"] - res["D"]["d"]:+.4f}   '
          f'(B9 这组 1min 基线特征吃掉了 5min 隐层多少增量)')
    print(f'  B − C = {res["B"]["d"] - res["C"]["d"]:+.4f}   '
          f'(基线没看见的 1min 信息值多少 = baseline-sees-fewer-inputs 的量化)')
    print(f'  基线本身 B8 {res["A"]["base"]:.4f} -> B8+B9 {res["D"]["base"]:.4f} '
          f'({res["D"]["base"]-res["A"]["base"]:+.4f})')
    pd.DataFrame(res).T.to_csv(f'{OUT}/v2_1m_ab_H{a.H}.csv')
    print(f'\n写出 {OUT}/v2_1m_ab_H{a.H}.csv    ★ test/holdout 未计算')


if __name__ == '__main__':
    main()
