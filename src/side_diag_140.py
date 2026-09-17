# -*- coding: utf-8 -*-
"""140 指数上的分侧诊断 + 显著性: B8 vs B8+B9 —— ★所有显著性以此面板为准(用户指示)。

回答: 补上 1min 派生的基线特征(B9)后, 隐层的增量是否从"只在高估侧"变成"两侧都有",
      以及严重低估的尾部(风控最在乎的那一档)是否转正。
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB, har_1m_features as H1, extract_universe as U
from run_universe_eval import build_panel_u, load_hidden
from run_layerscan import ridge_path, LAMS
from dm_test_v2 import dm_stat, cw_stat, qlike_loss

OUT = f'{_R}/out'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--H', type=int, default=6)
    a = ap.parse_args()
    cur = json.load(open(f'{OUT}/index_universe_curated.json'))
    cu0 = pd.read_csv(f'{OUT}/universe_curated_list.csv')
    bs = set(cu0[cu0['名称'].astype(str).str.contains('B股|B指')]['代码'])
    want = set(cur['codes']) - bs
    _, _, allc = U.load_grid()
    jsel = np.array([i for i, c in enumerate(allc) if c in want])
    print(f'140 指数(已排 B 股): {len(jsel)}', flush=True)
    Hm, key = load_hidden()
    d, y, F, codes = build_panel_u(a.H, jsel, 'rv')

    # B9: 140 指数的 1min 派生特征
    m1 = json.load(open(f'{OUT}/index1m_grid_140_meta.json'))
    arr1m = np.load(f'{OUT}/index1m_grid_140.npy', mmap_mode='r')
    loc = {c: i for i, c in enumerate(m1['codes'])}
    order = np.array([loc[allc[j]] for j in jsel])          # 对齐到 jsel 顺序
    f1 = H1.build_1m_features(np.asarray(arr1m[:, order, :]))
    for k, v in f1.items():
        F[k] = v[d['t'].values, d['jloc'].values]
    del arr1m, f1

    k = d['t'].values * 1000 + d['j'].values
    p = np.clip(np.searchsorted(key, k), 0, len(key) - 1); good = key[p] == k
    d, y = d[good].reset_index(drop=True), y[good]
    F = {kk: v[good] for kk, v in F.items()}
    Z = Hm[p[good], 1, :].astype(np.float32)
    tr = (d.seg == 'train').values; va = (d.seg == 'val').values
    dts = d['date'].values
    print(f'  对齐后 {len(d):,} 点 (train {tr.sum():,} / val {va.sum():,})', flush=True)

    for tag, b9 in [('B8', False), ('B8+B9', True)]:
        Xb = HB.baseline_design(d, F, len(jsel), 'B8')
        if b9:
            Xb = np.column_stack([Xb] + [F[c] for c in H1.B9_EXTRA])
        Xb = np.nan_to_num(Xb, nan=0., posinf=0., neginf=0.)
        base = HB.ridge_fit(Xb, y, tr)
        pr = ridge_path(Z, y - base, tr, LAMS)
        lam = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va)); full = base + pr[lam]
        r0, r1 = HB.r2(y, base, va), HB.r2(y, full, va)
        tcw, nd = cw_stat(y[va], base[va], full[va], dts[va])
        q0, q1 = qlike_loss(y, base, tr), qlike_loss(y, full, tr)
        tq, _ = dm_stat(q0[va], q1[va], dts[va])
        dd = pd.DataFrame({'d': (q0 - q1)[va], 'date': dts[va]}).groupby('date')['d'].mean().values
        print(f'\n=== 基线 {tag}  (140指数, H={a.H}) ===')
        # ★train 与 val 并报 —— 增量的 train/val 比是判过拟合最直接的诊断
        for sn, mm in [('train', tr), ('val', va)]:
            rr0, rr1 = HB.r2(y, base, mm), HB.r2(y, full, mm)
            tc, nn = cw_stat(y[mm], base[mm], full[mm], dts[mm])
            tqq, _ = dm_stat(q0[mm], q1[mm], dts[mm])
            print(f'  [{sn:>5s}] R² {rr0:.4f} -> {rr1:.4f}  Δ{rr1-rr0:+.4f}  ★CW t={tc:.2f}  '
                  f'QLIKE改善 {(1-q1[mm].mean()/q0[mm].mean())*100:+.1f}% (DM t={tqq:.2f})  n天={nn}')
        print(f'  ★train/val 增量比 = {(HB.r2(y,full,tr)-HB.r2(y,base,tr))/max(r1-r0,1e-9):.2f}'
              f'   (>1.3 = 过拟合特征; ≈1 = 无)')
        print(f'  基线自身 train/val R² 差 = {HB.r2(y,base,tr)-r0:+.4f}'
              f'   (基线过拟合 ≠ 增量过拟合, 要分开看)')
        e0, e1 = (y - base) ** 2, (y - full) ** 2
        err = y - base
        print(f"  {'情形':<20s}{'占比':>7s}{'基线MSE':>10s}{'MSE下降%':>10s}"
              f"{'QLIKE下降%':>11s}{'占QLIKE总量':>12s}")
        for lab, m in [('基线低估(实际>预测)', va & (err > 0)),
                       ('基线高估(实际<预测)', va & (err <= 0))]:
            print(f'  {lab:<20s}{m.sum()/va.sum()*100:>6.1f}%{e0[m].mean():>10.4f}'
                  f'{(1-e1[m].mean()/e0[m].mean())*100:>9.1f}%'
                  f'{(1-q1[m].mean()/q0[m].mean())*100:>10.1f}%'
                  f'{q0[m].sum()/q0[va].sum()*100:>11.1f}%')
        thr = np.quantile(err[va], 0.9); m = va & (err > thr)
        print(f'  {"★严重低估(前10%)":<20s}{10.0:>6.1f}%{e0[m].mean():>10.4f}'
              f'{(1-e1[m].mean()/e0[m].mean())*100:>9.1f}%'
              f'{(1-q1[m].mean()/q0[m].mean())*100:>10.1f}%'
              f'{q0[m].sum()/q0[va].sum()*100:>11.1f}%')


if __name__ == '__main__':
    main()
