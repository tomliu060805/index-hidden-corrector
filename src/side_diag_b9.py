# -*- coding: utf-8 -*-
"""分侧诊断: 隐层的增量落在"基线低估"还是"基线高估"那一侧 —— 跨基线对照。

★为什么必须做: 今天测出 B8 下增量全在高估侧(+12.0%)、低估侧为负(-5.4%),
  据此判了"风控用途不成立"(波动预测最危险的失败是低估)。
  但 A/D 又显示 B8+B9 下 QLIKE 从 1.8%(t=1.30) 跳到 4.0%(t=2.80)。
  若为真, 说明补上 1min 基线特征后隐层的价值**移到了低估侧** —— 那个判负要改。
  若分侧没动, 则 t=2.80 只是 3 指数小样本的抖动(跨基线 3.75->4.07->1.30->2.80 非单调,
  本就更像后者)。这是唯一能区分两者的检验。

用法: python side_diag_b9.py --H 6
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB, har_1m_features as H1, extract_1m as X1
from run_layerscan import (build_panel, load_layers, segmask, ridge_path,
                           LAMS, POS_NAMES, CODES, SEAL_OK)
from dm_test_v2 import qlike_loss


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--H', type=int, default=6)
    a = ap.parse_args(); assert not SEAL_OK
    d, y, F = build_panel(a.H, 'rv')
    arr1m, _, _ = X1.load_grid()
    for k, v in H1.build_1m_features(np.asarray(arr1m)).items():
        F[k] = v[d['t'].values, d['j'].values]
    Hm5, key5 = load_layers()
    k = d['t'].values.astype(np.int64) * 100 + d['j'].values.astype(np.int64)
    p5 = np.clip(np.searchsorted(key5, k), 0, len(key5) - 1); g5 = key5[p5] == k
    d, y = d[g5].reset_index(drop=True), y[g5]; F = {kk: v[g5] for kk, v in F.items()}
    Z = Hm5[p5[g5], POS_NAMES.index('L8n'), 0, :].astype(np.float32)
    tr, va = segmask(d, 'train'), segmask(d, 'val')
    for tag, b9 in [('B8', False), ('B8+B9', True)]:
        Xb = HB.baseline_design(d, F, len(CODES), 'B8')
        if b9:
            Xb = np.column_stack([Xb] + [F[c] for c in H1.B9_EXTRA])
        Xb = np.nan_to_num(Xb, nan=0., posinf=0., neginf=0.)
        base = HB.ridge_fit(Xb, y, tr)
        pr = ridge_path(Z, y - base, tr, LAMS)
        lam = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va)); full = base + pr[lam]
        e0, e1 = (y - base) ** 2, (y - full) ** 2
        q0, q1 = qlike_loss(y, base, tr), qlike_loss(y, full, tr)
        err = y - base                                   # >0 = 基线低估
        print(f'\n=== 基线 {tag}  (val, 3指数, H={a.H}) ===')
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
