"""v2 增量的统计显著性: Diebold-Mariano + Clark-West + QLIKE。只用 train/val。

为什么两个检验都要:
  DM (Diebold-Mariano 1995) 比较两个预测的损失差, 但**在嵌套模型上有偏** ——
  大模型包含小模型, 原假设下大模型的多余参数只带来估计噪声, 损失差的均值天然为负,
  DM 会系统性低估。
  CW (Clark-West 2007) 对这项偏差做了修正, 是嵌套比较的正确检验。
  B4 与 B4+隐层 正是嵌套的, 所以以 CW 为准, DM 仅作参照。

损失口径两套:
  MSE   —— 直接在 log 目标上, 与 R² 同源
  QLIKE —— Patton 2011 证明其对波动代理的噪声稳健; 在方差口径上算, 带 train 段乘性偏差校正

误差聚合到**日**再做 HAC(Newey-West), 因为同一天内相邻决策点的未来窗高度重叠。

用法: python dm_test_v2.py --H 6 --pos L8n
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import argparse, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB
from run_layerscan import (build_panel, load_layers, align, segmask,
                           ridge_path, LAMS, POS_NAMES, CODES, SEAL_OK)


def nw_var(x, lag):
    n = len(x); e = x - x.mean()
    v = (e @ e) / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * ((e[l:] @ e[:-l]) / n)
    return max(v, 1e-30)


def dm_stat(l1, l2, dates, lag=10):
    """DM: 日均损失差 d̄ 的 HAC t 值。正 = 模型2(大)更好。"""
    d = pd.DataFrame({'d': l1 - l2, 'date': dates}).groupby('date')['d'].mean().values
    return d.mean() / np.sqrt(nw_var(d, lag) / len(d)), len(d)


def cw_stat(y, f1, f2, dates, lag=10):
    """Clark-West: f_adj = (y−f1)² − [(y−f2)² − (f1−f2)²], 对嵌套的估计噪声做修正。"""
    a = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    d = pd.DataFrame({'d': a, 'date': dates}).groupby('date')['d'].mean().values
    return d.mean() / np.sqrt(nw_var(d, lag) / len(d)), len(d)


def qlike_loss(y, yh, tr):
    """方差口径 QLIKE, 带 train 段乘性偏差校正(每个模型算自己的)。"""
    c = np.exp((y - yh)[tr].mean())
    ratio = np.exp(y) / (c * np.exp(yh))
    return ratio - np.log(ratio) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=int, default=6)
    ap.add_argument('--pos', default='L8n')
    ap.add_argument('--kind', default='rv')
    ap.add_argument('--base', default='B4')
    a = ap.parse_args()
    assert not SEAL_OK, 'SEAL_OK 必须为 False'
    pi = POS_NAMES.index(a.pos)

    d, y, F = build_panel(a.H, a.kind)
    Hm, key = load_layers()
    p, good = align(d, key, Hm)
    d, y = d[good].reset_index(drop=True), y[good]
    F = {k: v[good] for k, v in F.items()}
    Z = Hm[p[good], pi, 0, :].astype(np.float32)
    tr, va = segmask(d, 'train'), segmask(d, 'val')

    Xb = HB.baseline_design(d, F, len(CODES), a.base)
    base = HB.ridge_fit(Xb, y, tr)
    pr = ridge_path(Z, y - base, tr, LAMS)
    lam = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va))
    full = base + pr[lam]

    dates = d['date'].values
    print(f'\nH={a.H} ({a.H*5}min)  位置={a.pos}  基线={a.base}  n={len(d):,}  λ={lam:.0f}')
    print(f'{"段":>6s}{"R² 基线":>9s}{"R² +隐层":>10s}{"ΔR²":>9s}'
          f'{"DM t":>9s}{"CW t":>9s}{"QLIKE改善%":>11s}{"QLIKE DM t":>12s}{"n天":>7s}')
    for s in ['train', 'val']:
        m = segmask(d, s)
        r0, r1 = HB.r2(y, base, m), HB.r2(y, full, m)
        # MSE 口径
        t_dm, nd = dm_stat(((y - base) ** 2)[m], ((y - full) ** 2)[m], dates[m])
        t_cw, _ = cw_stat(y[m], base[m], full[m], dates[m])
        # QLIKE 口径
        q0 = qlike_loss(y, base, tr)[m]
        q1 = qlike_loss(y, full, tr)[m]
        t_q, _ = dm_stat(q0, q1, dates[m])
        print(f'{s:>6s}{r0:9.4f}{r1:10.4f}{r1-r0:+9.4f}'
              f'{t_dm:9.2f}{t_cw:9.2f}{(1-q1.mean()/q0.mean())*100:10.1f}%{t_q:12.2f}{nd:7d}')
    print('\n嵌套比较以 CW 为准; DM 在嵌套下有偏(系统性低估大模型)。')
    print('★ test/holdout 未计算 —— SEAL_OK=False。')


if __name__ == '__main__':
    main()
