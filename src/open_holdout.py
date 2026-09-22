"""★★★ holdout 一次性开封 —— 预注册见 docs/PREREG_HOLDOUT_OPENING.md

开什么: holdout = 2025-07-18 ~ 2026-09-16 (285 个交易日)
这是本项目最后一段未读数据。**读完之后这条线只能前向累计。**

配置与 test 开封完全一致(不扫参不选优): 5min×L=128 / L8n-last / H=6 / B8+B9 / 140 指数。
拟合只用 train, λ 只在 val 上选; test 不参与拟合也不参与选参, 仅作对照报告。

主判据: holdout ΔR² > 0 且 CW t > 3。其余全部量只报告、不作判据。
★这个脚本只应该跑一次。
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB
import har_1m_features as H1
import extract_universe as U
from run_universe_eval import load_hidden
from run_layerscan import ridge_path, LAMS, pit_pct_same_bar
from dm_test_v2 import dm_stat, cw_stat, qlike_loss
import open_test as OT
from target_measerr_check import target_subsampled

OUT = f'{_R}/out'
TRAIN_END, VAL_END, TEST_END = '2023-04-27', '2024-06-07', '2025-07-17'
H = 6


def build_all(jsel):
    """与 open_test.build 同口径, 但 holdout 不再标 SEALED 而是单独一段。"""
    d, y, F, codes, close = OT.build(jsel)
    d = d.copy()
    d['seg'] = np.where(d.date <= TRAIN_END, 'train',
               np.where(d.date <= VAL_END, 'val',
               np.where(d.date <= TEST_END, 'TEST', 'HOLDOUT')))
    return d, y, F, codes, close


def report(tag, y, base, full, d, segs, tr, dts, crit=False):
    q0, q1 = qlike_loss(y, base, tr), qlike_loss(y, full, tr)
    e0, e1 = (y - base) ** 2, (y - full) ** 2
    print(f"\n--- {tag} ---")
    print(f"{'段':>8s}{'n点':>11s}{'基线R²':>9s}{'+隐层R²':>10s}{'ΔR²':>9s}"
          f"{'DM t':>8s}{'★CW t':>8s}{'QLIKE%':>9s}{'Q t':>7s}{'逐日为正':>9s}{'n天':>6s}")
    out = {}
    for nm in ['train', 'val', 'TEST', 'HOLDOUT']:
        m = segs[nm]
        r0, r1 = HB.r2(y, base, m), HB.r2(y, full, m)
        tdm, nd = dm_stat(e0[m], e1[m], dts[m])
        tcw, _ = cw_stat(y[m], base[m], full[m], dts[m])
        tq, _ = dm_stat(q0[m], q1[m], dts[m])
        dd = pd.DataFrame({'d': (q0 - q1)[m], 'date': dts[m]}).groupby('date')['d'].mean().values
        ql = (1 - q1[m].mean() / q0[m].mean()) * 100
        out[nm] = dict(base=r0, full=r1, d=r1 - r0, cw=tcw, dm=tdm, ql=ql, qt=tq,
                       qpos=(dd > 0).mean() * 100, nd=nd, n=int(m.sum()))
        star = '**' if (nm == 'HOLDOUT' and crit) else ''
        print(f'{star+nm:>8s}{m.sum():>11,d}{r0:>9.4f}{r1:>10.4f}{r1-r0:>+9.4f}'
              f'{tdm:>8.2f}{tcw:>8.2f}{ql:>8.1f}%{tq:>7.2f}{(dd>0).mean()*100:>8.1f}%{nd:>6d}')
    return out


def main():
    print('=' * 96)
    print('★★★ holdout 一次性开封 —— 本项目最后一段未读数据')
    print('    预注册: docs/PREREG_HOLDOUT_OPENING.md')
    print('    主判据: holdout ΔR² > 0 且 CW t > 3')
    print('    预期带: ≥+0.008 无衰减 / (0,+0.008) 继续衰减但成立 / ≤0 判负')
    print('=' * 96, flush=True)

    cur = json.load(open(f'{OUT}/index_universe_curated.json'))
    cu0 = pd.read_csv(f'{OUT}/universe_curated_list.csv')
    bs = set(cu0[cu0['名称'].astype(str).str.contains('B股|B指')]['代码'])
    want = set(cur['codes']) - bs
    _, dates, allc = U.load_grid()
    jsel = np.array([i for i, c in enumerate(allc) if c in want])
    Hm, key = load_hidden()
    d, yA, F, codes, close = build_all(jsel)

    m1 = json.load(open(f'{OUT}/index1m_grid_140_meta.json'))
    a1m = np.load(f'{OUT}/index1m_grid_140.npy', mmap_mode='r')
    loc = {c: i for i, c in enumerate(m1['codes'])}
    order = np.array([loc[allc[j]] for j in jsel])
    a1 = np.asarray(a1m[:, order, :])
    for k, v in H1.build_1m_features(a1).items():
        F[k] = v[d['t'].values, d['jloc'].values]
    # 目标 C(子采样 5min RV) —— 稳健性用, 非头条
    yC_tab, okC_tab, _ = target_subsampled(a1[:, :, 3], np.arange(close.shape[0]))
    del a1m, a1

    ti, jl = d['t'].values, d['jloc'].values
    yC = yC_tab[ti, jl]; okC = okC_tab[ti, jl]
    kk = ti * 1000 + d['j'].values
    p = np.clip(np.searchsorted(key, kk), 0, len(key) - 1); good = key[p] == kk
    d = d[good].reset_index(drop=True)
    yA, yC, okC = yA[good], yC[good], okC[good]
    F = {k: v[good] for k, v in F.items()}
    Z = Hm[p[good], 1, :].astype(np.float32)          # L8n / last
    del Hm
    segs = {nm: (d.seg == nm).values for nm in ['train', 'val', 'TEST', 'HOLDOUT']}
    tr, va = segs['train'], segs['val']
    dts = d['date'].values
    hd = segs['HOLDOUT']
    print(f"\n面板 {len(d):,} 点 | train {tr.sum():,} / val {va.sum():,} / "
          f"TEST {segs['TEST'].sum():,} / **HOLDOUT {hd.sum():,}**", flush=True)
    print(f"holdout 区间 {dts[hd].min()} ~ {dts[hd].max()}  "
          f"({pd.Series(dts[hd]).nunique()} 个交易日)", flush=True)

    Xb = np.column_stack([HB.baseline_design(d, F, len(jsel), 'B8')]
                         + [F[c] for c in H1.B9_EXTRA])
    Xb = np.nan_to_num(Xb, nan=0., posinf=0., neginf=0.)
    base = HB.ridge_fit(Xb, yA, tr)                   # ★只用 train 拟合
    pr = ridge_path(Z, yA - base, tr, LAMS)
    lam = max(LAMS, key=lambda L: HB.r2(yA, base + pr[L], va))   # ★λ 只在 val 选
    full = base + pr[lam]
    print(f'λ = {lam:.0f} (在 val 上选定, 与 test 开封一致)', flush=True)

    rA = report('★头条 目标 A = 6 根 5min 收益 (与 test 开封同口径)', yA, base, full,
                d, segs, tr, dts, crit=True)

    h = rA['HOLDOUT']; t = rA['TEST']; v = rA['val']
    print('\n' + '=' * 96)
    print('★ 主判据')
    print('=' * 96)
    ok1, ok2 = h['d'] > 0, h['cw'] > 3
    print(f"  holdout ΔR² = {h['d']:+.4f} (>0) {'✓' if ok1 else '✗'}   "
          f"CW t = {h['cw']:.2f} (>3) {'✓' if ok2 else '✗'}   =>  "
          f"**{'通过' if (ok1 and ok2) else '未通过'}**")
    band = ('≥+0.008 无衰减' if h['d'] >= 0.008 else
            '(0,+0.008) 继续衰减但成立' if h['d'] > 0 else '≤0 判负')
    print(f"  落在预注册的哪一带: {band}")
    print(f"  衰减链: val {v['d']:+.4f} -> test {t['d']:+.4f} -> holdout {h['d']:+.4f}"
          f"   (holdout/test = {h['d']/t['d']:.2f}, holdout/val = {h['d']/v['d']:.2f})")

    # 逐指数
    print('\n逐指数(holdout):')
    per = []
    for jg in np.unique(d['j'].values):
        m = hd & (d['j'].values == jg)
        if m.sum() < 200:
            continue
        per.append(dict(code=codes[jg], n=int(m.sum()),
                        d=HB.r2(yA, full, m) - HB.r2(yA, base, m)))
    pdf = pd.DataFrame(per); q = pdf['d']
    print(f"  n={len(pdf)}  为正 {int((q>0).sum())}/{len(pdf)} ({(q>0).mean()*100:.1f}%)  "
          f"中位 {q.median():+.4f}  p10 {q.quantile(.1):+.4f}  最差 {q.min():+.4f}")
    big = {'000016.XSHG': '上证50', '000300.XSHG': '沪深300',
           '000905.XSHG': '中证500', '000852.XSHG': '中证1000'}
    s = ' | '.join(f"{n} {pdf[pdf.code==c]['d'].iloc[0]:+.4f}"
                   for c, n in big.items() if (pdf.code == c).any())
    print(f'  四宽基: {s}')

    # 极端波动分位
    print('\n极端波动分位(holdout, PIT 同 bar 分位):')
    pct_tab = pit_pct_same_bar(close, HB.build_features(close)['rv12'])
    pct = pct_tab[d['t'].values, d['jloc'].values]
    okp = np.isfinite(pct)
    for lo, hi in [(0, .5), (.5, .9), (.9, .95), (.95, .99), (.99, 1.0)]:
        m = hd & okp & (pct >= lo) & (pct < hi + (1e-9 if hi == 1.0 else 0))
        if m.sum() < 200:
            continue
        r0, r1 = HB.r2(yA, base, m), HB.r2(yA, full, m)
        mse0 = ((yA - base) ** 2)[m].mean(); mse1 = ((yA - full) ** 2)[m].mean()
        print(f'  {int(lo*100):>3d}-{int(hi*100):>3d}%  ΔR²={r1-r0:+.4f}  '
              f'MSE 下降 {(1-mse1/mse0)*100:.1f}%  n={m.sum():,}')

    # 稳健性: 目标 C
    both = np.isfinite(yC) & okC
    yCf = np.where(both, yC, 0.0)
    segsC = {k: v & both for k, v in segs.items()}
    baseC = HB.ridge_fit(Xb, yCf, tr & both)
    prC = ridge_path(Z, yCf - baseC, tr & both, LAMS)
    lamC = max(LAMS, key=lambda L: HB.r2(yCf, baseC + prC[L], va & both))
    rC = report(f'稳健性 目标 C = 子采样 5min RV (无偏+低噪, λ={lamC:.0f}) —— 非头条',
                yCf, baseC, baseC + prC[lamC], d, segsC, tr & both, dts)
    print(f"\n  C 口径衰减链: val {rC['val']['d']:+.4f} -> test {rC['TEST']['d']:+.4f} "
          f"-> holdout {rC['HOLDOUT']['d']:+.4f}")

    print('\n' + '=' * 96)
    print('★ holdout 已开封。此后本项目转为**前向累计**, holdout 不得再用于任何调参。')
    print('=' * 96)


if __name__ == '__main__':
    main()
