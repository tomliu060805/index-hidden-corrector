"""★★★ 测试段一次性开封 —— 预注册见 docs/PREREG_TEST_OPENING.md

开什么: test = 2024-06-08 ~ 2025-07-17
不开:   holdout = 2025-07-18 之后 (留作最后一次独立确认)

唯一配置(不扫参、不选优): 5min×L=128 输入 / L8n-last / H=6 / 基线 B8+B9 / 140 策展指数
拟合只用 train, λ 只在 val 上选, **test 只用于评估**。

主判据: test ΔR² > 0 且 CW t > 3。其余全部量只报告、不作判据。

★这个脚本只应该跑一次。跑完 test 段不得再用于任何调参。
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, json, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB
import har_1m_features as H1
import extract_universe as U
from run_universe_eval import load_hidden
from run_layerscan import ridge_path, LAMS, pit_pct_same_bar
from dm_test_v2 import dm_stat, cw_stat, qlike_loss

OUT = f'{_R}/out'
TRAIN_END, VAL_END, TEST_END = '2023-04-27', '2024-06-07', '2025-07-17'
H = 6


def build(jsel, kind='rv'):
    """与 run_universe_eval.build_panel_u 同口径, 但 seg 区分 test 并把 holdout 标成 SEALED。"""
    arr, dates, codes = U.load_grid()
    G = {nm: np.ascontiguousarray(arr[:, jsel, k]).astype(np.float64)
         for k, nm in enumerate(U.COLS)}
    close = G['close']
    feats = HB.build_features(close)
    feats['ewma'] = HB.ewma_rv(close, 0.94)
    ex = HB.build_ohlcv_features(G['open'], G['high'], G['low'], G['close'],
                                 G['volume'], G['money'])
    ex.pop('_dow_raw', None)
    ex.update(HB.build_calendar(dates, close.shape[0], close.shape[1]))
    feats.update(ex)
    del G
    ytab = HB.future_target(close, H, kind)
    # 退化点过滤(与 build_panel_u 同, 2026-09-17 修)
    lr_ = HB.log_returns(close); NGd = close.shape[0]; bard = np.arange(NGd) % HB.NBAR
    degen = np.zeros_like(close, bool)
    for i in range(NGd - H):
        if bard[i] > HB.NBAR - 1 - H:
            continue
        seg = lr_[i + 1:i + 1 + H]
        fin = np.isfinite(seg).any(0)
        with np.errstate(invalid='ignore'):
            v = np.nansum(seg ** 2, 0)
        degen[i] = fin & (np.nan_to_num(v, nan=1.0) <= 0)
    need = (['rv12', 'rv48', 'rv240', 'sqrt_rq', 'rs_up', 'rs_dn', 'aret', 'ewma']
            + [f'lag{k}' for k in range(1, HB.NLAG + 1)] + HB.B8_EXTRA)
    ok = np.isfinite(ytab) & ~degen
    for k in need:
        ok &= np.isfinite(feats[k])
    bar = np.arange(NGd) % HB.NBAR
    ok &= (bar[:, None] <= HB.NBAR - 1 - H) & (bar[:, None] % U.STRIDE == 0)
    ti, jj = np.where(ok)
    d = pd.DataFrame({'t': ti, 'jloc': jj})
    d['j'] = jsel[jj]
    d['date'] = np.asarray(dates)[ti // HB.NBAR]
    d['bar'] = ti % HB.NBAR
    d['seg'] = np.where(d.date <= TRAIN_END, 'train',
               np.where(d.date <= VAL_END, 'val',
               np.where(d.date <= TEST_END, 'TEST', 'SEALED')))
    return d, ytab[ti, jj], {k: feats[k][ti, jj] for k in need}, np.asarray(codes), close


def main():
    print('=' * 78)
    print('★★★ 测试段一次性开封  (holdout 仍然封存, 本次不碰)')
    print('    预注册: docs/PREREG_TEST_OPENING.md')
    print('    主判据: test ΔR² > 0 且 CW t > 3')
    print('=' * 78, flush=True)
    cur = json.load(open(f'{OUT}/index_universe_curated.json'))
    cu0 = pd.read_csv(f'{OUT}/universe_curated_list.csv')
    bs = set(cu0[cu0['名称'].astype(str).str.contains('B股|B指')]['代码'])
    want = set(cur['codes']) - bs
    _, _, allc = U.load_grid()
    jsel = np.array([i for i, c in enumerate(allc) if c in want])
    Hm, key = load_hidden()
    d, y, F, codes, close = build(jsel)

    m1 = json.load(open(f'{OUT}/index1m_grid_140_meta.json'))
    arr1m = np.load(f'{OUT}/index1m_grid_140.npy', mmap_mode='r')
    loc = {c: i for i, c in enumerate(m1['codes'])}
    order = np.array([loc[allc[j]] for j in jsel])
    for k, v in H1.build_1m_features(np.asarray(arr1m[:, order, :])).items():
        F[k] = v[d['t'].values, d['jloc'].values]
    del arr1m

    kk = d['t'].values * 1000 + d['j'].values
    p = np.clip(np.searchsorted(key, kk), 0, len(key) - 1); good = key[p] == kk
    d, y = d[good].reset_index(drop=True), y[good]
    F = {k: v[good] for k, v in F.items()}
    Z = Hm[p[good], 1, :].astype(np.float32)     # L8n / last
    tr = (d.seg == 'train').values
    va = (d.seg == 'val').values
    te = (d.seg == 'TEST').values
    dts = d['date'].values
    print(f'\n面板 {len(d):,} 点 | train {tr.sum():,} / val {va.sum():,} / '
          f'**TEST {te.sum():,}** / 封存(holdout) {(d.seg=="SEALED").sum():,}', flush=True)
    print(f'test 区间 {dts[te].min()} ~ {dts[te].max()}  ({pd.Series(dts[te]).nunique()} 天)',
          flush=True)

    Xb = np.column_stack([HB.baseline_design(d, F, len(jsel), 'B8')]
                         + [F[c] for c in H1.B9_EXTRA])
    Xb = np.nan_to_num(Xb, nan=0., posinf=0., neginf=0.)
    base = HB.ridge_fit(Xb, y, tr)                       # ★只用 train 拟合
    pr = ridge_path(Z, y - base, tr, LAMS)
    lam = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va))   # ★λ 只在 val 上选
    full = base + pr[lam]
    print(f'λ = {lam:.0f} (在 val 上选定)', flush=True)

    q0, q1 = qlike_loss(y, base, tr), qlike_loss(y, full, tr)
    e0, e1 = (y - base) ** 2, (y - full) ** 2
    rows = []
    print(f"\n{'段':>6s}{'基线R²':>9s}{'+隐层R²':>10s}{'ΔR²':>9s}{'DM t':>8s}"
          f"{'★CW t':>8s}{'QLIKE%':>9s}{'Q t':>7s}{'逐日为正':>9s}{'n天':>7s}")
    for nm, m in [('train', tr), ('val', va), ('TEST', te)]:
        r0, r1 = HB.r2(y, base, m), HB.r2(y, full, m)
        tdm, nd = dm_stat(e0[m], e1[m], dts[m])
        tcw, _ = cw_stat(y[m], base[m], full[m], dts[m])
        tq, _ = dm_stat(q0[m], q1[m], dts[m])
        dd = pd.DataFrame({'d': (q0 - q1)[m], 'date': dts[m]}).groupby('date')['d'].mean().values
        rows.append(dict(seg=nm, n=int(m.sum()), base=r0, full=r1, d=r1 - r0,
                         dm_t=tdm, cw_t=tcw, qlike=(1 - q1[m].mean() / q0[m].mean()) * 100,
                         q_t=tq, q_pos=(dd > 0).mean() * 100, n_days=nd))
        print(f'{nm:>6s}{r0:>9.4f}{r1:>10.4f}{r1-r0:>+9.4f}{tdm:>8.2f}{tcw:>8.2f}'
              f'{(1-q1[m].mean()/q0[m].mean())*100:>8.1f}%{tq:>7.2f}'
              f'{(dd>0).mean()*100:>8.1f}%{nd:>7d}', flush=True)

    t = [r for r in rows if r['seg'] == 'TEST'][0]
    v = [r for r in rows if r['seg'] == 'val'][0]
    print(f"\n{'='*78}\n★ 主判据: test ΔR² = {t['d']:+.4f} (>0?) , CW t = {t['cw_t']:.2f} (>3?)"
          f"  =>  {'**通过**' if (t['d'] > 0 and t['cw_t'] > 3) else '**未通过**'}")
    print(f"  test/val 增量比 = {t['d']/max(v['d'],1e-9):.2f}"
          f"   (预注册预期: test ΔR² 落在 [+0.010,+0.020] = 无衰减)")
    print('=' * 78, flush=True)

    # ---- 以下只报告, 不作判据 ----
    print('\n--- 逐指数分布 (test) ---')
    per = []
    for jg in np.unique(d['j'].values):
        m = te & (d['j'].values == jg)
        if m.sum() < 300:
            continue
        per.append(dict(code=codes[jg], n=int(m.sum()), base=HB.r2(y, base, m),
                        d=HB.r2(y, full, m) - HB.r2(y, base, m)))
    pdf = pd.DataFrame(per); q = pdf.d
    print(f'  n={len(pdf)}  为正 {int((q>0).sum())}/{len(pdf)} ({(q>0).mean()*100:.1f}%)  '
          f'中位 {q.median():+.4f}  p10 {q.quantile(.1):+.4f}  p90 {q.quantile(.9):+.4f}')
    KEY = {'000016.XSHG': '上证50', '000300.XSHG': '沪深300',
           '000905.XSHG': '中证500', '000852.XSHG': '中证1000'}
    print('  四个主力宽基:')
    for cd, nmz in KEY.items():
        rr = pdf[pdf.code == cd]
        if len(rr):
            rr = rr.iloc[0]
            print(f'    {cd} {nmz:<9s} 基线{rr.base:.4f}  Δ{rr.d:+.4f}  ({rr.d/rr.base*100:+.1f}%)')

    print('\n--- 分侧诊断 (test) ---')
    err = y - base
    print(f"  {'情形':<20s}{'占比':>7s}{'基线MSE':>10s}{'MSE下降%':>10s}{'QLIKE下降%':>11s}")
    for lab, m in [('基线低估(实际>预测)', te & (err > 0)), ('基线高估(实际<预测)', te & (err <= 0))]:
        print(f'  {lab:<20s}{m.sum()/te.sum()*100:>6.1f}%{e0[m].mean():>10.4f}'
              f'{(1-e1[m].mean()/e0[m].mean())*100:>9.1f}%{(1-q1[m].mean()/q0[m].mean())*100:>10.1f}%')
    thr = np.quantile(err[te], 0.9); m = te & (err > thr)
    print(f'  {"★严重低估(前10%)":<20s}{10.0:>6.1f}%{e0[m].mean():>10.4f}'
          f'{(1-e1[m].mean()/e0[m].mean())*100:>9.1f}%{(1-q1[m].mean()/q0[m].mean())*100:>10.1f}%')
    print(f'  基线校准: 高估占比 {(err<=0)[te].mean()*100:.1f}% (50% = 无偏)')

    print('\n--- 极端波动分位 (test, PIT 同 bar 分位) ---')
    pct_tab = pit_pct_same_bar(close, HB.build_features(close)['rv12'])
    pct = pct_tab[d['t'].values, d['jloc'].values]
    okp = np.isfinite(pct)
    print(f"  {'分位桶':<12s}{'n':>9s}{'基线R²':>9s}{'ΔR²':>9s}{'MSE下降%':>10s}")
    for a_, b_ in zip([0, .5, .8, .9, .95, .99], [.5, .8, .9, .95, .99, 1.01]):
        m = te & okp & (pct >= a_) & (pct < b_)
        if m.sum() < 500:
            continue
        print(f'  {a_*100:.0f}-{min(b_,1)*100:.0f}%{"":<6s}{m.sum():>9,}{HB.r2(y,base,m):>9.4f}'
              f'{HB.r2(y,full,m)-HB.r2(y,base,m):>+9.4f}'
              f'{(1-e1[m].mean()/e0[m].mean())*100:>9.1f}%')

    pd.DataFrame(rows).to_csv(f'{OUT}/TEST_OPENED_summary.csv', index=False)
    pdf.to_csv(f'{OUT}/TEST_OPENED_perindex.csv', index=False)
    print(f'\n写出 {OUT}/TEST_OPENED_summary.csv 与 TEST_OPENED_perindex.csv')
    print('★ holdout (2025-07-18 之后) 未计算、未打印 —— 仍然封存。')


if __name__ == '__main__':
    main()
