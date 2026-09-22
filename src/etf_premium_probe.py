"""ETF 折价溢价对指数日内已实现波动的增量 —— 预注册见 docs/PREREG_ETF_PREMIUM.md

★核心设计: gap-ahead。指数价格因成分股不同步成交而滞后(1min 收益 ro1=+0.266),
  ETF 不滞后 ⇒ (ETF价 − 指数价) 很大程度上是"指数还没走完的那一段",
  而指数未来的 RV 机械地包含这段补涨。跳过 1~2 根 bar 再测, 才能分开机械与信息。

★信息集对齐: ETF 特征拆两组按序加 —— ①ETF 自身(RV/成交额/价差) ②折价溢价(水平/波动/|pd|/变化)。
  净效果 = ②加在 (B8+B9+①) 之上。

只用 train 拟合 / val 评估。test、holdout 一律不读。
用法: python etf_premium_probe.py
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB
import har_1m_features as H1
import extract_universe as U
from run_layerscan import ridge_path, LAMS
from dm_test_v2 import cw_stat, dm_stat

OUT = f'{_R}/out'
TRAIN_END, VAL_END, TEST_END = '2023-04-27', '2024-06-07', '2025-07-17'
SEAL_OPEN = os.environ.get('ETF_OPEN_TEST') == '1'   # ★只有显式置位才读 test
H = 6                                   # 30min
GAPS = [0, 1, 2]                        # 跳过几根 bar
PAIR = {'510300.XSHG': '000300.XSHG', '510500.XSHG': '000905.XSHG',
        '512100.XSHG': '000852.XSHG'}
ANCHOR_D = 20                           # 折价溢价锚: 过去 20 日收盘比值中位数


def build_panel(jsel, dates, codes):
    """与 open_test.build 同口径, 但① 不做 stride 过滤(要功效) ② 造三个 gap 目标。"""
    arr, _, _ = U.load_grid()
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
    del G, arr

    lr = HB.log_returns(close)
    NG = close.shape[0]; bar = np.arange(NG) % HB.NBAR
    r2 = lr ** 2
    fin = np.isfinite(lr).astype(float)
    cs = np.vstack([np.zeros((1, close.shape[1])), np.cumsum(np.nan_to_num(r2), 0)])
    cn = np.vstack([np.zeros((1, close.shape[1])), np.cumsum(fin, 0)])
    Y = {}
    maxg = max(GAPS)
    for g in GAPS:                       # 目标 = bar t+1+g .. t+H+g 的 RV
        i0, i1 = np.arange(NG) + 1 + g, np.arange(NG) + 1 + g + H
        okb = bar <= HB.NBAR - 1 - H - g
        i0c, i1c = np.clip(i0, 0, NG), np.clip(i1, 0, NG)
        s = np.where(okb[:, None], cs[i1c] - cs[i0c], np.nan)
        n = np.where(okb[:, None], cn[i1c] - cn[i0c], 0.0)
        v = np.where((n >= H * 0.5) & np.isfinite(s) & (s > 0), s, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            Y[g] = np.log(v + HB.EPS)

    need = (['rv12', 'rv48', 'rv240', 'sqrt_rq', 'rs_up', 'rs_dn', 'aret', 'ewma']
            + [f'lag{k}' for k in range(1, HB.NLAG + 1)] + HB.B8_EXTRA)
    ok = np.ones_like(close, bool)
    for g in GAPS:
        ok &= np.isfinite(Y[g])
    for k in need:
        ok &= np.isfinite(feats[k])
    ok &= (bar[:, None] <= HB.NBAR - 1 - H - maxg)
    ti, jj = np.where(ok)
    d = pd.DataFrame({'t': ti, 'jloc': jj})
    d['j'] = jsel[jj]
    d['date'] = np.asarray(dates)[ti // HB.NBAR]
    d['bar'] = ti % HB.NBAR
    d['code'] = [codes[x] for x in d['j']]
    F = {k: feats[k][ti, jj] for k in need}
    Ys = {g: Y[g][ti, jj] for g in GAPS}
    return d, Ys, F


USE_IOPV = os.path.exists(f'{OUT}/etf_iopv5m.parquet')


def etf_features(d, dates, close_idx):
    """由 ETF 5min 网格造 ①ETF 自身 与 ②折价溢价 两组特征。

    ★USE_IOPV=True 时用 CSMAR 的**真实 IOPV**((LastPrice/IOPV−1)×1e4, 交易所口径),
      不需要自造锚; 否则退回比值代理(锚=过去20日收盘比值中位数)。
      实测真实折溢价比代理窄 3~5 倍(510300: p1 −44bp/p99 +21bp vs 代理 −153/+102)。
    """
    e = pd.read_parquet(f'{OUT}/etf_iopv5m.parquet' if USE_IOPV else f'{OUT}/etf5m.parquet')
    if USE_IOPV:
        e = e.rename(columns={'close': 'mid'})
    e['iname'] = e.code.map(PAIR)
    di = pd.Series(np.arange(len(dates)), index=dates)
    e['t'] = di.reindex(e.date).values * HB.NBAR + e.bar.values
    e = e.dropna(subset=['t'])
    e['t'] = e['t'].astype(int)

    jloc_of = {c: k for k, c in enumerate(sorted(set(d.code)))}
    NG, J = close_idx.shape
    keys = ['mid', 'money', 'spr_bp', 'dep'] + (['prem_bp', 'prem_sd_bp'] if USE_IOPV else [])
    grid = {c: np.full((NG,), np.nan) for c in keys}
    P = {jl: {k: np.full(NG, np.nan) for k in grid} for jl in range(J)}
    codes_sorted = sorted(set(d.code))
    for etf, idx in PAIR.items():
        if idx not in jloc_of:
            continue
        jl = jloc_of[idx]
        s = e[e.code == etf]
        for k in grid:
            P[jl][k][s['t'].values] = s[k].values

    out = {}
    for jl in range(J):
        mid = P[jl]['mid']
        lm = np.log(np.where(mid > 0, mid, np.nan))
        li = np.log(np.where(close_idx[:, jl] > 0, close_idx[:, jl], np.nan))
        ratio = lm - li
        if USE_IOPV:
            pd_bp = P[jl]['prem_bp']                        # ★交易所口径真实折溢价 bp
        else:
            # 代理: 锚 = 过去 ANCHOR_D 日收盘比值中位数, 严格用 t 之前的日子
            eod = ratio[np.arange(NG) % HB.NBAR == HB.NBAR - 1]
            anc_d = pd.Series(eod).rolling(ANCHOR_D, min_periods=5).median().shift(1).values
            pd_bp = (ratio - np.repeat(anc_d, HB.NBAR)[:NG]) * 1e4
        # 组① ETF 自身
        r = np.full(NG, np.nan); r[1:] = lm[1:] - lm[:-1]
        r[np.arange(NG) % HB.NBAR == 0] = np.nan
        rs = pd.Series(np.nan_to_num(r ** 2)); ns = pd.Series(np.isfinite(r).astype(float))
        for w in (12, 48, 240):
            out.setdefault(f'etf_rv{w}', np.full((NG, J), np.nan))[:, jl] = \
                HB._safe_log(rs.rolling(w, min_periods=w // 2).sum().values
                             / np.maximum(ns.rolling(w, min_periods=w // 2).sum().values, 1))
            out.setdefault(f'etf_amt{w}', np.full((NG, J), np.nan))[:, jl] = \
                np.log(np.maximum(pd.Series(P[jl]['money']).rolling(w, min_periods=w // 2).sum().values, 1.0))
        for w in (12, 48):
            out.setdefault(f'etf_spr{w}', np.full((NG, J), np.nan))[:, jl] = \
                pd.Series(P[jl]['spr_bp']).rolling(w, min_periods=w // 2).mean().values
        out.setdefault('etf_dep12', np.full((NG, J), np.nan))[:, jl] = \
            pd.Series(P[jl]['dep']).rolling(12, min_periods=6).mean().values
        # 组② 折价溢价
        S = pd.Series(pd_bp)
        out.setdefault('pd_now', np.full((NG, J), np.nan))[:, jl] = pd_bp
        for w in (12, 48, 240):
            out.setdefault(f'pd_mean{w}', np.full((NG, J), np.nan))[:, jl] = S.rolling(w, min_periods=w // 2).mean().values
            out.setdefault(f'pd_std{w}', np.full((NG, J), np.nan))[:, jl] = \
                np.log(S.rolling(w, min_periods=w // 2).std().values + 1e-3)     # ★pd 的波动
        for w in (12, 48):
            out.setdefault(f'pd_abs{w}', np.full((NG, J), np.nan))[:, jl] = \
                np.log(S.abs().rolling(w, min_periods=w // 2).mean().values + 1e-3)
        out.setdefault('pd_chg12', np.full((NG, J), np.nan))[:, jl] = (S - S.shift(12)).values
        if USE_IOPV:   # ★bar 内折溢价的波动 —— 代理口径给不出这个量
            W = pd.Series(P[jl]['prem_sd_bp'])
            for w in (1, 12, 48):
                out.setdefault(f'pd_wsd{w}', np.full((NG, J), np.nan))[:, jl] = \
                    np.log(W.rolling(w, min_periods=max(1, w // 2)).mean().values + 1e-3)
    return out, codes_sorted


G1 = ([f'etf_rv{w}' for w in (12, 48, 240)] + [f'etf_amt{w}' for w in (12, 48, 240)]
      + [f'etf_spr{w}' for w in (12, 48)] + ['etf_dep12'])
G2_LVL = ['pd_now'] + [f'pd_mean{w}' for w in (12, 48, 240)] + ['pd_chg12']
G2_VOL = ([f'pd_std{w}' for w in (12, 48, 240)] + [f'pd_abs{w}' for w in (12, 48)]
          + ([f'pd_wsd{w}' for w in (1, 12, 48)] if os.path.exists(f'{OUT}/etf_iopv5m.parquet') else []))
G2 = G2_LVL + G2_VOL


def r2(y, yh, m):
    return 1 - ((y[m] - yh[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


def main():
    cur = json.load(open(f'{OUT}/index_universe_curated.json'))
    _, dates, allc = U.load_grid()
    want = set(PAIR.values())
    jsel = np.array([i for i, c in enumerate(allc) if c in want])
    print(f'标的 {len(jsel)} 个指数: {[allc[j] for j in jsel]}', flush=True)
    d, Ys, F = build_panel(jsel, np.asarray(dates), allc)

    # B9(1min 派生)
    m1 = json.load(open(f'{OUT}/index1m_grid_140_meta.json'))
    a1m = np.load(f'{OUT}/index1m_grid_140.npy', mmap_mode='r')
    loc = {c: i for i, c in enumerate(m1['codes'])}
    order = np.array([loc[allc[j]] for j in jsel])
    a1 = np.asarray(a1m[:, order, :])
    for k, v in H1.build_1m_features(a1).items():
        F[k] = v[d['t'].values, d['jloc'].values]
    del a1m, a1

    arr, _, _ = U.load_grid()
    close_idx = np.ascontiguousarray(arr[:, jsel, 3]).astype(np.float64)
    del arr
    EF, _ = etf_features(d, np.asarray(dates), close_idx)
    for k, v in EF.items():
        F[k] = v[d['t'].values, d['jloc'].values]

    good = np.ones(len(d), bool)
    for k in G1 + G2:
        good &= np.isfinite(F[k])
    for g in GAPS:
        good &= np.isfinite(Ys[g])
    d = d[good].reset_index(drop=True)
    F = {k: v[good] for k, v in F.items()}
    Ys = {g: v[good] for g, v in Ys.items()}
    d['seg'] = np.where(d.date <= TRAIN_END, 'train',
               np.where(d.date <= VAL_END, 'val',
               np.where(d.date <= TEST_END, 'TEST', 'SEALED')))
    tr = (d.seg == 'train').values; va = (d.seg == 'val').values
    te = (d.seg == 'TEST').values & SEAL_OPEN
    print(f'面板 {len(d):,} 点 | train {tr.sum():,} ({d.date[tr].min()}~{d.date[tr].max()}) '
          f'/ val {va.sum():,} ({d.date[va].min()}~{d.date[va].max()}) '
          f'/ 封存 {(~tr & ~va).sum():,}', flush=True)
    if SEAL_OPEN:
        nd = pd.Series(d.date[te]).nunique()
        print(f'★TEST 开封: {te.sum():,} 点 / {nd} 天 ({d.date[te].min()}~{d.date[te].max()})')
        print(f'  ★功效警告: ETF tick 止于 2024-10-15, test 应有约 270 天, 实得 {nd} 天 '
              f'({nd/270*100:.0f}%); val 上效应已只有 CW t≈1.7, 本段功效更低, '
              f'"测不出"与"确实为零"在此难以区分。', flush=True)
    print(f'折价溢价口径: {"★CSMAR 真实 IOPV" if USE_IOPV else "比值代理"}')
    print(f'折价溢价(bp) 分位: ' + '  '.join(
        f'p{q}={np.nanpercentile(F["pd_now"], q):+.1f}' for q in (1, 25, 50, 75, 99)), flush=True)

    Xb0 = np.column_stack([HB.baseline_design(d, F, len(jsel), 'B8')]
                          + [F[c] for c in H1.B9_EXTRA])
    sets = {'B8+B9 (基线)': Xb0,
            '+①ETF自身': np.column_stack([Xb0] + [F[c] for c in G1]),
            '+①+②pd水平': np.column_stack([Xb0] + [F[c] for c in G1 + G2_LVL]),
            '+①+②pd波动': np.column_stack([Xb0] + [F[c] for c in G1 + G2_VOL]),
            '+①+②全部': np.column_stack([Xb0] + [F[c] for c in G1 + G2])}
    sets = {k: np.nan_to_num(v, nan=0., posinf=0., neginf=0.) for k, v in sets.items()}
    dts = d['date'].values

    print('\n' + '=' * 92)
    print('val 段 R² / ΔR²  (拟合只用 train)   ★主判据 = gap1 上 ②的净增量 > +0.002 且 CW t > 3')
    print('=' * 92)
    hdr = f"{'模型':<16s}" + ''.join(f"{'gap'+str(g):>11s}{'':>9s}" for g in GAPS)
    print(f"{'模型':<16s}" + ''.join(f"{f'gap{g} R²':>11s}{f'gap{g} Δ净':>10s}" for g in GAPS))
    print('-' * 92)
    res = {}
    for nm, X in sets.items():
        row = f'{nm:<16s}'
        for g in GAPS:
            y = Ys[g]
            yh = HB.ridge_fit(X, y, tr)
            rv = r2(y, yh, va)
            res[(nm, g)] = (rv, yh)
            base_nm = '+①ETF自身' if nm.startswith('+①+②') else 'B8+B9 (基线)'
            dnet = rv - res[(base_nm, g)][0] if (base_nm, g) in res else np.nan
            row += f'{rv:>11.4f}' + (f'{dnet:>+10.4f}' if np.isfinite(dnet) else f'{"":>10s}')
        print(row, flush=True)
    print('-' * 92)
    print('注: ①的Δ相对基线; ②各行的Δ相对「基线+①」= 折价溢价的**净**增量')

    print('\n显著性(CW t, 误差按天聚合) —— ②全部 vs 基线+①:')
    for g in GAPS:
        y = Ys[g]
        f1 = res[('+①ETF自身', g)][1]; f2 = res[('+①+②全部', g)][1]
        t_cw, nd = cw_stat(y[va], f1[va], f2[va], dts[va])
        t_dm, _ = dm_stat(((y - f1) ** 2)[va], ((y - f2) ** 2)[va], dts[va])
        print(f'  gap{g}:  ΔR²={res[("+①+②全部",g)][0]-res[("+①ETF自身",g)][0]:+.4f}  '
              f'CW t={t_cw:.2f}  DM t={t_dm:.2f}  n天={nd}')

    print('\n逐指数(gap1, ②全部 vs 基线+①):')
    g = 1; y = Ys[g]
    f1 = res[('+①ETF自身', g)][1]; f2 = res[('+①+②全部', g)][1]
    for c in sorted(set(d.code)):
        m = va & (d.code.values == c)
        print(f'  {c}: n={m.sum():>6d}  基线+① {r2(y,f1,m):.4f} -> {r2(y,f2,m):.4f}  '
              f'Δ={r2(y,f2,m)-r2(y,f1,m):+.4f}')

    print('\n安慰剂(pd 特征在同一(指数,bar)内跨日打乱, gap1):')
    rng = np.random.default_rng(0)
    Fp = dict(F)
    key = d['code'].astype(str) + '_' + d['bar'].astype(str)
    dfp = pd.DataFrame({k: F[k] for k in G2}); dfp['k'] = key.values
    sh = dfp.groupby('k', sort=False)[G2].transform(lambda s: s.sample(frac=1, random_state=0).values)
    Xp = np.nan_to_num(np.column_stack([Xb0] + [F[c] for c in G1] + [sh[c].values for c in G2]),
                       nan=0., posinf=0., neginf=0.)
    yp = HB.ridge_fit(Xp, Ys[1], tr)
    print(f'  ΔR² = {r2(Ys[1],yp,va)-res[("+①ETF自身",1)][0]:+.4f}   (应塌到 0)')
    if SEAL_OPEN:
        print('\n' + '=' * 92)
        print('★★ TEST 段(一次性开封) —— 判据同 val: ②净增量 > +0.002 且 CW t > 3')
        print('=' * 92)
        print(f"{'模型':<16s}" + ''.join(f"{f'gap{g} R²':>11s}{f'gap{g} Δ净':>10s}" for g in GAPS))
        print('-' * 92)
        rt = {}
        for nm, X in sets.items():
            row = f'{nm:<16s}'
            for g in GAPS:
                y = Ys[g]; yh = HB.ridge_fit(X, y, tr)
                rv = r2(y, yh, te); rt[(nm, g)] = (rv, yh)
                bn = '+①ETF自身' if nm.startswith('+①+②') else 'B8+B9 (基线)'
                dn = rv - rt[(bn, g)][0] if (bn, g) in rt else np.nan
                row += f'{rv:>11.4f}' + (f'{dn:>+10.4f}' if np.isfinite(dn) else f'{"":>10s}')
            print(row, flush=True)
        print('-' * 92)
        for g in GAPS:
            y = Ys[g]; f1 = rt[('+①ETF自身', g)][1]; f2 = rt[('+①+②全部', g)][1]
            t_cw, nd2 = cw_stat(y[te], f1[te], f2[te], dts[te])
            print(f'  gap{g}: ΔR²={rt[("+①+②全部",g)][0]-rt[("+①ETF自身",g)][0]:+.4f}  '
                  f'CW t={t_cw:.2f}  n天={nd2}')
        v = rt[('+①+②全部', 1)][0] - rt[('+①ETF自身', 1)][0]
        print(f'\n★主判据 gap1: ΔR²={v:+.4f} (需>+0.002)  => '
              + ('通过' if v > 0.002 else '未通过, 确认判负'))
    print('\n★ holdout 未读。')


if __name__ == '__main__':
    main()
