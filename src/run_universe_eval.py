"""全指数(策展后 144 个)上的波动预报增量复查。

★ 只看预报层, 不涉及任何经济量。test/holdout 硬封锁。
★ 报告口径: 逐指数分布 + 池化, 并同时给出**有效独立维数**。
   480 个指数的参与率有效维数只有 3.1, 策展后仍远小于名义个数 ——
   「在 N 个指数上成立」不等于 N 次独立验证, 这一点必须和数字一起报。

用法: python run_universe_eval.py --H 6,12,24 --pos L8n --workers 40
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB
import extract_universe as U
from run_layerscan import ridge_path, LAMS, pit_pct_same_bar
from dm_test_v2 import dm_stat, cw_stat, qlike_loss

OUT = f'{_R}/out'
TRAIN_END, VAL_END = '2023-04-27', '2024-06-07'
SEAL_OK = False
POS = ['L6', 'L8n']


def load_hidden():
    fs = sorted(glob.glob(f'{OUT}/hidden_universe/chunk_*.npz'))
    Hs, Ms = [], []
    for f in fs:
        z = np.load(f)
        Hs.append(z['h']); Ms.append(z['meta'])
    Hm = np.concatenate(Hs); M = np.concatenate(Ms).astype(np.int64)
    key = M[:, 0] * 1000 + M[:, 1]
    o = np.argsort(key)
    print(f'  隐层库 {Hm.shape}  {Hm.nbytes/1e9:.1f}GB  chunks={len(fs)}', flush=True)
    return Hm[o], key[o]


def build_panel_u(H, jsel, kind='rv'):
    """144 指数版面板。逐指数计算 HAR 特征(窗口跨日, nan-skip), 与 3 指数版同口径。"""
    arr, dates, codes = U.load_grid()
    # ★六列都取 —— Kronos 的输入是 [open,high,low,close,volume,money], 基线不能只看 close
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
    # ★★退化点过滤 —— 3 指数版 build_panel 有, 这个 universe 版**曾经漏掉**(2026-09-17 修)。
    #   RV 恰为 0 的点 y=log(EPS)=-27.63, 而正常 y≈-12; 平方损失下单点权重约 400 倍,
    #   train 里 1153 个这样的点占了约 25% 的有效拟合权重 => 系数被拽走, 所有预测带偏。
    lr_ = HB.log_returns(close); NGd = close.shape[0]; bard = np.arange(NGd) % HB.NBAR
    degen = np.zeros_like(close, bool)
    for i in range(NGd - H):
        if bard[i] > HB.NBAR - 1 - H:
            continue
        seg = lr_[i + 1:i + 1 + H]
        fin = np.isfinite(seg).any(0)
        with np.errstate(invalid='ignore'):
            v = np.nansum(seg ** 2, 0) if kind == 'rv' else np.nanmean(np.abs(seg), 0)
        degen[i] = fin & (np.nan_to_num(v, nan=1.0) <= 0)
    need = (['rv12', 'rv48', 'rv240', 'sqrt_rq', 'rs_up', 'rs_dn', 'aret', 'ewma']
            + [f'lag{k}' for k in range(1, HB.NLAG + 1)] + HB.B8_EXTRA)
    NG = close.shape[0]; bar = np.arange(NG) % HB.NBAR
    ok = np.isfinite(ytab) & ~degen
    for k in need:
        ok &= np.isfinite(feats[k])
    ok &= (bar[:, None] <= HB.NBAR - 1 - H)
    ok &= (bar[:, None] % U.STRIDE == 0)                 # 只在提取过的 bar 上评估
    ti, jj = np.where(ok)
    d = pd.DataFrame({'t': ti, 'jloc': jj})
    d['j'] = jsel[jj]                                     # 全局指数索引(与隐层 meta 对齐)
    d['date'] = np.asarray(dates)[ti // HB.NBAR]
    d['bar'] = ti % HB.NBAR
    d['seg'] = np.where(d.date <= TRAIN_END, 'train',
               np.where(d.date <= VAL_END, 'val', 'SEALED'))
    y = ytab[ti, jj]
    F = {k: feats[k][ti, jj] for k in need}
    print(f'  H={H}: {len(d):,} 点  train {int((d.seg=="train").sum()):,} / '
          f'val {int((d.seg=="val").sum()):,}  (封存 {int((d.seg=="SEALED").sum()):,} 不参与)',
          flush=True)
    return d, y, F, np.asarray(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', default='6,12,24')
    ap.add_argument('--pos', default='L8n')
    ap.add_argument('--kind', default='rv')
    ap.add_argument('--base', default='B6', help='B4=HAR-RS, B6=+滞后收益, B7=+逐根平方')
    ap.add_argument('--keep-bshare', action='store_true',
                    help='默认排除 B 股(H=6 上 4/4 为负, 结构性隔离市场)')
    a = ap.parse_args()
    assert not SEAL_OK
    pi = POS.index(a.pos)
    cur = json.load(open(f'{OUT}/index_universe_curated.json'))
    _, _, allcodes = U.load_grid()
    want = set(cur['codes'])
    if not a.keep_bshare:
        # ★B 股是结构性隔离市场(外币计价/投资者群体不同/成交极稀), H=6 上 4/4 为负,
        #   Kronos 预训练语料里大概率没有同类。默认剔除并显式报告。
        cu0 = pd.read_csv(f'{OUT}/universe_curated_list.csv')
        bs = set(cu0[cu0['名称'].astype(str).str.contains('B股|B指')]['代码'])
        drop = want & bs
        want = want - bs
        print(f'★排除 B 股 {len(drop)} 个: {sorted(drop)}', flush=True)
    jsel = np.array([i for i, c in enumerate(allcodes) if c in want])
    print(f"策展 {len(jsel)} 个指数 | 基线={a.base} | {cur['rule']}", flush=True)

    Hm, key = load_hidden()
    rows = []
    for H in [int(x) for x in a.H.split(',')]:
        d, y, F, codes = build_panel_u(H, jsel, a.kind)
        k = d['t'].values * 1000 + d['j'].values
        p = np.clip(np.searchsorted(key, k), 0, len(key) - 1)
        good = key[p] == k
        d, y = d[good].reset_index(drop=True), y[good]
        F = {kk: v[good] for kk, v in F.items()}
        Z = Hm[p[good], pi, :].astype(np.float32)
        tr = (d.seg == 'train').values; va = (d.seg == 'val').values
        print(f'  对齐后 {len(d):,} 点  (train {tr.sum():,} / val {va.sum():,})', flush=True)

        Xb = HB.baseline_design(d, F, len(jsel), a.base)
        base = HB.ridge_fit(Xb, y, tr)
        r2b = HB.r2(y, base, va)
        pr = ridge_path(Z, y - base, tr, LAMS)
        bl = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va))
        yh = base + pr[bl]
        r2t = HB.r2(y, yh, va)
        print(f'  H={H}  池化 B*={r2b:.4f} -> +隐层 {r2t:.4f}  Δ={r2t-r2b:+.4f}  (λ={bl:.0f})',
              flush=True)

        # ★★显著性一律在 140 指数面板上判(用户 2026-09-17 指示)。
        #   DM/CW 先按**天**聚合再做 HAC, 故 n 是天数而非点数 ——
        #   140 个指数不会通过横截面伪重复吹大 t 值, 只是把每天的损失差估得更准。
        dts = d['date'].values
        for sname, mm in [('train', tr), ('val', va)]:
            t_dm, nd = dm_stat(((y - base) ** 2)[mm], ((y - yh) ** 2)[mm], dts[mm])
            t_cw, _ = cw_stat(y[mm], base[mm], yh[mm], dts[mm])
            q0 = qlike_loss(y, base, tr)[mm]; q1 = qlike_loss(y, yh, tr)[mm]
            t_q, _ = dm_stat(q0, q1, dts[mm])
            r0, r1 = HB.r2(y, base, mm), HB.r2(y, yh, mm)
            print(f'    [{sname}] R² {r0:.4f}->{r1:.4f} Δ{r1-r0:+.4f}  '
                  f'DM t={t_dm:.2f}  ★CW t={t_cw:.2f}  '
                  f'QLIKE改善 {(1-q1.mean()/q0.mean())*100:.1f}% (DM t={t_q:.2f})  n天={nd}',
                  flush=True)
            rows.append(dict(H=H, scope=f'sig_{sname}', code='SIG', n=int(mm.sum()),
                             base=r0, full=r1, d_val=r1 - r0, dm_t=t_dm, cw_t=t_cw,
                             qlike_gain_pct=(1 - q1.mean() / q0.mean()) * 100, qlike_dm_t=t_q,
                             n_days=nd))

        # 逐指数
        per = []
        for jg in np.unique(d['j'].values):
            m = va & (d['j'].values == jg)
            if m.sum() < 500:
                continue
            per.append(dict(H=H, code=codes[jg], n=int(m.sum()),
                            base=HB.r2(y, base, m), full=HB.r2(y, yh, m),
                            d_val=HB.r2(y, yh, m) - HB.r2(y, base, m)))
        pdf = pd.DataFrame(per)
        rows.append(dict(H=H, scope='pooled', code='POOLED', n=int(va.sum()),
                         base=r2b, full=r2t, d_val=r2t - r2b, lam=bl))
        for r in per:
            rows.append(dict(scope='per_index', **r))
        q = pdf.d_val
        print(f'    逐指数: n={len(pdf)}  正 {int((q>0).sum())}/{len(pdf)} '
              f'({(q>0).mean()*100:.1f}%)  中位 {q.median():+.4f}  '
              f'均值 {q.mean():+.4f}  p10 {q.quantile(.1):+.4f}  p90 {q.quantile(.9):+.4f}',
              flush=True)
        # ★四个主力宽基单独报
        KEY = {'000016.XSHG': '上证50', '000300.XSHG': '沪深300',
               '000905.XSHG': '中证500', '000852.XSHG': '中证1000'}
        print('    ★四个主力宽基:')
        print(f"      {'代码':<14s}{'名称':<10s}{'n':>8s}{'基线R²':>9s}{'+隐层R²':>10s}{'ΔR²':>9s}{'相对增量':>10s}")
        for cd, nmz in KEY.items():
            rr = pdf[pdf.code == cd]
            if len(rr):
                rr = rr.iloc[0]
                print(f'      {cd:<14s}{nmz:<10s}{int(rr.n):>8,}{rr.base:>9.4f}'
                      f'{rr.full:>10.4f}{rr.d_val:>+9.4f}{rr.d_val/rr.base*100:>9.1f}%')
        worst = pdf.nsmallest(3, 'd_val'); best = pdf.nlargest(3, 'd_val')
        print('    最好三个: ' + ', '.join(f'{r.code} {r.d_val:+.4f}' for r in best.itertuples()))
        print('    最差三个: ' + ', '.join(f'{r.code} {r.d_val:+.4f}' for r in worst.itertuples()))

        # ★★按"过去已实现波动在过去一年同时段的分位"分桶 —— 极端波动下增量是否更大
        try:
            arr_, _, _ = U.load_grid()
            cl_ = np.ascontiguousarray(arr_[:, jsel, 3]).astype(np.float64)
            pct_tab = pit_pct_same_bar(cl_, HB.build_features(cl_)['rv12'])
            pct = pct_tab[d['t'].values, d['jloc'].values]
            # ★决定性零基准: 按天整体打乱隐层后, 同样分桶。若安慰剂在高分位桶也显示
            #   上升的"增量", 说明这个模式是**分桶本身的机械效应**而非信息。
            rng = np.random.default_rng(0)
            ud = np.array(sorted(set(d['date'].values)))
            shuf = dict(zip(ud, rng.permutation(ud)))
            idx_by = {}
            for key_, q in d.groupby(['date', 'jloc']).indices.items():
                q = np.array(q)
                idx_by[key_] = q[np.argsort(d['bar'].values[q])]
            permA = np.arange(len(d))
            for (dt, jl), q in idx_by.items():
                src = idx_by.get((shuf[dt], jl))
                if src is None or len(src) == 0:
                    continue
                permA[q] = src[np.arange(len(q)) % len(src)]
            prp = ridge_path(Z[permA], y - base, tr, LAMS)
            blp = max(LAMS, key=lambda L: HB.r2(y, base + prp[L], va))
            yhp = base + prp[blp]
            e0, e1 = (y - base) ** 2, (y - yh) ** 2
            e2 = (y - yhp) ** 2
            okp = np.isfinite(pct)
            print('    按波动分位分桶(PIT: 过去252天同一bar的滚动分位):')
            print(f"      {'分位桶':<12s}{'n':>10s}{'基线R²':>9s}{'ΔR²':>9s}"
                  f"{'基线MSE':>10s}{'MSE下降%':>10s}{'安慰剂ΔR²':>11s}{'安慰剂MSE降%':>13s}")
            for a_, b_ in zip([0, .5, .7, .8, .9, .95, .99],
                              [.5, .7, .8, .9, .95, .99, 1.01]):
                m = va & okp & (pct >= a_) & (pct < b_)
                if m.sum() < 1000:
                    continue
                r0, r1 = HB.r2(y, base, m), HB.r2(y, yh, m)
                m0, m1 = e0[m].mean(), e1[m].mean()
                lab = f'{a_*100:.0f}-{min(b_,1)*100:.0f}%'
                rp = HB.r2(y, yhp, m); m2 = e2[m].mean()
                rows.append(dict(H=H, scope='regime', code=lab, n=int(m.sum()),
                                 base=r0, full=r1, d_val=r1 - r0,
                                 base_mse=m0, d_mse_pct=(m0 - m1) / m0 * 100,
                                 placebo_d_r2=rp - r0,
                                 placebo_d_mse_pct=(m0 - m2) / m0 * 100))
                print(f'      {lab:<12s}{m.sum():>10,}{r0:>9.4f}{r1-r0:>+9.4f}'
                      f'{m0:>10.4f}{(m0-m1)/m0*100:>9.1f}%'
                      f'{rp-r0:>+11.4f}{(m0-m2)/m0*100:>12.1f}%')
            del cl_, pct_tab
        except Exception as ex:
            print(f'    (分位分桶跳过: {ex})')

        # ★按指数类型分组 —— 扩到 144 个的主要价值: 3 个指数答不了"增量在什么样的
        #   指数上更强"。注意 universe 的有效独立维数远小于名义个数, 故这是**广度**
        #   陈述(各类指数上普遍成立), 不是 N 次独立验证。
        try:
            cu = pd.read_csv(f'{OUT}/universe_curated_list.csv')
            nm = dict(zip(cu['代码'], cu['名称'])); cl = dict(zip(cu['代码'], cu['类']))
            pdf['类'] = pdf['code'].map(cl); pdf['名称'] = pdf['code'].map(nm)
            g = pdf.groupby('类')['d_val'].agg(['size', 'mean', 'median',
                                               lambda x: (x > 0).mean()])
            g.columns = ['个数', '均值Δ', '中位Δ', '为正占比']
            print('    按类型:')
            for k, r in g.iterrows():
                print(f'      {k:<10s} n={int(r.个数):3d}  均值Δ={r.均值Δ:+.4f}  '
                      f'中位Δ={r.中位Δ:+.4f}  为正 {r.为正占比*100:.0f}%')
            # 基线难度 vs 增量 —— 越难预测的指数增量越大?
            rho = pdf[['base', 'd_val']].corr(method='spearman').iloc[0, 1]
            print(f'    corr(基线R², 增量) = {rho:+.3f} (Spearman) '
                  f'{"—— 基线越差增量越大" if rho < -0.2 else ""}')
            pdf.to_csv(f'{OUT}/v2_universe_perindex_H{H}.csv', index=False)
        except Exception as e:
            print(f'    (分组汇总跳过: {e})')
        del d, y, F, Z

    pd.DataFrame(rows).to_csv(f'{OUT}/v2_universe_{a.kind}_{a.pos}.csv', index=False)
    print(f'\n写出 {OUT}/v2_universe_{a.kind}_{a.pos}.csv')


if __name__ == '__main__':
    main()
