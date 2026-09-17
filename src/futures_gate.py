"""增量实验1+2: 暴拉保护(趋势门) × 预报期限(1h/2h) —— IC/IM 加仓覆盖层, 成本C平今现实

门: w = 1 + (w_lev−1)·1[mom_k(昨收) > 0], k∈{5,20,60}日; 'none'=无门
预报: preds npz (1h: bar0..35 / 2h: bar0..23), 仓位 w_lev=clip(tgt/pv2, 1, 3)
δ 无交易带 ∈ {0.15, 0.3}; 全网格 train+val 净Sharpe 选定, test/holdout 冻结
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import numpy as np, pandas as pd

B = f'{_R}'
PROD = {'IC': 1, 'IM': 2}
CAP, BARS_Y = 3.0, 242.0
fut = pd.read_parquet(f'{B}/out/futures_dom5m.parquet')
idx5 = pd.read_parquet(f'{B}/out/index5m.parquet')
CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']

seg_of = lambda dt: np.where(dt <= '2023-04-27', 'train', np.where(dt <= '2024-06-07', 'val',
                    np.where(dt <= '2025-07-17', 'test', 'holdout')))

# 指数日收盘 -> 动量 (T-1收盘可知)
eodc = idx5[idx5.bar == 47].pivot(index='date', columns='code', values='close')

def build(prod, src):
    z = np.load(f'{B}/out/{src}')
    j = PROD[prod]
    f = fut[fut['product'] == prod].sort_values(['date', 'bar']).reset_index(drop=True)
    days = np.array(sorted(f.date.unique())); Nd = len(days)
    di = pd.Series(np.arange(Nd), index=days)
    g = di.reindex(f.date).values * 48 + f.bar.values
    r = np.full(Nd * 48, np.nan)
    lc = np.log(f.close.values); ret = f.ret.values.copy()
    b0 = f.bar.values == 0
    ret[b0] = f.ov_ret.values[b0] + (lc[b0] - np.log(f.open.values[b0]))
    r[g] = ret
    roll_day = f.groupby('date')['roll'].first().reindex(days).values.astype(bool)
    mj = z['j'] == j
    pv2_all = np.exp(z['yh0'] + z['p2'])
    fmap = pd.Series(np.arange(mj.sum()), index=pd.MultiIndex.from_arrays([z['date'][mj], z['bar'][mj]]))
    pv2 = np.full(Nd * 48, np.nan)
    dd = np.repeat(days, 48); bb = np.tile(np.arange(48), Nd)
    idx = fmap.reindex(pd.MultiIndex.from_arrays([dd, bb]))
    ok = idx.notna().values
    pv2[ok] = pv2_all[mj][idx.values[ok].astype(int)]
    tgt = np.median(pv2_all[(z['seg'] == 'train') & mj])
    # 动量 (T-1 收盘)
    c = eodc[CODES[j]].reindex(days)
    mom = {k: np.log(c / c.shift(k)).shift(1).values for k in [5, 20, 60]}
    return dict(r=r, pv2=pv2, tgt=tgt, days=days, Nd=Nd, roll_day=roll_day, mom=mom,
                seg_days=seg_of(days))

def make_w(P, gate_k, delta):
    w = np.clip(P['tgt'] / P['pv2'], 1.0, CAP)
    w = pd.Series(w).ffill().fillna(1.0).values
    if gate_k != 'none':
        gd = np.nan_to_num(P['mom'][gate_k]) > 0
        w = 1.0 + (w - 1.0) * np.repeat(gd, 48)
    if delta > 0:
        out = np.empty_like(w); cur = w[0]
        for i in range(len(w)):
            if abs(w[i] - cur) >= delta: cur = w[i]
            out[i] = cur
        w = out
    return w

def run(P, w):
    wl = np.r_[1.0, w[:-1]]
    dw = w - wl
    c = np.where(dw > 0, 0.55e-4 * dw, 3.77e-4 * (-dw))
    c = np.where((np.arange(len(w)) % 48 == 0) & (dw < 0), 0.55e-4 * (-dw), c)
    rd = np.repeat(P['roll_day'], 48) & (np.arange(len(w)) % 48 == 0)
    c += np.where(rd, 2 * 0.55e-4 * wl, 0)
    return (wl * np.nan_to_num(P['r']) - c).reshape(-1, 48).sum(1)

def sh(x): return x.mean() / x.std() * np.sqrt(BARS_Y) if len(x) > 30 and x.std() > 0 else np.nan

def main():
    for prod in ['IC', 'IM']:
        rows, cache = [], {}
        for src, hz in [('paper1_preds.npz', '1h'), ('preds_2h.npz', '2h')]:
            P = build(prod, src)
            sd = P['seg_days']; mtv = np.isin(sd, ['train', 'val'])
            for gk in ['none', 5, 20, 60]:
                for delta in [0.15, 0.3]:
                    dr = run(P, make_w(P, gk, delta))
                    rows.append(dict(hz=hz, gate=gk, delta=delta, tv=sh(dr[mtv]),
                                     test=sh(dr[sd == 'test']), holdout=sh(dr[sd == 'holdout'])))
                    cache[(hz, gk, delta)] = (dr, P)
        t = pd.DataFrame(rows)
        print(f'\n### {prod} 全网格 (成本C, Sharpe; 选择只看 tv=train+val 列)')
        print(t.round(3).to_string(index=False))
        b = t.loc[t.tv.idxmax()]
        print(f'>> train+val 选定: {b.hz} gate={b.gate} δ={b.delta}  -> test={b.test:.3f} holdout={b.holdout:.3f}')
        # BH 对照
        P = cache[(b.hz, b.gate, b.delta)][1]
        sd = P['seg_days']
        bh = run(P, np.ones(P['Nd'] * 48))
        dr = cache[(b.hz, b.gate, b.delta)][0]
        for s in ['test', 'holdout']:
            x, y_ = dr[sd == s], bh[sd == s]
            d = x - y_
            print(f'   {s}: 选定Sharpe {sh(x):.3f} vs BH {sh(y_):.3f}; 差 {d.mean()*BARS_Y*1e4:+.0f}bp/年 t={d.mean()/d.std()*np.sqrt(len(d)):+.2f}')

if __name__ == '__main__':
    main()
