"""ETF 真实折溢价 5min 网格 —— 数据源 CSMAR ETF L2 TAQ(带 IOPV)。

★为什么重做: 先前用 L1 tick 源(IHC_ETF_TICK)的**比值代理**(该源无 IOPV), 且止于 2024-10-15
  (只覆盖 test 的 84/270 天)。CSMAR 的 SEL2_TAQ 直接带 IOPV 列, 覆盖 2015-01-05~2026-09-21。
  两个已知局限(代理带噪 / test 截断)因此都可以去掉。

折溢价 = (LastPrice / IOPV − 1) × 1e4  (bp), 交易所口径, 无需自造锚。
输出 out/etf_iopv5m.parquet: date,bar,code,close,prem_bp,prem_sd_bp,spr_bp,dep,money,nt
用法: python build_etf_iopv5m.py --workers 40
"""
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, glob, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

SRC = CFG.ETF_TAQ_DIR
OUT = f'{_R}/out'
SYM = {'510300': '510300.XSHG', '510500': '510500.XSHG', '512100': '512100.XSHG'}
END = '2025-07-17'          # ★holdout(2025-07-18 起)不建, 从源头保证不被读到
COLS = ['Symbol', 'TradingTime', 'LastPrice', 'IOPV', 'TotalVolume', 'TotalAmount',
        'BuyPrice01', 'SellPrice01', 'BuyVolume01', 'SellVolume01']


def one_day(f):
    date = os.path.basename(f)[:10]
    try:
        d = pd.read_parquet(f, columns=COLS)
    except Exception:
        return None
    d['Symbol'] = d.Symbol.astype(str).str.zfill(6)
    d = d[d.Symbol.isin(SYM)]
    if not len(d):
        return None
    t = pd.to_datetime(d.TradingTime)
    m = t.dt.hour * 60 + t.dt.minute
    d = d[(m.between(571, 690)) | (m.between(781, 900))].copy()
    if not len(d):
        return None
    mm = pd.to_datetime(d.TradingTime).dt.hour * 60 + pd.to_datetime(d.TradingTime).dt.minute
    d['bar'] = np.where(mm <= 690, (mm - 571) // 5, 24 + (mm - 781) // 5)
    d = d[(d.bar >= 0) & (d.bar < 48)].sort_values('TradingTime')
    ok = (d.IOPV > 0) & (d.LastPrice > 0)
    d['prem'] = np.where(ok, (d.LastPrice / d.IOPV - 1) * 1e4, np.nan)   # ★真实折溢价 bp
    b1, a1 = d.BuyPrice01.values, d.SellPrice01.values
    mid = np.where((b1 > 0) & (a1 > 0), (a1 + b1) / 2, np.nan)
    d['spr'] = np.where((b1 > 0) & (a1 > 0), (a1 - b1) / np.maximum(mid, 1e-9) * 1e4, np.nan)
    d['dep'] = np.log(np.maximum(d.BuyVolume01 + d.SellVolume01, 1.0))
    g = d.groupby(['Symbol', 'bar'], sort=True)
    o = pd.DataFrame({
        'close':      g['LastPrice'].last(),
        'iopv':       g['IOPV'].last(),
        'prem_bp':    g['prem'].last(),          # bar 末的折溢价
        'prem_mu':    g['prem'].mean(),          # bar 内均值
        'prem_sd_bp': g['prem'].std(),           # ★bar 内折溢价的波动
        'spr_bp':     g['spr'].mean(),
        'dep':        g['dep'].mean(),
        'volume':     g['TotalVolume'].last(),
        'money':      g['TotalAmount'].last(),
        'nt':         g.size(),
    }).reset_index().rename(columns={'Symbol': 'sym'})
    o['code'] = o.sym.map(SYM)
    o['date'] = date
    return o.drop(columns=['sym'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=40)
    a = ap.parse_args()
    fs = [f for f in sorted(glob.glob(f'{SRC}/*.parquet')) if os.path.basename(f)[:10] <= END]
    print(f'{len(fs)} 天 {os.path.basename(fs[0])[:10]} ~ {os.path.basename(fs[-1])[:10]} '
          f'(holdout 从源头排除)', flush=True)
    t0 = time.time(); parts = []
    with Pool(a.workers) as p:
        for k, r in enumerate(p.imap_unordered(one_day, fs, chunksize=2), 1):
            if r is not None:
                parts.append(r)
            if k % 400 == 0:
                print(f'  {k}/{len(fs)}  {time.time()-t0:.0f}s', flush=True)
    df = pd.concat(parts, ignore_index=True).sort_values(['code', 'date', 'bar'])
    for c in ['volume', 'money']:
        v = df.groupby(['code', 'date'])[c].diff()
        first = df.groupby(['code', 'date'])[c].transform('first')
        df[c] = np.where(df.groupby(['code', 'date']).cumcount() == 0, first, v)
        df[c] = np.where(df[c] >= 0, df[c], np.nan)
    df.to_parquet(f'{OUT}/etf_iopv5m.parquet', index=False)
    print(f'=== DONE === {len(df):,} 行 {df.date.nunique()} 天 {time.time()-t0:.0f}s', flush=True)
    print(df.groupby('code').agg(n=('date', 'size'), d1=('date', 'min'), d2=('date', 'max')).to_string())
    print('\n真实折溢价(bp) 分位:')
    for c, s in df.groupby('code')['prem_bp']:
        print(f'  {c}: p1={s.quantile(.01):+.1f} p25={s.quantile(.25):+.1f} '
              f'p50={s.median():+.1f} p75={s.quantile(.75):+.1f} p99={s.quantile(.99):+.1f}')


if __name__ == '__main__':
    main()
