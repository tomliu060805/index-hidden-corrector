"""ETF tick(3秒快照, 5档) -> 5min 网格, 与指数 5min 网格同一时间轴。

用于 docs/PREREG_ETF_PREMIUM.md 的折价溢价检验。
口径与 build_index5m.py 一致: 09:31-11:30 / 13:01-15:00, 48 根/日, 右边界。
输出 out/etf5m.parquet: date,bar,code,close,high,low,money,volume,spr_bp,dep
"""
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, glob, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

SRC = CFG.ETF_TICK_DIR
OUT = f'{_R}/out'
CODES = ['510300.XSHG', '510500.XSHG', '512100.XSHG']      # 300 / 500 / 1000
COLS = ['datetime', 'code', 'current', 'volume', 'money', 'a1_p', 'a1_v', 'b1_p', 'b1_v']


def one_day(f):
    date = os.path.basename(f)[:10]
    try:
        d = pd.read_parquet(f, columns=COLS)
    except Exception:
        return None
    d = d[d.code.isin(CODES)]
    if not len(d):
        return None
    t = pd.to_datetime(d.datetime)
    m = t.dt.hour * 60 + t.dt.minute
    am, pm = m.between(571, 690), m.between(781, 900)
    d = d[am | pm].copy()
    if not len(d):
        return None
    mm = pd.to_datetime(d.datetime).dt.hour * 60 + pd.to_datetime(d.datetime).dt.minute
    d['bar'] = np.where(mm <= 690, (mm - 571) // 5, 24 + (mm - 781) // 5)
    d = d[(d.bar >= 0) & (d.bar < 48)]
    d = d.sort_values('datetime')
    mid = np.where((d.a1_p > 0) & (d.b1_p > 0), (d.a1_p + d.b1_p) / 2, np.nan)
    d['mid'] = mid
    d['spr'] = np.where((d.a1_p > 0) & (d.b1_p > 0),
                        (d.a1_p - d.b1_p) / np.maximum(mid, 1e-9) * 1e4, np.nan)
    d['dep'] = np.log(np.maximum(d.a1_v + d.b1_v, 1.0))
    g = d.groupby(['code', 'bar'], sort=True)
    o = pd.DataFrame({
        'close':  g['current'].last(),
        'mid':    g['mid'].last(),
        'high':   g['current'].max(),
        'low':    g['current'].min(),
        'volume': g['volume'].last(),          # 累计量, 后面差分
        'money':  g['money'].last(),           # 累计额
        'spr_bp': g['spr'].mean(),
        'dep':    g['dep'].mean(),
        'nt':     g.size(),
    }).reset_index()
    o['date'] = date
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=40)
    a = ap.parse_args()
    fs = sorted(glob.glob(f'{SRC}/*.parquet'))
    print(f'{len(fs)} 个交易日 {os.path.basename(fs[0])[:10]} ~ {os.path.basename(fs[-1])[:10]}',
          flush=True)
    t0 = time.time(); parts = []
    with Pool(a.workers) as p:
        for k, r in enumerate(p.imap_unordered(one_day, fs, chunksize=4), 1):
            if r is not None:
                parts.append(r)
            if k % 400 == 0:
                print(f'  {k}/{len(fs)}  {time.time()-t0:.0f}s', flush=True)
    df = pd.concat(parts, ignore_index=True).sort_values(['code', 'date', 'bar'])
    # 累计量 -> 逐 bar 增量(日内差分, 跨日重置)
    for c in ['volume', 'money']:
        v = df.groupby(['code', 'date'])[c].diff()
        first = df.groupby(['code', 'date'])[c].transform('first')
        df[c] = np.where(df.groupby(['code', 'date']).cumcount() == 0, first, v)
        df[c] = np.where(df[c] >= 0, df[c], np.nan)
    df.to_parquet(f'{OUT}/etf5m.parquet', index=False)
    print(f'=== DONE === {len(df):,} 行  {df.date.nunique()} 天  {time.time()-t0:.0f}s', flush=True)
    print(df.groupby('code').agg(n=('date', 'size'), d1=('date', 'min'), d2=('date', 'max'),
                                 bars=('bar', 'nunique')).to_string())


if __name__ == '__main__':
    main()
