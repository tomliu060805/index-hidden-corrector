"""三宽基指数 5m OHLCV 面板: 000300/000905/000852, 2014-2026, 源=权威1min指数库

输出: out/index5m_transfer.parquet  列 date,bar,code,open,high,low,close,volume,amount
bar 0..47 (上午24+下午24, 右边界), 缺失bar不出行 (下游按稠密网格重排后=NaN)
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import glob, os, numpy as np, pandas as pd
from multiprocessing import Pool

SRC = CFG.INDEX_1M_DIR
OUT = f'{_R}/out'
CODES = ['000016.XSHG', '399006.XSHE', '000688.XSHG']


def one_day(f):
    date = os.path.basename(f)[:10]
    df = pd.read_parquet(f, columns=['datetime', 'code', 'open', 'high', 'low',
                                     'close', 'volume', 'money'])
    df = df[df.code.isin(CODES)]
    if not len(df):
        return None
    t = df['datetime']
    mins = t.dt.hour * 60 + t.dt.minute            # 右边界 09:31=571 .. 11:30=690, 13:01=781 .. 15:00=900
    am, pm = mins.between(571, 690), mins.between(781, 900)
    df = df[am | pm].copy()
    m = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    bar = np.where(m <= 690, (m - 571) // 5, 24 + (m - 781) // 5)
    df['bar'] = bar
    g = df.groupby(['code', 'bar'])
    out = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(),
                        'low': g['low'].min(), 'close': g['close'].last(),
                        'volume': g['volume'].sum(), 'amount': g['money'].sum(),
                        'n1m': g.size()}).reset_index()
    out.insert(0, 'date', date)
    return out


def main():
    fs = sorted(glob.glob(f'{SRC}/*.parquet'))
    print(f'{len(fs)} 天', flush=True)
    with Pool(32) as p:
        parts = [r for r in p.imap(one_day, fs, chunksize=20) if r is not None]
    df = pd.concat(parts, ignore_index=True)
    # 数据质量: 非满5根1min的bar记录下来但保留 (熔断/异常日)
    short = df[df.n1m < 5]
    print(f'总行 {len(df):,}; 不满5根1min的bar {len(short):,} 行, 涉及 {short.date.nunique()} 天')
    print(short.groupby('date').size().sort_values(ascending=False).head(10))
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(f'{OUT}/index5m_transfer.parquet', index=False)
    for c in CODES:
        sub = df[df.code == c]
        print(c, sub.date.min(), '~', sub.date.max(), f'{sub.date.nunique()}天',
              f'bar均值/天={len(sub)/max(sub.date.nunique(),1):.1f}')


if __name__ == '__main__':
    main()
