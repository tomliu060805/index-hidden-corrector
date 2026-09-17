"""全指数 5m OHLCV 面板: 由 screen_indices.py 筛出的 universe, 2014-2026。

源 = 权威 1min 指数库, 右边界口径(09:31 起, 240根/日) -> 聚合成 48 根 5min bar。
聚合规则与 build_index5m.py 完全一致(open=首, high=max, low=min, close=末, volume/money=和)。

★ universe 来自 out/index_universe.json, 由 train+val 筛出, 不含封存段信息。
★ 输出按年分片, 防止单文件过大。

用法: python build_index5m_universe.py --workers 80
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import argparse, glob, json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

SRC = CFG.INDEX_1M_DIR
OUT = f'{_R}/out/index5m_universe'
_G = {}


def init(codes):
    _G['codes'] = set(codes)


def one_day(path):
    date = os.path.basename(path)[:10]
    try:
        df = pd.read_parquet(path, columns=['datetime', 'code', 'open', 'high',
                                            'low', 'close', 'volume', 'money'])
    except Exception:
        return None
    df = df[df['code'].isin(_G['codes'])]
    if not len(df):
        return None
    df = df.sort_values(['code', 'datetime'])
    # 1min 右边界 09:31..15:00 -> 每 5 根一个 bar, 组内序号 0..47
    n = df.groupby('code', sort=False).cumcount()
    df['bar'] = n // 5
    g = df.groupby(['code', 'bar'], sort=True)
    out = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                close=('close', 'last'), volume=('volume', 'sum'),
                money=('money', 'sum')).reset_index()
    out = out[out['bar'] < 48]
    out['date'] = date
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=80)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    codes = json.load(open(f'{_R}/out/index_universe.json'))['codes']
    print(f'universe {len(codes)} 个指数', flush=True)

    files = sorted(glob.glob(f'{SRC}/*.parquet'))
    byyear = {}
    for f in files:
        byyear.setdefault(os.path.basename(f)[:4], []).append(f)
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(codes,)) as p:
        for yr in sorted(byyear):
            fo = f'{OUT}/{yr}.parquet'
            if os.path.exists(fo):
                print(f'  {yr} 已存在, 跳过', flush=True)
                continue
            parts = [r for r in p.imap_unordered(one_day, byyear[yr], chunksize=2)
                     if r is not None]
            df = pd.concat(parts, ignore_index=True)
            df['code'] = df['code'].astype('category')
            df.to_parquet(fo, index=False, compression='zstd')
            print(f'  {yr}: {len(df):,} 行 -> {fo}  ({time.time()-t0:.0f}s)', flush=True)
    tot = sum(os.path.getsize(f) for f in glob.glob(f'{OUT}/*.parquet'))
    print(f'=== UNIVERSE PANEL DONE === {tot/1e9:.2f}GB  {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
