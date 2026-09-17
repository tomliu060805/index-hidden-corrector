"""IC/IM 全合约 5m 面板 + 主力连续序列 (2015-04起, 合约内收益无换月跳价)

- 所有具体合约 (剔 8888/9999), 1m→5m, 时间窗与指数一致 09:31-11:30/13:01-15:00
- 主力映射 shift(1) 防前视: 今日持仓合约 = 昨日官方主力
- 隔夜收益 = 同一合约 今日bar0.open/昨日15:00收盘 -> 无跳价; roll日标记(成本另计)
输出: out/futures_dom5m.parquet 列 date,bar,product,code,open,close,ret(合约内5m logret),
     ov_ret(仅bar0行, 同合约隔夜), roll(当日换合约)
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import glob, os, numpy as np, pandas as pd
from multiprocessing import Pool

SRC = CFG.FUTURES_1M_DIR
DOM = CFG.DOMINANT_MAP_DIR
OUT = f'{_R}/out'
PRODUCTS = ('IC', 'IM')


def one_day(f):
    date = os.path.basename(f)[:10]
    if date < '2015-04-01':
        return None
    df = pd.read_parquet(f, columns=['datetime', 'code', 'open', 'close'])
    m = df.code.str.match(r'^(IC|IM)\d{4}\.CCFX$')
    df = df[m]
    if not len(df):
        return None
    mins = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    am, pm = mins.between(571, 690), mins.between(781, 900)
    df = df[am | pm].copy()
    mm = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['bar'] = np.where(mm <= 690, (mm - 571) // 5, 24 + (mm - 781) // 5)
    g = df.groupby(['code', 'bar'])
    out = pd.DataFrame({'open': g['open'].first(), 'close': g['close'].last()}).reset_index()
    out.insert(0, 'date', date)
    return out


def main():
    fs = sorted(glob.glob(f'{SRC}/*.parquet'))
    with Pool(30) as p:
        parts = [r for r in p.imap(one_day, fs, chunksize=20) if r is not None]
    allc = pd.concat(parts, ignore_index=True)
    allc['product'] = allc.code.str[:2]
    print(f'全合约5m: {len(allc):,} 行, {allc.date.nunique()} 天')

    # 主力映射 (shift1 防前视)
    dm = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{DOM}/*.parquet'))],
                   ignore_index=True)
    dm['date'] = pd.to_datetime(dm['date']).dt.strftime('%Y-%m-%d')
    dm = dm[dm['product'].isin(PRODUCTS)].sort_values(['product', 'date'])
    dm['code_use'] = dm.groupby('product')['code'].shift(1)
    dm = dm.dropna(subset=['code_use'])
    dm['roll'] = dm['code_use'] != dm.groupby('product')['code_use'].shift(1)
    dm.loc[dm.groupby('product').head(1).index, 'roll'] = False

    # 每合约逐日收盘 (bar47) 与前收 (合约内 shift)
    eod = allc[allc.bar == 47][['code', 'date', 'close']].sort_values(['code', 'date'])
    eod['prev_close'] = eod.groupby('code')['close'].shift(1)

    key = dm.drop(columns=['code']).rename(columns={'code_use': 'code'})[['date', 'product', 'code', 'roll']]
    dom = allc.drop(columns=['product']).merge(key, on=['date', 'code'], how='inner')
    dom = dom.sort_values(['product', 'date', 'bar']).reset_index(drop=True)
    lc = np.log(dom['close'].values)
    ret = np.full(len(dom), np.nan)
    same = (dom['date'] == dom['date'].shift(1)) & (dom['code'] == dom['code'].shift(1))
    ret[1:] = lc[1:] - lc[:-1]
    ret[~same.values] = np.nan
    dom['ret'] = ret
    fb = dom.groupby(['product', 'date'], sort=False)['bar'].transform('min') == dom['bar']
    dom = dom.merge(eod[['code', 'date', 'prev_close']], on=['code', 'date'], how='left')
    dom['ov_ret'] = np.where(fb.values, np.log(dom['open'].values) - np.log(dom['prev_close'].values), np.nan)
    dom.to_parquet(f'{OUT}/futures_dom5m.parquet', index=False)
    for prod in PRODUCTS:
        s = dom[dom['product'] == prod]
        nroll = s.groupby('date')['roll'].first().sum()
        print(f'{prod}: {s.date.min()}~{s.date.max()} {s.date.nunique()}天 '
              f'bar/天={len(s)/s.date.nunique():.1f} 换月{nroll}次 '
              f'隔夜覆盖={s.ov_ret.notna().sum()}/{s.date.nunique()}')


if __name__ == '__main__':
    main()
