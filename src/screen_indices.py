"""指数 universe 筛查: 找出研究区间内 1min K 线干净可用的全部指数。

为什么要筛: 1min 指数库里有**静默死值**序列 —— 例如中证全指 000985 在 2020-04-29
之后 1m/30m 全部被填成 pre_close, 2024-12-31 起彻底冻结在 4859.79。这类序列不会报错,
只会让波动预测在"零方差目标"上跑出漂亮数字。

三类判据(全部机械化, 不看结果):
  A 覆盖   —— 出现天数占应有交易日的比例, 以及首个观测日
  B 活性   —— 日内 close 恒定的天数占比 / 日内不同取值个数 / 零成交 bar 占比
  C 溯源   —— index_info.start_date 与 1m 首个观测日的关系; 1m 早于发布日 = 回填

★ 筛查只用 train+val (<= VAL_END) 的数据。用全区间筛会让"哪些指数进面板"这件事
   带上封存段的信息。test/holdout 的活性另行记录, 开封时才读。

用法: python screen_indices.py --workers 80
输出: out/index_screen.parquet (逐码统计) + out/index_universe.json (通过清单)
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

SRC = CFG.INDEX_1M_DIR
INFO = CFG.INDEX_INFO
OUT = f'{_R}/out'

VAL_END = '2024-06-07'          # 与项目既有切分一致: train<=2023-04-27, val<=2024-06-07
START = '2014-01-02'

# 通过门槛(先写下来, 不看结果再改)
# ★覆盖率必须相对指数自己的起点算, 不能相对 2014-01-02 —— 否则每个晚发布的指数
#   (中证1000 发布于 2014-10-17)都会被自己的发布日刷掉。
MIN_COVER = 0.98                # 自 first_obs 起的交易日覆盖率
MIN_DAYS = 1000                 # train+val 内最少可用天数(够训练)
MAX_TAIL_GAP = 10               # last_obs 距 VAL_END 的交易日数上限(防中途停更)
MAX_FROZEN = 0.01               # 日内 close 恒定的天数占比上限
MIN_DISTINCT = 30               # 日内不同 close 取值数的中位数下限(240根里至少30个不同值)
MAX_ZEROMONEY = 0.20            # 零成交额 bar 占比上限


def day_stats(path):
    """单日逐码统计。只读需要的列。"""
    try:
        df = pd.read_parquet(path, columns=['datetime', 'code', 'close', 'money'])
    except Exception:
        return None
    df = df.sort_values(['code', 'datetime'])
    g = df.groupby('code', sort=False)
    close = g['close']
    out = pd.DataFrame({
        'n_bars': g.size(),
        'n_distinct': close.nunique(),
        'c_std': close.std(),
        'c_mean': close.mean(),
        'n_zero_money': g['money'].apply(lambda s: int((s <= 0).sum())),
    })
    out['date'] = os.path.basename(path)[:10]
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=80)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    files = sorted(glob.glob(f'{SRC}/*.parquet'))
    files = [f for f in files if START <= os.path.basename(f)[:10] <= VAL_END]
    print(f'扫描 {len(files)} 个交易日 ({START} ~ {VAL_END}, 仅 train+val)', flush=True)

    t0 = time.time()
    parts = []
    with Pool(args.workers) as p:
        for k, r in enumerate(p.imap_unordered(day_stats, files, chunksize=4), 1):
            if r is not None:
                parts.append(r)
            if k % 400 == 0:
                print(f'  {k}/{len(files)}  {time.time()-t0:.0f}s', flush=True)
    raw = pd.concat(parts, ignore_index=True)
    ndays = len(files)

    # ---------- 逐码汇总 ----------
    raw['frozen'] = (raw['c_std'].fillna(0) == 0).astype(int)
    agg = raw.groupby('code').agg(
        days=('date', 'size'),
        first_obs=('date', 'min'),
        last_obs=('date', 'max'),
        frozen_frac=('frozen', 'mean'),
        distinct_med=('n_distinct', 'median'),
        bars_med=('n_bars', 'median'),
        zeromoney_frac=('n_zero_money', lambda s: float(s.sum()) / max(1, 240 * len(s))),
    ).reset_index()
    # ★覆盖率相对指数自己的首个观测日: 分母 = [first_obs, VAL_END] 之间的交易日数
    alldays = np.array([os.path.basename(f)[:10] for f in files])
    pos = pd.Series(np.arange(ndays), index=alldays)
    agg['i_first'] = pos.reindex(agg['first_obs']).values
    agg['i_last'] = pos.reindex(agg['last_obs']).values
    agg['cover'] = agg['days'] / (ndays - agg['i_first'])
    agg['tail_gap'] = (ndays - 1) - agg['i_last']      # 距 val 末尾还差几个交易日

    # ---------- 溯源: 发布日 ----------
    info = pd.read_parquet(INFO)[['code', 'start_date', 'end_date', 'display_name']]
    info['start_date'] = info['start_date'].astype(str)
    agg = agg.merge(info, on='code', how='left')
    # 1m 首观测早于发布日 => 回填(该段历史不是当时可见的)
    agg['backfilled'] = (agg['first_obs'] < agg['start_date'].fillna('1900-01-01')).astype(int)

    # ---------- 判据 ----------
    agg['pass_cover'] = (agg['cover'] >= MIN_COVER) & (agg['days'] >= MIN_DAYS)
    agg['pass_fresh'] = agg['tail_gap'] <= MAX_TAIL_GAP        # 中途停更的刷掉
    agg['pass_live'] = (agg['frozen_frac'] <= MAX_FROZEN) & (agg['distinct_med'] >= MIN_DISTINCT)
    agg['pass_liquid'] = agg['zeromoney_frac'] <= MAX_ZEROMONEY
    agg['pass_bars'] = agg['bars_med'] == 240
    agg['pass_pit'] = agg['backfilled'] == 0
    agg['PASS'] = (agg['pass_cover'] & agg['pass_fresh'] & agg['pass_live']
                   & agg['pass_liquid'] & agg['pass_bars'] & agg['pass_pit'])

    agg = agg.sort_values(['PASS', 'code'], ascending=[False, True])
    agg.to_parquet(f'{OUT}/index_screen.parquet', index=False)

    n = len(agg)
    print(f'\n扫描完成 {time.time()-t0:.0f}s —— {n} 个指数码')
    print(f'  通过 {int(agg.PASS.sum())}')
    for c, lab in [('pass_cover', f'覆盖>={MIN_COVER}且天数>={MIN_DAYS}'), ('pass_fresh', '未中途停更'), ('pass_bars', '每日240根'),
                   ('pass_live', f'非死值(冻结<={MAX_FROZEN}, 取值数>={MIN_DISTINCT})'),
                   ('pass_liquid', f'零成交额<={MAX_ZEROMONEY}'), ('pass_pit', '非回填')]:
        print(f'    被 {lab} 刷掉: {int((~agg[c]).sum())}')

    keep = agg[agg.PASS]['code'].tolist()
    json.dump({'val_end': VAL_END, 'start': START, 'n_days': ndays,
               'thresholds': {'MIN_COVER': MIN_COVER, 'MAX_FROZEN': MAX_FROZEN,
                              'MIN_DISTINCT': MIN_DISTINCT, 'MAX_ZEROMONEY': MAX_ZEROMONEY},
               'codes': keep},
              open(f'{OUT}/index_universe.json', 'w'), indent=1)
    print(f'\n写出 {OUT}/index_screen.parquet 与 index_universe.json ({len(keep)} 个指数)')

    # 已知陷阱对拍
    print('\n已知案例对拍:')
    for c in ['000985.XSHG', '000300.XSHG', '000905.XSHG', '000852.XSHG', '399317.XSHE']:
        r = agg[agg.code == c]
        if len(r):
            r = r.iloc[0]
            print(f"  {c} {str(r.display_name)[:8]:10s} PASS={bool(r.PASS)} "
                  f"cover={r.cover:.3f} frozen={r.frozen_frac:.3f} distinct_med={r.distinct_med:.0f} "
                  f"first={r.first_obs} start_date={r.start_date} backfill={r.backfilled}")


if __name__ == '__main__':
    main()
