"""把 train 段的逐日探针跑成一张面板。只跑 <=2023-04-27(父项目 train 段边界)。"""
import os, sys, glob, time
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_day import one_day

SRC = CFG.CCFX_L2_DIR
TRAIN_END = '20250717'   # ★开封: 扩到 test 末(holdout 仍不建)
START = '20200101'
OUT = '{OUTDIR}/panel_full.parquet'

def run(day):
    try:
        return one_day(day)
    except Exception:
        return None

if __name__ == '__main__':
    days = [os.path.basename(p) for p in sorted(glob.glob(f'{SRC}/2*'))]
    days = [d for d in days if START <= d <= TRAIN_END]
    print(f'{len(days)} 个交易日 ({days[0]} ~ {days[-1]}, 全部在 train 段内)', flush=True)
    t0 = time.time(); parts = []
    with Pool(40) as p:
        for k, r in enumerate(p.imap_unordered(run, days, chunksize=2), 1):
            if r is not None:
                parts.append(r)
            if k % 100 == 0:
                print(f'  {k}/{len(days)}  {time.time()-t0:.0f}s', flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f'=== DONE === {len(df):,} 行 {df.date.nunique()} 天 -> {OUT}  {time.time()-t0:.0f}s', flush=True)
    print(df.groupby('prod').agg(n=('date','size'), d1=('date','min'), d2=('date','max')).to_string())
