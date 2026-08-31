"""补提收盘决策点 (bar 47, 15:00) 的 512 维隐层 —— 供次日RV预报 (任务3)
复用 extract_index_hidden 的全部口径, 仅决策点不同; 输出 out/hidden_eod/
"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT
import extract_index_hidden as E
from multiprocessing import Pool

OUT = os.path.join(PROJECT_ROOT, 'out/hidden_eod')
E.OUT = OUT   # run_chunk 写到这里


def main():
    os.makedirs(OUT, exist_ok=True)
    arr, dates = E.load_grid()
    NG = arr.shape[0]
    close = arr[:, :, 3]
    pts = []
    for t in range(E.L, NG):
        if t % 48 != 47:
            continue
        for j in range(3):
            if np.isfinite(close[t - E.L + 1:t + 1, j]).all():
                pts.append((t, j))
    print(f'{len(pts):,} 个收盘决策点', flush=True)
    per = 500
    jobs = [(c, pts[i:i + per]) for c, i in enumerate(range(0, len(pts), per))]
    t0 = time.time()
    with Pool(6, initializer=E.init_worker, initargs=(5,)) as p:
        done = 0
        for cid, cnt in p.imap_unordered(E.run_chunk, jobs):
            done += 1
            if done % 5 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f'{done}/{len(jobs)}  {el/60:.1f}min', flush=True)
    print('完成', flush=True)


if __name__ == '__main__':
    main()
