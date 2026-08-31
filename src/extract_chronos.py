"""实验③ 换基础模型: Chronos-t5-small 编码器嵌入 -> out/hidden_chronos/
输入=单变量收盘价窗口128 (Chronos内部自做mean-scale); 存 EOS位置 与 mean池化 两路 (各512维)
决策点与原提取同构; meta=(t,j)"""
import os, sys, time, argparse, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT
import extract_index_hidden as E

OUT = os.path.join(PROJECT_ROOT, 'out/hidden_chronos')
L, H = E.L, E.H
_G = {}


def init_worker(nthreads):
    torch.set_num_threads(nthreads)
    from chronos import ChronosPipeline
    arr, dates = E.load_grid()
    _G['close'] = arr[:, :, 3]
    _G['pipe'] = ChronosPipeline.from_pretrained('amazon/chronos-t5-small',
                                                 device_map='cpu', torch_dtype=torch.float32)


def run_chunk(job):
    cid, pts = job
    fout = f'{OUT}/chunk_{cid:04d}.npz'
    if os.path.exists(fout):
        return cid, -1
    close, pipe = _G['close'], _G['pipe']
    B = 32
    Hs, meta = [], []
    for i in range(0, len(pts), B):
        blk = pts[i:i + B]
        ws, keep = [], []
        for (t, j) in blk:
            w = close[t - L + 1:t + 1, j]
            if not np.isfinite(w).all():
                continue
            ws.append(w); keep.append((t, j))
        if not ws:
            continue
        ctx = torch.from_numpy(np.stack(ws).astype(np.float32))
        with torch.no_grad():
            emb, scale = pipe.embed(ctx)               # (n, L+1, 512)
            e_last = emb[:, -1, :]                     # EOS (双向编码器, 可见全窗)
            e_mean = emb[:, :-1, :].mean(1)
        Hs.append(torch.stack([e_last, e_mean], 1).numpy().astype(np.float16))
        meta.extend(keep)
    if not Hs:
        return cid, 0
    np.savez_compressed(fout, h=np.concatenate(Hs, 0), meta=np.array(meta, np.int64))
    return cid, len(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    arr, dates = E.load_grid()
    NG = arr.shape[0]
    close = arr[:, :, 3]
    pts = []
    for t in range(L, NG - H):
        if t % 48 > 47 - H:
            continue
        for j in range(3):
            if np.isfinite(close[t - L + 1:t + 1 + H, j]).all():
                pts.append((t, j))
    print(f'{len(pts):,} 决策点', flush=True)
    per = 2000
    jobs = [(c, pts[i:i + per]) for c, i in enumerate(range(0, len(pts), per))]
    t0 = time.time()
    with Pool(args.workers, initializer=init_worker, initargs=(args.threads,)) as p:
        done = n = 0
        for cid, cnt in p.imap_unordered(run_chunk, jobs):
            done += 1; n += max(cnt, 0)
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f'{done}/{len(jobs)} {n:,}行 {el/60:.1f}min 预计总{el/done*len(jobs)/60:.0f}min', flush=True)


if __name__ == '__main__':
    main()
