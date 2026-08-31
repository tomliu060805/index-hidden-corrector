"""实验① 多层/多位置提取 (Kronos-small): 4路表示
  h0 final_last  = norm(最后层)[:, -1]   (与现有存储等价, 存作对齐/sanity)
  h1 mid_last    = norm(第n/2层)[:, -1]  (logit-lens 式中间层)
  h2 final_mean  = norm(最后层).mean(全128位置)
  h3 final_mean12= norm(最后层)[-12:].mean (最近1小时池化)
输出 out/hidden_multi/chunk_XXXX.npz: h (n,4,512) fp16, meta (t,j)
决策点与原提取完全同构; eval 用 (t,j) merge。
"""
import os, sys, time, argparse, numpy as np, pandas as pd, torch
from multiprocessing import Pool
sys.path.insert(0, KRONOS_REPO)
from config import KRONOS_REPO, PROJECT_ROOT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_index_hidden as E

OUT = os.path.join(PROJECT_ROOT, 'out/hidden_multi')
_G = E._G
L, CLIP, H = E.L, E.CLIP, E.H


def run_chunk(job):
    cid, pts = job
    fout = f'{OUT}/chunk_{cid:04d}.npz'
    if os.path.exists(fout):
        return cid, -1
    arr, stamp = _G['arr'], _G['stamp']
    tok, mdl = _G['tok'], _G['mdl']
    nmid = mdl.n_layers // 2
    B = 16
    Hs, meta = [], []
    for i in range(0, len(pts), B):
        blk = pts[i:i + B]
        ws, sts, keep = [], [], []
        for (t, j) in blk:
            w = arr[t - L + 1:t + 1, j, :]
            if not np.isfinite(w).all():
                continue
            m, s = w.mean(0), w.std(0)
            ws.append(np.clip((w - m) / (s + 1e-5), -CLIP, CLIP))
            sts.append(stamp[t - L + 1:t + 1]); keep.append((t, j))
        if not ws:
            continue
        x = torch.from_numpy(np.stack(ws).astype(np.float32))
        st = torch.from_numpy(np.stack(sts).astype(np.float32))
        with torch.no_grad():
            s1, s2 = tok.encode(x, half=True)
            z = mdl.embedding([s1, s2]) + mdl.time_emb(st)
            z = mdl.token_drop(z)
            xm = None
            for li, layer in enumerate(mdl.transformer):
                z = layer(z)
                if li == nmid - 1:
                    xm = z
            zn = mdl.norm(z)
            h0 = zn[:, -1]
            h1 = mdl.norm(xm)[:, -1]
            h2 = zn.mean(1)
            h3 = zn[:, -12:].mean(1)
            h4 = torch.stack([h0, h1, h2, h3], 1)      # (n,4,512)
        Hs.append(h4.numpy().astype(np.float16))
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
    with Pool(args.workers, initializer=E.init_worker, initargs=(args.threads,)) as p:
        done = n = 0
        for cid, cnt in p.imap_unordered(run_chunk, jobs):
            done += 1; n += max(cnt, 0)
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f'{done}/{len(jobs)} {n:,}行 {el/60:.1f}min 预计总{el/done*len(jobs)/60:.0f}min', flush=True)


if __name__ == '__main__':
    main()
