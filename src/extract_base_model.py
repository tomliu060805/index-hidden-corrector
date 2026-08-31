"""实验② 规模消融: Kronos-base 提取 final_last 隐层 -> out/hidden_base/
决策点与原提取同构; meta=(t,j)"""
import os, sys, time, argparse, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, KRONOS_REPO)
from config import KRONOS_REPO, PROJECT_ROOT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_index_hidden as E

OUT = os.path.join(PROJECT_ROOT, 'out/hidden_base')
L, CLIP, H = E.L, E.CLIP, E.H
_G = E._G


def init_worker(nthreads):
    from model import Kronos, KronosTokenizer
    from model.kronos import calc_time_stamps
    import pandas as pd
    torch.set_num_threads(nthreads)
    arr, dates = E.load_grid()
    ts = pd.to_datetime([f'{d} {t}' for d in dates for t in E.T48])
    _G['arr'] = arr
    _G['stamp'] = calc_time_stamps(ts.to_series()).values.astype(np.float32)
    _G['tok'] = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    _G['mdl'] = Kronos.from_pretrained("NeoQuasar/Kronos-base").eval()


def run_chunk(job):
    cid, pts = job
    fout = f'{OUT}/chunk_{cid:04d}.npz'
    if os.path.exists(fout):
        return cid, -1
    arr, stamp = _G['arr'], _G['stamp']
    tok, mdl = _G['tok'], _G['mdl']
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
            logits, ctx = mdl.decode_s1(s1, s2, st)
            h = ctx[:, -1, :]
        Hs.append(h.numpy().astype(np.float16))
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
