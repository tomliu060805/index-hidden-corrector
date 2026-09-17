"""指数版 512 维隐层提取: 000300/000905/000852 三宽基, 5min, 2014-2026

复刻 kronos_riskguard 场景三口径:
  - L=128 上下文, 窗口自身 z-score (mean/std), clip±5
  - 每个决策点独立前向 (坑#2: 不能滑窗复用, 归一化参数会带未来)
  - decode_s2 用单位置 sibling 调用 (坑#1: eval 下整序列 teacher-forcing 非因果)
  - h = ctx[:, -1, :] 最后一根bar的512维hidden; nll = n1+n2; ent = 下一bar s1 熵
决策点: 所有 bar 0..35 (未来12根5min留在日内), 36点/天 × 3指数
输出: out/hidden/chunk_XXXX.npz (h: float16, meta: [gt, ci, nll, ent])
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import os, sys, time, argparse, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from multiprocessing import Pool

sys.path.insert(0, CFG.KRONOS_REPO)
BASE = f'{_R}'
OUT = f'{BASE}/out/hidden_transfer'
COLS = ['open', 'high', 'low', 'close', 'volume', 'amount']
CODES = ['000016.XSHG', '399006.XSHE', '000688.XSHG']
CLIP, H, L = 5.0, 12, 128
_G = {}

T48 = ['09:35','09:40','09:45','09:50','09:55','10:00','10:05','10:10','10:15','10:20','10:25','10:30',
       '10:35','10:40','10:45','10:50','10:55','11:00','11:05','11:10','11:15','11:20','11:25','11:30',
       '13:05','13:10','13:15','13:20','13:25','13:30','13:35','13:40','13:45','13:50','13:55','14:00',
       '14:05','14:10','14:15','14:20','14:25','14:30','14:35','14:40','14:45','14:50','14:55','15:00']


def load_grid():
    df = pd.read_parquet(f'{BASE}/out/index5m_transfer.parquet')
    dates = np.array(sorted(df['date'].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    ci = pd.Series(np.arange(len(CODES)), index=CODES)
    g = di.reindex(df['date']).values * 48 + df['bar'].values
    s = ci.reindex(df['code']).values
    NG = len(dates) * 48
    arr = np.full((NG, len(CODES), len(COLS)), np.nan, np.float32)
    for k, c in enumerate(COLS):
        arr[g, s, k] = df[c].values.astype(np.float32)
    return arr, dates


def init_worker(nthreads):
    from model import Kronos, KronosTokenizer
    from model.kronos import calc_time_stamps
    torch.set_num_threads(nthreads)
    arr, dates = load_grid()
    ts = pd.to_datetime([f'{d} {t}' for d in dates for t in T48])
    _G['arr'] = arr
    _G['stamp'] = calc_time_stamps(ts.to_series()).values.astype(np.float32)
    _G['tok'] = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    _G['mdl'] = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()


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
            b = s1.shape[0]; ar = torch.arange(b)
            n1 = -F.log_softmax(logits[:, L - 2, :].float(), -1)[ar, s1[:, L - 1]]
            s2lg = mdl.decode_s2(ctx[:, :L - 1, :], s1[:, L - 1:L])
            n2 = -F.log_softmax(s2lg[:, -1, :].float(), -1)[ar, s2[:, L - 1]]
            pn = F.softmax(logits[:, L - 1, :].float(), -1)
            ent = -(pn * torch.log(pn + 1e-12)).sum(-1)
            h = ctx[:, -1, :]
        Hs.append(h.numpy().astype(np.float16))
        for k, (t, j) in enumerate(keep):
            meta.append((t, j, float(n1[k] + n2[k]), float(ent[k])))
    if not Hs:
        return cid, 0
    np.savez_compressed(fout, h=np.concatenate(Hs, 0), meta=np.array(meta, np.float64))
    return cid, len(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    arr, dates = load_grid()
    NG = arr.shape[0]
    close = arr[:, :, 3]
    pts = []
    for t in range(L, NG - H):
        if t % 48 > 47 - H:                      # 未来12根须留在日内
            continue
        for j in range(len(CODES)):
            if np.isfinite(close[t - L + 1:t + 1 + H, j]).all():
                pts.append((t, j))
    print(f'{len(pts):,} 个决策点 ({len(dates)}天 × ≤36bar × 3指数)', flush=True)

    per = 2000
    jobs = [(c, pts[i:i + per]) for c, i in enumerate(range(0, len(pts), per))]
    t0 = time.time()
    with Pool(args.workers, initializer=init_worker, initargs=(args.threads,)) as p:
        done = n = 0
        for cid, cnt in p.imap_unordered(run_chunk, jobs):
            done += 1; n += max(cnt, 0)
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f'{done}/{len(jobs)}  {n:,}行  {el/60:.1f}min  预计总{el/done*len(jobs)/60:.0f}min', flush=True)
    print('完成', flush=True)


if __name__ == '__main__':
    main()
