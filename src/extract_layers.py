"""逐层隐状态提取: Kronos-small 的 10 个位置 × 2 种池化。

动机: TimeRep (arXiv 2509.12650) 在 MOMENT-Large 的 24 层里扫出第 16 层最好, 并报告
**最后一层往往过拟合预训练目标**; arXiv 2511.15324 的逐层探针显示深层编码 dispersion
(离散度), 正是波动预测要的东西。本项目一直只取最后一层, 从未系统扫过。

位置 (见 Kronos.decode_s1 源码):
    L0            embedding + time_emb + token_drop 之后, 进第一个 block 之前
    L1..L8        第 k 个 TransformerBlock 的输出
    L8n           self.norm(L8) —— decode_s1 实际返回的 context, **现行生产用的就是这个**
池化:
    last          序列最后一个位置 (现行做法)
    mean          全序列 128 个位置的均值

★ 用 forward hook 采集, 不重写前向。项目历史上 decode_s2 在 eval 下非因果那个坑
  (extract_index_hidden.py 坑#1) 说明重写 Kronos 内部逻辑风险很高。

★ 口径与 extract_index_hidden.py 完全一致(直接 import 复用 load_grid/T48/L/CLIP):
  L=128 上下文, 窗口自身 z-score, clip±5, 每个决策点独立前向(不滑窗复用)。
  唯一差别: 决策点扩到 bar 0..41 —— 因为最短视界 H=6 需要到 bar 41。

输出: out/hidden_layers/chunk_XXXX.npz
      h: float16 [n, 10, 2, 512]   meta: [t, j]
用法: python extract_layers.py --workers 10 --threads 8
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np
import torch

sys.path.insert(0, CFG.KRONOS_REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_index_hidden as E

OUT = f'{_R}/out/hidden_layers'
HMIN = 6                     # 最短视界: 决策点须留 6 根在日内 => bar <= 41
NPOS, NPOOL, DM = 10, 2, 512
POS_NAMES = ['L0'] + [f'L{k}' for k in range(1, 9)] + ['L8n']
_G = {}


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
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
    assert mdl.n_layers == 8, f'层数变了: {mdl.n_layers}'

    # ---- forward hook: 每个 block 的输出立刻压成 (last, mean), 不留整条序列 ----
    buf = {}

    def mk(k):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out          # [B, L, D]
            buf[k] = torch.stack([h[:, -1, :], h.mean(1)], 1)      # [B, 2, D]
        return hook

    for k, blk in enumerate(mdl.transformer, start=1):
        blk.register_forward_hook(mk(k))
    _G['mdl'] = mdl
    _G['buf'] = buf


def run_chunk(job):
    cid, pts = job
    fout = f'{OUT}/chunk_{cid:04d}.npz'
    if os.path.exists(fout):
        return cid, -1
    arr, stamp = _G['arr'], _G['stamp']
    tok, mdl, buf = _G['tok'], _G['mdl'], _G['buf']
    L, CLIP = E.L, E.CLIP
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
        buf.clear()
        with torch.no_grad():
            s1, s2 = tok.encode(x, half=True)
            # L0: 复刻 decode_s1 前三行(embedding + time_emb + token_drop)
            e = mdl.embedding([s1, s2]) + mdl.time_emb(st)
            e = mdl.token_drop(e)                        # eval 下是恒等, 但照复刻
            l0 = torch.stack([e[:, -1, :], e.mean(1)], 1)
            _, ctx = mdl.decode_s1(s1, s2, st)           # 触发 hook 填 buf[1..8]
            l8n = torch.stack([ctx[:, -1, :], ctx.mean(1)], 1)
            # [B, 10, 2, D]
            h = torch.stack([l0] + [buf[k] for k in range(1, 9)] + [l8n], 1)
        Hs.append(h.numpy().astype(np.float16))
        meta.extend(keep)
    if not Hs:
        return cid, 0
    np.savez_compressed(fout, h=np.concatenate(Hs, 0),
                        meta=np.array(meta, np.int64), pos=np.array(POS_NAMES))
    return cid, len(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    arr, dates = E.load_grid()
    NG = arr.shape[0]
    close = arr[:, :, 3]
    L = E.L
    pts = []
    for t in range(L, NG - HMIN):
        if t % 48 > 47 - HMIN:                   # 最短视界的未来窗须留在日内
            continue
        for j in range(arr.shape[1]):
            if np.isfinite(close[t - L + 1:t + 1 + HMIN, j]).all():
                pts.append((t, j))
    est = len(pts) * NPOS * NPOOL * DM * 2 / 1e9
    print(f'{len(pts):,} 个决策点 ({len(dates)}天 × ≤42bar × {arr.shape[1]}指数), '
          f'预计落盘 ~{est:.1f}GB', flush=True)

    per = 2000
    jobs = [(c, pts[i:i + per]) for c, i in enumerate(range(0, len(pts), per))]
    t0 = time.time()
    done = n = 0
    with Pool(args.workers, initializer=init_worker, initargs=(args.threads,)) as p:
        for cid, k in p.imap_unordered(run_chunk, jobs):
            done += 1
            if k > 0:
                n += k
            if done % 20 == 0:
                el = time.time() - t0
                print(f'  {done}/{len(jobs)} chunks  {n:,} 点  {el:.0f}s  '
                      f'eta {el/done*(len(jobs)-done):.0f}s', flush=True)
    print(f'=== LAYERS ALL DONE === {n:,} 点  {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
