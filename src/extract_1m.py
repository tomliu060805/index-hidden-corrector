"""把 **1 分钟** K 线喂给 Kronos —— 与现行 5min 输入做干净的 A/B。

动机: 这条线变过模型规模(small/base)、取哪一层(10个位置)、池化(last/mean)、
目标视界(5min~2h)、基线强度(B1~B8), **唯独输入频率从没变过**(一直是 5min)。
而在决策时刻 t, 现行配置看到的最近观测是**一根已完成的 5min bar**;
换成 1min 输入, 同一时刻看到的是最近 5 根 1min bar 各自的 OHLCV ——
严格更多的信息, 不是等价重排。

★A/B 的干净性: **决策点与目标完全不变**(仍是 5min bar 边界, 目标仍是 H 根 5min bar),
  唯一变的是喂给 Kronos 的输入粒度与上下文长度。任何差异只能归因于输入频率。

  现行:  5min × L=128  -> 回看 640 分钟 = 2.67 个交易日
  本脚本: 1min × L=512  -> 回看 512 分钟 = 2.13 个交易日   (512 = Kronos-small 的 max_context)

  回看几乎不损失, 粒度细 5 倍。

★必须配套(今天新增的失败模式 baseline-sees-fewer-inputs): Kronos 吃 1min, 基线也必须
  吃 1min。1min 数据最直接的受益者恰恰是 RV 估计量本身(同窗口从 12 个收益变 60 个)。
  见 src/har_baselines.py 的 B9。
★反方向的已知事实: 1min RV 被微观结构噪声污染(Liu-Patton-Sheppard: 5min 是甜点),
  这对**基线**不利, 对 Kronos 未必 —— 它可能从噪声模式本身提取信号。故先验不判死。

输出: out/hidden_1m/chunk_XXXX.npz   h: float16 [n, 2, 512](L6/L8n 的 last)  meta: [t5, j]
      ★meta 的 t5 是**5min 网格**的下标, 直接与现有面板对齐。
用法: python extract_1m.py --workers 10 --threads 8
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, CFG.KRONOS_REPO)

SRC1M = CFG.INDEX_1M_DIR
OUT = f'{_R}/out/hidden_1m'
GRID = f'{_R}/out/index1m_grid.npy'
GRID_META = f'{_R}/out/index1m_grid_meta.json'
CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']
COLS = ['open', 'high', 'low', 'close', 'volume', 'money']
NB1, NB5 = 240, 48          # 每日 1min / 5min 根数
L, CLIP = 512, 5.0          # ★Kronos-small 的 max_context = 512
HMIN = 1                    # 决策点须留至少 1 根 5min bar 在日内
POS = ['L6', 'L8n']
_G = {}


def build_grid():
    """[NG_1m, 3, 6] float32, 约 53MB。1min 源列名 money/amount 两种都兼容。"""
    files = sorted(glob.glob(f'{SRC1M}/*.parquet'))
    dates = [os.path.basename(f)[:10] for f in files]
    ci = pd.Series(np.arange(len(CODES)), index=CODES)
    arr = np.full((len(files) * NB1, len(CODES), len(COLS)), np.nan, np.float32)
    for di, f in enumerate(files):
        df = pd.read_parquet(f)
        df = df[df['code'].isin(CODES)]
        if not len(df):
            continue
        amt = 'money' if 'money' in df.columns else 'amount'
        df = df.sort_values(['code', 'datetime'])
        n = df.groupby('code', sort=False).cumcount().values      # 组内序号 0..239
        s = ci.reindex(df['code']).values.astype(int)
        keep = n < NB1
        g = di * NB1 + n[keep]
        for k, c in enumerate(COLS):
            arr[g, s[keep], k] = df[c if c != 'money' else amt].values[keep].astype(np.float32)
        if di % 500 == 0:
            print(f'  {di}/{len(files)}', flush=True)
    np.save(GRID, arr)
    json.dump({'dates': dates, 'codes': CODES}, open(GRID_META, 'w'))
    print(f'1min 网格落盘 {GRID} ({os.path.getsize(GRID)/1e6:.0f}MB, {len(files)}天)', flush=True)


def load_grid():
    m = json.load(open(GRID_META))
    return np.load(GRID, mmap_mode='r'), np.array(m['dates']), m['codes']


def init_worker(nthreads):
    from model import Kronos, KronosTokenizer
    from model.kronos import calc_time_stamps
    torch.set_num_threads(nthreads)
    arr, dates, codes = load_grid()
    # 1min 时间戳: 09:31..11:30, 13:01..15:00
    tt = ([f'09:{m:02d}' for m in range(31, 60)] + [f'10:{m:02d}' for m in range(60)]
          + [f'11:{m:02d}' for m in range(31)]
          + [f'13:{m:02d}' for m in range(1, 60)] + [f'14:{m:02d}' for m in range(60)]
          + ['15:00'])
    assert len(tt) == NB1, len(tt)
    ts = pd.to_datetime([f'{d} {t}' for d in dates for t in tt])
    _G['arr'] = arr
    _G['stamp'] = calc_time_stamps(ts.to_series()).values.astype(np.float32)
    _G['tok'] = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
    buf = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        buf['L6'] = h[:, -1, :]
    mdl.transformer[5].register_forward_hook(hook)
    _G['mdl'] = mdl
    _G['buf'] = buf


def run_chunk(job):
    cid, pts = job
    fout = f'{OUT}/chunk_{cid:04d}.npz'
    if os.path.exists(fout):
        return cid, -1
    arr, stamp = _G['arr'], _G['stamp']
    tok, mdl, buf = _G['tok'], _G['mdl'], _G['buf']
    B = 8                                  # L=512 时注意力是 L=128 的 16 倍, batch 调小
    Hs, meta = [], []
    for i in range(0, len(pts), B):
        blk = pts[i:i + B]
        ws, sts, keep = [], [], []
        for (i1, t5, j) in blk:
            w = arr[i1 - L + 1:i1 + 1, j, :]
            if not np.isfinite(w).all():
                continue
            m, s = w.mean(0), w.std(0)
            ws.append(np.clip((w - m) / (s + 1e-5), -CLIP, CLIP))
            sts.append(stamp[i1 - L + 1:i1 + 1]); keep.append((t5, j))
        if not ws:
            continue
        x = torch.from_numpy(np.stack(ws).astype(np.float32))
        st = torch.from_numpy(np.stack(sts).astype(np.float32))
        buf.clear()
        with torch.no_grad():
            s1, s2 = tok.encode(x, half=True)
            _, ctx = mdl.decode_s1(s1, s2, st)
            h = torch.stack([buf['L6'], ctx[:, -1, :]], 1)
        Hs.append(h.numpy().astype(np.float16))
        meta.extend(keep)
    if not Hs:
        return cid, 0
    np.savez(fout, h=np.concatenate(Hs, 0), meta=np.array(meta, np.int64),
             pos=np.array(POS))
    return cid, len(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--stride5', type=int, default=1, help='5min 决策点抽样步长')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not (os.path.exists(GRID) and os.path.exists(GRID_META)):
        print('构建 1min 网格...', flush=True)
        build_grid()
    arr, dates, codes = load_grid()
    NG1 = arr.shape[0]
    close1 = np.ascontiguousarray(arr[:, :, 3])
    pts = []
    for d in range(len(dates)):
        for b5 in range(0, NB5 - HMIN, a.stride5):
            i1 = d * NB1 + (b5 + 1) * 5 - 1        # 该 5min bar 的最后一根 1min bar
            t5 = d * NB5 + b5
            if i1 - L + 1 < 0:
                continue
            for j in range(len(CODES)):
                if np.isfinite(close1[i1 - L + 1:i1 + 1, j]).all():
                    pts.append((i1, t5, j))
    est = len(pts) * len(POS) * 512 * 2 / 1e9
    print(f'{len(pts):,} 个决策点 (5min边界, stride={a.stride5}), 预计落盘 ~{est:.2f}GB',
          flush=True)
    del arr, close1

    per = 1000
    jobs = [(c, pts[i:i + per]) for c, i in enumerate(range(0, len(pts), per))]
    t0 = time.time(); done = n = 0
    with Pool(a.workers, initializer=init_worker, initargs=(a.threads,)) as p:
        for cid, k in p.imap_unordered(run_chunk, jobs):
            done += 1
            if k > 0:
                n += k
            if done % 25 == 0:
                el = time.time() - t0
                print(f'  {done}/{len(jobs)}  {n:,} 点  {el/60:.1f}min  '
                      f'eta {el/done*(len(jobs)-done)/60:.0f}min', flush=True)
    print(f'=== 1M HIDDEN DONE === {n:,} 点  {(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
