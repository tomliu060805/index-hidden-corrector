"""全指数隐层提取: universe(480 个指数) × {L6, L8n} × last 池化。

为什么只存两个位置: 逐层扫描(out/v2_layerscan_rv.csv)显示
  - `mean` 池化全面弱于 `last`(均值 0.010 vs 0.026) -> 只留 last
  - L5/L6/L7/L8/L8n 五者几乎并列, L8n 是现行生产位置, L6 是 last 池化下三视界均值最高者
  -> 存这两个, 在 480 指数的规模上复查层排序是否翻转(算力几乎不增, 只多存一个向量)

口径与 extract_layers.py / extract_index_hidden.py 严格一致:
  L=128 上下文, 窗口自身 z-score, clip±5, 每个决策点独立前向。
决策点: bar ∈ {0,6,12,18,24,30,36} (stride 6) —— 相邻 bar 的隐层高度冗余,
  而 480 指数 × 3091 天 × 7 bar ≈ 1040 万点, 对 ΔR² 的功效远远够用。

输出: out/hidden_universe/chunk_XXXX.npz  h: float16 [n,2,512]  meta: [t, j]
用法: python extract_universe.py --workers 10 --threads 8
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

OUT = f'{_R}/out/hidden_universe'
PANEL = f'{_R}/out/index5m_universe'
L, CLIP, NBAR = 128, 5.0, 48
HMIN = 6                                   # 最短视界, 决策点须留 6 根在日内
STRIDE = 6
POS = ['L6', 'L8n']                        # 存这两个位置的 last 池化
COLS = ['open', 'high', 'low', 'close', 'volume', 'money']
_G = {}


GRID_NPY = f'{_R}/out/universe_grid.npy'
GRID_META = f'{_R}/out/universe_grid_meta.json'
# ★策展子集的**转置**紧凑网格 [J_sel, NG, 6]。
#   全量网格是 [NG, 480, 6], 读一个指数的 128 根窗口要跨 128 个不相邻页(行距 11520B),
#   实测把单点成本抬高 8 倍(67 点/秒 vs 3 指数版的 549 点/秒)。
#   转置后一个窗口是一次连续读; 144 指数只占 512MB, 可常驻。
SUB_NPY = f'{_R}/out/universe_grid_sub.npy'
SUB_META = f'{_R}/out/universe_grid_sub_meta.json'


def build_grid():
    """把年度分片拼成 [NG, J, 6] float32 网格并落盘为 .npy。

    ★ 只在主进程做一次。逐年处理而不是一次 concat —— 实测一次性 concat 71M 行
      的峰值 RSS 达 31GB, 若每个 worker 各做一次(10 workers)就是 310GB。
      落盘后 worker 用 mmap 只读共享, 常驻内存 = 1.7GB 的页缓存一份。
    """
    codes = json.load(open(f'{_R}/out/index_universe.json'))['codes']
    ci = pd.Series(np.arange(len(codes)), index=codes)
    files = sorted(glob.glob(f'{PANEL}/*.parquet'))
    dates = []
    for f in files:
        dates.append(pd.read_parquet(f, columns=['date'])['date'].unique())
    dates = np.array(sorted(set(np.concatenate(dates))))
    di = pd.Series(np.arange(len(dates)), index=dates)
    arr = np.lib.format.open_memmap(GRID_NPY, mode='w+', dtype=np.float32,
                                    shape=(len(dates) * NBAR, len(codes), len(COLS)))
    arr[:] = np.nan
    for f in files:
        df = pd.read_parquet(f)
        df['code'] = df['code'].astype(str)
        s = ci.reindex(df['code']).values
        keep = np.isfinite(s.astype(float))
        g = (di.reindex(df['date']).values * NBAR + df['bar'].values)[keep].astype(np.int64)
        sj = s[keep].astype(np.int64)
        for k, c in enumerate(COLS):
            arr[g, sj, k] = df[c].values[keep].astype(np.float32)
        del df
        print(f'  网格写入 {os.path.basename(f)}', flush=True)
    arr.flush(); del arr
    json.dump({'dates': list(map(str, dates)), 'codes': codes}, open(GRID_META, 'w'))
    print(f'网格落盘 {GRID_NPY}  ({os.path.getsize(GRID_NPY)/1e9:.2f}GB)', flush=True)


def load_grid():
    """只读 mmap 打开已落盘的全量网格 [NG, J, 6](不复制)。"""
    m = json.load(open(GRID_META))
    arr = np.load(GRID_NPY, mmap_mode='r')
    return arr, np.array(m['dates']), m['codes']


def build_sub(jsel, codes):
    """抽出策展子集并转置成 [J_sel, NG, 6] 连续布局。"""
    arr, dates, _ = load_grid()
    NG = arr.shape[0]
    out = np.lib.format.open_memmap(SUB_NPY, mode='w+', dtype=np.float32,
                                    shape=(len(jsel), NG, len(COLS)))
    for k, j in enumerate(jsel):
        out[k] = arr[:, j, :]
    out.flush(); del out, arr
    json.dump({'dates': list(map(str, dates)),
               'codes_sel': [codes[j] for j in jsel],
               'j_global': [int(j) for j in jsel]}, open(SUB_META, 'w'))
    print(f'子网格落盘 {SUB_NPY} ({os.path.getsize(SUB_NPY)/1e9:.2f}GB, '
          f'{len(jsel)} 指数)', flush=True)


def load_sub():
    m = json.load(open(SUB_META))
    return (np.load(SUB_NPY, mmap_mode='r'), np.array(m['dates']),
            m['codes_sel'], np.array(m['j_global']))


def init_worker(nthreads):
    from model import Kronos, KronosTokenizer
    from model.kronos import calc_time_stamps
    torch.set_num_threads(nthreads)
    arr, dates, codes, jglob = load_sub()
    _G['jglob'] = jglob
    T48 = [f'{9+(i*5+35)//60}:{(i*5+35)%60:02d}' for i in range(24)]      # 占位, 下面重建
    # 与既有实现一致的 48 个 bar 时间戳
    tt = (['09:%02d' % m for m in (35, 40, 45, 50, 55)] + ['10:%02d' % m for m in range(0, 60, 5)]
          + ['11:%02d' % m for m in (0, 5, 10, 15, 20, 25, 30)]
          + ['13:%02d' % m for m in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)]
          + ['14:%02d' % m for m in range(0, 60, 5)] + ['15:00'])
    assert len(tt) == 48, len(tt)
    ts = pd.to_datetime([f'{d} {t}' for d in dates for t in tt])
    _G['arr'] = arr
    _G['stamp'] = calc_time_stamps(ts.to_series()).values.astype(np.float32)
    _G['tok'] = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
    buf = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        buf['L6'] = h[:, -1, :]
    mdl.transformer[5].register_forward_hook(hook)      # block 6 (0-indexed 5)
    _G['mdl'] = mdl
    _G['buf'] = buf


def run_chunk(job):
    cid, pts = job
    fout = f'{OUT}/chunk_{cid:05d}.npz'
    if os.path.exists(fout):
        return cid, -1
    arr, stamp = _G['arr'], _G['stamp']
    tok, mdl, buf = _G['tok'], _G['mdl'], _G['buf']
    B = 32
    Hs, meta = [], []
    for i in range(0, len(pts), B):
        blk = pts[i:i + B]
        ws, sts, keep = [], [], []
        for (t, j) in blk:
            w = arr[j, t - L + 1:t + 1, :]      # ★ [J, NG, 6] 连续读
            if not np.isfinite(w).all():
                continue
            m, s = w.mean(0), w.std(0)
            ws.append(np.clip((w - m) / (s + 1e-5), -CLIP, CLIP))
            sts.append(stamp[t - L + 1:t + 1])
            keep.append((t, int(_G['jglob'][j])))      # meta 存全局 j, 与评估端对齐
        if not ws:
            continue
        x = torch.from_numpy(np.stack(ws).astype(np.float32))
        st = torch.from_numpy(np.stack(sts).astype(np.float32))
        buf.clear()
        with torch.no_grad():
            s1, s2 = tok.encode(x, half=True)
            _, ctx = mdl.decode_s1(s1, s2, st)
            h = torch.stack([buf['L6'], ctx[:, -1, :]], 1)       # [B, 2, 512]
        Hs.append(h.numpy().astype(np.float16))
        meta.extend(keep)
    if not Hs:
        return cid, 0
    np.savez(fout, h=np.concatenate(Hs, 0), meta=np.array(meta, np.int32),
             pos=np.array(POS))
    return cid, len(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not (os.path.exists(GRID_NPY) and os.path.exists(GRID_META)):
        print('首次运行: 构建共享网格...', flush=True)
        build_grid()

    arr, dates, codes = load_grid()
    NG, J = arr.shape[0], arr.shape[1]
    # ★只提取策展后的子集(地域+主题+其他行业族已砍, 见 src/curate_universe.py)
    cur = f'{_R}/out/index_universe_curated.json'
    if os.path.exists(cur):
        meta_ = json.load(open(cur))
        want = set(meta_['codes'])
        jsel = np.array([i for i, c in enumerate(codes) if c in want])
        print(f"策展清单: {len(jsel)}/{J} 个指数  ({meta_['rule']})", flush=True)
    else:
        jsel = np.arange(J)
    if not os.path.exists(SUB_NPY):
        print('构建策展子网格(转置, 连续布局)...', flush=True)
        build_sub(jsel, codes)
    sub, _, _, _ = load_sub()
    close = np.ascontiguousarray(sub[:, :, 3])     # [J_sel, NG]
    pts = []
    for t in range(L, NG - HMIN):
        b = t % NBAR
        if b > NBAR - 1 - HMIN or b % STRIDE != 0:
            continue
        ok = np.isfinite(close[:, t - L + 1:t + 1 + HMIN]).all(1)
        for j in np.where(ok)[0]:
            pts.append((t, int(j)))                # ★ j 是局部下标
    del sub, close
    est = len(pts) * len(POS) * 512 * 2 / 1e9
    print(f'{len(pts):,} 个决策点 ({len(dates)}天 × {NBAR//STRIDE}bar × {len(jsel)}指数), '
          f'预计落盘 ~{est:.1f}GB', flush=True)
    del arr

    per = 4000
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
    print(f'=== UNIVERSE HIDDEN DONE === {n:,} 点  {(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
