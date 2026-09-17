"""HAR 基线阶梯 + 逐层扫描 + 残差修正 vs 等权集成。

预注册: docs/PREREG_v2_layerscan.md (写于 2026-09-17, 数字产生之前)
★ 本脚本对 test/holdout 硬封锁。任何评估都不读这两段, 解锁需显式改 SEAL_OK。

阶段:
  baselines  B0..B5 基线阶梯, 三视界, 选出 val 最强的 B*
  scan       60 格逐层扫描 (10位置 × 2池化 × 3视界), 岭头
  seeds      对 L8n 与扫描最优层做 5 种子 GatedLinear 对照
  combine    T1(同回归) / T2(残差修正) / T3(等权集成) 三法对照
  nulls      M4 最笨版本 / M3 正交化 / S6 安慰剂+功效正对照

用法: python run_layerscan.py --stage baselines --H 6,12,24
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB

OUT = f'{_R}/out'
CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']
TRAIN_END, VAL_END = '2023-04-27', '2024-06-07'
SEAL_OK = False                      # ★ test/holdout 封存开关, 本次全程 False
POS_NAMES = ['L0'] + [f'L{k}' for k in range(1, 9)] + ['L8n']
POOLS = ['last', 'mean']


# ------------------------------------------------------------------ 面板
def close_grid(full=False):
    """full=False 只返回 close(向后兼容); full=True 返回 OHLCV 六列 —— B8 要用。
    ★原实现只取 close, 而 Kronos 的输入是六列 => 基线看不见量和高低价, 隐层看得见。"""
    df = pd.read_parquet(f'{OUT}/index5m.parquet')
    dates = np.array(sorted(df['date'].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    ci = pd.Series(np.arange(len(CODES)), index=CODES)
    g = di.reindex(df['date']).values * HB.NBAR + df['bar'].values
    s = ci.reindex(df['code']).values
    NG = len(dates) * HB.NBAR
    amt = 'amount' if 'amount' in df.columns else 'money'
    cols = {}
    for nm, src in [('open', 'open'), ('high', 'high'), ('low', 'low'),
                    ('close', 'close'), ('volume', 'volume'), ('money', amt)]:
        a = np.full((NG, len(CODES)), np.nan)
        a[g, s] = df[src].values
        cols[nm] = a
    if not full:
        return cols['close'], dates
    return cols, dates


def build_panel(H, kind='rv'):
    """返回 (d, y, F) —— d 含 t/j/date/bar/seg, y 是 log 目标, F 是逐点特征 dict。"""
    cache = f'{OUT}/har_panel_H{H}_{kind}.npz'
    G, dates = close_grid(full=True)
    close = G['close']
    feats = HB.build_features(close)
    feats['ewma'] = HB.ewma_rv(close, 0.94)              # RiskMetrics 日频惯用值
    # ★B8 扩展: 成交额多尺度 / 日内累计RV / 隔夜跳空 / 跳跃分解 / 高低价区间 / 日历
    ex = HB.build_ohlcv_features(G['open'], G['high'], G['low'], G['close'],
                                 G['volume'], G['money'])
    ex.pop('_dow_raw', None)
    ex.update(HB.build_calendar(dates, close.shape[0], close.shape[1]))
    feats.update(ex)
    ytab = HB.future_target(close, H, kind)
    feats['b0'] = HB.past_target(close, H, kind)         # B0 零拟合: 同统计量的过去值
    # 退化点必须分两类统计:
    #   allnan  = 该指数当时没有数据(例如中证1000在2014-10-17之前) —— 不是数据缺陷
    #   degen   = 有数据但未来窗波动恰好为 0 —— 真退化, log 后是 EPS 的对数, 必须丢
    # ★np.nansum 对全 NaN 切片返回 0.0, 若不先判 isfinite.any() 会把前者误计为后者。
    lr = HB.log_returns(close)
    NG = close.shape[0]; bar = np.arange(NG) % HB.NBAR
    degen = np.zeros_like(close, bool); allnan = np.zeros_like(close, bool)
    for i in range(NG - H):
        if bar[i] > HB.NBAR - 1 - H:
            continue
        seg = lr[i + 1:i + 1 + H]
        fin = np.isfinite(seg).any(0)
        allnan[i] = ~fin
        with np.errstate(invalid='ignore'):
            v = np.nansum(seg ** 2, 0) if kind == 'rv' else np.nanmean(np.abs(seg), 0)
        degen[i] = fin & (np.nan_to_num(v, nan=1.0) <= 0)

    need = (['rv12', 'rv48', 'rv240', 'sqrt_rq', 'rs_up', 'rs_dn', 'aret', 'ewma', 'b0']
            + [f'lag{k}' for k in range(1, HB.NLAG + 1)] + HB.B8_EXTRA)
    ok = np.isfinite(ytab) & ~degen
    for k in need:
        ok &= np.isfinite(feats[k])
    ok &= (bar[:, None] <= HB.NBAR - 1 - H)
    ti, ji = np.where(ok)
    d = pd.DataFrame({'t': ti, 'j': ji})
    d['date'] = dates[ti // HB.NBAR]
    d['bar'] = ti % HB.NBAR
    d['seg'] = np.where(d.date <= TRAIN_END, 'train',
               np.where(d.date <= VAL_END, 'val',
               np.where(d.date <= '2025-07-17', 'test', 'holdout')))
    y = ytab[ti, ji]
    F = {k: feats[k][ti, ji] for k in need}
    n_deg, n_nan = int(degen.sum()), int(allnan.sum())
    tot = n_deg + n_nan + len(d)
    print(f'  H={H} {kind}: 有效 {len(d):,} ({len(d)/tot*100:.2f}%)  '
          f'无数据 {n_nan:,} ({n_nan/tot*100:.2f}%)  '
          f'真零波动 {n_deg:,} ({n_deg/tot*100:.3f}%)  |  '
          f'train {int((d.seg=="train").sum()):,} / val {int((d.seg=="val").sum()):,}', flush=True)
    return d, y, F


def load_layers():
    """[n,10,2,512] float16 + meta(t,j) -> 按 (t,j) 建索引"""
    fs = sorted(glob.glob(f'{OUT}/hidden_layers/chunk_*.npz'))
    Hs, Ms = [], []
    for f in fs:
        z = np.load(f)
        Hs.append(z['h']); Ms.append(z['meta'])
    Hm = np.concatenate(Hs); M = np.concatenate(Ms)
    key = M[:, 0].astype(np.int64) * 100 + M[:, 1].astype(np.int64)
    order = np.argsort(key)
    return Hm[order], key[order]


def align(d, key_sorted, Hm):
    k = d['t'].values.astype(np.int64) * 100 + d['j'].values.astype(np.int64)
    pos = np.searchsorted(key_sorted, k)
    pos = np.clip(pos, 0, len(key_sorted) - 1)
    good = key_sorted[pos] == k
    return pos, good


# ------------------------------------------------------------------ 评估
def segmask(d, s):
    if not SEAL_OK and s in ('test', 'holdout'):
        raise RuntimeError(f'封存段 {s} 被访问 —— SEAL_OK=False')
    return (d.seg == s).values


def eval_pred(d, y, yh):
    tr, va = segmask(d, 'train'), segmask(d, 'val')
    return {'train_R2': HB.r2(y, yh, tr), 'val_R2': HB.r2(y, yh, va),
            'val_QLIKE': HB.qlike(y, yh, va)}


# ------------------------------------------------------------------ 阶段
def stage_baselines(Hs, kind):
    rows = []
    for H in Hs:
        d, y, F = build_panel(H, kind)
        tr = segmask(d, 'train')
        # B0 零拟合: 同一个统计量在过去 H 根上的取值, 直接当预测(无参数)
        yh0 = F['b0']
        rows.append(dict(H=H, base='B0', **eval_pred(d, y, yh0),
                         desc=HB.BASELINE_DESC['B0']))
        for b in HB.BASELINES[1:]:
            X = HB.baseline_design(d, F, len(CODES), b)
            yh = HB.ridge_fit(X, y, tr)
            rows.append(dict(H=H, base=b, **eval_pred(d, y, yh),
                             desc=HB.BASELINE_DESC[b], ncol=X.shape[1]))
        del d, y, F
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT}/v2_baselines_{kind}.csv', index=False)
    print(f'\n=== 基线阶梯 ({kind}) ===')
    for H in Hs:
        s = r[r.H == H].sort_values('val_R2', ascending=False)
        print(f'\n  H={H} ({H*5}min)   val R² 排序:')
        for _, x in s.iterrows():
            star = ' ★B*' if x.base == s.iloc[0].base else ''
            print(f'    {x.base}  val R²={x.val_R2:7.4f}  QLIKE={x.val_QLIKE:7.4f}  '
                  f'train={x.train_R2:7.4f}  {x.desc}{star}')
    return r


def ridge_path(X, y, tr, lams):
    """Gram 只算一次, 对多个 λ 求解。逐列标准化(各层尺度差 400 倍, 不标准化比的是尺度)。"""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xn = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    A = Xn[tr]
    G = A.T @ A
    b0 = A.T @ y[tr]
    I = np.eye(G.shape[0]); I[-1, -1] = 0.0            # 截距不罚
    return {lam: Xn @ np.linalg.solve(G + lam * I, b0) for lam in lams}


# ★λ 网格必须够宽: 首跑 60 格全部选到上界 1000, 说明最优正则在网格外,
#   每格都在次优点评估会扭曲层排序(噪声大的层从更强收缩里获益更多)。
LAMS = [10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]


def stage_scan(Hs, kind):
    """60 格逐层扫描: 10 位置 × 2 池化 × 3 视界, 岭头, 每格在 LAMS 上选 λ。"""
    print('载入逐层隐层库...', flush=True)
    Hm, key = load_layers()
    print(f'  {Hm.shape}  {Hm.nbytes/1e9:.1f}GB', flush=True)
    rows = []
    for H in Hs:
        d, y, F = build_panel(H, kind)
        pos, good = align(d, key, Hm)
        d, y = d[good].reset_index(drop=True), y[good]
        F = {k: v[good] for k, v in F.items()}
        pos = pos[good]
        print(f'  H={H} 对齐后 {len(d):,} 点', flush=True)
        tr, va = segmask(d, 'train'), segmask(d, 'val')
        Xb = HB.baseline_design(d, F, len(CODES), 'B4')          # B* = B4
        base = HB.ridge_fit(Xb, y, tr)
        r2b = HB.r2(y, base, va)
        rows.append(dict(H=H, pos='(B*=B4)', pool='-', lam=np.nan,
                         val_R2=r2b, d_val=0.0, train_R2=HB.r2(y, base, tr)))
        print(f'    B* val R²={r2b:.4f}', flush=True)
        resid = y - base          # ★ 学的是"最强基线 B4 的残差", 基线系数固定不再重估
        for pi, pname in enumerate(POS_NAMES):
            for qi, qname in enumerate(POOLS):
                Hs_ = Hm[pos, pi, qi, :].astype(np.float32)
                # --- T2r: 残差修正(主口径) —— B* 固定, 隐层只学它剩下的 ---
                pr = ridge_path(Hs_, resid, tr, LAMS)
                bl = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va))
                yh_r = base + pr[bl]
                rr = HB.r2(y, yh_r, va)
                # --- T1: 联合回归(对照) —— 基线系数被一起重估并收缩 ---
                pj = ridge_path(np.column_stack([Xb, Hs_]), y, tr, LAMS)
                bj = max(LAMS, key=lambda L: HB.r2(y, pj[L], va))
                rj = HB.r2(y, pj[bj], va)
                # 分指数(预注册判据3), 用主口径
                per = [HB.r2(y, yh_r, va & (d.j == j).values) -
                       HB.r2(y, base, va & (d.j == j).values) for j in range(len(CODES))]
                rows.append(dict(H=H, pos=pname, pool=qname, lam=bl,
                                 val_R2=rr, d_val=rr - r2b, train_R2=HB.r2(y, yh_r, tr),
                                 val_R2_joint=rj, d_val_joint=rj - r2b, lam_joint=bj,
                                 **{f'd_idx{j}': per[j] for j in range(len(CODES))}))
                print(f'    {pname:4s}/{qname:4s} λ={bl:6.0f}  残差口径 val R²={rr:.4f} '
                      f'Δ={rr-r2b:+.4f}   联合口径 Δ={rj-r2b:+.4f}   '
                      f'分指数Δ={np.round(per,4)}', flush=True)
                del Hs_, pr, pj
        del d, y, F
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT}/v2_layerscan_{kind}.csv', index=False)
    print(f'\n写出 {OUT}/v2_layerscan_{kind}.csv')
    return r


def gated_fit(X, target, tr, va, seed, rank=32, lr=1e-3, epochs=60, bs=8192, wd=1e-4):
    """项目现有的 GatedLinear 修正器(低秩+门控), val 早停。返回全样本预测。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(30)

    class GatedLinear(nn.Module):
        def __init__(self, dim, rank):
            super().__init__()
            self.proj_r = nn.Linear(dim, rank); self.proj_g = nn.Linear(dim, rank)
            self.head = nn.Linear(rank, 1)

        def forward(self, x):
            return self.head(torch.sigmoid(self.proj_g(x)) * self.proj_r(x)).squeeze(-1)

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(target.astype(np.float32))
    net = GatedLinear(X.shape[1], rank)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    idx = np.where(tr)[0]
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            ((net(Xt[b]) - yt[b]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(Xt[va]).numpy() - target[va]) ** 2).mean()
        if vl < best - 1e-7:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return net(Xt).numpy()


def stage_combine(Hs, kind, pos_name='L8n', seeds=5):
    """T1 联合回归 / T2r 残差+岭 / T2 残差+GatedLinear(多种子) / T3 等权集成。

    T3 的动机: arXiv 2607.05291 报告 TTM 与 Log-HAR 的**等权集成**胜过任一单独,
    这是"为什么用残差学习而不用简单集成"这个必问问题的现成对照组。
    """
    pi = POS_NAMES.index(pos_name)
    Hm, key = load_layers()
    rows = []
    for H in Hs:
        d, y, F = build_panel(H, kind)
        p, good = align(d, key, Hm)
        d, y = d[good].reset_index(drop=True), y[good]
        F = {k: v[good] for k, v in F.items()}
        Z = Hm[p[good], pi, 0, :].astype(np.float32)          # last 池化
        tr, va = segmask(d, 'train'), segmask(d, 'val')
        Xb = HB.baseline_design(d, F, len(CODES), 'B4')
        base = HB.ridge_fit(Xb, y, tr)
        r2b = HB.r2(y, base, va)
        resid = y - base
        print(f'\n  H={H}  B* val R²={r2b:.4f}   位置={pos_name}/last  n={len(d):,}', flush=True)
        rows.append(dict(H=H, method='B*(HAR-RS)', seed=-1, val_R2=r2b, d_val=0.0))

        def rec(name, yh, seed=-1):
            v = HB.r2(y, yh, va)
            rows.append(dict(H=H, method=name, seed=seed, val_R2=v, d_val=v - r2b))
            print(f'    {name:28s} seed={seed:2d}  val R²={v:.4f}  Δ={v-r2b:+.4f}', flush=True)
            return v

        # T1 联合回归
        pj = ridge_path(np.column_stack([Xb, Z]), y, tr, LAMS)
        rec('T1 联合回归(岭)', pj[max(LAMS, key=lambda L: HB.r2(y, pj[L], va))])
        # T2r 残差 + 岭
        pr = ridge_path(Z, resid, tr, LAMS)
        lr_ = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va))
        rec('T2r 残差修正(岭)', base + pr[lr_])
        # T3 等权集成: 隐层单独预测目标, 与 B* 各占一半
        ps = ridge_path(Z, y, tr, LAMS)
        ls = max(LAMS, key=lambda L: HB.r2(y, ps[L], va))
        rec('   (隐层单独预测)', ps[ls])
        rec('T3 等权集成(½B*+½隐层)', 0.5 * base + 0.5 * ps[ls])
        # T2 残差 + GatedLinear, 多种子
        vs = []
        for s in range(seeds):
            vs.append(rec('T2 残差修正(GatedLinear)', base + gated_fit(Z, resid, tr, va, s), s))
        print(f'    -> T2 多种子 Δ 均值={np.mean(vs)-r2b:+.4f}  sd={np.std(vs):.4f}  '
              f'全正={all(v > r2b for v in vs)}', flush=True)
        del d, y, F, Z
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT}/v2_combine_{kind}_{pos_name}.csv', index=False)
    print(f'\n写出 {OUT}/v2_combine_{kind}_{pos_name}.csv')
    return r


def load_scalar_store():
    """原 out/hidden 库的 meta = [t, j, nll, ent] —— M4 最笨版本要用这两个标量。"""
    fs = sorted(glob.glob(f'{OUT}/hidden/chunk_*.npz'))
    Ms = [np.load(f)['meta'] for f in fs]
    M = np.concatenate(Ms)
    key = M[:, 0].astype(np.int64) * 100 + M[:, 1].astype(np.int64)
    o = np.argsort(key)
    return key[o], M[o, 2], M[o, 3]


def stage_nulls(Hs, kind, pos_name='L8n'):
    """判负库为这个形状的研究点名的三个陷阱: M4 / M3 / S6。"""
    pi = POS_NAMES.index(pos_name)
    Hm, key = load_layers()
    skey, snll, sent = load_scalar_store()
    rows = []
    for H in Hs:
        d, y, F = build_panel(H, kind)
        p, good = align(d, key, Hm)
        d, y = d[good].reset_index(drop=True), y[good]
        F = {k: v[good] for k, v in F.items()}
        Z = Hm[p[good], pi, 0, :].astype(np.float32)
        tr, va = segmask(d, 'train'), segmask(d, 'val')
        Xb = HB.baseline_design(d, F, len(CODES), 'B4')
        base = HB.ridge_fit(Xb, y, tr); r2b = HB.r2(y, base, va)
        resid = y - base
        print(f'\n  H={H}  B*={r2b:.4f}  n={len(d):,}', flush=True)

        def inc(name, Zx, target=None, note=''):
            t_ = resid if target is None else target
            pr = ridge_path(Zx, t_, tr, LAMS)
            b = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va))
            v = HB.r2(y, base + pr[b], va)
            rows.append(dict(H=H, check=name, val_R2=v, d_val=v - r2b, note=note))
            print(f'    {name:34s} Δ={v-r2b:+.4f}   {note}', flush=True)
            return v - r2b

        d_full = inc('隐层 512 维 (参照)', Z)

        # ---- M4: 最笨的同信息版本 —— 两个标量 nll/ent ----
        kk = d['t'].values.astype(np.int64) * 100 + d['j'].values.astype(np.int64)
        ps = np.searchsorted(skey, kk); ps = np.clip(ps, 0, len(skey) - 1)
        hit = skey[ps] == kk
        if hit.sum() > 1000:
            S = np.column_stack([np.where(hit, snll[ps], np.nan),
                                 np.where(hit, sent[ps], np.nan)])
            m = np.isfinite(S).all(1)
            if m.all():
                d_s = inc('[M4] 只用 nll+ent 两个标量', S)
                print(f'      -> 标量版拿到 512 维版的 {d_s/max(d_full,1e-9)*100:.1f}%', flush=True)
            else:
                print(f'      [M4] 标量库只覆盖 {m.mean()*100:.0f}% 的点, 跳过', flush=True)
        else:
            print('      [M4] 标量库(out/hidden)与本视界决策点不重合, 跳过', flush=True)

        # ---- M3: 只是"更多已实现波动尺度"换个名字? ----
        close, _ = close_grid()
        lr = HB.log_returns(close); r2_ = lr ** 2
        extra = []
        for w in (3, 6, 24, 96, 480):
            v = np.log(pd.DataFrame(r2_).rolling(w, min_periods=max(2, w // 2)).sum().values / w + HB.EPS)
            extra.append(v[d['t'].values, d['j'].values])
        Xrich = np.column_stack([Xb] + extra)
        ok = np.isfinite(Xrich).all(1)
        rich = HB.ridge_fit(Xrich[ok], y[ok], tr[ok])
        r2_rich = HB.r2(y[ok], rich, va[ok])
        pr = ridge_path(Z[ok], y[ok] - rich, tr[ok], LAMS)
        b = max(LAMS, key=lambda L: HB.r2(y[ok], rich + pr[L], va[ok]))
        v = HB.r2(y[ok], rich + pr[b], va[ok])
        rows.append(dict(H=H, check='[M3] 基线扩到7个RV尺度后的增量',
                         val_R2=v, d_val=v - r2_rich,
                         note=f'扩尺度基线={r2_rich:.4f}(原{r2b:.4f})'))
        print(f'    [M3] 基线扩到7个RV尺度: 基线 {r2b:.4f}->{r2_rich:.4f}, '
              f'隐层增量 {d_full:+.4f}->{v-r2_rich:+.4f}', flush=True)

        # ---- S6a 安慰剂 A: **按天整体**打乱 —— 把 A 天的隐层配给 B 天 ----
        # 这是预注册写的那个。它切断隐层与时间的一切联系, Δ 必须塌到 0 附近。
        rng = np.random.default_rng(0)
        dl = d['date'].values
        udates = np.array(sorted(set(dl)))
        shuf = dict(zip(udates, rng.permutation(udates)))
        # 同一天内按 (j, bar) 排序后与目标日的同序位配对, 保证是"整天换掉"
        idx_by = {}
        for key_, g in d.groupby('date').indices.items():
            q = np.array(g)
            idx_by[key_] = q[np.lexsort((d['bar'].values[q], d['j'].values[q]))]
        permA = np.arange(len(d))
        for key_, q in idx_by.items():
            src = idx_by[shuf[key_]]
            permA[q] = src[np.arange(len(q)) % len(src)]
        inc('[S6a] 安慰剂·按天整体打乱', Z[permA], note='应塌到 0 附近')

        # ---- S6b 安慰剂 B: **日内**打乱 —— 保留"今天是什么样的一天", 只打乱 bar ----
        # 不是零基准, 是一把尺子: 它量出增量里有多少是日级的、多少是 bar 级的。
        permB = np.arange(len(d))
        for _, g in d.groupby('date').indices.items():
            q = np.array(g); permB[q] = rng.permutation(q)
        inc('[S6b] 日内打乱(跨bar+跨指数)', Z[permB],
            note='与真值之比 = 日级成分占比(含跨指数混淆)')

        # ---- S6d: 只在 (日期, 指数) 内打乱 —— 把 S6b 的跨指数混淆排掉 ----
        # S6b 同时打乱了 bar 和三个指数; 若隐层信息是全市场共同的日级波动水平,
        # 跨指数混合无害, 两者会接近。差值就是"指数特异"成分。
        permD = np.arange(len(d))
        for _, q in d.groupby(['date', 'j']).indices.items():
            q = np.array(q); permD[q] = rng.permutation(q)
        inc('[S6d] 同指数同日内打乱(只打乱bar)', Z[permD],
            note='★含前视: 会取到当天更晚时刻的隐层')

        # ---- S6e: 只用**当日第一根 bar** 的隐层 —— 严格因果的"日级"版本 ----
        # ★S6b/S6d 会把当天更晚的隐层配给当前点 = 未来信息, 不能直接读成"日级就够了"。
        #   本检验只用 09:35 那根的隐层(早于当日全部决策点), 若增量仍在, 结论才成立,
        #   且立刻可落地: 每天 1 次前向而不是 36~48 次。
        first = {}
        for key_, q in d.groupby(['date', 'j']).indices.items():
            q = np.array(q)
            first[key_] = q[np.argmin(d['bar'].values[q])]
        permE = np.array([first[(dt, jj)] for dt, jj in
                          zip(d['date'].values, d['j'].values)])
        inc('[S6e] 只用当日首根bar隐层(严格因果)', Z[permE],
            note='与真值之比 = 可落地的日级成分')

        # ---- S6c 功效正对照: 把 12 根窗口的**全部**信息抽掉再喂回 ----
        # ★第一版只抽了 rv12 却留着 rs_up/rs_dn, 而 rs_up+rs_dn≈rv12 ⇒ 等于没抽,
        #   正对照必然给 0, 会被误读成"管线没功效"。这里把 12 根尺度整个拿掉。
        Xw = np.column_stack([F['rv48'], F['rv240'],
                              pd.get_dummies(d['bar']).values.astype(float),
                              pd.get_dummies(d['j']).values.astype(float)])
        weak = HB.ridge_fit(Xw, y, tr); r2w = HB.r2(y, weak, va)
        back = np.column_stack([F['rv12'], F['rs_up'], F['rs_dn'], F['aret']])
        pr = ridge_path(back, y - weak, tr, LAMS)
        b = max(LAMS, key=lambda L: HB.r2(y, weak + pr[L], va))
        v = HB.r2(y, weak + pr[b], va)
        rows.append(dict(H=H, check='[S6c] 功效正对照(抽掉12根尺度再喂回)',
                         val_R2=v, d_val=v - r2w, note=f'弱基线={r2w:.4f}'))
        print(f'    [S6c] 功效正对照: 弱基线 {r2w:.4f}, 喂回 12根尺度后 Δ={v-r2w:+.4f} '
              f'(必须明显为正, 否则管线无功效)', flush=True)
        del d, y, F, Z
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT}/v2_nulls_{kind}.csv', index=False)
    print(f'\n写出 {OUT}/v2_nulls_{kind}.csv')
    return r


def pit_pct_same_bar(close, feat, win=252):
    """当前值在**过去 win 个交易日同一 bar** 的分布里的分位 —— 严格 PIT。

    为什么要按 bar 分组: 日内节律让 09:40 的波动天生是 15:00 的 3.4 倍,
    不分 bar 直接算分位会把"现在是早盘"误当成"现在很极端"。
    pandas rolling(win).rank(pct=True) 的窗口含当前值本身, 不含任何未来值。
    """
    NG, J = feat.shape
    out = np.full_like(feat, np.nan)
    for b in range(HB.NBAR):
        idx = np.arange(b, NG, HB.NBAR)
        sub = pd.DataFrame(feat[idx])
        out[idx] = sub.rolling(win, min_periods=60).rank(pct=True).values
    return out


def stage_regime(Hs, kind, pos_name='L8n', base_id='B6'):
    """按"过去已实现波动在过去一年同时段的分位"分桶, 看增量是否在极端波动时更大。

    ★三条读数纪律:
      1) 桶内 R² 不可跨桶比较 —— 分母是桶内 y 的方差, 高波动桶天生不同。
      2) 必须并报**绝对 MSE 下降**, 那才是风控关心的"少错多少"。
      3) 报整条曲线而非最好的桶 —— 单调才有说服力, 尖峰更可能是分桶搜索的产物。
    """
    pi = POS_NAMES.index(pos_name)
    Hm, key = load_layers()
    EDGES = [0, .5, .7, .8, .9, .95, 1.01]
    rows = []
    for H in Hs:
        d, y, F = build_panel(H, kind)
        close, _ = close_grid()
        pct_tab = pit_pct_same_bar(close, HB.build_features(close)['rv12'])
        p, good = align(d, key, Hm)
        d, y = d[good].reset_index(drop=True), y[good]
        F = {k: v[good] for k, v in F.items()}
        pct = pct_tab[d['t'].values, d['j'].values]
        Z = Hm[p[good], pi, 0, :].astype(np.float32)
        tr, va = segmask(d, 'train'), segmask(d, 'val')
        Xb = HB.baseline_design(d, F, len(CODES), base_id)
        base = HB.ridge_fit(Xb, y, tr)
        pr = ridge_path(Z, y - base, tr, LAMS)
        lam = max(LAMS, key=lambda L: HB.r2(y, base + pr[L], va))
        full = base + pr[lam]
        e0, e1 = (y - base) ** 2, (y - full) ** 2
        ok = np.isfinite(pct)
        print(f"\n  H={H} ({H*5}min) 基线={base_id}  val 分桶 "
              f"(分位缺失 {int((~ok&va).sum()):,} 点不计)", flush=True)
        print(f"    {'分位桶':<12s}{'n':>9s}{'基线R²':>9s}{'+隐层R²':>10s}{'ΔR²':>9s}"
              f"{'基线MSE':>10s}{'MSE下降':>10s}{'下降%':>8s}")
        for a_, b_ in zip(EDGES[:-1], EDGES[1:]):
            m = va & ok & (pct >= a_) & (pct < b_)
            if m.sum() < 500:
                continue
            r0, r1 = HB.r2(y, base, m), HB.r2(y, full, m)
            m0, m1 = e0[m].mean(), e1[m].mean()
            lab = f'{a_*100:.0f}-{b_*100 if b_<=1 else 100:.0f}%'
            rows.append(dict(H=H, bucket=lab, n=int(m.sum()), base_r2=r0, full_r2=r1,
                             d_r2=r1 - r0, base_mse=m0, d_mse=m0 - m1,
                             d_mse_pct=(m0 - m1) / m0 * 100))
            print(f'    {lab:<12s}{m.sum():>9,}{r0:>9.4f}{r1:>10.4f}{r1-r0:>+9.4f}'
                  f'{m0:>10.4f}{m0-m1:>+10.4f}{(m0-m1)/m0*100:>7.1f}%')
        del d, y, F, Z
    pd.DataFrame(rows).to_csv(f'{OUT}/v2_regime_{kind}_{base_id}.csv', index=False)
    print(f'\n写出 {OUT}/v2_regime_{kind}_{base_id}.csv')
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='baselines')
    ap.add_argument('--base', default='B6')
    ap.add_argument('--pos', default='L8n')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--H', default='6,12,24')
    ap.add_argument('--kind', default='rv')
    a = ap.parse_args()
    Hs = [int(x) for x in a.H.split(',')]
    print(f'口径={a.kind}  视界={Hs}  SEAL_OK={SEAL_OK}', flush=True)
    if a.stage == 'baselines':
        stage_baselines(Hs, a.kind)
    elif a.stage == 'scan':
        stage_scan(Hs, a.kind)
    elif a.stage == 'combine':
        stage_combine(Hs, a.kind, a.pos, a.seeds)
    elif a.stage == 'nulls':
        stage_nulls(Hs, a.kind, a.pos)
    elif a.stage == 'regime':
        stage_regime(Hs, a.kind, a.pos, a.base)
    else:
        raise SystemExit(f'阶段 {a.stage} 尚未实现')


if __name__ == '__main__':
    main()
