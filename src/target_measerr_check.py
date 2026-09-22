"""★目标测量误差检验: 把 30min 的 RV 目标从「6 根 5min 收益」换成「30 根 1min 收益」。

动机(2026-09-22, 来自 l2vol30s 项目的旁证):
  同样是 30 分钟视界, 股指期货用 1800 个 1 秒收益算 RV 时, 秒级 HAR 基线 R² = 0.836;
  本项目用 **6 根** 5min 收益算指数 RV 时, B8+B9 基线 R² 只有 0.493。
  差的那一大块很可能不是"指数更难预测", 而是**目标本身的测量噪声** ——
  它对谁都不可预测, 却坐在 R² 的分母里, 于是同时**压低基线 R²** 并**放大表观增量**。

  RV 估计量的渐近方差 ∝ 1/n(n=参与的收益个数): 6 根 -> 30 根, 测量噪声方差降 5 倍。

判据(写在看数之前):
  若换成 1min 目标后 ΔR² 掉幅 > 40%, 则 test 段那个 +0.0127 的**表述**必须改
  (「在一个测量很吵的目标上的增量」而不是「对真实波动的增量」)。
  若掉幅 < 20%, 原结论稳健, 这条威胁排除。

★其余一切不变: 同样的决策点、同样的隐层库(L8n/last)、同样的 B8+B9、同样的切分、
  同样只用 train 拟合 + λ 只在 val 选。**决定在 val 上下, test 只作已开封的附带报告。**

用法: python target_measerr_check.py
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import har_baselines as HB
import har_1m_features as H1
import extract_universe as U
from run_universe_eval import load_hidden
from run_layerscan import ridge_path, LAMS
from dm_test_v2 import dm_stat, cw_stat, qlike_loss
import open_test as OT

OUT = f'{_R}/out'
H = 6
NB1 = 240


def target_from_1m(arr1m_close, t_idx):
    """y[t] = log( 未来 H*5=30 根 1min 收益的 RV )。

    决策点 t 是 5min 网格下标: day = t//48, b = t%48;
    对应的最后一根 1min bar = day*240 + b*5 + 4;
    未来窗 = 该 bar 之后的 30 根 1min bar(build 已保证 b <= 47-H, 不跨日)。
    """
    c = arr1m_close.astype(np.float64)
    NG1, J = c.shape
    lr = np.full_like(c, np.nan)
    lr[1:] = np.log(c[1:] / c[:-1])
    lr[np.arange(NG1) % NB1 == 0] = np.nan          # 每日第一根无前收
    r2 = np.nan_to_num(lr ** 2)
    fin = np.isfinite(lr).astype(np.float64)
    cs = np.vstack([np.zeros((1, J)), np.cumsum(r2, 0)])
    cn = np.vstack([np.zeros((1, J)), np.cumsum(fin, 0)])
    day, b = t_idx // HB.NBAR, t_idx % HB.NBAR
    valid = b <= HB.NBAR - 1 - H                     # ★未来窗须留在日内(与 5min 版同规矩)
    i0 = np.clip(day * NB1 + b * 5 + 5, 0, NG1)      # 未来窗第一根 1min bar
    i1 = np.clip(i0 + H * 5, 0, NG1)                 # 开区间右端
    s = np.where(valid[:, None], cs[i1] - cs[i0], np.nan)
    n = np.where(valid[:, None], cn[i1] - cn[i0], 0.0)
    y = np.where((n >= H * 5 * 0.5) & np.isfinite(s), s, np.nan)   # 至少一半有效才算
    ok = np.isfinite(y) & (y > 0)                    # ★退化点(真零波动)剔掉, 与 5min 版同规矩
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log(y + HB.EPS)
    return out, ok, n


def target_subsampled(arr1m_close, t_idx, K=5):
    """★目标 C: 子采样 5min RV (Zhang-Mykland-Ait-Sahalia)。

    为什么需要它: 指数 1min 收益有 +0.266 的**正**自相关(成分股不同步成交 => 陈旧价格),
      所以 B(1min 收益)测的是被平滑过的波动, 有偏; 而 A(6 根 5min 收益)无偏但只有 6 个收益。
    做法: 仍用**5 分钟跨度**的收益(避开陈旧价格), 但把起点按 1 分钟错开 K=5 组, 5 组 RV 取平均。
      无偏 + 方差约降到 1/K。
    """
    c = arr1m_close.astype(np.float64)
    NG1, J = c.shape
    lp = np.log(np.where(c > 0, c, np.nan))
    day, b = t_idx // HB.NBAR, t_idx % HB.NBAR
    valid = b <= HB.NBAR - 1 - H
    anchor0 = np.clip(day * NB1 + b * 5 + 4, 0, NG1 - 1)     # 决策点那根 1min bar
    last = anchor0 + H * 5                                    # 未来窗最后一根
    acc = np.zeros((len(t_idx), J)); cntK = np.zeros((len(t_idx), J))
    for o in range(K):
        so = np.zeros((len(t_idx), J)); no = np.zeros((len(t_idx), J))
        for j in range(H + 1):
            a = anchor0 + o + 5 * j
            bnd = a + 5
            m = valid & (bnd <= last) & (bnd < NG1)
            if not m.any():
                continue
            r = lp[np.clip(bnd, 0, NG1 - 1)] - lp[np.clip(a, 0, NG1 - 1)]
            r = np.where(m[:, None] & np.isfinite(r), r, np.nan)
            so += np.nan_to_num(r ** 2); no += np.isfinite(r)
        good = no >= 3
        acc += np.where(good, so * (H / np.maximum(no, 1)), 0.0)   # 缩放到整窗 H 根
        cntK += good
    y = np.where(cntK >= 3, acc / np.maximum(cntK, 1), np.nan)
    ok = np.isfinite(y) & (y > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(y + HB.EPS), ok, cntK


def fit_report(tag, y, ok, d, F, Z, jsel, segs):
    m = ok
    Xb = np.column_stack([HB.baseline_design(d, F, len(jsel), 'B8')]
                         + [F[c] for c in H1.B9_EXTRA])
    Xb = np.nan_to_num(Xb, nan=0., posinf=0., neginf=0.)
    yy = np.where(np.isfinite(y), y, 0.0)
    tr = segs['train'] & m
    base = HB.ridge_fit(Xb, yy, tr)
    pr = ridge_path(Z, yy - base, tr, LAMS)
    lam = max(LAMS, key=lambda L: HB.r2(yy, base + pr[L], segs['val'] & m))
    full = base + pr[lam]
    q0, q1 = qlike_loss(yy, base, tr), qlike_loss(yy, full, tr)
    dts = d['date'].values
    out = {}
    print(f"\n--- {tag}  (λ={lam:.0f}) ---")
    print(f"{'段':>6s}{'n点':>11s}{'基线R²':>9s}{'+隐层R²':>10s}{'ΔR²':>9s}{'★CW t':>8s}{'QLIKE%':>9s}")
    for nm in ['train', 'val', 'TEST']:
        mm = segs[nm] & m
        if mm.sum() < 1000:
            continue
        r0, r1 = HB.r2(yy, base, mm), HB.r2(yy, full, mm)
        tcw, _ = cw_stat(yy[mm], base[mm], full[mm], dts[mm])
        ql = (1 - q1[mm].mean() / q0[mm].mean()) * 100
        out[nm] = (r0, r1, r1 - r0, tcw, ql)
        print(f'{nm:>6s}{mm.sum():>11,d}{r0:>9.4f}{r1:>10.4f}{r1-r0:>+9.4f}{tcw:>8.2f}{ql:>8.1f}%')
    return out


def main():
    print('=' * 84)
    print('★ 目标测量误差检验 —— 同一批决策点/同一隐层/同一基线, 只换目标的构造方式')
    print('   A: 未来 30min 的 RV, 用 **6 根 5min 收益**  (现行口径, 已发布结论用的就是这个)')
    print('   B: 未来 30min 的 RV, 用 **30 根 1min 收益** (同一段时间, 收益数 ×5)')
    print('=' * 84, flush=True)

    cur = json.load(open(f'{OUT}/index_universe_curated.json'))
    cu0 = pd.read_csv(f'{OUT}/universe_curated_list.csv')
    bs = set(cu0[cu0['名称'].astype(str).str.contains('B股|B指')]['代码'])
    want = set(cur['codes']) - bs
    _, _, allc = U.load_grid()
    jsel = np.array([i for i, c in enumerate(allc) if c in want])
    Hm, key = load_hidden()
    d, yA, F, codes, close = OT.build(jsel)

    m1 = json.load(open(f'{OUT}/index1m_grid_140_meta.json'))
    arr1m = np.load(f'{OUT}/index1m_grid_140.npy', mmap_mode='r')
    loc = {c: i for i, c in enumerate(m1['codes'])}
    order = np.array([loc[allc[j]] for j in jsel])
    a1 = np.asarray(arr1m[:, order, :])
    for k, v in H1.build_1m_features(a1).items():
        F[k] = v[d['t'].values, d['jloc'].values]

    # ★B 目标
    tix = np.arange(close.shape[0])
    yB_tab, okB_tab, nB_tab = target_from_1m(a1[:, :, 3], tix)
    yC_tab, okC_tab, _ = target_subsampled(a1[:, :, 3], tix)
    del arr1m, a1
    ti, jl = d['t'].values, d['jloc'].values
    yB = yB_tab[ti, jl]; okB = okB_tab[ti, jl]; nB = nB_tab[ti, jl]
    yC = yC_tab[ti, jl]; okC = okC_tab[ti, jl]

    kk = ti * 1000 + d['j'].values
    p = np.clip(np.searchsorted(key, kk), 0, len(key) - 1); good = key[p] == kk
    d = d[good].reset_index(drop=True)
    yA, yB, okB, nB = yA[good], yB[good], okB[good], nB[good]
    yC, okC = yC[good], okC[good]
    F = {k: v[good] for k, v in F.items()}
    Z = Hm[p[good], 1, :].astype(np.float32)
    del Hm
    segs = {nm: (d.seg == nm).values for nm in ['train', 'val', 'TEST']}
    okA = np.isfinite(yA)
    both = okA & okB & okC
    print(f'\n面板 {len(d):,} 点 | A 可用 {okA.sum():,} | B 可用 {okB.sum():,} | 两者都可用 {both.sum():,}')
    print(f'B 目标每点平均用到 {np.nanmean(nB[okB]):.1f} 根 1min 收益 (A 是 {H} 根 5min 收益)')

    # ---- 目标本身的性质 ----
    print('\n' + '=' * 84)
    print('目标的性质 (train+val 段, 两者都可用的点)')
    print('=' * 84)
    mtv = both & (segs['train'] | segs['val'])
    a, b = yA[mtv], yB[mtv]
    print(f'  Var(y)        A={a.var():.4f}   B={b.var():.4f}   (B 更小 = 噪声被削掉)')
    print(f'  corr(A,B)     {np.corrcoef(a, b)[0,1]:.4f}')
    # log RV 的渐近测量方差 ≈ 2/n (常波动近似)
    vA, vB = 2.0 / H, 2.0 / (H * 5)
    print(f'  理论测量噪声方差 ≈ 2/n:  A={vA:.3f} (n={H})   B={vB:.3f} (n={H*5})')
    print(f'  ★可达 R² 上限 = 1 - 噪声/Var(y):  A={1-vA/a.var():.3f}   B={1-vB/b.var():.3f}')
    print('   (完美预测「真实积分波动」时能拿到的 R²; 超不过它)')

    print(f'  Var(y) C(子采样5min)={yC[mtv].var():.4f}  corr(A,C)={np.corrcoef(a, yC[mtv])[0,1]:.4f}'
          f'  corr(B,C)={np.corrcoef(b, yC[mtv])[0,1]:.4f}')
    print(f'  ★可达 R² 上限 C = {1-(2.0/H/5)/yC[mtv].var():.3f} (无偏 + 方差降到约 1/5)')
    rA = fit_report('A  目标=6 根 5min 收益 (现行, 无偏但噪声大)', yA, both, d, F, Z, jsel, segs)
    rB = fit_report('B  目标=30 根 1min 收益 (低噪声但被陈旧价格平滑, rho1=+0.266)', yB, both, d, F, Z, jsel, segs)
    rC = fit_report('★C 目标=子采样 5min RV (无偏 + 低噪声) —— 应以此为准', yC, both, d, F, Z, jsel, segs)

    print('\n' + '=' * 84)
    print('判读')
    print('=' * 84)
    print(f"{'段':>6s}{'A基线':>9s}{'A ΔR²':>9s}{'B基线':>9s}{'B ΔR²':>9s}{'C基线':>9s}{'★C ΔR²':>9s}{'C/A':>7s}")
    for nm in ['train', 'val', 'TEST']:
        if nm in rA and nm in rC:
            print(f'  {nm:>4s}{rA[nm][0]:>9.4f}{rA[nm][2]:>+9.4f}{rB[nm][0]:>9.4f}{rB[nm][2]:>+9.4f}'
                  f'{rC[nm][0]:>9.4f}{rC[nm][2]:>+9.4f}{rC[nm][2]/rA[nm][2]:>7.2f}')
    drop = (1 - rC['val'][2] / rA['val'][2]) * 100
    print(f'\n  ★以 C 为准, val 段增量掉幅 = {drop:.0f}%   (预注册: >40% 需改表述 / <20% 排除威胁)')
    print('  ' + ('=> 需要改表述' if drop > 40 else '=> 威胁排除' if drop < 20 else '=> 中间地带, 按实际幅度报'))
    print(f"  test/val 增量比: A={rA['TEST'][2]/rA['val'][2]:.2f}  C={rC['TEST'][2]/rC['val'][2]:.2f}")


if __name__ == '__main__':
    main()
