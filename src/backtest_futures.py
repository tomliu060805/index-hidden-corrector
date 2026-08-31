"""任务1+2: IC/IM 期货真实口径回测 + 加仓半边不对称化

口径:
  - 主力连续(shift1防前视), 合约内5m收益+同合约隔夜(roll视为前收盘完成, roll日计2×w成本)
  - 全天净值 = 隔夜 + 48根日内, 符合净值记账铁律(收盘-收盘含隔夜)
  - 权重: 指数模型预报 (500→IC, 1000→IM), bar t 收盘决策作用于 t+1 起, 跨日ffill
  - 成本场景: A对称0.5bp / B对称1bp / C平今现实(加仓0.55bp, 日内减仓3.77bp, roll按平昨0.55×2)
  - 变体: SYM_V1/SYM_V2 对称VT(cap3) | LEV 仅加仓floor1 | ASYM(θ) pv2<θ·pv1才加仓 | 带δ无交易带
  - θ∈{0.85,0.9,0.95,1.0}×δ∈{0,0.15,0.3} 只在 train+val 选(成本C), test/holdout 冻结报告
"""
import os
import numpy as np, pandas as pd

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD = {'IC': 1, 'IM': 2}          # product -> npz j
CAP = 3.0
BARS_Y = 242.0

z = np.load(f'{B}/out/paper1_preds.npz')
pv1_all, pv2_all = np.exp(z['yh0']), np.exp(z['yh0'] + z['p2'])
fut = pd.read_parquet(f'{B}/out/futures_dom5m.parquet')

seg_of = lambda dates: np.where(dates <= '2023-04-27', 'train',
                       np.where(dates <= '2024-06-07', 'val',
                       np.where(dates <= '2025-07-17', 'test', 'holdout')))

def build_product(prod):
    j = PROD[prod]
    f = fut[fut['product'] == prod].sort_values(['date', 'bar']).reset_index(drop=True)
    days = np.array(sorted(f.date.unique()))
    Nd = len(days)
    di = pd.Series(np.arange(Nd), index=days)
    g = di.reindex(f.date).values * 48 + f.bar.values
    r = np.full(Nd * 48, np.nan)
    lc = np.log(f.close.values)
    ret = f.ret.values.copy()
    b0 = f.bar.values == 0
    ret[b0] = f.ov_ret.values[b0] + (lc[b0] - np.log(f.open.values[b0]))
    r[g] = ret
    roll_day = f.groupby('date')['roll'].first().reindex(days).values.astype(bool)

    # 预报 -> 期货网格
    mj = z['j'] == j
    fmap = pd.Series(np.arange(mj.sum()), index=pd.MultiIndex.from_arrays(
        [z['date'][mj], z['bar'][mj]]))
    pv1 = np.full(Nd * 48, np.nan); pv2 = np.full(Nd * 48, np.nan)
    dd = np.repeat(days, 48); bb = np.tile(np.arange(48), Nd)
    idx = fmap.reindex(pd.MultiIndex.from_arrays([dd, bb]))
    ok = idx.notna().values
    pv1[ok] = pv1_all[mj][idx.values[ok].astype(int)]
    pv2[ok] = pv2_all[mj][idx.values[ok].astype(int)]
    # tgt 用指数时间轴 train 段中位数 (与前一致)
    tr_idx = (z['seg'] == 'train') & mj
    tgt1, tgt2 = np.median(pv1_all[tr_idx]), np.median(pv2_all[tr_idx])
    seg = np.repeat(seg_of(days), 48)
    return dict(r=r, pv1=pv1, pv2=pv2, tgt1=tgt1, tgt2=tgt2, seg=seg,
                days=days, roll_day=roll_day, Nd=Nd)

def weights(P, kind, theta=1.0, delta=0.0):
    pv1, pv2 = P['pv1'], P['pv2']
    if kind == 'SYM_V1': w_raw = np.clip(P['tgt1'] / pv1, 0, CAP)
    elif kind == 'SYM_V2': w_raw = np.clip(P['tgt2'] / pv2, 0, CAP)
    elif kind == 'BH': return np.ones(P['Nd'] * 48)
    else:                                            # LEV / ASYM
        w_raw = np.clip(P['tgt2'] / pv2, 1.0, CAP)
        if theta < 1.0:
            gate = pv2 < theta * pv1
            w_raw = np.where(gate, w_raw, 1.0)
    w = pd.Series(w_raw).ffill().fillna(1.0).values
    if delta > 0:                                     # 无交易带
        out = np.empty_like(w); cur = w[0]
        for i in range(len(w)):
            if abs(w[i] - cur) >= delta: cur = w[i]
            out[i] = cur
        w = out
    return w

def run(P, w, cost_mode):
    r = P['r']
    wl = np.r_[1.0, w[:-1]]                           # bar t 收益用 t-1 决策的仓位
    dw = w - np.r_[1.0, w[:-1]]
    if cost_mode == 'A': c = 0.5e-4 * np.abs(dw)
    elif cost_mode == 'B': c = 1.0e-4 * np.abs(dw)
    else:                                             # C 平今现实
        c = np.where(dw > 0, 0.55e-4 * dw, 3.77e-4 * (-dw))
        c = np.where((np.arange(len(w)) % 48 == 0) & (dw < 0), 0.55e-4 * (-dw), c)  # 日首减仓=平昨
    rd = np.repeat(P['roll_day'], 48) & (np.arange(len(w)) % 48 == 0)
    c = c + np.where(rd, 2 * 0.55e-4 * wl, 0)
    sr = wl * np.nan_to_num(r) - c
    day_ret = sr.reshape(-1, 48).sum(1)               # 收盘-收盘日收益(log近似)
    return day_ret, w

def stats(day_ret, w, seg_days, s):
    m = seg_days == s
    x = day_ret[m]
    if len(x) < 30: return None
    mu, sd = x.mean() * BARS_Y, x.std() * np.sqrt(BARS_Y)
    eq = np.cumprod(1 + x)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    wj = w.reshape(-1, 48)[m]
    turn = np.abs(np.diff(np.r_[1.0, w])).reshape(-1, 48)[m].sum(1).mean()
    return dict(年化=mu, 波动=sd, Sharpe=mu / sd if sd > 0 else np.nan, MDD=mdd,
                均仓=wj.mean(), 峰仓=wj.max(), 日换手=turn)

def main():
    for prod in ['IC', 'IM']:
        P = build_product(prod)
        seg_days = seg_of(P['days'])
        print(f'\n{"="*100}\n### {prod} ({P["days"][0]}~{P["days"][-1]}, {P["Nd"]}天)  保证金~12%, 峰仓3=36%占用\n{"="*100}')

        # --- 主表: 基准与对称VT, 三种成本 ---
        for cm in ['A', 'B', 'C']:
            rows = []
            for kind in ['BH', 'SYM_V1', 'SYM_V2', 'LEV']:
                dr, w = run(P, weights(P, kind), cm)
                for s in ['train', 'val', 'test', 'holdout']:
                    st = stats(dr, w, seg_days, s)
                    if st: rows.append({'策略': kind, '段': s, **st})
            t = pd.DataFrame(rows)
            print(f'\n-- 成本{cm} --')
            print(t.round(4).to_string(index=False))

        # --- 任务2: θ×δ 网格, train+val 选, 冻结报 test/holdout (成本C) ---
        print(f'\n-- ASYM 网格 (成本C, 按 train+val Sharpe 选) --')
        best, brow = None, None
        rows = []
        for theta in [0.85, 0.9, 0.95, 1.0]:
            for delta in [0.0, 0.15, 0.3]:
                dr, w = run(P, weights(P, 'ASYM', theta, delta), 'C')
                mtv = np.isin(seg_days, ['train', 'val'])
                x = dr[mtv]
                sh = x.mean() / x.std() * np.sqrt(BARS_Y)
                rows.append(dict(theta=theta, delta=delta, trainval_Sharpe=sh))
                if best is None or sh > best:
                    best, brow = sh, (theta, delta)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        theta, delta = brow
        print(f'>> 选定 θ={theta} δ={delta} (train+val Sharpe={best:.3f}), 冻结报告:')
        rows = []
        for nm, kind, kw in [('BH', 'BH', {}), ('LEV_纯加仓', 'LEV', {}),
                              (f'ASYM_选定', 'ASYM', dict(theta=theta, delta=delta))]:
            dr, w = run(P, weights(P, kind, **kw), 'C')
            for s in ['test', 'holdout']:
                st = stats(dr, w, seg_days, s)
                if st: rows.append({'策略': nm, '段': s, **st})
        print(pd.DataFrame(rows).round(4).to_string(index=False))

        # V2−V1 对称VT 日差 t 值 (成本B)
        d1, _ = run(P, weights(P, 'SYM_V1'), 'B')
        d2, _ = run(P, weights(P, 'SYM_V2'), 'B')
        print('\nV2−V1 (对称VT, 成本B) 日差:')
        for s in ['val', 'test', 'holdout']:
            x = (d2 - d1)[seg_days == s]
            if len(x) < 30: continue
            print(f'  {s}: {x.mean()*BARS_Y*1e4:+.0f}bp/年 t={x.mean()/x.std()*np.sqrt(len(x)):+.2f}')

if __name__ == '__main__':
    main()
