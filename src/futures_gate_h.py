"""按指定视界重选 IC/IM 覆盖层的趋势门与 δ 无交易带 —— **只用 train+val**。

与 src/futures_gate.py 同协议同网格({无门,5,20,60日门} × {δ0.15,0.3}), 两处差别:
  ★1 视界参数化(原脚本写死 1h/2h 两个源)。
  ★2 **不计算、不打印 test/holdout**。原脚本会把两段的 Sharpe 打出来 —— 那条线在 v1
     已经开封过, 本次换 30min 属于新的调参, 按项目规矩「读过之后不能再调」,
     唯一诚实的续法是向前累计。故本脚本连算都不算。

用法: python futures_gate_h.py --src preds_H6.npz --tag H6
输出: out/gate_<tag>.csv + 每个品种的选定参数(打印 + json)
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import argparse, json
import numpy as np
import pandas as pd

B = f'{_R}'
PROD = {'IC': 1, 'IM': 2}
CAP, BARS_Y = 3.0, 242.0
CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']
GATES = ['none', 5, 20, 60]
DELTAS = [0.15, 0.3]

seg_of = lambda dt: np.where(dt <= '2023-04-27', 'train',
                    np.where(dt <= '2024-06-07', 'val', 'SEALED'))


def build(prod, src, fut, eodc, use_hidden=True):
    """use_hidden=True -> V2(岭基线+隐层修正); False -> V1(只用岭基线预报)。
    ★真正可部署的量是 V2−V1(同规则、只换 σ 来源), 不是独立策略 Sharpe ——
      v1 已判定'独立 VT/加仓策略不部署', IM 的独立 Sharpe 在 train+val 全格为负。"""
    z = np.load(f'{B}/out/{src}')
    j = PROD[prod]
    f = fut[fut['product'] == prod].sort_values(['date', 'bar']).reset_index(drop=True)
    days = np.array(sorted(f.date.unique())); Nd = len(days)
    di = pd.Series(np.arange(Nd), index=days)
    g = di.reindex(f.date).values * 48 + f.bar.values
    r = np.full(Nd * 48, np.nan)
    lc = np.log(f.close.values); ret = f.ret.values.copy()
    b0 = f.bar.values == 0
    ret[b0] = f.ov_ret.values[b0] + (lc[b0] - np.log(f.open.values[b0]))
    r[g] = ret
    roll_day = f.groupby('date')['roll'].first().reindex(days).values.astype(bool)
    mj = z['j'] == j
    pv2_all = np.exp(z['yh0'] + z['p2']) if use_hidden else np.exp(z['yh0'])
    fmap = pd.Series(np.arange(mj.sum()),
                     index=pd.MultiIndex.from_arrays([z['date'][mj], z['bar'][mj]]))
    pv2 = np.full(Nd * 48, np.nan)
    dd = np.repeat(days, 48); bb = np.tile(np.arange(48), Nd)
    idx = fmap.reindex(pd.MultiIndex.from_arrays([dd, bb]))
    ok = idx.notna().values
    pv2[ok] = pv2_all[mj][idx.values[ok].astype(int)]
    tgt = float(np.median(pv2_all[(z['seg'] == 'train') & mj]))
    c = eodc[CODES[j]].reindex(days)
    mom = {k: np.log(c / c.shift(k)).shift(1).values for k in [5, 20, 60]}
    return dict(r=r, pv2=pv2, tgt=tgt, days=days, Nd=Nd, roll_day=roll_day,
                mom=mom, seg_days=seg_of(days))


def make_w(P, gate_k, delta):
    w = np.clip(P['tgt'] / P['pv2'], 1.0, CAP)
    w = pd.Series(w).ffill().fillna(1.0).values
    if gate_k != 'none':
        gd = np.nan_to_num(P['mom'][gate_k]) > 0
        w = 1.0 + (w - 1.0) * np.repeat(gd, 48)
    if delta > 0:
        out = np.empty_like(w); cur = w[0]
        for i in range(len(w)):
            if abs(w[i] - cur) >= delta:
                cur = w[i]
            out[i] = cur
        w = out
    return w


def run(P, w):
    """平今现实成本 C: 开仓 0.55bp, 平今 3.77bp; 隔夜减仓按开仓价; 换月双边。"""
    wl = np.r_[1.0, w[:-1]]
    dw = w - wl
    c = np.where(dw > 0, 0.55e-4 * dw, 3.77e-4 * (-dw))
    c = np.where((np.arange(len(w)) % 48 == 0) & (dw < 0), 0.55e-4 * (-dw), c)
    rd = np.repeat(P['roll_day'], 48) & (np.arange(len(w)) % 48 == 0)
    c += np.where(rd, 2 * 0.55e-4 * wl, 0)
    return (wl * np.nan_to_num(P['r']) - c).reshape(-1, 48).sum(1)


def sh(x):
    return x.mean() / x.std() * np.sqrt(BARS_Y) if len(x) > 30 and x.std() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    fut = pd.read_parquet(f'{B}/out/futures_dom5m.parquet')
    idx5 = pd.read_parquet(f'{B}/out/index5m.parquet')
    eodc = idx5[idx5.bar == 47].pivot(index='date', columns='code', values='close')

    rows, chosen = [], {}
    for prod in ['IC', 'IM']:
        P2 = build(prod, a.src, fut, eodc, use_hidden=True)    # V2: 岭 + 隐层修正
        P1 = build(prod, a.src, fut, eodc, use_hidden=False)   # V1: 只用岭基线预报
        sd = P2['seg_days']
        mtv = np.isin(sd, ['train', 'val'])
        print(f'\n### {prod}  (n_days train+val = {mtv.sum()}, 封存段 {int((sd=="SEALED").sum())} 天不参与)')
        print('  判据 = V2−V1 的年化收益差(同规则, 只换 σ 来源); 独立 Sharpe 仅供参考')
        best = (None, None, -9e9)
        for gk in GATES:
            for delta in DELTAS:
                d2 = run(P2, make_w(P2, gk, delta))
                d1 = run(P1, make_w(P1, gk, delta))
                inc_bp = float((d2[mtv] - d1[mtv]).mean() * BARS_Y * 1e4)   # bp/年
                s2, s1 = sh(d2[mtv]), sh(d1[mtv])
                w = make_w(P2, gk, delta)
                turn = float(np.abs(np.diff(np.r_[1.0, w])).sum() / (mtv.sum() or 1))
                rows.append(dict(prod=prod, gate=gk, delta=delta, inc_bp_yr=inc_bp,
                                 v2_sharpe=s2, v1_sharpe=s1, d_sharpe=s2 - s1,
                                 mean_w=float(np.nanmean(w[np.repeat(mtv, 48)])),
                                 turn_per_day=turn))
                print(f'  门={str(gk):>4s} δ={delta:<5.2f}  V2−V1={inc_bp:+8.1f}bp/年  '
                      f'ΔSharpe={s2-s1:+6.3f}  (V2 {s2:+.3f} / V1 {s1:+.3f})  '
                      f'日换手={turn:.3f}')
                if inc_bp > best[2]:
                    best = (gk, delta, inc_bp)
        chosen[prod] = dict(gate_days=best[0], delta=best[1], inc_bp_yr=best[2],
                            sigma_star=P2['tgt'])
        print(f'  ★选定: 门={best[0]}  δ={best[1]}  (V2−V1 = {best[2]:+.1f} bp/年)')

    pd.DataFrame(rows).to_csv(f'{B}/out/gate_{a.tag}.csv', index=False)
    json.dump(chosen, open(f'{B}/out/gate_{a.tag}_chosen.json', 'w'), indent=1, default=str)
    print(f'\n写出 out/gate_{a.tag}.csv 与 gate_{a.tag}_chosen.json')
    print('★ test/holdout 未计算、未打印 —— 本配置的验证只能靠前向累计。')


if __name__ == '__main__':
    main()
