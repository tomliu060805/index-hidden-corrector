"""波动目标化回测: V1(M0基线预报) vs V2(M2隐层预报) —— 经济检验主门

口径对齐此前内部研究的波动目标化回测:
  仓位 w_t = clip(target/pred_vol_t, 0, 3), target=train中位数(使train均仓≈1)
  bar t 决策用 t 收盘信息, 作用于 t→t+1 收益; 日内5min不含隔夜; 成本=cost·|Δw|
  bar 36-47 无预报, 日内ffill仓位; 主指标 Sharpe(尺度无关) + 波动离散度(std日波动/均值)
"""
import os
import numpy as np, pandas as pd

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']
BARS_Y = 48 * 242.0

z = np.load(f'{B}/out/paper1_preds.npz')
pred = {'V1_M0': np.exp(z['yh0']), 'V2_M2': np.exp(z['yh0'] + z['p2'])}

df = pd.read_parquet(f'{B}/out/index5m.parquet')
dates = np.array(sorted(df['date'].unique()))
di = pd.Series(np.arange(len(dates)), index=dates)
ci = pd.Series(np.arange(len(CODES)), index=CODES)
NG = len(dates) * 48
close = np.full((NG, 3), np.nan, np.float32)
close[di.reindex(df['date']).values * 48 + df['bar'].values, ci.reindex(df['code']).values] = df['close'].values
lr = np.full_like(close, np.nan); lr[1:] = np.log(close[1:] / close[:-1])
lr[np.arange(NG) % 48 == 0] = np.nan                      # 隔夜剔除
fwd1 = np.full_like(close, np.nan); fwd1[:-1] = lr[1:]    # bar t -> t+1 收益
fwd1[np.arange(NG) % 48 == 47] = np.nan                   # 日末无下一bar

# 决策点预报 -> (NG,3) 网格
gt = (di.reindex(z['date']).values * 48 + z['bar']).astype(int)
gj = z['j'].astype(int)
seg_of_date = pd.Series(np.where(dates <= '2023-04-27', 'train',
                        np.where(dates <= '2024-06-07', 'val',
                        np.where(dates <= '2025-07-17', 'test', 'holdout'))), index=dates)

def perf(ret, w):
    ok = np.isfinite(ret)
    ret = ret[ok];
    if len(ret) < 100: return {}
    mu, sd = ret.mean() * BARS_Y, ret.std() * np.sqrt(BARS_Y)
    eq = np.cumprod(1 + ret)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(年化收益=mu, 年化波动=sd, Sharpe=mu/sd if sd > 0 else np.nan, 最大回撤=mdd,
                Calmar=mu/abs(mdd) if mdd < 0 else np.nan, 平均仓位=np.nanmean(w))

def vol_dispersion(ret, ndays):
    r = pd.Series(ret)
    dv = r.groupby(np.arange(len(r)) // 48).apply(lambda g: g.abs().mean())
    dv = dv.dropna()
    return dv.std() / dv.mean() if dv.mean() > 0 else np.nan

for jj, c in enumerate(CODES):
    print(f'\n{"="*90}\n### {c}\n{"="*90}')
    W = {}
    for nm, p in pred.items():
        pv = np.full(NG, np.nan)
        mj = gj == jj
        pv[gt[mj]] = p[mj]
        w = np.full(NG, np.nan)
        trm = np.isfinite(pv) & np.repeat(seg_of_date.values == 'train', 48)
        tgt = np.nanmedian(pv[trm])
        w[np.isfinite(pv)] = np.clip(tgt / pv[np.isfinite(pv)], 0, 3.0)
        w = pd.Series(w).groupby(np.arange(NG) // 48).ffill().values   # 日内ffill, 不跨日
        w = np.where(np.isfinite(w), w, 1.0)                            # 无预报日=满仓
        W[nm] = w
    W['V0_买入持有'] = np.ones(NG)

    for COST in [0.0, 0.5, 1.0]:
        rows = []
        for nm, w in W.items():
            w_lag = np.r_[1.0, w[:-1]]
            ret = w * fwd1 [:, jj] - COST * 1e-4 * np.abs(w - w_lag)
            for s in ['val', 'test', 'holdout']:
                m = np.repeat(seg_of_date.values == s, 48)
                d = perf(ret[m], w[m])
                if not d: continue
                d['波动离散'] = vol_dispersion(np.where(np.isfinite(ret[m]), ret[m], 0), m.sum()//48)
                turn = np.abs(w - w_lag)[m]
                d['日均换手'] = np.nansum(turn) / (m.sum() / 48)
                d.update(模型=nm, 段=s); rows.append(d)
        t = pd.DataFrame(rows)[['模型','段','年化收益','年化波动','Sharpe','最大回撤','Calmar','平均仓位','日均换手','波动离散']]
        print(f'\n-- 成本 {COST}bp --')
        print(t.sort_values(['段','模型']).round(4).to_string(index=False))
