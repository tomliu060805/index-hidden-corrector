"""B9: 从 **1 分钟** 数据派生的基线特征 —— 1min 输入实验的配套。

★为什么必须有: 今天(2026-09-17)新增的失败模式 `baseline-sees-fewer-inputs` ——
  Kronos 吃什么, 基线就必须吃什么。1min 实验里 Kronos 看到 1min OHLCV,
  若基线仍只有 5min 派生量, 测出来的"改善"会混进"基线没看见的东西",
  和今天上午 B6->B8 暴露的问题一模一样。

  而且这次的不对称会更严重: **1min 数据对波动预测最直接的价值就是更精确的 RV 估计量**
  (同样 1 小时窗口, 从 12 个 5min 收益变成 60 个 1min 收益, 估计方差 ∝ 1/n)。
  这块红利本该先归基线。

★反方向的已知事实(不能只加不想): 1min RV 被微观结构噪声污染 ——
  Liu-Patton-Sheppard "Does anything beat 5-minute RV?" 的结论是基本没有,
  5 分钟这个惯例正是偏差-方差权衡的最优点。所以 B9 里同时给出
  **1min RV** 与 **1min 子采样平均 RV**(subsampled, 降噪的标准做法), 让数据自己选。

特征(全部只用 t 时刻及以前, t 是 5min bar 的收盘):
  rv1m_{w}     过去 w 根 5min 对应的 1min RV (w=12/48/240 对齐 HAR 三尺度)
  rv1m_sub_{w} 同窗口的 5 组子采样 RV 平均(每组隔 5 取一, 降微观结构噪声)
  park1m_{w}   1min Parkinson 区间估计量
  bv1m_12      1min bipower(跳跃分解的连续部分)
  jump1m_12    1min 跳跃成分
  rq1m_12      1min 已实现四次方差(测量误差代理)
  amt1m_ratio  当前 5min 内 1min 成交额的离散度(量的日内分布形状)
"""
import numpy as np
import pandas as pd
from har_baselines import _safe_log

NB1, NB5 = 240, 48
EPS = 1e-12
W5 = {12: 'h1', 48: 'd1', 240: 'd5'}      # 以 5min bar 计的窗口


def _agg5(x1m, how='sum'):
    """[NG1, J] 的 1min 逐根量 -> [NG5, J] 的每 5min bar 聚合。"""
    NG1, J = x1m.shape
    a = x1m.reshape(-1, 5, J)
    return np.nansum(a, 1) if how == 'sum' else np.nanmean(a, 1)


def build_1m_features(arr1m, cols=('open', 'high', 'low', 'close', 'volume', 'money')):
    """arr1m: [NG1, J, 6] 的 1min 网格 -> 落在 **5min 网格** 上的特征 dict。"""
    ci = {c: k for k, c in enumerate(cols)}
    o = arr1m[:, :, ci['open']].astype(np.float64)
    h = arr1m[:, :, ci['high']].astype(np.float64)
    l = arr1m[:, :, ci['low']].astype(np.float64)
    c = arr1m[:, :, ci['close']].astype(np.float64)
    m = arr1m[:, :, ci['money']].astype(np.float64)
    NG1, J = c.shape

    # 1min 对数收益, 跨日边界置 NaN(每日第一根 09:31 没有前收)
    lr = np.full_like(c, np.nan)
    lr[1:] = np.log(c[1:] / c[:-1])
    lr[np.arange(NG1) % NB1 == 0] = np.nan
    r2 = lr ** 2

    f = {}
    # --- 1min RV 三尺度(对齐 HAR 的 12/48/240 根 5min) ---
    rv5 = _agg5(np.nan_to_num(r2))                         # 每个 5min bar 内的 1min RV
    cnt5 = _agg5(np.isfinite(r2).astype(float))
    for w, nm in W5.items():
        s = pd.DataFrame(rv5).rolling(w, min_periods=w // 2).sum().values
        n = pd.DataFrame(cnt5).rolling(w, min_periods=w // 2).sum().values
        f[f'rv1m_{nm}'] = _safe_log(s / np.maximum(n, 1))

    # --- 子采样 RV(降微观结构噪声的标准做法): 每隔 5 根取一条 1min 序列, 5 条平均 ---
    sub = np.zeros_like(rv5)
    for off in range(5):
        lr_s = np.full_like(c, np.nan)
        lr_s[5:] = np.log(c[5:] / c[:-5])                  # 5 分钟间隔但错开 off 的收益
        lr_s[np.arange(NG1) % NB1 < 5] = np.nan
        idx = np.arange(NG1) % 5 == off
        tmp = np.where(idx[:, None], lr_s ** 2, 0.0)
        sub += _agg5(np.nan_to_num(tmp))
    sub /= 5.0
    for w, nm in W5.items():
        s = pd.DataFrame(sub).rolling(w, min_periods=w // 2).sum().values
        f[f'rv1m_sub_{nm}'] = _safe_log(np.maximum(s, 0) / w)

    # --- 1min Parkinson ---
    hl = np.log(np.maximum(h, 1e-12) / np.maximum(l, 1e-12))
    park5 = _agg5(np.nan_to_num(hl ** 2 / (4 * np.log(2))))
    for w, nm in W5.items():
        s = pd.DataFrame(park5).rolling(w, min_periods=w // 2).sum().values
        f[f'park1m_{nm}'] = _safe_log(np.maximum(s, 0) / w)

    # --- 1min 跳跃分解(bipower) + 已实现四次方差 ---
    a = np.abs(lr)
    bp = np.full_like(a, np.nan)
    bp[1:] = a[1:] * a[:-1]
    bp[np.arange(NG1) % NB1 <= 1] = np.nan
    bv5 = _agg5(np.nan_to_num(bp)) * (np.pi / 2)
    BV = pd.DataFrame(bv5).rolling(12, min_periods=6).sum().values
    RV = pd.DataFrame(rv5).rolling(12, min_periods=6).sum().values
    f['bv1m_h1'] = _safe_log(np.maximum(BV, 0) / 12)
    f['jump1m_h1'] = _safe_log(np.maximum(RV - BV, 0.0) / 12)
    rq5 = _agg5(np.nan_to_num(lr ** 4))
    RQ = pd.DataFrame(rq5).rolling(12, min_periods=6).sum().values
    f['rq1m_h1'] = _safe_log(np.maximum(RQ, 0) * 1e8)

    # --- 5min bar 内成交额的分布形状(1min 才看得见) ---
    am = arr1m[:, :, ci['money']].astype(np.float64).reshape(-1, 5, J)
    tot = np.nansum(am, 1)
    f['amt1m_disp'] = _safe_log(np.nanstd(am, 1) / np.maximum(tot / 5, 1.0))
    f['amt1m_max'] = _safe_log(np.nanmax(am, 1) / np.maximum(tot, 1.0))
    return f


B9_EXTRA = ([f'rv1m_{n}' for n in W5.values()] + [f'rv1m_sub_{n}' for n in W5.values()]
            + [f'park1m_{n}' for n in W5.values()]
            + ['bv1m_h1', 'jump1m_h1', 'rq1m_h1', 'amt1m_disp', 'amt1m_max'])
