"""高频 HAR 家族基线 + 目标构造。

为什么要换掉原来的岭回归: 判负库里 LM-GARCH 那条的教训是
「波动率模型的复杂度几乎从不赢过 HAR-RV, 一个新模型要先过这个基线再谈别的」。
项目原基线是 log(mean|r|) 的三尺度 + 日内节律独热, 结构上接近 HAR 但
(a) 统计量用的是平均绝对收益 MAD 而不是文献标准的已实现波动 RV;
(b) 缺 HARQ 的测量误差修正与 HAR-RS 的半方差分解。

本模块给出一整排前人基线, Kronos 隐层要打的是其中 val 最强的那个。

口径
----
r_t      = log(close_t / close_{t-1}), 跨日边界置 NaN(不跨日拼收益)
RV_W(t)  = Σ_{i=t-W+1}^{t} r_i²      —— 含当前 bar t, 即 t 时刻的完整信息集
目标      y = log RV_{t+1 : t+H}     —— 未来 H 根, 强制留在日内
副口径    y = log MAD_{t+1 : t+H} = log mean|r| —— 仅用于接回项目历史数字

窗口跨日: 过去 1 日(48根)/5 日(240根)窗必然跨日, 按 nan-skip 处理(与项目原实现一致)。
"""
import numpy as np
import pandas as pd

NBAR = 48                       # 每日 5min bar 数
EPS = 1e-12
SCALES = {'rv12': 12, 'rv48': 48, 'rv240': 240}     # 过去1h / 1日 / 5日
NLAG = 12                       # B6 用的带符号滞后收益根数(过去 1 小时)


def log_returns(close):
    """close: [NG, J] -> 对数收益, 每日第一根(bar 0)置 NaN(不跨日)"""
    lr = np.full_like(close, np.nan, dtype=np.float64)
    lr[1:] = np.log(close[1:] / close[:-1])
    lr[np.arange(close.shape[0]) % NBAR == 0] = np.nan
    return lr


def _roll_sum(a, w, minp):
    return pd.DataFrame(a).rolling(w, min_periods=minp).sum().values


# ★log 特征的地板: 原来用 log(x + 1e-12), 当滚动窗内波动恰为 0 时给出 -27.63,
#   而特征的正常范围是 -18~-10。线性模型遇到这种 10 个标准差之外的输入会**灾难性外推**
#   (实测预测 RV 小了一百万倍, u=RV/预测 到 6e7, QLIKE 爆到 6e7)。
#   rs_up/rs_dn 尤其容易踩 —— 单向行情里上行或下行半方差天然可以是 0(实测 0.13%/0.14%)。
#   改为在 log 之后 clip 到一个合理下限, 保留"极低波动"的信息但不让它变成数值炸弹。
LOG_FLOOR = -20.0


def _safe_log(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.log(np.maximum(x, 0.0) + EPS)
    return np.maximum(v, LOG_FLOOR)


AMT_SCALES = [3, 6, 12, 24, 48, 96, 240, 480]     # 成交额多尺度(用户要求多加几个)
RNG_SCALES = [3, 6, 12, 48]                       # 高低价区间估计量的尺度


def build_ohlcv_features(o, h, l, c, v, m):
    """B8 用的全套扩展特征。只用 t 时刻及以前, 逐项说明见下。

    ★动机: Kronos 的输入是 [open,high,low,close,volume,amount] 六列, 而原基线**只用了 close**。
      "隐层有增量"里可能有一部分只是"隐层把成交量/高低价编码进去了"。加上这些之后
      增量若不缩水, 才能说增量不是 OHLCV 的简单聚合。
    """
    f = {}
    NG = c.shape[0]
    bar = np.arange(NG) % NBAR
    lr = log_returns(c)
    r2 = lr ** 2

    # ---- 1) 成交额多尺度(MDH: 成交量与波动同源) ----
    # ★成交额也要地板: log(max(money,1)) 在 money=0 时给 0, 而正常范围 16-25
    #   —— 与 LOG_FLOOR 同一类数值炸弹(实测 amt_ratio 最低 -21.6 vs 正常 ±1.5,
    #   标准化时会抬高 std 把该特征压没)。按非零成交额的低分位设地板。
    lm = np.maximum(np.log(np.maximum(m, 1.0)), 12.0)
    for w in AMT_SCALES:
        f[f'amt{w}'] = pd.DataFrame(lm).rolling(w, min_periods=max(2, w // 2)).mean().values
    # 量的**相对**水平: 当前 bar 成交额 / 过去一天同尺度均值 (量比)
    f['amt_ratio'] = lm - f['amt48']

    # ---- 2) 日内累计 RV(★不跨日 —— 跨日滚动窗会稀释"今天到现在为止有多颠") ----
    cum = np.zeros_like(r2); cnt = np.zeros(NG)
    run = np.zeros(c.shape[1]); k = 0
    for i in range(NG):
        if bar[i] == 0:
            run = np.zeros(c.shape[1]); k = 0
        run = run + np.nan_to_num(r2[i]); k += 1
        cum[i] = run; cnt[i] = k
    f['irv_cum'] = _safe_log(cum / cnt[:, None])        # 当日均 r²
    f['irv_frac'] = _safe_log(cum) - _safe_log(_roll_sum(r2, 48, 24))  # 当日累计 / 过去一天
    f['bar_frac'] = np.repeat((bar / NBAR)[:, None], c.shape[1], 1)

    # ---- 3) 隔夜跳空(原实现把每日第一根置 NaN, 隔夜信息完全丢掉) ----
    ov = np.full_like(c, np.nan, dtype=np.float64)
    d0 = np.arange(NG) % NBAR == 0
    idx0 = np.where(d0)[0]
    gap = np.full((len(idx0), c.shape[1]), np.nan)
    gap[1:] = np.log(o[idx0[1:]] / c[idx0[1:] - 1])        # 今日首根 open / 昨日末根 close
    for k2, i in enumerate(idx0):                          # 当日全 bar 广播(开盘即可知)
        ov[i:i + NBAR] = gap[k2]
    f['ov_gap'] = np.nan_to_num(ov) * 1e2
    f['ov_abs'] = np.abs(np.nan_to_num(ov)) * 1e2

    # ---- 4) 跳跃分解(Barndorff-Nielsen bipower; HAR-CJ) ----
    a = np.abs(lr)
    bp = np.full_like(a, np.nan)
    bp[1:] = a[1:] * a[:-1]
    bp[bar == 0] = np.nan; bp[np.r_[False, bar[:-1] == 0]] = np.nan
    BV = _roll_sum(bp, 12, 6) * (np.pi / 2)
    RV12 = _roll_sum(r2, 12, 6)
    f['bv12'] = _safe_log(BV / 12)
    f['jump12'] = _safe_log(np.maximum(RV12 - BV, 0.0) / 12)

    # ---- 5) 高低价区间估计量(Parkinson / Garman-Klass; 短窗比 Σr² 有效率) ----
    hl = np.log(np.maximum(h, 1e-12) / np.maximum(l, 1e-12))
    park = hl ** 2 / (4 * np.log(2))
    co = np.log(np.maximum(c, 1e-12) / np.maximum(o, 1e-12))
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    for w in RNG_SCALES:
        f[f'park{w}'] = _safe_log(_roll_sum(park, w, max(2, w // 2)) / w)
        f[f'gk{w}'] = _safe_log(np.maximum(_roll_sum(gk, w, max(2, w // 2)), 0.0) / w)
    f['range_now'] = hl * 1e2

    # ---- 6) 日历(现在只有 bar 独热=日内周期, 没有日间周期) ----
    f['_dow_raw'] = None       # 由调用方按日期填(见 build_calendar)
    return f


def build_calendar(dates, NG, J):
    """星期几独热(5) + 月内位置(月初/月末各3日) —— 日间周期, 与 bar 独热互补。"""
    dt = pd.to_datetime(pd.Series(dates))
    dow = dt.dt.dayofweek.values
    dom = dt.dt.day.values
    mth = dt.dt.month.values
    # 月末: 该日期是否是当月在样本里的最后3个交易日
    key = mth + dt.dt.year.values * 100
    lastk = np.zeros(len(dates), bool); firstk = np.zeros(len(dates), bool)
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        lastk[idx[-3:]] = True; firstk[idx[:3]] = True
    out = {}
    for w in range(5):
        out[f'dow{w}'] = np.repeat(np.repeat((dow == w).astype(float), NBAR)[:, None], J, 1)[:NG]
    out['mon_first3'] = np.repeat(np.repeat(firstk.astype(float), NBAR)[:, None], J, 1)[:NG]
    out['mon_last3'] = np.repeat(np.repeat(lastk.astype(float), NBAR)[:, None], J, 1)[:NG]
    return out


B8_EXTRA = ([f'amt{w}' for w in AMT_SCALES] + ['amt_ratio']
            + ['irv_cum', 'irv_frac', 'bar_frac']
            + ['ov_gap', 'ov_abs']
            + ['bv12', 'jump12']
            + [f'park{w}' for w in RNG_SCALES] + [f'gk{w}' for w in RNG_SCALES]
            + ['range_now']
            + [f'dow{w}' for w in range(5)] + ['mon_first3', 'mon_last3'])


def build_features(close):
    """返回特征 dict, 每项形状 [NG, J]。全部只用到 t 时刻及以前。"""
    lr = log_returns(close)
    r2 = lr ** 2
    r4 = lr ** 4
    f = {}
    # --- HAR 三尺度: 用每根 bar 的平均 r² (= RV/W), 取 log ---
    for name, w in SCALES.items():
        f[name] = _safe_log(_roll_sum(r2, w, w // 2) / w)
    # --- HARQ: 已实现四次方差 RQ = (W/3)Σr⁴, 取 √RQ 与短期 RV 交互 ---
    rq = _roll_sum(r4, SCALES['rv12'], SCALES['rv12'] // 2) * (SCALES['rv12'] / 3.0)
    f['sqrt_rq'] = np.sqrt(np.maximum(rq, 0.0)) * 1e4          # 缩放到 O(1)
    # --- HAR-RS: 上下行半方差(过去 1h) ---
    up = np.where(lr > 0, r2, 0.0); up[np.isnan(lr)] = np.nan
    dn = np.where(lr < 0, r2, 0.0); dn[np.isnan(lr)] = np.nan
    f['rs_up'] = _safe_log(_roll_sum(up, 12, 6) / 12)
    f['rs_dn'] = _safe_log(_roll_sum(dn, 12, 6) / 12)
    # --- 当前 bar 自身 ---
    f['aret'] = np.abs(lr)
    f['_lr'] = lr
    # --- B6: 过去 K 根的**带符号**收益(原始路径, 不是平方后聚合) ---
    # HAR 三尺度只保留了 Σr² —— 丢掉了符号与逐根的形状。
    # 把原始滞后收益放进来, 既是更强的基线, 也是更强的 M3 检验:
    # 若隐层增量有一部分是"最近几根的路径形状", 这组特征应当吃掉它。
    # ★跨日的滞后填 0 而不是 NaN, 否则每天前 K 根决策点会被整体丢掉, 样本与 B4 不可比。
    NGl = lr.shape[0]
    bar = np.arange(NGl) % NBAR
    for k in range(1, NLAG + 1):
        v = np.full_like(lr, np.nan)
        v[k:] = lr[:-k]
        v[bar < k] = 0.0                 # 当日不足 k 根 -> 0(无信息), 不跨日借
        f[f'lag{k}'] = np.nan_to_num(v, nan=0.0) * 1e4      # 缩放到 O(1)
    return f


def ewma_rv(close, lam):
    """RiskMetrics: σ²_t = λσ²_{t-1} + (1-λ)r²_t, 跨日不重置(与 HAR 窗口跨日一致)"""
    lr = log_returns(close)
    r2 = np.nan_to_num(lr ** 2, nan=0.0)
    out = np.empty_like(r2)
    s = np.nanmean(r2[:NBAR * 5], 0)
    for i in range(r2.shape[0]):
        s = lam * s + (1 - lam) * r2[i]
        out[i] = s
    return np.log(out + EPS)


def past_target(close, H, kind='rv'):
    """B0 零拟合基线用: 与目标**同一个统计量**在过去 H 根上的取值。
    ★必须随 kind 变 —— 拿 RV 尺度的量去外推 MAD 目标会得到 R²=−80。"""
    lr = log_returns(close)
    if kind == 'rv':
        v = _roll_sum(lr ** 2, H, min(H, max(2, H // 2)))
    else:
        v = pd.DataFrame(np.abs(lr)).rolling(H, min_periods=min(H, max(2, H // 2))).mean().values
    return _safe_log(v)


def future_target(close, H, kind='rv'):
    """y[t] = log 未来 H 根的 RV(或 MAD)。bar > 47-H 的置 NaN(不跨日)。"""
    lr = log_returns(close)
    NG = close.shape[0]
    bar = np.arange(NG) % NBAR
    out = np.full_like(lr, np.nan)
    for i in range(NG - H):
        if bar[i] > NBAR - 1 - H:
            continue
        seg = lr[i + 1:i + 1 + H]
        if kind == 'rv':
            out[i] = np.nansum(seg ** 2, 0)
        else:                                   # mad —— 项目历史口径
            out[i] = np.nanmean(np.abs(seg), 0)
    with np.errstate(divide='ignore'):
        return np.log(out + EPS)


# ---------------------------------------------------------------- 基线规格
def baseline_design(d, feats_at, J, which):
    """按基线 id 组装设计矩阵。d 是逐点 DataFrame(含 bar/j), feats_at 是取好点的特征 dict。"""
    cols = []
    if which == 'B0':                                   # 零拟合 RW: 无参数, 直接外推
        return None
    cols += [feats_at['rv12'], feats_at['rv48'], feats_at['rv240']]      # B1 三尺度
    if which == 'B1':
        return np.column_stack(cols)
    onehot_bar = pd.get_dummies(d['bar']).values.astype(np.float64)
    onehot_idx = pd.get_dummies(d['j']).values.astype(np.float64)
    cols += [onehot_bar, onehot_idx, feats_at['aret']]
    if which == 'B2':
        return np.column_stack(cols)
    if which == 'B3':                                   # HARQ: √RQ × 短期 RV 交互
        cols += [feats_at['sqrt_rq'], feats_at['sqrt_rq'] * feats_at['rv12']]
        return np.column_stack(cols)
    if which == 'B4':                                   # HAR-RS: 短期拆上下行
        cols += [feats_at['rs_up'], feats_at['rs_dn']]
        return np.column_stack(cols)
    if which == 'B5':                                   # + EWMA
        cols += [feats_at['ewma']]
        return np.column_stack(cols)
    if which == 'B6':                                   # B4 + 过去12根带符号收益
        cols += [feats_at['rs_up'], feats_at['rs_dn']]
        cols += [feats_at[f'lag{k}'] for k in range(1, NLAG + 1)]
        return np.column_stack(cols)
    if which == 'B7':                                   # B6 + 滞后收益的平方(非线性)
        cols += [feats_at['rs_up'], feats_at['rs_dn']]
        cols += [feats_at[f'lag{k}'] for k in range(1, NLAG + 1)]
        cols += [feats_at[f'lag{k}'] ** 2 for k in range(1, NLAG + 1)]
        return np.column_stack(cols)
    if which == 'B8':                                   # ★全家桶: B6 + OHLCV 全部信息 + 日历
        cols += [feats_at['rs_up'], feats_at['rs_dn']]
        cols += [feats_at[f'lag{k}'] for k in range(1, NLAG + 1)]
        cols += [feats_at[k] for k in B8_EXTRA if k in feats_at]
        return np.column_stack(cols)
    raise ValueError(which)


BASELINES = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
BASELINE_DESC = {
    'B0': '零拟合 RW (过去H根RV外推, 无参数)',
    'B1': 'HAR-RV 高频版 (Corsi 2009: 1h/1日/5日三尺度)',
    'B2': 'B1 + 日内节律 + 指数独热 + 当前bar (Andersen-Bollerslev)',
    'B3': 'HARQ (Bollerslev-Patton-Quaedvlieg 2016: √RQ 交互)',
    'B4': 'HAR-RS (Barndorff-Nielsen: 上下行半方差)',
    'B5': 'B2 + EWMA/RiskMetrics',
    'B6': 'B4 + 过去12根带符号收益(原始路径)',
    'B7': 'B6 + 滞后收益的平方(逐根非线性)',
    'B8': '★B6 + 成交额8尺度 + 日内累计RV + 隔夜跳空 + 跳跃分解 + Parkinson/GK + 日历',
}


# ---------------------------------------------------------------- 拟合与评估
def ridge_fit(X, y, tr, lam=1.0):
    """带截距的岭回归; 只用 tr 拟合。★逐列标准化 —— 各层隐状态尺度差 400 倍,
    不标准化的话比的是尺度不是信息。"""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xn = (X - mu) / sd
    Xa = np.column_stack([Xn, np.ones(len(Xn))])
    A = Xa[tr]
    G = A.T @ A + lam * np.eye(Xa.shape[1])
    G[-1, -1] -= lam                                   # 截距不罚
    b = np.linalg.solve(G, A.T @ y[tr])
    return Xa @ b


def r2(y, yh, m):
    """样本外 R²: 基准取该段自身均值(不是训练期均值 —— FA-GSTN 项目的教训:
    用训练期均值会让 R² 在测'知不知道当前波动水平'而不是'预测得准不准')。"""
    yy, hh = y[m], yh[m]
    return 1 - ((yy - hh) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def qlike(y, yh, m):
    """QLIKE 损失(对数方差空间): 对波动预测比 MSE 更常用, Patton 2011 证明其稳健。"""
    a = np.exp(y[m] - yh[m])
    return float(np.mean(a - (y[m] - yh[m]) - 1))
