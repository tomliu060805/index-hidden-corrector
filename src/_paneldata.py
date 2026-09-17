"""无副作用的公共小工具: 三宽基 5m 面板的对数收益 + 切分标签。

★为什么单独抽出来: 这两个函数原来住在 task3_horizon.py 里, 而那个脚本**没有
  `if __name__ == '__main__'` 保护** —— 一 import 就把整个视界实验跑一遍,
  并把 test/holdout 的 R² 打到屏幕上。2026-09-17 实测踩到过一次。
  本模块只有 import 和 def, 不执行任何东西。

src/ 下另有 20 个脚本同样没有 main 保护(backtest_vt / freeze_production /
seed_robustness / gen_charts_v1 ...)。它们都是"跑一次的实验脚本", 不被别处 import,
所以没改; 但**任何新代码都不要 import 它们**, 需要复用就往本模块里搬。
"""
import numpy as np
import pandas as pd
import os as _os

_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
CODES = ['000300.XSHG', '000905.XSHG', '000852.XSHG']
NBAR = 48
TRAIN_END, VAL_END, TEST_END = '2023-04-27', '2024-06-07', '2025-07-17'


def seg_of(dates):
    return np.where(dates <= TRAIN_END, 'train',
           np.where(dates <= VAL_END, 'val',
           np.where(dates <= TEST_END, 'test', 'holdout')))


def load_close():
    """返回 (对数收益 [NG,3], 日期数组)。跨日边界置 NaN, 不拼接隔夜收益。"""
    df = pd.read_parquet(f'{_R}/out/index5m.parquet')
    dates = np.array(sorted(df['date'].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    ci = pd.Series(np.arange(len(CODES)), index=CODES)
    NG = len(dates) * NBAR
    close = np.full((NG, len(CODES)), np.nan, np.float32)
    close[di.reindex(df['date']).values * NBAR + df['bar'].values,
          ci.reindex(df['code']).values] = df['close'].values
    lr = np.full_like(close, np.nan)
    lr[1:] = np.log(close[1:] / close[:-1])
    lr[np.arange(NG) % NBAR == 0] = np.nan
    return lr, dates


def ridge_fit(X, y, tr, lam=1.0):
    """带截距的岭回归, 只用 tr 拟合(与 eval_index_ridge.ridge_fit 同口径)。"""
    Xa = np.column_stack([X, np.ones(len(X))]).astype(np.float64)
    A = Xa[tr]
    G = A.T @ A + lam * np.eye(Xa.shape[1])
    G[-1, -1] -= lam
    return Xa @ np.linalg.solve(G, A.T @ y[tr])


def r2(y, yh, m):
    return 1 - ((y[m] - yh[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
