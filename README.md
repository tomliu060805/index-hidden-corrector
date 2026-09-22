# 指数隐层修正

冻结时序基础模型（Kronos）的 512 维隐状态 + 门控线性修正器，预测宽基指数日内波动，
并落地为 IC/IM 股指期货的波动目标化覆盖层。方法框架源自 arXiv 2608.08825
（Hybrid Neural-Classical Correction for Frozen Time Series Foundation Models）。

**核心数字**（holdout = 2025-07-18~2026-08-14，预注册一次性开封，262 天）：

- 未来 1h 波动 ΔR² **+0.095±0.004**（100 种子全正；旧标量提取 +0.005）
- 残差日 IC +0.377 / 日度 IR 1.41（t=22.9）；QLIKE 改善 20.5%（DM t=7.6）
- IC/IM 期货覆盖层（平今真实成本）：相对朴素波动预报 +460~970bp/年，六段全正
- 方向预测 = 0；日频（次日 RV）= 判负；信息期限 = 日内 ≤2h

## 目录

```
src/            全部脚本（数据→提取→建模→回测→稳健性→图表）
docs/            无前视审计 · 稳健性检验
figures/            v1 净值/超额/逐年超额 + 4 幅逻辑图（README 索引）
out/             中间产物（隐层库/预测/面板，不入 git）
RESULTS.md   全量结论正文（按阶段一~五组织）
```

## 复现（conda env: alphagen，torch CPU + lightgbm）

```bash
PY=python   # conda env: alphagen (torch CPU + lightgbm)
$PY src/build_index5m.py                 # ① 三宽基 5m 面板 (2014-2026)
HF_HUB_OFFLINE=1 $PY src/extract_index_hidden.py --workers 6 --threads 5   # ② 32.4万点隐层, 30核约40min
HF_HUB_OFFLINE=1 $PY src/extract_eod_hidden.py   # ③ 收盘点隐层 (次日RV用)
$PY src/eval_index_ridge.py              # ④ 岭基线 B1-B4
$PY src/run_paper1_stack.py              # ⑤ 论文一栈 M0-M4 → out/paper1_preds.npz
$PY src/eval_holdout.py                  # ⑥ holdout 一次性开封
$PY src/orth_check.py                    # ⑦ 广度正交对拍
$PY src/build_futures5m.py               # ⑧ IC/IM 主力连续面板
$PY src/backtest_futures.py              # ⑨ 期货真实口径回测
$PY src/futures_gate.py                  # ⑩ 趋势门×期限×δ带网格
$PY src/task3_horizon.py                 # ⑪ 期限结构 (2h/次日RV)
$PY src/seed_robustness.py               # ⑫ 10种子
$PY src/qlike_dm.py                      # ⑬ QLIKE/DM/安慰剂
$PY src/rolling_refit.py                 # ⑭ 滚动重训对照
$PY src/gen_charts_v1.py                 # ⑮ 图表
```

数据依赖（只读）：权威指数 1min 库（`IHC_INDEX_1M`）、期货 1min 与主力映射
（`IHC_FUTURES_1M` / `IHC_DOMINANT_MAP`）、
本地 HF 缓存的 NeoQuasar/Kronos-small 与 Tokenizer。

## 三个防坑要点（改代码前必读）

1. Kronos 窗口归一化用窗口自身 μ/σ → 每个决策点必须独立前向，不能滑窗复用。
2. `decode_s2` 在 eval 下非因果，必须单位置 sibling 调用。
3. 期货收益必须合约内计算（主力映射 shift1，隔夜用同合约前收盘），否则换月跳价污染。

完整审计见 `docs/lookahead-audit.md`。
