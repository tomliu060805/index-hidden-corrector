# 子研究：L2 盘口能不能在更高频上拿到更大增量

预注册 `docs/PREREG_L2_HIGHFREQ.md`，结论见 `RESULTS_v2.md` 的
「能不能往更高频走」一节。**结论：成立但太小，不值得立项**（不是"没有效应"）。

数据源：中金所 L2 tick（通联），0.5 秒快照 / 5 档，IC/IF/IH/IM。
路径经环境变量 `IHC_CCFX_L2` 指定（见 `probe_day.py` 顶部）。

- `probe_day.py`   单日 tick → 1 秒中价网格 → 决策点特征（成交流组 / 订单簿组）
- `build_panel.py` 多日并行汇总成面板
- `eval_probe.py`  基线阶梯与逐组拆分
- `open_test_l2.py` test 段一次性开封（判据写在脚本 docstring 里）
- `memguard.py`    内存看门狗（PSS 口径；不用 `pgrep -f`，遍历 /proc 并校验 exe）
