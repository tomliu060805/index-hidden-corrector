# -*- coding: utf-8 -*-
"""图表 v1: 期货覆盖层净值/超额/逐年超额 + 4 幅逻辑图
输出 -> figures/v1-futures-overlay/
"""
import sys, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{B}/figures/v1-futures-overlay'
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f'{B}/code')
import futures_gate as FG
import backtest_futures as BF

plt.rcParams.update({
    'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False,
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 10.5,
    'axes.edgecolor': '#c9cbc4', 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': '#e3e4df', 'grid.linewidth': 0.6,
})
C_BLUE, C_ORANGE, C_AQUA, C_INK, C_MUT = '#2a78d6', '#eb6834', '#1baf7a', '#191b1f', '#8a8e99'
SEG_BOUND = {'val': '2023-04-27', 'test': '2024-06-07', 'holdout': '2025-07-17'}
CFG = {'IC': dict(src='preds_2h.npz', gate=60, delta=0.3, name='IC（中证500期货）'),
       'IM': dict(src='preds_2h.npz', gate=5, delta=0.15 if False else 0.3, name='IM（中证1000期货）')}
# 冻结配置: IC 2h+60日门+δ0.3; IM 2h+5日门+δ0.3 (train+val 选定)


def series(prod):
    cfg = CFG[prod]
    P = FG.build(prod, cfg['src'])
    w = FG.make_w(P, cfg['gate'], cfg['delta'])
    dr = FG.run(P, w)                       # 成本C 平今现实
    bh = FG.run(P, np.ones(P['Nd'] * 48))
    dts = pd.to_datetime(P['days'])
    return dts, dr, bh


def shade_segments(ax, dts):
    spans = [(dts.min(), pd.Timestamp(SEG_BOUND['val']), '训练', '#ffffff'),
             (pd.Timestamp(SEG_BOUND['val']), pd.Timestamp(SEG_BOUND['test']), '验证', '#eef3fb'),
             (pd.Timestamp(SEG_BOUND['test']), pd.Timestamp(SEG_BOUND['holdout']), '测试', '#fdf1ea'),
             (pd.Timestamp(SEG_BOUND['holdout']), dts.max(), '封存段', '#eaf6f1')]
    for a, b, lab, col in spans:
        if b <= dts.min() or a >= dts.max():
            continue
        a = max(a, dts.min()); b = min(b, dts.max())
        ax.axvspan(a, b, color=col, zorder=0)
        ax.text((a + (b - a) / 2), 0.985, lab, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=9, color=C_MUT)


# ============ 图1/2: 净值曲线 ============
for prod in ['IC', 'IM']:
    dts, dr, bh = series(prod)
    nav_s, nav_b = np.cumprod(1 + dr), np.cumprod(1 + bh)
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    shade_segments(ax, dts)
    ax.plot(dts, nav_b, color=C_MUT, lw=1.4, label='直接持有主力合约（含贴水与隔夜）')
    ax.plot(dts, nav_s, color=C_BLUE, lw=1.7, label='覆盖层：波动预报加仓 + 趋势门 + 调仓死区')
    ax.set_yscale('log')
    ax.set_title(f'净值曲线 · {CFG[prod]["name"]} · 真实口径（平今成本, 收盘-收盘含隔夜）', fontsize=12)
    ax.legend(loc='upper left', frameon=False, fontsize=9.5)
    ax.annotate(f'{nav_s[-1]:.2f} 覆盖层', (dts[-1], nav_s[-1]), xytext=(6, 9), textcoords='offset points',
                color=C_BLUE, va='center', fontsize=9.5, fontweight='bold')
    ax.annotate(f'{nav_b[-1]:.2f} 买持', (dts[-1], nav_b[-1]), xytext=(6, -9), textcoords='offset points',
                color=C_MUT, va='center', fontsize=9.5)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    fig.savefig(f'{OUT}/nav-curve-{prod}.png')
    plt.close(fig)

# ============ 图3: 超额曲线 (覆盖层-买持) ============
fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
for prod, col in [('IC', C_BLUE), ('IM', C_ORANGE)]:
    dts, dr, bh = series(prod)
    ex = np.cumsum(dr - bh) * 100
    shade_segments(ax, dts) if prod == 'IC' else None
    ax.plot(dts, ex, color=col, lw=1.7, label=f'{CFG[prod]["name"]}')
    ax.text(dts[-1], ex[-1], f'  {ex[-1]:+.1f}pp', color=col, va='center', fontsize=9.5, fontweight='bold')
ax.axhline(0, color=C_INK, lw=0.8)
ax.set_ylabel('累计超额（百分点）')
ax.set_title('超额曲线 · 覆盖层 − 直接持有 · 冻结配置（验证段之后未做任何选择）', fontsize=12)
ax.legend(loc='upper left', frameon=False, fontsize=9.5)
fig.savefig(f'{OUT}/excess-overlay-vs-buyhold.png')
plt.close(fig)

# ============ 图4: 超额曲线 (信号增量 V2-V1, 对称VT 成本B) ============
fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
first = True
for prod, col in [('IC', C_BLUE), ('IM', C_ORANGE)]:
    P = BF.build_product(prod)
    d1, _ = BF.run(P, BF.weights(P, 'SYM_V1'), 'B')
    d2, _ = BF.run(P, BF.weights(P, 'SYM_V2'), 'B')
    dts = pd.to_datetime(P['days'])
    if first:
        shade_segments(ax, dts); first = False
    ex = np.cumsum(d2 - d1) * 100
    ax.plot(dts, ex, color=col, lw=1.7, label=f'{CFG[prod]["name"]}')
    ax.text(dts[-1], ex[-1], f'  {ex[-1]:+.1f}pp', color=col, va='center', fontsize=9.5, fontweight='bold')
ax.axhline(0, color=C_INK, lw=0.8)
ax.set_ylabel('累计超额（百分点）')
ax.set_title('信号增量曲线 · 老师傅预报 − 常识预报（同构VT, 1bp成本）· 隐层的净贡献', fontsize=12)
ax.legend(loc='upper left', frameon=False, fontsize=9.5)
fig.savefig(f'{OUT}/excess-signal-increment.png')
plt.close(fig)

# ============ 图5: 逐年超额 ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for k, (mode, ttl) in enumerate([('prod', '覆盖层 − 直接持有（平今成本）'),
                                 ('sig', '信号增量 V2−V1（对称VT, 1bp）')]):
    ax = axes[k]
    allyrs = None
    for prod, col, off in [('IC', C_BLUE, -0.2), ('IM', C_ORANGE, 0.2)]:
        if mode == 'prod':
            dts, dr, bh = series(prod); diff = dr - bh
        else:
            P = BF.build_product(prod)
            d1, _ = BF.run(P, BF.weights(P, 'SYM_V1'), 'B')
            d2, _ = BF.run(P, BF.weights(P, 'SYM_V2'), 'B')
            dts = pd.to_datetime(P['days']); diff = d2 - d1
        yr = pd.Series(diff * 100, index=dts).groupby(dts.year).sum()
        yr = yr[yr.index >= 2016] if prod == 'IC' else yr
        ax.bar(yr.index + off, yr.values, width=0.38, color=col,
               label='IC' if prod == 'IC' else 'IM')
        allyrs = yr.index if allyrs is None else allyrs.union(yr.index)
    ax.axhline(0, color=C_INK, lw=0.8)
    ax.set_title(ttl, fontsize=11.5)
    ax.set_ylabel('年度超额（百分点）')
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_xticks(list(allyrs))
    ax.tick_params(axis='x', rotation=45)
fig.suptitle('逐年超额 · 左=产品口径 右=信号口径', fontsize=12.5)
fig.savefig(f'{OUT}/annual-excess.png')
plt.close(fig)


# ============ 逻辑图工具 ============
def box(ax, x, y, w, h, text, fc='#ffffff', ec='#c9cbc4', fs=10, tc=C_INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.015',
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal', linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, text='', fs=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
                                 color=C_MUT, lw=1.4))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha='center', fontsize=fs, color=C_MUT)


def canvas(figsize=(11, 5)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.grid(False)
    return fig, ax


# ============ 逻辑图1: 总体流程 ============
fig, ax = canvas((12, 5.2))
ax.text(0.01, 0.97, '逻辑图 1 · 总体流程：从行情到交易指令', fontsize=13, fontweight='bold', va='top')
box(ax, 0.01, 0.55, 0.16, 0.24, '输入\n三大指数 5 分钟K线\n(2014起, 每天48根)\n+ IC/IM 期货行情', fc='#eef3fb', fs=9.5)
box(ax, 0.21, 0.55, 0.17, 0.24, '老师傅看盘\n冻结的K线基础模型\nKronos (不再训练)\n每5分钟看过去2.5天', fc='#ffffff', fs=9.5)
box(ax, 0.42, 0.55, 0.17, 0.24, '完整心理活动\n512个数字(隐层)\n⚠ 7月只问2个数字\n所以判了死刑', fc='#fdf1ea', fs=9.5)
box(ax, 0.63, 0.55, 0.16, 0.24, '小翻译官\n门控线性修正器\n(3.5万参数)\n修正常识预报', fc='#ffffff', fs=9.5)
box(ax, 0.83, 0.55, 0.15, 0.24, '输出\n未来1~2小时\n市场颠簸程度预报\n(不预测涨跌!)', fc='#eaf6f1', fs=9.5, bold=True)
arrow(ax, 0.17, 0.67, 0.21, 0.67); arrow(ax, 0.38, 0.67, 0.42, 0.67)
arrow(ax, 0.59, 0.67, 0.63, 0.67); arrow(ax, 0.79, 0.67, 0.83, 0.67)
box(ax, 0.21, 0.12, 0.17, 0.26, '常识预报(基线)\n刚才颠→接着颠\n+ 一天内固定节奏\n(它单独已答对6成)', fc='#f6f6f3', fs=9.5)
arrow(ax, 0.38, 0.25, 0.665, 0.55)
box(ax, 0.63, 0.12, 0.35, 0.26, '交易化 (只用于期货覆盖层)\n预报平静 → 加仓至最多3倍   预报颠簸 → 回到1倍\n+ 顺风才加速(趋势门)   + 方向盘别乱动(调仓死区)', fc='#eef3fb', fs=9.5)
arrow(ax, 0.905, 0.55, 0.85, 0.38)
fig.savefig(f'{OUT}/diagram1-pipeline.png'); plt.close(fig)

# ============ 逻辑图2: 单个决策点的输入输出 ============
fig, ax = canvas((12, 5.2))
ax.text(0.01, 0.97, '逻辑图 2 · 每个决策点(每5分钟一次)的输入与输出', fontsize=13, fontweight='bold', va='top')
box(ax, 0.01, 0.52, 0.30, 0.30, '输入窗口\n过去128根5分钟K线 (≈2.5天)\n开高低收+量+额 六列\n先做"窗口内标准化":\n只用窗口自己的均值方差\n(防止未来信息泄漏)', fc='#eef3fb', fs=9.5)
box(ax, 0.37, 0.52, 0.26, 0.30, '冻结模型一次前向\n得到三样东西:\n① 512维隐层 (主角)\n② 意外度 NLL (旧标量)\n③ 犹豫度 熵 (旧标量)', fs=9.5)
box(ax, 0.69, 0.52, 0.29, 0.30, '输出\n预报目标 = log(未来H根平均|涨跌幅|)\nH=12根(1小时) 或 24根(2小时)\n\n换算成年化波动即"路况预报"', fc='#eaf6f1', fs=9.5)
arrow(ax, 0.31, 0.67, 0.37, 0.67); arrow(ax, 0.63, 0.67, 0.69, 0.67)
box(ax, 0.01, 0.08, 0.45, 0.32, '预报合成公式\n最终预报 = 常识预报 + 音量旋钮 × 翻译量\n翻译量 = W·(512维隐层等全部特征)\n音量旋钮 = sigmoid门控, 模型自己决定采纳多少\n系数全部冻结在2023-04, 之后从未再动', fc='#f6f6f3', fs=9.5)
box(ax, 0.52, 0.08, 0.46, 0.32, '质量 (2025-07后从未参与选择的封存段)\n解释力提升 +0.096 (旧标量法只有+0.005)\n日内排序相关 +0.38, 信息比 22.9\n行情越难预测的时期, 帮忙越大\n对"涨跌方向"贡献 = 0 (试过N次)', fc='#fdf1ea', fs=9.5)
fig.savefig(f'{OUT}/diagram2-input-output.png'); plt.close(fig)

# ============ 逻辑图3: 如何转化为可交易逻辑 ============
fig, ax = canvas((12, 5.6))
ax.text(0.01, 0.97, '逻辑图 3 · 预报如何变成交易指令 (IC/IM 期货覆盖层)', fontsize=13, fontweight='bold', va='top')
box(ax, 0.01, 0.60, 0.20, 0.22, '① 波动预报\n未来2小时路况\n(每5分钟更新)', fc='#eaf6f1', fs=9.5)
box(ax, 0.25, 0.60, 0.22, 0.22, '② 目标仓位\nw = clip(σ* ÷ 预报, 1, 3)\nσ* = 训练段预报中位数\n只加油不刹车(下限1)', fs=9.5)
box(ax, 0.51, 0.60, 0.22, 0.22, '③ 趋势门(暴拉保护)\n指数近期在跌?\n→ 杠杆归1不加仓\nIC看60日 IM看5日', fs=9.5)
box(ax, 0.77, 0.60, 0.21, 0.22, '④ 调仓死区 δ=0.3\n目标与当前仓位\n差距<0.3倍就不动\n(平今费太贵)', fs=9.5)
arrow(ax, 0.21, 0.71, 0.25, 0.71); arrow(ax, 0.47, 0.71, 0.51, 0.71); arrow(ax, 0.73, 0.71, 0.77, 0.71)
box(ax, 0.25, 0.24, 0.48, 0.24, '⑤ 下单执行\n主力合约按官方映射、换月提前一天已知\n隔夜持仓不平 (隔夜跳空+贴水是收益来源, 实测日末平仓更差)\n日均调仓量 ~0.4倍 · 保证金占用 峰值36%', fc='#eef3fb', fs=9.5)
arrow(ax, 0.875, 0.60, 0.60, 0.48)
box(ax, 0.01, 0.02, 0.46, 0.16, '成本真账: 开仓万0.23+冲击 ≈0.55bp\n日内平仓(平今) 万3.45+冲击 ≈3.77bp\n→ 死区把换手从1.4压到0.4才活下来', fc='#fdf1ea', fs=9)
box(ax, 0.52, 0.02, 0.46, 0.16, '成绩(封存段): 比常识预报开车 +460~970bp/年\n比躺着持有 +330~606bp/年 (方向对但证据未板上钉钉)\n适用: 本来就要持有IC/IM敞口的账户', fc='#eaf6f1', fs=9)
fig.savefig(f'{OUT}/diagram3-tradability.png'); plt.close(fig)

# ============ 逻辑图4: 三道门与数据切分 ============
fig, ax = canvas((12, 5.2))
ax.text(0.01, 0.97, '逻辑图 4 · 检验纪律：数据怎么切、结论怎么过关', fontsize=13, fontweight='bold', va='top')
segs = [(0.01, 0.44, '训练段\n2014-01 ~ 2023-04\n学习全部参数', '#f6f6f3'),
        (0.47, 0.135, '验证段\n~2024-06\n调参/选模型', '#eef3fb'),
        (0.615, 0.135, '测试段\n~2025-07\n只看分不选择', '#fdf1ea'),
        (0.76, 0.155, '封存段 ★\n~2026-08\n一次性拆封', '#eaf6f1')]
for x, w, t, c in segs:
    box(ax, x, 0.62, w, 0.2, t, fc=c, fs=9.5)
ax.annotate('', xy=(0.98, 0.575), xytext=(0.01, 0.575),
            arrowprops=dict(arrowstyle='-|>', color=C_MUT, lw=1.4))
ax.text(0.5, 0.545, '时间 →', ha='center', fontsize=9, color=C_MUT)
box(ax, 0.01, 0.10, 0.29, 0.30, '门1 统计门\n预报增量三段一致\n且脱离±0.004死区?\n✓ +0.042~0.096', fs=9.5)
box(ax, 0.35, 0.10, 0.29, 0.30, '门2 查重门\n对已有信号(500股\n集体情绪等)有新信息?\n✓ 反向包含了它们', fs=9.5)
box(ax, 0.69, 0.10, 0.29, 0.30, '门3 赚钱门\n真实成本下逐段同号?\n✓ 信号增量6格全正\n△ 独立策略证据不足', fs=9.5)
arrow(ax, 0.30, 0.25, 0.35, 0.25); arrow(ax, 0.64, 0.25, 0.69, 0.25)
fig.savefig(f'{OUT}/diagram4-validation-discipline.png'); plt.close(fig)

print('输出完成:', sorted(os.listdir(OUT)))
