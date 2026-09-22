"""单日探针: 由中金所 L2 tick 建 1s 中价网格, 算 30s/60s RV 目标 + 两组特征。

★为什么直接做线性探针而不先上模型(M4 陷阱: the machinery is not paying for itself):
  问题是"L2 盘口对超短波动有没有增量", 不是"某个模型好不好"。
  最笨的版本 = 秒级 HAR 基线 vs 秒级 HAR + 盘口派生量, 两个都是岭回归。
  这个不过线, 上任何模型都没有意义。

★信息集对齐(本项目最贵的教训 baseline-sees-fewer-inputs):
  基线 B0 只吃价格; L 组是盘口独有的量(价差/深度/队列不平衡/OFI)。
  两组的差 = 盘口的真实增量, 不混"基线没看见的价格信息"。
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG
import os, sys, zipfile, io
import numpy as np, pandas as pd

SRC = CFG.CCFX_L2_DIR
COLS = ['UpdateTime','InstruID','LastPrice','Volume','Turnover',
        'BidPrice1','BidVolume1','BidPrice2','BidVolume2','BidPrice3','BidVolume3',
        'AskPrice1','AskVolume1','AskPrice2','AskVolume2','AskPrice3','AskVolume3']
AM = (9*3600+31*60, 11*3600+30*60)      # 09:31:00 - 11:30:00
PM = (13*3600+ 1*60, 15*3600+ 0*60)
STEP = 10          # 决策点间隔(秒)
HORIZ = (30, 60, 300, 900, 1800)   # 30s 1min 5min 15min 30min


def _grid(d, lo, hi):
    """把不规则 0.5s tick 落到 1s 网格(取该秒最后一条, 前值填充)。"""
    n = hi - lo
    sec = d['sec'].values.astype(int)
    k = np.clip(sec - lo, 0, n - 1)
    out = {}
    for c in ['mid','spr','dep','qi','last','vol','turn','slope']:
        g = np.full(n, np.nan)
        g[k] = d[c].values                      # 同秒多条取最后一条(顺序即时间序)
        s = pd.Series(g).ffill()
        out[c] = s.values
    for c in ['ofi_t','upd','dvol_t','trd_t']:  # ★这几个是流量, 逐秒**求和**
        g = np.zeros(n)
        np.add.at(g, k, np.nan_to_num(d[c].values))
        out[c] = g
    return out


def one_day(day, products=("IC","IF","IM")):
    z = f'{SRC}/{day}/future_ccfxl2_{day}.zip'
    if not os.path.exists(z):
        return None
    try:
        with zipfile.ZipFile(z) as zf:
            nm = zf.namelist()[0]
            with zf.open(nm) as fh:
                df = pd.read_csv(fh, usecols=COLS)
    except Exception as e:
        return None
    out = []
    for product in products:
        sub = df[df.InstruID.str.match(rf'^{product}\d{{4}}$', na=False)]
        if not len(sub):
            continue
        # 主力 = 当日成交量最大的合约
        code = sub.groupby('InstruID')['Volume'].max().idxmax()
        r = _one(sub[sub.InstruID == code].copy(), day, code, product)
        if r is not None:
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else None


def _one(d, day, code, product):
    t = pd.to_timedelta(d.UpdateTime, errors='coerce')
    d['sec'] = t.dt.total_seconds()
    d = d.dropna(subset=['sec'])
    d = d[(d.BidPrice1 > 0) & (d.AskPrice1 > 0)]
    if len(d) < 5000:
        return None
    d['mid']  = (d.BidPrice1 + d.AskPrice1) / 2
    d['spr']  = (d.AskPrice1 - d.BidPrice1) / d['mid'] * 1e4          # bp
    bd = d.BidVolume1 + d.BidVolume2 + d.BidVolume3
    ad = d.AskVolume1 + d.AskVolume2 + d.AskVolume3
    d['dep']   = np.log(np.maximum(bd + ad, 1.0))                      # 三档总深度
    d['qi']    = (bd - ad) / np.maximum(bd + ad, 1.0)                  # 队列不平衡 [-1,1]
    d['last']  = d.LastPrice
    d['vol']   = d.Volume
    d['turn']  = d.Turnover
    # 深度斜率: 价格每走 1bp 累积多少量(盘口形状)
    dpb = (d['mid'] - d.BidPrice3) / d['mid'] * 1e4
    dpa = (d.AskPrice3 - d['mid']) / d['mid'] * 1e4
    d['slope'] = np.log(np.maximum(bd, 1)) / np.maximum(dpb, .1) - np.log(np.maximum(ad, 1)) / np.maximum(dpa, .1)
    d = d.sort_values('sec')
    # ★真 OFI (Cont-Kukanov-Stoikov 2014): 按最优档**队列量的变化**定义, 不是价格方向x成交量
    bp1 = d.BidPrice1.values; bv1 = d.BidVolume1.values
    ap1 = d.AskPrice1.values; av1 = d.AskVolume1.values
    e = np.zeros(len(d))
    pb, pv = bp1[:-1], bv1[:-1]; cb, cv_ = bp1[1:], bv1[1:]
    e[1:] += np.where(cb > pb, cv_, np.where(cb < pb, -pv, cv_ - pv))
    pa, pav = ap1[:-1], av1[:-1]; ca, cav = ap1[1:], av1[1:]
    e[1:] -= np.where(ca < pa, cav, np.where(ca > pa, -pav, cav - pav))
    d['ofi_t'] = e
    d['upd'] = 1.0                                   # 每条快照计 1, 聚合成报价更新强度
    dv = np.diff(d.Volume.values, prepend=np.nan)
    d['dvol_t'] = np.where(dv >= 0, dv, np.nan)      # 逐 tick 成交量
    d['trd_t'] = (np.nan_to_num(d['dvol_t'].values) > 0).astype(float)

    rows = []
    for lo, hi in (AM, PM):
        g = _grid(d[(d.sec >= lo) & (d.sec < hi)], lo, hi)
        mid = g['mid']
        if np.isnan(mid).all():
            continue
        lm = np.log(np.where(np.isfinite(mid) & (mid > 0), mid, np.nan))
        r1 = np.diff(lm, prepend=np.nan)                    # 1s 对数收益
        r2 = np.nan_to_num(r1 ** 2)
        fin = np.isfinite(r1).astype(float)
        cs  = np.concatenate([[0.], np.cumsum(r2)])          # RV 前缀和
        cn  = np.concatenate([[0.], np.cumsum(fin)])
        dvol = np.diff(g['vol'], prepend=np.nan); dvol = np.where(dvol >= 0, dvol, np.nan)
        cv  = np.concatenate([[0.], np.cumsum(np.nan_to_num(dvol))])
        # OFI 代理: 中价变化方向 × 成交量
        sgn = np.sign(np.nan_to_num(r1))
        cofi = np.concatenate([[0.], np.cumsum(sgn * np.nan_to_num(dvol))])
        cai  = np.concatenate([[0.], np.cumsum(np.nan_to_num(np.abs(r1)))])   # 绝对收益和(bipower 用)
        cofi2= np.concatenate([[0.], np.cumsum(g['ofi_t'])])
        caofi= np.concatenate([[0.], np.cumsum(np.abs(g['ofi_t']))])
        cupd = np.concatenate([[0.], np.cumsum(g['upd'])])
        ctrd = np.concatenate([[0.], np.cumsum(g['trd_t'])])
        n = len(mid)

        def win(c, i, w):                                    # (i-w, i] 的和
            return c[i] - c[np.maximum(i - w, 0)]

        idx = np.arange(0, n - max(HORIZ) - 1, STEP)
        idx = idx[idx >= 600]                                # 留 10min 回看
        for i in idx:
            rec = {'date': day, 'code': code, 'prod': product, 'sess': 0 if lo == AM[0] else 1,
                   'tod': (lo + i) / 3600.0}
            ok = True
            for H in HORIZ:                                  # ★目标: 未来 H 秒的 RV
                nf = win(cn, i + H, H)
                if nf < H * 0.5:
                    ok = False; break
                rec[f'y{H}'] = np.log(win(cs, i + H, H) / nf + 1e-14)
            if not ok:
                continue
            for w in (30, 120, 600):                          # 基线: 多尺度过去 RV
                nb = max(win(cn, i, w), 1)
                rec[f'rv{w}'] = np.log(win(cs, i, w) / nb + 1e-14)
                rec[f'bp{w}'] = np.log(win(cai, i, w) / nb + 1e-9)
            rec['rvday'] = np.log(cs[i] / max(cn[i], 1) + 1e-14)   # 当日累计(不跨日)
            for w in (30, 120, 600):                          # L2 独有
                rec[f'spr{w}']  = np.nanmean(g['spr'][max(i-w,0):i])
                rec[f'dep{w}']  = np.nanmean(g['dep'][max(i-w,0):i])
                rec[f'qi{w}']   = np.nanmean(g['qi'][max(i-w,0):i])
                rec[f'slp{w}']  = np.nanmean(g['slope'][max(i-w,0):i])
                rec[f'lv{w}']   = np.log(win(cv, i, w) + 1.0)
                rec[f'ofi{w}']  = win(cofi, i, w) / (win(cv, i, w) + 1.0)
            # ★补强组 L2b
            for w in (30, 120, 600):
                lo_ = max(i - w, 0)
                rec[f'tofi{w}'] = win(cofi2, i, w) / (win(cupd, i, w) + 1.0)      # 真OFI/更新数
                rec[f'atofi{w}'] = np.log(win(caofi, i, w) / (win(cupd, i, w) + 1.0) + 1e-6)
                rec[f'upd{w}']  = np.log(win(cupd, i, w) / w + 1e-6)              # 报价更新强度/秒
                rec[f'trd{w}']  = np.log(win(ctrd, i, w) / w + 1e-6)              # 成交笔数/秒
                rec[f'sz{w}']   = np.log(win(cv, i, w) / (win(ctrd, i, w) + 1.0) + 1e-6)  # 单笔规模
                rec[f'sprsd{w}'] = np.log(np.nanstd(g['spr'][lo_:i]) + 1e-6)      # ★价差的波动
                rec[f'qisd{w}']  = np.log(np.nanstd(g['qi'][lo_:i]) + 1e-6)       # ★QI 的波动
                rec[f'depsd{w}'] = np.log(np.nanstd(g['dep'][lo_:i]) + 1e-6)      # 深度的波动
            rec['qi_now']  = g['qi'][i]
            rec['spr_now'] = g['spr'][i]
            rec['dep_now'] = g['dep'][i]
            rec['mlp']     = (g['mid'][i] - g['last'][i]) / g['mid'][i] * 1e4   # 中价-成交价
            rows.append(rec)
    if not rows:
        return None
    return pd.DataFrame(rows)


if __name__ == '__main__':
    r = one_day(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'IC')
    print(r.shape if r is not None else 'None')
    if r is not None:
        print(r.head(2).T.to_string())
