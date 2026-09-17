# -*- coding: utf-8 -*-
"""从 MO(中证1000)期权 tick 构建 5min ATM 跨式隐含波动 —— 期权载体实验用
口径: 每5min取近月ATM认购+认沽中价之和 / 标的 ≈ 0.7979*IV*sqrt(T)  =>  IV
输出 out/mo_iv_5m.parquet: date,bar,iv,straddle_mid,spread_bp,K,T
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import glob, os, re, numpy as np, pandas as pd
from multiprocessing import Pool
OUT=f'{_R}/out'
IDX=CFG.INDEX_1M_DIR
OPT=CFG.OPTIONS_DIR

def one_day(day):
    f=f'{OPT}/{day}/MO.parquet'
    if not os.path.exists(f): return None
    df=pd.read_parquet(f, columns=['order_book_id','datetime','a1','b1','volume'])
    ex=df.order_book_id.str.extract(r'^MO(\d{4})([CP])(\d+)$')
    df['ym'],df['cp'],df['K']=ex[0],ex[1],pd.to_numeric(ex[2],errors='coerce')
    df=df.dropna(subset=['K'])
    if not len(df): return None
    df['ym']=df['ym'].astype(int)
    front=df.ym.min()                                  # 近月
    df=df[df.ym==front]
    # 标的指数 1min
    d=f'{day[:4]}-{day[4:6]}-{day[6:]}'
    ip=f'{IDX}/{d}.parquet'
    if not os.path.exists(ip): return None
    idx=pd.read_parquet(ip, columns=['datetime','code','close'])
    idx=idx[idx.code=='000852.XSHG'][['datetime','close']]
    mins=df.datetime.dt.hour*60+df.datetime.dt.minute
    df=df[(mins.between(571,690))|(mins.between(781,900))].copy()
    mm=df.datetime.dt.hour*60+df.datetime.dt.minute
    df['bar']=np.where(mm<=690,(mm-571)//5,24+(mm-781)//5)
    df['mid']=np.where((df.a1>0)&(df.b1>0),(df.a1+df.b1)/2,np.nan)
    df['spr']=np.where((df.a1>0)&(df.b1>0),df.a1-df.b1,np.nan)
    # 每 bar 每合约取最后一笔
    last=df.sort_values('datetime').groupby(['bar','cp','K']).agg(mid=('mid','last'),spr=('spr','last')).reset_index()
    im=idx.copy(); imm=im.datetime.dt.hour*60+im.datetime.dt.minute
    im=im[(imm.between(571,690))|(imm.between(781,900))]
    m2=im.datetime.dt.hour*60+im.datetime.dt.minute
    im['bar']=np.where(m2<=690,(m2-571)//5,24+(m2-781)//5)
    spot=im.groupby('bar')['close'].last()
    rows=[]
    for bar,g in last.groupby('bar'):
        S=spot.get(bar,np.nan)
        if not np.isfinite(S): continue
        piv=g.pivot_table(index='K',columns='cp',values='mid')
        sp=g.pivot_table(index='K',columns='cp',values='spr')
        if 'C' not in piv or 'P' not in piv: continue
        piv=piv.dropna()
        if not len(piv): continue
        K=piv.index[np.argmin(np.abs(piv.index.values-S))]
        if abs(K-S)/S>0.03: continue                    # 太偏离ATM则跳过
        strad=piv.loc[K,'C']+piv.loc[K,'P']
        try: sprd=(sp.loc[K,'C']+sp.loc[K,'P'])/strad*1e4
        except Exception: sprd=np.nan
        rows.append((d,bar,strad,S,K,sprd))
    if not rows: return None
    r=pd.DataFrame(rows,columns=['date','bar','straddle','S','K','spread_bp'])
    r['front_ym']=front
    return r

if __name__=='__main__':
    days=sorted(os.path.basename(p) for p in glob.glob(f'{OPT}/2026*'))
    with Pool(24) as p:
        parts=[x for x in p.imap(one_day,days) if x is not None]
    df=pd.concat(parts,ignore_index=True)
    # 到期日近似: 月份第三个周五 -> 用 front_ym 估剩余年化时间
    def texp(row):
        y=2000+row.front_ym//100; m=row.front_ym%100
        d3=pd.date_range(f'{y}-{m:02d}-01',periods=31,freq='D')
        fri=[x for x in d3 if x.month==m and x.weekday()==4]
        exp=fri[2]
        return max((exp-pd.Timestamp(row.date)).days,1)/365.0
    df['T']=df.apply(texp,axis=1)
    df['iv']=df.straddle/df.S/(0.7979*np.sqrt(df['T']))     # ATM近似
    df.to_parquet(f'{OUT}/mo_iv_5m.parquet',index=False)
    print(f'{len(df):,} 行, {df.date.nunique()} 天, IV中位={df.iv.median():.3f}, 价差中位={df.spread_bp.median():.0f}bp')
