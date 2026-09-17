# -*- coding: utf-8 -*-
"""改进②: 执行择时 —— 本来就要交易的量, 用波动预报挑低冲击的bar下手 (零额外换手)
因果调度: 每bar基础切片=剩余/剩余bar数, 乘以 clip((σ*/σ̂)^κ, 0.5, 2), 收盘前必须执行完
成本代理: 冲击 ∝ σ_realized (平方根律下同量同ADV时冲击正比于波动)
指标: 成交量加权实现波动 Σq·|r| / Σq  (越低越省)
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys, numpy as np, pandas as pd
sys.path.insert(0,f'{_R}/src')
from eval_index_ridge import load_all, ridge_fit
CODES=['000300.XSHG','000905.XSHG','000852.XSHG']

d,Hm=load_all(); seg=d.seg.values; tr=seg=='train'
z=np.load(f'{_R}/out/paper1_preds.npz')
d=d.copy(); d['pv1']=np.exp(z['yh0']); d['pv2']=np.exp(z['yh0']+z['p2']); d['seg']=seg
# 教科书: 滚动1日窗已实现波动 (v48 就是过去48根|r|均值, 已 shift1)
d['tb']=d['v48'].values
d['real']=d['aret'].values           # 该bar的实现|收益| = 冲击代理

def schedule(sig, sig_star, kappa=1.0, lo=0.5, hi=2.0):
    """因果切片调度: 返回每行的执行权重 q (按 date×j 分组, 每组合计=1)"""
    q=np.zeros(len(sig))
    return q

res={}
for name in ['uniform','tb','pv1','pv2']:
    ws=[]
    for (dt,j),g in d.groupby(['date','j'],sort=False):
        n=len(g); idx=g.index.values
        if name=='uniform':
            q=np.full(n,1.0/n)
        else:
            s=np.clip(g[name].values,1e-8,None)
            star=np.nanmedian(np.clip(d.loc[tr,name].values,1e-8,None))
            mult=np.clip((star/s),0.5,2.0)
            q=np.zeros(n); R=1.0
            for i in range(n):
                base=R/(n-i)
                qq=min(R, base*mult[i])
                if i==n-1: qq=R
                q[i]=qq; R-=qq
        ws.append(pd.Series(q,index=idx))
    res[name]=pd.concat(ws).reindex(d.index).values

print(f"{'调度信号':>26s}{'val':>10s}{'test':>10s}{'holdout':>10s}{'vs均匀节省':>12s}")
base_cost={}
for name,lab in [('uniform','均匀TWAP(基准)'),('tb','教科书滚动1日窗'),('pv1','V1 HAR型预报'),('pv2','V2 隐层预报')]:
    q=res[name]; row=f'{lab:>26s}'
    saves=[]
    for s_ in ['val','test','holdout']:
        m=(seg==s_)
        c=(q[m]*d['real'].values[m]).sum()/q[m].sum()*1e4
        if name=='uniform': base_cost[s_]=c
        row+=f'{c:>10.3f}'
        saves.append(100*(1-c/base_cost[s_]))
    row+=f'{np.mean(saves):>+11.2f}%'
    print(row)
print('\n(单位: 成交量加权实现|收益| bp, 越低越省; 冲击成本按平方根律与σ成正比)')

# 分指数看 holdout
print('\n分指数 holdout 节省 (vs 均匀TWAP):')
for jj,c in enumerate(CODES):
    m=(seg=='holdout')&(d.j==jj).values
    cu=(res['uniform'][m]*d['real'].values[m]).sum()/res['uniform'][m].sum()
    out=[]
    for name,lab in [('tb','教科书'),('pv1','V1'),('pv2','V2')]:
        cc=(res[name][m]*d['real'].values[m]).sum()/res[name][m].sum()
        out.append(f'{lab}={100*(1-cc/cu):+.2f}%')
    print(f'  {c}: '+'  '.join(out))
