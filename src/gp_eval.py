# -*- coding: utf-8 -*-
"""③按经济指标重评三种训练目标 + ④Garleanu-Pedersen最优交易框架应用"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys, numpy as np, pandas as pd
sys.path.insert(0,f'{_R}/src')
from eval_index_ridge import load_all, ridge_fit
import backtest_futures as BF
B=f'{_R}'; BY=242.0
d,Hm=load_all(); seg=d.seg.values; tr=seg=='train'
y=np.log(d['fvol'].values)
Xb=np.column_stack([np.column_stack([np.log(np.clip(d[c].values,1e-8,None)) for c in ['v12','v48','v240','aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
yh0=ridge_fit(Xb,y,tr)
ca=np.load(f'{B}/out/costaware_preds.npz')
z=np.load(f'{B}/out/paper1_preds.npz')
PRED={'(a)标准MSE':np.exp(yh0+ca['mse']),'(b)截断加权MSE':np.exp(yh0+ca['wmse']),
      '(c)决策导向净P&L':np.exp(yh0+ca['pnl'])}
# 映射到期货网格
def to_grid(P, j, pv_flat):
    mj=z['j']==j
    fm=pd.Series(np.arange(mj.sum()),index=pd.MultiIndex.from_arrays([z['date'][mj],z['bar'][mj]]))
    out=np.full(P['Nd']*48,np.nan)
    dd=np.repeat(P['days'],48); bb=np.tile(np.arange(48),P['Nd'])
    idx=fm.reindex(pd.MultiIndex.from_arrays([dd,bb])); ok=idx.notna().values
    out[ok]=pv_flat[mj][idx.values[ok].astype(int)]
    return out
def ema(w,a):
    o=np.empty_like(w); cur=w[0]
    for i in range(len(w)): cur=(1-a)*cur+a*w[i]; o[i]=cur
    return o
def evaluate(P,sig,trm,a_ema=0.0,delta=0.0,floor=0.0,cap=3.0):
    s=np.clip(sig,1e-8,None); tgt=np.nanmedian(s[trm&np.isfinite(s)])
    w=pd.Series(np.clip(tgt/s,floor,cap)).ffill().fillna(1.0).values
    if a_ema>0: w=ema(w,a_ema)
    if delta>0:
        o=np.empty_like(w); cur=w[0]
        for i in range(len(w)):
            if abs(w[i]-cur)>=delta: cur=w[i]
            o[i]=cur
        w=o
    return BF.run(P,w,'C')[0], w

print('=== ③ 成本感知训练目标 —— 按经济指标 (对称VT+EMA a=0.1, 平今成本) ===')
print(f"{'训练目标':>20s}{'IC test':>10s}{'IC hold':>10s}{'IM test':>10s}{'IM hold':>10s}")
for lab,pv in PRED.items():
    row=f'{lab:>20s}'
    for prod,j in [('IC',1),('IM',2)]:
        P=BF.build_product(prod); sd=BF.seg_of(P['days']); trm=np.repeat(sd=='train',48)
        dd,_=evaluate(P,to_grid(P,j,pv),trm,a_ema=0.1)
        for s_ in ['test','holdout']:
            m=sd==s_; row+=f'{dd[m].mean()/dd[m].std()*np.sqrt(BY):>10.2f}'
    print(row)

print('\n=== ④ Garleanu-Pedersen 最优交易框架 ===')
# 参数估计 (IC, train段)
P=BF.build_product('IC'); sd=BF.seg_of(P['days']); trm=np.repeat(sd=='train',48)
pv=P['pv2']; s=np.clip(pv,1e-8,None); tgt=np.nanmedian(s[trm&np.isfinite(s)])
aim=pd.Series(np.clip(tgt/s,0,3.0)).ffill().fillna(1.0).values          # 目标(aim)仓位
am=aim[trm]
phi=1-pd.Series(am).autocorr(1)              # 信号(aim)每bar均值回复速度
r=np.nan_to_num(P['r']); sig2=r[trm].var()   # 每bar收益方差
dw=np.abs(np.diff(aim[trm])); typ=np.median(dw[dw>1e-9])
c_lin=1.5e-4                                  # 平均单边成本(开0.55/平3.77混合近似)
lam=2*c_lin/max(typ,1e-3)                     # 二次成本系数 (在典型交易量处匹配线性成本)
rho=1-np.exp(-1/(242*48))                     # 每bar贴现
# 关键: GP公式里的风险项是 γ·Σ (风险厌恶×方差), 与成本λ同量纲
gamma=1.0/(3.0*np.sqrt(sig2))                 # 风险厌恶(使最优仓位量级≈1~3)
G=gamma*sig2                                  # 有效风险惩罚系数
a_star=(np.sqrt((G+lam*rho)**2+4*G*lam)-(G+lam*rho))/(2*lam)
shrink_f=G/(G+a_star*lam*phi)                 # GP: 信号衰减越快, aim收缩越多
print(f'  参数估计: 信号回复速度φ={phi:.4f}/bar  收益方差σ²={sig2:.2e}  典型调仓{typ:.3f}')
print(f'            二次成本λ={lam:.2e}  有效风险G=γΣ={G:.2e}  G/λ={G/lam:.4f}  贴现ρ={rho:.2e}')
print(f'  >> GP 最优交易速率 a* = {a_star:.4f}   (经验网格最优 a=0.10)')
print(f'  >> GP 目标收缩因子 = G/(G+a*·λ·φ) = {shrink_f:.3f}  (信号衰减越快收缩越多)')
print('\n  GP规则 vs 经验EMA vs 生产截断规则 (平今成本):')
print(f"{'规则':>26s}{'IC test':>10s}{'IC hold':>10s}{'IM test':>10s}{'IM hold':>10s}{'IC换手':>9s}")
rules=[('生产截断(floor1,cap3,δ0.3)',dict(floor=1.0,delta=0.3)),
       ('经验EMA a=0.10',dict(a_ema=0.10)),
       (f'GP最优 a*={a_star:.3f}',dict(a_ema=float(np.clip(a_star,0.005,0.9)))),
       (f'GP+收缩aim',dict(a_ema=float(np.clip(a_star,0.005,0.9)),shrink=True))]
for lab,cfg in rules:
    row=f'{lab:>26s}'; turn_ic=None
    for prod,j in [('IC',1),('IM',2)]:
        Pp=BF.build_product(prod); sdd=BF.seg_of(Pp['days']); trmm=np.repeat(sdd=='train',48)
        sig=Pp['pv2'].copy()
        if cfg.get('shrink'):
            sh=shrink_f
            ss=np.clip(sig,1e-8,None); tg=np.nanmedian(ss[trmm&np.isfinite(ss)])
            wraw=np.clip(tg/ss,0,3.0); wraw=1.0+(wraw-1.0)*sh
            w=pd.Series(wraw).ffill().fillna(1.0).values
            w=ema(w,cfg['a_ema']); dd=BF.run(Pp,w,'C')[0]
        else:
            dd,w=evaluate(Pp,sig,trmm,a_ema=cfg.get('a_ema',0.0),delta=cfg.get('delta',0.0),floor=cfg.get('floor',0.0))
        for s_ in ['test','holdout']:
            m=sdd==s_; row+=f'{dd[m].mean()/dd[m].std()*np.sqrt(BY):>10.2f}'
        if prod=='IC':
            m2=np.isin(sdd,['test','holdout'])
            turn_ic=np.abs(np.diff(np.r_[1.0,w])).reshape(-1,48)[m2].sum(1).mean()
    print(row+f'{turn_ic:>9.2f}')
