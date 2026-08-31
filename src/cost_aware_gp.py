# -*- coding: utf-8 -*-
"""改进③成本感知端到端训练 + ④Garleanu-Pedersen最优交易框架
③ 三种训练目标: (a)标准MSE (b)截断加权MSE (c)决策导向(直接优化净P&L)
④ GP: 二次成本->部分调整(EMA), 线性成本->无交易带; 由参数解出最优交易速率a, 与经验最优对比
"""
import os
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_index_ridge import load_all, ridge_fit, r2
from run_paper1_stack import GatedLinear
from task3_horizon import load_close
torch.set_num_threads(40)
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d,Hm=load_all(); seg=d.seg.values; tr=seg=='train'; va=seg=='val'
y=np.log(d['fvol'].values)
Xb=np.column_stack([np.column_stack([np.log(np.clip(d[c].values,1e-8,None)) for c in ['v12','v48','v240','aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
X=np.column_stack([Xb,d.nll.values,d.ent.values,Hm]).astype(np.float32)
yh0=ridge_fit(Xb,y,tr); resid0=(y-yh0).astype(np.float32)
# 下一根bar收益 (决策t作用于t+1)
lr,dates=load_close(); NG=lr.shape[0]
nxt=np.full(lr.shape,np.nan); nxt[:-1]=lr[1:]
rn=nxt[d.t.values,d.j.values].astype(np.float32)
rn=np.nan_to_num(rn)
mu,sd=X[tr].mean(0),X[tr].std(0)+1e-8
Xn=((X-mu)/sd).astype(np.float32)
Xt=torch.from_numpy(Xn); yt=torch.from_numpy(resid0); rt=torch.from_numpy(rn)
y0t=torch.from_numpy(yh0.astype(np.float32))
logstar=float(np.median(y[tr]))          # 目标波动水平(log)
# 截断权重: 仓位处于内点(1<w<3)的指示 -> 预报差异才有意义
w_ref=np.clip(np.exp(logstar-(yh0+np.load(f'{B}/out/paper1_preds.npz')['p2'])),1.0,3.0)
interior=((w_ref>1.001)&(w_ref<2.999)).astype(np.float32)
it=torch.from_numpy(interior)
print(f'内点样本占比: train={interior[tr].mean():.1%}  holdout={interior[seg=="holdout"].mean():.1%}')

def train(mode, epochs=60, bs=8192, lr_=1e-3):
    torch.manual_seed(0); np.random.seed(0)
    net=GatedLinear(X.shape[1],32); opt=torch.optim.Adam(net.parameters(),lr=lr_,weight_decay=1e-4)
    idx=np.where(tr)[0]; vi=np.where(va)[0]
    best,bs_,pat=np.inf,None,0
    for ep in range(epochs):
        net.train(); perm=np.random.permutation(idx)
        for i in range(0,len(perm),bs):
            b=perm[i:i+bs]; opt.zero_grad()
            p=net(Xt[b])
            if mode=='mse': loss=((p-yt[b])**2).mean()
            elif mode=='wmse': loss=(it[b]*(p-yt[b])**2).mean()/(it[b].mean()+1e-6)
            else:                                    # 决策导向: 直接优化净收益
                yh=y0t[b]+p
                w=1.0+2.0*torch.sigmoid((logstar-yh)*3.0)      # 软截断到[1,3]
                loss=-(w*rt[b]).mean()*1e4 + 0.02*((w-1.0)**2).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv=net(Xt[vi])
            if mode=='mse': v=((pv-yt[vi])**2).mean().item()
            elif mode=='wmse': v=(it[vi]*(pv-yt[vi])**2).mean().item()/(it[vi].mean().item()+1e-6)
            else:
                yh=y0t[vi]+pv; w=1.0+2.0*torch.sigmoid((logstar-yh)*3.0)
                v=-(w*rt[vi]).mean().item()*1e4
        if v<best-1e-7: best,bs_,pat=v,{k:t.clone() for k,t in net.state_dict().items()},0
        else:
            pat+=1
            if pat>=8: break
    net.load_state_dict(bs_); net.eval()
    with torch.no_grad(): return net(Xt).numpy(), ep

print(f"\n=== ③ 成本感知训练目标对照 (预报增量 ΔR²) ===")
print(f"{'训练目标':>26s}{'val':>10s}{'test':>10s}{'holdout':>10s}")
preds={}
for mode,lab in [('mse','(a)标准MSE=生产版'),('wmse','(b)截断加权MSE'),('pnl','(c)决策导向净P&L')]:
    p,ep=train(mode); preds[mode]=p
    row=f'{lab:>26s}'
    for s_ in ['val','test','holdout']:
        m=seg==s_; row+=f'{r2(y,yh0+p,m)-r2(y,yh0,m):>+10.4f}'
    print(row+f'  ({ep}ep)')
np.savez(f'{B}/out/costaware_preds.npz',**preds)
