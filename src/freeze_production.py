# -*- coding: utf-8 -*-
"""固化生产模型包: 岭系数 + GatedLinear权重 + 全部配置常量 -> production/model_bundle.pt
使 production/ 完全自包含, 推理不再依赖训练数据
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0,f'{_R}/src')
from eval_index_ridge import load_all, ridge_fit
from task3_horizon import load_close, seg_of
import glob
B=f'{_R}'
H24=24            # 生产用 2h 预报 (阶段四选定)

# —— 重建 2h 训练面板 (与 save_preds_2h.py 完全一致) ——
lr,dates=load_close(); NG=lr.shape[0]; alr=np.abs(lr)
fvol24=np.full(lr.shape,np.nan)
for i in range(NG-H24):
    if i%48>47-H24: continue
    fvol24[i]=np.nanmean(alr[i+1:i+1+H24],0)
pv=pd.DataFrame(alr).shift(1)
v12=pv.rolling(12,min_periods=6).mean().values
v48=pv.rolling(48,min_periods=24).mean().values
v240=pv.rolling(240,min_periods=120).mean().values
Hs,Ms=[],[]
for f in sorted(glob.glob(f'{B}/out/hidden/chunk_*.npz')):
    z=np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
Hi=np.concatenate(Hs).astype(np.float32); Mi=np.concatenate(Ms)
ti=Mi[:,0].astype(np.int64); ji=Mi[:,1].astype(np.int64)
keep=(ti%48)<=47-H24
d=pd.DataFrame({'t':ti[keep],'j':ji[keep],'nll':Mi[keep,2],'ent':Mi[keep,3]})
d['y']=np.log(np.clip(fvol24[d.t,d.j],1e-8,None))
for nm,a in [('v12',v12),('v48',v48),('v240',v240)]: d[nm]=a[d.t,d.j]
d['aret']=alr[d.t,d.j]; d['date']=dates[d.t//48]; d['bar']=d.t%48
ok=d[['y','v12','v48','v240','aret']].notna().all(1).values&np.isfinite(d.y.values)
d=d[ok].reset_index(drop=True); Hk=Hi[keep][ok]
seg=seg_of(d.date.values); tr=seg=='train'; va=seg=='val'
y=d.y.values
bars=np.sort(d.bar.unique()); idxs=np.sort(d.j.unique())
def build_Xb(dd):
    rv=np.column_stack([np.log(np.clip(dd[c].values,1e-8,None)) for c in ['v12','v48','v240','aret']])
    bo=np.zeros((len(dd),len(bars)),np.float32)
    bo[np.arange(len(dd)),np.searchsorted(bars,dd.bar.values)]=1
    io=np.zeros((len(dd),len(idxs)),np.float32)
    io[np.arange(len(dd)),np.searchsorted(idxs,dd.j.values)]=1
    return np.column_stack([rv,bo,io]).astype(np.float32)
Xb=build_Xb(d)
X=np.column_stack([Xb,d.nll.values,d.ent.values,Hk]).astype(np.float32)
# 岭系数(显式求解并保存)
Xa=np.column_stack([Xb,np.ones(len(Xb))]).astype(np.float64)
lam=1.0*tr.sum()/1e4
beta=np.linalg.solve(Xa[tr].T@Xa[tr]+lam*np.eye(Xa.shape[1]), Xa[tr].T@y[tr])
yh0=Xa@beta
resid0=(y-yh0).astype(np.float32)
# GatedLinear (seed0, 与生产一致)
from run_paper1_stack import GatedLinear
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(30)
mu,sd=X[tr].mean(0),X[tr].std(0)+1e-8
Xn=((X-mu)/sd).astype(np.float32)
Xt=torch.from_numpy(Xn); yt=torch.from_numpy(resid0)
net=GatedLinear(X.shape[1],32); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
idx=np.where(tr)[0]; vi=np.where(va)[0]; best,bst,pat=np.inf,None,0
for ep in range(60):
    net.train(); perm=np.random.permutation(idx)
    for i in range(0,len(perm),8192):
        b=perm[i:i+8192]; opt.zero_grad(); ((net(Xt[b])-yt[b])**2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad(): v=((net(Xt[vi]).numpy()-resid0[vi])**2).mean()
    if v<best-1e-6: best,bst,pat=v,{k:t.clone() for k,t in net.state_dict().items()},0
    else:
        pat+=1
        if pat>=8: break
net.load_state_dict(bst); net.eval()
with torch.no_grad(): p2=net(Xt).numpy()
from eval_index_ridge import r2
for s_ in ['val','test','holdout']:
    m=seg==s_
    print(f'  {s_}: M0={r2(y,yh0,m):.4f} -> M2={r2(y,yh0+p2,m):.4f} (Δ{r2(y,yh0+p2,m)-r2(y,yh0,m):+.4f})')
sigma_star=float(np.median(np.exp(yh0[tr]+p2[tr])))
torch.save(dict(ridge_beta=beta, bars=bars, idxs=idxs,
                train_codes=['000300.XSHG','000905.XSHG','000852.XSHG'], gl_state=bst, gl_dim=X.shape[1], gl_rank=32,
                mu=mu, sd=sd, sigma_star=sigma_star, H=H24,
                cfg=dict(IC=dict(index='000905.XSHG', gate_days=60, delta=0.3, cap=3.0, floor=1.0),
                         IM=dict(index='000852.XSHG', gate_days=5,  delta=0.3, cap=3.0, floor=1.0)),
                note='2h horizon production bundle; ridge on [logRV4, bar-onehot, idx-onehot, 1]'),
           f'{B}/production/model_bundle.pt')
print(f'\nσ*={sigma_star:.6f}  bundle已保存 production/model_bundle.pt')
