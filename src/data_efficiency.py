"""数据效率: 冻结TSFM隐层+GL vs 从零Transformer, 训练数据只用train末尾{5%,15%,50%}天
两臂都在同一缩减数据上重拟合M0基线; 预训练表示的经典优势应在低数据端显现
"""
import os
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_index_ridge import load_all, ridge_fit, r2
from run_paper1_stack import train_gated
import extract_index_hidden as E

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGS = ['val', 'test', 'holdout']
torch.set_num_threads(100)
L = 128

d, Hm = load_all()
seg = d.seg.values; va = seg == 'val'
dates = d.date.values
y = np.log(d['fvol'].values)
Xb = np.column_stack([
    np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12','v48','v240','aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)

print('构建原始窗口...', flush=True)
arr, _dts = E.load_grid()
W = np.empty((len(d), L, 6), np.float32)
t_i = d.t.values; j_i = d.j.values
for k in range(len(d)):
    w = arr[t_i[k]-L+1:t_i[k]+1, j_i[k], :]
    m, s = w.mean(0), w.std(0)
    W[k] = np.clip((w-m)/(s+1e-5), -5, 5)

class TF(nn.Module):
    def __init__(self, side):
        super().__init__()
        self.proj = nn.Linear(6, 64)
        lyr = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lyr, 2)
        self.pos = nn.Parameter(torch.randn(1, L, 64)*0.02)
        self.head = nn.Sequential(nn.Linear(128+side, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, w, s):
        z = self.enc(self.proj(w) + self.pos)
        return self.head(torch.cat([z[:, -1], z.mean(1), s], -1)).squeeze(-1)

def train_tf(trm, resid, Xbn, epochs=40, bs=4096):
    torch.manual_seed(0); np.random.seed(0)
    net = TF(Xbn.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Wt = torch.from_numpy(W); St = torch.from_numpy(Xbn); yt = torch.from_numpy(resid)
    idx = np.where(trm)[0]; vai = np.where(va)[0]
    best, bstate, pat = np.inf, None, 0
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i+bs]; opt.zero_grad()
            ((net(Wt[b], St[b]) - yt[b])**2).mean().backward(); opt.step()
        net.eval(); pv = []
        with torch.no_grad():
            for i in range(0, len(vai), 16384):
                pv.append(net(Wt[vai[i:i+16384]], St[vai[i:i+16384]]).numpy())
        v = ((np.concatenate(pv) - resid[vai])**2).mean()
        if v < best - 1e-6: best, bstate, pat = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 5: break
    net.load_state_dict(bstate); net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(d), 16384):
            out.append(net(Wt[i:i+16384], St[i:i+16384]).numpy())
    return np.concatenate(out)

trdays = np.array(sorted(pd.unique(dates[seg == 'train'])))
for frac in [0.05, 0.15, 0.5, 1.0]:
    keep = trdays[-int(len(trdays)*frac):]
    trm = np.isin(dates, keep)
    yh0 = ridge_fit(Xb, y, trm)
    resid = (y - yh0).astype(np.float32)
    mu, sd = Xb[trm].mean(0), Xb[trm].std(0)+1e-8
    Xbn = ((Xb-mu)/sd).astype(np.float32)
    # GL on hidden
    Xg = np.column_stack([Xb, d.nll.values, d.ent.values, Hm]).astype(np.float32)
    torch.manual_seed(0); np.random.seed(0)
    pg, _ = train_gated(Xg, resid, trm, va)
    # scratch TF
    pt = train_tf(trm, resid, Xbn)
    row = f'frac={frac:.2f} (train天={len(keep)}):'
    for s in SEGS:
        m = seg == s
        dg = r2(y, yh0+pg, m)-r2(y, yh0, m)
        dt_ = r2(y, yh0+pt, m)-r2(y, yh0, m)
        row += f'  {s}: GL={dg:+.4f} TF={dt_:+.4f}'
    print(row, flush=True)
