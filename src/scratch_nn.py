"""致命消融: 同一窗口从零训练 vs 冻结TSFM表示
输入 = 与Kronos完全相同的128×6窗口(窗口内z-score, clip±5) + 基线特征Xb (不给nll/熵, 那是预训练产物)
模型: (a) GRU-64  (b) Transformer d64L2  (c) PCA-512+GatedLinear (线性表示对照)
协议与生产M2一致: 预测M0残差, val早停, Adam; 目标=1h波动
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, f'{_R}/src')
from eval_index_ridge import load_all, ridge_fit, r2
import extract_index_hidden as E

B = f'{_R}'
SEGS = ['val', 'test', 'holdout']
torch.set_num_threads(100)
L = 128

d, Hm = load_all()
seg = d.seg.values; tr = seg == 'train'; va = seg == 'val'
y = np.log(d['fvol'].values)
Xb = np.column_stack([
    np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
yh0 = ridge_fit(Xb, y, tr); resid0 = (y - yh0).astype(np.float32)
prod = {'val': +0.0647, 'test': +0.0415, 'holdout': +0.0964}

print('构建原始窗口...', flush=True)
arr, dates = E.load_grid()
W = np.empty((len(d), L, 6), np.float32)
t_i = d.t.values; j_i = d.j.values
for k in range(len(d)):
    w = arr[t_i[k] - L + 1:t_i[k] + 1, j_i[k], :]
    m, s = w.mean(0), w.std(0)
    W[k] = np.clip((w - m) / (s + 1e-5), -5, 5)
print(f'窗口张量 {W.shape} {W.nbytes/1e9:.1f}GB', flush=True)

mu_b, sd_b = Xb[tr].mean(0), Xb[tr].std(0) + 1e-8
Xbn = ((Xb - mu_b) / sd_b).astype(np.float32)


def report(nm, p):
    row = f'{nm:>24s}:'
    for s in SEGS:
        m = seg == s
        row += f'  {s}Δ={r2(y, yh0 + p, m) - r2(y, yh0, m):+.4f}(生产{prod[s]:+.3f})'
    print(row, flush=True)


class SeqModel(nn.Module):
    def __init__(self, kind, side_dim):
        super().__init__()
        self.kind = kind
        if kind == 'gru':
            self.enc = nn.GRU(6, 64, batch_first=True)
            rep = 64
        else:
            self.proj = nn.Linear(6, 64)
            lyr = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(lyr, 2)
            self.pos = nn.Parameter(torch.randn(1, L, 64) * 0.02)
            rep = 128
        self.head = nn.Sequential(nn.Linear(rep + side_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, w, side):
        if self.kind == 'gru':
            _, h = self.enc(w)
            r = h[-1]
        else:
            z = self.enc(self.proj(w) + self.pos)
            r = torch.cat([z[:, -1], z.mean(1)], -1)
        return self.head(torch.cat([r, side], -1)).squeeze(-1)


def train_seq(kind, epochs=40, bs=4096, lr=1e-3):
    torch.manual_seed(0); np.random.seed(0)
    net = SeqModel(kind, Xbn.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    Wt = torch.from_numpy(W); St = torch.from_numpy(Xbn); yt = torch.from_numpy(resid0)
    idx = np.where(tr)[0]; vai = np.where(va)[0]
    best, bstate, pat = np.inf, None, 0
    t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            ((net(Wt[b], St[b]) - yt[b]) ** 2).mean().backward(); opt.step()
        net.eval(); pv = []
        with torch.no_grad():
            for i in range(0, len(vai), 16384):
                pv.append(net(Wt[vai[i:i + 16384]], St[vai[i:i + 16384]]).numpy())
        vloss = ((np.concatenate(pv) - resid0[vai]) ** 2).mean()
        print(f'  [{kind}] ep{ep} val={vloss:.5f} {(time.time()-t0)/60:.1f}min', flush=True)
        if vloss < best - 1e-6:
            best, bstate, pat = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 5: break
    net.load_state_dict(bstate); net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(d), 16384):
            out.append(net(Wt[i:i + 16384], St[i:i + 16384]).numpy())
    return np.concatenate(out)


# (c) PCA-512 线性表示 + GatedLinear
print('PCA-512 ...', flush=True)
from sklearn.decomposition import PCA
flat = W.reshape(len(d), -1)
pca = PCA(n_components=512, svd_solver='randomized', random_state=0)
pca.fit(flat[tr][::4])
P512 = pca.transform(flat).astype(np.float32)
from run_paper1_stack import train_gated
torch.manual_seed(0); np.random.seed(0)
p_pca, ep = train_gated(np.column_stack([Xb, P512]).astype(np.float32), resid0, tr, va)
report(f'PCA512+GatedLinear({ep}ep)', p_pca)

p_tf = train_seq('tfm')
report('Transformer-d64L2', p_tf)
p_gru = train_seq('gru')
report('GRU-64', p_gru)
np.savez(f'{B}/out/scratch_preds.npz', p_pca=p_pca, p_tf=p_tf, p_gru=p_gru)
