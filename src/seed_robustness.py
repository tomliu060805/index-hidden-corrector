"""种子稳健性: GatedLinear 10个种子重训 (1h与2h目标), 报告各段增量的 mean±std/min/max
岭基线确定性; 只有修正器初始化与batch顺序受种子影响。输出 out/seed_robustness.csv
"""
import os
import glob, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_index_ridge import load_all, ridge_fit, r2, daily_ic
from run_paper1_stack import train_gated
from task3_horizon import load_close, seg_of

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGS = ['train', 'val', 'test', 'holdout']
SEEDS = list(range(100))
rows = []

# ---------- 1h 目标 ----------
d, Hm = load_all()
seg1 = d.seg.values
tr, va = seg1 == 'train', seg1 == 'val'
y = np.log(d['fvol'].values)
Xb = np.column_stack([
    np.column_stack([np.log(np.clip(d[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    pd.get_dummies(d['bar']).values, pd.get_dummies(d['j']).values]).astype(np.float32)
Xfull = np.column_stack([Xb, d.nll.values, d.ent.values, Hm]).astype(np.float32)
yh0 = ridge_fit(Xb, y, tr)
base = {s: r2(y, yh0, seg1 == s) for s in SEGS}
resid0 = y - yh0
print(f'[1h] 基线 R²: ' + ' '.join(f'{s}={base[s]:.4f}' for s in SEGS), flush=True)
for sd in SEEDS:
    torch.manual_seed(sd); np.random.seed(sd)
    p, ep = train_gated(Xfull, resid0, tr, va)
    rec = dict(target='1h', seed=sd, epochs=ep)
    for s in SEGS:
        rec[f'd_{s}'] = r2(y, yh0 + p, seg1 == s) - base[s]
    ic, icir, n = daily_ic(d, p, resid0, seg1 == 'holdout')
    rec['ic_holdout'], rec['icir_holdout'] = ic, icir
    rows.append(rec)
    print(f"[1h] seed {sd}: Δval={rec['d_val']:+.4f} Δtest={rec['d_test']:+.4f} "
          f"Δholdout={rec['d_holdout']:+.4f} IC={ic:+.3f} ({ep}ep)", flush=True)

# ---------- 2h 目标 ----------
lr, dates = load_close()
NG = lr.shape[0]; H24 = 24
alr = np.abs(lr)
fvol24 = np.full(lr.shape, np.nan)
for i in range(NG - H24):
    if i % 48 > 47 - H24: continue
    fvol24[i] = np.nanmean(alr[i + 1:i + 1 + H24], 0)
pv = pd.DataFrame(alr).shift(1)
v12 = pv.rolling(12, min_periods=6).mean().values
v48 = pv.rolling(48, min_periods=24).mean().values
v240 = pv.rolling(240, min_periods=120).mean().values
fs = sorted(glob.glob(f'{B}/out/hidden/chunk_*.npz'))
Hs, Ms = [], []
for f in fs:
    z = np.load(f); Hs.append(z['h']); Ms.append(z['meta'])
Hi = np.concatenate(Hs).astype(np.float32); Mi = np.concatenate(Ms)
ti = Mi[:, 0].astype(np.int64); ji = Mi[:, 1].astype(np.int64)
keep = (ti % 48) <= 47 - H24
d2 = pd.DataFrame({'t': ti[keep], 'j': ji[keep], 'nll': Mi[keep, 2], 'ent': Mi[keep, 3]})
d2['y'] = np.log(np.clip(fvol24[d2.t, d2.j], 1e-8, None))
for nm, a in [('v12', v12), ('v48', v48), ('v240', v240)]:
    d2[nm] = a[d2.t, d2.j]
d2['aret'] = alr[d2.t, d2.j]
d2['date'] = dates[d2.t // 48]; d2['bar'] = d2.t % 48
ok = d2[['y', 'v12', 'v48', 'v240', 'aret']].notna().all(1).values & np.isfinite(d2.y.values)
d2 = d2[ok].reset_index(drop=True); Hk = Hi[keep][ok]
seg2 = seg_of(d2.date.values)
y2 = d2.y.values
Xb2 = np.column_stack([
    np.column_stack([np.log(np.clip(d2[c].values, 1e-8, None)) for c in ['v12', 'v48', 'v240', 'aret']]),
    pd.get_dummies(d2.bar).values, pd.get_dummies(d2.j).values]).astype(np.float32)
Xf2 = np.column_stack([Xb2, d2.nll.values, d2.ent.values, Hk]).astype(np.float32)
tr2, va2 = seg2 == 'train', seg2 == 'val'
yh02 = ridge_fit(Xb2, y2, tr2)
base2 = {s: r2(y2, yh02, seg2 == s) for s in SEGS}
resid02 = y2 - yh02
print(f'[2h] 基线 R²: ' + ' '.join(f'{s}={base2[s]:.4f}' for s in SEGS), flush=True)
for sd in SEEDS:
    torch.manual_seed(sd); np.random.seed(sd)
    p, ep = train_gated(Xf2, resid02, tr2, va2)
    rec = dict(target='2h', seed=sd, epochs=ep)
    for s in SEGS:
        rec[f'd_{s}'] = r2(y2, yh02 + p, seg2 == s) - base2[s]
    rec['ic_holdout'] = rec['icir_holdout'] = np.nan
    rows.append(rec)
    print(f"[2h] seed {sd}: Δval={rec['d_val']:+.4f} Δtest={rec['d_test']:+.4f} "
          f"Δholdout={rec['d_holdout']:+.4f} ({ep}ep)", flush=True)

t = pd.DataFrame(rows)
t.to_csv(f'{B}/out/seed_robustness.csv', index=False)
print('\n===== 汇总 (10种子) =====')
for tgt in ['1h', '2h']:
    s = t[t.target == tgt]
    for c in ['d_val', 'd_test', 'd_holdout']:
        print(f'{tgt} {c}: mean={s[c].mean():+.4f} std={s[c].std():.4f} '
              f'min={s[c].min():+.4f} max={s[c].max():+.4f}')
    if tgt == '1h':
        print(f"1h holdout日IC: mean={s.ic_holdout.mean():+.4f} std={s.ic_holdout.std():.4f} "
              f"min={s.ic_holdout.min():+.4f}")
