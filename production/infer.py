# -*- coding: utf-8 -*-
"""生产推理引擎: 每 5 分钟输出 IC/IM 的目标仓位
用法:
  python infer.py --date 2026-08-14                 # 复算某日全天(回放/对账)
  python infer.py --date 2026-08-14 --bar 20        # 只算某个决策点(实盘)
  python infer.py --replay 2026-08-01 2026-08-14    # 区间回放, 输出CSV
自包含: 只依赖 model_bundle.pt + 权威 1min 行情源 + 冻结 Kronos 权重
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根(随目录搬迁自动跟随)
import sys as _sys; _sys.path.insert(0, _os.path.join(_R, 'src'))
import config as CFG   # 数据路径走环境变量, 见 src/config.py
import os, sys, json, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0, CFG.KRONOS_REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
IDX_SRC = CFG.INDEX_1M_DIR
L, CLIP = 128, 5.0
COLS = ['open', 'high', 'low', 'close', 'volume', 'amount']
T48 = ['09:35','09:40','09:45','09:50','09:55','10:00','10:05','10:10','10:15','10:20','10:25','10:30',
       '10:35','10:40','10:45','10:50','10:55','11:00','11:05','11:10','11:15','11:20','11:25','11:30',
       '13:05','13:10','13:15','13:20','13:25','13:30','13:35','13:40','13:45','13:50','13:55','14:00',
       '14:05','14:10','14:15','14:20','14:25','14:30','14:35','14:40','14:45','14:50','14:55','15:00']


class Engine:
    def __init__(self, bundle=f'{HERE}/model_bundle.pt', threads=4):
        from model import Kronos, KronosTokenizer
        torch.set_num_threads(threads)
        self.B = torch.load(bundle, weights_only=False)
        self.tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
        self.mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
        sys.path.insert(0, f'{_R}/src'); from run_paper1_stack import GatedLinear
        self.gl = GatedLinear(self.B['gl_dim'], self.B['gl_rank'])
        self.gl.load_state_dict(self.B['gl_state']); self.gl.eval()
        # ★必须与训练时的指数集合与顺序一致(指数独热按此编码), 产品只取其中两个
        self.codes = self.B.get('train_codes', ['000300.XSHG','000905.XSHG','000852.XSHG'])
        self.prods = list(self.B['cfg'].keys())

    # ---------- 数据: 取到 date 为止的足量历史 5m bar ----------
    def load_bars(self, date, lookback_days=12):
        days = sorted(os.path.basename(f)[:10] for f in os.listdir(IDX_SRC) if f.endswith('.parquet'))
        i = days.index(date)
        use = days[max(0, i - lookback_days):i + 1]
        parts = []
        for d in use:
            df = pd.read_parquet(f'{IDX_SRC}/{d}.parquet',
                                 columns=['datetime', 'code', 'open', 'high', 'low', 'close', 'volume', 'money'])
            df = df[df.code.isin(self.codes)]
            m = df.datetime.dt.hour * 60 + df.datetime.dt.minute
            df = df[(m.between(571, 690)) | (m.between(781, 900))].copy()
            mm = df.datetime.dt.hour * 60 + df.datetime.dt.minute
            df['bar'] = np.where(mm <= 690, (mm - 571) // 5, 24 + (mm - 781) // 5)
            g = df.groupby(['code', 'bar'])
            o = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                              'close': g['close'].last(), 'volume': g['volume'].sum(),
                              'amount': g['money'].sum()}).reset_index()
            o.insert(0, 'date', d); parts.append(o)
        return pd.concat(parts, ignore_index=True), use

    # ---------- 单个决策点推理 ----------
    def forecast(self, panel, days, date, bar):
        from model.kronos import calc_time_stamps
        di = {d: k for k, d in enumerate(days)}
        NG = len(days) * 48
        arr = np.full((NG, len(self.codes), 6), np.nan, np.float32)
        ci = {c: k for k, c in enumerate(self.codes)}
        g = panel.date.map(di).values * 48 + panel.bar.values
        s = panel.code.map(ci).values
        for k, c in enumerate(COLS):
            arr[g, s, k] = panel[c].values
        ts = pd.to_datetime([f'{d} {t}' for d in days for t in T48])
        stamp = calc_time_stamps(ts.to_series()).values.astype(np.float32)
        t = di[date] * 48 + bar
        if t - L + 1 < 0:
            raise RuntimeError('历史不足 128 根 bar')
        out = {}
        ws, sts, keep = [], [], []
        for j in range(len(self.codes)):
            w = arr[t - L + 1:t + 1, j, :]
            if not np.isfinite(w).all(): continue
            mu_, sd_ = w.mean(0), w.std(0)
            ws.append(np.clip((w - mu_) / (sd_ + 1e-5), -CLIP, CLIP)); sts.append(stamp[t - L + 1:t + 1]); keep.append(j)
        if not ws: raise RuntimeError('该决策点无有效窗口')
        x = torch.from_numpy(np.stack(ws).astype(np.float32))
        st = torch.from_numpy(np.stack(sts).astype(np.float32))
        with torch.no_grad():
            s1, s2 = self.tok.encode(x, half=True)
            import torch.nn.functional as F
            logits, ctx = self.mdl.decode_s1(s1, s2, st)
            b = s1.shape[0]; ar = torch.arange(b)
            n1 = -F.log_softmax(logits[:, L - 2, :].float(), -1)[ar, s1[:, L - 1]]
            s2lg = self.mdl.decode_s2(ctx[:, :L - 1, :], s1[:, L - 1:L])
            n2 = -F.log_softmax(s2lg[:, -1, :].float(), -1)[ar, s2[:, L - 1]]
            pn = F.softmax(logits[:, L - 1, :].float(), -1)
            ent = -(pn * torch.log(pn + 1e-12)).sum(-1)
            hid = ctx[:, -1, :].numpy().astype(np.float32)
        # 已实现波动特征 (与训练同口径: shift(1) 后滚动)
        close = arr[:, :, 3]
        lr = np.full_like(close, np.nan); lr[1:] = np.log(close[1:] / close[:-1])
        lr[np.arange(NG) % 48 == 0] = np.nan
        al = pd.DataFrame(np.abs(lr)).shift(1)
        feats = {k: al.rolling(w_, min_periods=w_ // 2).mean().values[t]
                 for k, w_ in [('v12', 12), ('v48', 48), ('v240', 240)]}
        aret = np.abs(lr)[t]
        bars_, idxs_ = self.B['bars'], self.B['idxs']
        rows = []
        for k, j in enumerate(keep):
            rv = [np.log(max(feats[c][j], 1e-8)) for c in ['v12', 'v48', 'v240']] + [np.log(max(aret[j], 1e-8))]
            bo = np.zeros(len(bars_)); bi = np.searchsorted(bars_, bar)
            if bi < len(bars_) and bars_[bi] == bar: bo[bi] = 1
            io = np.zeros(len(idxs_)); ii = np.searchsorted(idxs_, j)
            if ii < len(idxs_) and idxs_[ii] == j: io[ii] = 1
            rows.append((j, np.concatenate([rv, bo, io]).astype(np.float32)))
        res = {}
        for j, xb in rows:
            yh0 = float(np.concatenate([xb, [1.0]]) @ self.B['ridge_beta'])
            k = keep.index(j)
            xf = np.concatenate([xb, [float(n1[k] + n2[k]), float(ent[k])], hid[k]]).astype(np.float32)
            xn = ((xf - self.B['mu']) / self.B['sd']).astype(np.float32)
            with torch.no_grad():
                dy = float(self.gl(torch.from_numpy(xn[None, :]))[0])
            res[self.codes[j]] = dict(sigma_hat=float(np.exp(yh0 + dy)),
                                      sigma_baseline=float(np.exp(yh0)), correction=dy)
        return res

    # ---------- 仓位 ----------
    def positions(self, sig, mom_ok, w_prev):
        out = {}
        for prod, cfg in self.B['cfg'].items():
            c = cfg['index']
            if c not in sig: continue
            raw = float(np.clip(self.B['sigma_star'] / sig[c]['sigma_hat'], cfg['floor'], cfg['cap']))
            gated = 1.0 + (raw - 1.0) * (1.0 if mom_ok.get(prod, False) else 0.0)
            wp = w_prev.get(prod, 1.0)
            w = gated if abs(gated - wp) >= cfg['delta'] else wp
            out[prod] = dict(target_raw=round(raw, 4), after_gate=round(gated, 4),
                             w_prev=round(wp, 4), w_final=round(w, 4),
                             traded=abs(w - wp) > 1e-9,
                             sigma_hat_bp=round(sig[c]['sigma_hat'] * 1e4, 2),
                             sigma_base_bp=round(sig[c]['sigma_baseline'] * 1e4, 2))
        return out

    def momentum_ok(self, date):
        """趋势门: T-1 收盘的 k 日动量 > 0"""
        days = sorted(os.path.basename(f)[:10] for f in os.listdir(IDX_SRC) if f.endswith('.parquet'))
        i = days.index(date)
        need = max(c['gate_days'] for c in self.B['cfg'].values()) + 5
        use = days[max(0, i - need):i]              # 严格到 T-1
        cl = {}
        for d in use:
            df = pd.read_parquet(f'{IDX_SRC}/{d}.parquet', columns=['datetime', 'code', 'close'])
            df = df[df.code.isin(self.codes)]
            cl[d] = df.groupby('code')['close'].last()
        C = pd.DataFrame(cl).T
        ok = {}
        for prod, cfg in self.B['cfg'].items():
            k = cfg['gate_days']; c = cfg['index']
            ok[prod] = bool(len(C) > k and np.isfinite(C[c].iloc[-1]) and C[c].iloc[-1] > C[c].iloc[-1 - k])
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date'); ap.add_argument('--bar', type=int, default=None)
    ap.add_argument('--replay', nargs=2, metavar=('FROM', 'TO'))
    ap.add_argument('--out', default=None); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    eng = Engine(threads=a.threads)
    days_all = sorted(os.path.basename(f)[:10] for f in os.listdir(IDX_SRC) if f.endswith('.parquet'))
    targets = ([d for d in days_all if a.replay[0] <= d <= a.replay[1]] if a.replay else [a.date])
    recs = []
    for date in targets:
        panel, days = eng.load_bars(date)
        mom = eng.momentum_ok(date)
        w_prev = {p: 1.0 for p in eng.prods}
        bars = [a.bar] if a.bar is not None else list(range(0, 24))    # 2h视界: bar 0..23
        for bar in bars:
            try:
                sig = eng.forecast(panel, days, date, bar)
            except Exception as e:
                continue
            pos = eng.positions(sig, mom, w_prev)
            for p, v in pos.items(): w_prev[p] = v['w_final']
            rec = dict(date=date, bar=bar, time=T48[bar], **{f'{p}_{k}': v for p, d_ in pos.items() for k, v in d_.items()})
            recs.append(rec)
            if not a.replay:
                print(json.dumps(dict(date=date, bar=bar, time=T48[bar], momentum_gate=mom, positions=pos),
                                 ensure_ascii=False, indent=2))
    if a.replay:
        df = pd.DataFrame(recs)
        out = a.out or f'{HERE}/replay_{a.replay[0]}_{a.replay[1]}.csv'
        df.to_csv(out, index=False)
        print(f'{len(df)} 个决策点 -> {out}')
        for p in eng.prods:
            c = f'{p}_w_final'
            if c in df: print(f'  {p}: 均仓{df[c].mean():.3f} 峰仓{df[c].max():.2f} 调仓{int(df[f"{p}_traded"].sum())}次/{len(df)}点')


if __name__ == '__main__':
    main()
