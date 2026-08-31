# index-hidden-corrector

**Hidden states from a frozen time-series foundation model beat its scalar summaries
for intraday index volatility forecasting.**

A frozen, zero-shot K-line foundation model (Kronos-small) is used purely as a feature
extractor: its full 512-dimensional decoder hidden state feeds a 35K-parameter gated
linear corrector sitting on top of a strong HAR-type baseline. The same model's scalar
outputs (negative log-likelihood, predictive entropy) had previously been evaluated on
the same data and judged useless — changing only the *extraction interface* raises the
out-of-sample gain by an order of magnitude.

Method framework follows arXiv:2608.08825 (*Hybrid Neural-Classical Correction for
Frozen Time Series Foundation Models*), with ablations that revise its attribution.

## Headline results

All figures below are on a **pre-registered holdout year** (262 trading days) opened
exactly once after every modeling decision was frozen.

| | |
|---|---|
| Forecast gain over strong baseline | **ΔR² = +0.095 ± 0.004** (100 seeds, none negative) |
| Same model, scalar interface | +0.005 (≈20× weaker) |
| QLIKE improvement | −20.5% (Diebold–Mariano *t* = 7.6; Clark–West *t* = 16.9) |
| Model confidence set | eliminates every specification without hidden states (*p* ≤ 0.021) |
| Placebo (hidden states shuffled across days) | collapses to +0.008 |
| Direction (returns) | ≈ 0 — the representation carries second-moment information only |

**What the representation knows**

- **Subsumes cross-sectional breadth**: conditioning on the single-index hidden state
  drives a 500-constituent breadth signal's incremental value to +0.001; the converse
  does not hold.
- **Subsumes the option market's own expectation**: on 133 holdout days of five-minute
  index-option order-book data, the hidden-state increment retains a daily rank IC of
  +0.405 after controlling for implied volatility, while implied volatility's own
  increment turns negative (−0.086) after controlling for the hidden state.
- **Characteristic timescale ≈ 1 hour**: the increment traces an inverted-U across
  forecast horizons (+0.013 at 5 min → **+0.084 at 1 h** → +0.009 next day). This curve
  predicts which downstream applications work and which fail.
- **Pretraining buys sample efficiency, not asymptotic accuracy**: a scratch Transformer
  on the identical input matches the frozen pipeline at full sample size, but with half
  the data the frozen representation retains 77% of its gain versus 34%.

**Economic application** — a volatility-targeted overlay on CSI 500/1000 index futures
under realistic Chinese fees (intraday close-outs cost ~7× opens), with within-contract
returns, overnight gaps and roll costs: **+460~970 bp/year** versus an identical overlay
driven by the baseline forecast (six of six index-segment cells positive). Against
*textbook* trailing-volatility targeting the two are statistically indistinguishable —
the position map truncates 79% of observations at unit exposure, so most of the forecast
advantage never reaches the portfolio. This gap between forecast significance and
portfolio significance is documented rather than hidden.

## Layout

```
src/            pipeline: panel construction → hidden-state extraction → modeling → backtests
production/     self-contained inference engine (5-min target positions) + ops notes
docs/           lookahead audit (12 design checks + 6 empirical) · robustness summary
figures/        chart sets, versioned directories with an index
RESULTS.md      complete findings log, nine stages including all negative results
out/            data artifacts (git-ignored)
```

## Configuration

External data directories are read from environment variables (see `src/config.py`);
`PROJECT_ROOT` resolves from the file location, so nothing is hard-coded to a machine.

```bash
export IHC_INDEX_1M=/path/to/index_1min        # per-day parquet of 1-minute index bars
export IHC_FUTURES_1M=/path/to/futures_1min    # 1-minute index-futures bars
export IHC_DOMINANT_MAP=/path/to/dominant_map  # product -> dominant contract, per day
export IHC_OPTIONS=/path/to/options_tick       # index-option ticks with order book
export IHC_KRONOS_REPO=/path/to/Kronos         # local clone of the Kronos repository
```

Alternatively create an untracked `src/local_config.py` defining the same names.

## Reproduction

```bash
python src/build_index5m.py              # 5-minute panel, three broad-based indices
python src/extract_index_hidden.py       # 324k decision points × 512-d hidden states (~40 min, 30 cores)
python src/eval_index_ridge.py           # ridge baselines B1–B4
python src/run_paper1_stack.py           # corrector stack M0–M4  → out/paper1_preds.npz
python src/eval_holdout.py               # one-shot holdout opening
python src/build_futures5m.py            # IC/IM dominant-contract panel
python src/backtest_futures.py           # futures backtest under realistic costs
python src/futures_gate.py               # trend gate × horizon × no-trade-band grid
python src/seed_robustness.py            # 100-seed retraining
python src/qlike_dm.py                   # QLIKE / DM / placebo
python src/gp_eval.py                    # Gârleanu–Pedersen optimal trading calibration
python production/infer.py --date YYYY-MM-DD --bar 20
```

Data requirements are described in `src/config.py`. The market data itself is not
redistributed here.

## Three implementation notes

1. Window normalization uses the window's own mean/std, so **every decision point needs
   its own forward pass** — reusing a sliding window leaks future bars into the normalizer.
2. The model's second-token decoder is non-causal under `eval()`; the single-position
   sibling call in `extract_index_hidden.py` works around it.
3. Futures returns must be computed **within contract** (dominant map lagged one day,
   overnight from the same contract), otherwise roll jumps contaminate the series.

See `docs/lookahead-audit.md` for the full no-lookahead argument.
