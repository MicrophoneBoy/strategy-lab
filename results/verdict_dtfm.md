# Verdict: dtfm

Data: QQQ plus 9 sector ETFs (QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB), daily bars from yahoo, adjusted for splits and dividends, on the dates they all share: 1999-03-10 to 2025-11-13 (6714 bars, fingerprint `d5889bd1bcb8bc20`). Costs: 0 bps fee plus 2 bps slippage on whatever changes hands. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Cash earns 0. Bars from 2025-11-14 on are the locked holdout and were not read. For a portfolio, "Trades" counts the days on which something was traded.

Tests counted so far: **2 idea(s), 91 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | PASS | excess CAGR +5.78% after costs, +5.79% after tax |
| 2. In-sample permutation | FAIL | p = 0.8432 (needs < 0.0050) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 2.** Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (1999-03-11 to 2003-03-13, best parameters {'fast': 20, 'slow': 60, 'stop': 0.08, 'slots': 3, 'buy_weight': 0.1, 'max_weight': 0.15})

| Metric | dtfm | Buy-and-hold SPY |
|---|---:|---:|
| Total return | -12.73% | -31.80% |
| CAGR (return per year) | -3.35% | -9.13% |
| Sharpe ratio | -0.64 | -0.30 |
| Max drawdown | -16.59% | -47.52% |
| Profit factor | 0.87 | 0.93 |
| Trades | 117 | 1 |
| Exposure (share of days invested) | 19.31% | 100.00% |
| After-tax total return | -12.73% | -31.81% |
| After-tax CAGR | -3.35% | -9.14% |

## Gate 2 detail

- Real optimised log return: -0.1361
- Shuffled data, median optimised log return: -0.0565
- Shuffled data, 95th percentile: 0.0710
- 843 of 1000 shuffles did at least as well as the real data
