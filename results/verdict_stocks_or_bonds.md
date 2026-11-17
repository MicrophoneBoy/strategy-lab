# Verdict: stocks_or_bonds

Data: SPY and IEF, daily bars from yahoo, adjusted for splits and dividends, on the dates they all share: 2002-07-30 to 2025-11-13 (5863 bars, fingerprint `d5889bd1bcb8bc20`, IEF `4ae5dda4a4733c64`). Costs: 0 bps fee plus 2 bps slippage on whatever changes hands. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Cash earns 0. Bars from 2025-11-14 on are the locked holdout and were not read. For a portfolio, "Trades" counts the days on which something was traded.

Tests counted so far: **4 idea(s), 104 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | PASS | excess CAGR +0.82% after costs, +0.41% after tax |
| 2. In-sample permutation | FAIL | p = 0.2785 (needs < 0.0025) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 2.** Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (2002-07-31 to 2006-07-28, best parameters {'lookback': 126})

| Metric | stocks_or_bonds | Buy-and-hold SPY |
|---|---:|---:|
| Total return | 55.45% | 50.96% |
| CAGR (return per year) | 11.67% | 10.86% |
| Sharpe ratio | 1.11 | 0.75 |
| Max drawdown | -8.87% | -18.86% |
| Profit factor | 1.21 | 1.12 |
| Trades | 6 | 1 |
| Exposure (share of days invested) | 87.29% | 100.00% |
| After-tax total return | 45.45% | 43.29% |
| After-tax CAGR | 9.83% | 9.42% |

## Gate 2 detail

- Real optimised log return: 0.4411
- Shuffled data, median optimised log return: 0.3677
- Shuffled data, 95th percentile: 0.5975
- 1392 of 5000 shuffles did at least as well as the real data
