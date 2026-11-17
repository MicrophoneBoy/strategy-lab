# Verdict: mean_reversion

Data: SPY, daily bars from yahoo, adjusted for splits and dividends, on the dates they all share: 1993-01-29 to 2025-11-13 (8256 bars, fingerprint `d5889bd1bcb8bc20`). Costs: 0 bps fee plus 2 bps slippage on whatever changes hands. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Cash earns 0. Bars from 2025-11-14 on are the locked holdout and were not read. For a portfolio, "Trades" counts the days on which something was traded.

Tests counted so far: **8 idea(s), 146 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | FAIL | excess CAGR -10.63% after costs, -10.10% after tax |
| 2. In-sample permutation | FAIL | p = 0.0166 (needs < 0.0013) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 1, gate 2.** Even with the best parameters in hindsight, it did not beat buy-and-hold on the development window after costs and tax. Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (1993-02-01 to 1997-01-23, best parameters {'lookback': 5, 'drop': 0.01})

| Metric | mean_reversion | Buy-and-hold SPY |
|---|---:|---:|
| Total return | 33.29% | 94.29% |
| CAGR (return per year) | 7.46% | 18.08% |
| Sharpe ratio | 1.36 | 1.63 |
| Max drawdown | -5.62% | -8.54% |
| Profit factor | 1.74 | 1.31 |
| Trades | 57 | 1 |
| Exposure (share of days invested) | 16.39% | 100.00% |
| After-tax total return | 25.09% | 80.11% |
| After-tax CAGR | 5.76% | 15.86% |

## Gate 2 detail

- Real optimised log return: 0.2873
- Shuffled data, median optimised log return: 0.1291
- Shuffled data, 95th percentile: 0.2519
- 82 of 5000 shuffles did at least as well as the real data
