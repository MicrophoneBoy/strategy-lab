# Verdict: volatility_filter

Data: SPY, daily bars from yahoo, adjusted for splits and dividends, on the dates they all share: 1993-01-29 to 2025-11-13 (8256 bars, fingerprint `d5889bd1bcb8bc20`). Costs: 0 bps fee plus 2 bps slippage on whatever changes hands. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Cash earns 0. Bars from 2025-11-14 on are the locked holdout and were not read. For a portfolio, "Trades" counts the days on which something was traded.

Tests counted so far: **7 idea(s), 134 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | FAIL | excess CAGR -3.23% after costs, -4.03% after tax |
| 2. In-sample permutation | FAIL | p = 0.2154 (needs < 0.0014) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 1, gate 2.** Even with the best parameters in hindsight, it did not beat buy-and-hold on the development window after costs and tax. Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (1993-02-01 to 1997-01-23, best parameters {'window': 10, 'percentile': 90})

| Metric | volatility_filter | Buy-and-hold SPY |
|---|---:|---:|
| Total return | 73.89% | 94.29% |
| CAGR (return per year) | 14.85% | 18.08% |
| Sharpe ratio | 1.69 | 1.63 |
| Max drawdown | -5.71% | -8.54% |
| Profit factor | 1.42 | 1.31 |
| Trades | 19 | 1 |
| Exposure (share of days invested) | 63.36% | 100.00% |
| After-tax total return | 56.36% | 80.11% |
| After-tax CAGR | 11.83% | 15.86% |

## Gate 2 detail

- Real optimised log return: 0.5533
- Shuffled data, median optimised log return: 0.4717
- Shuffled data, 95th percentile: 0.6425
- 1076 of 5000 shuffles did at least as well as the real data
