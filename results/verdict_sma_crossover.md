# Verdict: sma_crossover

Data: SPY daily bars from yahoo, adjusted for splits and dividends, 1993-01-29 to 2025-11-13 (8256 bars, fingerprint `d5889bd1bcb8bc20`). Costs: 0 bps fee plus 2 bps slippage per position change. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Bars from 2025-11-14 on are the locked holdout and were not read.

Tests counted so far: **1 idea(s), 90 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | FAIL | excess CAGR -1.94% after costs, -2.19% after tax |
| 2. In-sample permutation | FAIL | p = 0.4635 (needs < 0.0100) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 1, gate 2.** Even with the best parameters in hindsight, it did not beat buy-and-hold on the development window after costs and tax. Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (1993-02-01 to 1997-01-23, best parameters {'fast': 35, 'slow': 60})

| Metric | sma_crossover | Buy-and-hold SPY |
|---|---:|---:|
| Total return | 81.82% | 94.29% |
| CAGR (return per year) | 16.14% | 18.08% |
| Sharpe ratio | 1.72 | 1.63 |
| Max drawdown | -7.63% | -8.54% |
| Profit factor | 1.39 | 1.31 |
| Trades | 8 | 1 |
| Exposure (share of days invested) | 75.77% | 100.00% |
| After-tax total return | 66.90% | 80.11% |
| After-tax CAGR | 13.68% | 15.86% |

## Gate 2 detail

- Real optimised log return: 0.5979
- Shuffled data, median optimised log return: 0.5916
- Shuffled data, 95th percentile: 0.7150
- 463 of 1000 shuffles did at least as well as the real data
