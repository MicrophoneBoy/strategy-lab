# Verdict: turn_of_month

Data: SPY, daily bars from yahoo, adjusted for splits and dividends, on the dates they all share: 1993-01-29 to 2025-11-13 (8256 bars, fingerprint `d5889bd1bcb8bc20`). Costs: 0 bps fee plus 2 bps slippage on whatever changes hands. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Cash earns 0. Bars from 2025-11-14 on are the locked holdout and were not read. For a portfolio, "Trades" counts the days on which something was traded.

Tests counted so far: **6 idea(s), 125 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | FAIL | excess CAGR -12.22% after costs, -11.25% after tax |
| 2. In-sample permutation | FAIL | p = 0.4659 (needs < 0.0017) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 1, gate 2.** Even with the best parameters in hindsight, it did not beat buy-and-hold on the development window after costs and tax. Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (1993-02-01 to 1997-01-23, best parameters {'last': 3, 'first': 3})

| Metric | turn_of_month | Buy-and-hold SPY |
|---|---:|---:|
| Total return | 25.54% | 94.29% |
| CAGR (return per year) | 5.86% | 18.08% |
| Sharpe ratio | 1.03 | 1.63 |
| Max drawdown | -7.33% | -8.54% |
| Profit factor | 1.38 | 1.31 |
| Trades | 48 | 1 |
| Exposure (share of days invested) | 27.81% | 100.00% |
| After-tax total return | 19.75% | 80.11% |
| After-tax CAGR | 4.61% | 15.86% |

## Gate 2 detail

- Real optimised log return: 0.2275
- Shuffled data, median optimised log return: 0.2189
- Shuffled data, 95th percentile: 0.3759
- 2329 of 5000 shuffles did at least as well as the real data
