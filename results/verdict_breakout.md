# Verdict: breakout

Data: QQQ, daily bars from yahoo, adjusted for splits and dividends, on the dates they all share: 1999-03-10 to 2025-11-13 (6714 bars, fingerprint `d5889bd1bcb8bc20`). Costs: 0 bps fee plus 2 bps slippage on whatever changes hands. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Cash earns 0. Bars from 2025-11-14 on are the locked holdout and were not read. For a portfolio, "Trades" counts the days on which something was traded.

Tests counted so far: **9 idea(s), 154 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | PASS | excess CAGR +13.29% after costs, +12.10% after tax |
| 2. In-sample permutation | FAIL | p = 0.3687 (needs < 0.0011) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 2.** Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (1999-03-11 to 2003-03-13, best parameters {'entry': 100, 'exit': 20})

| Metric | breakout | Buy-and-hold SPY |
|---|---:|---:|
| Total return | 17.67% | -31.80% |
| CAGR (return per year) | 4.16% | -9.13% |
| Sharpe ratio | 0.32 | -0.30 |
| Max drawdown | -26.76% | -47.52% |
| Profit factor | 1.10 | 0.93 |
| Trades | 5 | 1 |
| Exposure (share of days invested) | 15.79% | 100.00% |
| After-tax total return | 12.36% | -31.81% |
| After-tax CAGR | 2.96% | -9.14% |

## Gate 2 detail

- Real optimised log return: 0.1627
- Shuffled data, median optimised log return: 0.0698
- Shuffled data, 95th percentile: 0.6158
- 1843 of 5000 shuffles did at least as well as the real data
