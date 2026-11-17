# Verdict: low_vol_sectors

Data: 9 sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB), daily bars from yahoo, adjusted for splits and dividends, on the dates they all share: 1998-12-22 to 2025-11-13 (6766 bars, fingerprint `d5889bd1bcb8bc20`). Costs: 0 bps fee plus 2 bps slippage on whatever changes hands. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Cash earns 0. Bars from 2025-11-14 on are the locked holdout and were not read. For a portfolio, "Trades" counts the days on which something was traded.

Tests counted so far: **5 idea(s), 113 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | PASS | excess CAGR +1.75% after costs, +1.58% after tax |
| 2. In-sample permutation | FAIL | p = 0.9858 (needs < 0.0020) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 2.** Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (1998-12-23 to 2002-12-26, best parameters {'window': 120, 'hold': 4})

| Metric | low_vol_sectors | Buy-and-hold SPY |
|---|---:|---:|
| Total return | -16.21% | -22.18% |
| CAGR (return per year) | -4.33% | -6.08% |
| Sharpe ratio | -0.17 | -0.16 |
| Max drawdown | -35.50% | -47.52% |
| Profit factor | 0.95 | 0.96 |
| Trades | 42 | 1 |
| Exposure (share of days invested) | 87.09% | 100.00% |
| After-tax total return | -16.82% | -22.20% |
| After-tax CAGR | -4.51% | -6.09% |

## Gate 2 detail

- Real optimised log return: -0.1769
- Shuffled data, median optimised log return: 0.1241
- Shuffled data, 95th percentile: 0.3810
- 4929 of 5000 shuffles did at least as well as the real data
