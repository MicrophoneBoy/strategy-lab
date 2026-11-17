# Verdict: sector_rotation

Data: 9 sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB), daily bars from yahoo, adjusted for splits and dividends, on the dates they all share: 1998-12-22 to 2025-11-13 (6766 bars, fingerprint `d5889bd1bcb8bc20`). Costs: 0 bps fee plus 2 bps slippage on whatever changes hands. Tax estimate: 22% on gains held a year or less, 15% on longer holds. Cash earns 0. Bars from 2025-11-14 on are the locked holdout and were not read. For a portfolio, "Trades" counts the days on which something was traded.

Tests counted so far: **3 idea(s), 100 parameter sets**. Permutation pass bars are divided by the number of ideas.

| Gate | Result | Key number |
|---|---|---|
| 1. In-sample vs SPY | PASS | excess CAGR +3.18% after costs, +2.70% after tax |
| 2. In-sample permutation | FAIL | p = 0.8616 (needs < 0.0033) |
| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |
| 4. Walk-forward permutation | not run | |

**Verdict: dead at gate 2.** Its in-sample result is no better than what the same optimisation finds on shuffled prices, so it looks like luck.

## Gate 1 detail (1998-12-23 to 2002-12-26, best parameters {'lookback': 126, 'hold': 3})

| Metric | sector_rotation | Buy-and-hold SPY |
|---|---:|---:|
| Total return | -11.09% | -22.18% |
| CAGR (return per year) | -2.90% | -6.08% |
| Sharpe ratio | -0.05 | -0.16 |
| Max drawdown | -34.50% | -47.52% |
| Profit factor | 0.97 | 0.96 |
| Trades | 42 | 1 |
| Exposure (share of days invested) | 87.09% | 100.00% |
| After-tax total return | -12.88% | -22.20% |
| After-tax CAGR | -3.39% | -6.09% |

## Gate 2 detail

- Real optimised log return: -0.1176
- Shuffled data, median optimised log return: 0.0835
- Shuffled data, 95th percentile: 0.5279
- 4308 of 5000 shuffles did at least as well as the real data
