# Test the tester

The gates are only worth trusting if they fail what should fail and pass what should pass.

| Strategy | Data | Gate | p-value | Pass bar | Gate says | Expected | Check |
|---|---|---:|---:|---:|---|---|---|
| random signal | real SPY development window | 2 | 0.1489 | < 0.01 | fail | fail | OK |
| random signal | random walk | 4 | 0.9851 | < 0.01 | fail | fail | OK |
| SMA crossover | random walk | 2 | 0.6294 | < 0.01 | fail | fail | OK |
| SMA crossover | random walk | 4 | 0.5920 | < 0.01 | fail | fail | OK |
| SMA crossover | planted trend | 2 | 0.0010 | < 0.01 | pass | pass | OK |
| SMA crossover | planted trend | 4 | 0.0050 | < 0.01 | pass | pass | OK |

Planted edge: a drift of 0.3% a day that flips direction about every 80 days, against 1% daily noise.

- Planted edge vs buy-and-hold, after costs and tax: in-sample +31.99% a year, walk-forward +20.99% a year. OK
- Calibration: on 100 separate random walks, gate 2 raised a false alarm at the 5% level 8% of the time and at the 1% level 2% of the time (median p-value 0.54; about 5%, 1% and 0.50 expected). OK

## How small an edge can the permutation gate see?

Gate 2 on separate planted-trend series, at the 1% pass bar. A weak real edge can be missed, so a FAIL only means the result was not shown to be better than luck.

| Planted trend per day | Series | Detected |
|---:|---:|---:|
| 0.1% | 20 | 1 |
| 0.2% | 20 | 9 |
| 0.3% | 20 | 17 |

**Tester verdict: all checks behave as expected.**
