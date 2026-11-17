# Every idea tested so far

The bar: beat buy-and-hold SPY after costs and tax, and show the result is better than luck. An idea is dead at the first gate it fails.

| # | Idea | Tested on | Versions tried | Gate 1: beat SPY in-sample | Gate 2: better than luck | Gate 3: beat SPY out of sample | Gate 4: better than luck | Verdict | What happened |
|---:|---|---|---:|---|---|---|---|---|---|
| 1 | Moving-average crossover | SPY | 90 | FAIL | FAIL | not run | not run | Dead at gate 1, gate 2 | In its first 4 years (1993 to 1997) the best version trailed SPY by 1.94% a year. 463 of 1,000 shuffled price histories did at least as well (p = 0.4635, needed below 0.0100), so even that result looks like luck. |
| 2 | DTFM (dual-timeframe momentum) | QQQ plus 9 sector ETFs | 1 | PASS | FAIL | not run | not run | Dead at gate 2 | In its first 4 years (1999 to 2003) the best version beat SPY by 5.78% a year, but 843 of 1,000 shuffled price histories did at least as well (p = 0.8432, needed below 0.0050), so that looks like luck. |
| 3 | Sector momentum rotation | 9 sector ETFs | 9 | PASS | FAIL | not run | not run | Dead at gate 2 | In its first 4 years (1998 to 2002) the best version beat SPY by 3.18% a year, but 4,308 of 5,000 shuffled price histories did at least as well (p = 0.8616, needed below 0.0033), so that looks like luck. |
| 4 | Stocks or bonds by momentum | SPY and IEF | 4 | PASS | FAIL | not run | not run | Dead at gate 2 | In its first 4 years (2002 to 2006) the best version beat SPY by 0.82% a year, but 1,392 of 5,000 shuffled price histories did at least as well (p = 0.2785, needed below 0.0025), so that looks like luck. |
| 5 | Low-volatility sectors | 9 sector ETFs | 9 | PASS | FAIL | not run | not run | Dead at gate 2 | In its first 4 years (1998 to 2002) the best version beat SPY by 1.75% a year, but 4,929 of 5,000 shuffled price histories did at least as well (p = 0.9858, needed below 0.0020), so that looks like luck. |
| 6 | Turn of the month | SPY | 12 | FAIL | FAIL | not run | not run | Dead at gate 1, gate 2 | In its first 4 years (1993 to 1997) the best version trailed SPY by 12.22% a year. 2,329 of 5,000 shuffled price histories did at least as well (p = 0.4659, needed below 0.0017), so even that result looks like luck. |
| 7 | Volatility filter | SPY | 9 | FAIL | FAIL | not run | not run | Dead at gate 1, gate 2 | In its first 4 years (1993 to 1997) the best version trailed SPY by 3.23% a year. 1,076 of 5,000 shuffled price histories did at least as well (p = 0.2154, needed below 0.0014), so even that result looks like luck. |
| 8 | Short-term mean reversion | SPY | 12 | FAIL | FAIL | not run | not run | Dead at gate 1, gate 2 | In its first 4 years (1993 to 1997) the best version trailed SPY by 10.63% a year. 82 of 5,000 shuffled price histories did at least as well (p = 0.0166, needed below 0.0013), so even that result looks like luck. |
| 9 | Breakout | QQQ | 8 | PASS | FAIL | not run | not run | Dead at gate 2 | In its first 4 years (1999 to 2003) the best version beat SPY by 13.29% a year, but 1,843 of 5,000 shuffled price histories did at least as well (p = 0.3687, needed below 0.0011), so that looks like luck. |

**9 ideas tested, 154 versions in total, 0 survivor(s).**
