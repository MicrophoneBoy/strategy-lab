# Strategy Lab

A small backtesting lab that takes a trading idea, tests it properly, and gives an honest
verdict: does it have a real edge, or did it only look good by luck?

**The bar is buy-and-hold SPY, after costs and taxes.** "It made money" is not enough, because
SPY makes money most years. If nothing clears the bar, the answer is "buy the index fund", and
that counts as a correct result.

Status: work in progress. The engine, the 4 gates and the tester checks are built. One idea has
been tested so far (the SMA crossover baseline), and it died. More ideas come next.

## How it works

**One engine rule.** A position decided at the close of bar `t` earns the return of bar `t+1`.
A signal can never profit from the bar it was computed on. Costs are charged on every position
change. Every number in `results/` comes out of this engine.

Multi-asset ideas use the same rule with portfolio weights in place of a single position. The
weights are long only and never add up to more than 100% of the account, so there is no
leverage. A test checks that on one asset the two engines give identical results. When prices
are shuffled for a multi-asset idea, every asset gets the same shuffle, so each day's moves stay
together.

**Four gates, in order.** An idea dies at the first gate it fails.

1. **In-sample.** On the first 4 years of data, does the best parameter set beat buy-and-hold
   SPY after costs and after an estimated tax?
2. **In-sample permutation test.** Shuffle the price history 1,000 times (same bars, random
   order, so any real pattern is destroyed), re-optimise on each copy, and count how often the
   shuffled data does as well as the real data. Pass if that happens less than 1% of the time.
3. **Walk-forward.** Optimise on the past 4 years, trade the next month, roll forward. Every
   traded day is out-of-sample. Does it still beat buy-and-hold SPY?
4. **Walk-forward permutation test.** Shuffle everything after the first training window 200
   times and rerun the whole walk-forward. Pass if p is below 1%.

**Guard rails.**

- **Locked holdout.** The most recent 12 months of data were locked away before any research
  (`holdout.lock.json`). They are stored in a separate folder that research code never opens.
  A strategy that passes all 4 gates gets exactly one run on them.
- **Walk-forward data is single use.** Gates 3 and 4 only run for an idea that passed gates 1
  and 2, so out-of-sample results are never used to pick between ideas.
- **Every test is counted.** `ledger/tests.json` counts every idea and parameter set tried, and
  the permutation pass bars are divided by the number of ideas tested. Dead ideas stay in
  `ledger/verdicts.json` with the reason they died.
- **The tester is tested.** `uv run lab tester` checks that the gates fail a random signal, fail
  a real strategy on a random walk, and pass a strategy on data with a planted edge.

## Results so far

Full output is in [`results/`](results). Headline numbers:

| What | Result |
|---|---|
| Baseline, SMA 50/200 on SPY, 1993 to 1997 | trails buy-and-hold SPY by 3.81% a year after costs, 3.41% after tax |
| Best of 90 SMA parameter sets, same window | still trails SPY by 1.94% a year; 0% of the grid beat SPY |
| SMA crossover, gate 2 | p = 0.46: no better than luck. **Dead at gates 1 and 2.** |
| Tester: random signals and no-edge data | all fail, as they should |
| Tester: planted edge | passes gates 2 and 4, as it should |
| Tester: the same checks on multi-asset portfolios | random portfolios fail, planted edge passes |

Ideas tested: 1. Parameter sets tested: 90. Survivors: 0.

See [`results/baseline.md`](results/baseline.md),
[`results/verdict_sma_crossover.md`](results/verdict_sma_crossover.md) and
[`results/tester.md`](results/tester.md).

## Run it

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run lab fetch      # download daily bars, cache them, lock the holdout
uv run pytest         # 80 tests
uv run lab baseline   # baseline strategy and parameter sweep vs buy-and-hold SPY
uv run lab tester     # test the tester (about 5 minutes)
uv run lab gates      # run the baseline idea through the gates
```

Results are reproducible: the same cached data gives byte-identical output on rerun. Each result
prints a fingerprint of the data it used.

Optional: to pull data from Alpaca or read a paper account, copy `.env.example` to `.env` and
fill in **paper** keys. The code refuses keys that are not paper keys. Nothing here places
orders.

## Layout

```
config.toml           every assumption: symbols, costs, tax rates, window sizes, seeds
holdout.lock.json     the date the final holdout starts
src/strategy_lab/
  data.py             download, cache, and the locked holdout
  engine.py           positions times next-bar log returns, minus costs
  portfolio.py        the same rule for several assets: weights, drift, weekly rebalancing,
                      trailing stops, and a tax estimate with first-in-first-out lots
  metrics.py          total return, CAGR, Sharpe, max drawdown, profit factor, trades, exposure
  tax.py              the after-tax estimate
  strategies.py       strategy families (an idea plus its parameter grid)
  sweep.py            parameter sweep, heatmap, plateau-or-spike check
  permutation.py      bar shuffling and the in-sample permutation test
  walkforward.py      walk-forward and its permutation test
  gates.py            the 4 gates and the verdict
  tester.py           test the tester
  ledger.py           the count of tests and the verdict on every idea
tests/                hand-checked examples, lookahead tests, holdout tests
results/              output of the commands above
ledger/               tests.json, verdicts.json
```

## Assumptions and limits

- Daily bars from Yahoo Finance, adjusted for splits and dividends. Yahoo revises adjusted
  prices over time, so a fresh download can change results slightly. The fingerprint shows it.
  Alpaca's bars (which start in 2016) were compared with Yahoo's for SPY and XLK: the daily
  returns match closely except on 7 days for SPY, 6 of them in March 2020.
- Trades happen at the close that produced the signal. Slippage (2 basis points per change by
  default) stands in for the spread and for not getting exactly the closing price.
- Cash earns nothing while a strategy is flat. Shorts are modelled as minus the log return, with
  no borrowing cost.
- The tax estimate uses flat federal rates (22% short-term, 15% long-term), nets gains and
  losses each year and carries losses forward. It leaves out state tax, tax on dividends, and
  the wash-sale rule. It is a rough estimate, not tax advice.
- A permutation test can miss a weak real edge. `results/tester.md` shows how often it detects
  planted edges of different sizes. A failed gate means the result was not shown to be better
  than luck.
- Past results say nothing certain about the future. This is a research project, not advice.

## License

MIT
