# DTFM: the lab's verdict next to the real paper run

## The real paper run

- Universe: 7 mega-cap stocks: NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA
- 2026-07-17 to 2026-11-16: account went from $100,000.00 to $97,839.79, **-2.16%**
- Positions at the end: AAPL, MSFT, NVDA (confirmed through Alpaca's API)

## SPY over the same dates

- Price only, from the experiment's own baseline ($742.17 at the 2026-07-17 open) to the 2026-11-16 close ($739.95): **-0.30%**
- With dividends (Alpaca adjusted open $740.24 to adjusted close $739.95): **-0.04%**
- Alpaca's own raw open on 2026-07-17 was $742.08, against the logged baseline of $742.17

## Paper run minus SPY

- Against SPY price only: **-1.86%**
- Against SPY with dividends: **-2.12%**
- Did the paper run beat buy-and-hold SPY? **NO**

## The lab's verdict on the same rules

Run on QQQ plus 9 sector ETFs (a survivorship-clean stand-in for the mega-cap stocks), on data before the locked holdout.

- Gate 1, in-sample vs SPY: PASS
- Gate 2, in-sample permutation: FAIL (p = 0.8432, needs < 0.0050)
- Gate 3, walk-forward vs SPY: not run
- Gate 4, walk-forward permutation: not run

**Lab verdict: dead at gate 2.**

The two agree: the lab found no edge it could trust, and the paper run trailed SPY.

## How much weight this comparison can carry

- The paper run is one stretch of about four months. That is far too short to prove or disprove anything by itself; the gates use decades of data.
- The universes differ: the paper run traded 7 mega-cap stocks, the lab run used ETFs.
- The paper run was not a clean run of the rules. Its log records wrong price data behind several decisions, including a first purchase that never met the entry rule.
- The account equity was read during the end date, and SPY is taken at that day's close.
- The two SPY bars sit inside the lab's locked holdout. They were read only as a benchmark for this comparison and are recorded in `ledger/holdout_reads.json`.
