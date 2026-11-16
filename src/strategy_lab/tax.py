"""A rough US tax estimate, so results can be compared after tax.

The model, stated plainly:

- A trade is one unbroken stretch in the same non-flat position.
- When a trade closes, its dollar gain or loss is realised. Held more than 365
  days: long-term rate. Otherwise: short-term rate (ordinary income).
- At each year end, gains and losses are netted. Net losses carry forward.
  Tax on net gains is paid out of the account.
- On the last bar any open position is sold (paying the exit cost) and taxed,
  so strategy and benchmark are both compared as cash in hand.

Left out: state tax, tax on dividends, the wash-sale rule, and the $3,000 a year
of losses that can offset ordinary income. It is an estimate, not tax advice.
"""

from __future__ import annotations

import math

import numpy as np

from .config import TaxRates
from .metrics import BARS_PER_YEAR


def settle_year(short: float, long: float, carry: float, rates: TaxRates) -> tuple[float, float]:
    """Net one year's realised gains. Returns (tax due, loss carried forward <= 0)."""
    short += carry
    if short < 0 < long:
        long += short
        short = 0.0
    elif long < 0 < short:
        short += long
        long = 0.0
    tax = rates.short_term * max(short, 0.0) + rates.long_term * max(long, 0.0)
    return tax, min(short, 0.0) + min(long, 0.0)


def after_tax(
    net: np.ndarray,
    held: np.ndarray,
    dates: np.ndarray,
    rates: TaxRates,
    exit_cost_rate: float = 0.0,
) -> dict:
    """Simulate the account bar by bar, paying estimated tax each year end.

    net, held and dates are per return bar, as produced by engine.backtest.
    """
    n = len(net)
    year_of = dates.astype("datetime64[Y]").astype(int)
    equity, carry, taxes = 1.0, 0.0, 0.0
    realised = {"short": 0.0, "long": 0.0}
    closed = {"short": 0, "long": 0}
    entry_equity, entry_date = 0.0, None
    previous = 0.0

    def close_trade(exit_date) -> None:
        bucket = "long" if (exit_date - entry_date).astype(int) > 365 else "short"
        realised[bucket] += equity - entry_equity
        closed[bucket] += 1

    for t in range(n):
        h = float(held[t])
        if h != previous:
            if previous != 0.0:  # the old trade ended at the close of bar t-1
                close_trade(dates[t - 1])
            if h != 0.0:  # a new trade starts earning on bar t
                entry_equity, entry_date = equity, dates[t]

        equity *= math.exp(net[t])

        last = t == n - 1
        if last and h != 0.0:  # sell everything on the final bar
            equity *= 1.0 - exit_cost_rate * abs(h)
            close_trade(dates[t])
        if last or year_of[t + 1] != year_of[t]:
            tax, carry = settle_year(realised["short"], realised["long"], carry, rates)
            equity -= tax
            taxes += tax
            realised["short"] = realised["long"] = 0.0
        previous = h

    years = n / BARS_PER_YEAR
    return {
        "after_tax_total_return": equity - 1.0,
        "after_tax_cagr": equity ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else -1.0,
        "tax_paid": taxes,  # as a fraction of starting capital
        "short_term_trades": closed["short"],
        "long_term_trades": closed["long"],
    }
