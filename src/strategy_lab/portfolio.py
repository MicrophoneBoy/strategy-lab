"""The engine for several assets at once. Same rule as the single-asset engine:

    weights[t] are decided with data up to and including the close of bar t,
    and they earn the returns of bar t+1.

weights[t, j] is the share of the account wanted in asset j after the close of
bar t. Weights are long only and sum to at most 1; the rest is cash earning 0.
Between trades the weights drift as prices move, and the cost of a trade is
charged on what actually changes hands: |wanted weight - drifted weight|.

On one asset with weights of 0 or 1 this gives exactly the single-asset engine
(a test checks it).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import Costs, TaxRates
from .data import Panel
from .engine import Backtest
from .metrics import BARS_PER_YEAR, summarize
from .tax import settle_year

TRADE_TOLERANCE = 1e-9  # weight changes smaller than this are rounding noise, not trades


def simple_returns(close: np.ndarray) -> np.ndarray:
    """s[t] = close[t] / close[t-1] - 1 along the time axis; s[0] = 0."""
    s = np.zeros_like(close, dtype=float)
    s[1:] = close[1:] / close[:-1] - 1.0
    return s


def _shift(x: np.ndarray, axis: int) -> np.ndarray:
    """Move values one step later along `axis`, filling the first with zero."""
    out = np.zeros_like(x)
    src = [slice(None)] * x.ndim
    dst = [slice(None)] * x.ndim
    src[axis], dst[axis] = slice(None, -1), slice(1, None)
    out[tuple(dst)] = x[tuple(src)]
    return out


def portfolio_net_log_returns(
    weights: np.ndarray, close: np.ndarray, costs: Costs
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar portfolio log returns after costs, and the turnover at each close.

    weights  (..., n_bars, n_assets)
    close    (n_bars, n_assets)

    For bar t:
      held      = weights[t-1]
      gross     = sum over assets of held * simple return of bar t
      drifted   = held * (1 + return) / (1 + gross)      weights just before trading
      turnover  = sum |weights[t] - drifted|             what changes hands at close t
      net       = ln(1 + gross) + ln(1 - cost_rate * turnover at close t-1)
    """
    weights = np.asarray(weights, dtype=float)
    s = simple_returns(close)
    held = _shift(weights, axis=-2)
    gross = (held * s).sum(axis=-1)
    drifted = held * (1.0 + s) / (1.0 + gross)[..., None]
    turnover = np.abs(weights - drifted).sum(axis=-1)
    net = np.log1p(gross) + np.log1p(-costs.rate * _shift(turnover, axis=-1))
    return net, turnover


def last_trading_day_of_week(dates: np.ndarray) -> np.ndarray:
    """True on each week's final trading day (Friday, or Thursday before a holiday).

    The trading calendar is known in advance, so using tomorrow's date is not
    lookahead. The very last bar counts only if it is a Friday.
    """
    day = dates.astype("datetime64[D]").astype(int) + 3  # 1970-01-01 was a Thursday
    week, weekday = day // 7, day % 7  # weekday 0 = Monday
    mask = np.zeros(len(dates), dtype=bool)
    mask[:-1] = week[1:] != week[:-1]
    mask[-1] = weekday[-1] == 4
    return mask


def hold_between_rebalances(
    targets: np.ndarray,
    rebalance: np.ndarray,
    close: np.ndarray,
    trailing_stop: float | np.ndarray | None = None,
) -> np.ndarray:
    """Turn 'what I want on rebalance days' into weights for every day.

    targets        (..., n_bars, n_assets), read only on rebalance days
    rebalance      (n_bars,) True where the portfolio is reset to the targets
    trailing_stop  optional. An asset is sold to cash at the close where it sits
                   this fraction below its highest close since it was bought.
                   It may be bought back at the next rebalance. One value, or
                   one per leading row of `targets`.

    On other days nothing is traded: weights just drift with prices. Day t uses
    closes up to t only.
    """
    targets = np.asarray(targets, dtype=float)
    s = simple_returns(close)
    out = np.zeros_like(targets)
    held = np.zeros(targets.shape[:-2] + targets.shape[-1:])
    peak = np.zeros_like(held)
    stop = None if trailing_stop is None else np.asarray(trailing_stop, dtype=float)[..., None]

    for t in range(targets.shape[-2]):
        growth = 1.0 + (held * s[t]).sum(axis=-1, keepdims=True)
        held = held * (1.0 + s[t]) / growth
        if stop is not None:
            invested = held > 0.0
            peak = np.where(invested, np.maximum(peak, close[t]), 0.0)
            stopped = invested & (close[t] <= peak * (1.0 - stop))
            held = np.where(stopped, 0.0, held)
        if rebalance[t]:
            wanted = targets[..., t, :]
            newly_bought = (wanted > 0.0) & (held <= 0.0)
            peak = np.where(newly_bought, close[t], np.where(wanted > 0.0, peak, 0.0))
            held = wanted.copy()
        out[..., t, :] = held
    return out


# --------------------------------------------------------------------------
# Tax estimate with first-in-first-out lots
# --------------------------------------------------------------------------


def after_tax_portfolio(
    weights: np.ndarray,
    close: np.ndarray,
    dates: np.ndarray,
    net: np.ndarray,
    turnover: np.ndarray,
    rates: TaxRates,
    cost_rate: float = 0.0,
) -> dict:
    """Simulate the account in dollars and pay estimated tax each year end.

    Every purchase is a lot with its own price and date. Sales use the oldest
    lots first (FIFO, the US default). A lot sold after more than 365 days is
    long-term. Year-end netting, loss carryforward and the final liquidation
    work as in tax.py. Costs and tax payments shrink every holding in
    proportion.

    All arrays are full length: index t is the close of bar t.
    """
    n, m = close.shape
    year_of = dates.astype("datetime64[Y]").astype(int)
    equity, carry, taxes = 1.0, 0.0, 0.0
    scale = 1.0  # real shares = stored units * scale, so "shrink everything" is one multiply
    lots: list[deque] = [deque() for _ in range(m)]  # each lot: [units, price, date]
    units = [0.0] * m  # total units held per asset
    realised = {"short": 0.0, "long": 0.0}
    sold = {"short": 0, "long": 0}

    def sell(j: int, shares: float, price: float, date) -> None:
        while shares > 0.0 and lots[j]:
            lot = lots[j][0]
            available = lot[0] * scale
            quantity = min(shares, available)
            bucket = "long" if (date - lot[2]).astype(int) > 365 else "short"
            realised[bucket] += quantity * (price - lot[1])
            sold[bucket] += 1
            shares -= quantity
            if quantity >= available * (1.0 - 1e-12):
                units[j] -= lot[0]
                lots[j].popleft()
            else:
                lot[0] -= quantity / scale
                units[j] -= quantity / scale
        if not lots[j]:
            units[j] = 0.0

    for t in range(n):
        if t > 0:
            scale *= 1.0 - cost_rate * turnover[t - 1]  # cost of yesterday's trade
            equity *= float(np.exp(net[t]))
        for j in range(m):
            price = close[t, j]
            change = weights[t, j] * equity - units[j] * scale * price
            if change > TRADE_TOLERANCE * equity:
                lots[j].append([change / price / scale, price, dates[t]])
                units[j] += change / price / scale
            elif change < -TRADE_TOLERANCE * equity:
                sell(j, -change / price, price, dates[t])

        last = t == n - 1
        if last:  # sell everything on the final bar
            equity *= 1.0 - cost_rate * float(weights[t].sum())
            for j in range(m):
                sell(j, float("inf"), close[t, j], dates[t])
        if last or year_of[t + 1] != year_of[t]:
            tax, carry = settle_year(realised["short"], realised["long"], carry, rates)
            if tax > 0.0:
                scale *= (equity - tax) / equity
                equity -= tax
                taxes += tax
            realised["short"] = realised["long"] = 0.0

    years = (n - 1) / BARS_PER_YEAR
    return {
        "after_tax_total_return": equity - 1.0,
        "after_tax_cagr": equity ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else -1.0,
        "tax_paid": taxes,
        "short_term_trades": sold["short"],  # lots sold within a year
        "long_term_trades": sold["long"],
    }


# --------------------------------------------------------------------------
# Backtest and strategy families
# --------------------------------------------------------------------------


def portfolio_backtest(
    panel: Panel, weights: np.ndarray, costs: Costs, tax: TaxRates | None = None
) -> Backtest:
    """Run one weight path and compute every metric."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != panel.close.shape:
        raise ValueError("weights must have one row per bar and one column per asset")
    if not np.isfinite(weights).all() or (weights < -1e-12).any():
        raise ValueError("weights must be finite and long only")
    if (weights.sum(axis=-1) > 1.0 + 1e-9).any():
        raise ValueError("weights may not sum to more than 1 (no leverage)")
    net, turnover = portfolio_net_log_returns(weights, panel.close, costs)
    invested = _shift(weights.sum(axis=-1), axis=-1)  # share of the account invested during each bar
    metrics = summarize(net[1:], invested[1:])
    metrics["exposure"] = float(invested[1:].mean())
    metrics["trades"] = int(np.sum(turnover[:-1] > 1e-6))  # days on which something was traded
    metrics["turnover_per_year"] = float(turnover[:-1].sum() / metrics["years"])
    if tax is not None:
        metrics.update(
            after_tax_portfolio(weights, panel.close, panel.dates, net, turnover, tax, costs.rate)
        )
    return Backtest(dates=panel.dates[1:], held=invested[1:], net=net[1:], metrics=metrics)


@dataclass(frozen=True)
class PortfolioFamily:
    """One multi-asset idea plus its parameter grid.

    weight_matrix turns a panel into (n_params, n_bars, n_assets) weights.
    Row t may only use bars 0..t (tests check this).
    """

    name: str
    rationale: str
    grid: tuple[dict, ...]
    weight_matrix: Callable[[Panel], np.ndarray]

    # The interface the permutation tests, walk-forward and gates use.
    def decisions(self, data: Panel) -> np.ndarray:
        return self.weight_matrix(data)

    def net(self, decisions: np.ndarray, data: Panel, costs: Costs) -> np.ndarray:
        return portfolio_net_log_returns(decisions, data.close, costs)[0]

    def backtest(
        self, data: Panel, decision: np.ndarray, costs: Costs, tax: TaxRates | None = None
    ) -> Backtest:
        return portfolio_backtest(data, decision, costs, tax)
