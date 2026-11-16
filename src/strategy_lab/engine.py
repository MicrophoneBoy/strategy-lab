"""The backtest engine. One rule, applied everywhere:

    position[t] is decided with data up to and including the close of bar t,
    and it earns the log return of bar t+1.

So a signal can never profit from the bar it was computed on. Costs are charged
on the bar where a new position starts earning.

All functions work on the last axis, so a (n_params, n_bars) matrix of
positions is backtested in one call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Costs, TaxRates
from .data import Bars
from .metrics import summarize
from .tax import after_tax


def log_returns(close: np.ndarray) -> np.ndarray:
    """r[t] = ln(close[t] / close[t-1]); r[0] = 0 because bar 0 has no previous close."""
    r = np.zeros(len(close), dtype=float)
    r[1:] = np.diff(np.log(close))
    return r


def held_positions(position: np.ndarray) -> np.ndarray:
    """The position actually held during each bar: yesterday's decision."""
    position = np.asarray(position, dtype=float)
    held = np.zeros_like(position)
    held[..., 1:] = position[..., :-1]
    return held


def net_log_returns(
    position: np.ndarray, close: np.ndarray, costs: Costs
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar strategy log returns after costs, and the held positions.

    net[t] = held[t] * r[t] + ln(1 - cost_rate * |held[t] - held[t-1]|)
    """
    held = held_positions(position)
    turnover = np.abs(np.diff(held, axis=-1, prepend=0.0))
    net = held * log_returns(close) + np.log1p(-costs.rate * turnover)
    return net, held


@dataclass(frozen=True)
class Backtest:
    dates: np.ndarray  # dates of the return bars (bar 0 is dropped: it has no return)
    held: np.ndarray
    net: np.ndarray  # per-bar log returns after costs
    metrics: dict


def backtest(
    bars: Bars, position: np.ndarray, costs: Costs, tax: TaxRates | None = None
) -> Backtest:
    """Run one position series and compute every metric."""
    position = np.asarray(position, dtype=float)
    if position.shape != (len(bars),):
        raise ValueError("position must have one value per bar")
    if not np.isin(position, (-1.0, 0.0, 1.0)).all():
        raise ValueError("positions must be 1 (long), 0 (flat) or -1 (short)")
    net, held = net_log_returns(position, bars.close, costs)
    dates, held, net = bars.dates[1:], held[1:], net[1:]
    metrics = summarize(net, held)
    if tax is not None:
        metrics.update(after_tax(net, held, dates, tax, costs.rate))
    return Backtest(dates=dates, held=held, net=net, metrics=metrics)


def buy_and_hold(bars: Bars, costs: Costs, tax: TaxRates | None = None) -> Backtest:
    """The benchmark: buy at the first close, never sell."""
    return backtest(bars, np.ones(len(bars)), costs, tax)


def compare(strategy: dict, benchmark: dict) -> dict:
    """Score a strategy against the benchmark over the same bars."""
    out = {
        "excess_total_return": strategy["total_return"] - benchmark["total_return"],
        "excess_cagr": strategy["cagr"] - benchmark["cagr"],
        "sharpe_difference": strategy["sharpe"] - benchmark["sharpe"],
        "beats_benchmark_after_costs": strategy["cagr"] > benchmark["cagr"],
    }
    if "after_tax_cagr" in strategy and "after_tax_cagr" in benchmark:
        out["excess_after_tax_cagr"] = strategy["after_tax_cagr"] - benchmark["after_tax_cagr"]
        out["beats_benchmark_after_costs_and_taxes"] = (
            strategy["after_tax_cagr"] > benchmark["after_tax_cagr"]
        )
    return out
