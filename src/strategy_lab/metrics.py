"""Performance metrics from per-bar log returns. Daily bars, 252 per year."""

from __future__ import annotations

import numpy as np

BARS_PER_YEAR = 252


def max_drawdown(net: np.ndarray) -> float:
    """Worst peak-to-trough fall of the equity curve, as a negative fraction."""
    equity = np.exp(np.concatenate([[0.0], np.cumsum(net)]))
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def count_trades(held: np.ndarray) -> int:
    """Number of times a non-flat position was opened (a flip opens a new one)."""
    previous = np.concatenate([[0.0], held[:-1]])
    return int(np.sum((held != previous) & (held != 0.0)))


def summarize(net: np.ndarray, held: np.ndarray) -> dict:
    """All headline metrics for one run.

    net   per-bar log returns after costs
    held  position held during each bar
    """
    n = len(net)
    years = n / BARS_PER_YEAR
    total_log = float(np.sum(net))
    simple = np.expm1(net)
    std = float(np.std(simple, ddof=1)) if n > 1 else 0.0
    gains = float(np.sum(net[net > 0]))
    losses = float(-np.sum(net[net < 0]))
    return {
        "bars": n,
        "years": years,
        "total_return": float(np.expm1(total_log)),
        "cagr": float(np.expm1(total_log / years)) if years > 0 else 0.0,
        # Annualised mean over standard deviation of daily returns, cash rate taken as 0.
        "sharpe": float(np.mean(simple) / std * np.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0,
        "max_drawdown": max_drawdown(net),
        # Sum of winning bars over sum of losing bars (log returns).
        "profit_factor": gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0,
        "trades": count_trades(held),
        "exposure": float(np.mean(held != 0.0)) if n else 0.0,
    }
