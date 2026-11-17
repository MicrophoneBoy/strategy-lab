"""Strategy families.

A family is one idea plus the grid of parameters it may choose from. It turns
bars into a (n_params, n_bars) matrix of positions: 1 long, 0 flat, -1 short.
Row k uses grid[k]. Column t may only use bars 0..t (tests check this).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import Costs, TaxRates
from .data import Bars, Panel
from .engine import Backtest, backtest, net_log_returns
from .portfolio import PortfolioFamily, hold_between_rebalances, last_trading_day_of_week


@dataclass(frozen=True)
class Family:
    name: str
    rationale: str  # why there might be an edge, and who loses on the other side
    grid: tuple[dict, ...]
    position_matrix: Callable[[Bars], np.ndarray]

    def positions(self, bars: Bars, index: int) -> np.ndarray:
        return self.position_matrix(bars)[index]

    # The interface the permutation tests, walk-forward and gates use.
    # PortfolioFamily has the same three methods.
    def decisions(self, data: Bars) -> np.ndarray:
        return self.position_matrix(data)

    def net(self, decisions: np.ndarray, data: Bars, costs: Costs) -> np.ndarray:
        return net_log_returns(decisions, data.close, costs)[0]

    def backtest(
        self, data: Bars, decision: np.ndarray, costs: Costs, tax: TaxRates | None = None
    ) -> Backtest:
        return backtest(data, decision, costs, tax)


def sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average. NaN until `window` values exist."""
    out = np.full(len(x), np.nan)
    if window <= len(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        out[window - 1 :] = (c[window:] - c[:-window]) / window
    return out


def sma_crossover(
    fast: tuple[int, ...] = tuple(range(5, 51, 5)),
    slow: tuple[int, ...] = tuple(range(20, 201, 20)),
    allow_short: bool = False,
) -> Family:
    """Long when the fast average is above the slow one, otherwise flat (or short)."""
    grid = tuple({"fast": f, "slow": s} for f in fast for s in slow if f < s)
    windows = sorted({w for g in grid for w in g.values()})
    out_of_market = -1.0 if allow_short else 0.0

    def matrix(bars: Bars) -> np.ndarray:
        averages = {w: sma(bars.close, w) for w in windows}
        rows = []
        for g in grid:
            f, s = averages[g["fast"]], averages[g["slow"]]
            pos = np.where(f > s, 1.0, out_of_market)
            pos[np.isnan(s)] = 0.0  # no signal until the slow average exists
            rows.append(pos)
        return np.array(rows)

    return Family(
        name="sma_crossover" + ("_long_short" if allow_short else ""),
        rationale=(
            "Trend following: prices that have been rising keep rising for a while because "
            "investors react slowly to news. The loser is whoever sells early in a trend."
        ),
        grid=grid,
        position_matrix=matrix,
    )


def random_signal(n_seeds: int = 50, mean_hold: int = 20, base_seed: int = 7) -> Family:
    """Coin-flip positions that ignore prices. Used only to test the tester:
    picking the best of many random signals looks good in-sample and must fail."""
    grid = tuple({"seed": k} for k in range(n_seeds))

    def matrix(bars: Bars) -> np.ndarray:
        rows = []
        for g in grid:
            rng = np.random.default_rng([base_seed, g["seed"]])
            first = rng.integers(0, 2)
            flips = rng.random(len(bars)) < 1.0 / mean_hold
            rows.append(((first + np.cumsum(flips)) % 2).astype(float))
        return np.array(rows)

    return Family(
        name="random_signal",
        rationale="None. A control that must fail.",
        grid=grid,
        position_matrix=matrix,
    )


# --------------------------------------------------------------------------
# Multi-asset families
# --------------------------------------------------------------------------


def trailing_return(close: np.ndarray, lookback: int) -> np.ndarray:
    """Return over the last `lookback` bars. NaN until that many bars exist."""
    out = np.full(close.shape, np.nan)
    if lookback < len(close):
        out[lookback:] = close[lookback:] / close[:-lookback] - 1.0
    return out


def capped_equal_weight(eligible: np.ndarray, max_weight: float) -> np.ndarray:
    """Equal weight across the eligible assets of each row, at most max_weight each.
    Whatever is left over stays in cash."""
    count = np.maximum(eligible.sum(axis=-1, keepdims=True), 1)
    return np.where(eligible, np.minimum(1.0 / count, max_weight), 0.0)


def dual_momentum(
    fast: tuple[int, ...] = (10, 20, 40),
    slow: tuple[int, ...] = (40, 60, 120),
    stop: tuple[float | None, ...] = (0.08, None),
    max_weight: float = 0.15,
) -> PortfolioFamily:
    """Hold every asset whose fast and slow trailing returns are both positive.

    Equal weight, capped per asset, reset on the last trading day of each week.
    With a trailing stop, an asset is sold when it closes that far below its
    highest close since purchase, and can come back at the next weekly reset.
    """
    grid = tuple(
        {"fast": f, "slow": s, "stop": st} for f in fast for s in slow if f < s for st in stop
    )
    lookbacks = sorted({g[key] for g in grid for key in ("fast", "slow")})
    # A stop of 100% can never trigger, so it stands in for "no stop".
    stops = np.array([1.0 if g["stop"] is None else g["stop"] for g in grid])

    def matrix(panel: Panel) -> np.ndarray:
        returns = {w: trailing_return(panel.close, w) for w in lookbacks}
        targets = np.array(
            [
                capped_equal_weight((returns[g["fast"]] > 0) & (returns[g["slow"]] > 0), max_weight)
                for g in grid
            ]
        )
        return hold_between_rebalances(
            targets, last_trading_day_of_week(panel.dates), panel.close, stops
        )

    return PortfolioFamily(
        name="dual_momentum",
        rationale=(
            "Momentum on two timeframes: assets rising over both the short and the longer "
            "window tend to keep rising, as investors are slow to price in good news. The "
            "loser is whoever sells a winner early."
        ),
        grid=grid,
        weight_matrix=matrix,
    )


def random_portfolio(n_seeds: int = 30, max_weight: float = 0.15, base_seed: int = 11) -> PortfolioFamily:
    """Each week, hold a random half of the assets. Ignores prices. Used only to
    test the tester: the best of many random portfolios must fail."""
    grid = tuple({"seed": k} for k in range(n_seeds))

    def matrix(panel: Panel) -> np.ndarray:
        targets = np.array(
            [
                capped_equal_weight(
                    np.random.default_rng([base_seed, g["seed"]]).random(panel.close.shape) < 0.5,
                    max_weight,
                )
                for g in grid
            ]
        )
        return hold_between_rebalances(targets, last_trading_day_of_week(panel.dates), panel.close)

    return PortfolioFamily(
        name="random_portfolio",
        rationale="None. A control that must fail.",
        grid=grid,
        weight_matrix=matrix,
    )


def dtfm_weights(
    panel: Panel,
    fast: int,
    slow: int,
    stop: float,
    slots: int,
    buy_weight: float,
    max_weight: float,
) -> np.ndarray:
    """Dual-Timeframe Momentum as it was run in the paper account.

    Everything happens at the weekly review (the week's last close):
      1. Sell a holding if its fast or slow trailing return is 0 or below, or
         if it closed `stop` or more below its highest close since purchase.
      2. Trim any holding that has drifted above `max_weight`.
      3. Fill free slots (at most `slots` positions) with the qualifying assets
         not already held, strongest slow return first, `buy_weight` of the
         account each. An asset sold at this review cannot be bought back
         until a later one.
    Between reviews nothing is traded and holdings are never topped up.
    """
    close = panel.close
    n, m = close.shape
    returns_fast, returns_slow = trailing_return(close, fast), trailing_return(close, slow)
    review = last_trading_day_of_week(panel.dates)
    step = np.ones_like(close)
    step[1:] = close[1:] / close[:-1]  # 1 + simple return of each bar

    out = np.zeros((n, m))
    held = np.zeros(m)
    peak = np.zeros(m)
    for t in range(n):
        held = held * step[t] / (1.0 + (held * (step[t] - 1.0)).sum())
        invested = held > 0.0
        peak = np.where(invested, np.maximum(peak, close[t]), 0.0)
        if review[t]:
            qualifies = (returns_fast[t] > 0.0) & (returns_slow[t] > 0.0)
            stopped = invested & (close[t] <= peak * (1.0 - stop))
            sold = invested & (~qualifies | stopped)
            held[sold] = 0.0
            peak[sold] = 0.0
            held = np.minimum(held, max_weight)
            free = slots - int(np.sum(held > 0.0))
            if free > 0:
                candidates = qualifies & ~(held > 0.0) & ~sold
                ranking = np.where(candidates, returns_slow[t], -np.inf)
                for j in np.argsort(-ranking, kind="stable")[:free]:
                    if not candidates[j]:
                        break
                    held[j] = min(buy_weight, 1.0 - held.sum())
                    peak[j] = close[t, j]
        out[t] = held
    return out


def dtfm(
    fast: int = 20,
    slow: int = 60,
    stop: float = 0.08,
    slots: int = 3,
    buy_weight: float = 0.10,
    max_weight: float = 0.15,
) -> PortfolioFamily:
    """DTFM with its rules fixed in advance: one parameter set, nothing to optimise."""
    params = {
        "fast": fast, "slow": slow, "stop": stop,
        "slots": slots, "buy_weight": buy_weight, "max_weight": max_weight,
    }
    return PortfolioFamily(
        name="dtfm",
        rationale=(
            "Dual-timeframe momentum: assets rising over both 20 and 60 sessions tend to keep "
            "rising, as investors are slow to price in good news, and a trailing stop cuts "
            "losers. The loser is whoever sells a winner early."
        ),
        grid=(params,),
        weight_matrix=lambda panel: dtfm_weights(panel, **params)[None],
    )
