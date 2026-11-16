"""Strategy families.

A family is one idea plus the grid of parameters it may choose from. It turns
bars into a (n_params, n_bars) matrix of positions: 1 long, 0 flat, -1 short.
Row k uses grid[k]. Column t may only use bars 0..t (tests check this).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import Bars


@dataclass(frozen=True)
class Family:
    name: str
    rationale: str  # why there might be an edge, and who loses on the other side
    grid: tuple[dict, ...]
    position_matrix: Callable[[Bars], np.ndarray]

    def positions(self, bars: Bars, index: int) -> np.ndarray:
        return self.position_matrix(bars)[index]


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
