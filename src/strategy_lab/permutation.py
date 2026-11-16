"""Permutation tests: could this result have come from luck alone?

Shuffling the order of the bars keeps every bar's size (so the mean, the
volatility and the total buy-and-hold return are unchanged) but destroys any
real pattern in time. If a strategy, re-optimised on the shuffled data, does as
well as it did on the real data, its result is what luck looks like.

Overnight gaps and intraday moves are shuffled separately, following Masters,
"Permutation and Randomization Tests for Trading System Development".
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import Costs


def permute_bars(bars, rng: np.random.Generator, start: int = 0):
    """Shuffle the bars after index `start`. Bars 0..start are left untouched.

    The last close is preserved exactly, so buy-and-hold earns the same.

    Works on single-asset Bars and on a multi-asset Panel. For a panel the same
    shuffle is applied to every asset, so each day's moves stay together and
    the relationships between assets survive.
    """
    n = len(bars)
    count = n - start - 1
    if count < 2:
        return bars
    lo, lh, ll, lc = (np.log(a) for a in (bars.open, bars.high, bars.low, bars.close))
    gap = lo[start + 1 :] - lc[start:-1]  # previous close to this open
    high = lh[start + 1 :] - lo[start + 1 :]  # open to high
    low = ll[start + 1 :] - lo[start + 1 :]  # open to low
    close = lc[start + 1 :] - lo[start + 1 :]  # open to close

    intraday_order = rng.permutation(count)
    gap = gap[rng.permutation(count)]
    high, low, close = high[intraday_order], low[intraday_order], close[intraday_order]

    new_close = lc[start] + np.cumsum(gap + close, axis=0)
    new_open = np.concatenate([lc[start][None], new_close[:-1]]) + gap

    def rebuilt(original: np.ndarray, tail: np.ndarray) -> np.ndarray:
        return np.concatenate([original[: start + 1], np.exp(tail)])

    return replace(
        bars,
        open=rebuilt(bars.open, new_open),
        high=rebuilt(bars.high, new_open + high),
        low=rebuilt(bars.low, new_open + low),
        close=rebuilt(bars.close, new_close),
    )


def best_in_sample(family, bars, costs: Costs) -> tuple[float, int]:
    """Optimise: total log return after costs of the best parameter set, and its index."""
    net = family.net(family.decisions(bars), bars, costs)
    totals = net.sum(axis=1)
    k = int(np.argmax(totals))
    return float(totals[k]), k


def p_value(real: float, permuted: np.ndarray) -> float:
    """Share of runs (the real one included) that scored at least as well as the real one."""
    return float((1 + np.sum(permuted >= real)) / (1 + len(permuted)))


def insample_permutation_test(family, bars, costs: Costs, n_perm: int, seed: int) -> dict:
    """Gate 2. Optimise on the real window, then on n_perm shuffled copies of it."""
    real, k = best_in_sample(family, bars, costs)
    rng = np.random.default_rng(seed)
    permuted = np.array(
        [best_in_sample(family, permute_bars(bars, rng), costs)[0] for _ in range(n_perm)]
    )
    return {
        "real_score": real,
        "best_params": family.grid[k],
        "permutations": n_perm,
        "permuted_median": float(np.median(permuted)),
        "permuted_95th": float(np.quantile(permuted, 0.95)),
        "permuted_at_least_as_good": int(np.sum(permuted >= real)),
        "p_value": p_value(real, permuted),
    }
