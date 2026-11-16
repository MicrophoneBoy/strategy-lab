"""Walk-forward: optimise on the past 4 years, trade the next month, roll on.

Every traded bar uses parameters chosen only from earlier bars, so the stitched
result is out-of-sample from start to finish.
"""

from __future__ import annotations

import numpy as np

from .config import Costs, TaxRates
from .data import Bars
from .engine import backtest, buy_and_hold, compare, net_log_returns
from .permutation import p_value, permute_bars
from .strategies import Family


def walk_forward_positions(
    family: Family, bars: Bars, costs: Costs, train_bars: int, step_bars: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Stitch one position series from monthly re-optimisation.

    For a step whose first traded bar is s:
      - score every parameter set on return bars s-train_bars .. s-1
      - the winner sets the positions decided at closes s-1 .. s+step_bars-2,
        which earn the returns of bars s .. s+step_bars-1.
    Returns the positions and a list of (s, winning grid index).
    """
    n = len(bars)
    if n <= train_bars:
        raise ValueError("not enough bars for one training window")
    matrix = family.position_matrix(bars)
    net, _ = net_log_returns(matrix, bars.close, costs)
    cumulative = np.concatenate([np.zeros((len(matrix), 1)), np.cumsum(net, axis=1)], axis=1)
    position = np.zeros(n)
    choices = []
    for s in range(train_bars, n, step_bars):
        scores = cumulative[:, s] - cumulative[:, s - train_bars]
        k = int(np.argmax(scores))
        end = min(s + step_bars, n)
        position[s - 1 : end - 1] = matrix[k, s - 1 : end - 1]
        choices.append((s, k))
    return position, choices


def walk_forward_score(
    family: Family, bars: Bars, costs: Costs, train_bars: int, step_bars: int
) -> float:
    """Total out-of-sample log return after costs."""
    position, _ = walk_forward_positions(family, bars, costs, train_bars, step_bars)
    net, _ = net_log_returns(position, bars.close, costs)
    return float(net[train_bars:].sum())


def walk_forward(
    family: Family,
    bars: Bars,
    benchmark_bars: Bars,
    costs: Costs,
    tax: TaxRates,
    train_bars: int,
    step_bars: int,
) -> dict:
    """Gate 3. Out-of-sample results against buy-and-hold over the same bars."""
    if not np.array_equal(bars.dates, benchmark_bars.dates):
        raise ValueError("strategy and benchmark must cover the same bars")
    position, choices = walk_forward_positions(family, bars, costs, train_bars, step_bars)
    start = train_bars - 1  # the close at which the first out-of-sample position is taken
    strategy = backtest(bars.slice(start), position[start:], costs, tax)
    benchmark = buy_and_hold(benchmark_bars.slice(start), costs, tax)
    picks: dict[str, int] = {}
    for _, k in choices:
        key = ", ".join(f"{name}={value}" for name, value in family.grid[k].items())
        picks[key] = picks.get(key, 0) + 1
    return {
        "first_bar": str(strategy.dates[0]),
        "last_bar": str(strategy.dates[-1]),
        "steps": len(choices),
        "strategy": strategy.metrics,
        "benchmark": benchmark.metrics,
        "versus_benchmark": compare(strategy.metrics, benchmark.metrics),
        "most_chosen_params": dict(sorted(picks.items(), key=lambda kv: -kv[1])[:5]),
    }


def walk_forward_permutation_test(
    family: Family,
    bars: Bars,
    costs: Costs,
    train_bars: int,
    step_bars: int,
    n_perm: int,
    seed: int,
) -> dict:
    """Gate 4. Shuffle only the bars after the first training window, rerun the
    whole walk-forward on each shuffled copy, and compare."""
    real = walk_forward_score(family, bars, costs, train_bars, step_bars)
    rng = np.random.default_rng(seed)
    permuted = np.array(
        [
            walk_forward_score(
                family, permute_bars(bars, rng, start=train_bars - 1), costs, train_bars, step_bars
            )
            for _ in range(n_perm)
        ]
    )
    return {
        "real_score": real,
        "permutations": n_perm,
        "permuted_median": float(np.median(permuted)),
        "permuted_95th": float(np.quantile(permuted, 0.95)),
        "permuted_at_least_as_good": int(np.sum(permuted >= real)),
        "p_value": p_value(real, permuted),
    }
