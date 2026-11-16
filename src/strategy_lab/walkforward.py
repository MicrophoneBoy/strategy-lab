"""Walk-forward: optimise on the past 4 years, trade the next month, roll on.

Every traded bar uses parameters chosen only from earlier bars, so the stitched
result is out-of-sample from start to finish.

Works for single-asset families (decisions are positions) and multi-asset
families (decisions are weights). `data` is Bars or a Panel to match.
"""

from __future__ import annotations

import numpy as np

from .config import Costs, TaxRates
from .data import Bars
from .engine import buy_and_hold, compare
from .permutation import p_value, permute_bars


def walk_forward_positions(
    family, data, costs: Costs, train_bars: int, step_bars: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Stitch one decision series from monthly re-optimisation.

    For a step whose first traded bar is s:
      - score every parameter set on return bars s-train_bars .. s-1
      - the winner sets the decisions made at closes s-1 .. s+step_bars-2,
        which earn the returns of bars s .. s+step_bars-1.
    Returns the stitched decisions and a list of (s, winning grid index).
    """
    n = len(data)
    if n <= train_bars:
        raise ValueError("not enough bars for one training window")
    decisions = family.decisions(data)
    net = family.net(decisions, data, costs)
    cumulative = np.concatenate([np.zeros((len(net), 1)), np.cumsum(net, axis=1)], axis=1)
    stitched = np.zeros(decisions.shape[1:])
    choices = []
    for s in range(train_bars, n, step_bars):
        scores = cumulative[:, s] - cumulative[:, s - train_bars]
        k = int(np.argmax(scores))
        end = min(s + step_bars, n)
        stitched[s - 1 : end - 1] = decisions[k, s - 1 : end - 1]
        choices.append((s, k))
    return stitched, choices


def walk_forward_score(family, data, costs: Costs, train_bars: int, step_bars: int) -> float:
    """Total out-of-sample log return after costs."""
    stitched, _ = walk_forward_positions(family, data, costs, train_bars, step_bars)
    return float(family.net(stitched, data, costs)[train_bars:].sum())


def walk_forward(
    family,
    data,
    benchmark_bars: Bars,
    costs: Costs,
    tax: TaxRates,
    train_bars: int,
    step_bars: int,
) -> dict:
    """Gate 3. Out-of-sample results against buy-and-hold over the same bars."""
    if not np.array_equal(data.dates, benchmark_bars.dates):
        raise ValueError("strategy and benchmark must cover the same bars")
    stitched, choices = walk_forward_positions(family, data, costs, train_bars, step_bars)
    start = train_bars - 1  # the close at which the first out-of-sample decision is made
    strategy = family.backtest(data.slice(start), stitched[start:], costs, tax)
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
    family,
    data,
    costs: Costs,
    train_bars: int,
    step_bars: int,
    n_perm: int,
    seed: int,
) -> dict:
    """Gate 4. Shuffle only the bars after the first training window, rerun the
    whole walk-forward on each shuffled copy, and compare."""
    real = walk_forward_score(family, data, costs, train_bars, step_bars)
    rng = np.random.default_rng(seed)
    permuted = np.array(
        [
            walk_forward_score(
                family, permute_bars(data, rng, start=train_bars - 1), costs, train_bars, step_bars
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
