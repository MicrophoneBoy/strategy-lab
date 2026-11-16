"""Strategies must not see the future. Sweep and plateau check behave."""

import numpy as np
import pandas as pd
import pytest

from strategy_lab.config import Costs
from strategy_lab.strategies import random_signal, sma, sma_crossover
from strategy_lab.sweep import plateau_check, sweep

FAMILIES = [sma_crossover(), sma_crossover(allow_short=True), random_signal(n_seeds=5)]


def test_sma_matches_a_plain_average():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(x, 3)
    assert np.isnan(out[:2]).all()
    assert out[2:] == pytest.approx([2.0, 3.0, 4.0])


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_positions_never_use_future_bars(family, random_bars):
    """Positions computed on the first k bars must equal the first k positions
    computed on all bars. If they differ, a signal looked ahead."""
    full = family.position_matrix(random_bars)
    for k in (250, 777, 1200):
        assert np.array_equal(family.position_matrix(random_bars.slice(0, k)), full[:, :k])


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_positions_are_long_flat_or_short(family, random_bars):
    matrix = family.position_matrix(random_bars)
    assert matrix.shape == (len(family.grid), len(random_bars))
    assert np.isin(matrix, (-1.0, 0.0, 1.0)).all()


def test_sma_crossover_is_flat_until_the_slow_average_exists(random_bars):
    family = sma_crossover()
    k = family.grid.index({"fast": 50, "slow": 200})
    assert np.all(family.positions(random_bars, k)[:199] == 0.0)


def test_sweep_has_one_row_per_parameter_set(random_bars):
    family = sma_crossover()
    table = sweep(family, random_bars, random_bars, Costs())
    assert len(table) == len(family.grid)
    assert {"fast", "slow", "cagr", "excess_cagr", "sharpe", "trades"} <= set(table.columns)


def _grid(values: np.ndarray) -> pd.DataFrame:
    rows = [
        {"fast": j, "slow": i, "excess_cagr": values[i, j]}
        for i in range(values.shape[0])
        for j in range(values.shape[1])
    ]
    return pd.DataFrame(rows)


def test_plateau_check_flags_a_lone_spike():
    values = np.zeros((5, 5))
    values[2, 2] = 0.10  # one great cell surrounded by nothing
    result = plateau_check(_grid(values), "fast", "slow")
    assert result["shape"] == "spike"
    assert result["best"]["fast"] == 2 and result["best"]["slow"] == 2


def test_plateau_check_accepts_a_broad_hill():
    i, j = np.mgrid[0:5, 0:5]
    values = 0.10 - 0.01 * ((i - 2) ** 2 + (j - 2) ** 2)  # smooth hill
    result = plateau_check(_grid(values), "fast", "slow")
    assert result["shape"] == "plateau"
