"""Permutation, walk-forward and the gates."""

import numpy as np
import pytest

from strategy_lab.config import Config, Costs, TaxRates, Validation
from strategy_lab.engine import net_log_returns
from strategy_lab.gates import run_gates
from strategy_lab.permutation import insample_permutation_test, p_value, permute_bars
from strategy_lab.strategies import Family, random_signal, sma_crossover
from strategy_lab.synthetic import synthetic_bars
from strategy_lab.walkforward import walk_forward_positions, walk_forward_score

from conftest import make_bars

COSTS = Costs()
SMALL = Config(
    symbols=("SPY",),
    benchmark="SPY",
    source="test",
    start="2000-01-01",
    holdout_months=12,
    costs=COSTS,
    tax=TaxRates(),
    validation=Validation(
        train_bars=504, step_bars=21, n_perm_insample=200, n_perm_walkforward=100, seed=5
    ),
)


# ---------------------------------------------------------------- permutation


def test_permutation_keeps_the_first_and_last_close():
    bars = synthetic_bars(600, seed=1)
    shuffled = permute_bars(bars, np.random.default_rng(0))
    assert shuffled.close[0] == bars.close[0]
    assert shuffled.close[-1] == pytest.approx(bars.close[-1], rel=1e-10)
    assert not np.allclose(shuffled.close, bars.close)


def test_permutation_keeps_every_gap_and_intraday_move():
    bars = synthetic_bars(600, seed=1)
    shuffled = permute_bars(bars, np.random.default_rng(0))

    def moves(b):
        gap = np.log(b.open[1:]) - np.log(b.close[:-1])
        intraday = np.log(b.close[1:]) - np.log(b.open[1:])
        return np.sort(gap), np.sort(intraday)

    for original, permuted in zip(moves(bars), moves(shuffled)):
        assert permuted == pytest.approx(original, abs=1e-10)


def test_permuted_bars_are_valid_bars():
    shuffled = permute_bars(synthetic_bars(600, seed=1), np.random.default_rng(0))
    assert np.all(shuffled.high >= np.maximum(shuffled.open, shuffled.close) - 1e-12)
    assert np.all(shuffled.low <= np.minimum(shuffled.open, shuffled.close) + 1e-12)


def test_permutation_leaves_bars_up_to_start_untouched():
    bars = synthetic_bars(600, seed=1)
    shuffled = permute_bars(bars, np.random.default_rng(0), start=299)
    assert np.array_equal(shuffled.close[:300], bars.close[:300])
    assert np.array_equal(shuffled.open[:300], bars.open[:300])
    assert not np.allclose(shuffled.close[300:], bars.close[300:])
    assert shuffled.close[-1] == pytest.approx(bars.close[-1], rel=1e-10)


def test_permutation_is_reproducible_from_its_seed():
    bars = synthetic_bars(600, seed=1)
    a = permute_bars(bars, np.random.default_rng(3))
    b = permute_bars(bars, np.random.default_rng(3))
    c = permute_bars(bars, np.random.default_rng(4))
    assert np.array_equal(a.close, b.close)
    assert not np.array_equal(a.close, c.close)


def test_p_value_counts_the_real_run():
    assert p_value(5.0, np.array([1.0, 2.0, 3.0])) == pytest.approx(1 / 4)
    assert p_value(2.0, np.array([1.0, 2.0, 3.0])) == pytest.approx(3 / 4)


def test_permutation_test_is_reproducible():
    bars = synthetic_bars(504, seed=9)
    a = insample_permutation_test(sma_crossover(), bars, COSTS, 50, seed=1)
    b = insample_permutation_test(sma_crossover(), bars, COSTS, 50, seed=1)
    assert a == b


# --------------------------------------------------------------- walk-forward


LONG_OR_FLAT = Family(
    name="long_or_flat",
    rationale="test",
    grid=({"long": 1}, {"long": 0}),
    position_matrix=lambda bars: np.array([np.ones(len(bars)), np.zeros(len(bars))]),
)


def test_walk_forward_hand_checked():
    """10 closes rising 1% a bar, then 10 falling 1% a bar. Train on 4 bars, trade 2.

    While the last 4 returns are up, 'always long' wins the training window.
    At the step starting on bar 12 the last 4 returns are up, up, down, down,
    which nets slightly negative, so 'always flat' wins from then on.
    So: long is decided at closes 3..10, flat everywhere else.
    """
    close = 100.0 * np.cumprod([1.0] + [1.01] * 9 + [0.99] * 10)
    position, choices = walk_forward_positions(
        LONG_OR_FLAT, make_bars(close), Costs(0.0, 0.0), train_bars=4, step_bars=2
    )
    expected = np.zeros(20)
    expected[3:11] = 1.0
    assert np.array_equal(position, expected)
    assert [s for s, _ in choices] == list(range(4, 20, 2))


def test_walk_forward_never_uses_future_bars():
    """Replace every bar after index k with different prices. Positions decided
    up to k must not move."""
    k, train, step = 900, 504, 21
    a = synthetic_bars(1400, seed=11, trend=0.001)
    other = synthetic_bars(1400, seed=12, trend=0.001)
    scale = a.close[k] / other.close[k]

    def spliced(x, y):
        return np.concatenate([x[: k + 1], y[k + 1 :] * scale])

    b = type(a)(a.dates, *(spliced(getattr(a, f), getattr(other, f)) for f in ("open", "high", "low", "close")))
    family = sma_crossover()
    pos_a, _ = walk_forward_positions(family, a, COSTS, train, step)
    pos_b, _ = walk_forward_positions(family, b, COSTS, train, step)
    assert np.array_equal(pos_a[: k + 1], pos_b[: k + 1])
    assert not np.array_equal(pos_a, pos_b)


def test_walk_forward_trades_nothing_before_the_first_training_window_ends():
    bars = synthetic_bars(1000, seed=11, trend=0.001)
    position, _ = walk_forward_positions(sma_crossover(), bars, COSTS, 504, 21)
    assert np.all(position[:503] == 0.0)
    net, _ = net_log_returns(position, bars.close, COSTS)
    assert np.all(net[:504] == 0.0)
    assert walk_forward_score(sma_crossover(), bars, COSTS, 504, 21) == pytest.approx(net.sum())


# ---------------------------------------------------------------------- gates


def test_gates_pass_a_planted_edge():
    bars = synthetic_bars(504 + 3 * 252, seed=202, trend=0.004)
    verdict = run_gates(sma_crossover(), bars, bars, SMALL, {"ideas": 1, "param_sets": 91}, say=lambda _: None)
    assert verdict["passed_all_gates"], verdict["died_at"]
    assert [verdict["gates"][g]["passed"] for g in "1234"] == [True] * 4


def test_gates_kill_a_random_signal_and_leave_walk_forward_data_alone():
    bars = synthetic_bars(504 + 3 * 252, seed=101, trend=0.0)
    verdict = run_gates(random_signal(), bars, bars, SMALL, {"ideas": 1, "param_sets": 50}, say=lambda _: None)
    assert not verdict["passed_all_gates"]
    assert verdict["gates"]["2"]["passed"] is False
    assert verdict["gates"]["3"]["passed"] is None  # never run
    assert verdict["gates"]["4"]["passed"] is None


def test_more_ideas_tested_means_a_stricter_bar():
    bars = synthetic_bars(504 + 252, seed=101, trend=0.0)
    one = run_gates(random_signal(), bars, bars, SMALL, {"ideas": 1, "param_sets": 50}, say=lambda _: None)
    ten = run_gates(random_signal(), bars, bars, SMALL, {"ideas": 10, "param_sets": 500}, say=lambda _: None)
    assert ten["gates"]["2"]["alpha"] == pytest.approx(one["gates"]["2"]["alpha"] / 10)
