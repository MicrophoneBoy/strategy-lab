"""The engine: a hand-checked example, no lookahead, costs, reproducibility."""

import math

import numpy as np
import pytest

from strategy_lab.config import Costs
from strategy_lab.engine import backtest, buy_and_hold, log_returns

from conftest import make_bars

NO_COSTS = Costs(fee_bps=0.0, slippage_bps=0.0)
TEN_BPS = Costs(fee_bps=0.0, slippage_bps=10.0)


def test_hand_checked_example():
    """Small enough to do on paper.

    Closes: 100 -> 110 -> 121 -> 108.9 -> 108.9   (+10%, +10%, -10%, flat)
    Decide long at closes 0 and 1, then go flat. Cost 10 bps per change.

      bar 1: long, just bought   1.10 * 0.999
      bar 2: long                1.10
      bar 3: flat, just sold     0.999   (the -10% bar is avoided)
      bar 4: flat                1
      final equity = 1.21 * 0.999^2 = 1.20758121
    """
    bars = make_bars([100.0, 110.0, 121.0, 108.9, 108.9])
    result = backtest(bars, [1, 1, 0, 0, 0], TEN_BPS)

    expected = [
        math.log(1.10) + math.log(0.999),
        math.log(1.10),
        math.log(0.999),
        0.0,
    ]
    assert result.net == pytest.approx(expected, abs=1e-12)
    assert result.metrics["total_return"] == pytest.approx(0.20758121, abs=1e-9)
    assert result.metrics["trades"] == 1
    assert result.metrics["exposure"] == pytest.approx(0.5)
    assert result.metrics["max_drawdown"] == pytest.approx(-0.001, abs=1e-9)

    # Buy-and-hold over the same bars: 1.089 * 0.999 = 1.087911
    benchmark = buy_and_hold(bars, TEN_BPS)
    assert benchmark.metrics["total_return"] == pytest.approx(0.087911, abs=1e-9)


def test_signal_cannot_profit_from_the_bar_it_was_computed_on(random_bars):
    """A signal that is long whenever *today* was up knows nothing about tomorrow.
    An engine with lookahead would hand it every winning bar."""
    r = log_returns(random_bars.close)
    long_if_today_was_up = (r > 0).astype(float)
    result = backtest(random_bars, long_if_today_was_up, NO_COSTS)

    honest = float(np.sum(long_if_today_was_up[:-1] * r[1:]))
    every_winning_bar = float(r[r > 0].sum())
    assert result.net.sum() == pytest.approx(honest, abs=1e-12)
    assert result.net.sum() < 0.25 * every_winning_bar


def test_shifting_signals_forward_changes_results(random_bars):
    """Hand the engine tomorrow's signal today and the result must change (here,
    into a perfect one). If it did not change, the engine was already peeking."""
    r = log_returns(random_bars.close)
    signal = (r > 0).astype(float)
    peeking = np.roll(signal, -1)
    peeking[-1] = 0.0

    honest = backtest(random_bars, signal, NO_COSTS).net.sum()
    cheating = backtest(random_bars, peeking, NO_COSTS).net.sum()
    assert cheating == pytest.approx(r[1:][r[1:] > 0].sum(), abs=1e-12)
    assert cheating > 4 * abs(honest)


def test_costs_scale_with_position_change():
    bars = make_bars([100.0, 100.0, 100.0, 100.0])
    # Flat prices, so only costs remain. Long, then flip short: 1 unit, then 2 units.
    result = backtest(bars, [1, -1, -1, 0], TEN_BPS)
    assert result.net == pytest.approx(
        [math.log(1 - 0.001), math.log(1 - 0.002), 0.0], abs=1e-12
    )


def test_no_trades_means_no_costs_and_no_return(random_bars):
    result = backtest(random_bars, np.zeros(len(random_bars)), TEN_BPS)
    assert np.all(result.net == 0.0)
    assert result.metrics["trades"] == 0
    assert result.metrics["exposure"] == 0.0


def test_more_cost_means_less_return(random_bars):
    rng = np.random.default_rng(0)
    position = rng.integers(0, 2, len(random_bars)).astype(float)
    cheap = backtest(random_bars, position, Costs(0.0, 1.0)).metrics["total_return"]
    dear = backtest(random_bars, position, Costs(0.0, 20.0)).metrics["total_return"]
    assert dear < cheap


def test_results_are_identical_on_rerun(random_bars):
    rng = np.random.default_rng(1)
    position = rng.integers(-1, 2, len(random_bars)).astype(float)
    a = backtest(random_bars, position, TEN_BPS)
    b = backtest(random_bars, position, TEN_BPS)
    assert np.array_equal(a.net, b.net)
    assert a.metrics == b.metrics


def test_rejects_positions_outside_long_flat_short(random_bars):
    position = np.zeros(len(random_bars))
    position[5] = 2.0
    with pytest.raises(ValueError):
        backtest(random_bars, position, NO_COSTS)
