"""DTFM's rules, each checked on made-up prices where the right answer is known.

Dates are consecutive business days starting Wednesday 2020-01-01, so the
weekly reviews (each week's last close) fall on rows 2, 7, 12, ... The slow
window needs 60 earlier closes, so the first review that can buy is row 62.
"""

import numpy as np
import pytest

from strategy_lab.config import Config, Costs, TaxRates, Validation
from strategy_lab.data import Panel
from strategy_lab.gates import run_gates
from strategy_lab.portfolio import last_trading_day_of_week, portfolio_net_log_returns
from strategy_lab.report import verdict_markdown
from strategy_lab.strategies import dtfm, dtfm_weights
from strategy_lab.synthetic import synthetic_panel

RULES = dict(fast=20, slow=60, stop=0.08, slots=3, buy_weight=0.10, max_weight=0.15)


def panel_from_growth(daily_growth: np.ndarray) -> Panel:
    """daily_growth[t, j] is the return of asset j on bar t (row 0 is ignored)."""
    growth = np.asarray(daily_growth, dtype=float).copy()
    growth[0] = 0.0
    close = 100.0 * np.cumprod(1.0 + growth, axis=0)
    dates = np.busday_offset(np.datetime64("2020-01-01"), np.arange(len(close)), roll="forward")
    symbols = tuple(f"A{j}" for j in range(close.shape[1]))
    return Panel(dates.astype("datetime64[D]"), symbols, close, close, close, close)


def steady(n: int, rates) -> np.ndarray:
    return np.tile(np.asarray(rates, dtype=float), (n, 1))


def test_reviews_fall_on_the_rows_the_other_tests_assume():
    panel = panel_from_growth(steady(20, [0.001]))
    assert np.flatnonzero(last_trading_day_of_week(panel.dates)).tolist() == [2, 7, 12, 17]


def test_buys_the_three_strongest_at_ten_percent_each():
    """Five assets rise steadily at different speeds. All qualify, there are
    3 slots, so the three with the highest 60-session return are bought."""
    panel = panel_from_growth(steady(80, [0.001, 0.005, 0.003, 0.004, 0.002]))
    weights = dtfm_weights(panel, **RULES)
    assert np.all(weights[:62] == 0.0)  # no 60-session return exists before row 60
    assert weights[62] == pytest.approx([0.0, 0.10, 0.10, 0.10, 0.0])


def test_holdings_are_left_alone_between_and_at_later_reviews():
    panel = panel_from_growth(steady(80, [0.001, 0.005, 0.003, 0.004, 0.002]))
    weights = dtfm_weights(panel, **RULES)
    _, turnover = portfolio_net_log_returns(weights, panel.close, Costs())
    assert turnover[62] == pytest.approx(0.30)  # the only trade: three buys
    assert turnover[63:] == pytest.approx(0.0, abs=1e-12)  # row 67, 72, 77 reviews trade nothing
    assert weights[67, 1] > 0.10  # the fastest riser has drifted up, not been reset


def test_sells_when_a_window_turns_negative_and_fills_the_slot_with_the_next_best():
    """Asset 1 falls 5% a day for four days after it is bought. At the next
    review its 20-session return is negative: it is sold, and the free slot
    goes to the strongest asset not yet held (asset 4, not asset 0)."""
    growth = steady(80, [0.001, 0.005, 0.003, 0.004, 0.002])
    growth[63:67, 1] = -0.05
    weights = dtfm_weights(panel_from_growth(growth), **RULES)
    assert weights[66, 1] > 0.0  # still held until the review
    assert weights[67, 1] == 0.0
    assert weights[67, 4] == pytest.approx(0.10)
    assert weights[67, 0] == 0.0
    assert weights[67, 2] > 0.0 and weights[67, 3] > 0.0


def one_asset_with_a_midweek_drop(recovery: float) -> np.ndarray:
    """Up 1% a day, then -9% on Tuesday (row 74), then `recovery` a day until
    Friday (rows 75 to 77), then flat."""
    growth = steady(90, [0.01])
    growth[74, 0] = -0.09
    growth[75:78, 0] = recovery
    growth[78:, 0] = 0.0
    return growth


def test_stop_is_checked_only_at_the_weekly_review():
    """Peak close is Monday's (row 73). Tuesday closes 9% below it, already past
    the 8% stop, but nothing is sold until Friday's review (row 77), where the
    close is still 9% below the peak."""
    weights = dtfm_weights(panel_from_growth(one_asset_with_a_midweek_drop(0.0)), **RULES)
    assert np.all(weights[74:77, 0] > 0.0)  # Tuesday to Thursday: still held
    assert weights[77, 0] == 0.0  # Friday: stopped out


def test_no_stop_if_the_price_recovers_by_the_review():
    """Same drop, but +1% a day afterwards: Friday closes 6.2% below the peak,
    inside the 8% stop, so the position stays through that review and the next.
    (At row 87 it is sold for a different reason: with prices flat since Friday,
    the 20-session return has turned negative, 1.01^9 * 0.91 = 0.995.)"""
    weights = dtfm_weights(panel_from_growth(one_asset_with_a_midweek_drop(0.01)), **RULES)
    assert np.all(weights[62:87, 0] > 0.0)
    assert weights[87, 0] == 0.0


def test_a_stopped_asset_waits_for_a_later_review_to_come_back():
    """At the stop-out review the asset still has positive 20 and 60-session
    returns and a slot is free, but it may not be bought back that day. At the
    next review (row 82) it still qualifies and is bought again at 10%."""
    weights = dtfm_weights(panel_from_growth(one_asset_with_a_midweek_drop(0.0)), **RULES)
    assert np.all(weights[77:82, 0] == 0.0)
    assert weights[82, 0] == pytest.approx(0.10)


def test_a_position_that_outgrows_the_cap_is_trimmed_at_the_review():
    """One asset up 1.5% a day while the rest of the account is cash. Bought at
    10%, it passes 15% of the account after about 32 days and is cut back to
    15% at each review from then on."""
    weights = dtfm_weights(panel_from_growth(steady(110, [0.015])), **RULES)
    review = last_trading_day_of_week(np.busday_offset(np.datetime64("2020-01-01"), np.arange(110), roll="forward"))
    assert weights[92, 0] < 0.15  # review row 92: not there yet
    assert weights[97, 0] == pytest.approx(0.15)
    assert weights[review].max() <= 0.15 + 1e-12


def test_never_more_than_three_positions_and_never_leveraged():
    panel = synthetic_panel(1500, 10, seed=7, trend=0.002)
    weights = dtfm_weights(panel, **RULES)
    assert (np.sum(weights > 0.0, axis=1) <= 3).all()
    assert (weights >= 0.0).all() and (weights.sum(axis=1) <= 1.0).all()
    assert np.sum(weights > 0.0) > 0  # it does trade


def test_dtfm_never_uses_future_bars():
    panel = synthetic_panel(700, 10, seed=7, trend=0.002)
    family = dtfm()
    full = family.weight_matrix(panel)
    assert full.shape == (1, 700, 10)
    for k in (150, 333, 600):
        assert np.array_equal(family.weight_matrix(panel.slice(0, k))[:, : k - 1], full[:, : k - 1])


SMALL = Config(
    symbols=("A0",),
    benchmark="A0",
    source="test",
    start="2000-01-01",
    holdout_months=12,
    costs=Costs(),
    tax=TaxRates(),
    validation=Validation(
        train_bars=504, step_bars=21, n_perm_insample=200, n_perm_walkforward=100, seed=5
    ),
)


@pytest.mark.parametrize("trend", [0.0, 0.004])
def test_dtfm_runs_through_the_gates_and_the_verdict_renders(trend):
    """The whole path the real run takes, on made-up data: gates, then the report."""
    panel = synthetic_panel(504 + 3 * 252, 10, seed=41, trend=trend)
    verdict = run_gates(
        dtfm(), panel, panel.bars("A0"), SMALL, {"ideas": 2, "param_sets": 91}, say=lambda _: None
    )
    text = verdict_markdown(verdict, "synthetic data")
    assert "# Verdict: dtfm" in text
    assert verdict["gates"]["2"]["alpha"] == pytest.approx(0.005)
    if trend == 0.0:
        assert not verdict["passed_all_gates"]  # no edge in random walks
