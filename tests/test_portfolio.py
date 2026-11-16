"""The multi-asset engine: agrees with the single-asset engine, hand-checked
examples, weekly rebalancing, trailing stops, FIFO tax lots, joint shuffles."""

import math

import numpy as np
import pytest

from strategy_lab.config import Config, Costs, TaxRates, Validation
from strategy_lab.data import Panel
from strategy_lab.engine import backtest
from strategy_lab.gates import run_gates
from strategy_lab.permutation import permute_bars
from strategy_lab.portfolio import (
    after_tax_portfolio,
    hold_between_rebalances,
    last_trading_day_of_week,
    portfolio_backtest,
    portfolio_net_log_returns,
)
from strategy_lab.strategies import dual_momentum, random_portfolio, random_signal
from strategy_lab.synthetic import synthetic_panel

NO_COSTS = Costs(0.0, 0.0)
TEN_BPS = Costs(0.0, 10.0)
RATES = TaxRates(short_term=0.22, long_term=0.15)


def panel_from(close, dates=None) -> Panel:
    close = np.asarray(close, dtype=float)
    if dates is None:
        dates = np.busday_offset(np.datetime64("2020-01-01"), np.arange(len(close)), roll="forward")
    dates = np.asarray(dates, dtype="datetime64[D]")
    symbols = tuple(f"A{j}" for j in range(close.shape[1]))
    return Panel(dates, symbols, close, close, close, close)


# ------------------------------------------------------------------ the engine


def test_one_asset_matches_the_single_asset_engine(random_bars):
    """Weights of 0 or 1 on one asset must reproduce the single-asset engine."""
    position = random_signal(n_seeds=1).positions(random_bars, 0)
    single = backtest(random_bars, position, TEN_BPS)
    panel = panel_from(random_bars.close[:, None], random_bars.dates)
    multi = portfolio_backtest(panel, position[:, None], TEN_BPS)
    assert multi.net == pytest.approx(single.net, abs=1e-12)
    for key in ("total_return", "cagr", "sharpe", "max_drawdown", "exposure"):
        assert multi.metrics[key] == pytest.approx(single.metrics[key], abs=1e-10)


def test_two_assets_hand_checked_buy_and_drift():
    """Put half in A and half in B, then never trade. Cost 10 bps.

    A: 100 -> 110 -> 110      B: 100 -> 100 -> 90
    bar 1: 0.5 * 10% = +5%, minus the cost of buying 100% of the account
    bar 2: weights have drifted to 0.55/1.05 and 0.50/1.05; B falls 10%
    In dollars: 0.55 + 0.45 = 1.00, times 0.999 for the cost.
    """
    panel = panel_from([[100.0, 100.0], [110.0, 100.0], [110.0, 90.0]])
    drifted = [0.55 / 1.05, 0.50 / 1.05]
    weights = np.array([[0.5, 0.5], drifted, drifted])
    net, turnover = portfolio_net_log_returns(weights, panel.close, TEN_BPS)

    assert turnover[:2] == pytest.approx([1.0, 0.0], abs=1e-12)
    assert net[1] == pytest.approx(math.log(1.05) + math.log(0.999), abs=1e-12)
    assert net[2] == pytest.approx(math.log(1.0 / 1.05), abs=1e-12)
    assert math.exp(net.sum()) == pytest.approx(0.999, abs=1e-12)


def test_two_assets_hand_checked_rebalance():
    """Same prices, but reset to 50/50 at close 1. That trade moves
    |0.5 - 0.5238| + |0.5 - 0.4762| = 4.76% of the account."""
    panel = panel_from([[100.0, 100.0], [110.0, 100.0], [110.0, 90.0]])
    weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    net, turnover = portfolio_net_log_returns(weights, panel.close, TEN_BPS)

    moved = abs(0.5 - 0.55 / 1.05) + abs(0.5 - 0.50 / 1.05)
    assert turnover[1] == pytest.approx(moved, abs=1e-12)
    assert net[2] == pytest.approx(math.log(0.95) + math.log(1 - 0.001 * moved), abs=1e-12)


def test_cash_earns_nothing():
    panel = panel_from([[100.0], [110.0], [121.0]])
    net, _ = portfolio_net_log_returns(np.full((3, 1), 0.5), panel.close, NO_COSTS)
    assert net[1] == pytest.approx(math.log(1.05))  # half invested in a +10% bar


def test_rejects_leverage_and_shorts():
    panel = panel_from([[100.0, 100.0], [101.0, 101.0]])
    with pytest.raises(ValueError, match="leverage"):
        portfolio_backtest(panel, np.full((2, 2), 0.6), NO_COSTS)
    with pytest.raises(ValueError, match="long only"):
        portfolio_backtest(panel, np.array([[0.5, -0.1], [0.5, 0.0]]), NO_COSTS)


# ------------------------------------------------- rebalancing and trailing stops


def test_last_trading_day_of_week_handles_a_holiday_friday():
    # Good Friday 2021-04-02 was a market holiday, so Thursday closes that week.
    dates = np.array(
        ["2021-03-29", "2021-03-30", "2021-03-31", "2021-04-01",
         "2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08", "2021-04-09"],
        dtype="datetime64[D]",
    )
    expected = [False, False, False, True, False, False, False, False, True]
    assert last_trading_day_of_week(dates).tolist() == expected


def test_weights_drift_between_rebalances():
    close = np.array([[100.0, 100.0], [110.0, 100.0], [110.0, 90.0]])
    targets = np.full((3, 2), 0.5)
    weights = hold_between_rebalances(targets, np.array([True, False, False]), close)
    assert weights[0] == pytest.approx([0.5, 0.5])
    assert weights[1] == pytest.approx([0.55 / 1.05, 0.50 / 1.05])
    assert weights[2] == pytest.approx([0.55, 0.45])
    # Holding the drifted weights costs nothing after the first purchase.
    _, turnover = portfolio_net_log_returns(weights, close, TEN_BPS)
    assert turnover[1:] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_trailing_stop_hand_checked():
    """Bought at 100, peak close 120. An 8% stop sits at 110.4, so the close at
    109 sells. Without a later rebalance the money stays in cash."""
    close = np.array([[100.0], [110.0], [120.0], [109.0], [115.0], [130.0]])
    targets = np.ones((6, 1))
    only_first = np.array([True, False, False, False, False, False])
    weights = hold_between_rebalances(targets, only_first, close, trailing_stop=0.08)
    assert weights[:, 0].tolist() == [1, 1, 1, 0, 0, 0]


def test_trailing_stop_not_hit_and_buy_back_at_next_rebalance():
    close = np.array([[100.0], [110.0], [120.0], [111.0], [115.0], [130.0]])
    targets = np.ones((6, 1))
    only_first = np.array([True, False, False, False, False, False])
    held = hold_between_rebalances(targets, only_first, close, trailing_stop=0.08)
    assert held[:, 0].tolist() == [1, 1, 1, 1, 1, 1]  # 111 is above 110.4

    close[3, 0] = 109.0  # now the stop is hit, and close 4 is a rebalance day
    rebalance = np.array([True, False, False, False, True, False])
    bought_back = hold_between_rebalances(targets, rebalance, close, trailing_stop=0.08)
    assert bought_back[:, 0].tolist() == [1, 1, 1, 0, 1, 1]


def test_each_parameter_row_can_have_its_own_stop():
    close = np.array([[100.0], [120.0], [109.0], [109.0]])
    targets = np.ones((2, 4, 1))
    rebalance = np.array([True, False, False, False])
    weights = hold_between_rebalances(targets, rebalance, close, trailing_stop=np.array([0.08, 1.0]))
    assert weights[0, :, 0].tolist() == [1, 1, 0, 0]  # 8% stop sells
    assert weights[1, :, 0].tolist() == [1, 1, 1, 1]  # a 100% stop never triggers


# ------------------------------------------------------------------- families

FAMILIES = [dual_momentum(), random_portfolio(n_seeds=4)]


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_weights_never_use_future_bars(family):
    """Weights from the first k bars must equal the first rows of weights from
    all bars. (The final row is left out: whether it is a week's last trading
    day depends on the calendar, which is known ahead of time.)"""
    panel = synthetic_panel(700, 6, seed=5, trend=0.002)
    full = family.weight_matrix(panel)
    for k in (150, 333, 600):
        part = family.weight_matrix(panel.slice(0, k))
        assert np.array_equal(part[:, : k - 1], full[:, : k - 1])


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.name)
def test_weights_are_long_only_without_leverage(family):
    panel = synthetic_panel(700, 6, seed=5, trend=0.002)
    weights = family.weight_matrix(panel)
    assert weights.shape == (len(family.grid), 700, 6)
    assert (weights >= 0).all()
    assert (weights.sum(axis=-1) <= 1 + 1e-9).all()


def test_dual_momentum_respects_the_cap_on_rebalance_days():
    panel = synthetic_panel(700, 6, seed=5, trend=0.002)
    weights = dual_momentum(max_weight=0.15).weight_matrix(panel)
    rebalance = last_trading_day_of_week(panel.dates)
    assert (weights[:, rebalance] <= 0.15 + 1e-12).all()
    assert weights[:, rebalance].max() == pytest.approx(0.15)


def test_dual_momentum_holds_nothing_until_the_slow_window_exists():
    panel = synthetic_panel(300, 4, seed=5, trend=0.002)
    family = dual_momentum()
    k = family.grid.index({"fast": 20, "slow": 60, "stop": None})
    assert np.all(family.weight_matrix(panel)[k, :60] == 0.0)


# ------------------------------------------------------------------ tax lots


def test_single_asset_tax_matches_the_single_asset_model(random_bars):
    """With no costs and every trade shorter than a year, the FIFO model and
    the single-asset model must agree."""
    position = random_signal(n_seeds=1).positions(random_bars, 0)
    single = backtest(random_bars, position, NO_COSTS, RATES).metrics
    panel = panel_from(random_bars.close[:, None], random_bars.dates)
    multi = portfolio_backtest(panel, position[:, None], NO_COSTS, RATES).metrics
    assert multi["after_tax_total_return"] == pytest.approx(single["after_tax_total_return"], rel=1e-9)
    assert multi["tax_paid"] == pytest.approx(single["tax_paid"], rel=1e-9)


def test_oldest_lot_is_sold_first():
    """Buy half at 100 (Jan 2020). It doubles. Buy more at 200 (Jun 2021),
    then sell half the position the next day.

    FIFO sells the 2020 lot: 0.00375 shares * $100 gain = 0.375, long-term,
    tax 0.05625 for 2021. (Selling the new lot first would owe nothing.)
    In 2022 the rest is sold: the tax payment shrank holdings by 0.9625, so the
    old lot's remaining gain is 0.125 * 0.9625, taxed at 15%.
    """
    dates = np.array(["2020-01-02", "2021-06-01", "2021-06-02", "2022-01-03"], dtype="datetime64[D]")
    close = np.array([[100.0], [200.0], [200.0], [200.0]])
    weights = np.array([[0.5], [1.0], [0.5], [0.5]])
    net, turnover = portfolio_net_log_returns(weights, close, NO_COSTS)
    result = after_tax_portfolio(weights, close, dates, net, turnover, RATES)

    tax_2021 = 0.15 * 0.375
    tax_2022 = 0.15 * 0.125 * (1.5 - tax_2021) / 1.5
    assert result["tax_paid"] == pytest.approx(tax_2021 + tax_2022)
    assert result["after_tax_total_return"] == pytest.approx(0.5 - tax_2021 - tax_2022)
    assert result["long_term_trades"] == 2 and result["short_term_trades"] == 1


def test_realised_gains_add_up_to_the_actual_profit():
    """Inside one calendar year, with no costs and one flat tax rate, the tax
    must be exactly rate * profit however the weights moved. This fails if any
    lot is lost or double counted."""
    rng = np.random.default_rng(3)
    n, m = 200, 4
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.01, (n, m)), axis=0))
    raw = rng.random((n, m))
    weights = raw / raw.sum(axis=1, keepdims=True) * rng.uniform(0.3, 1.0, (n, 1))
    dates = np.busday_offset(np.datetime64("2021-01-04"), np.arange(n), roll="forward")
    net, turnover = portfolio_net_log_returns(weights, close, NO_COSTS)
    profit = math.exp(net.sum()) - 1.0
    assert profit > 0
    flat = TaxRates(short_term=0.25, long_term=0.25)
    result = after_tax_portfolio(weights, close, dates.astype("datetime64[D]"), net, turnover, flat)
    assert result["after_tax_total_return"] == pytest.approx(0.75 * profit, rel=1e-9)


# ------------------------------------------------------------- joint shuffles


def test_panel_shuffle_keeps_each_days_moves_together():
    """The same shuffle is applied to every asset, so the set of daily
    (asset 0, asset 1, ...) move vectors is unchanged."""
    panel = synthetic_panel(400, 3, seed=8)
    shuffled = permute_bars(panel, np.random.default_rng(0))
    assert isinstance(shuffled, Panel) and shuffled.symbols == panel.symbols

    def intraday_rows(p):
        rows = np.log(p.close[1:]) - np.log(p.open[1:])
        return rows[np.lexsort(rows.T[::-1])]

    assert intraday_rows(shuffled) == pytest.approx(intraday_rows(panel), abs=1e-10)
    assert shuffled.close[-1] == pytest.approx(panel.close[-1], rel=1e-10)
    assert not np.allclose(shuffled.close, panel.close)


def test_panel_shuffle_leaves_bars_up_to_start_untouched():
    panel = synthetic_panel(400, 3, seed=8)
    shuffled = permute_bars(panel, np.random.default_rng(0), start=199)
    assert np.array_equal(shuffled.close[:200], panel.close[:200])
    assert not np.allclose(shuffled.close[200:], panel.close[200:])


# ---------------------------------------------------------------------- gates

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


def test_gates_pass_a_planted_edge_in_a_portfolio():
    panel = synthetic_panel(504 + 3 * 252, 6, seed=21, trend=0.004)
    verdict = run_gates(
        dual_momentum(), panel, panel.bars("A0"), SMALL, {"ideas": 1, "param_sets": 16}, say=lambda _: None
    )
    assert verdict["passed_all_gates"], (verdict["died_at"], verdict["gates"]["2"]["p_value"])


def test_gates_kill_a_random_portfolio():
    panel = synthetic_panel(504 + 3 * 252, 6, seed=22, trend=0.0)
    verdict = run_gates(
        random_portfolio(), panel, panel.bars("A0"), SMALL, {"ideas": 1, "param_sets": 30}, say=lambda _: None
    )
    assert not verdict["passed_all_gates"]
    assert verdict["gates"]["2"]["passed"] is False
