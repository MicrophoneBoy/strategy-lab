"""Ideas 3 to 9: hand-checked rules, no lookahead, and the full gate path on made-up data."""

import numpy as np
import pytest

from strategy_lab.config import Config, Costs, TaxRates, Validation
from strategy_lab.data import Panel
from strategy_lab.gates import run_gates
from strategy_lab.ideas import (
    breakout,
    breakout_positions,
    last_trading_day_of_month,
    low_vol_sectors,
    mean_reversion,
    pick,
    rolling_extreme,
    rolling_std,
    sector_rotation,
    stocks_or_bonds,
    turn_of_month,
    turn_of_month_days,
    volatility_filter,
)
from strategy_lab.report import summary_markdown, verdict_markdown
from strategy_lab.synthetic import synthetic_bars, synthetic_panel

from conftest import make_bars

SINGLE = [turn_of_month(), volatility_filter(), mean_reversion(), breakout()]
MULTI = [sector_rotation(), low_vol_sectors()]


def days(*dates):
    return np.array(dates, dtype="datetime64[D]")


# ------------------------------------------------------------ building blocks


def test_month_end_mask():
    dates = days("2021-01-28", "2021-01-29", "2021-02-01", "2021-02-26", "2021-03-01")
    assert last_trading_day_of_month(dates).tolist() == [False, True, False, True, False]


def test_rolling_std_matches_a_direct_calculation():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 50)
    r[0] = 0.0
    out = rolling_std(r, 10)
    assert np.isnan(out[:10]).all()
    assert out[10] == pytest.approx(np.std(r[1:11], ddof=1))
    assert out[49] == pytest.approx(np.std(r[40:50], ddof=1))


def test_rolling_extreme():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert rolling_extreme(x, 3, largest=True)[2:].tolist() == [4.0, 4.0, 5.0]
    assert rolling_extreme(x, 3, largest=False)[2:].tolist() == [1.0, 1.0, 1.0]


def test_pick_takes_the_top_k_equally():
    scores = np.array([[0.1, 0.5, 0.3, 0.2], [np.nan, 0.5, 0.3, 0.2]])
    weights = pick(scores, 2)
    assert weights[0].tolist() == [0.0, 0.5, 0.5, 0.0]
    assert weights[1].tolist() == [0.0, 0.0, 0.0, 0.0]  # a missing score: hold nothing


# --------------------------------------------------------------- hand checks


def test_turn_of_month_window_hand_checked():
    """January 2021 into February: the last 2 business days of January are the
    28th and 29th; the first 2 trading days of February are the 1st and 2nd."""
    dates = days(
        "2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29",
        "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04",
    )
    inside = turn_of_month_days(dates, last=2, first=2)
    assert inside.tolist() == [False, False, False, True, True, True, True, False, False]


def test_turn_of_month_is_long_going_into_each_window_day():
    dates = days(
        "2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29",
        "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04",
    )
    bars = make_bars(np.linspace(100, 108, 9))
    bars = type(bars)(dates, bars.open, bars.high, bars.low, bars.close)
    family = turn_of_month()
    k = family.grid.index({"last": 2, "first": 2})
    # Decided at the close before each window day: closes of Jan 27, 28, 29 and Feb 1.
    assert family.positions(bars, k).tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]


def test_mean_reversion_hand_checked():
    bars = make_bars([100.0, 97.0, 97.5, 96.0, 99.0])
    family = mean_reversion()
    one_day_2pct = family.grid.index({"lookback": 1, "drop": 0.02})
    # 1-day returns: -3.0%, +0.5%, -1.5%, +3.1%. Only the first is below -2%.
    assert family.positions(bars, one_day_2pct).tolist() == [0, 1, 0, 0, 0]


def test_breakout_hand_checked():
    """3-day high to enter, 2-day low to exit."""
    close = np.array([10.0, 9.0, 9.5, 11.0, 12.0, 11.5, 11.0, 11.2, 13.0])
    #  new 3-day high at rows 3, 4 and 8; new 2-day low at rows 1, 5 and 6.
    assert breakout_positions(close, 3, 2).tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 1]


def test_volatility_filter_is_long_in_calm_and_flat_in_storm():
    """300 calm days, then 60 wild ones. With the 50th percentile, the filter
    must be out of the market once the wild days dominate the short window."""
    rng = np.random.default_rng(1)
    returns = np.concatenate([rng.normal(0, 0.005, 300), rng.normal(0, 0.03, 60)])
    bars = make_bars(100.0 * np.exp(np.cumsum(returns)))
    family = volatility_filter()
    k = family.grid.index({"window": 10, "percentile": 50})
    position = family.positions(bars, k)
    assert np.all(position[:261] == 0.0)  # not enough history yet (10 + 252 - 1)
    assert position[262:300].mean() > 0.2  # trades during the calm stretch
    assert np.all(position[320:] == 0.0)  # flat through the storm


def test_sector_rotation_holds_the_strongest_sector():
    n = 200
    growth = np.tile([0.001, 0.004, 0.002], (n, 1))
    close = 100.0 * np.cumprod(1.0 + growth, axis=0)
    dates = np.busday_offset(np.datetime64("2020-01-01"), np.arange(n), roll="forward").astype("datetime64[D]")
    panel = Panel(dates, ("A", "B", "C"), close, close, close, close)
    family = sector_rotation()
    k = family.grid.index({"lookback": 63, "hold": 1})
    weights = family.weight_matrix(panel)[k]
    month_ends = np.flatnonzero(last_trading_day_of_month(dates))
    first = month_ends[month_ends >= 63][0]
    assert np.all(weights[:first] == 0.0)
    assert weights[first:, 1] == pytest.approx(1.0)  # all in B, the fastest riser


def test_stocks_or_bonds_switches_on_the_stock_trend():
    """Stocks rise for 150 days then fall; bonds are flat. After the first
    month end where the 63-day stock return is negative, everything is in bonds."""
    n = 300
    stock = 100.0 * np.cumprod(np.r_[np.full(150, 1.002), np.full(150, 0.997)])
    close = np.column_stack([stock, np.full(n, 100.0)])
    dates = np.busday_offset(np.datetime64("2020-01-01"), np.arange(n), roll="forward").astype("datetime64[D]")
    panel = Panel(dates, ("SPY", "IEF"), close, close, close, close)
    family = stocks_or_bonds()
    weights = family.weight_matrix(panel)[family.grid.index({"lookback": 63})]
    assert weights[140] == pytest.approx([1.0, 0.0])
    assert weights[-1] == pytest.approx([0.0, 1.0])
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_low_vol_sectors_holds_the_calmest():
    rng = np.random.default_rng(2)
    n = 400
    returns = rng.normal(0, 1, (n, 4)) * np.array([0.02, 0.005, 0.03, 0.006])
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = np.busday_offset(np.datetime64("2020-01-01"), np.arange(n), roll="forward").astype("datetime64[D]")
    panel = Panel(dates, ("A", "B", "C", "D"), close, close, close, close)
    family = low_vol_sectors()
    weights = family.weight_matrix(panel)[family.grid.index({"window": 60, "hold": 2})]
    assert np.all(weights[-1, [0, 2]] == 0.0)  # the two wild ones are never held
    assert weights[-1, [1, 3]].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- no lookahead


@pytest.mark.parametrize("family", SINGLE, ids=lambda f: f.name)
def test_single_asset_ideas_never_use_future_bars(family):
    """The last row is left out: for calendar ideas it depends on the next date,
    which the trading calendar gives in advance."""
    bars = synthetic_bars(900, seed=3, trend=0.002)
    full = family.position_matrix(bars)
    assert np.isin(full, (0.0, 1.0)).all()
    for k in (300, 555, 800):
        assert np.array_equal(family.position_matrix(bars.slice(0, k))[:, : k - 1], full[:, : k - 1])


@pytest.mark.parametrize("family", MULTI + [stocks_or_bonds()], ids=lambda f: f.name)
def test_multi_asset_ideas_never_use_future_bars(family):
    panel = synthetic_panel(900, 9 if family.name != "stocks_or_bonds" else 2, seed=4, trend=0.002)
    full = family.weight_matrix(panel)
    assert (full >= 0).all() and (full.sum(axis=-1) <= 1 + 1e-9).all()
    for k in (300, 555, 800):
        assert np.array_equal(family.weight_matrix(panel.slice(0, k))[:, : k - 1], full[:, : k - 1])


# ------------------------------------------------------- the whole gate path

SMALL = Config(
    symbols=("A0",),
    benchmark="A0",
    source="test",
    start="2000-01-01",
    holdout_months=12,
    costs=Costs(),
    tax=TaxRates(),
    validation=Validation(
        train_bars=504, step_bars=21, n_perm_insample=100, n_perm_walkforward=50, seed=5
    ),
)


@pytest.mark.parametrize("family", SINGLE + MULTI + [stocks_or_bonds()], ids=lambda f: f.name)
def test_every_idea_runs_through_the_gates_and_the_reports_render(family):
    multi = hasattr(family, "weight_matrix")
    if multi:
        data = synthetic_panel(504 + 2 * 252, 2 if family.name == "stocks_or_bonds" else 9, seed=6, trend=0.003)
        benchmark = data.bars("A0")
    else:
        data = benchmark = synthetic_bars(504 + 2 * 252, seed=6, trend=0.003)
    verdict = run_gates(family, data, benchmark, SMALL, {"ideas": 3, "param_sets": 100}, say=lambda _: None)
    assert f"# Verdict: {family.name}" in verdict_markdown(verdict, "synthetic data")
    table = summary_markdown(
        {family.name: verdict},
        [{"idea": family.name, "data": "synthetic", "param_sets": len(family.grid)}],
        {"ideas": 1, "param_sets": len(family.grid)},
    )
    assert "What happened" in table and family.name.split("_")[0][:4].lower() in table.lower()
