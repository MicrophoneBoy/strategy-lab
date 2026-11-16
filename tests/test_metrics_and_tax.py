"""Metrics and the tax estimate, each checked against numbers worked by hand."""

import math

import numpy as np
import pytest

from strategy_lab.config import TaxRates
from strategy_lab.metrics import count_trades, max_drawdown, summarize
from strategy_lab.tax import after_tax, settle_year

RATES = TaxRates(short_term=0.22, long_term=0.15)


def days(*dates):
    return np.array(dates, dtype="datetime64[D]")


def test_max_drawdown():
    # Equity: 1 -> 1.1 -> 0.55 -> 0.66. Worst fall is 1.1 to 0.55 = -50%.
    net = np.log([1.1, 0.5, 1.2])
    assert max_drawdown(net) == pytest.approx(-0.5)


def test_cagr_and_total_return():
    # 504 bars (2 years) that compound to +21% total is 10% a year.
    net = np.full(504, math.log(1.21) / 504)
    m = summarize(net, np.ones(504))
    assert m["total_return"] == pytest.approx(0.21)
    assert m["cagr"] == pytest.approx(0.10)
    assert m["years"] == pytest.approx(2.0)


def test_sharpe():
    # Daily returns alternate +2% and -1%: mean 0.5%, sample st. dev. known.
    simple = np.tile([0.02, -0.01], 126)
    m = summarize(np.log1p(simple), np.ones(252))
    expected = simple.mean() / simple.std(ddof=1) * math.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected)


def test_profit_factor():
    net = np.array([0.03, -0.01, 0.02, -0.01])  # gains 0.05, losses 0.02
    assert summarize(net, np.ones(4))["profit_factor"] == pytest.approx(2.5)


def test_trades_and_exposure():
    held = np.array([0, 1, 1, 0, -1, -1, 1], dtype=float)
    # Opens: flat to long, flat to short, short to long.
    assert count_trades(held) == 3
    assert summarize(np.zeros(7), held)["exposure"] == pytest.approx(5 / 7)


def test_long_term_gain_is_taxed_at_the_long_term_rate():
    # Doubles over 17 months in one trade: gain 1.0, tax 15% -> keeps 0.85.
    result = after_tax(
        net=np.log([1.5, 4 / 3]),
        held=np.ones(2),
        dates=days("2020-01-02", "2021-06-01"),
        rates=RATES,
    )
    assert result["after_tax_total_return"] == pytest.approx(0.85)
    assert result["long_term_trades"] == 1 and result["short_term_trades"] == 0


def test_short_term_gain_is_taxed_as_ordinary_income():
    # Same doubling inside 5 months: tax 22% -> keeps 0.78.
    result = after_tax(
        net=np.log([1.5, 4 / 3]),
        held=np.ones(2),
        dates=days("2020-01-02", "2020-06-01"),
        rates=RATES,
    )
    assert result["after_tax_total_return"] == pytest.approx(0.78)
    assert result["short_term_trades"] == 1


def test_losses_carry_forward_to_the_next_year():
    # 2020: lose half (1.0 -> 0.5). 2021: double back (0.5 -> 1.0).
    # The 2021 gain of 0.5 is fully offset by the 2020 loss, so no tax at all.
    result = after_tax(
        net=np.array([math.log(0.5), 0.0, math.log(2.0), 0.0]),
        held=np.array([1.0, 0.0, 1.0, 0.0]),
        dates=days("2020-03-02", "2020-03-03", "2021-03-01", "2021-03-02"),
        rates=RATES,
    )
    assert result["tax_paid"] == pytest.approx(0.0)
    assert result["after_tax_total_return"] == pytest.approx(0.0)


def test_gains_are_taxed_every_year_they_are_realised():
    # +10% in 2020 and +10% in 2021, each a separate short-term trade.
    # 2020: 1.10, tax 0.022 -> 1.078. 2021: 1.1858, tax 22% of 0.1078 -> 1.162084.
    result = after_tax(
        net=np.array([math.log(1.1), 0.0, math.log(1.1), 0.0]),
        held=np.array([1.0, 0.0, 1.0, 0.0]),
        dates=days("2020-03-02", "2020-03-03", "2021-03-01", "2021-03-02"),
        rates=RATES,
    )
    assert result["after_tax_total_return"] == pytest.approx(0.162084)


def test_exit_cost_is_paid_when_the_final_position_is_sold():
    result = after_tax(
        net=np.zeros(2),
        held=np.ones(2),
        dates=days("2020-01-02", "2020-01-03"),
        rates=RATES,
        exit_cost_rate=0.001,
    )
    assert result["after_tax_total_return"] == pytest.approx(-0.001)


@pytest.mark.parametrize(
    "short, long, carry, tax, new_carry",
    [
        (10.0, 20.0, 0.0, 0.22 * 10 + 0.15 * 20, 0.0),  # both gains
        (-5.0, 3.0, 0.0, 0.0, -2.0),  # short loss wipes out the long gain
        (5.0, -3.0, 0.0, 0.22 * 2, 0.0),  # long loss reduces the short gain
        (-4.0, -6.0, 0.0, 0.0, -10.0),  # all losses carry forward
        (10.0, 0.0, -4.0, 0.22 * 6, 0.0),  # last year's loss used first
    ],
)
def test_settle_year(short, long, carry, tax, new_carry):
    got_tax, got_carry = settle_year(short, long, carry, RATES)
    assert got_tax == pytest.approx(tax)
    assert got_carry == pytest.approx(new_carry)
