"""The HTML report: builds from made-up verdicts, is well formed, and shows the numbers it was given."""

from html.parser import HTMLParser

import pytest

from strategy_lab.config import Config, Costs, TaxRates, Validation
from strategy_lab.gates import run_gates
from strategy_lab.html_report import build, sweep_table
from strategy_lab.ideas import sector_rotation
from strategy_lab.strategies import dtfm, sma_crossover
from strategy_lab.sweep import sweep
from strategy_lab.synthetic import synthetic_bars, synthetic_panel

SMALL = Config(
    symbols=("A0",), benchmark="A0", source="test", start="2000-01-01", holdout_months=12,
    costs=Costs(), tax=TaxRates(),
    validation=Validation(train_bars=504, step_bars=21, n_perm_insample=50, n_perm_walkforward=30, seed=5),
)
VOID = {"meta", "br"}
OPTIONAL_END = {"li", "p"}


class Balance(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], 0

    def handle_starttag(self, tag, attrs):
        if tag not in VOID | OPTIONAL_END:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in OPTIONAL_END:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors += 1


def idea(family, data, benchmark, trend_label):
    verdict = run_gates(family, data, benchmark, SMALL, {"ideas": 2, "param_sets": 10}, say=lambda _: None)
    dev, bench = data.slice(0, 504), benchmark.slice(0, 504)
    return {
        "name": family.name,
        "tested_on": trend_label,
        "verdict": verdict,
        "sweep": sweep(family, dev, bench, SMALL.costs).to_dict("records"),
        "grid": family.grid,
    }


@pytest.fixture(scope="module")
def page():
    bars = synthetic_bars(504 + 2 * 252, seed=6, trend=0.004)
    panel = synthetic_panel(504 + 2 * 252, 9, seed=6, trend=0.004)
    ideas = [
        idea(sma_crossover(), bars, bars, "made-up prices"),  # 2 parameters, triangular grid
        idea(dtfm(), panel, panel.bars("A0"), "made-up panel"),  # a single version
        idea(sector_rotation(), panel, panel.bars("A0"), "made-up panel"),  # a multi-asset grid
    ]
    return ideas, build(ideas, {"ideas": 3, "param_sets": 100}, None, None, "made-up data")


def test_page_is_well_formed(page):
    _, text = page
    checker = Balance()
    checker.feed(text)
    assert checker.errors == 0 and checker.stack == []
    assert "nan" not in text.lower().replace("finance", "") and "None" not in text


def test_page_has_a_section_per_idea_and_the_counts(page):
    ideas, text = page
    assert text.count('<section class="idea"') == len(ideas)
    assert "<b>3</b><span>ideas tested</span>" in text
    assert "<b>100</b><span>versions tried</span>" in text


def test_page_shows_the_p_value_it_was_given(page):
    ideas, text = page
    for item in ideas:
        assert f"p = {item['verdict']['gates']['2']['p_value']:.4f}" in text


def test_sweep_grid_marks_exactly_one_best_cell_and_leaves_gaps_blank():
    family = sma_crossover()
    rows = [{**g, "excess_cagr": 0.001 * i - 0.02} for i, g in enumerate(family.grid)]
    table = sweep_table(rows, family.grid)
    assert table.count(' best"') == 1  # the class on one cell (the caption also says "best")
    assert table.count('class="empty"') == 100 - len(family.grid)  # fast must be below slow
    assert "+7.0" not in table and "-2.0" in table
