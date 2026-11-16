"""The 4 gates an idea must pass, in order. It dies at the first one it fails.

1. In-sample: on the first 4 years, does the best parameter set beat
   buy-and-hold after costs and after tax?
2. In-sample permutation test: is that result better than luck?
3. Walk-forward: does it still beat buy-and-hold out-of-sample?
4. Walk-forward permutation test: is the out-of-sample result better than luck?

Gates 1 and 2 only read the development window. Walk-forward data is single
use, so gates 3 and 4 run only for an idea that passed both.

`family` is a single-asset Family with Bars, or a PortfolioFamily with a Panel.
The benchmark is always single-asset Bars on the same dates.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .data import Bars
from .engine import buy_and_hold, compare
from .ledger import adjusted_alpha
from .metrics import BARS_PER_YEAR
from .permutation import best_in_sample, insample_permutation_test
from .strategies import Family
from .walkforward import walk_forward, walk_forward_permutation_test


def gate1(family: Family, dev: Bars, benchmark_dev: Bars, cfg: Config) -> dict:
    _, k = best_in_sample(family, dev, cfg.costs)
    strategy = family.backtest(dev, family.decisions(dev)[k], cfg.costs, cfg.tax).metrics
    benchmark = buy_and_hold(benchmark_dev, cfg.costs, cfg.tax).metrics
    versus = compare(strategy, benchmark)
    return {
        "passed": bool(
            versus["beats_benchmark_after_costs"] and versus["beats_benchmark_after_costs_and_taxes"]
        ),
        "first_bar": str(dev.dates[1]),
        "last_bar": str(dev.dates[-1]),
        "best_params": family.grid[k],
        "strategy": strategy,
        "benchmark": benchmark,
        "versus_benchmark": versus,
    }


def gate2(family: Family, dev: Bars, cfg: Config, ideas: int) -> dict:
    v = cfg.validation
    result = insample_permutation_test(family, dev, cfg.costs, v.n_perm_insample, v.seed)
    alpha = adjusted_alpha(v.alpha_insample, ideas)
    return {"passed": bool(result["p_value"] < alpha), "alpha": alpha, **result}


def gate3(family: Family, bars: Bars, benchmark_bars: Bars, cfg: Config) -> dict:
    v = cfg.validation
    result = walk_forward(
        family, bars, benchmark_bars, cfg.costs, cfg.tax, v.train_bars, v.step_bars
    )
    versus = result["versus_benchmark"]
    return {
        "passed": bool(
            versus["beats_benchmark_after_costs"] and versus["beats_benchmark_after_costs_and_taxes"]
        ),
        **result,
    }


def gate4(family: Family, bars: Bars, cfg: Config, ideas: int) -> dict:
    v = cfg.validation
    result = walk_forward_permutation_test(
        family, bars, cfg.costs, v.train_bars, v.step_bars, v.n_perm_walkforward, v.seed + 1
    )
    years = (len(bars) - v.train_bars) / BARS_PER_YEAR
    base = v.alpha_walkforward if years >= 2 else v.alpha_walkforward_short
    alpha = adjusted_alpha(base, ideas)
    return {
        "passed": bool(result["p_value"] < alpha),
        "alpha": alpha,
        "out_of_sample_years": years,
        **result,
    }


NOT_RUN = {"passed": None, "status": "not run: the idea had already died"}

REASONS = {
    1: "Even with the best parameters in hindsight, it did not beat buy-and-hold on the "
    "development window after costs and tax.",
    2: "Its in-sample result is no better than what the same optimisation finds on shuffled "
    "prices, so it looks like luck.",
    3: "Out-of-sample, it did not beat buy-and-hold after costs and tax.",
    4: "Its out-of-sample result is no better than what walk-forward finds on shuffled prices.",
}


def run_gates(
    family: Family, bars: Bars, benchmark_bars: Bars, cfg: Config, tests: dict, say=print
) -> dict:
    """Run an idea through the gates and return its verdict."""
    if not np.array_equal(bars.dates, benchmark_bars.dates):
        raise ValueError("strategy and benchmark must cover the same bars")
    train = cfg.validation.train_bars
    dev, benchmark_dev = bars.slice(0, train), benchmark_bars.slice(0, train)
    ideas = tests["ideas"]
    gates: dict = {}

    say("gate 1: in-sample ...")
    gates["1"] = gate1(family, dev, benchmark_dev, cfg)
    say("gate 2: in-sample permutation test ...")
    gates["2"] = gate2(family, dev, cfg, ideas)

    if gates["1"]["passed"] and gates["2"]["passed"]:
        say("gate 3: walk-forward ...")
        gates["3"] = gate3(family, bars, benchmark_bars, cfg)
        if gates["3"]["passed"]:
            say("gate 4: walk-forward permutation test ...")
            gates["4"] = gate4(family, bars, cfg, ideas)
        else:
            gates["4"] = dict(NOT_RUN)
    else:
        gates["3"] = dict(NOT_RUN)
        gates["4"] = dict(NOT_RUN)

    failed = [int(g) for g in ("1", "2", "3", "4") if gates[g]["passed"] is False]
    return {
        "idea": family.name,
        "rationale": family.rationale,
        "param_sets": len(family.grid),
        "tests_so_far": tests,
        "gates": gates,
        "passed_all_gates": not failed and all(gates[g]["passed"] for g in gates),
        "died_at": [f"gate {g}" for g in failed],
        "reason": " ".join(REASONS[g] for g in failed),
    }
