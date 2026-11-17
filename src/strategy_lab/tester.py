"""Test the tester. Before any verdict is trusted, the gates must:

- FAIL a random signal (best-of-many coin flips looks good in-sample),
- FAIL a real strategy on data with no edge in it (a random walk),
- PASS a real strategy on data with a planted edge,
- and raise false alarms on no-edge data about as often as the pass bar says.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .data import Bars
from .engine import backtest, buy_and_hold, compare
from .permutation import best_in_sample, insample_permutation_test
from .strategies import dtfm, dual_momentum, random_portfolio, random_signal, sma_crossover
from .synthetic import synthetic_bars, synthetic_panel
from .walkforward import walk_forward, walk_forward_permutation_test

PLANTED_TREND = 0.003  # daily drift of the planted edge, against 1% daily noise
CALIBRATION_SERIES = 100
CALIBRATION_PERMUTATIONS = 200
POWER_TRENDS = (0.001, 0.002, 0.003)  # how small an edge can gate 2 still see?
POWER_SERIES = 20
PANEL_ASSETS = 8  # size of the synthetic multi-asset panels
TESTER_INSAMPLE_SHUFFLES = 1000  # fixed here so the tester does not change with config.toml
TESTER_WALKFORWARD_SHUFFLES = 200


def run_tester(cfg: Config, spy_dev_window: Bars, say=print) -> dict:
    v = cfg.validation
    n = v.train_bars + 5 * 252  # one training window plus 5 walk-forward years
    random_walk = synthetic_bars(n, seed=101, trend=0.0)
    planted = synthetic_bars(n, seed=202, trend=PLANTED_TREND)
    coin_flips, trend_follower = random_signal(), sma_crossover()
    checks = []

    def permutation_check(name, data_name, family, bars, gate, expect_pass):
        say(f"  {name} on {data_name}, gate {gate} ...")
        if gate == 2:
            result = insample_permutation_test(
                family, bars.slice(0, v.train_bars), cfg.costs, TESTER_INSAMPLE_SHUFFLES, v.seed
            )
            alpha = v.alpha_insample
        else:
            result = walk_forward_permutation_test(
                family, bars, cfg.costs, v.train_bars, v.step_bars, TESTER_WALKFORWARD_SHUFFLES, v.seed + 1
            )
            alpha = v.alpha_walkforward
        passed = result["p_value"] < alpha
        checks.append(
            {
                "strategy": name,
                "data": data_name,
                "gate": gate,
                "p_value": result["p_value"],
                "alpha": alpha,
                "gate_result": "pass" if passed else "fail",
                "expected": "pass" if expect_pass else "fail",
                "ok": bool(passed == expect_pass),
            }
        )

    permutation_check("random signal", "real SPY development window", coin_flips, spy_dev_window, 2, False)
    permutation_check("random signal", "random walk", coin_flips, random_walk, 4, False)
    permutation_check("SMA crossover", "random walk", trend_follower, random_walk, 2, False)
    permutation_check("SMA crossover", "random walk", trend_follower, random_walk, 4, False)
    permutation_check("SMA crossover", "planted trend", trend_follower, planted, 2, True)
    permutation_check("SMA crossover", "planted trend", trend_follower, planted, 4, True)

    # The same checks for the multi-asset engine.
    walk_panel = synthetic_panel(n, PANEL_ASSETS, seed=31, trend=0.0)
    planted_panel = synthetic_panel(n, PANEL_ASSETS, seed=32, trend=PLANTED_TREND)
    walks, trends = f"{PANEL_ASSETS} random walks", f"{PANEL_ASSETS} planted trends"
    permutation_check("random portfolio", walks, random_portfolio(), walk_panel, 2, False)
    permutation_check("random portfolio", walks, random_portfolio(), walk_panel, 4, False)
    permutation_check("dual momentum", walks, dual_momentum(), walk_panel, 2, False)
    permutation_check("dual momentum", walks, dual_momentum(), walk_panel, 4, False)
    permutation_check("dual momentum", trends, dual_momentum(), planted_panel, 2, True)
    permutation_check("dual momentum", trends, dual_momentum(), planted_panel, 4, True)
    # DTFM has one fixed parameter set, so these also check the gates when nothing is optimised.
    permutation_check("DTFM", walks, dtfm(), walk_panel, 2, False)
    permutation_check("DTFM", walks, dtfm(), walk_panel, 4, False)
    permutation_check("DTFM", trends, dtfm(), planted_panel, 2, True)
    permutation_check("DTFM", trends, dtfm(), planted_panel, 4, True)

    # Gates 1 and 3 on the planted edge: it must also beat buy-and-hold.
    dev = planted.slice(0, v.train_bars)
    _, k = best_in_sample(trend_follower, dev, cfg.costs)
    in_sample = compare(
        backtest(dev, trend_follower.positions(dev, k), cfg.costs, cfg.tax).metrics,
        buy_and_hold(dev, cfg.costs, cfg.tax).metrics,
    )
    out_of_sample = walk_forward(
        trend_follower, planted, planted, cfg.costs, cfg.tax, v.train_bars, v.step_bars
    )["versus_benchmark"]
    planted_beats = {
        "gate_1_excess_after_tax_cagr": in_sample["excess_after_tax_cagr"],
        "gate_3_excess_after_tax_cagr": out_of_sample["excess_after_tax_cagr"],
        "ok": bool(
            in_sample["beats_benchmark_after_costs_and_taxes"]
            and out_of_sample["beats_benchmark_after_costs_and_taxes"]
        ),
    }

    # Calibration: on no-edge data a 5% test should raise a false alarm about 5% of the time.
    say(f"  calibration: gate 2 on {CALIBRATION_SERIES} separate random walks ...")
    p_values = np.array(
        [
            insample_permutation_test(
                trend_follower,
                synthetic_bars(v.train_bars, seed=1000 + i, trend=0.0),
                cfg.costs,
                CALIBRATION_PERMUTATIONS,
                v.seed + i,
            )["p_value"]
            for i in range(CALIBRATION_SERIES)
        ]
    )
    false_alarm_rate = float(np.mean(p_values < 0.05))
    calibration = {
        "series": CALIBRATION_SERIES,
        "permutations_each": CALIBRATION_PERMUTATIONS,
        "false_alarm_rate_at_5_percent": false_alarm_rate,
        "false_alarm_rate_at_1_percent": float(np.mean(p_values < 0.01)),
        "median_p_value": float(np.median(p_values)),
        # With 100 series and a true 5% rate, more than 10 false alarms is a 1-in-100 event.
        "ok": bool(false_alarm_rate <= 0.10),
    }

    # Power: how often does gate 2 catch a planted edge, as the edge gets smaller?
    power = []
    for trend in POWER_TRENDS:
        say(f"  power: gate 2 on {POWER_SERIES} series with a {trend * 100:.1f}% a day trend ...")
        detected = sum(
            insample_permutation_test(
                trend_follower,
                synthetic_bars(v.train_bars, seed=3000 + i, trend=trend),
                cfg.costs,
                CALIBRATION_PERMUTATIONS,
                v.seed + i,
            )["p_value"]
            < v.alpha_insample
            for i in range(POWER_SERIES)
        )
        power.append({"trend_per_day": trend, "series": POWER_SERIES, "detected": int(detected)})

    return {
        "planted_trend_per_day": PLANTED_TREND,
        "checks": checks,
        "planted_edge_beats_buy_and_hold": planted_beats,
        "calibration": calibration,
        "power": power,
        "all_ok": bool(all(c["ok"] for c in checks) and planted_beats["ok"] and calibration["ok"]),
    }


def tester_markdown(result: dict) -> str:
    lines = [
        "# Test the tester",
        "",
        "The gates are only worth trusting if they fail what should fail and pass what should pass.",
        "",
        "| Strategy | Data | Gate | p-value | Pass bar | Gate says | Expected | Check |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for c in result["checks"]:
        lines.append(
            f"| {c['strategy']} | {c['data']} | {c['gate']} | {c['p_value']:.4f} | "
            f"< {c['alpha']:.2f} | {c['gate_result']} | {c['expected']} | "
            f"{'OK' if c['ok'] else 'WRONG'} |"
        )
    b, cal = result["planted_edge_beats_buy_and_hold"], result["calibration"]
    lines += [
        "",
        f"Planted edge: a drift of {result['planted_trend_per_day'] * 100:.1f}% a day that flips "
        "direction about every 80 days, against 1% daily noise.",
        "",
        f"- Planted edge vs buy-and-hold, after costs and tax: in-sample "
        f"{b['gate_1_excess_after_tax_cagr'] * 100:+.2f}% a year, walk-forward "
        f"{b['gate_3_excess_after_tax_cagr'] * 100:+.2f}% a year. "
        f"{'OK' if b['ok'] else 'WRONG'}",
        f"- Calibration: on {cal['series']} separate random walks, gate 2 raised a false alarm at "
        f"the 5% level {cal['false_alarm_rate_at_5_percent'] * 100:.0f}% of the time and at the 1% "
        f"level {cal['false_alarm_rate_at_1_percent'] * 100:.0f}% of the time "
        f"(median p-value {cal['median_p_value']:.2f}; about 5%, 1% and 0.50 expected). "
        f"{'OK' if cal['ok'] else 'WRONG'}",
        "",
        "## How small an edge can the permutation gate see?",
        "",
        "Gate 2 on separate planted-trend series, at the 1% pass bar. A weak real edge can be "
        "missed, so a FAIL only means the result was not shown to be better than luck.",
        "",
        "| Planted trend per day | Series | Detected |",
        "|---:|---:|---:|",
        *[
            f"| {p['trend_per_day'] * 100:.1f}% | {p['series']} | {p['detected']} |"
            for p in result["power"]
        ],
        "",
        "**Tester verdict: "
        + (
            "all checks behave as expected"
            if result["all_ok"]
            else "AT LEAST ONE CHECK IS WRONG, do not trust any verdict"
        )
        + ".**",
        "",
    ]
    return "\n".join(lines)
