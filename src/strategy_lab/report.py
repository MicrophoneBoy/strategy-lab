"""Turn engine output into markdown and JSON. Formatting only, no arithmetic
beyond rounding for display."""

from __future__ import annotations

import json
from pathlib import Path

from .config import repo_root


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def signed_pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:+.{digits}f}%"


def results_dir() -> Path:
    path = repo_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(name: str, value) -> Path:
    path = results_dir() / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    return path


def write_text(name: str, text: str) -> Path:
    path = results_dir() / name
    path.write_text(text, encoding="utf-8")
    return path


METRIC_ROWS = [
    ("Total return", "total_return", pct),
    ("CAGR (return per year)", "cagr", pct),
    ("Sharpe ratio", "sharpe", lambda x: f"{x:.2f}"),
    ("Max drawdown", "max_drawdown", pct),
    ("Profit factor", "profit_factor", lambda x: f"{x:.2f}"),
    ("Trades", "trades", lambda x: f"{x:d}"),
    ("Exposure (share of days invested)", "exposure", pct),
    ("After-tax total return", "after_tax_total_return", pct),
    ("After-tax CAGR", "after_tax_cagr", pct),
]


def metrics_table(strategy: dict, benchmark: dict, strategy_name: str) -> str:
    lines = [f"| Metric | {strategy_name} | Buy-and-hold SPY |", "|---|---:|---:|"]
    for label, key, fmt in METRIC_ROWS:
        if key in strategy:
            lines.append(f"| {label} | {fmt(strategy[key])} | {fmt(benchmark[key])} |")
    return "\n".join(lines)


def versus_lines(versus: dict) -> str:
    lines = [
        f"- Excess CAGR over SPY after costs: **{signed_pct(versus['excess_cagr'])}** a year",
        f"- Sharpe difference: {versus['sharpe_difference']:+.2f}",
    ]
    if "excess_after_tax_cagr" in versus:
        lines.append(
            f"- Excess CAGR over SPY after costs and tax: "
            f"**{signed_pct(versus['excess_after_tax_cagr'])}** a year"
        )
        verdict = "YES" if versus["beats_benchmark_after_costs_and_taxes"] else "NO"
        lines.append(f"- Beats buy-and-hold SPY after costs and tax: **{verdict}**")
    return "\n".join(lines)


def gate_status(gate: dict) -> str:
    if gate["passed"] is None:
        return "not run"
    return "PASS" if gate["passed"] else "FAIL"


def verdict_markdown(verdict: dict, data_note: str) -> str:
    g = verdict["gates"]
    tests = verdict["tests_so_far"]
    out = [
        f"# Verdict: {verdict['idea']}",
        "",
        data_note,
        "",
        f"Tests counted so far: **{tests['ideas']} idea(s), {tests['param_sets']} parameter sets**. "
        "Permutation pass bars are divided by the number of ideas.",
        "",
        "| Gate | Result | Key number |",
        "|---|---|---|",
    ]
    v1 = g["1"]["versus_benchmark"]
    out.append(
        f"| 1. In-sample vs SPY | {gate_status(g['1'])} | excess CAGR "
        f"{signed_pct(v1['excess_cagr'])} after costs, "
        f"{signed_pct(v1['excess_after_tax_cagr'])} after tax |"
    )
    out.append(
        f"| 2. In-sample permutation | {gate_status(g['2'])} | p = {g['2']['p_value']:.4f} "
        f"(needs < {g['2']['alpha']:.4f}) |"
    )
    if g["3"]["passed"] is None:
        out.append("| 3. Walk-forward vs SPY | not run | walk-forward data left untouched |")
    else:
        v3 = g["3"]["versus_benchmark"]
        out.append(
            f"| 3. Walk-forward vs SPY | {gate_status(g['3'])} | excess CAGR "
            f"{signed_pct(v3['excess_cagr'])} after costs, "
            f"{signed_pct(v3['excess_after_tax_cagr'])} after tax |"
        )
    if g["4"]["passed"] is None:
        out.append("| 4. Walk-forward permutation | not run | |")
    else:
        out.append(
            f"| 4. Walk-forward permutation | {gate_status(g['4'])} | p = {g['4']['p_value']:.4f} "
            f"(needs < {g['4']['alpha']:.4f}) |"
        )
    out.append("")
    if verdict["passed_all_gates"]:
        out.append("**Verdict: passed all 4 gates.** It has earned its one run on the locked holdout.")
    else:
        out.append(f"**Verdict: dead at {', '.join(verdict['died_at'])}.** {verdict['reason']}")
    out += [
        "",
        f"## Gate 1 detail ({g['1']['first_bar']} to {g['1']['last_bar']}, "
        f"best parameters {g['1']['best_params']})",
        "",
        metrics_table(g["1"]["strategy"], g["1"]["benchmark"], verdict["idea"]),
        "",
        "## Gate 2 detail",
        "",
        f"- Real optimised log return: {g['2']['real_score']:.4f}",
        f"- Shuffled data, median optimised log return: {g['2']['permuted_median']:.4f}",
        f"- Shuffled data, 95th percentile: {g['2']['permuted_95th']:.4f}",
        f"- {g['2']['permuted_at_least_as_good']} of {g['2']['permutations']} shuffles did at "
        "least as well as the real data",
    ]
    if g["3"]["passed"] is not None:
        out += [
            "",
            f"## Gate 3 detail ({g['3']['first_bar']} to {g['3']['last_bar']}, "
            f"{g['3']['steps']} monthly steps)",
            "",
            metrics_table(g["3"]["strategy"], g["3"]["benchmark"], verdict["idea"]),
        ]
    if g["4"]["passed"] is not None:
        out += [
            "",
            "## Gate 4 detail",
            "",
            f"- Real walk-forward log return: {g['4']['real_score']:.4f}",
            f"- Shuffled data, median: {g['4']['permuted_median']:.4f}",
            f"- {g['4']['permuted_at_least_as_good']} of {g['4']['permutations']} shuffles did at "
            "least as well as the real data",
        ]
    return "\n".join(out) + "\n"
