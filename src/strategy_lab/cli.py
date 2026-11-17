"""Command line: `uv run lab <command>`.

    fetch         download daily bars, cache them, lock the holdout
    check-alpaca  read the Alpaca paper account (needs paper keys in .env)
    baseline      SMA crossover on SPY vs buy-and-hold, plus the parameter sweep
    tester        test the tester: random signals must fail, a planted edge must pass
    gates         run an idea through the 4 gates (--idea sma_crossover or dtfm)
    paper-compare the real DTFM paper run vs SPY over the same dates, next to the lab verdict
"""

from __future__ import annotations

import argparse
import json

from . import html_report, ledger, report
from .config import Config, load_config
from .data import (
    add_to_cache,
    fetch_and_cache,
    load_dev_bars,
    load_dev_panel,
    paper_keys,
    read_lock,
    read_meta,
)
from .engine import backtest, buy_and_hold, compare
from .gates import run_gates
from .ideas import (
    breakout,
    low_vol_sectors,
    mean_reversion,
    sector_rotation,
    stocks_or_bonds,
    turn_of_month,
    volatility_filter,
)
from .paper import compare_paper, comparison_markdown, load_paper_run, spy_reference_prices
from .strategies import dtfm, sma_crossover
from .sweep import heatmap, plateau_check, sweep
from .tester import run_tester, tester_markdown

BASELINE = {"fast": 50, "slow": 200}
SECTORS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB")
DTFM_UNIVERSE = ("QQQ", *SECTORS)
SECTOR_IDEAS = {"sector_rotation": sector_rotation, "low_vol_sectors": low_vol_sectors}
SPY_IDEAS = {
    "turn_of_month": turn_of_month,
    "volatility_filter": volatility_filter,
    "mean_reversion": mean_reversion,
}
IDEAS = (
    "sma_crossover", "dtfm", "sector_rotation", "stocks_or_bonds", "low_vol_sectors",
    "turn_of_month", "volatility_filter", "mean_reversion", "breakout",
)


def data_note(cfg: Config, symbol: str) -> str:
    meta = read_meta()
    s = meta["symbols"][symbol]
    return (
        f"Data: {symbol} daily bars from {meta['source']}, adjusted for splits and dividends, "
        f"{s['dev_first']} to {s['dev_last']} ({s['dev_bars']} bars, fingerprint "
        f"`{meta['dev_fingerprint']}`). Costs: {cfg.costs.fee_bps:g} bps fee plus "
        f"{cfg.costs.slippage_bps:g} bps slippage per position change. Tax estimate: "
        f"{cfg.tax.short_term:.0%} on gains held a year or less, {cfg.tax.long_term:.0%} on longer "
        f"holds. Bars from {meta['holdout_start']} on are the locked holdout and were not read."
    )


def panel_note(cfg: Config, description: str, panel, extra: tuple[str, ...] = ()) -> str:
    meta = read_meta()
    fingerprint = f"`{meta['dev_fingerprint']}`" + "".join(
        f", {s} `{meta['symbols'][s]['fingerprint']}`" for s in extra
    )
    return (
        f"Data: {description}, daily bars from {meta['source']}, adjusted for splits and "
        f"dividends, on the dates they all share: {panel.dates[0]} to {panel.dates[-1]} "
        f"({len(panel)} bars, fingerprint {fingerprint}). Costs: "
        f"{cfg.costs.fee_bps:g} bps fee plus {cfg.costs.slippage_bps:g} bps slippage on whatever "
        f"changes hands. Tax estimate: {cfg.tax.short_term:.0%} on gains held a year or less, "
        f"{cfg.tax.long_term:.0%} on longer holds. Cash earns 0. Bars from "
        f"{meta['holdout_start']} on are the locked holdout and were not read. For a portfolio, "
        "\"Trades\" counts the days on which something was traded."
    )


def cmd_fetch(cfg: Config, args) -> None:
    meta = fetch_and_cache(cfg, args.source)
    print(f"source: {meta['source']}   holdout locked from: {meta['holdout_start']}")
    print(f"development data fingerprint: {meta['dev_fingerprint']}")
    for symbol, s in meta["symbols"].items():
        print(
            f"  {symbol:<4} {s['dev_bars']:>5} dev bars  {s['dev_first']} to {s['dev_last']}"
            f"   {s['holdout_bars']:>3} holdout bars (locked)"
        )


def cmd_check_alpaca(cfg: Config, args) -> None:
    from alpaca.trading.client import TradingClient

    key, secret = paper_keys()
    client = TradingClient(key, secret, paper=True)  # paper is hard-coded on purpose
    account = client.get_account()
    positions = client.get_all_positions()
    print("Alpaca PAPER account reached.")
    print(f"  status: {account.status}")
    print(f"  equity: {account.equity} {account.currency}")
    print(f"  open positions: {len(positions)}")
    for p in positions:
        print(f"    {p.symbol}: {p.qty}")


def cmd_baseline(cfg: Config, args) -> None:
    train = cfg.validation.train_bars
    dev = load_dev_bars(cfg.benchmark).slice(0, train)
    family = sma_crossover()
    k = family.grid.index(BASELINE)

    strategy = backtest(dev, family.positions(dev, k), cfg.costs, cfg.tax)
    benchmark = buy_and_hold(dev, cfg.costs, cfg.tax)
    versus = compare(strategy.metrics, benchmark.metrics)

    table = sweep(family, dev, dev, cfg.costs)
    plateau = plateau_check(table, "fast", "slow")
    window = f"{strategy.dates[0]} to {strategy.dates[-1]}"
    heatmap(
        table, "fast", "slow", report.results_dir() / "baseline_heatmap.png",
        f"SMA crossover on SPY, {window}: excess return over buy-and-hold",
    )
    ledger.record_test(family.name, family.grid, f"{cfg.benchmark} development window")
    tests = ledger.counts()

    name = f"SMA {BASELINE['fast']}/{BASELINE['slow']}"
    best = plateau["best"]
    text = "\n".join(
        [
            f"# Baseline: {name} crossover on SPY",
            "",
            data_note(cfg, cfg.benchmark),
            "",
            f"Window: the development window, {window} ({strategy.metrics['bars']} bars). "
            "Walk-forward data after it is single use and was left alone.",
            "",
            report.metrics_table(strategy.metrics, benchmark.metrics, name),
            "",
            report.versus_lines(versus),
            "",
            f"## Parameter sweep ({len(family.grid)} parameter sets)",
            "",
            "![heatmap](baseline_heatmap.png)",
            "",
            f"- Best cell: fast {best['fast']}, slow {best['slow']}, excess CAGR "
            f"{report.signed_pct(best['excess_cagr'])} a year",
            f"- Average of its {plateau['neighbours']} neighbours: "
            f"{report.signed_pct(plateau['neighbour_mean'])} a year",
            f"- Shape: **{plateau['shape']}** (spike score {plateau['spike_score']:.2f}; above 1 "
            "means the best cell stands alone)",
            f"- Share of the grid that beat SPY: "
            f"{report.pct(plateau['share_of_grid_beating_benchmark'], 0)}",
            "",
            f"Tests counted so far: {tests['ideas']} idea(s), {tests['param_sets']} parameter sets.",
            "",
        ]
    )
    report.write_text("baseline.md", text)
    report.write_json(
        "baseline.json",
        {
            "window": window,
            "strategy": strategy.metrics,
            "benchmark": benchmark.metrics,
            "versus_benchmark": versus,
            "plateau": plateau,
            "sweep": json.loads(table.to_json(orient="records")),
        },
    )
    print(text)


def cmd_tester(cfg: Config, args) -> None:
    dev = load_dev_bars(cfg.benchmark).slice(0, cfg.validation.train_bars)
    result = run_tester(cfg, dev)
    text = tester_markdown(result)
    report.write_text("tester.md", text)
    report.write_json("tester.json", result)
    print(text)


def load_idea(name: str, cfg: Config):
    """Return (family, its data, benchmark bars on the same dates, ledger label, data note)."""
    if name == "sma_crossover":
        bars = load_dev_bars(cfg.benchmark)
        label = f"{cfg.benchmark} development window"
        return sma_crossover(), bars, bars, label, data_note(cfg, cfg.benchmark)
    if name == "dtfm":
        panel, benchmark = load_dev_panel(DTFM_UNIVERSE, cfg.benchmark)
        note = panel_note(cfg, "QQQ plus 9 sector ETFs (" + ", ".join(DTFM_UNIVERSE) + ")", panel)
        return dtfm(), panel, benchmark, "QQQ plus 9 sector ETFs", note
    if name in SECTOR_IDEAS:
        panel, benchmark = load_dev_panel(SECTORS, cfg.benchmark)
        note = panel_note(cfg, "9 sector ETFs (" + ", ".join(SECTORS) + ")", panel)
        return SECTOR_IDEAS[name](), panel, benchmark, "9 sector ETFs", note
    if name == "stocks_or_bonds":
        panel, benchmark = load_dev_panel(("SPY", "IEF"), cfg.benchmark)
        note = panel_note(cfg, "SPY and IEF", panel, extra=("IEF",))
        return stocks_or_bonds(), panel, benchmark, "SPY and IEF", note
    if name in SPY_IDEAS:
        bars = load_dev_bars(cfg.benchmark)
        return SPY_IDEAS[name](), bars, bars, "SPY", panel_note(cfg, "SPY", bars)
    if name == "breakout":
        panel, benchmark = load_dev_panel(("QQQ",), cfg.benchmark)
        bars = panel.bars("QQQ")
        return breakout(), bars, benchmark, "QQQ", panel_note(cfg, "QQQ", bars)
    raise ValueError(f"unknown idea: {name}")


def cmd_gates(cfg: Config, args) -> None:
    family, data, benchmark, label, note = load_idea(args.idea, cfg)
    ledger.record_test(family.name, family.grid, label)
    verdict = run_gates(family, data, benchmark, cfg, ledger.counts())
    verdict["tested_on"] = label
    ledger.record_verdict(family.name, verdict)
    text = report.verdict_markdown(verdict, note)
    report.write_text(f"verdict_{family.name}.md", text)
    report.write_json(f"verdict_{family.name}.json", verdict)
    print(text)


def cmd_add_data(cfg: Config, args) -> None:
    meta = add_to_cache(cfg, tuple(args.symbols))
    for symbol in args.symbols:
        s = meta["symbols"][symbol]
        print(
            f"  {symbol:<4} {s['dev_bars']:>5} dev bars  {s['dev_first']} to {s['dev_last']}"
            f"   {s['holdout_bars']:>3} holdout bars (locked)   fingerprint {s['fingerprint']}"
        )


def cmd_summary(cfg: Config, args) -> None:
    text = report.summary_markdown(ledger.verdicts(), ledger.test_order(), ledger.counts())
    report.write_text("ideas.md", text)
    print(text)


def cmd_report(cfg: Config, args) -> None:
    """Build results/report.html from the ledger, the result files and fresh sweeps.
    The sweeps rerun only versions already counted, on the development window."""
    verdicts, train, ideas, seen = ledger.verdicts(), cfg.validation.train_bars, [], set()
    for test in ledger.test_order():
        name = test["idea"]
        if name in seen or name not in verdicts:
            continue
        seen.add(name)
        family, data, benchmark, label, _ = load_idea(name, cfg)
        table = sweep(family, data.slice(0, train), benchmark.slice(0, train), cfg.costs)
        ideas.append(
            {
                "name": name,
                "tested_on": label.replace(" development window", ""),
                "verdict": verdicts[name],
                "sweep": table.to_dict("records"),
                "grid": family.grid,
            }
        )
        print(f"  swept {name}")

    def load(name: str):
        path = report.results_dir() / name
        return json.loads(path.read_text()) if path.exists() else None

    meta = read_meta()
    note = (
        f"Daily bars from {meta['source']}, adjusted for splits and dividends, up to "
        f"{meta['symbols'][cfg.benchmark]['dev_last']}. Costs: {cfg.costs.fee_bps:g} bps fee plus "
        f"{cfg.costs.slippage_bps:g} bps slippage on whatever changes hands. Tax estimate: "
        f"{cfg.tax.short_term:.0%} on gains held a year or less, {cfg.tax.long_term:.0%} on longer "
        "holds. Cash earns 0. Trades happen at the close that produced the signal."
    )
    page = html_report.build(
        ideas, ledger.counts(), load("tester.json"), load("dtfm_paper_comparison.json"), note
    )
    path = report.write_text("report.html", page)
    print(f"wrote {path}")


def cmd_paper_compare(cfg: Config, args) -> None:
    run = load_paper_run()
    result = compare_paper(run, spy_reference_prices(run["start_date"], run["end_date"]))
    text = comparison_markdown(run, result, ledger.verdicts().get(run["strategy"]))
    report.write_text("dtfm_paper_comparison.md", text)
    report.write_json("dtfm_paper_comparison.json", {"paper_run": run, **result})
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="lab", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download and cache daily bars, lock the holdout")
    fetch.add_argument("--source", choices=["yahoo", "alpaca"], default=None)
    sub.add_parser("check-alpaca", help="read the Alpaca paper account")
    sub.add_parser("baseline", help="baseline strategy and sweep vs buy-and-hold SPY")
    sub.add_parser("tester", help="test the tester")
    gates = sub.add_parser("gates", help="run an idea through the 4 gates")
    gates.add_argument("--idea", choices=IDEAS, default="sma_crossover")
    sub.add_parser("paper-compare", help="the real DTFM paper run vs SPY and the lab verdict")
    add = sub.add_parser("add-data", help="download extra symbols into the existing cache")
    add.add_argument("symbols", nargs="+")
    sub.add_parser("summary", help="one table with every idea's verdict in plain language")
    sub.add_parser("report", help="build results/report.html: every idea, sweep, gate and verdict")
    args = parser.parse_args()
    cfg = load_config()
    if read_lock() is None and args.command not in ("fetch", "check-alpaca"):
        parser.error("no data yet: run `uv run lab fetch` first")
    {
        "fetch": cmd_fetch,
        "check-alpaca": cmd_check_alpaca,
        "baseline": cmd_baseline,
        "tester": cmd_tester,
        "gates": cmd_gates,
        "paper-compare": cmd_paper_compare,
        "add-data": cmd_add_data,
        "summary": cmd_summary,
        "report": cmd_report,
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()
