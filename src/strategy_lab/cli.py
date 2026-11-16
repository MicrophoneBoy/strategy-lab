"""Command line: `uv run lab <command>`.

    fetch         download daily bars, cache them, lock the holdout
    check-alpaca  read the Alpaca paper account (needs paper keys in .env)
    baseline      SMA crossover on SPY vs buy-and-hold, plus the parameter sweep
    tester        test the tester: random signals must fail, a planted edge must pass
    gates         run the baseline idea through the 4 gates
"""

from __future__ import annotations

import argparse
import json

from . import ledger, report
from .config import Config, load_config
from .data import fetch_and_cache, load_dev_bars, paper_keys, read_lock, read_meta
from .engine import backtest, buy_and_hold, compare
from .gates import run_gates
from .strategies import sma_crossover
from .sweep import heatmap, plateau_check, sweep
from .tester import run_tester, tester_markdown

BASELINE = {"fast": 50, "slow": 200}


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


def cmd_gates(cfg: Config, args) -> None:
    bars = load_dev_bars(cfg.benchmark)
    family = sma_crossover()
    ledger.record_test(family.name, family.grid, f"{cfg.benchmark} development window")
    verdict = run_gates(family, bars, bars, cfg, ledger.counts())
    ledger.record_verdict(family.name, verdict)
    text = report.verdict_markdown(verdict, data_note(cfg, cfg.benchmark))
    report.write_text(f"verdict_{family.name}.md", text)
    report.write_json(f"verdict_{family.name}.json", verdict)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="lab", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download and cache daily bars, lock the holdout")
    fetch.add_argument("--source", choices=["yahoo", "alpaca"], default=None)
    sub.add_parser("check-alpaca", help="read the Alpaca paper account")
    sub.add_parser("baseline", help="baseline strategy and sweep vs buy-and-hold SPY")
    sub.add_parser("tester", help="test the tester")
    sub.add_parser("gates", help="run the baseline idea through the 4 gates")
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
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()
