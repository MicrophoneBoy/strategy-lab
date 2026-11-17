"""Compare the real DTFM paper run with SPY over the same dates, next to the
lab's verdict.

The paper run happened inside the locked holdout period. This module reads
exactly two SPY daily bars (the first and last day of the run), straight from
Alpaca and never from the research cache. They are a benchmark for a result
that already exists; no strategy is fitted or scored on them. Each read is
written to ledger/holdout_reads.json.
"""

from __future__ import annotations

import tomllib

import pandas as pd

from . import ledger
from .config import repo_root
from .data import paper_keys
from .report import gate_status, signed_pct


def load_paper_run() -> dict:
    with open(repo_root() / "paper" / "dtfm_paper_run.toml", "rb") as f:
        return tomllib.load(f)


def _session_is_over(day: str) -> bool:
    """True once the day's closing bar is final and old enough for Alpaca's free data plan."""
    close = pd.Timestamp(f"{day} 16:00", tz="America/New_York")
    return pd.Timestamp.now(tz="America/New_York") > close + pd.Timedelta(minutes=20)


def _daily_bar(client, symbol: str, day: str, adjustment) -> pd.Series:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start = pd.Timestamp(day, tz="America/New_York")
    end = min(
        start + pd.Timedelta(days=1), pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(minutes=16)
    )
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        adjustment=adjustment,
        feed=DataFeed.SIP,
    )
    frame = client.get_stock_bars(request).df
    if len(frame) != 1:
        raise RuntimeError(f"expected exactly one {symbol} bar for {day}, got {len(frame)}")
    return frame.iloc[0]


def spy_reference_prices(start_date: str, end_date: str) -> dict:
    """SPY's open on the first day and close on the last day, raw and dividend-adjusted."""
    from alpaca.data.enums import Adjustment
    from alpaca.data.historical import StockHistoricalDataClient

    if not _session_is_over(end_date):
        raise RuntimeError(f"The {end_date} session is not over yet. Run this after the close.")
    client = StockHistoricalDataClient(*paper_keys())
    prices = {
        "raw_open_start": float(_daily_bar(client, "SPY", start_date, Adjustment.RAW)["open"]),
        "raw_close_end": float(_daily_bar(client, "SPY", end_date, Adjustment.RAW)["close"]),
        "adjusted_open_start": float(_daily_bar(client, "SPY", start_date, Adjustment.ALL)["open"]),
        "adjusted_close_end": float(_daily_bar(client, "SPY", end_date, Adjustment.ALL)["close"]),
    }
    ledger.record_holdout_read(
        symbol="SPY",
        dates=[start_date, end_date],
        purpose="benchmark for the DTFM paper run; no strategy fitted or scored",
    )
    return prices


def compare_paper(run: dict, prices: dict) -> dict:
    paper = run["end_equity"] / run["start_equity"] - 1.0
    spy_price = prices["raw_close_end"] / run["spy_baseline"] - 1.0
    spy_with_dividends = prices["adjusted_close_end"] / prices["adjusted_open_start"] - 1.0
    return {
        "paper_return": paper,
        "spy_price_return_from_log_baseline": spy_price,
        "spy_return_with_dividends": spy_with_dividends,
        "paper_minus_spy_price": paper - spy_price,
        "paper_minus_spy_with_dividends": paper - spy_with_dividends,
        "paper_beat_spy": bool(paper > spy_with_dividends),
        "prices": prices,
    }


def comparison_markdown(run: dict, result: dict, verdict: dict | None) -> str:
    p = result["prices"]
    lines = [
        "# DTFM: the lab's verdict next to the real paper run",
        "",
        "## The real paper run",
        "",
        f"- Universe: {run['universe']}",
        f"- {run['start_date']} to {run['end_date']}: account went from "
        f"${run['start_equity']:,.2f} to ${run['end_equity']:,.2f}, "
        f"**{signed_pct(result['paper_return'])}**",
        f"- Positions at the end: {run['positions_at_end']}",
        "",
        "## SPY over the same dates",
        "",
        f"- Price only, from the experiment's own baseline (${run['spy_baseline']:.2f} at the "
        f"{run['start_date']} open) to the {run['end_date']} close (${p['raw_close_end']:.2f}): "
        f"**{signed_pct(result['spy_price_return_from_log_baseline'])}**",
        f"- With dividends (Alpaca adjusted open ${p['adjusted_open_start']:.2f} to adjusted close "
        f"${p['adjusted_close_end']:.2f}): **{signed_pct(result['spy_return_with_dividends'])}**",
        f"- Alpaca's own raw open on {run['start_date']} was ${p['raw_open_start']:.2f}, against "
        f"the logged baseline of ${run['spy_baseline']:.2f}",
        "",
        "## Paper run minus SPY",
        "",
        f"- Against SPY price only: **{signed_pct(result['paper_minus_spy_price'])}**",
        f"- Against SPY with dividends: **{signed_pct(result['paper_minus_spy_with_dividends'])}**",
        f"- Did the paper run beat buy-and-hold SPY? **{'YES' if result['paper_beat_spy'] else 'NO'}**",
        "",
        "## The lab's verdict on the same rules",
        "",
    ]
    if verdict is None:
        lines.append("DTFM has not been through the gates yet.")
    else:
        g = verdict["gates"]
        lines += [
            "Run on QQQ plus 9 sector ETFs (a survivorship-clean stand-in for the mega-cap "
            "stocks), on data before the locked holdout.",
            "",
            f"- Gate 1, in-sample vs SPY: {gate_status(g['1'])}",
            f"- Gate 2, in-sample permutation: {gate_status(g['2'])} (p = {g['2']['p_value']:.4f}, "
            f"needs < {g['2']['alpha']:.4f})",
            f"- Gate 3, walk-forward vs SPY: {gate_status(g['3'])}",
            f"- Gate 4, walk-forward permutation: {gate_status(g['4'])}",
            "",
            "**Lab verdict: "
            + (
                "passed all 4 gates."
                if verdict["passed_all_gates"]
                else f"dead at {', '.join(verdict['died_at'])}."
            )
            + "**",
            "",
        ]
        lab_ok, paper_ok = verdict["passed_all_gates"], result["paper_beat_spy"]
        if lab_ok == paper_ok:
            lines.append(
                "The two agree: "
                + (
                    "the lab found an edge and the paper run beat SPY."
                    if lab_ok
                    else "the lab found no edge it could trust, and the paper run trailed SPY."
                )
            )
        else:
            lines.append(
                "The two disagree: "
                + (
                    "the lab found an edge, but the paper run trailed SPY."
                    if lab_ok
                    else "the lab found no edge it could trust, but the paper run beat SPY."
                )
            )
    lines += [
        "",
        "## How much weight this comparison can carry",
        "",
        "- The paper run is one stretch of about four months. That is far too short to prove or "
        "disprove anything by itself; the gates use decades of data.",
        "- The universes differ: the paper run traded 7 mega-cap stocks, the lab run used ETFs.",
        "- The paper run was not a clean run of the rules. Its log records wrong price data "
        "behind several decisions, including a first purchase that never met the entry rule.",
        "- The account equity was read during the end date, and SPY is taken at that day's close.",
        "- The two SPY bars sit inside the lab's locked holdout. They were read only as a "
        "benchmark for this comparison and are recorded in `ledger/holdout_reads.json`.",
        "",
    ]
    return "\n".join(lines)
