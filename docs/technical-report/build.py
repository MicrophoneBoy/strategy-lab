"""Build the technical report (HTML and PDF) from the lab's own results.

Every number in the report is read from results/, ledger/, config.toml, holdout.lock.json
and data/meta.json, or computed by the engine on the development data. The script also
checks the claims the text makes (for example that no rule passed gate 2) and stops if
one of them stops being true. Run it from the repository root:

    uv run python docs/technical-report/build.py

It needs the data cache (`uv run lab fetch`). The PDF is printed with Edge or Chrome in
headless mode; without either, open the HTML and print it.
"""

from __future__ import annotations

import io
import json
import shutil
import string
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from strategy_lab.cli import SECTORS, load_idea  # noqa: E402
from strategy_lab.config import load_config  # noqa: E402
from strategy_lab.engine import buy_and_hold  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_HTML = HERE / "strategy-lab-technical-report.html"
OUT_PDF = HERE / "strategy-lab-technical-report.pdf"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
PCT_KEYS = {"drop", "stop", "buy_weight", "max_weight"}

NAMES = {
    "sma_crossover": "Moving-average crossover",
    "dtfm": "Dual-timeframe momentum (DTFM)",
    "sector_rotation": "Sector momentum rotation",
    "stocks_or_bonds": "Stocks or bonds by momentum",
    "low_vol_sectors": "Low-volatility sectors",
    "turn_of_month": "Turn of the month",
    "volatility_filter": "Volatility filter",
    "mean_reversion": "Short-term mean reversion",
    "breakout": "Breakout",
}
HOW = {
    "sma_crossover": "Long while the <i>fast</i>-day simple moving average is above the <i>slow</i>-day one, otherwise flat.",
    "dtfm": ("Weekly, hold up to {slots} assets whose {fast}- and {slow}-session returns are both positive, "
             "best {slow}-session return first. Each buy is {buy_weight} of the account (cap {max_weight}), "
             "with a {stop} trailing stop; the rest is cash."),
    "sector_rotation": "Each month end, hold the <i>hold</i> sectors with the best <i>lookback</i>-day return, equal weight.",
    "stocks_or_bonds": "Each month end, hold SPY if its <i>lookback</i>-day return is positive, otherwise IEF.",
    "low_vol_sectors": "Each month end, hold the <i>hold</i> sectors with the lowest <i>window</i>-day volatility, equal weight.",
    "turn_of_month": "Long for the last <i>last</i> trading days of each month and the first <i>first</i> of the next.",
    "volatility_filter": "Long only while <i>window</i>-day volatility is below its <i>percentile</i>-th percentile of the past year.",
    "mean_reversion": "Long while the <i>lookback</i>-day return is below minus <i>drop</i>, otherwise flat.",
    "breakout": "Buy at a new <i>entry</i>-day closing high, sell at a new <i>exit</i>-day closing low.",
}
SPY_ONLY = ("sma_crossover", "turn_of_month", "volatility_filter", "mean_reversion")


# ---------- formatting ----------

def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"


def spct(x: float, d: int = 2) -> str:
    return f"{x * 100:+.{d}f}%"


def apct(x: float, d: int = 2) -> str:
    return pct(abs(x), d)


def pval(p: float) -> str:
    return f"{p:.4f}"


def num(n: int) -> str:
    return f"{n:,}"


def value(key: str, v) -> str:
    return pct(v, 0) if key in PCT_KEYS else f"{v:g}"


def setting(params: dict) -> str:
    return ", ".join(f"{k.replace('_', ' ')} {value(k, v)}" for k, v in params.items())


def grid_text(grid: tuple[dict, ...]) -> str:
    if len(grid) == 1:
        return "one fixed setting"
    parts = []
    for key in grid[0]:
        vals = sorted({g[key] for g in grid})
        steps = set(np.round(np.diff(vals), 9))
        if len(vals) > 4 and len(steps) == 1:
            parts.append(f"<i>{key}</i> {value(key, vals[0])} to {value(key, vals[-1])} by {value(key, steps.pop())}")
        else:
            parts.append(f"<i>{key}</i> " + ", ".join(value(key, v) for v in vals))
    return "; ".join(parts)


def month(d: str) -> str:
    return d[:7]


def table(n: int, caption: str, head: list[str], rows: list[list[str]], align: str) -> str:
    th = "".join(f'<th class="{a}">{h}</th>' for h, a in zip(head, align.split(",")))
    body = "".join(
        "<tr>" + "".join(f'<td class="{a}">{c}</td>' for c, a in zip(r, align.split(","))) + "</tr>" for r in rows
    )
    cls = "tablewrap split" if len(rows) > 8 and n != 1 else "tablewrap"
    return (f'<div class="{cls}"><p class="tcap"><b>Table {n}.</b> {caption}</p>'
            f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>")


# ---------- figures ----------

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "svg.fonttype": "none",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": INK, "legend.frameon": False,
})
WIDTH = 6.6


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg"):]


def figure(n: int, body: str, caption: str) -> str:
    return f'<figure>{body}<figcaption><b>Figure {n}.</b> {caption}</figcaption></figure>'


def fig_power(tester: dict) -> str:
    cal, power = tester["calibration"], tester["power"]
    labels = [f"No edge\n({cal['series']} random walks)"] + [f"Trend {pct(p['trend_per_day'], 1)} a day\n({p['series']} series)" for p in power]
    rates = [cal["false_alarm_rate_at_1_percent"]] + [p["detected"] / p["series"] for p in power]
    counts = [f"{round(cal['false_alarm_rate_at_1_percent'] * cal['series'])} of {cal['series']}"] + [f"{p['detected']} of {p['series']}" for p in power]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    x = np.arange(len(rates))
    ax.bar(x, rates, width=0.5, color=BLUE, zorder=2)
    for xi, r, c in zip(x, rates, counts):
        ax.text(xi, r + 0.03, c, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Flagged at the 1% level")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="x", length=0)
    return svg(fig)


def fig_heatmap(baseline: dict) -> str:
    rows = baseline["sweep"]
    fasts = sorted({r["fast"] for r in rows})
    slows = sorted({r["slow"] for r in rows})
    grid = np.full((len(fasts), len(slows)), np.nan)
    for r in rows:
        grid[fasts.index(r["fast"]), slows.index(r["slow"])] = r["excess_cagr"] * 100
    lo = np.nanmin(grid)
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, NEUTRAL, BLUE])
    norm = TwoSlopeNorm(vmin=min(lo, -0.01), vcenter=0.0, vmax=max(np.nanmax(grid), -lo))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto", origin="lower")
    for i in range(len(fasts)):
        for j in range(len(slows)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color=INK)
    best = baseline["plateau"]["best"]
    bi, bj = fasts.index(best["fast"]), slows.index(best["slow"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.4))
    ax.set_xticks(range(len(slows)), slows)
    ax.set_yticks(range(len(fasts)), fasts)
    ax.set_xlabel("Slow moving average (days)")
    ax.set_ylabel("Fast moving average (days)")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Excess return over SPY, % a year after costs", color=INK)
    bar.outline.set_visible(False)
    return svg(fig)


def fig_gate1(order: list[str], verdicts: dict) -> str:
    names = [f"{i + 1}. {NAMES[n]} ({'pass' if verdicts[n]['gates']['1']['passed'] else 'fail'})" for i, n in enumerate(order)]
    costs = [verdicts[n]["gates"]["1"]["versus_benchmark"]["excess_cagr"] * 100 for n in order]
    taxed = [verdicts[n]["gates"]["1"]["versus_benchmark"]["excess_after_tax_cagr"] * 100 for n in order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    h = 0.36
    ax.barh(y + h / 2, costs, height=h - 0.04, color=BLUE, label="After costs", zorder=2)
    ax.barh(y - h / 2, taxed, height=h - 0.04, color=ORANGE, label="After costs and tax", zorder=2)
    for yi, c in zip(y, costs):
        ax.text(c + (0.3 if c >= 0 else -0.3), yi + h / 2, f"{c:+.2f}", va="center",
                ha="left" if c >= 0 else "right", fontsize=7, color=INK)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0)
    lim = max(abs(min(costs + taxed)), max(costs + taxed)) + 3
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Best in-sample setting minus buy-and-hold SPY, % a year")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.5)
    return svg(fig)


def fig_pvalues(order: list[str], verdicts: dict) -> str:
    g2 = [verdicts[n]["gates"]["2"] for n in order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    for yi, g in zip(y, g2):
        ax.plot([g["alpha"], g["p_value"]], [yi, yi], color=GRID, linewidth=2, zorder=1)
    ax.scatter([g["alpha"] for g in g2], y, marker="|", s=160, linewidths=2, color=ORANGE, label="Pass bar (p must be below)", zorder=3)
    ax.scatter([g["p_value"] for g in g2], y, s=36, color=BLUE, edgecolors="white", linewidths=1.2, label="Real p-value", zorder=4)
    for yi, g in zip(y, g2):
        ax.text(g["p_value"] * 1.25, yi, pval(g["p_value"]), va="center", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 3)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1], ["0.001", "0.01", "0.1", "1"])
    ax.set_yticks(y, [f"{i + 1}. {NAMES[n]}" for i, n in enumerate(order)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel("Permutation p-value (log scale)")
    ax.legend(loc="lower left", fontsize=7.5, ncols=2, bbox_to_anchor=(0, 1.0))
    return svg(fig)


def fig_curves(curves: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    for ax, (title, (bt, bh, start)) in zip(axes, curves.items()):
        dates = np.concatenate([[start], bt.dates]).astype("datetime64[D]")
        for series, color, label in ((bt.net, BLUE, "Rule, after costs"), (bh.net, ORANGE, "Buy-and-hold SPY")):
            ax.plot(dates, np.exp(np.concatenate([[0.0], np.cumsum(series)])), color=color, linewidth=1.6, label=label)
        ax.axhline(1, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="lower left", fontsize=7.5)
    return svg(fig)


# ---------- build ----------

def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    cfg = load_config()
    ledger = load("ledger/tests.json")
    order = [t["idea"] for t in ledger]
    universe = {t["idea"]: t["data"].replace(" development window", "") for t in ledger}
    verdicts = {n: load(f"results/verdict_{n}.json") for n in order}
    baseline, tester = load("results/baseline.json"), load("results/tester.json")
    paper, lock, meta = load("results/dtfm_paper_comparison.json"), load("holdout.lock.json"), load("data/meta.json")
    families = {n: load_idea(n, cfg) for n in order}

    g1 = {n: verdicts[n]["gates"]["1"] for n in order}
    g2 = {n: verdicts[n]["gates"]["2"] for n in order}
    passes = [n for n in order if g1[n]["passed"]]
    share = {n: g2[n]["permuted_at_least_as_good"] / g2[n]["permutations"] for n in order}

    # Claims the text makes. If one stops being true, the text must change.
    assert not any(g2[n]["passed"] for n in order), "a rule passed gate 2"
    assert not any(verdicts[n]["passed_all_gates"] for n in order)
    assert all(not g1[n]["passed"] for n in SPY_ONLY), "a SPY-only rule passed gate 1"
    assert all(g1[n]["first_bar"] < "2002-10-10" and g1[n]["last_bar"] > "2000-03-24" for n in passes), "a gate 1 win missed the bear market"
    assert all(g1[n]["versus_benchmark"]["beats_benchmark_after_costs"] == g1[n]["versus_benchmark"]["beats_benchmark_after_costs_and_taxes"] for n in order)
    assert tester["all_ok"] and all(c["ok"] for c in tester["checks"])
    assert baseline["plateau"]["share_of_grid_beating_benchmark"] == 0
    assert g2["dtfm"]["real_score"] < g2["dtfm"]["permuted_median"]
    assert sum(len(f[0].grid) for f in families.values()) == verdicts[order[-1]]["tests_so_far"]["param_sets"]
    assert not any("IWM" in t["data"] for t in ledger), "a rule used IWM"
    mr = "mean_reversion"
    assert min(order, key=lambda n: g2[n]["p_value"]) == mr and not g1[mr]["passed"]

    # Figure 5: the two biggest gate 1 wins, recomputed by the engine and checked against the verdicts.
    train = cfg.validation.train_bars
    curves = {}
    for n, title in (("dtfm", "DTFM on QQQ and 9 sector ETFs"), ("breakout", "Breakout on QQQ")):
        family, data, bench, *_ = families[n]
        dev, bdev = data.slice(0, train), bench.slice(0, train)
        k = family.grid.index(g1[n]["best_params"])
        bt = family.backtest(dev, family.decisions(dev)[k], cfg.costs, cfg.tax)
        bh = buy_and_hold(bdev, cfg.costs, cfg.tax)
        assert abs(bt.metrics["total_return"] - g1[n]["strategy"]["total_return"]) < 1e-9
        assert abs(bh.metrics["total_return"] - g1[n]["benchmark"]["total_return"]) < 1e-9
        curves[f"{title}, {setting(g1[n]['best_params']) if n == 'breakout' else 'fixed rules'}"] = (bt, bh, dev.dates[0])

    # Tables.
    checks = tester["checks"]
    table_tester = table(
        1, "Test the tester: every check on data with a known answer. Gate 2 is the in-sample permutation test, gate 4 the walk-forward one.",
        ["Strategy", "Data", "Gate", "Expected", "p-value", "Result"],
        [[c["strategy"], c["data"], str(c["gate"]), c["expected"], pval(c["p_value"]), c["gate_result"]] for c in checks],
        ",,c,c,n,c",
    )
    dt = dict(families["dtfm"][0].grid[0])
    dtfm_how = HOW["dtfm"].format(**{k: (pct(v, 0) if k in PCT_KEYS else f"{v:g}") for k, v in dt.items()})
    table_rules = table(
        2, "The nine rules in the order they were tested. Grids and order were fixed before the first run. The pass bar is the in-sample permutation bar, 0.01 divided by the rule's position.",
        ["#", "Rule", "Data", "How it trades", "Grid", "Versions", "Pass bar"],
        [[str(i + 1), NAMES[n], universe[n], dtfm_how if n == "dtfm" else HOW[n], grid_text(families[n][0].grid),
          str(len(families[n][0].grid)), pval(g2[n]["alpha"])] for i, n in enumerate(order)],
        "n,,,,,n,n",
    )
    table_results = table(
        3, "Gate 1 and gate 2 for every rule. Excess is the best in-sample setting's CAGR minus buy-and-hold SPY's over the same development window. Shuffles counts how many shuffled histories scored at least as well as the real data.",
        ["#", "Rule", "Window", "Excess after costs", "Excess after tax", "Gate 1", "Shuffles", "p", "Bar", "Gate 2"],
        [[str(i + 1), NAMES[n], f"{month(g1[n]['first_bar'])} to {month(g1[n]['last_bar'])}",
          spct(g1[n]["versus_benchmark"]["excess_cagr"]), spct(g1[n]["versus_benchmark"]["excess_after_tax_cagr"]),
          "pass" if g1[n]["passed"] else "fail",
          f"{num(g2[n]['permuted_at_least_as_good'])} of {num(g2[n]['permutations'])}",
          pval(g2[n]["p_value"]), pval(g2[n]["alpha"]), "pass" if g2[n]["passed"] else "fail"] for i, n in enumerate(order)],
        "n,,,n,n,c,n,n,n,c",
    )
    table_rationale = table(
        4, "Stated reason for each rule, written before it was run, and the best setting it chose in sample.",
        ["#", "Rule", "Why an edge might exist, and who loses", "Best setting"],
        [[str(i + 1), NAMES[n], verdicts[n]["rationale"], "fixed rules" if n == "dtfm" else setting(g1[n]["best_params"])] for i, n in enumerate(order)],
        "n,,,",
    )

    # Figures.
    n_fig = {"power": 1, "heatmap": 2, "gate1": 3, "pvalues": 4, "curves": 5}
    cal, power = tester["calibration"], tester["power"]
    b, bb, bv = baseline["strategy"], baseline["benchmark"], baseline["versus_benchmark"]
    best = baseline["plateau"]["best"]
    figs = {
        "fig_power": figure(n_fig["power"], fig_power(tester),
            f"How often the in-sample permutation test flags a series at the 1% level: random walks with no edge, and synthetic series with a planted daily trend. Each series used {cal['permutations_each']} shuffles."),
        "fig_heatmap": figure(n_fig["heatmap"], fig_heatmap(baseline),
            f"Moving-average crossover on SPY, {baseline['window']}: excess return over buy-and-hold for each of the {len(baseline['sweep'])} settings. The outlined cell is the best setting. Blank cells are settings where the fast average would not be shorter than the slow one."),
        "fig_gate1": figure(n_fig["gate1"], fig_gate1(order, verdicts),
            "Gate 1: the best in-sample setting of each rule against buy-and-hold SPY over the same development window. A rule passes only if both bars are above zero."),
        "fig_pvalues": figure(n_fig["pvalues"], fig_pvalues(order, verdicts),
            "Gate 2: each rule's permutation p-value against its Bonferroni pass bar. A rule passes only if its dot lies left of its bar. None does."),
        "fig_curves": figure(n_fig["curves"], fig_curves(curves),
            "The two largest gate 1 wins, recomputed by the engine over their development window. Both beat SPY by staying mostly out of a falling market."),
    }

    def p_trend(i: int) -> dict:
        return {f"p{i + 1}_trend": pct(power[i]["trend_per_day"], 1), f"p{i + 1}_det": str(power[i]["detected"]), f"p{i + 1}_n": str(power[i]["series"])}

    syms = list(cfg.symbols)
    others = [s for s in syms if s not in SECTORS]
    d, bo = g1["dtfm"], g1["breakout"]
    v = cfg.validation
    values = {
        "report_month": date.today().strftime("%B %Y"),
        "n_ideas": str(len(order)),
        "n_sets": num(verdicts[order[-1]]["tests_so_far"]["param_sets"]),
        "n_g1_pass_word": WORDS[len(passes)],
        "n_g1_pass_cap": WORDS[len(passes)].capitalize(),
        "share_min_g1": pct(min(share[n] for n in passes), 1),
        "share_max_g1": pct(max(share[n] for n in passes), 1),
        "n_symbols": str(len(syms)),
        "symbols_sentence": ", ".join(others[:-1]) + f", {others[-1]} and nine sector ETFs ({', '.join(SECTORS)})",
        "spy_first": meta["symbols"]["SPY"]["dev_first"],
        "sector_first": meta["symbols"][SECTORS[0]]["dev_first"],
        "qqq_first": meta["symbols"]["QQQ"]["dev_first"],
        "ief_first": meta["symbols"]["IEF"]["dev_first"],
        "holdout_months": str(lock["months"]),
        "holdout_start": lock["holdout_start"],
        "holdout_last": lock["last_bar_when_locked"],
        "holdout_bars": str(meta["symbols"]["SPY"]["holdout_bars"]),
        "dev_end": meta["symbols"]["SPY"]["dev_last"],
        "train_bars": num(v.train_bars),
        "step_bars": str(v.step_bars),
        "spy_window": f"{g1['sma_crossover']['first_bar']} to {g1['sma_crossover']['last_bar']}",
        "fee_bps": f"{cfg.costs.fee_bps:g}",
        "slippage_bps": f"{cfg.costs.slippage_bps:g}",
        "st_rate": pct(cfg.tax.short_term, 0),
        "lt_rate": pct(cfg.tax.long_term, 0),
        "n_perm_is": num(v.n_perm_insample),
        "n_perm_is_early": num(min(g2[n]["permutations"] for n in order)),
        "n_perm_wf": num(v.n_perm_walkforward),
        "alpha": f"{v.alpha_insample:g}",
        "alpha_1": pval(g2[order[0]]["alpha"]),
        "alpha_9": pval(g2[order[-1]]["alpha"]),
        "min_possible_p": pval(1 / (1 + v.n_perm_insample)),
        "planted_trend": pct(tester["planted_trend_per_day"], 1),
        "n_checks": str(len(checks)),
        "planted_g1": pct(tester["planted_edge_beats_buy_and_hold"]["gate_1_excess_after_tax_cagr"]),
        "planted_g3": pct(tester["planted_edge_beats_buy_and_hold"]["gate_3_excess_after_tax_cagr"]),
        "table_tester": table_tester,
        "calib_series": str(cal["series"]),
        "calib_perms": str(cal["permutations_each"]),
        "fa5": pct(cal["false_alarm_rate_at_5_percent"], 0),
        "fa1": pct(cal["false_alarm_rate_at_1_percent"], 0),
        **p_trend(0), **p_trend(1), **p_trend(2),
        "table_rules": table_rules,
        "base_window": baseline["window"],
        "base_cagr": pct(b["cagr"]), "bh_cagr": pct(bb["cagr"]),
        "base_sharpe": f"{b['sharpe']:.2f}", "bh_sharpe": f"{bb['sharpe']:.2f}",
        "base_mdd": apct(b["max_drawdown"]), "bh_mdd": apct(bb["max_drawdown"]),
        "base_at": pct(b["after_tax_cagr"]), "bh_at": pct(bb["after_tax_cagr"]),
        "sweep_n": str(len(baseline["sweep"])),
        "best_fast": str(best["fast"]), "best_slow": str(best["slow"]), "best_excess": apct(best["excess_cagr"]),
        "g1_min": pct(min(g1[n]["versus_benchmark"]["excess_cagr"] for n in passes)),
        "g1_max": pct(max(g1[n]["versus_benchmark"]["excess_cagr"] for n in passes)),
        "mr_p": pval(g2[mr]["p_value"]), "mr_bar": pval(g2[mr]["alpha"]),
        "mr_ratio": f"{g2[mr]['p_value'] / g2[mr]['alpha']:.0f}",
        "mr_excess": apct(g1[mr]["versus_benchmark"]["excess_cagr"]),
        "table_results": table_results,
        "dtfm_window": f"{d['first_bar']} to {d['last_bar']}",
        "dtfm_tr": apct(d["strategy"]["total_return"]), "spy_tr": apct(d["benchmark"]["total_return"]),
        "dtfm_exp": pct(d["strategy"]["exposure"], 1),
        "dtfm_mdd": apct(d["strategy"]["max_drawdown"]), "spy_mdd": apct(d["benchmark"]["max_drawdown"]),
        "bo_tr": pct(bo["strategy"]["total_return"]), "bo_exp": pct(bo["strategy"]["exposure"], 1),
        "bo_trades": str(bo["strategy"]["trades"]),
        "dtfm_real": f"{g2['dtfm']['real_score']:.3f}", "dtfm_med": f"{g2['dtfm']['permuted_median']:.3f}",
        "dtfm_k": num(g2["dtfm"]["permuted_at_least_as_good"]), "dtfm_n": num(g2["dtfm"]["permutations"]),
        "bo_k": num(g2["breakout"]["permuted_at_least_as_good"]), "bo_n": num(g2["breakout"]["permutations"]),
        "paper_start": paper["paper_run"]["start_date"], "paper_end": paper["paper_run"]["end_date"],
        "paper_start_eq": f"${paper['paper_run']['start_equity']:,.2f}",
        "paper_end_eq": f"${paper['paper_run']['end_equity']:,.2f}",
        "paper_ret": spct(paper["paper_return"]),
        "spy_price_ret": spct(paper["spy_price_return_from_log_baseline"]),
        "spy_div_ret": spct(paper["spy_return_with_dividends"]),
        "paper_gap": apct(paper["paper_minus_spy_with_dividends"]),
        "p_min_g1": pval(min(g2[n]["p_value"] for n in passes)),
        "p_max_g1": pval(max(g2[n]["p_value"] for n in passes)),
        "n_median_better": WORDS[sum(g2[n]["permuted_median"] >= g2[n]["real_score"] for n in order)],
        "n_tax_lower": WORDS[sum(g1[n]["versus_benchmark"]["excess_after_tax_cagr"] < g1[n]["versus_benchmark"]["excess_cagr"] for n in passes)],
        "table_rationale": table_rationale,
        "seed": str(v.seed),
        "fingerprint": meta["dev_fingerprint"],
        **figs,
    }
    html = string.Template((HERE / "template.html").read_text(encoding="utf-8")).substitute(values)

    for bad in ("\u2013", "\u2014", "${"):
        assert bad not in html, f"found {bad!r} in the report"

    OUT_HTML.write_text(html, encoding="utf-8")
    print("wrote", OUT_HTML.relative_to(ROOT))
    print_pdf(OUT_HTML, OUT_PDF)


def print_pdf(html: Path, pdf: Path) -> None:
    candidates = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome"),
    ]
    exe = next((c for c in candidates if c and Path(c).exists()), None)
    if exe is None:
        print("No Edge or Chrome found: open the HTML in a browser and print it to PDF.")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run(
            [exe, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
             "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
            check=True, timeout=180, capture_output=True,
        )
    print("wrote", pdf.relative_to(ROOT))


if __name__ == "__main__":
    main()
