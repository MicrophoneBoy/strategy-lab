"""One self-contained HTML page with every idea, its sweep, its gates and its
verdict in plain language. Formatting only: every number comes from the
ledger, the result files or the sweep passed in."""

from __future__ import annotations

import html

from .report import METRIC_ROWS, PLAIN_NAMES, gate_status, pct, signed_pct, what_happened

RULES = {
    "sma_crossover": "Hold SPY while its fast moving average is above its slow one, otherwise hold cash.",
    "dtfm": "Hold up to 3 assets whose 20-session and 60-session returns are both positive, 10% of "
    "the account each, reviewed weekly, with an 8% trailing stop. Rules fixed in advance.",
    "sector_rotation": "Each month end, hold the top sectors by trailing return, equal weight, fully invested.",
    "stocks_or_bonds": "Each month end, hold SPY if its trailing return is positive, otherwise the bond ETF IEF.",
    "low_vol_sectors": "Each month end, hold the calmest sectors (lowest recent volatility), equal weight.",
    "turn_of_month": "Hold SPY only for the last few business days of a month and the first few of the next.",
    "volatility_filter": "Hold SPY only while its recent volatility is below a percentile of the past year.",
    "mean_reversion": "Hold SPY only while its return over the last few days is below a negative threshold.",
    "breakout": "Buy QQQ at a new N-day closing high and sell at a new M-day closing low.",
}

CSS = """
:root { --bg:#fcfcfb; --panel:#f4f3f0; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#dedcd6;
        --pos:#2a6cb5; --neg:#b3372e; --mid:#f0efec; --pass:#1f7a4d; --fail:#b3372e; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1a1a19; --panel:#242423; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#3a3a37;
          --pos:#4a8fd6; --neg:#d2564c; --mid:#383835; --pass:#5fbf8f; --fail:#e0776e; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 system-ui, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 30px; margin: 0 0 4px; } h2 { font-size: 22px; margin: 44px 0 10px; }
h3 { font-size: 16px; margin: 22px 0 8px; color: var(--ink2); }
p { margin: 8px 0; } .lede { color: var(--ink2); max-width: 70ch; }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin:20px 0; }
.tile { background:var(--panel); border-radius:10px; padding:14px 18px; min-width:150px; }
.tile b { display:block; font-size:28px; line-height:1.2; } .tile span { color:var(--ink2); font-size:14px; }
.scroll { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:14px; margin:8px 0; }
th, td { padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { color:var(--ink2); font-weight:600; } td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
.badge { display:inline-block; padding:1px 8px; border-radius:999px; font-size:13px; font-weight:600;
         border:1px solid currentColor; white-space:nowrap; }
.PASS { color:var(--pass); } .FAIL { color:var(--fail); } .none { color:var(--muted); }
section.idea { border-top:1px solid var(--line); margin-top:36px; }
.heat { width:auto; } .heat td, .heat th { text-align:center; border:2px solid var(--bg); padding:6px 9px;
        font-variant-numeric: tabular-nums; }
.heat td.cell { min-width:58px; }
.heat td.up { background: color-mix(in srgb, var(--pos) calc(var(--v) * 100%), var(--mid)); }
.heat td.down { background: color-mix(in srgb, var(--neg) calc(var(--v) * 100%), var(--mid)); }
.heat td.strong { color:#fff; } .heat td.best { outline:2px solid var(--ink); outline-offset:-2px; font-weight:700; }
.heat td.empty { background:transparent; }
.note { color:var(--ink2); font-size:14px; } ol, ul { padding-left: 22px; } li { margin: 4px 0; }
"""


def esc(value) -> str:
    return html.escape(str(value))


def badge(gate: dict) -> str:
    status = gate_status(gate)
    return f'<span class="badge {"none" if status == "not run" else status}">{status}</span>'


def heat_cell(value: float, reach: float, best: bool) -> str:
    share = min(abs(value) / reach, 1.0) if reach > 0 else 0.0
    classes = ["cell", "up" if value >= 0 else "down"]
    if share > 0.6:
        classes.append("strong")
    if best:
        classes.append("best")
    return f'<td class="{" ".join(classes)}" style="--v:{share:.3f}">{value * 100:+.1f}</td>'


def sweep_table(rows: list[dict], grid: tuple[dict, ...]) -> str:
    """Excess return over SPY for every version tried, as a coloured grid."""
    keys = [k for k in grid[0] if len({str(g[k]) for g in grid}) > 1]
    values = [r["excess_cagr"] for r in rows]
    reach, top = max(abs(v) for v in values), max(values)
    caption = (
        '<p class="note">Return per year minus buy-and-hold SPY, in percentage points, after costs, '
        "on the first 4 years. Blue beats SPY, red trails it. The outlined cell is the best version.</p>"
    )
    if len(rows) == 1:
        return caption + f'<table class="heat"><tr>{heat_cell(values[0], reach, True)}</tr></table>'
    if len(keys) == 1:
        head = "".join(f"<th>{esc(r[keys[0]])}</th>" for r in rows)
        cells = "".join(heat_cell(r["excess_cagr"], reach, r["excess_cagr"] == top) for r in rows)
        return (
            caption
            + f'<div class="scroll"><table class="heat"><tr><th>{esc(keys[0])}</th>{head}</tr>'
            f"<tr><th>vs SPY</th>{cells}</tr></table></div>"
        )
    if len(keys) != 2:
        return ""
    x, y = keys
    xs, ys = list(dict.fromkeys(r[x] for r in rows)), list(dict.fromkeys(r[y] for r in rows))
    lookup = {(r[x], r[y]): r["excess_cagr"] for r in rows}
    out = [f'<div class="scroll"><table class="heat"><tr><th>{esc(y)} \\ {esc(x)}</th>']
    out += [f"<th>{esc(v)}</th>" for v in xs] + ["</tr>"]
    for yv in ys:
        out.append(f"<tr><th>{esc(yv)}</th>")
        for xv in xs:
            v = lookup.get((xv, yv))
            out.append('<td class="empty"></td>' if v is None else heat_cell(v, reach, v == top))
        out.append("</tr>")
    return caption + "".join(out) + "</table></div>"


def metrics_table(strategy: dict, benchmark: dict, name: str) -> str:
    rows = "".join(
        f'<tr><td>{esc(label)}</td><td class="num">{esc(fmt(strategy[key]))}</td>'
        f'<td class="num">{esc(fmt(benchmark[key]))}</td></tr>'
        for label, key, fmt in METRIC_ROWS
        if key in strategy
    )
    return (
        f'<div class="scroll"><table><tr><th>Metric</th><th class="num">{esc(name)}, best version</th>'
        f'<th class="num">Buy-and-hold SPY</th></tr>{rows}</table></div>'
    )


def gates_table(g: dict) -> str:
    v1 = g["1"]["versus_benchmark"]
    rows = [
        ("1. Beat SPY on the first 4 years", g["1"],
         f"{signed_pct(v1['excess_cagr'])} a year after costs, "
         f"{signed_pct(v1['excess_after_tax_cagr'])} after tax"),
        ("2. Better than luck (shuffle test)", g["2"],
         f"{g['2']['permuted_at_least_as_good']:,} of {g['2']['permutations']:,} shuffled histories did "
         f"at least as well: p = {g['2']['p_value']:.4f}, needed below {g['2']['alpha']:.4f}"),
    ]
    if g["3"]["passed"] is None:
        rows.append(("3. Beat SPY out of sample (walk-forward)", g["3"], "not run, data left unspent"))
    else:
        v3 = g["3"]["versus_benchmark"]
        rows.append(("3. Beat SPY out of sample (walk-forward)", g["3"],
                     f"{g['3']['first_bar']} to {g['3']['last_bar']}: "
                     f"{signed_pct(v3['excess_cagr'])} a year after costs, "
                     f"{signed_pct(v3['excess_after_tax_cagr'])} after tax"))
    if g["4"]["passed"] is None:
        rows.append(("4. Better than luck out of sample", g["4"], "not run"))
    else:
        rows.append(("4. Better than luck out of sample", g["4"],
                     f"{g['4']['permuted_at_least_as_good']:,} of {g['4']['permutations']:,} shuffles: "
                     f"p = {g['4']['p_value']:.4f}, needed below {g['4']['alpha']:.4f}"))
    body = "".join(
        f"<tr><td>{esc(label)}</td><td>{badge(gate)}</td><td>{esc(text)}</td></tr>"
        for label, gate, text in rows
    )
    return f'<div class="scroll"><table><tr><th>Gate</th><th>Result</th><th>Key number</th></tr>{body}</table></div>'


def idea_section(number: int, idea: dict) -> str:
    v, name = idea["verdict"], idea["name"]
    g = v["gates"]
    plain = PLAIN_NAMES.get(name, name)
    outcome = "Survived all 4 gates" if v["passed_all_gates"] else "Dead at " + ", ".join(v["died_at"])
    shuffle = g["2"]
    return f"""
<section class="idea" id="{esc(name)}">
<h2>{number}. {esc(plain)}</h2>
<p><b>{esc(outcome)}.</b> {esc(what_happened(v))}</p>
<p class="note"><b>Rule:</b> {esc(RULES.get(name, ""))}<br>
<b>Why it might work, and who loses:</b> {esc(v["rationale"])}<br>
<b>Tested on:</b> {esc(idea["tested_on"])}, first 4 years {esc(g["1"]["first_bar"])} to {esc(g["1"]["last_bar"])}.
<b>Versions tried:</b> {v["param_sets"]}. <b>Best version:</b> {esc(", ".join(f"{k} {val}" for k, val in g["1"]["best_params"].items()))}.</p>
<h3>Gates</h3>
{gates_table(g)}
<h3>Every version tried</h3>
{sweep_table(idea["sweep"], idea["grid"])}
<h3>Best version against SPY, first 4 years</h3>
{metrics_table(g["1"]["strategy"], g["1"]["benchmark"], plain)}
<h3>Shuffle test detail</h3>
<p class="note">Total log return of the best version on the real prices: {shuffle["real_score"]:.4f}.
On shuffled prices, re-optimised each time: median {shuffle["permuted_median"]:.4f},
95th percentile {shuffle["permuted_95th"]:.4f}. A real edge should sit far above the 95th percentile.</p>
</section>"""


def summary_rows(ideas: list[dict]) -> str:
    out = []
    for i, idea in enumerate(ideas, 1):
        v = idea["verdict"]
        g = v["gates"]
        outcome = "Survived" if v["passed_all_gates"] else "Dead at " + ", ".join(v["died_at"])
        out.append(
            f'<tr><td class="num">{i}</td><td><a href="#{esc(idea["name"])}">'
            f'{esc(PLAIN_NAMES.get(idea["name"], idea["name"]))}</a></td><td>{esc(idea["tested_on"])}</td>'
            f'<td class="num">{v["param_sets"]}</td><td class="num">'
            f'{signed_pct(g["1"]["versus_benchmark"]["excess_cagr"])}</td><td>{badge(g["1"])}</td>'
            f'<td class="num">{g["2"]["p_value"]:.4f}</td><td class="num">{g["2"]["alpha"]:.4f}</td>'
            f"<td>{badge(g['2'])}</td><td>{esc(outcome)}</td></tr>"
        )
    return "".join(out)


def tester_section(tester: dict | None) -> str:
    if not tester:
        return ""
    rows = "".join(
        f'<tr><td>{esc(c["strategy"])}</td><td>{esc(c["data"])}</td><td class="num">{c["gate"]}</td>'
        f'<td class="num">{c["p_value"]:.4f}</td><td>{esc(c["gate_result"])}</td>'
        f'<td>{esc(c["expected"])}</td><td>{"OK" if c["ok"] else "WRONG"}</td></tr>'
        for c in tester["checks"]
    )
    cal = tester["calibration"]
    power = "".join(
        f'<tr><td class="num">{p["trend_per_day"] * 100:.1f}%</td><td class="num">{p["detected"]} of {p["series"]}</td></tr>'
        for p in tester["power"]
    )
    return f"""
<h2>Is the tester itself trustworthy?</h2>
<p class="lede">Before any verdict is believed, the gates are run on made-up data where the right answer
is known. Random strategies must fail. A strategy on data with a planted edge must pass.</p>
<div class="scroll"><table><tr><th>Strategy</th><th>Made-up data</th><th class="num">Gate</th><th class="num">p</th>
<th>Gate says</th><th>Expected</th><th>Check</th></tr>{rows}</table></div>
<p class="note">False alarms: on {cal["series"]} random walks the shuffle gate fired
{pct(cal["false_alarm_rate_at_5_percent"], 0)} of the time at the 5% level and
{pct(cal["false_alarm_rate_at_1_percent"], 0)} at the 1% level.</p>
<h3>How small an edge can it see?</h3>
<table class="heat" style="text-align:left"><tr><th>Planted trend per day</th><th>Detected at the 1% bar</th></tr>{power}</table>
<p class="note">A weak real edge can be missed. A dead verdict means the idea was not shown to be better than luck.</p>"""


def paper_section(paper: dict | None) -> str:
    if not paper:
        return ""
    run = paper["paper_run"]
    return f"""
<h2>A reality check: DTFM in a real paper account</h2>
<p class="lede">The DTFM rules were also run with fake money in a paper-trading account, on 7 large
stocks, from {esc(run["start_date"])} to {esc(run["end_date"])}.</p>
<div class="tiles">
<div class="tile"><b>{signed_pct(paper["paper_return"])}</b><span>paper account</span></div>
<div class="tile"><b>{signed_pct(paper["spy_return_with_dividends"])}</b><span>SPY with dividends, same dates</span></div>
<div class="tile"><b>{signed_pct(paper["paper_minus_spy_with_dividends"])}</b><span>paper minus SPY</span></div>
</div>
<p class="note">The lab verdict and the paper run point the same way. Four months is far too short to
prove anything alone, the paper run used stocks where the lab used ETFs, and the paper run was not a
clean run of the rules.</p>"""


def build(ideas: list[dict], counts: dict, tester: dict | None, paper: dict | None, note: str) -> str:
    """ideas: dicts with name, tested_on, verdict, sweep (rows), grid."""
    survivors = sum(bool(i["verdict"]["passed_all_gates"]) for i in ideas)
    conclusion = (
        "Nothing here beat simply holding SPY in a way that could be told apart from luck. "
        "The answer so far is: buy the index fund."
        if survivors == 0
        else f"{survivors} idea(s) passed all 4 gates and earned one run on the locked final year."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Lab report</title><style>{CSS}</style></head><body><main>
<h1>Strategy Lab report</h1>
<p class="lede">Does any simple, well-known trading idea beat buying and holding SPY, after costs and
taxes, by more than luck? {esc(conclusion)}</p>
<div class="tiles">
<div class="tile"><b>{counts["ideas"]}</b><span>ideas tested</span></div>
<div class="tile"><b>{counts["param_sets"]}</b><span>versions tried</span></div>
<div class="tile"><b>{survivors}</b><span>survivors</span></div>
</div>
<h2>How an idea is judged</h2>
<ol>
<li><b>Beat SPY on the first 4 years.</b> The best version must earn more per year than buy-and-hold SPY, after costs and an estimated tax.</li>
<li><b>Be better than luck.</b> The price history is shuffled thousands of times, which keeps every day's move but destroys any real pattern. The idea is re-optimised on each shuffled copy. If shuffled data does as well as real data, the result is luck.</li>
<li><b>Beat SPY out of sample.</b> Pick settings on the past 4 years, trade the next month, roll forward.</li>
<li><b>Be better than luck out of sample.</b> The shuffle test again, on the walk-forward run.</li>
</ol>
<p class="note">An idea is dead at the first gate it fails, and gates 3 and 4 only run for ideas that pass 1 and 2.
Every idea tested makes a lucky survivor more likely, so the bar for the shuffle tests is 0.01 divided by the
number of ideas tested so far. The most recent 12 months of data are locked away for a final exam that only a
survivor may take.</p>
<h2>All ideas at a glance</h2>
<div class="scroll"><table><tr><th class="num">#</th><th>Idea</th><th>Tested on</th><th class="num">Versions</th>
<th class="num">Best version vs SPY a year</th><th>Gate 1</th><th class="num">Shuffle p</th><th class="num">Needed below</th>
<th>Gate 2</th><th>Verdict</th></tr>{summary_rows(ideas)}</table></div>
{"".join(idea_section(i, idea) for i, idea in enumerate(ideas, 1))}
{tester_section(tester)}
{paper_section(paper)}
<h2>Assumptions and limits</h2>
<ul>
<li>{esc(note)}</li>
<li>Each verdict rests on one 4-year window. Ideas that beat SPY did so in windows where SPY fell or lagged, and the shuffle test showed no timing skill behind it.</li>
<li>The shuffle test credits timing only. It keeps each asset's total return, so an asset that simply did better earns the idea nothing.</li>
<li>Past results say nothing certain about the future. This is a research project, not advice.</li>
</ul>
</main></body></html>
"""
