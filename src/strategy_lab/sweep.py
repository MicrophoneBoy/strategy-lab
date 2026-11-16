"""Parameter sweep, heatmap, and the plateau-or-spike check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Costs
from .data import Bars
from .engine import backtest, buy_and_hold
from .strategies import Family


def sweep(family: Family, bars: Bars, benchmark_bars: Bars, costs: Costs) -> pd.DataFrame:
    """Backtest every parameter set. One row per set, scored against the benchmark."""
    if not np.array_equal(bars.dates, benchmark_bars.dates):
        raise ValueError("strategy and benchmark must cover the same bars")
    positions = family.position_matrix(bars)
    bench = buy_and_hold(benchmark_bars, costs).metrics
    rows = []
    for k, params in enumerate(family.grid):
        m = backtest(bars, positions[k], costs).metrics
        rows.append({**params, **m, "excess_cagr": m["cagr"] - bench["cagr"]})
    return pd.DataFrame(rows)


def plateau_check(table: pd.DataFrame, x: str, y: str, value: str = "excess_cagr") -> dict:
    """Is the best cell part of a broad good region, or one lucky spike?

    Compares the best cell with the average of its neighbours on the grid.
    `spike_score` is that gap measured in standard deviations of the whole grid.
    Above 1 means the best cell stands alone, which is what an overfit looks like.
    """
    grid = table.pivot(index=y, columns=x, values=value)
    values = grid.to_numpy()
    i, j = np.unravel_index(np.nanargmax(values), values.shape)
    neighbours = [
        values[a, b]
        for a in range(i - 1, i + 2)
        for b in range(j - 1, j + 2)
        if (a, b) != (i, j)
        and 0 <= a < values.shape[0]
        and 0 <= b < values.shape[1]
        and not np.isnan(values[a, b])
    ]
    best = float(values[i, j])
    spread = float(np.nanstd(values))
    neighbour_mean = float(np.mean(neighbours)) if neighbours else float("nan")
    spike_score = (best - neighbour_mean) / spread if spread > 0 and neighbours else 0.0
    return {
        "best": {x: int(grid.columns[j]), y: int(grid.index[i]), value: best},
        "neighbour_mean": neighbour_mean,
        "neighbours": len(neighbours),
        "spike_score": float(spike_score),
        "shape": "spike" if spike_score > 1.0 else "plateau",
        "share_of_grid_beating_benchmark": float(np.nanmean(values[~np.isnan(values)] > 0)),
    }


def heatmap(
    table: pd.DataFrame, x: str, y: str, path: Path, title: str, value: str = "excess_cagr"
) -> None:
    """Grid of excess return over the benchmark, in percent a year.

    Blue beats the benchmark, red trails it, gray is a tie.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#898781"
    cmap = LinearSegmentedColormap.from_list("lag_tie_beat", ["#b3372e", "#f0efec", "#2a6cb5"])
    cmap.set_bad(surface)

    grid = table.pivot(index=y, columns=x, values=value) * 100.0
    values = grid.to_numpy()
    reach = max(float(np.nanmax(np.abs(values))), 1e-9)
    norm = TwoSlopeNorm(vmin=-reach, vcenter=0.0, vmax=reach)

    fig, ax = plt.subplots(figsize=(8.4, 6.0), facecolor=surface)
    ax.set_facecolor(surface)
    image = ax.imshow(values, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    # Thin surface-coloured gaps between cells.
    ax.set_xticks(np.arange(-0.5, values.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0]), minor=True)
    ax.grid(which="minor", color=surface, linewidth=2)
    ax.tick_params(which="both", length=0, colors=muted)
    ax.set_xticks(range(values.shape[1]), [str(c) for c in grid.columns])
    ax.set_yticks(range(values.shape[0]), [str(r) for r in grid.index])
    for spine in ax.spines.values():
        spine.set_visible(False)

    best = np.unravel_index(np.nanargmax(values), values.shape)
    for (i, j), v in np.ndenumerate(values):
        if np.isnan(v):
            continue
        strong = abs(v) > 0.6 * reach
        ax.text(
            j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
            color="#ffffff" if strong else ink,
            fontweight="bold" if (i, j) == best else "normal",
        )
    ax.add_patch(
        plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor=ink, linewidth=2)
    )
    ax.set_xlabel(f"{x} window (days)", color=ink)
    ax.set_ylabel(f"{y} window (days)", color=ink)
    ax.set_title(title, color=ink, loc="left", fontsize=11)
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("excess return over the benchmark, % a year", color=ink)
    bar.ax.tick_params(colors=muted, length=0)
    bar.outline.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
