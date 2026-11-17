"""Ideas 3 to 9. Each has its rules and parameter grid fixed before it was run.

Single-asset ideas return positions (1 long, 0 flat) decided at each close.
Multi-asset ideas return portfolio weights. Row t uses bars 0..t only; the
trading calendar (which day ends a month) is known ahead of time and is the one
thing that may look at the next date.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .data import Bars, Panel
from .engine import log_returns
from .portfolio import PortfolioFamily, hold_between_rebalances
from .strategies import Family, trailing_return

MONTH = 21  # trading days


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def last_trading_day_of_month(dates: np.ndarray) -> np.ndarray:
    """True where the next bar falls in a new month. The final bar is never marked."""
    month = dates.astype("datetime64[M]").astype(int)
    mask = np.zeros(len(dates), dtype=bool)
    mask[:-1] = month[1:] != month[:-1]
    return mask


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Standard deviation of the last `window` daily log returns, along axis 0.

    x holds log returns with x[0] = 0 (bar 0 has none), so the first value that
    uses `window` real returns is at index `window`. Earlier rows are NaN.
    """
    out = np.full(x.shape, np.nan)
    if window < len(x):
        windows = sliding_window_view(x[1:], window, axis=0)  # (n - window, ..., window)
        out[window:] = windows.std(axis=-1, ddof=1)
    return out


def rolling_extreme(x: np.ndarray, window: int, largest: bool) -> np.ndarray:
    """Highest or lowest of the last `window` values, today included. NaN until enough exist."""
    out = np.full(len(x), np.nan)
    if window <= len(x):
        windows = sliding_window_view(x, window)
        out[window - 1 :] = windows.max(axis=1) if largest else windows.min(axis=1)
    return out


def pick(scores: np.ndarray, k: int) -> np.ndarray:
    """Equal weight in the k highest-scoring assets of each row. Rows with any
    missing score hold nothing."""
    n, m = scores.shape
    weights = np.zeros((n, m))
    ready = ~np.isnan(scores).any(axis=1)
    order = np.argsort(-scores[ready], axis=1, kind="stable")[:, :k]
    chosen = np.zeros((int(ready.sum()), m))
    np.put_along_axis(chosen, order, 1.0 / k, axis=1)
    weights[ready] = chosen
    return weights


def panel_log_returns(close: np.ndarray) -> np.ndarray:
    r = np.zeros_like(close, dtype=float)
    r[1:] = np.diff(np.log(close), axis=0)
    return r


# --------------------------------------------------------------------------
# 3. Sector momentum rotation
# --------------------------------------------------------------------------


def sector_rotation(
    lookbacks: tuple[int, ...] = (3 * MONTH, 6 * MONTH, 12 * MONTH),
    holdings: tuple[int, ...] = (1, 3, 5),
) -> PortfolioFamily:
    grid = tuple({"lookback": w, "hold": k} for w in lookbacks for k in holdings)

    def matrix(panel: Panel) -> np.ndarray:
        returns = {w: trailing_return(panel.close, w) for w in lookbacks}
        targets = np.array([pick(returns[g["lookback"]], g["hold"]) for g in grid])
        return hold_between_rebalances(targets, last_trading_day_of_month(panel.dates), panel.close)

    return PortfolioFamily(
        name="sector_rotation",
        rationale=(
            "Investors are slow to move money toward sectors with improving news, so winners "
            "keep winning for some months. The loser is whoever sells a strong sector early "
            "or holds a weak one too long."
        ),
        grid=grid,
        weight_matrix=matrix,
    )


# --------------------------------------------------------------------------
# 4. Stocks or bonds by momentum
# --------------------------------------------------------------------------


def stocks_or_bonds(
    lookbacks: tuple[int, ...] = (3 * MONTH, 6 * MONTH, 9 * MONTH, 12 * MONTH),
) -> PortfolioFamily:
    """Panel columns must be (stocks, bonds)."""
    grid = tuple({"lookback": w} for w in lookbacks)

    def matrix(panel: Panel) -> np.ndarray:
        targets = []
        for g in grid:
            stock_return = trailing_return(panel.close[:, 0], g["lookback"])
            target = np.zeros(panel.close.shape)
            target[stock_return > 0.0, 0] = 1.0
            target[stock_return <= 0.0, 1] = 1.0  # rows with no return yet stay in cash
            targets.append(target)
        return hold_between_rebalances(
            np.array(targets), last_trading_day_of_month(panel.dates), panel.close
        )

    return PortfolioFamily(
        name="stocks_or_bonds",
        rationale=(
            "Big stock declines unfold over months, so a falling trend warns that more may "
            "follow, and bonds often rise then. The loser is the investor who holds stocks "
            "through the whole decline."
        ),
        grid=grid,
        weight_matrix=matrix,
    )


# --------------------------------------------------------------------------
# 5. Low-volatility sectors
# --------------------------------------------------------------------------


def low_vol_sectors(
    windows: tuple[int, ...] = (60, 120, 250), holdings: tuple[int, ...] = (2, 3, 4)
) -> PortfolioFamily:
    grid = tuple({"window": w, "hold": k} for w in windows for k in holdings)

    def matrix(panel: Panel) -> np.ndarray:
        returns = panel_log_returns(panel.close)
        calm = {w: -rolling_std(returns, w) for w in windows}  # higher score = calmer
        targets = np.array([pick(calm[g["window"]], g["hold"]) for g in grid])
        return hold_between_rebalances(targets, last_trading_day_of_month(panel.dates), panel.close)

    return PortfolioFamily(
        name="low_vol_sectors",
        rationale=(
            "Many investors overpay for exciting, jumpy sectors hoping for big wins, which "
            "leaves calm sectors cheap for their risk. The loser is the lottery-ticket buyer."
        ),
        grid=grid,
        weight_matrix=matrix,
    )


# --------------------------------------------------------------------------
# 6. Turn of the month
# --------------------------------------------------------------------------


def turn_of_month_days(dates: np.ndarray, last: int, first: int) -> np.ndarray:
    """True for bars inside the window: the last `last` business days of a month
    (by the calendar, holidays not skipped) and the first `first` trading days
    of a month (counted from the bars themselves)."""
    days = dates.astype("datetime64[D]")
    month = days.astype("datetime64[M]")
    next_month = (month + np.timedelta64(1, "M")).astype("datetime64[D]")
    business_days_left = np.busday_count(days, next_month) - 1  # after this bar, in its month
    index_in_month = np.zeros(len(days), dtype=int)
    for t in range(1, len(days)):
        index_in_month[t] = index_in_month[t - 1] + 1 if month[t] == month[t - 1] else 0
    # The first bar of the data may be mid-month, so its count is unknown: use the calendar.
    start = np.flatnonzero(index_in_month == 0)
    first_run_end = start[1] if len(start) > 1 else len(days)
    index_in_month[:first_run_end] += int(np.busday_count(month[0].astype("datetime64[D]"), days[0]))
    return (business_days_left < last) | (index_in_month < first)


def turn_of_month(
    last_days: tuple[int, ...] = (1, 2, 3), first_days: tuple[int, ...] = (1, 2, 3, 4)
) -> Family:
    grid = tuple({"last": a, "first": b} for a in last_days for b in first_days)

    def matrix(bars: Bars) -> np.ndarray:
        rows = []
        for g in grid:
            inside = turn_of_month_days(bars.dates, g["last"], g["first"])
            position = np.zeros(len(bars))
            position[:-1] = inside[1:]  # be long going into each bar that is inside the window
            rows.append(position)
        return np.array(rows)

    return Family(
        name="turn_of_month",
        rationale=(
            "Paychecks and pension contributions hit the market around month end, which pushes "
            "prices up on those days. The loser is whoever sells into that predictable buying, "
            "or buys just after it."
        ),
        grid=grid,
        position_matrix=matrix,
    )


# --------------------------------------------------------------------------
# 7. Volatility filter
# --------------------------------------------------------------------------


def volatility_filter(
    windows: tuple[int, ...] = (10, 20, 40),
    percentiles: tuple[int, ...] = (50, 70, 90),
    history: int = 252,
) -> Family:
    grid = tuple({"window": w, "percentile": p} for w in windows for p in percentiles)

    def matrix(bars: Bars) -> np.ndarray:
        returns = log_returns(bars.close)
        rows = {}
        for w in windows:
            vol = rolling_std(returns, w)
            first = w + history - 1  # first bar with `history` volatility readings behind it
            if first >= len(vol):
                for p in percentiles:
                    rows[(w, p)] = np.zeros(len(vol))
                continue
            past = sliding_window_view(vol[w:], history)  # each row: the last `history` readings
            thresholds = np.percentile(past, percentiles, axis=1)
            for p, threshold in zip(percentiles, thresholds):
                position = np.zeros(len(vol))
                position[first:] = vol[first:] < threshold
                rows[(w, p)] = position
        return np.array([rows[(g["window"], g["percentile"])] for g in grid])

    return Family(
        name="volatility_filter",
        rationale=(
            "Stormy markets tend to stay stormy, and those stretches have had poor returns for "
            "the risk taken. The loser is whoever must stay fully invested through them."
        ),
        grid=grid,
        position_matrix=matrix,
    )


# --------------------------------------------------------------------------
# 8. Short-term mean reversion
# --------------------------------------------------------------------------


def mean_reversion(
    lookbacks: tuple[int, ...] = (1, 2, 3, 5), drops: tuple[float, ...] = (0.01, 0.02, 0.03)
) -> Family:
    grid = tuple({"lookback": n, "drop": x} for n in lookbacks for x in drops)

    def matrix(bars: Bars) -> np.ndarray:
        returns = {n: trailing_return(bars.close, n) for n in lookbacks}
        return np.array([(returns[g["lookback"]] < -g["drop"]).astype(float) for g in grid])

    return Family(
        name="mean_reversion",
        rationale=(
            "After a sharp drop, some sellers are forced or panicked, and prices tend to bounce "
            "once that selling is done. The loser is the panicked seller."
        ),
        grid=grid,
        position_matrix=matrix,
    )


# --------------------------------------------------------------------------
# 9. Breakout
# --------------------------------------------------------------------------


def breakout_positions(close: np.ndarray, entry: int, exit_: int) -> np.ndarray:
    """Long from a new `entry`-day closing high until a new `exit_`-day closing low."""
    buy = close >= rolling_extreme(close, entry, largest=True)  # NaN compares False
    sell = close <= rolling_extreme(close, exit_, largest=False)
    event = np.where(buy, 1.0, np.where(sell, 0.0, np.nan))
    last_event = np.maximum.accumulate(np.where(np.isnan(event), -1, np.arange(len(close))))
    return np.where(last_event >= 0, event[np.maximum(last_event, 0)], 0.0)


def breakout(
    entries: tuple[int, ...] = (50, 100, 200, 250), exits: tuple[int, ...] = (20, 50)
) -> Family:
    grid = tuple({"entry": n, "exit": m} for n in entries for m in exits)
    return Family(
        name="breakout",
        rationale=(
            "People anchor on old highs and sell there, so a price that pushes through has "
            "absorbed that selling and often keeps going. The loser is whoever sold at the "
            "old high."
        ),
        grid=grid,
        position_matrix=lambda bars: np.array(
            [breakout_positions(bars.close, g["entry"], g["exit"]) for g in grid]
        ),
    )
