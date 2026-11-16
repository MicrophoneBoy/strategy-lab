"""Made-up price series with a known answer, used to test the tester.

`trend=0` gives a random walk: no strategy can have a real edge on it.
`trend>0` plants one: the drift flips between +trend and -trend in long
stretches, so a trend follower that is long in the up stretches and flat in the
down stretches really does beat buy-and-hold.
"""

from __future__ import annotations

import numpy as np

from .data import Bars


def synthetic_bars(
    n: int,
    seed: int,
    trend: float = 0.0,
    regime_length: int = 80,
    volatility: float = 0.01,
) -> Bars:
    rng = np.random.default_rng(seed)
    flips = rng.random(n) < 1.0 / regime_length
    direction = np.where((np.cumsum(flips) % 2) == 0, 1.0, -1.0)
    gap = rng.normal(0.0, volatility * 0.5, n)
    intraday = rng.normal(0.0, volatility * 0.87, n) + trend * direction
    gap[0] = intraday[0] = 0.0
    log_close = np.log(100.0) + np.cumsum(gap + intraday)
    log_open = np.concatenate([[log_close[0]], log_close[:-1]]) + gap
    wiggle_up = np.abs(rng.normal(0.0, volatility * 0.4, n))
    wiggle_down = np.abs(rng.normal(0.0, volatility * 0.4, n))
    dates = np.busday_offset(np.datetime64("2000-01-03"), np.arange(n), roll="forward")
    return Bars(
        dates=dates.astype("datetime64[D]"),
        open=np.exp(log_open),
        high=np.exp(np.maximum(log_open, log_close) + wiggle_up),
        low=np.exp(np.minimum(log_open, log_close) - wiggle_down),
        close=np.exp(log_close),
    )
