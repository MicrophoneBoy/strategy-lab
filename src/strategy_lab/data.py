"""Download daily bars, cache them, and keep the final holdout locked away.

The cache is split in two folders on purpose:

    data/dev/      everything before the holdout. Research reads only this.
    data/holdout/  the most recent 12 months at the time the lock was made.

`load_dev` never opens the holdout folder. The only way in is `open_holdout`,
which works once per strategy and only for a strategy that passed all 4 gates.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, repo_root

COLUMNS = ["open", "high", "low", "close", "volume"]


class HoldoutError(RuntimeError):
    """Raised when something tries to use the locked holdout improperly."""


# --------------------------------------------------------------------------
# Bars: plain numpy arrays, fast enough for thousands of permutations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Bars:
    dates: np.ndarray  # datetime64[D]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, start: int | None = None, stop: int | None = None) -> "Bars":
        s = np.s_[start:stop]
        return Bars(self.dates[s], self.open[s], self.high[s], self.low[s], self.close[s])

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Bars":
        return cls(
            dates=df.index.to_numpy().astype("datetime64[D]"),
            open=df["open"].to_numpy(dtype=float),
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            close=df["close"].to_numpy(dtype=float),
        )


@dataclass(frozen=True)
class Panel:
    """Several symbols on the same dates. Price arrays are (n_bars, n_symbols)."""

    dates: np.ndarray  # datetime64[D]
    symbols: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, start: int | None = None, stop: int | None = None) -> "Panel":
        s = np.s_[start:stop]
        return Panel(
            self.dates[s], self.symbols, self.open[s], self.high[s], self.low[s], self.close[s]
        )

    def bars(self, symbol: str) -> Bars:
        j = self.symbols.index(symbol)
        return Bars(self.dates, self.open[:, j], self.high[:, j], self.low[:, j], self.close[:, j])

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]) -> "Panel":
        """Keep only the dates every symbol has."""
        common = None
        for df in frames.values():
            common = df.index if common is None else common.intersection(df.index)
        common = common.sort_values()

        def stack(column: str) -> np.ndarray:
            return np.column_stack(
                [df.loc[common, column].to_numpy(dtype=float) for df in frames.values()]
            )

        return cls(
            dates=common.to_numpy().astype("datetime64[D]"),
            symbols=tuple(frames),
            open=stack("open"),
            high=stack("high"),
            low=stack("low"),
            close=stack("close"),
        )


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def _data_dir() -> Path:
    return repo_root() / "data"


def _lock_path() -> Path:
    return repo_root() / "holdout.lock.json"


def _ledger_dir() -> Path:
    return repo_root() / "ledger"


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def paper_keys() -> tuple[str, str]:
    """Read Alpaca PAPER keys from .env. Refuses anything that looks live."""
    from dotenv import load_dotenv

    load_dotenv(repo_root() / ".env")
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    paper = os.environ.get("ALPACA_PAPER_TRADE", "true").strip().lower()
    if not key or not secret:
        raise RuntimeError(
            "No Alpaca keys found. Copy .env.example to .env and fill in your PAPER keys."
        )
    if paper != "true":
        raise RuntimeError("ALPACA_PAPER_TRADE must be true. This project is paper only.")
    if not key.startswith("PK"):
        raise RuntimeError(
            "This does not look like a paper key (paper key IDs start with 'PK'). Refusing."
        )
    return key, secret


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[COLUMNS].dropna(how="any")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype({c: float for c in COLUMNS})


def _fetch_yahoo(symbols: tuple[str, ...], start: str) -> dict[str, pd.DataFrame]:
    """Split- and dividend-adjusted daily bars from Yahoo Finance (no key needed)."""
    import yfinance as yf

    raw = yf.download(
        list(symbols),
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    return {s: _clean(raw[s]) for s in symbols}


def _fetch_alpaca(symbols: tuple[str, ...], start: str) -> dict[str, pd.DataFrame]:
    """Split- and dividend-adjusted daily bars from Alpaca's data API."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, secret = paper_keys()
    client = StockHistoricalDataClient(key, secret)
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start, tz="UTC").to_pydatetime(),
        # The free plan serves consolidated (SIP) history older than 15 minutes.
        end=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).to_pydatetime(),
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
    )
    frame = client.get_stock_bars(request).df
    out = {}
    for s in symbols:
        d = frame.xs(s, level="symbol")
        d.index = d.index.tz_convert("America/New_York")
        out[s] = _clean(d)
    return out


def fetch(symbols: tuple[str, ...], start: str, source: str) -> dict[str, pd.DataFrame]:
    if source == "yahoo":
        frames = _fetch_yahoo(symbols, start)
    elif source == "alpaca":
        frames = _fetch_alpaca(symbols, start)
    else:
        raise ValueError(f"unknown data source: {source}")
    # Today's bar may still be forming. Drop it so every cached bar is complete.
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    return {s: df[df.index < today] for s, df in frames.items()}


# --------------------------------------------------------------------------
# Holdout lock
# --------------------------------------------------------------------------


def read_lock() -> dict | None:
    path = _lock_path()
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _create_lock(last_bar: pd.Timestamp, months: int) -> dict:
    """Written once, on the first fetch. It never moves afterwards."""
    start = last_bar - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    lock = {
        "holdout_start": start.strftime("%Y-%m-%d"),
        "last_bar_when_locked": last_bar.strftime("%Y-%m-%d"),
        "months": months,
        "rule": (
            "Bars on or after holdout_start are the final exam. No idea, sweep or gate may "
            "read them. A strategy that passes all 4 gates gets exactly one run on them."
        ),
    }
    _lock_path().write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def holdout_start() -> pd.Timestamp:
    lock = read_lock()
    if lock is None:
        raise HoldoutError("No holdout lock yet. Run `uv run lab fetch` first.")
    return pd.Timestamp(lock["holdout_start"])


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _fingerprint(frames: dict[str, pd.DataFrame]) -> str:
    """Short hash of the development data, printed next to every result."""
    h = hashlib.sha256()
    for s in sorted(frames):
        df = frames[s]
        h.update(s.encode())
        h.update(df.index.to_numpy().astype("datetime64[D]").astype("int64").tobytes())
        h.update(np.ascontiguousarray(df[COLUMNS].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:16]


def fetch_and_cache(cfg: Config, source: str | None = None) -> dict:
    """Download every symbol, create the lock if needed, write the split cache."""
    source = source or cfg.source
    frames = fetch(cfg.symbols, cfg.start, source)
    lock = read_lock() or _create_lock(frames[cfg.benchmark].index[-1], cfg.holdout_months)
    cut = pd.Timestamp(lock["holdout_start"])

    dev_dir, hold_dir = _data_dir() / "dev", _data_dir() / "holdout"
    dev_dir.mkdir(parents=True, exist_ok=True)
    hold_dir.mkdir(parents=True, exist_ok=True)

    dev_frames, summary = {}, {}
    for s, df in frames.items():
        dev, hold = df[df.index < cut], df[df.index >= cut]
        dev.to_parquet(dev_dir / f"{s}.parquet")
        hold.to_parquet(hold_dir / f"{s}.parquet")
        dev_frames[s] = dev
        summary[s] = {
            "dev_bars": len(dev),
            "dev_first": dev.index[0].strftime("%Y-%m-%d"),
            "dev_last": dev.index[-1].strftime("%Y-%m-%d"),
            "holdout_bars": len(hold),
        }
    meta = {
        "source": source,
        "adjusted_for": "splits and dividends",
        "fetched_on": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"),
        "holdout_start": lock["holdout_start"],
        "dev_fingerprint": _fingerprint(dev_frames),
        "symbols": summary,
    }
    (_data_dir() / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def read_meta() -> dict:
    path = _data_dir() / "meta.json"
    if not path.exists():
        raise FileNotFoundError("No cached data. Run `uv run lab fetch` first.")
    return json.loads(path.read_text())


def load_dev(symbol: str) -> pd.DataFrame:
    """Development data for one symbol. Never touches the holdout folder."""
    path = _data_dir() / "dev" / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {symbol}. Run `uv run lab fetch` first.")
    df = pd.read_parquet(path)
    cut = holdout_start()
    if len(df) and df.index.max() >= cut:
        raise HoldoutError(f"{symbol}: development cache contains holdout bars. Refetch.")
    return df


def load_dev_bars(symbol: str) -> Bars:
    return Bars.from_frame(load_dev(symbol))


def load_dev_panel(symbols: tuple[str, ...], benchmark: str) -> tuple[Panel, Bars]:
    """Development data for several symbols plus the benchmark, on the dates
    they all share. Returns (panel of the symbols, benchmark bars)."""
    frames = {s: load_dev(s) for s in dict.fromkeys((*symbols, benchmark))}
    everything = Panel.from_frames(frames)
    keep = [everything.symbols.index(s) for s in symbols]
    panel = Panel(
        everything.dates,
        tuple(symbols),
        everything.open[:, keep],
        everything.high[:, keep],
        everything.low[:, keep],
        everything.close[:, keep],
    )
    return panel, everything.bars(benchmark)


# --------------------------------------------------------------------------
# The one door into the holdout
# --------------------------------------------------------------------------


def _read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def open_holdout(symbol: str, strategy_id: str) -> pd.DataFrame:
    """Return the locked holdout bars for a strategy's single final exam.

    Refuses unless the strategy is recorded as having passed all 4 gates, and
    refuses a second visit by the same strategy. Each use is written to
    ledger/holdout_runs.json.
    """
    verdicts = _read_json(_ledger_dir() / "verdicts.json", {})
    verdict = verdicts.get(strategy_id)
    if not verdict or not verdict.get("passed_all_gates"):
        raise HoldoutError(f"{strategy_id} has not passed all 4 gates. Holdout stays locked.")
    runs_path = _ledger_dir() / "holdout_runs.json"
    runs = _read_json(runs_path, {})
    if strategy_id in runs:
        raise HoldoutError(f"{strategy_id} already used its one holdout run on {runs[strategy_id]}.")
    runs[strategy_id] = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")
    return pd.read_parquet(_data_dir() / "holdout" / f"{symbol}.parquet")


def add_to_cache(cfg: Config, symbols: tuple[str, ...], source: str | None = None) -> dict:
    """Download extra symbols into the existing cache without touching the rest.
    The holdout lock already exists and applies to them as well."""
    cut = holdout_start()
    frames = fetch(tuple(symbols), cfg.start, source or cfg.source)
    meta = read_meta()
    for s, df in frames.items():
        dev, hold = df[df.index < cut], df[df.index >= cut]
        dev.to_parquet(_data_dir() / "dev" / f"{s}.parquet")
        hold.to_parquet(_data_dir() / "holdout" / f"{s}.parquet")
        meta["symbols"][s] = {
            "dev_bars": len(dev),
            "dev_first": dev.index[0].strftime("%Y-%m-%d"),
            "dev_last": dev.index[-1].strftime("%Y-%m-%d"),
            "holdout_bars": len(hold),
            "fingerprint": _fingerprint({s: dev}),  # added later, so it has its own
        }
    (_data_dir() / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
