"""The holdout lock, the cache, and the ledger of tests."""

import json

import numpy as np
import pandas as pd
import pytest

from strategy_lab import data, ledger
from strategy_lab.config import Config, Costs, TaxRates, Validation, load_config, repo_root

CFG = Config(
    symbols=("SPY", "QQQ"),
    benchmark="SPY",
    source="test",
    start="2015-01-01",
    holdout_months=12,
    costs=Costs(),
    tax=TaxRates(),
    validation=Validation(),
)


def fake_frames(end: str) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2015-01-01", end, name="date")
    rng = np.random.default_rng(0)
    frames = {}
    for symbol in CFG.symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
        frames[symbol] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
            index=index,
        )
    return frames


@pytest.fixture
def cached(lab_root, monkeypatch):
    monkeypatch.setattr(data, "fetch", lambda symbols, start, source: fake_frames("2024-06-28"))
    return data.fetch_and_cache(CFG)


def test_lock_covers_the_most_recent_12_months(cached):
    lock = data.read_lock()
    assert lock["last_bar_when_locked"] == "2024-06-28"
    assert lock["holdout_start"] == "2023-06-29"


def test_development_data_stops_before_the_holdout(cached):
    for symbol in CFG.symbols:
        dev = data.load_dev(symbol)
        assert dev.index.max() < data.holdout_start()
        assert len(dev) > 2000


def test_lock_does_not_move_when_newer_data_arrives(cached, monkeypatch):
    monkeypatch.setattr(data, "fetch", lambda symbols, start, source: fake_frames("2025-03-31"))
    data.fetch_and_cache(CFG)
    assert data.read_lock()["holdout_start"] == "2023-06-29"
    assert data.load_dev("SPY").index.max() < pd.Timestamp("2023-06-29")


def test_load_dev_refuses_a_cache_that_contains_holdout_bars(cached, lab_root):
    leaked = fake_frames("2024-06-28")["SPY"]
    leaked.to_parquet(lab_root / "data" / "dev" / "SPY.parquet")
    with pytest.raises(data.HoldoutError):
        data.load_dev("SPY")


def test_holdout_stays_locked_without_a_passing_verdict(cached):
    with pytest.raises(data.HoldoutError):
        data.open_holdout("SPY", "some_idea")
    ledger.record_verdict("some_idea", {"passed_all_gates": False})
    with pytest.raises(data.HoldoutError):
        data.open_holdout("SPY", "some_idea")


def test_holdout_opens_exactly_once_for_a_strategy_that_passed(cached):
    ledger.record_verdict("winner", {"passed_all_gates": True})
    holdout = data.open_holdout("SPY", "winner")
    assert holdout.index.min() >= data.holdout_start()
    with pytest.raises(data.HoldoutError):
        data.open_holdout("SPY", "winner")


def test_ledger_counts_each_idea_and_grid_once(lab_root):
    grid_a = ({"fast": 5, "slow": 20}, {"fast": 10, "slow": 20})
    grid_b = ({"lookback": 3},)
    ledger.record_test("sma", grid_a, "SPY")
    ledger.record_test("sma", grid_a, "SPY")  # rerun: not counted twice
    ledger.record_test("reversion", grid_b, "SPY")
    assert ledger.counts() == {"ideas": 2, "param_sets": 3}
    assert ledger.adjusted_alpha(0.01, 2) == pytest.approx(0.005)


def test_paper_keys_refuse_live_looking_keys(lab_root, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "AKLIVEKEY")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "x")
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "true")
    with pytest.raises(RuntimeError, match="paper key"):
        data.paper_keys()
    monkeypatch.setenv("ALPACA_API_KEY", "PKPAPERKEY")
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "false")
    with pytest.raises(RuntimeError, match="paper only"):
        data.paper_keys()


# ------------------------------------------------- smoke test on the real cache


needs_cache = pytest.mark.skipif(
    not (repo_root() / "data" / "meta.json").exists(),
    reason="no cached data: run `uv run lab fetch`",
)


@needs_cache
def test_smoke_cached_data_loads():
    cfg = load_config()
    cut = data.holdout_start()
    for symbol in cfg.symbols:
        df = data.load_dev(symbol)
        assert list(df.columns) == data.COLUMNS
        assert df.index.is_monotonic_increasing and df.index.is_unique
        assert not df.isna().any().any()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()
        assert (df["high"] >= df["low"]).all()
        assert df.index.max() < cut
    assert len(data.load_dev(cfg.benchmark)) > cfg.validation.train_bars


@needs_cache
def test_smoke_holdout_is_about_12_months():
    lock = json.loads((repo_root() / "holdout.lock.json").read_text())
    span = pd.Timestamp(lock["last_bar_when_locked"]) - pd.Timestamp(lock["holdout_start"])
    assert 360 <= span.days <= 366
