import numpy as np
import pytest

from strategy_lab.data import Bars


def make_bars(close) -> Bars:
    """Bars where open = high = low = close, on consecutive business days."""
    close = np.asarray(close, dtype=float)
    dates = np.busday_offset(np.datetime64("2020-01-01"), np.arange(len(close)), roll="forward")
    return Bars(dates.astype("datetime64[D]"), close, close, close, close)


@pytest.fixture
def random_bars() -> Bars:
    rng = np.random.default_rng(42)
    return make_bars(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 1500))))


@pytest.fixture
def lab_root(tmp_path, monkeypatch):
    """Point the lab at an empty temporary folder (no real data, ledger or lock)."""
    monkeypatch.setenv("STRATEGY_LAB_ROOT", str(tmp_path))
    return tmp_path
