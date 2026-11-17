"""The running count of everything tried, and the verdict on every idea.

The more ideas you test, the more of them pass by luck. So every idea and
parameter grid is counted here, and the pass bar for the permutation gates is
divided by the number of ideas tested so far (a Bonferroni correction).
Dead ideas stay in the ledger with the reason they died.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import repo_root


def _path(name: str) -> Path:
    return repo_root() / "ledger" / name


def _read(name: str, default):
    path = _path(name)
    return json.loads(path.read_text()) if path.exists() else default


def _write(name: str, value) -> None:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def record_test(idea: str, grid: tuple[dict, ...], data: str) -> None:
    """Count an idea and its parameter grid. Rerunning the same grid counts once."""
    grid_hash = hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest()[:12]
    tests = _read("tests.json", [])
    if any(t["idea"] == idea and t["grid_hash"] == grid_hash and t["data"] == data for t in tests):
        return
    tests.append({"idea": idea, "grid_hash": grid_hash, "param_sets": len(grid), "data": data})
    _write("tests.json", tests)


def counts() -> dict:
    tests = _read("tests.json", [])
    return {
        "ideas": len({t["idea"] for t in tests}),
        "param_sets": sum(t["param_sets"] for t in tests),
    }


def adjusted_alpha(alpha: float, ideas: int) -> float:
    """The pass bar after counting every idea tested so far."""
    return alpha / max(1, ideas)


def record_verdict(idea: str, verdict: dict) -> None:
    verdicts = _read("verdicts.json", {})
    verdicts[idea] = verdict
    _write("verdicts.json", verdicts)


def verdicts() -> dict:
    return _read("verdicts.json", {})


def record_holdout_read(symbol: str, dates: list[str], purpose: str) -> None:
    """Log a benchmark-only read of bars inside the locked holdout period."""
    reads = _read("holdout_reads.json", [])
    entry = {"symbol": symbol, "dates": dates, "purpose": purpose}
    if entry not in reads:
        reads.append(entry)
        _write("holdout_reads.json", reads)


def test_order() -> list[dict]:
    """Every counted test, in the order the ideas were first run."""
    return _read("tests.json", [])
