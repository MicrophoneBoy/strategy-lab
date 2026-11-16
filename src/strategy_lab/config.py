"""Load config.toml into plain dataclasses. All assumptions live in that file."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Folder that holds config.toml, data/, ledger/ and results/."""
    override = os.environ.get("STRATEGY_LAB_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Costs:
    """Trading costs in basis points per unit of position change."""

    fee_bps: float = 0.0
    slippage_bps: float = 2.0

    @property
    def rate(self) -> float:
        """Fraction of equity lost per unit of position change."""
        return (self.fee_bps + self.slippage_bps) / 10_000.0


@dataclass(frozen=True)
class TaxRates:
    short_term: float = 0.22
    long_term: float = 0.15


@dataclass(frozen=True)
class Validation:
    train_bars: int = 1008
    step_bars: int = 21
    n_perm_insample: int = 1000
    n_perm_walkforward: int = 200
    alpha_insample: float = 0.01
    alpha_walkforward: float = 0.01
    alpha_walkforward_short: float = 0.05
    seed: int = 20261116


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...]
    benchmark: str
    source: str
    start: str
    holdout_months: int
    costs: Costs
    tax: TaxRates
    validation: Validation


def load_config(path: Path | None = None) -> Config:
    path = path or repo_root() / "config.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    data = raw["data"]
    return Config(
        symbols=tuple(data["symbols"]),
        benchmark=data["benchmark"],
        source=data["source"],
        start=data["start"],
        holdout_months=int(data["holdout_months"]),
        costs=Costs(**raw["costs"]),
        tax=TaxRates(
            short_term=raw["tax"]["short_term_rate"],
            long_term=raw["tax"]["long_term_rate"],
        ),
        validation=Validation(**raw["validation"]),
    )
