"""Project-wide settings: paths, horizons, backtest protocol and crisis windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"

TRADING_DAYS = 252
HORIZONS: tuple[int, ...] = (1, 22)

# Named stress periods for zoom presets and breakdowns (inclusive dates).
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "2008 crisis": ("2008-09-01", "2009-03-31"),
    "2010 flash crash": ("2010-04-15", "2010-07-15"),
    "2011 downgrade": ("2011-07-15", "2011-10-31"),
    "2015 China devaluation": ("2015-08-01", "2015-10-15"),
    "Feb 2018 vol spike": ("2018-01-15", "2018-04-15"),
    "Q4 2018 selloff": ("2018-10-01", "2019-01-31"),
    "COVID 2020": ("2020-02-01", "2020-06-30"),
    "2022 rate hikes": ("2022-01-01", "2022-10-31"),
    "April 2025 tariff shock": ("2025-03-15", "2025-05-31"),
}


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward protocol (DESIGN.md §5)."""

    oos_start: str = "2005-01-01"  # first out-of-sample forecast origin (on or after)
    refit_every: int = 22  # trading days between parameter refits
    horizons: tuple[int, ...] = HORIZONS
    min_train: int = 500  # rows required before the first out-of-sample origin
