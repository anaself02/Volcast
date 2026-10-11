"""Volatility targets (DESIGN.md §3).

A forecast made at the close of day t (the origin) covers days t+1..t+h. All quantities are
decimal variances; `to_annual_vol` converts to annualized percent for display only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS

LN2 = float(np.log(2.0))
GK_COEF = 2.0 * LN2 - 1.0
V_FLOOR = 1e-10  # keeps logs finite on (rare) days with no range and no gap


def log_return(close: pd.Series) -> pd.Series:
    """Close-to-close log return."""
    return np.log(close).diff()


def overnight_return(bars: pd.DataFrame) -> pd.Series:
    """ln(open_t / close_{t-1}): the gap between yesterday's close and today's open."""
    return np.log(bars["open"] / bars["close"].shift(1))


def garman_klass(bars: pd.DataFrame) -> pd.Series:
    """Garman–Klass open-to-close variance estimator. Non-negative for any consistent bar."""
    hl = np.log(bars["high"] / bars["low"])
    co = np.log(bars["close"] / bars["open"])
    return 0.5 * hl**2 - GK_COEF * co**2


def parkinson(bars: pd.DataFrame) -> pd.Series:
    """Parkinson high-low variance estimator."""
    return np.log(bars["high"] / bars["low"]) ** 2 / (4.0 * LN2)


def variance_proxy(bars: pd.DataFrame, floor: float = V_FLOOR) -> pd.Series:
    """Daily close-to-close variance proxy: squared overnight gap plus Garman–Klass.

    The first row is NaN (no previous close).
    """
    return (overnight_return(bars) ** 2 + garman_klass(bars)).clip(lower=floor).rename("v")


def forward_variance(v: pd.Series, horizon: int) -> pd.Series:
    """Target at origin t: total variance over t+1..t+horizon. NaN where the window isn't complete.

    Computed on a truncated series, the last `horizon` rows are NaN. That is what makes the
    walk-forward embargo structural (DESIGN.md §5).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return v.rolling(horizon).sum().shift(-horizon)


def to_annual_vol(var, horizon: int):
    """Decimal variance over `horizon` days -> annualized volatility in percent.

    Accepts a scalar, array or Series and returns the same kind.
    """
    out = 100.0 * np.sqrt(TRADING_DAYS / horizon * np.asarray(var, dtype=float))
    if isinstance(var, pd.Series):
        return pd.Series(out, index=var.index, name=var.name)
    return out if np.ndim(out) else float(out)


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Derived daily panel the models consume: prices, returns, the variance proxy and VIX."""
    panel = raw[["open", "high", "low", "close"]].copy()
    panel["ret"] = log_return(raw["close"])
    panel["r_on"] = overnight_return(raw)
    panel["gk"] = garman_klass(raw)
    panel["v"] = variance_proxy(raw)
    panel["vix"] = raw["vix"] if "vix" in raw.columns else np.nan
    return panel.iloc[1:]  # the first row has no previous close
