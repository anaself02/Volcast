"""Shared fixtures: a synthetic market with known volatility, and a short backtest protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast.config import BacktestConfig


def simulate_bars(
    n_days: int = 900,
    seed: int = 0,
    steps: int = 78,
    overnight_share: float = 0.2,
    phi: float = 0.97,
    vol_of_vol: float = 0.25,
    start: str = "2001-01-02",
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily OHLC from a stochastic-volatility random walk with an explicit intraday path.

    Log daily variance follows an AR(1) around 1e-4 (16% annualized). Each day has an overnight
    gap carrying `overnight_share` of the variance and `steps` intraday increments carrying the
    rest; high and low come from the simulated path. Returns (raw bars with a vix column, true
    total daily variance).
    """
    rng = np.random.default_rng(seed)
    mu = np.log(1e-4)
    h = np.empty(n_days)
    h[0] = mu
    for t in range(1, n_days):
        h[t] = mu + phi * (h[t - 1] - mu) + vol_of_vol * rng.standard_normal()
    var = np.exp(h)

    gap = rng.standard_normal(n_days) * np.sqrt(overnight_share * var)
    inc = rng.standard_normal((n_days, steps)) * np.sqrt((1 - overnight_share) * var / steps)[:, None]
    path = np.cumsum(inc, axis=1)  # log price relative to the open
    oc = path[:, -1]
    log_close = np.log(100.0) + np.cumsum(gap + oc)
    log_open = log_close - oc
    log_high = log_open + np.maximum(0.0, path.max(axis=1))
    log_low = log_open + np.minimum(0.0, path.min(axis=1))

    index = pd.bdate_range(start, periods=n_days, name="date")
    bars = pd.DataFrame(
        {
            "open": np.exp(log_open),
            "high": np.exp(log_high),
            "low": np.exp(log_low),
            "close": np.exp(log_close),
            "volume": rng.integers(1_000_000, 2_000_000, n_days).astype(float),
            "vix": 100 * np.sqrt(252 * var) * np.exp(0.1 * rng.standard_normal(n_days)),
        },
        index=index,
    )
    return bars, pd.Series(var, index=index, name="true_var")


@pytest.fixture(scope="session")
def synthetic():
    return simulate_bars()


@pytest.fixture(scope="session")
def raw(synthetic):
    return synthetic[0]


@pytest.fixture(scope="session")
def small_cfg(raw):
    """Out of sample from row 300, refit every 22 rows: 28 refit blocks on 900 days."""
    return BacktestConfig(oos_start=str(raw.index[300].date()), refit_every=22, min_train=200)
