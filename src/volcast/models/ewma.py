"""EWMA baseline (RiskMetrics-style exponential smoothing of the daily variance proxy)."""

from __future__ import annotations

import pandas as pd

from .base import VolModel


class EWMA(VolModel):
    """s_t = lam * s_{t-1} + (1 - lam) * v_t; forecast horizon * s_t. Nothing to estimate."""

    name = "ewma"
    label = "EWMA"

    def __init__(self, lam: float = 0.94) -> None:
        if not 0.0 < lam < 1.0:
            raise ValueError("lam must be in (0, 1)")
        self.lam = lam

    def fit(self, panel: pd.DataFrame, horizon: int) -> "EWMA":
        return self

    def forecast(self, panel: pd.DataFrame, horizon: int) -> pd.Series:
        smoothed = panel["v"].ewm(alpha=1.0 - self.lam, adjust=False).mean()
        return (horizon * smoothed).rename(self.name)
