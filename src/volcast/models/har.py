"""HAR-RV (Corsi 2009): regression on daily, weekly and monthly averages of the variance proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..targets import forward_variance
from .base import VolModel


class HAR(VolModel):
    """Direct h-step HAR on variance levels, fit by OLS on an expanding window.

    y_t = b0 + bd * v_t + bw * mean(v_{t-4..t}) + bm * mean(v_{t-21..t}) + e_t,
    where y_t is the total variance over t+1..t+h. Forecasts are floored at the smallest
    target seen in training so QLIKE never sees a non-positive forecast.
    """

    name = "har"
    label = "HAR-RV"
    MIN_ROWS = 100

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.floor_: float | None = None
        self.horizon_: int | None = None
        self.n_train_: int = 0

    @staticmethod
    def features(v: pd.Series) -> pd.DataFrame:
        return pd.concat(
            {"d": v, "w": v.rolling(5).mean(), "m": v.rolling(22).mean()},
            axis=1,
        )

    def fit(self, panel: pd.DataFrame, horizon: int) -> "HAR":
        X = self.features(panel["v"])
        y = forward_variance(panel["v"], horizon)  # last `horizon` rows are NaN: the embargo
        ok = X.notna().all(axis=1) & y.notna()
        if ok.sum() < self.MIN_ROWS:
            raise ValueError(f"HAR needs at least {self.MIN_ROWS} training rows, got {int(ok.sum())}")
        A = np.column_stack([np.ones(int(ok.sum())), X.loc[ok].to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, y.loc[ok].to_numpy(), rcond=None)
        self.floor_ = float(y.loc[ok].min())
        self.horizon_ = horizon
        self.n_train_ = int(ok.sum())
        return self

    def forecast(self, panel: pd.DataFrame, horizon: int) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("fit() must be called before forecast()")
        if horizon != self.horizon_:
            raise ValueError(f"fitted for horizon {self.horizon_}, asked for {horizon}")
        X = self.features(panel["v"]).to_numpy()
        f = self.coef_[0] + X @ self.coef_[1:]
        return pd.Series(np.maximum(f, self.floor_), index=panel.index, name=self.name)
