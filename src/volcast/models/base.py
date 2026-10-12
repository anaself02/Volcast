"""The contract every forecasting model follows (DESIGN.md §9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class VolModel(ABC):
    """A volatility model that forecasts total variance over the next `horizon` trading days.

    The walk-forward engine calls, for each refit block:

    - `fit(panel, horizon)` with `panel` truncated at the refit date. Everything derived from it
      was known at that close, and forward targets computed on it are automatically embargoed.
    - `forecast(panel, horizon)` with `panel` truncated at the block's last origin. It must return
      one value per row: the forecast made at that row's close, in decimal variance summed over
      the horizon (NaN where history is too short).

    Forecasts must be causal: the value at row t may depend only on rows up to t. The look-ahead
    check (`volcast.checks.assert_no_lookahead`) enforces this for every registered model.
    """

    name: ClassVar[str]
    label: ClassVar[str]

    @abstractmethod
    def fit(self, panel: pd.DataFrame, horizon: int) -> "VolModel":
        """Estimate parameters from `panel` (ends at the refit date). Returns self."""

    @abstractmethod
    def forecast(self, panel: pd.DataFrame, horizon: int) -> pd.Series:
        """Forecast for every row of `panel`, indexed like `panel`."""

    def info(self) -> dict:
        """Diagnostics about the latest fit, recorded per refit in results/refits.parquet.

        A `status` key other than "ok" flags a refit that needed special handling (for example a
        fallback after a failed optimization); everything else is free-form detail.
        """
        return {}
