"""Walk-forward engine (DESIGN.md §5).

Expanding window, parameters refit every `refit_every` trading days. The model is fit on the
panel truncated at the refit date, so forward targets inside it stop at R - h on their own:
the embargo is structural rather than something each model has to remember.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .models import REGISTRY
from .models.base import VolModel
from .targets import forward_variance

ModelFactory = Callable[[], VolModel]


def refit_positions(index: pd.DatetimeIndex, oos_start: str, refit_every: int) -> list[int]:
    """Row positions of the refit dates: the first origin on/after `oos_start`, then every `refit_every` rows."""
    start = int(index.searchsorted(pd.Timestamp(oos_start)))
    return list(range(start, len(index), refit_every))


def walk_forward(
    panel: pd.DataFrame,
    model_factory: ModelFactory,
    horizon: int,
    cfg: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """Out-of-sample forecasts for every origin from `cfg.oos_start` to the last row.

    Returns one row per origin: date, model, horizon, forecast, realized (NaN until the whole
    horizon has been observed) and the refit date whose parameters produced the forecast.
    """
    idx = panel.index
    starts = refit_positions(idx, cfg.oos_start, cfg.refit_every)
    if not starts:
        raise ValueError(f"oos_start {cfg.oos_start} is after the last date {idx[-1].date()}")
    if starts[0] < cfg.min_train:
        raise ValueError(
            f"only {starts[0]} rows before the first out-of-sample origin; need {cfg.min_train}"
        )

    blocks = []
    name = None
    for r in starts:
        end = min(r + cfg.refit_every, len(panel)) - 1
        model = model_factory()  # fresh instance per refit: no state leaks between blocks
        name = model.name
        model.fit(panel.iloc[: r + 1], horizon)
        f = model.forecast(panel.iloc[: end + 1], horizon)
        if len(f) != end + 1:
            raise ValueError(f"{name}.forecast returned {len(f)} rows for a {end + 1}-row panel")
        blocks.append(
            pd.DataFrame(
                {"forecast": f.to_numpy()[r : end + 1], "refit_date": idx[r]},
                index=idx[r : end + 1],
            )
        )

    out = pd.concat(blocks)
    bad = ~np.isfinite(out["forecast"]) | (out["forecast"] <= 0)
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} missing or non-positive out-of-sample forecasts")
    out["realized"] = forward_variance(panel["v"], horizon).reindex(out.index)
    out.index.name = "date"
    out = out.reset_index()
    out.insert(1, "model", name)
    out.insert(2, "horizon", horizon)
    return out[["date", "model", "horizon", "forecast", "realized", "refit_date"]]


def run_backtests(
    panel: pd.DataFrame,
    models: Mapping[str, ModelFactory] | None = None,
    cfg: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """Walk-forward every model at every horizon; long format."""
    models = REGISTRY if models is None else models
    frames = [walk_forward(panel, factory, h, cfg) for factory in models.values() for h in cfg.horizons]
    return pd.concat(frames, ignore_index=True)
