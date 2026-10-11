"""Forecast evaluation (DESIGN.md §6).

QLIKE and MSE on variance are the two standard losses that still rank forecasts correctly when
the "true" variance is itself a noisy estimate (Patton 2011). Phase 2 adds Diebold–Mariano,
the Model Confidence Set and Mincer–Zarnowitz regressions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QLIKE_FLOOR = 1e-12
PCT2 = 1e4  # decimal variance -> percent-squared (as if returns were in %)


def qlike(realized, forecast) -> np.ndarray:
    """y/f - ln(y/f) - 1: zero for a perfect forecast; under-forecasts cost more than over-forecasts."""
    y = np.maximum(np.asarray(realized, dtype=float), QLIKE_FLOOR)
    f = np.asarray(forecast, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("forecasts must be finite and positive")
    x = y / f
    return x - np.log(x) - 1.0


def squared_error(realized, forecast) -> np.ndarray:
    """Squared error on variance in %² units."""
    return (PCT2 * (np.asarray(realized, dtype=float) - np.asarray(forecast, dtype=float))) ** 2


def common_sample(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Wide frame (date x model) plus `realized`, restricted to dates every model covers and the target is known."""
    g = forecasts[forecasts["horizon"] == horizon]
    wide = g.pivot(index="date", columns="model", values="forecast")
    realized = g.groupby("date")["realized"].first()
    return wide.join(realized).dropna()


def summarize(
    forecasts: pd.DataFrame,
    baseline: str = "ewma",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Mean QLIKE and MSE per model and horizon on the common sample, plus ratios to `baseline`."""
    fc = forecasts
    if start is not None:
        fc = fc[fc["date"] >= pd.Timestamp(start)]
    if end is not None:
        fc = fc[fc["date"] <= pd.Timestamp(end)]

    rows = []
    for h in sorted(fc["horizon"].unique()):
        wide = common_sample(fc, int(h))
        if wide.empty:
            continue
        y = wide.pop("realized")
        for model in wide.columns:
            rows.append(
                {
                    "model": model,
                    "horizon": int(h),
                    "n": len(y),
                    "qlike": float(qlike(y, wide[model]).mean()),
                    "mse_pct2": float(squared_error(y, wide[model]).mean()),
                    "start": wide.index[0],
                    "end": wide.index[-1],
                }
            )
    cols = ["model", "horizon", "n", "qlike", "mse_pct2", "qlike_rel", "mse_rel", "start", "end"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    base = out[out["model"] == baseline].set_index("horizon")
    out["qlike_rel"] = out["qlike"] / out["horizon"].map(base["qlike"])
    out["mse_rel"] = out["mse_pct2"] / out["horizon"].map(base["mse_pct2"])
    return out[cols]
