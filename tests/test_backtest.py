from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from volcast.backtest import refit_positions, run_backtests, walk_forward
from volcast.models import EWMA, HAR
from volcast.models.base import VolModel
from volcast.targets import build_panel, forward_variance


class Spy(VolModel):
    """Records what each refit was allowed to see."""

    name = "spy"
    label = "Spy"
    calls: list = []

    def fit(self, panel, horizon):
        y = forward_variance(panel["v"], horizon)
        Spy.calls.append((panel.index[-1], y.last_valid_index()))
        return self

    def forecast(self, panel, horizon):
        return pd.Series(horizon * 1e-4, index=panel.index)


@pytest.mark.parametrize("horizon", [1, 22])
def test_each_refit_sees_data_up_to_the_refit_date_and_targets_up_to_r_minus_h(raw, small_cfg, horizon):
    panel = build_panel(raw)
    Spy.calls = []
    out = walk_forward(panel, Spy, horizon, small_cfg)
    refit_dates = out["refit_date"].unique()
    assert len(Spy.calls) == len(refit_dates)
    for (fit_end, last_target), refit_date in zip(Spy.calls, refit_dates):
        assert fit_end == refit_date
        assert panel.index.get_loc(last_target) == panel.index.get_loc(fit_end) - horizon


def test_output_shape_and_timing(raw, small_cfg):
    panel = build_panel(raw)
    out = walk_forward(panel, HAR, 22, small_cfg)
    assert list(out.columns) == ["date", "model", "horizon", "forecast", "realized", "refit_date"]
    assert out["date"].iloc[0] == pd.Timestamp(small_cfg.oos_start)
    assert out["date"].iloc[-1] == panel.index[-1]
    assert out["date"].is_unique and (out["forecast"] > 0).all()
    assert out["realized"].iloc[-22:].isna().all() and out["realized"].iloc[:-22].notna().all()
    assert (out["date"] >= out["refit_date"]).all()
    positions = refit_positions(panel.index, small_cfg.oos_start, small_cfg.refit_every)
    assert list(out["refit_date"].unique()) == list(panel.index[positions])
    assert np.diff(positions).max() == 22


def test_forecast_uses_the_parameters_of_its_own_block(raw, small_cfg):
    panel = build_panel(raw)
    out = walk_forward(panel, HAR, 1, small_cfg).set_index("date")
    r = out["refit_date"].unique()[3]
    manual = HAR().fit(panel.loc[:r], 1).forecast(panel, 1)
    block = out[out["refit_date"] == r]
    np.testing.assert_allclose(block["forecast"], manual.loc[block.index], rtol=1e-12)


def test_protocol_errors(raw, small_cfg):
    panel = build_panel(raw)
    with pytest.raises(ValueError, match="rows before"):
        walk_forward(panel, EWMA, 1, replace(small_cfg, min_train=10_000))
    with pytest.raises(ValueError, match="after the last date"):
        walk_forward(panel, EWMA, 1, replace(small_cfg, oos_start="2100-01-01"))


def test_bad_forecasts_are_rejected(raw, small_cfg):
    class Broken(Spy):
        name = "broken"

        def forecast(self, panel, horizon):
            return pd.Series(0.0, index=panel.index)

    with pytest.raises(ValueError, match="non-positive"):
        walk_forward(build_panel(raw), Broken, 1, small_cfg)


def test_run_backtests_covers_every_model_and_horizon(raw, small_cfg):
    out = run_backtests(build_panel(raw), cfg=small_cfg)
    assert set(map(tuple, out[["model", "horizon"]].drop_duplicates().to_numpy())) == {
        ("ewma", 1), ("ewma", 22), ("har", 1), ("har", 22),
    }
