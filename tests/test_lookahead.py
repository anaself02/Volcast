"""The look-ahead check: every registered model, every horizon, plus a planted leak it must catch."""

import numpy as np
import pandas as pd
import pytest

from volcast.checks import assert_no_lookahead, default_cutoffs, scramble_after
from volcast.config import HORIZONS
from volcast.models import REGISTRY
from volcast.models.base import VolModel
from volcast.targets import build_panel


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registered_models_do_not_look_ahead(raw, small_cfg, name, horizon):
    assert_no_lookahead(raw, REGISTRY[name], horizon, small_cfg)


class CenteredLeak(VolModel):
    """Smooths with a centered window: uses the next five days. A classic silent leak."""

    name = "centered_leak"
    label = "Leaky"

    def fit(self, panel, horizon):
        return self

    def forecast(self, panel, horizon):
        return horizon * panel["v"].rolling(11, center=True, min_periods=1).mean()


class TrainsOnTheFuture(VolModel):
    """Fits a scale factor on data it shouldn't have: the refit date plus the whole next month."""

    name = "trains_on_future"
    label = "Leaky"
    full_v: pd.Series | None = None

    def fit(self, panel, horizon):
        end = TrainsOnTheFuture.full_v.index.get_loc(panel.index[-1])
        self.scale_ = float(TrainsOnTheFuture.full_v.iloc[: end + 22].mean())
        return self

    def forecast(self, panel, horizon):
        return pd.Series(horizon * self.scale_, index=panel.index)


def test_the_check_catches_a_non_causal_feature(raw, small_cfg):
    with pytest.raises(AssertionError, match="uses information from the future"):
        assert_no_lookahead(raw, CenteredLeak, 1, small_cfg)


def test_the_check_catches_training_on_future_data(raw, small_cfg, monkeypatch):
    # The model peeks at a global copy of the data; point it at whichever panel is being tested.
    import volcast.checks as checks

    real_walk_forward = checks.walk_forward

    def walk_forward_and_expose(panel, factory, horizon, cfg):
        TrainsOnTheFuture.full_v = panel["v"]
        return real_walk_forward(panel, factory, horizon, cfg)

    monkeypatch.setattr(checks, "walk_forward", walk_forward_and_expose)
    with pytest.raises(AssertionError, match="uses information from the future"):
        assert_no_lookahead(raw, TrainsOnTheFuture, 1, small_cfg)


def test_scramble_changes_only_rows_after_the_cutoff(raw):
    cutoff = raw.index[400]
    out = scramble_after(raw, cutoff, seed=1)
    pd.testing.assert_frame_equal(out.loc[:cutoff], raw.loc[:cutoff])
    after = raw.index > cutoff
    assert not np.allclose(out.loc[after, "close"], raw.loc[after, "close"])
    np.testing.assert_allclose(np.sort(out.loc[after, "close"]), np.sort(raw.loc[after, "close"]))


def test_default_cutoffs_hit_a_refit_date_and_a_mid_block_date(raw, small_cfg):
    panel_index = build_panel(raw).index
    refit, mid = default_cutoffs(panel_index, small_cfg)
    start = panel_index.get_loc(pd.Timestamp(small_cfg.oos_start))
    assert (panel_index.get_loc(refit) - start) % small_cfg.refit_every == 0
    assert (panel_index.get_loc(mid) - start) % small_cfg.refit_every == small_cfg.refit_every // 2
