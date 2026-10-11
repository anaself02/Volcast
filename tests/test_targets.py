import numpy as np
import pandas as pd
import pytest

from conftest import simulate_bars
from volcast.targets import (
    build_panel,
    forward_variance,
    garman_klass,
    to_annual_vol,
    variance_proxy,
)


def test_gk_is_non_negative_for_any_consistent_bar():
    rng = np.random.default_rng(1)
    n = 10_000
    low = 100 * np.exp(-rng.uniform(0, 0.05, n))
    high = 100 * np.exp(rng.uniform(0, 0.05, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    close = low + rng.uniform(0, 1, n) * (high - low)
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    assert (garman_klass(bars) >= 0).all()


def test_gk_bias_is_small_and_shrinks_with_finer_sampling():
    # Constant variance, no overnight gap. Discretely sampled highs and lows understate the true
    # range, so GK runs low, less so the finer the sampling.
    ratios = {}
    for steps in (100, 1000):
        bars, true_var = simulate_bars(n_days=3000, seed=2, steps=steps, overnight_share=0.0, vol_of_vol=0.0)
        ratios[steps] = garman_klass(bars).mean() / true_var.mean()
    assert 0.92 < ratios[1000] < 1.02
    assert ratios[100] < ratios[1000] - 0.03


def test_gk_is_several_times_more_efficient_than_squared_returns():
    bars, _ = simulate_bars(n_days=3000, seed=4, steps=500, overnight_share=0.0, vol_of_vol=0.0)
    oc2 = np.log(bars["close"] / bars["open"]) ** 2
    assert oc2.var() / garman_klass(bars).var() > 5  # theory: ~7.4x


def test_proxy_matches_total_variance_including_the_overnight_gap():
    bars, true_var = simulate_bars(n_days=3000, seed=3, steps=2000, overnight_share=0.3, vol_of_vol=0.0)
    v = variance_proxy(bars).dropna()
    assert v.mean() == pytest.approx(true_var.mean(), rel=0.05)


def test_proxy_first_row_is_nan_and_rest_positive():
    bars, _ = simulate_bars(n_days=50)
    v = variance_proxy(bars)
    assert np.isnan(v.iloc[0])
    assert (v.iloc[1:] > 0).all()


def test_forward_variance_sums_the_next_h_days():
    v = pd.Series(np.arange(1.0, 11.0))
    y = forward_variance(v, 3)
    assert y.iloc[0] == 2 + 3 + 4
    assert y.iloc[6] == 8 + 9 + 10
    assert y.iloc[7:].isna().all()


def test_forward_variance_on_a_truncated_series_is_the_embargo():
    v = pd.Series(np.random.default_rng(0).uniform(1, 2, 100))
    full, cut = forward_variance(v, 22), forward_variance(v.iloc[:60], 22)
    pd.testing.assert_series_equal(cut.iloc[: 60 - 22], full.iloc[: 60 - 22])
    assert cut.iloc[60 - 22 :].isna().all()  # targets not yet observable at row 59


def test_forward_variance_rejects_bad_horizon():
    with pytest.raises(ValueError):
        forward_variance(pd.Series([1.0, 2.0]), 0)


def test_to_annual_vol_types_and_value():
    assert to_annual_vol(1e-4, 1) == pytest.approx(100 * np.sqrt(252e-4))
    assert to_annual_vol(22e-4, 22) == pytest.approx(to_annual_vol(1e-4, 1))
    s = to_annual_vol(pd.Series([1e-4, 4e-4], index=["a", "b"], name="x"), 1)
    assert isinstance(s, pd.Series) and list(s.index) == ["a", "b"] and s.name == "x"
    assert isinstance(to_annual_vol(np.array([1e-4]), 1), np.ndarray)


def test_build_panel(raw):
    panel = build_panel(raw)
    assert list(panel.columns) == ["open", "high", "low", "close", "ret", "r_on", "gk", "v", "vix"]
    assert panel.index[0] == raw.index[1]
    assert panel["v"].notna().all()
