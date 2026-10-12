"""LightGBM: causal features, the QLIKE (gamma) objective, tuning inside the training window only."""

import numpy as np
import pandas as pd
import pytest

from volcast.evaluate import qlike
from volcast.models import LightGBM
from volcast.targets import build_panel, forward_variance


def test_features_use_only_the_past(raw):
    panel = build_panel(raw)
    full = LightGBM.features(panel)
    cut = 500
    early = LightGBM.features(panel.iloc[: cut + 1])
    pd.testing.assert_frame_equal(early, full.iloc[: cut + 1])  # later rows can't move earlier features
    t = panel.index[cut]
    v, r, vix = panel["v"].loc[:t], panel["ret"].loc[:t], panel["vix"].loc[:t]
    row = full.loc[t]
    assert row["v66"] == pytest.approx(v.iloc[-66:].mean(), rel=1e-12)
    assert row["down5"] == pytest.approx(r.iloc[-5:].clip(upper=0).mean(), rel=1e-12)
    # VIX is lagged a day: its close is struck at 4:15 pm, after the 4:00 pm close the forecast is made at.
    assert row["vix"] == vix.iloc[-2]
    assert row["dvix5"] == pytest.approx(vix.iloc[-2] - vix.iloc[-7], rel=1e-12)
    assert row["vix_ratio"] == pytest.approx((vix.iloc[-2] / 100) ** 2 / 252 / v.iloc[-22:].mean(), rel=1e-12)


def _noisy_mean_panel(n=4000, seed=3):
    """v whose 1-day-ahead mean is a known function of today's v, with mean-one lognormal noise."""
    rng = np.random.default_rng(seed)
    v = np.empty(n)
    v[0] = 1e-4
    for t in range(n - 1):
        mean = 2e-5 + 0.8 * v[t]
        v[t + 1] = mean * np.exp(0.6 * rng.standard_normal() - 0.18)  # E[exp(0.6 z - 0.18)] = 1
    idx = pd.bdate_range("2000-01-03", periods=n, name="date")
    return pd.DataFrame({"v": v, "ret": np.sqrt(v) * rng.standard_normal(n), "vix": np.nan}, index=idx)


def test_gamma_objective_targets_the_mean_not_the_median():
    """QLIKE's minimizer is E[y | x]. A log-target fit would track the median and run low."""
    panel = _noisy_mean_panel()
    train, test = panel.iloc[:3000], panel.iloc[3000:]
    f = LightGBM().fit(train, 1).forecast(panel, 1).loc[test.index]
    y = forward_variance(panel["v"], 1).loc[test.index]
    ok = f.notna() & y.notna()
    assert f[ok].mean() / y[ok].mean() == pytest.approx(1.0, abs=0.06)
    median_like = np.exp(np.log(2e-5 + 0.8 * panel["v"]) - 0.18).loc[test.index][ok]  # what a log fit estimates
    assert median_like.mean() / y[ok].mean() < 0.9


def test_tuning_only_sees_the_training_window(raw, monkeypatch):
    panel = build_panel(raw).iloc[:800]
    calls = []
    real = LightGBM._train

    def spy(self, X, y, leaves, rounds, valid=None):
        calls.append((X.index[-1], None if valid is None else (valid[0].index[0], valid[0].index[-1])))
        return real(self, X, y, leaves, rounds, valid)

    monkeypatch.setattr(LightGBM, "_train", spy)
    h = 22
    m = LightGBM().fit(panel, h)
    last_target = panel.index[-1 - h]  # the last row whose h-day target is known at the refit date
    assert m.info()["tuned"] and m.info()["num_leaves"] in LightGBM.LEAVES
    assert len(calls) == len(LightGBM.LEAVES) + 1
    for fit_end, valid in calls[:-1]:
        v_start, v_end = valid
        assert v_end == last_target
        assert panel.index.get_loc(v_start) - panel.index.get_loc(fit_end) > h  # gap: targets never overlap
    assert calls[-1] == (last_target, None)  # the final model uses the whole window, nothing after it


def test_refits_are_deterministic_and_short_windows_skip_tuning(raw):
    panel = build_panel(raw)
    a = LightGBM().fit(panel.iloc[:700], 1).forecast(panel, 1)
    b = LightGBM().fit(panel.iloc[:700], 1).forecast(panel, 1)
    pd.testing.assert_series_equal(a, b)
    assert a.iloc[:65].isna().all() and (a.iloc[65:] > 0).all()
    small = LightGBM().fit(panel.iloc[:300], 1)
    assert not small.info()["tuned"] and small.info()["num_leaves"] == LightGBM.DEFAULT_LEAVES


def test_guards(raw):
    panel = build_panel(raw)
    with pytest.raises(ValueError, match="training rows"):
        LightGBM().fit(panel.iloc[:150], 22)
    with pytest.raises(RuntimeError):
        LightGBM().forecast(panel, 1)
    with pytest.raises(ValueError, match="horizon"):
        LightGBM().fit(panel.iloc[:400], 1).forecast(panel, 22)
    assert qlike([1.0], [1.0])[0] == 0.0
