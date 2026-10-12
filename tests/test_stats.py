"""Significance and calibration tests, checked against known answers and independent implementations."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from volcast.backtest import run_backtests
from volcast.evaluate import qlike
from volcast.models import EWMA, HAR
from volcast.stats import (
    by_regime,
    calibration,
    diebold_mariano,
    hac_cov,
    mincer_zarnowitz,
    model_confidence_set,
    mz_lags,
    newey_west_lrv,
    significance,
)
from volcast.targets import build_panel

def test_newey_west_known_answer():
    x = [1.0, 2.0, 3.0, 4.0]  # deviations -1.5, -0.5, 0.5, 1.5
    assert newey_west_lrv(x, 0) == pytest.approx(1.25)  # (2.25 + 0.25 + 0.25 + 2.25) / 4
    assert newey_west_lrv(x, 1) == pytest.approx(1.25 + 2 * 0.5 * (0.75 - 0.25 + 0.75) / 4)  # = 1.5625


def _ar1(n=600, phi=0.6, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x + 0.1


def test_newey_west_and_hac_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")  # installed with arch; only an independent cross-check
    x = _ar1()
    ols = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 7, "use_correction": False})
    assert newey_west_lrv(x, 7) == pytest.approx(ols.bse[0] ** 2 * len(x), rel=1e-10)
    f = np.abs(_ar1(seed=1)) + 1
    y = 0.3 + 0.8 * f + _ar1(seed=2)
    X = sm.add_constant(f)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
    np.testing.assert_allclose(hac_cov(X, ref.resid, 5), ref.cov_params(), rtol=1e-10)


def test_dm_for_one_step_forecasts_is_the_one_sample_t_test():
    rng = np.random.default_rng(4)
    a, b = rng.gamma(2.0, 1.0, 300), rng.gamma(2.0, 1.0, 300) + 0.15
    r = diebold_mariano(a, b, horizon=1)
    t = sps.ttest_1samp(a - b, 0.0)
    assert r.stat == pytest.approx(t.statistic, rel=1e-10) and r.p_value == pytest.approx(t.pvalue, rel=1e-10)
    assert r.lags == 0 and r.stat < 0 and r.mean_diff < 0  # a has the lower loss


def test_dm_for_overlapping_forecasts_uses_hac_and_the_hln_correction():
    sm = pytest.importorskip("statsmodels.api")  # installed with arch; only an independent cross-check
    d = _ar1(n=500, phi=0.9, seed=5)
    h = 22
    r = diebold_mariano(d, np.zeros_like(d), horizon=h)
    ols = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": h - 1, "use_correction": False})
    n = len(d)
    expected = ols.params[0] / ols.bse[0] * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert r.stat == pytest.approx(expected, rel=1e-10) and r.lags == h - 1
    assert r.p_value == pytest.approx(2 * sps.t.sf(abs(expected), df=n - 1), rel=1e-10)


def test_dm_size_and_power():
    rng = np.random.default_rng(6)
    null = [diebold_mariano(rng.standard_normal(400), np.zeros(400), 1).p_value < 0.05 for _ in range(400)]
    shifted = [diebold_mariano(rng.standard_normal(400) - 0.3, np.zeros(400), 1).p_value < 0.05 for _ in range(100)]
    assert 0.025 <= np.mean(null) <= 0.08
    assert np.mean(shifted) > 0.95


def test_dm_is_undefined_for_identical_forecasts():
    x = np.linspace(1, 2, 50)
    r = diebold_mariano(x, x, 22)
    assert np.isnan(r.stat) and np.isnan(r.p_value) and r.mean_diff == 0


def test_mcs_known_answers():
    rng = np.random.default_rng(7)
    base = rng.gamma(4.0, 0.25, 800)
    clear = pd.DataFrame({"best": base, "worse": base + 0.5, "worst": base + 1.0})
    out = model_confidence_set(clear, reps=300)
    assert out.set_index("model")["in_mcs"].to_dict() == {"best": True, "worse": False, "worst": False}
    assert out.set_index("model").loc["best", "mcs_pvalue"] == 1.0
    tied = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.01, 800), "bad": base + 1.0})
    assert set(model_confidence_set(tied, reps=300).query("in_mcs")["model"]) == {"a", "b"}
    pd.testing.assert_frame_equal(model_confidence_set(tied, reps=300), model_confidence_set(tied, reps=300))


def test_mincer_zarnowitz_recovers_coefficients_and_tests_unbiasedness():
    rng = np.random.default_rng(8)
    f = rng.gamma(3.0, 1.0, 2000)
    biased = 0.5 + 2.0 * f + rng.normal(0, 1, 2000)
    r = mincer_zarnowitz(biased, f, lags=5)
    b, a = np.polyfit(f, biased, 1)
    assert (r.alpha, r.beta) == (pytest.approx(a, rel=1e-10), pytest.approx(b, rel=1e-10))
    assert r.p_value < 1e-6 and r.n == 2000
    fair = mincer_zarnowitz(f + rng.normal(0, 1, 2000), f, lags=5)
    assert fair.beta == pytest.approx(1.0, abs=0.08) and fair.p_value > 0.05
    assert mz_lags(5455, 22) == 21 and mz_lags(5476, 1) == 9


def _frame(n=300, seed=9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    y = rng.gamma(2.0, 1e-4, n)
    rows = []
    for name, scale in (("ewma", 1.3), ("har", 1.0)):
        rows.append(pd.DataFrame({"date": dates, "model": name, "horizon": 1, "forecast": y * scale + 1e-5, "realized": y}))
    return pd.concat(rows, ignore_index=True), dates


def test_by_regime_counts_and_scores_by_hand():
    fc, dates = _frame()
    windows = {"A": (str(dates[10].date()), str(dates[29].date())), "B": (str(dates[100].date()), str(dates[104].date()))}
    vix = pd.Series(np.linspace(10, 40, len(dates)), index=dates)
    out = by_regime(fc, vix, windows).set_index(["regime", "model"])
    assert out.loc[("Inside stress windows", "har"), "n"] == 25 and out.loc[("Outside stress windows", "har"), "n"] == 275
    assert out.loc[("A", "ewma"), "n"] == 20 and out.loc[("B", "ewma"), "n"] == 5
    y = fc.query("model == 'har'")["realized"].to_numpy()
    f = fc.query("model == 'har'")["forecast"].to_numpy()
    assert out.loc[("A", "har"), "qlike"] == pytest.approx(qlike(y[10:30], f[10:30]).mean())
    vix_rows = out[out["kind"] == "vix"].xs("har", level="model")
    assert vix_rows["n"].sum() == 300 and vix_rows["n"].min() >= 99
    assert (out.xs("ewma", level="model")["qlike_rel"] == 1.0).all()


def test_tables_from_a_walk_forward(raw, small_cfg):
    fc = run_backtests(build_panel(raw), {"ewma": EWMA, "har": HAR}, small_cfg)
    dm, mcs = significance(fc, reps=200)
    assert set(map(tuple, dm[["model", "benchmark"]].drop_duplicates().to_numpy())) == {("har", "ewma"), ("ewma", "har")}
    assert dm.groupby("horizon")["lags"].first().to_dict() == {1: 0, 22: 21}
    pair = dm.set_index(["horizon", "model", "benchmark"])
    for h in (1, 22):  # the two directions are the same test with the sign flipped
        assert pair.loc[(h, "har", "ewma"), "stat"] == pytest.approx(-pair.loc[(h, "ewma", "har"), "stat"])
    assert len(mcs) == 4 and mcs["mcs_pvalue"].between(0, 1).all() and mcs.groupby("horizon")["in_mcs"].any().all()
    mz = calibration(fc)
    assert len(mz) == 4 and (mz["n"] > 0).all() and mz["p_value"].between(0, 1).all()
