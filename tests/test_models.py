import numpy as np
import pandas as pd
import pytest

from volcast.models import EWMA, HAR, REGISTRY, labels
from volcast.targets import build_panel


def test_registry_and_labels():
    assert set(REGISTRY) >= {"ewma", "har", "garch"}
    assert labels()["ewma"] == "EWMA" and labels()["har"] == "HAR-RV" and labels()["garch"] == "GJR-GARCH"


def test_ewma_matches_the_recursion(raw):
    panel = build_panel(raw)
    lam, v = 0.94, panel["v"].to_numpy()
    s = np.empty_like(v)
    s[0] = v[0]
    for t in range(1, len(v)):
        s[t] = lam * s[t - 1] + (1 - lam) * v[t]
    f = EWMA(lam).fit(panel, 22).forecast(panel, 22)
    np.testing.assert_allclose(f.to_numpy(), 22 * s, rtol=1e-12)


def test_ewma_rejects_bad_decay():
    with pytest.raises(ValueError):
        EWMA(1.0)


def _simulate_har(n=6000, coef=(2e-6, 0.3, 0.4, 0.2), seed=0):
    """v_{t+1} = E_t[v_{t+1}] * lognormal noise with mean one, E_t linear in the HAR terms."""
    rng = np.random.default_rng(seed)
    b0, bd, bw, bm = coef
    v = np.full(n, 1e-4)
    for t in range(22, n - 1):
        mean = b0 + bd * v[t] + bw * v[t - 4 : t + 1].mean() + bm * v[t - 21 : t + 1].mean()
        v[t + 1] = mean * np.exp(0.3 * rng.standard_normal() - 0.045)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"v": v}, index=idx)


def test_har_recovers_known_coefficients():
    truth = (2e-6, 0.3, 0.4, 0.2)
    panel = _simulate_har(coef=truth)
    m = HAR().fit(panel, 1)
    np.testing.assert_allclose(m.coef_[1:], truth[1:], atol=0.08)
    assert m.coef_[0] == pytest.approx(truth[0], abs=2e-6)


def test_har_forecasts_are_positive_and_floored(raw):
    panel = build_panel(raw)
    m = HAR().fit(panel, 22)
    f = m.forecast(panel, 22).dropna()
    assert (f >= m.floor_).all() and m.floor_ > 0
    assert f.index[0] == panel.index[21]  # needs 22 days of history


def test_har_guards():
    panel = _simulate_har(n=60)
    with pytest.raises(ValueError, match="training rows"):
        HAR().fit(panel, 1)
    big = _simulate_har(n=500)
    with pytest.raises(RuntimeError):
        HAR().forecast(big, 1)
    with pytest.raises(ValueError, match="horizon"):
        HAR().fit(big, 1).forecast(big, 22)
