"""GJR-GARCH(1,1,1)-t: estimation, the analytic h-day forecast, units, and the fallback on failed fits."""

import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from arch.univariate import GARCH, ConstantMean, StudentsT

from volcast.models import GJRGARCH

TRUE = {"mu": 0.05, "omega": 0.02, "alpha[1]": 0.03, "gamma[1]": 0.12, "beta[1]": 0.88, "nu": 7.0}


def _panel(n=6000, seed=1):
    """Returns simulated from a known GJR-GARCH-t, in decimal log-return units like build_panel's."""
    spec = ConstantMean(None, volatility=GARCH(p=1, o=1, q=1), distribution=StudentsT(seed=np.random.default_rng(seed)))
    sim = spec.simulate(np.array(list(TRUE.values())), n)
    idx = pd.bdate_range("2000-01-03", periods=n, name="date")
    return pd.DataFrame({"ret": sim["data"].to_numpy() / 100.0, "v": (sim["data"].to_numpy() / 100.0) ** 2}, index=idx)


def test_recovers_known_parameters():
    m = GJRGARCH().fit(_panel(), 1)
    for k, truth in TRUE.items():
        tol = {"nu": 2.5, "mu": 0.03}.get(k, 0.05 if k != "omega" else 0.015)
        assert m.params_[k] == pytest.approx(truth, abs=tol), k
    assert m.fallbacks_ == 0 and m.info()["status"] == "ok"


@pytest.mark.parametrize("horizon", [1, 22])
def test_forecast_is_arch_analytic_sum_in_decimal_variance(horizon):
    panel = _panel(n=3000)
    m = GJRGARCH().fit(panel.iloc[:2500], horizon)
    f = m.forecast(panel, horizon)
    spec = arch_model(100 * panel["ret"], mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
    ref = spec.forecast(params=m.params_, horizon=horizon, start=2600, method="analytic", reindex=False).variance
    np.testing.assert_allclose(f.iloc[2600:].to_numpy(), ref.sum(axis=1).to_numpy() / 1e4, rtol=1e-8)
    assert f.index.equals(panel.index) and (f.iloc[300:] > 0).all()


def test_falls_back_to_the_previous_refit_and_says_so(monkeypatch):
    panel = _panel(n=3000)
    real = GJRGARCH._fit_window
    calls = []

    def flaky(self, returns):
        calls.append(len(returns))
        if len(returns) == len(panel):  # this refit's own window fails
            return None, "optimizer did not converge (flag 9)"
        return real(self, returns)

    monkeypatch.setattr(GJRGARCH, "_fit_window", flaky)
    m = GJRGARCH(refit_every=22).fit(panel, 22)
    assert calls == [len(panel), len(panel) - 22]
    previous = GJRGARCH().fit(panel.iloc[:-22], 22)
    pd.testing.assert_series_equal(m.params_, previous.params_)
    info = m.info()
    assert info["status"] == "fallback" and info["fallback_refits"] == 1 and "flag 9" in info["rejected_because"]


def test_never_forecasts_from_a_failed_fit(monkeypatch):
    monkeypatch.setattr(GJRGARCH, "_fit_window", lambda self, returns: (None, "optimizer did not converge"))
    with pytest.raises(RuntimeError, match="no usable fit"):
        GJRGARCH(max_fallbacks=3).fit(_panel(n=1000), 1)
    with pytest.raises(RuntimeError, match="fit"):
        GJRGARCH().forecast(_panel(n=400), 1)
    with pytest.raises(ValueError, match="at least"):
        GJRGARCH().fit(_panel(n=100), 1)


def test_rejects_parameters_that_cannot_forecast():
    class Res:
        convergence_flag = 0
        params = pd.Series({"mu": 0.0, "omega": 0.01, "alpha[1]": 0.1, "gamma[1]": 0.2, "beta[1]": 0.85, "nu": 6.0})

    assert "persistence" in GJRGARCH._problem(Res())  # 0.1 + 0.1 + 0.85 >= 1: no finite long-run variance
    Res.params = Res.params.copy()
    Res.params["beta[1]"], Res.params["nu"] = 0.75, 1.9  # persistence 0.95 now, but nu too low
    assert "nu=1.9" in GJRGARCH._problem(Res())
    Res.params["nu"] = 6.0
    assert GJRGARCH._problem(Res()) is None
    Res.convergence_flag = 4
    assert "did not converge" in GJRGARCH._problem(Res())
