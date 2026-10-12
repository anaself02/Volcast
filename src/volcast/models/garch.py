"""GJR-GARCH(1,1,1) with Student-t errors on close-to-close returns (Glosten, Jagannathan & Runkle 1993)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from .base import VolModel

log = logging.getLogger(__name__)

PCT2 = 1e4  # returns are fit in percent, so variances come out in %^2


class GJRGARCH(VolModel):
    """sigma2_t = omega + (alpha + gamma * 1{e_{t-1} < 0}) * e_{t-1}^2 + beta * sigma2_{t-1}.

    Fit by maximum likelihood (`arch`) on percent close-to-close log returns with a constant mean
    and Student-t errors, on an expanding window. The forecast at origin t is the analytic sum of
    the next `horizon` conditional variances, converted from %^2 to decimal variance:

        sum_{k=1..h} E_t[sigma2_{t+k}] = h * s + (sigma2_{t+1} - s) * (1 - phi^h) / (1 - phi),

    with persistence phi = alpha + gamma / 2 + beta (symmetric errors) and long-run variance
    s = omega / (1 - phi). Between refits the parameters are frozen and sigma2 filters new returns.

    A fit is rejected if the optimizer didn't converge or the parameters can't produce sensible
    forecasts (non-finite, omega <= 0, phi >= 1, nu <= 2). The model then falls back to the
    parameters of the previous refit, re-estimated on the window that refit saw (data up to
    `refit_every` rows earlier, so still causal), and records that it did. If no window in the
    last `max_fallbacks` refits gives a usable fit, `fit` raises instead of forecasting garbage.
    """

    name = "garch"
    label = "GJR-GARCH"
    MIN_ROWS = 250

    def __init__(self, refit_every: int = 22, max_fallbacks: int = 12) -> None:
        self.refit_every = refit_every
        self.max_fallbacks = max_fallbacks
        self.params_: pd.Series | None = None
        self.fallbacks_ = 0  # how many refits back the parameters come from (0 = this one)
        self.rejected_: str | None = None  # why this refit's own fit was rejected, if it was
        self.n_train_ = 0

    @staticmethod
    def _spec(returns: pd.Series):
        return arch_model(returns, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)

    @staticmethod
    def _problem(res) -> str | None:
        """Why a fit result can't be used, or None if it can."""
        if res.convergence_flag != 0:
            return f"optimizer did not converge (flag {res.convergence_flag})"
        p = res.params
        if not np.isfinite(p.to_numpy()).all():
            return "non-finite parameters"
        phi = p["alpha[1]"] + 0.5 * p["gamma[1]"] + p["beta[1]"]
        if p["omega"] <= 0 or phi >= 1 or p["nu"] <= 2:
            return f"unusable parameters (omega={p['omega']:.3g}, persistence={phi:.4f}, nu={p['nu']:.3g})"
        return None

    def _fit_window(self, returns: pd.Series) -> tuple[pd.Series | None, str | None]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # convergence is checked explicitly below
                res = self._spec(returns).fit(disp="off", show_warning=False, options={"maxiter": 500})
        except Exception as exc:  # the optimizer and arch raise several types
            return None, f"fit raised {type(exc).__name__}: {exc}"
        problem = self._problem(res)
        return (None, problem) if problem else (res.params, None)

    def fit(self, panel: pd.DataFrame, horizon: int) -> "GJRGARCH":
        returns = 100.0 * panel["ret"].dropna()
        if len(returns) < self.MIN_ROWS:
            raise ValueError(f"GJR-GARCH needs at least {self.MIN_ROWS} returns, got {len(returns)}")
        for back in range(self.max_fallbacks + 1):
            window = returns.iloc[: len(returns) - back * self.refit_every]
            if len(window) < self.MIN_ROWS:
                break
            params, problem = self._fit_window(window)
            if back == 0:
                self.rejected_ = problem
            if params is not None:
                if back:
                    log.warning(
                        "GJR-GARCH refit on %s rejected (%s); using the parameters from %d refit(s) earlier",
                        returns.index[-1].date(), self.rejected_, back,
                    )
                self.params_, self.fallbacks_, self.n_train_ = params, back, len(window)
                return self
        raise RuntimeError(
            f"GJR-GARCH: no usable fit for the refit on {returns.index[-1].date()} or the "
            f"{self.max_fallbacks} before it (latest problem: {self.rejected_})"
        )

    def forecast(self, panel: pd.DataFrame, horizon: int) -> pd.Series:
        if self.params_ is None:
            raise RuntimeError("fit() must be called before forecast()")
        p = self.params_
        returns = 100.0 * panel["ret"].dropna()
        # Conditional variance through the panel with frozen parameters (arch's own filter).
        sigma2 = self._spec(returns).fix(p).conditional_volatility.to_numpy() ** 2
        e = returns.to_numpy() - p["mu"]
        next_var = p["omega"] + (p["alpha[1]"] + p["gamma[1]"] * (e < 0)) * e**2 + p["beta[1]"] * sigma2
        phi = p["alpha[1]"] + 0.5 * p["gamma[1]"] + p["beta[1]"]
        long_run = p["omega"] / (1.0 - phi)
        total = horizon * long_run + (next_var - long_run) * (1.0 - phi**horizon) / (1.0 - phi)
        return pd.Series(total / PCT2, index=returns.index, name=self.name).reindex(panel.index)

    def info(self) -> dict:
        p = {} if self.params_ is None else {k: round(float(v), 6) for k, v in self.params_.items()}
        out = {"status": "fallback" if self.fallbacks_ else "ok", "n_train": self.n_train_, "params": p}
        if self.fallbacks_:
            out |= {"fallback_refits": self.fallbacks_, "rejected_because": self.rejected_}
        return out
