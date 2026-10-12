"""LightGBM on causal volatility features, trained on the QLIKE loss (DESIGN.md §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import qlike
from ..targets import forward_variance
from .base import VolModel

PCT2 = 1e4  # targets are fit in %^2 so the booster works with numbers near one


class LightGBM(VolModel):
    """Gradient-boosted trees for the h-day variance, with features known at the close of day t.

    Objective: LightGBM's gamma regression (log link). Its loss per row, y/f + ln f, is QLIKE up to
    terms that don't depend on the forecast, so the booster minimizes the same loss the evaluation
    uses and estimates the conditional mean E[y | x] directly: no log transform of the target and
    no retransformation (smearing) correction, whose constant-variance assumption fails for
    volatility. Forecasts are positive by construction.

    Tuning happens inside each training window only. The last 20% of the rows with known targets
    (at least 60) are held out, the rows before them stop `horizon` rows earlier so no training
    target overlaps a validation one, and the number of leaves (7, 15 or 31) and of boosting rounds
    (early stopping) are picked by validation QLIKE. The final model is refit on the whole window
    with the chosen settings. Windows too short to split use a fixed default. Seeds, thread count
    and LightGBM's deterministic mode are fixed, so a refit on the same data gives the same model.
    """

    name = "gbm"
    label = "LightGBM"
    MIN_ROWS = 150  # training rows with a complete feature history and a known target
    TUNE_MIN_ROWS = 400  # below this the window isn't split for tuning
    LEAVES = (7, 15, 31)
    DEFAULT_LEAVES, DEFAULT_ROUNDS, MAX_ROUNDS, PATIENCE = 15, 100, 400, 30
    PARAMS = {
        "objective": "gamma",
        "learning_rate": 0.1,
        "min_data_in_leaf": 40,
        "lambda_l2": 1.0,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 7,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 2,  # fixed: with deterministic=True, results then repeat exactly
        "verbose": -1,
    }

    def __init__(self) -> None:
        self.booster_ = None
        self.horizon_: int | None = None
        self.info_: dict = {}

    @staticmethod
    def features(panel: pd.DataFrame) -> pd.DataFrame:
        """Everything at row t uses rows up to t only (rolling windows end at t).

        VIX enters with a one-day lag. Its daily close is struck at 4:15 pm ET, 15 minutes after the
        S&P 500 close the forecast is made at, so the same-day close would carry the first minutes of
        the period being forecast (the overnight part of the 1-day target).
        """
        v, r, vix = panel["v"], panel["ret"], panel["vix"].shift(1)
        down = r.clip(upper=0.0)  # negative returns, zero otherwise
        month = v.rolling(22).mean()
        return pd.DataFrame(
            {
                "v": v,  # today's variance proxy
                "v5": v.rolling(5).mean(),  # HAR-style averages: week, month, quarter
                "v22": month,
                "v66": v.rolling(66).mean(),
                "down5": down.rolling(5).mean(),  # leverage effect: falling markets raise volatility
                "down22": down.rolling(22).mean(),
                "vix": vix,  # the market's 30-day forecast, previous close (NaN where VIX isn't available)
                "dvix5": vix - vix.shift(5),  # its momentum
                "vix_ratio": (vix / 100) ** 2 / 252 / month,  # implied over realized variance
            },
            index=panel.index,
        )

    def _train(self, X: pd.DataFrame, y: pd.Series, leaves: int, rounds: int, valid=None):
        import lightgbm as lgb  # lazy: only the pipeline needs it

        params = self.PARAMS | {"num_leaves": leaves}
        data = lgb.Dataset(X, y, free_raw_data=False)
        if valid is None:
            return lgb.train(params, data, num_boost_round=rounds)
        Xv, yv = valid
        return lgb.train(
            params | {"metric": "gamma_deviance"}, data, num_boost_round=rounds,
            valid_sets=[lgb.Dataset(Xv, yv, reference=data)],
            callbacks=[lgb.early_stopping(self.PATIENCE, verbose=False)],
        )

    def fit(self, panel: pd.DataFrame, horizon: int) -> "LightGBM":
        X = self.features(panel)
        y = PCT2 * forward_variance(panel["v"], horizon)  # last `horizon` rows are NaN: the embargo
        ok = X["v66"].notna() & y.notna()
        X, y = X.loc[ok], y.loc[ok]
        n = len(y)
        if n < self.MIN_ROWS:
            raise ValueError(f"LightGBM needs at least {self.MIN_ROWS} training rows, got {n}")

        leaves, rounds, valid_qlike = self.DEFAULT_LEAVES, self.DEFAULT_ROUNDS, None
        if n >= self.TUNE_MIN_ROWS:
            n_valid = max(60, n // 5)
            fit_end = n - n_valid - horizon  # gap: no training target overlaps a validation target
            Xv, yv = X.iloc[n - n_valid :], y.iloc[n - n_valid :]
            scores = {}
            for candidate in self.LEAVES:
                b = self._train(X.iloc[:fit_end], y.iloc[:fit_end], candidate, self.MAX_ROUNDS, valid=(Xv, yv))
                best = max(int(b.best_iteration), 1)
                scores[candidate] = (float(qlike(yv, b.predict(Xv, num_iteration=best)).mean()), best)
            leaves = min(scores, key=lambda k: scores[k][0])
            valid_qlike, rounds = scores[leaves]

        self.booster_ = self._train(X, y, leaves, rounds)
        self.horizon_ = horizon
        self.info_ = {
            "status": "ok", "n_train": n, "tuned": valid_qlike is not None, "num_leaves": leaves,
            "rounds": rounds, "valid_qlike": None if valid_qlike is None else round(valid_qlike, 5),
        }
        return self

    def forecast(self, panel: pd.DataFrame, horizon: int) -> pd.Series:
        if self.booster_ is None:
            raise RuntimeError("fit() must be called before forecast()")
        if horizon != self.horizon_:
            raise ValueError(f"fitted for horizon {self.horizon_}, asked for {horizon}")
        X = self.features(panel)
        f = self.booster_.predict(X) / PCT2
        f[X["v66"].isna().to_numpy()] = np.nan  # not enough history yet
        return pd.Series(f, index=panel.index, name=self.name)

    def info(self) -> dict:
        return dict(self.info_)
