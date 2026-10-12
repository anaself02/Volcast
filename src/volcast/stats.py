"""Forecast comparison and calibration (DESIGN.md §6).

- Diebold–Mariano (1995) on the QLIKE loss differential, with a Newey–West long-run variance and the
  Harvey, Leybourne & Newbold (1997) small-sample correction.
- The Model Confidence Set (Hansen, Lunde & Nason 2011) via `arch.bootstrap.MCS`.
- Mincer–Zarnowitz regressions of realized on forecast variance, with HAC standard errors.
- Scores by regime: inside vs outside the named stress windows, each window, and VIX terciles.

Everything is computed on the common sample (`evaluate.common_sample`): every model has a forecast
and the target is known, the same rule as the headline scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .evaluate import PCT2, common_sample, qlike, squared_error

MCS_SIZE, MCS_REPS, MCS_BLOCK, MCS_SEED = 0.10, 1000, 22, 20_111  # 90% set, block ~ one month


# ----------------------------------------------------------------------------- building blocks


def newey_west_lrv(x, lags: int) -> float:
    """Long-run variance of `x` with Bartlett weights: g0 + 2 * sum_{j<=lags} (1 - j/(lags+1)) * gj.

    Autocovariances are taken around the sample mean and divided by T. Bartlett weights keep the
    estimate non-negative.
    """
    u = np.asarray(x, dtype=float)
    u = u - u.mean()
    n = len(u)
    lrv = float(u @ u) / n
    for j in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * float(u[j:] @ u[:-j]) / n
    return lrv


def hac_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey–West (Bartlett) covariance of OLS coefficients: (X'X)^-1 S (X'X)^-1, no small-sample scaling."""
    xe = X * resid[:, None]
    S = xe.T @ xe
    for j in range(1, min(lags, len(resid) - 1) + 1):
        G = xe[j:].T @ xe[:-j]
        S += (1.0 - j / (lags + 1.0)) * (G + G.T)
    bread = np.linalg.inv(X.T @ X)
    return bread @ S @ bread


# ----------------------------------------------------------------------------- Diebold–Mariano


@dataclass(frozen=True)
class DMResult:
    stat: float  # negative: the first forecast has the lower loss
    p_value: float  # two-sided
    mean_diff: float  # mean of loss_a - loss_b
    n: int
    lags: int


def diebold_mariano(loss_a, loss_b, horizon: int) -> DMResult:
    """Test H0: equal expected loss, with d_t = loss_a - loss_b.

    DM = mean(d) / sqrt(LRV / T), LRV by Newey–West with h - 1 lags (h-step forecasts overlap, so
    d is at most MA(h - 1) under the null). The HLN correction multiplies DM by
    sqrt((T + 1 - 2h + h(h - 1) / T) / T) and compares it with Student's t on T - 1 degrees of
    freedom. For h = 1 this is exactly the one-sample t-test of d. Identical losses give NaN.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n, lags = len(d), max(int(horizon) - 1, 0)
    if n < 3:
        raise ValueError("need at least three loss differentials")
    lrv = newey_west_lrv(d, lags)
    if not np.isfinite(lrv) or lrv <= 0:
        return DMResult(np.nan, np.nan, float(d.mean()), n, lags)
    dm = d.mean() / np.sqrt(lrv / n)
    stat = dm * np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    return DMResult(float(stat), float(2 * sps.t.sf(abs(stat), df=n - 1)), float(d.mean()), n, lags)


# ----------------------------------------------------------------------------- Model Confidence Set


def model_confidence_set(
    losses: pd.DataFrame, size: float = MCS_SIZE, reps: int = MCS_REPS, block: int = MCS_BLOCK, seed: int = MCS_SEED
) -> pd.DataFrame:
    """MCS p-value per model (columns of `losses`) and membership of the 1 - `size` confidence set.

    Range statistic ("R"), stationary block bootstrap, fixed seed so reruns give the same answer.
    """
    from arch.bootstrap import MCS

    mcs = MCS(losses, size=size, reps=reps, block_size=block, method="R", bootstrap="stationary", seed=seed)
    mcs.compute()
    p = mcs.pvalues.iloc[:, 0].reindex(losses.columns)
    return pd.DataFrame({"model": losses.columns, "mcs_pvalue": p.to_numpy(), "in_mcs": losses.columns.isin(mcs.included)})


# ----------------------------------------------------------------------------- Mincer–Zarnowitz


@dataclass(frozen=True)
class MZResult:
    alpha: float  # intercept, in the units of the inputs
    beta: float
    se_alpha: float
    se_beta: float
    r2: float
    wald: float  # joint test alpha = 0, beta = 1
    p_value: float
    n: int
    lags: int


def mz_lags(n: int, horizon: int) -> int:
    """HAC lags: at least h - 1 (overlapping targets), at least the Newey–West rule 4 (n/100)^(2/9)."""
    return max(int(horizon) - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def mincer_zarnowitz(realized, forecast, lags: int) -> MZResult:
    """OLS of realized on forecast, y = a + b f + e, with a Newey–West covariance and a chi-squared(2)
    Wald test of a = 0 and b = 1 (an unbiased forecast whose errors don't depend on its level)."""
    y = np.asarray(realized, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(forecast, dtype=float)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    V = hac_cov(X, resid, lags)
    q = coef - np.array([0.0, 1.0])
    wald = float(q @ np.linalg.solve(V, q))
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    se = np.sqrt(np.diag(V))
    return MZResult(float(coef[0]), float(coef[1]), float(se[0]), float(se[1]), r2, wald,
                    float(sps.chi2.sf(wald, df=2)), len(y), int(lags))


# ----------------------------------------------------------------------------- tables for results/


def _losses(forecasts: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    wide = common_sample(forecasts, horizon)
    y = wide.pop("realized")
    return wide, y, wide.apply(lambda f: pd.Series(qlike(y, f), index=wide.index))


def significance(forecasts: pd.DataFrame, benchmarks: tuple[str, ...] = ("ewma", "har"), **mcs_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DM of every model against each benchmark, and the MCS, per horizon, on QLIKE."""
    dm_rows, mcs_frames = [], []
    for h in sorted(forecasts["horizon"].unique()):
        _, _, losses = _losses(forecasts, int(h))
        for bench in benchmarks:
            if bench not in losses:
                continue
            for model in losses.columns:
                if model != bench:
                    r = diebold_mariano(losses[model], losses[bench], int(h))
                    dm_rows.append({"horizon": int(h), "model": model, "benchmark": bench, **asdict(r)})
        m = model_confidence_set(losses, **mcs_kwargs)
        m.insert(0, "horizon", int(h))
        m.insert(2, "qlike", losses.mean().reindex(m["model"]).to_numpy())
        mcs_frames.append(m)
    dm = pd.DataFrame(dm_rows, columns=["horizon", "model", "benchmark", "stat", "p_value", "mean_diff", "n", "lags"])
    return dm, pd.concat(mcs_frames, ignore_index=True)


def calibration(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Mincer–Zarnowitz per model and horizon on the common sample; intercepts in %² like the MSE."""
    rows = []
    for h in sorted(forecasts["horizon"].unique()):
        wide, y, _ = _losses(forecasts, int(h))
        lags = mz_lags(len(y), int(h))
        for model in wide.columns:
            r = mincer_zarnowitz(PCT2 * y, PCT2 * wide[model], lags)
            rows.append({"horizon": int(h), "model": model, **asdict(r)})
    return pd.DataFrame(rows)


def _score(losses: pd.DataFrame, sq: pd.DataFrame, mask: np.ndarray, baseline: str) -> list[dict]:
    if mask.sum() == 0:
        return []
    q, m = losses[mask].mean(), sq[mask].mean()
    base_q = q.get(baseline, np.nan)
    base_m = m.get(baseline, np.nan)
    return [
        {"model": k, "n": int(mask.sum()), "qlike": float(q[k]), "mse_pct2": float(m[k]),
         "qlike_rel": float(q[k] / base_q), "mse_rel": float(m[k] / base_m)}
        for k in losses.columns
    ]


def by_regime(
    forecasts: pd.DataFrame, vix: pd.Series, windows: dict[str, tuple[str, str]], baseline: str = "ewma"
) -> pd.DataFrame:
    """Scores inside vs outside the stress windows, per window, and by VIX tercile at the origin.

    Tercile cut-offs are the 1/3 and 2/3 quantiles of VIX over each horizon's scored origins. They
    only label the days: no forecast uses them.
    """
    rows = []
    for h in sorted(forecasts["horizon"].unique()):
        wide, y, losses = _losses(forecasts, int(h))
        sq = wide.apply(lambda f: pd.Series(squared_error(y, f), index=wide.index))
        dates = wide.index
        groups: list[tuple[str, str, np.ndarray, float, float]] = []
        stressed = np.zeros(len(dates), dtype=bool)
        for name, (a, b) in windows.items():
            inside = np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
            stressed |= inside
            groups.append(("window", name, inside, np.nan, np.nan))
        groups = [("stress", "Inside stress windows", stressed, np.nan, np.nan),
                  ("stress", "Outside stress windows", ~stressed, np.nan, np.nan)] + groups
        level = vix.reindex(dates).to_numpy(dtype=float)
        if np.isfinite(level).sum() >= 30:
            lo, hi = np.nanquantile(level, [1 / 3, 2 / 3])
            groups += [
                ("vix", f"VIX below {lo:.1f}", level < lo, -np.inf, lo),
                ("vix", f"VIX {lo:.1f} to {hi:.1f}", (level >= lo) & (level <= hi), lo, hi),
                ("vix", f"VIX above {hi:.1f}", level > hi, hi, np.inf),
            ]
        for kind, label, mask, lower, upper in groups:
            for r in _score(losses, sq, mask, baseline):
                rows.append({"horizon": int(h), "kind": kind, "regime": label, "lower": lower, "upper": upper, **r})
    cols = ["horizon", "kind", "regime", "lower", "upper", "model", "n", "qlike", "mse_pct2", "qlike_rel", "mse_rel"]
    return pd.DataFrame(rows, columns=cols)
