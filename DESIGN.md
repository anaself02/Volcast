# Volcast design

> Forecast S&P 500 volatility with five models, prove out of sample which ones actually work, and turn the forecasts into risk limits backtested through real crises.

**Status:** phase 1 (walking skeleton) complete. See [§11 Build phases](#11-build-phases).
This document is the source of truth for the project. Update it whenever a decision changes.

---

## 1. Goals and non-goals

**Goals**
- Forecast SPY variance at two horizons (1 and 22 trading days) with EWMA, HAR-RV, GJR-GARCH, a Kalman-filter stochastic volatility model and LightGBM, and compare them against VIX.
- Evaluate strictly out of sample with walk-forward refits, robust loss functions and significance tests.
- Turn the forecasts into 1-day VaR / Expected Shortfall and backtest them with standard regulatory tests.
- Ship it as a live Streamlit app plus a README that states the findings.

**Non-goals**
- Forecasting prices or returns. Returns are close to unpredictable; volatility clusters and is forecastable.
- Claiming a tradable strategy. The volatility-targeting backtest only demonstrates economic value.
- Intraday data. Everything runs on free daily data.

## 2. Data

| Series | Primary source | Backup | Notes |
|---|---|---|---|
| SPY daily OHLCV, 1993 to today | `yfinance`, `auto_adjust=True` | local cache | Adjusted OHLC keeps dividends out of overnight returns |
| VIX daily close, 1990 to today | `yfinance` `^VIX` | CBOE `VIX_History.csv` | Joined onto SPY dates, forward-filled at most 3 days |
| Offline sample | `arch.data.sp500` (S&P 500 index, 1999–2018) and `arch.data.vix` (2014–2019) | n/a | Development and CI only; never presented as results |

**Rules**
- Raw downloads are cached as Parquet in `data/raw/` (gitignored). `--no-refresh` runs from the cache.
- `validate_ohlc` fails loudly on unsorted or duplicate dates, non-positive prices, NaN rows above 1%, calendar gaps over 10 days, and OHLC inconsistencies (high below open/close or low above open/close) larger than 0.1% or on more than 2% of rows. Smaller inconsistencies (rounding in adjusted data) are repaired by widening high/low and counted in the run metadata.
- The app never calls Yahoo. `yfinance` is an unofficial scraper; only the pipeline touches it.

**Sample-data quirk (measured):** the bundled S&P 500 index sample records stale opens. The open equals the previous close on ~96% of days before 2006, and through 2016 overnight moves account for at most 5% of squared returns (16–26% in 2017–18). Garman–Klass then treats the overnight jump as an intraday move with almost no range (a pure jump keeps only ~11% of its squared return), so the proxy runs 30–50% below squared returns: `E[r²]/E[v]` is 1.61 overall, 1.4–2.0 through 2016 and 1.1–1.2 in 2017–18. Model comparisons stay like for like, since every model forecasts the same proxy, but absolute levels are low. SPY is a traded ETF with real opens and shouldn't have this problem; the pipeline checks it on every run (§3).

## 3. Target definition

**Conventions**
- A forecast made at the close of day *t* (the origin) covers days *t+1 … t+h*. Everything is dated at the origin.
- Units: decimal variance summed over the horizon. Annualized vol is for display only: `vol% = 100 · sqrt(252/h · var)` (`volcast.targets.to_annual_vol`).

**Daily variance proxy**

```
v_t    = r_on,t² + GK_t
r_on,t = ln(O_t / C_{t-1})                                  overnight gap
GK_t   = ½·ln(H_t / L_t)² − (2·ln2 − 1)·ln(C_t / O_t)²      Garman–Klass, intraday
```

- GK is non-negative for any consistent OHLC bar. `v` is floored at 1e-10 only to keep logs finite.
- Under Brownian motion GK is unbiased for open-to-close variance and about 7.4x more efficient than a squared return. With discretely observed prices it runs low because sampled highs and lows understate the true range: about 9% low at 500 observations a day, 2% at 5,000 (both checked by simulation in `tests/test_targets.py`).
- **Level check on every run.** The pipeline reports `E[r²]/E[v]` (close-to-close squared return over the proxy) overall and by year in `meta.json → proxy`, and warns outside 0.8–1.25. It should sit near 1 for SPY. If it doesn't, fix that before the phase 3 VIX comparison, where absolute levels matter.

**Targets**
- `h = 1`: `y_t = v_{t+1}` — feeds the VaR layer.
- `h = 22`: `y_t = Σ_{i=1..22} v_{t+i}` — matches the ~30-calendar-day window VIX covers.

## 4. Models

All models implement `VolModel` (§9) and forecast `y_t` directly for each horizon.

| Model | Phase | Specification |
|---|---|---|
| **EWMA** | 1 ✅ | `s_t = λ·s_{t−1} + (1−λ)·v_t`, λ = 0.94 (RiskMetrics daily decay). Forecast `h·s_t`. Nothing to fit. |
| **HAR-RV** (Corsi 2009) | 1 ✅ | Direct regression per horizon: `y_t = β0 + βd·v_t + βw·mean(v_{t−4..t}) + βm·mean(v_{t−21..t}) + e_t`. OLS on variance levels, expanding window. Forecasts floored at the smallest training target. |
| **GJR-GARCH(1,1)-t** | 2 | `arch` package on returns ×100, constant mean, Student-t errors. Parameters refit every 22 days; between refits, daily forecasts filter new data with frozen parameters. 22-day forecast = sum of the analytic 1…22-step variance forecasts. |
| **Range-based SV, Kalman filter** (Alizadeh, Brandt & Diebold 2002) | 3 | State `x_t = ln σ_t`: `x_t = μ + φ(x_{t−1} − μ) + η_t`, `η ~ N(0, ση²)`. Measurement `y_t = ln(ln H_t − ln L_t) = c + x_t + ε_t`, `ε ≈ N(0, σε²)`; under Brownian motion `c ≈ 0.43`, `σε ≈ 0.29` (our simulation: 0.415 and 0.291 at 5,000 steps a day). Fix `c`, estimate `μ, φ, ση, σε` by Gaussian QML with a scalar Kalman filter written from scratch in NumPy; optimize with SciPy under transforms that keep `\|φ\| < 1` and `σ > 0`. Forecast `Σ_{k=1..h} exp(2·m_k + 2·P_k)` with `m_k = μ + φ^k(x̂_t − μ)` and `P_k = φ^{2k}·P_t + ση²(1 − φ^{2k})/(1 − φ²)`. The range misses overnight gaps, so scale forecasts by `κ = mean(v) / mean(fitted σ²)` on the training window. |
| **LightGBM** | 2 | Features at origin *t* only: HAR components and their logs, negative-return terms (`r_t·1{r_t<0}`, 5-day sum), Parkinson and GK lags, VIX level, 5-day VIX change, implied-minus-realized spread `(VIX/100)²·22/252 − Σ_{i=0..21} v_{t−i}`. Target `ln y_t`; back-transform with Duan's smearing factor (mean of exp(residuals) on the training window), otherwise forecasts are biased low and QLIKE punishes that hardest. Hyperparameters fixed, or tuned by time-series CV inside the training window only. Known limitation: trees can't extrapolate beyond the largest training target (watch late 2008). |
| **VIX benchmark** | 3 | 22-day implied variance `(VIX/100)²·22/252`. Not a model: the market's forecast. Expect it to be biased high on average (the variance risk premium). |
| LSTM | stretch | Only if everything else is done. |

## 5. Walk-forward protocol

- Out-of-sample origins: 2005-01-03 to the latest close (~5,400 days on SPY). Initial training window 1993–2004, expanding.
- Refit every 22 trading days. Between refits, parameters are frozen and forecasts update daily with new data.
- **Embargo is structural.** A model is fit on the panel truncated at the refit date *R*. A target for row *s* needs `v` up to *s+h*, so inside the truncated panel only rows *s ≤ R − h* have targets. No per-model bookkeeping is needed, and none can be forgotten.
- Hyperparameters, when any, are chosen inside the training window only.
- Forecasts for a refit block are computed on the panel truncated at the block's last origin. The look-ahead check (§9) proves features are causal, so later rows in the block can't leak into earlier origins.

## 6. Evaluation

- **Common sample:** all models are scored on the same origins (every model has a forecast and the target is observed).
- **Losses** (both robust to noise in the proxy, Patton 2011):
  - QLIKE `L = y/f − ln(y/f) − 1`: zero when perfect, penalizes under-forecasting more than over-forecasting. Headline metric.
  - MSE on variance, reported in %² units.
  - Not MAE or MSE on log vol: neither is robust to proxy noise.
  - Every table reports each loss relative to EWMA.
- **Phase 2 additions**
  - Diebold–Mariano tests vs HAR and vs EWMA, Newey–West HAC with at least `h−1` lags (22-day forecasts overlap).
  - Model Confidence Set at 90% (`arch.bootstrap.MCS`), block bootstrap with block ≈ 22.
  - Mincer–Zarnowitz regressions `y = a + b·f + e`, testing `a = 0, b = 1` with HAC errors.
  - Regimes defined with origin-time information only: trailing 22-day realized-vol percentile vs the training sample (calm < 50th, normal, stressed > 90th), plus VIX < 20 / 20–30 / > 30 once VIX is in.
- **Crisis windows** for zoom presets and breakdowns live in `config.CRISIS_WINDOWS`: 2008 crisis, 2010 flash crash, 2011 downgrade, 2015 China devaluation, Feb 2018 vol spike, Q4 2018, COVID 2020, 2022 rate hikes, April 2025 tariff shock.

## 7. Risk layer (phase 4)

- **Position:** $1M long SPY. 1-day 99% VaR and 97.5% ES from each model's 1-day forecast.
- **Distribution:** filtered historical simulation. Standardized returns `z_s = r_s / sqrt(f_{s−1})` over a trailing window (~1,000 days) of forecasts available at the origin; VaR/ES from their empirical quantiles, with a location-scale Student-t fit as a comparison.
- **Calibration:** models forecast the range-based proxy, while VaR applies to close-to-close returns. FHS absorbs any level difference because `z` is standardized against the same forecast. Report `E[r²]/E[v]` on the training window anyway.
- **Backtests:** Kupiec POF (breach count), Christoffersen independence and conditional coverage (breaches shouldn't cluster), Basel traffic light on rolling 250-day windows (green 0–4 breaches, yellow 5–9, red 10+). Stretch: Acerbi–Székely ES test.
- **Stretch, volatility targeting:** `w_t = min(2, 0.10 / σ̂_{t+1, annualized})`, daily rebalance, 2 bps per unit of turnover, vs buy-and-hold: Sharpe, max drawdown, turnover.

## 8. App

The app reads `results/` only. It never fits models or downloads data.

**Pages** (phase 1 ships one page with the first two sections)
1. **Today:** latest 1-day and 22-day forecasts per model (annualized vol), VIX, as-of date and data source.
2. **Model race:** forecast vs realized for a chosen horizon, models and crisis preset, with an accuracy table for the selected window. Phase 2 adds DM p-values, MCS membership and regimes.
3. **Risk** (phase 4): VaR/ES bands against returns with breaches marked, backtest table and traffic light.
4. **Methodology:** target, protocol and metrics in plain language.

**Results contract** (written by `volcast.pipeline`)

| File | Contents |
|---|---|
| `forecasts.parquet` | `date` (origin), `model`, `horizon`, `forecast`, `realized` (NaN until observed), `refit_date` |
| `metrics.parquet` | `model`, `horizon`, `n`, `qlike`, `mse_pct2`, `qlike_rel`, `mse_rel`, `start`, `end` |
| `panel.parquet` | `date`, `close`, `ret`, `v`, `vix` |
| `meta.json` | `source`, `source_label`, `data_start`, `data_end`, `oos_start`, `generated_at`, `models` (name → label), `horizons`, `crisis_windows`, `validation` |

## 9. Engineering

**Model contract** (`volcast.models.base.VolModel`)
- `fit(panel, horizon)`: `panel` ends at the refit date; anything computed from it was known at that close.
- `forecast(panel, horizon) -> Series`: one forecast per row of `panel`, made at that row's close, of total decimal variance over the next `horizon` days; NaN where history is too short.
- Models must be causal. Register new ones in `volcast/models/__init__.py`. The look-ahead test is parametrized over the registry, so every new model is checked automatically.

**Look-ahead check** (`volcast.checks.assert_no_lookahead`)
- Shuffle every raw row after a cutoff date, rebuild the panel, rerun the walk-forward, and assert that every forecast with origin ≤ cutoff is unchanged.
- Cutoffs include a refit date (catches embargo leaks) and a mid-block date (catches non-causal features).
- Negative controls: a model with a non-causal feature (centered rolling mean) and a model that trains on the month after its refit date must both fail the check, which proves the test has teeth.

**Other rules**
- No random train/test splits anywhere.
- Heavy compute runs in the pipeline; the app only reads Parquet.
- CI (`.github/workflows/ci.yml`) runs `pytest` on every push and pull request.
- Phase 5 adds `.github/workflows/refresh.yml`: weekdays around 22:30 UTC, run the pipeline on Yahoo data, run the tests, commit `results/` if changed. If the download fails, keep the previous results and fail the job so it's visible.

**Repo layout**

```
volcast/
├── DESIGN.md              this spec
├── CLAUDE.md              working rules for Claude Code
├── src/volcast/
│   ├── config.py          paths, horizons, backtest settings, crisis windows
│   ├── data.py            Yahoo / CBOE / sample loaders, validation, cache
│   ├── targets.py         variance proxy, forward targets, annualization
│   ├── models/            base contract, registry, ewma, har (+ garch, kalman_sv, gbm)
│   ├── backtest.py        walk-forward engine
│   ├── evaluate.py        QLIKE, MSE (+ DM, MCS, Mincer–Zarnowitz)
│   ├── checks.py          look-ahead check
│   ├── pipeline.py        CLI: data → targets → walk-forward → results/
│   └── risk.py            phase 4: VaR/ES, Kupiec, Christoffersen
├── app/Home.py            Streamlit app (reads results/ only)
├── results/               pipeline output the app reads (committed)
├── data/                  raw download cache (gitignored)
├── tests/                 includes the look-ahead test
└── .github/workflows/     CI (+ daily refresh in phase 5)
```

## 10. Decisions and known issues

- **Proxy from daily OHLC, not intraday realized variance:** free intraday history doesn't go back far enough.
- **HAR on variance levels** (canonical). OLS in levels is dominated by crisis days; a log-HAR variant is a cheap phase 2 addition if its QLIKE suffers.
- **Refit every 22 days:** a compromise between realism and runtime (GARCH refits are the slow part).
- **Range proxy vs return-based GARCH:** GARCH models close-to-close returns while the target is range-based. Check levels with Mincer–Zarnowitz before reading too much into GARCH's QLIKE.
- **Sample-data quirk:** stale index opens push the proxy 30–50% low in most sample years (§2). Caught by the proxy level check, which is now permanent.
- **App colours:** Streamlit applies its light/dark theme in the browser and can't reliably tell the app which one is showing on first load, so the chart uses one set of palette steps validated against both backgrounds (`#ffffff` and `#0e1117`), and widgets use a neutral accent (`.streamlit/config.toml`).

## 11. Build phases

- [x] **Phase 1: walking skeleton**
  - [x] Data layer (Yahoo, CBOE backup, offline sample), validation, cache
  - [x] Variance proxy and forward targets
  - [x] EWMA and HAR-RV
  - [x] Walk-forward engine with structural embargo
  - [x] QLIKE and MSE evaluation, relative to EWMA
  - [x] Look-ahead check with negative control
  - [x] Single-page Streamlit app and CI
- [ ] **Phase 2:** GJR-GARCH-t, LightGBM, Diebold–Mariano, MCS, Mincer–Zarnowitz, regimes
- [ ] **Phase 3:** Kalman SV, VIX comparison
- [ ] **Phase 4:** risk layer (VaR/ES, Kupiec, Christoffersen, traffic light; volatility targeting as stretch)
- [ ] **Phase 5:** daily refresh workflow, full app pages, README with findings, deploy to Streamlit Community Cloud

Aim: phases 1–3 live within about four weeks. A finished simple version beats an unfinished ambitious one.

## 12. Expectations and positioning

- HAR and GARCH are hard to beat. Expect LightGBM to win modestly, mostly around regime shifts. A large win means leakage until proven otherwise.
- "The simple model was statistically tied" is a good result when it's rigorously shown.
- The README is the real portfolio piece: one headline chart, three findings, then the method.
- Engineering audiences: lead with the Kalman filter and the engineering (look-ahead test, pipeline, deployment). Finance and data audiences: lead with the evaluation and the VaR backtests.

## References

- Alizadeh, Brandt & Diebold (2002). Range-based estimation of stochastic volatility models. *Journal of Finance*.
- Christoffersen (1998). Evaluating interval forecasts. *International Economic Review*.
- Corsi (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*.
- Diebold & Mariano (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
- Duan (1983). Smearing estimate: a nonparametric retransformation method. *JASA*.
- Garman & Klass (1980). On the estimation of security price volatilities from historical data. *Journal of Business*.
- Glosten, Jagannathan & Runkle (1993). *Journal of Finance*.
- Hansen, Lunde & Nason (2011). The model confidence set. *Econometrica*.
- Kupiec (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*.
- Moreira & Muir (2017). Volatility-managed portfolios. *Journal of Finance*.
- Patton (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
