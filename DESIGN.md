# Volcast design

> Forecast S&P 500 volatility with five models, prove out of sample which ones actually work, and turn the forecasts into risk limits backtested through real crises.

**Status:** phases 1 and 2 complete. See [§11 Build phases](#11-build-phases).
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
| **GJR-GARCH(1,1,1)-t** (Glosten, Jagannathan & Runkle 1993) | 2 ✅ | `arch` on close-to-close log returns ×100, constant mean, Student-t errors, expanding window. Between refits the conditional variance filters new returns with frozen parameters. Forecast = analytic sum of the next h conditional variances, `h·s + (σ²_{t+1} − s)(1 − φ^h)/(1 − φ)` with `φ = α + γ/2 + β` and `s = ω/(1 − φ)`, divided by 10⁴ to decimal variance (it matches `arch`'s own analytic forecast to 1e-8, unit-tested). A fit is rejected if the optimizer doesn't converge or the parameters can't forecast (non-finite, ω ≤ 0, φ ≥ 1, ν ≤ 2). The model then re-estimates on the previous refit's window, which gives that refit's parameters because fits are deterministic, going back up to 12 refits, and records it in `refits.parquet`; if nothing works it raises instead of forecasting. |
| **Range-based SV, Kalman filter** (Alizadeh, Brandt & Diebold 2002) | 3 | State `x_t = ln σ_t`: `x_t = μ + φ(x_{t−1} − μ) + η_t`, `η ~ N(0, ση²)`. Measurement `y_t = ln(ln H_t − ln L_t) = c + x_t + ε_t`, `ε ≈ N(0, σε²)`; under Brownian motion `c ≈ 0.43`, `σε ≈ 0.29` (our simulation: 0.415 and 0.291 at 5,000 steps a day). Fix `c`, estimate `μ, φ, ση, σε` by Gaussian QML with a scalar Kalman filter written from scratch in NumPy; optimize with SciPy under transforms that keep `\|φ\| < 1` and `σ > 0`. Forecast `Σ_{k=1..h} exp(2·m_k + 2·P_k)` with `m_k = μ + φ^k(x̂_t − μ)` and `P_k = φ^{2k}·P_t + ση²(1 − φ^{2k})/(1 − φ²)`. The range misses overnight gaps, so scale forecasts by `κ = mean(v) / mean(fitted σ²)` on the training window. |
| **LightGBM** | 2 ✅ | Features at the close of day t: `v_t`; 5-, 22- and 66-day means of `v`; 5- and 22-day means of negative returns `min(r, 0)` (leverage effect); the previous day's VIX close, its 5-day change, and implied over realized variance `(VIX/100)²/252 ÷ mean(v, 22 days)`. VIX is lagged a day because its close is struck at 4:15 pm ET (§10). Objective: gamma regression with a log link on `y_t` in %². Its loss is QLIKE up to constants, so the model estimates `E[y | x]` under the evaluation loss, with no log target and no smearing (§10). Tuning inside each training window only: the last 20% of rows (at least 60) are held out, the rows before them stop h rows earlier so no targets overlap, and the leaves (7, 15 or 31) and rounds (early stopping, patience 30, at most 400) are picked by validation QLIKE before a refit on the whole window. Learning rate 0.1, 40 rows per leaf, L2 1, 90% of features and 80% of rows per tree, seed 7, deterministic mode, 2 threads. Windows under 400 rows skip tuning (15 leaves, 100 rounds). Known limitation, confirmed: trees can't extrapolate beyond the largest training target, so the model is weakest in stress periods (§10). |
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
- **Phase 2 tests** (`volcast.stats`, written to the files in §8), all on the full out-of-sample common sample:
  - Diebold–Mariano on QLIKE, every model against EWMA and against HAR: `d = L_model − L_benchmark`, Newey–West variance with `h − 1` lags (22-day forecasts overlap), the Harvey–Leybourne–Newbold small-sample correction, Student t with `T − 1` degrees of freedom, two-sided. For h = 1 it is exactly the one-sample t-test of `d` (unit-tested). The p-values aren't adjusted for multiple comparisons; the MCS is the multiple-comparison answer.
  - Model Confidence Set at 90% (`arch.bootstrap.MCS`): range statistic, stationary bootstrap with 22-day blocks, 1,000 draws, seed 20111.
  - Mincer–Zarnowitz `y = a + b·f + e` in levels (%²), Newey–West covariance with `max(h − 1, ⌊4 (n/100)^{2/9}⌋)` lags, Wald χ²(2) test of `a = 0, b = 1`.
  - Regimes: inside vs outside the named stress windows, each window on its own, and VIX terciles at the forecast origin. The tercile cut-offs come from the scored days; they only label days and no forecast uses them. The originally planned trailing realized-vol percentile regimes are not built (§10).
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

**Page, top to bottom** (`app/Home.py`). Dense and quiet, like an internal risk dashboard: a neutral interface, colour only for data series and status, tabular figures throughout, units in column headers, ISO dates.
- **Status strip:** one terminal-style line above everything: data source, data through, last refit, out-of-sample window and days scored per horizon. The source, dates and window come from `meta.json`; the last refit date from `forecasts.parquet` (`refit_date`) and the days scored from `metrics.parquet`, since `meta.json` doesn't carry them.
- **Header:** the name and one plain sentence on what the page shows. The offline sample shows an info banner and a failed proxy check (`meta.json → proxy.ok` false) shows a warning. Neither is ever hidden.
- **Headline card:** for each horizon, the most accurate model by name and how much lower its mean QLIKE is than EWMA's over the full out-of-sample period, from `metrics.parquet`. Under each number, whether that edge is significant at 5% against EWMA and against HAR-RV (Diebold–Mariano, `dm.parquet`), and the 90% Model Confidence Set per horizon (`mcs.parquet`). On the sample the card is labelled "Development sample, not a finding". (Phase 1 showed a "not yet tested for statistical significance" line here; the tests replaced it.)
- **Latest forecast card**, the same height as the headline card: the most accurate model's forecast per horizon and VIX, each with its change from the prior day and, for the models, its gap to realized volatility over the matching recent period ("3.3 pts above realized vol over the last 22 days (11.1%)"). An "All models" popover lists every model's latest forecast.
- **Forecast vs. realized:** one filter row (horizon, models, period, scale) scopes both the chart and the accuracy table. Period defaults to the last 2 years and offers the full history, each named stress period and a custom range. Scale defaults to log on the full history (so calm years aren't squashed by 2008 and 2020) and linear elsewhere, and the toggle overrides it. Realized volatility is drawn last, on top, a little thicker than the forecasts, in neutral ink (dark grey on the light theme, light grey on the dark one). Every stress period in view is shaded and labelled: full names when they all fit, short ones ("GFC", "COVID") when they don't, in up to three staggered rows sized for a 375px phone; a label that still can't fit is left to the tooltip, which names every band. Lines end in direct labels where they don't collide (most accurate model first), next to a legend, a crosshair and a unified tooltip. Zoom is by period only (no drag), so the chart doesn't capture scrolling on phones.
- **Accuracy:** both horizons for the selected window and models, on summarize()'s common sample. Columns: QLIKE, QLIKE vs. EWMA in words, MSE (%²), MSE vs. EWMA in words, Bias (×) and days scored. Bias is mean forecast over mean realized variance on the same days (1.00 is unbiased), computed in the app from `forecasts.parquet`; it is descriptive only. The best model per horizon is shown by a neutral row tint. A caption says to read accuracy and bias together, since QLIKE penalizes under-forecasting more than over-forecasting.
- **Model comparison** (full out-of-sample period, independent of the controls above), three tabs: *Significance* (QLIKE, Diebold–Mariano statistic and p-value against EWMA and HAR-RV, MCS p-value and membership, MCS rows shaded), *Calibration* (Mincer–Zarnowitz α, β, R² and the joint p-value), *By regime* (QLIKE per model inside and outside the stress windows and by VIX tercile, best per row shaded, each window in an expander). Results written before phase 2 lack these files; the section is then hidden and the headline says so.
- **Data behind the chart** (the plotted values, a dash where realized isn't known yet) and **How this works** (models, scores, tests, proxy check, GARCH fallbacks in this run, what isn't built yet) are expanders.
- **Empty and loading states:** no models selected, an unfinished custom range, or a window with no scored forecasts show a one-line instruction instead of an empty chart or table. Missing results show how to run the pipeline. The first load shows a spinner.
- **Chrome:** `toolbarMode = "minimal"` hides Deploy and the developer menu, so the theme follows the viewer's system setting. The only custom CSS is one block at the top of `app/Home.py`, hooked on `data-testid` attributes and the app's own `vc-*` classes.

**Results contract** (written by `volcast.pipeline`)

| File | Contents |
|---|---|
| `forecasts.parquet` | `date` (origin), `model`, `horizon`, `forecast`, `realized` (NaN until observed), `refit_date` |
| `metrics.parquet` | `model`, `horizon`, `n`, `qlike`, `mse_pct2`, `qlike_rel`, `mse_rel`, `start`, `end` |
| `panel.parquet` | `date`, `close`, `ret`, `v`, `vix` |
| `meta.json` | `source`, `source_label`, `data_start`, `data_end`, `oos_start`, `refit_every`, `generated_at`, `models` (name → label), `horizons`, `crisis_windows`, `proxy`, `validation`, `tests` (settings of the tests below), `refit_issues` (`"model/h"` → refits that needed special handling) |
| `dm.parquet` | `horizon`, `model`, `benchmark` (`ewma` or `har`), `stat`, `p_value`, `mean_diff` (mean QLIKE of model minus benchmark), `n`, `lags` |
| `mcs.parquet` | `horizon`, `model`, `qlike`, `mcs_pvalue`, `in_mcs` (in the 90% set) |
| `mz.parquet` | `horizon`, `model`, `alpha` (%²), `beta`, `se_alpha`, `se_beta`, `r2`, `wald`, `p_value`, `n`, `lags` |
| `regimes.parquet` | `horizon`, `kind` (`stress`, `window`, `vix`), `regime`, `lower`, `upper` (VIX bounds), `model`, `n`, `qlike`, `mse_pct2`, `qlike_rel`, `mse_rel` |
| `refits.parquet` | `model`, `horizon`, `refit_date`, `status` (`ok` or `fallback`), `n_train`, `detail` (JSON: GARCH parameters and any fallback; LightGBM's tuned leaves, rounds and validation QLIKE) |

The four phase-1 files keep their columns; the phase-2 tables are new files, so older readers are unaffected.

## 9. Engineering

**Model contract** (`volcast.models.base.VolModel`)
- `fit(panel, horizon)`: `panel` ends at the refit date; anything computed from it was known at that close.
- `forecast(panel, horizon) -> Series`: one forecast per row of `panel`, made at that row's close, of total decimal variance over the next `horizon` days; NaN where history is too short.
- `info() -> dict` (optional): diagnostics about the latest fit. `walk_forward(..., on_refit=...)` hands every fitted model to a callback, and the pipeline writes `info()` to `refits.parquet`; the callback can't change the forecasts.
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
- **App colours:** Streamlit applies its light/dark theme in the browser and can't reliably tell the app which one is showing on first load (`st.context.theme` can be wrong then), so the chart uses one set of palette steps validated against both backgrounds (`#ffffff` and `#0e1117`) and a neutral wash for stress periods. Widgets use one accent, a purple that no chart series uses, stepped per mode (`#7c3eb3` light, `#b893d4` dark, in `.streamlit/config.toml`). It has at least 4.5:1 contrast as text on its background and is at least 15 OKLab ΔE from every series colour (8 under protan/deutan simulation). It replaced a neutral grey accent that made selected controls look dimmer than unselected ones in dark mode and put white tag text on a 2.6:1 background. Models are now pills rather than a multiselect, so no white-on-accent text remains.
- **Chart labels:** the plot's width is only known in the browser, so stress-period labels are placed for a phone-sized plot and skipped when they would collide; the tooltip names every band. Line labels sit at the right edge and are dropped, not nudged, when two would overlap.
- **App colours, second pass (supersedes the purple accent above):** the interface went neutral for a sober, information-dense look, with colour reserved for data series and status. The accent is ink, `#31312e` light and `#e4e3dd` dark (13:1 and 14.7:1 on their backgrounds). Selected pills and segments read by their border and tint; the date picker shows its selected day as dark text on the light accent. The best-row tint uses the same neutral wash as the stress bands.
- **Realized line colour per theme:** Streamlit's theme colours don't reach Plotly traces (checked: an uncoloured trace stays Plotly blue in both modes), so the app picks the realized ink from `st.context.theme.type`: the palette's secondary ink, `#52514e` on light (7.9:1) and `#c3c2b7` on dark (10.6:1), with the old `#7d7c76` (4.2:1 and 4.5:1) when the type is unknown, as in tests. In Streamlit 1.64 the type was right on first load in both modes. A live OS theme switch doesn't rerun the app, so the line keeps the old grey until the next interaction (2.4:1 or 1.8:1 against the other background in the meantime). The palette validator's categorical checks don't apply to a neutral: it passes separation from the series (ΔE ≥ 22) and contrast, and fails the chroma floor by construction. This is the one place where the "validated against both backgrounds" rule above gives way to a theme-specific colour.
- **Dense daily realized line:** with more than about four years of the 1-day horizon, the daily realized line on top would bury the forecasts, so it becomes translucent there. It is still drawn on top.
- **Label widths:** stress-period labels are fitted with the chart font's measured glyph advances (Source Sans, `GLYPH_EM` in `app/Home.py`), within a pixel of the rendered width; a flat per-character estimate was wrong by up to 30% in both directions.
- **Defaults:** the chart opens on the last 2 years (readable, and what a desk would check first); the headline card always reports the full out-of-sample period from `metrics.parquet`, so the default window never changes the headline.
- **GARCH fallback without shared state:** `walk_forward` builds a fresh model per refit, so a failed fit can't reach the previous refit's object. It re-estimates on the window the previous refit saw (`refit_every` rows shorter), which reproduces that refit's parameters exactly because fits are deterministic, and is still causal. Every such case is logged and written to `refits.parquet`. On SPY 2005–2026 it never happened (0 of 498 fits).
- **GARCH forecasts in closed form:** `arch`'s analytic forecast recomputes every origin and took about 1.2 s per call (10 minutes over the backtest). The model takes the conditional variance from `arch` (`fix(params)`) and applies the closed-form h-step sum; it matches `arch`'s analytic forecast to 1e-8 after the first ~100 rows.
- **LightGBM objective, gamma instead of `ln y` plus smearing:** the plan was a log target with Duan's smearing factor. Smearing assumes the log residuals have the same variance everywhere, which volatility violates, so the correction is wrong where it matters. The gamma objective's loss is QLIKE up to constants, so the booster estimates the conditional mean under the evaluation loss directly (a unit test checks it tracks the mean, where a log fit tracks the median and runs low).
- **LightGBM run settings:** learning rate 0.1, at most 400 rounds, patience 30 and 2 threads were set for runtime (about 0.2 s per refit) before any out-of-sample result was seen, and weren't revisited after.
- **VIX enters LightGBM with a one-day lag:** VIX's daily close is struck at 4:15 pm ET, 15 minutes after the S&P 500 close the forecast is made at, so the same-day close overlaps the first minutes of the period being forecast (the overnight part of the 1-day target). On SPY, same-day VIX gave a 1-day QLIKE of 0.349 against 0.363 with the lag: up to 3.4 percentage points of the edge over HAR could come from those 15 minutes, and intraday VIX data would be needed to tell how much is legitimate. The lag removes the overlap by construction.
- **Leakage review of LightGBM's 1-day edge** (13.7% lower QLIKE than HAR before the VIX lag, 10.3% after): the look-ahead check passes unchanged at both horizons; tuning only touches rows inside the training window (unit-tested); the edge doesn't extend to MSE (16.31 against HAR's 13.95), where a leak would usually help too; it holds in 21 of 22 years (2008 is the exception); and it decomposes into plausible parts: fitting HAR's own inputs on QLIKE with a log link gains 3.8%, LightGBM on HAR's inputs alone gains nothing (+1.6%), negative returns bring it to 5.6% and lagged VIX to 10.3%. Conclusion: no leak found beyond the VIX timing, which is now removed.
- **HAR fitted on QLIKE is a cheap improvement, not built:** the same three HAR inputs with a log link, fitted by minimizing QLIKE, beat OLS-in-levels HAR by 3.8% at 1 day (DM p < 0.001) and 3.9% at 22 days (p = 0.36). A candidate for phase 3, since HAR is the benchmark everything is compared with.
- **Model colours:** LightGBM is plum `#a063a5`, not the documented fourth slot (amber `#c98500`): amber passes the adjacent check but fails all-pairs against HAR-RV's orange (normal-vision ΔE 10.6, CVD 4.8), and HAR-RV and LightGBM are the two lines most often side by side. With plum, EWMA, HAR-RV, GJR-GARCH and LightGBM pass every validator check with all pairs compared on both `#ffffff` and `#0e1117` (worst CVD ΔE 9.4, normal-vision 16.6, contrast ≥ 4.3:1). The old provisional slot for LightGBM, magenta, collapses with GJR-GARCH's aqua under deuteranopia (ΔE 1.6). Kalman SV and VIX keep provisional colours to be validated in phase 3.
- **Regimes, as built:** stress windows (pooled, and each window) and VIX terciles at the origin replace the planned trailing realized-vol percentile regimes; the terciles' cut-offs come from the scored days, which only labels them. The percentile regimes can be added later without changing the file format (a new `kind`).
- **Results, SPY 2005–2026 (first full run):** see the summary in the app and `dm.parquet`. LightGBM is best at 1 day and alone in the 90% MCS; at 22 days HAR has the lowest QLIKE but its edge over EWMA is not significant at 5% (p = 0.065), and HAR, GJR-GARCH and LightGBM can't be separated (all in the MCS). LightGBM is worst inside the stress windows at 22 days, as the tree-extrapolation limit predicted, and its forecasts run low on average (bias 0.84 at 1 day and 0.82 at 22 days; Mincer–Zarnowitz β 1.38 at 1 day): it can't reach crisis levels and earns its QLIKE edge in calm periods.

## 11. Build phases

- [x] **Phase 1: walking skeleton**
  - [x] Data layer (Yahoo, CBOE backup, offline sample), validation, cache
  - [x] Variance proxy and forward targets
  - [x] EWMA and HAR-RV
  - [x] Walk-forward engine with structural embargo
  - [x] QLIKE and MSE evaluation, relative to EWMA
  - [x] Look-ahead check with negative control
  - [x] Single-page Streamlit app and CI
  - [x] App design pass: headline result, latest forecast, stress-period bands, plain-language accuracy table, light/dark and phone layouts
  - [x] App design pass 2: status strip, neutral interface, significance caveat under the headline, 2-year default window, log/linear scale, realized line on top, labels on every stress band, Bias column
- [x] **Phase 2: models and tests**
  - [x] GJR-GARCH(1,1,1)-t with explicit fallback on failed fits
  - [x] LightGBM with causal features, QLIKE (gamma) objective, tuning inside each training window
  - [x] Diebold–Mariano (Newey–West, HLN correction) against EWMA and HAR
  - [x] Model Confidence Set (90%, fixed seed)
  - [x] Mincer–Zarnowitz with HAC errors and a joint test
  - [x] Scores by regime: stress windows and VIX terciles
  - [x] New results files and app sections (significance in the headline, model comparison tabs)
  - [ ] Trailing realized-vol percentile regimes (deferred, §10)
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
