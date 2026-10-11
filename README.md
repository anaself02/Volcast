# Volcast

**Forecasting S&P 500 volatility, tested out of sample.**

Volcast forecasts how volatile the S&P 500 (SPY) will be over the next day and the next month with a ladder of models, from a one-line baseline up to a Kalman-filter stochastic volatility model and gradient boosting. It then checks which ones actually work on data they never saw, and turns the forecasts into risk limits (Value-at-Risk and Expected Shortfall) backtested through real crises.

> **Status: phase 1 of 5.** Data pipeline, EWMA and HAR-RV models, walk-forward engine, look-ahead test and a first app page are done. Findings will go here once all models are in. The full design and roadmap are in [DESIGN.md](DESIGN.md).

## Why the results can be trusted

- **Strictly out of sample.** Every forecast comes from a model refit every 22 trading days on past data only, then run forward with frozen parameters.
- **Structural embargo.** A 22-day target isn't known until 22 days later. Models are fit on data truncated at the refit date, so the last 22 targets simply don't exist yet. Nothing to remember, nothing to forget.
- **A test that proves there's no look-ahead.** It scrambles every day after a cutoff and requires every earlier forecast to stay identical. It runs on every model automatically, and two deliberately leaky models prove it catches leaks.
- **Metrics that survive noisy data.** QLIKE and MSE on variance (Patton 2011) rank forecasts correctly even though "true" volatility is itself an estimate. Every model is compared against the same EWMA baseline, on the same days.
- **Data checks that fail loudly.** Bad bars, gaps and duplicates stop the pipeline. The variance proxy's level is checked against squared returns on every run.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"

python -m volcast.pipeline --source yahoo     # live SPY and VIX (or --source sample, offline)
streamlit run app/Home.py                     # the app
pytest -q                                     # tests
```

## How it's built

```
src/volcast/
  data.py       Yahoo / CBOE / offline sample loaders, validation, cache
  targets.py    daily variance proxy (Garman–Klass + overnight gap), forward targets
  models/       EWMA, HAR-RV (GARCH, Kalman SV and LightGBM next)
  backtest.py   walk-forward engine
  evaluate.py   QLIKE, MSE
  checks.py     look-ahead check
  pipeline.py   CLI that writes results/ for the app
app/Home.py     Streamlit app: reads results/ only
tests/          unit tests, look-ahead test, end-to-end pipeline and app test
```

## Roadmap

1. ~~Walking skeleton~~ ✅
2. GJR-GARCH, LightGBM, Diebold–Mariano tests, Model Confidence Set
3. Range-based stochastic volatility with a Kalman filter, VIX comparison
4. Risk layer: VaR and Expected Shortfall with Kupiec, Christoffersen and Basel traffic-light backtests
5. Daily refresh, full app, findings

Data from Yahoo Finance is for research and education.
