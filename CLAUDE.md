# CLAUDE.md

Volcast forecasts S&P 500 (SPY) volatility with several models, evaluates them walk-forward, turns the forecasts into risk numbers, and serves everything in a Streamlit app.

**DESIGN.md is the spec and source of truth.** Read it before changing anything. When you finish a piece of work or change a decision, update its phase checklist (§11) and decisions log (§10).

## Commands

```bash
pip install -e ".[app,dev]"                      # install
python -m volcast.pipeline --source sample       # offline run on the bundled sample (dev/CI)
python -m volcast.pipeline --source yahoo        # live SPY + VIX data
pytest -q                                        # tests, including the look-ahead check
streamlit run app/Home.py                        # app
```

## Rules

- Units: decimal variance summed over the horizon. Annualize only for display (`volcast.targets.to_annual_vol`).
- Forecasts are dated at the origin (close of day t) and cover days t+1..t+h.
- New models subclass `VolModel` (`src/volcast/models/base.py`), must be causal, and are added to `REGISTRY` in `src/volcast/models/__init__.py`. `tests/test_lookahead.py` then checks them automatically. Never weaken, skip or special-case that test.
- Evaluate only through `volcast.backtest.walk_forward`. No random splits. Tune hyperparameters inside the training window only.
- The app reads `results/` only. It must never call Yahoo or fit models.
- Results from `--source sample` are for development. Don't quote them as findings.
- If a model beats HAR by a large margin, look for leakage before reporting it. DESIGN.md §10 records what was checked for LightGBM.
- Features must be known at the close of day t. VIX closes at 4:15 pm ET, after the 4:00 pm S&P close: use it lagged a day.

## Status

Phases 1 and 2 are done: EWMA, HAR-RV, GJR-GARCH-t and LightGBM, scored with Diebold–Mariano, the Model Confidence Set, Mincer–Zarnowitz and regime breakdowns (DESIGN.md §4, §6, §10). Next: phase 3 — Kalman-filter SV and the VIX benchmark (validate their colours against the four in use).
