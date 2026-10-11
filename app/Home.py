"""Volcast app: S&P 500 volatility forecasts, tested out of sample.

Reads the files written by `python -m volcast.pipeline` (results/ by default, or the directory in
$VOLCAST_RESULTS_DIR). It never downloads data or fits models (DESIGN.md §8).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # light imports only: numpy/pandas, no network or model code

from volcast.evaluate import summarize  # noqa: E402
from volcast.targets import to_annual_vol  # noqa: E402

RESULTS_DIR = Path(os.environ.get("VOLCAST_RESULTS_DIR", ROOT / "results"))
BASELINE = "ewma"
HORIZON_LABELS = {1: "Next day", 22: "Next 22 days"}

# One fixed colour per entity, in the validated categorical order, so filtering never repaints a
# line. These steps pass the palette checks (CVD separation, lightness, >= 3:1 contrast) on both
# Streamlit backgrounds (#ffffff and #0e1117), because Streamlit applies its light/dark theme in the
# browser and the app can't reliably know which one is showing. Future models already have a slot.
SERIES_ORDER = ["ewma", "har", "garch", "kalman_sv", "gbm", "vix"]
SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
REALIZED_COLOR = "#7d7c76"  # neutral ink: 4.2:1 on white, 4.5:1 on the dark background
SPIKE_COLOR = "#898781"

st.set_page_config(page_title="Volcast", layout="wide")


# ----------------------------------------------------------------------------- data


def _stamp(directory: Path) -> float:
    files = list(directory.glob("*")) if directory.exists() else []
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner=False)
def load_results(directory: str, stamp: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    d = Path(directory)
    meta = json.loads((d / "meta.json").read_text())
    forecasts = pd.read_parquet(d / "forecasts.parquet")
    panel = pd.read_parquet(d / "panel.parquet").set_index("date")
    return meta, forecasts, panel


def fmt_vol(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.1f}%"


def series_color(name: str) -> str:
    return SERIES_COLORS[SERIES_ORDER.index(name)] if name in SERIES_ORDER else SERIES_COLORS[-1]


# ----------------------------------------------------------------------------- page

required = ["meta.json", "forecasts.parquet", "panel.parquet"]
if not all((RESULTS_DIR / f).exists() for f in required):
    st.title("Volcast")
    st.error(
        f"No results found in `{RESULTS_DIR}`. Run `python -m volcast.pipeline --source yahoo` "
        "(or `--source sample` offline) first."
    )
    st.stop()

meta, forecasts, panel = load_results(str(RESULTS_DIR), _stamp(RESULTS_DIR))
labels: dict[str, str] = meta["models"]
horizons: list[int] = meta["horizons"]
model_names = [m for m in SERIES_ORDER if m in labels] + [m for m in labels if m not in SERIES_ORDER]

st.title("Volcast")
st.markdown("S&P 500 volatility forecasts, tested out of sample.")
st.caption(
    f"Data through {meta['data_end']} · out-of-sample from {meta['oos_start']} · "
    f"parameters refit every {meta['refit_every']} trading days · generated {meta['generated_at'][:16].replace('T', ' ')} UTC"
)
if meta["source"] == "sample":
    st.info(
        "Showing the offline sample (S&P 500 index 1999–2018 bundled with the `arch` package), "
        "used for development. Live SPY results come from `python -m volcast.pipeline --source yahoo`."
    )

# ----------------------------------------------------------------------------- latest forecast

last_date = forecasts["date"].max()
latest = forecasts[forecasts["date"] == last_date]
st.subheader("Latest forecast")
st.caption(f"Made at the close on {last_date:%A %d %B %Y}, annualized.")

n_cols = len(model_names) + 2  # same grid on every row: models, realized, VIX
for h in horizons:
    row = latest[latest["horizon"] == h].set_index("model")["forecast"]
    st.markdown(f"**{HORIZON_LABELS.get(h, f'Next {h} days')}**")
    cols = st.columns(n_cols)
    for col, m in zip(cols, model_names):
        col.metric(labels[m], fmt_vol(to_annual_vol(row[m], h)) if m in row.index else "n/a", border=True)
    recent = panel["v"].iloc[-h:].sum()
    cols[len(model_names)].metric(
        "Realized, last day" if h == 1 else f"Realized, last {h} days", fmt_vol(to_annual_vol(recent, h)), border=True,
        help="What volatility has actually been over the most recent period, for context.",
    )
    if h == max(horizons):
        cols[-1].metric(
            "VIX", fmt_vol(panel["vix"].iloc[-1]), border=True,
            help="The market's own 30-day volatility forecast, implied by S&P 500 option prices.",
        )

st.divider()

# ----------------------------------------------------------------------------- model race

st.subheader("Forecast vs. realized")

oos_start, data_end = pd.Timestamp(meta["oos_start"]), pd.Timestamp(meta["data_end"])
windows = {
    name: (pd.Timestamp(a), pd.Timestamp(b))
    for name, (a, b) in meta.get("crisis_windows", {}).items()
    if pd.Timestamp(b) >= oos_start and pd.Timestamp(a) <= data_end
}
FULL, CUSTOM = "Full out-of-sample period", "Custom range"

f1, f2, f3 = st.columns([1.2, 2, 2])
with f1:
    horizon = st.segmented_control(
        "Horizon", horizons, default=horizons[-1], format_func=lambda h: HORIZON_LABELS.get(h, f"{h} days"),
        required=True, key="horizon",
    )
with f2:
    chosen = st.multiselect("Models", model_names, default=model_names, format_func=lambda m: labels[m], key="models")
with f3:
    period = st.selectbox("Period", [FULL, *windows, CUSTOM], key="period")

if period == FULL:
    start, end = oos_start, data_end
elif period == CUSTOM:
    picked = st.date_input(
        "Dates", value=(oos_start.date(), data_end.date()),
        min_value=oos_start.date(), max_value=data_end.date(), key="dates",
    )
    if not isinstance(picked, tuple) or len(picked) != 2:
        st.caption("Pick a start and an end date.")
        st.stop()
    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
else:
    start, end = windows[period]

if not chosen:
    st.caption("Select at least one model.")
    st.stop()

fc = forecasts[(forecasts["horizon"] == horizon) & forecasts["date"].between(start, end)]
wide = fc.pivot(index="date", columns="model", values="forecast")
realized = fc.groupby("date")["realized"].first()
vol = wide.apply(lambda s: to_annual_vol(s, horizon))
vol_realized = to_annual_vol(realized, horizon)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=vol_realized.index, y=vol_realized.to_numpy(), name="Realized", mode="lines",
        line={"color": REALIZED_COLOR, "width": 2}, hovertemplate="<b>%{y:.1f}%</b>",
    )
)
for m in chosen:
    if m in vol.columns:
        fig.add_trace(
            go.Scatter(
                x=vol.index, y=vol[m].to_numpy(), name=labels[m], mode="lines",
                line={"color": series_color(m), "width": 2}, hovertemplate="<b>%{y:.1f}%</b>",
            )
        )
fig.update_layout(
    height=440,
    margin={"l": 0, "r": 0, "t": 40, "b": 0},
    hovermode="x unified",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "title": None},
    xaxis={
        "automargin": True, "showgrid": False, "ticks": "outside", "hoverformat": "%a %d %b %Y",
        "showspikes": True, "spikemode": "across", "spikesnap": "cursor",
        "spikethickness": 1, "spikecolor": SPIKE_COLOR, "spikedash": "solid",
    },
    yaxis={
        "automargin": True, "title": {"text": "Annualized volatility"}, "ticksuffix": "%",
        "rangemode": "tozero", "zeroline": False,
    },
)
st.plotly_chart(fig, width="stretch", theme="streamlit", config={"displaylogo": False})
st.caption(
    "Each point is dated on the day the forecast was made. The realized line shows what actually "
    f"happened over the {'following day' if horizon == 1 else f'following {horizon} trading days'}."
)

# ----------------------------------------------------------------------------- accuracy

scored = summarize(forecasts[forecasts["model"].isin(chosen)], baseline=BASELINE, start=start, end=end)
scored = scored[scored["horizon"] == horizon].sort_values("qlike")
if scored.empty:
    st.caption("No scored forecasts in this window yet (realized volatility isn't known until the horizon has passed).")
else:
    table = pd.DataFrame(
        {
            "Model": scored["model"].map(labels),
            "Days scored": scored["n"],
            "QLIKE": scored["qlike"],
            "QLIKE vs EWMA": (scored["qlike_rel"] - 1) * 100,
            "MSE (%²)": scored["mse_pct2"],
            "MSE vs EWMA": (scored["mse_rel"] - 1) * 100,
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "QLIKE": st.column_config.NumberColumn(format="%.3f"),
            "QLIKE vs EWMA": st.column_config.NumberColumn(format="%+.1f%%"),
            "MSE (%²)": st.column_config.NumberColumn(format="%.2f"),
            "MSE vs EWMA": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    st.caption(
        "Lower is better. QLIKE is the headline metric: it stays reliable when realized volatility is itself "
        "estimated with noise, and it penalizes under-forecasting more than over-forecasting. "
        "All models are scored on the same days."
    )

with st.expander("Data behind the chart"):
    data = vol.rename(columns=labels).assign(Realized=vol_realized).round(2)
    st.dataframe(data, width="stretch")

# ----------------------------------------------------------------------------- method

with st.expander("How this works"):
    proxy = meta.get("proxy", {})
    st.markdown(
        f"""
**Target.** Daily variance is estimated from each day's open, high, low and close (Garman–Klass)
plus the squared overnight gap. It's several times less noisy than a squared daily return. Forecasts
cover the next day and the next 22 trading days (the window VIX covers).

**Walk-forward.** Every forecast is out of sample. Models are refit every {meta['refit_every']} trading
days on all data up to that date, then forecast daily with frozen parameters until the next refit. A
22-day target is only known 22 days later, so training stops 22 days before each refit date. A test
scrambles all data after a cutoff and checks that no earlier forecast changes.

**Models.** EWMA (λ = 0.94) is the naive baseline. HAR-RV regresses future variance on the last day,
week and month of variance. GARCH, a Kalman-filter stochastic volatility model and LightGBM come next.

**Proxy check.** Squared returns average {proxy.get('ret2_over_v', float('nan')):.2f}x the variance proxy
in this data{' (outside the expected 0.8–1.25 range: the index sample records stale opens, so absolute levels run low; model comparisons are still like for like)' if not proxy.get('ok', True) else ''}.
"""
    )
