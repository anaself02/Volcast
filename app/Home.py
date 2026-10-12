"""Volcast app: S&P 500 volatility forecasts, tested out of sample.

Reads the files written by `python -m volcast.pipeline` (results/ by default, or the directory in
$VOLCAST_RESULTS_DIR). It never downloads data or fits models (DESIGN.md §8).
"""

from __future__ import annotations

import html
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

from volcast.evaluate import common_sample, summarize  # noqa: E402
from volcast.targets import to_annual_vol  # noqa: E402

RESULTS_DIR = Path(os.environ.get("VOLCAST_RESULTS_DIR", ROOT / "results"))
REQUIRED = ["meta.json", "forecasts.parquet", "panel.parquet", "metrics.parquet"]
OPTIONAL = ["dm", "mcs", "mz", "regimes"]  # phase-2 tables; results written before them still load
BASELINE = "ewma"
ALPHA = 0.05  # significance level for the Diebold–Mariano verdicts
HORIZON_LABELS = {1: "Next day", 22: "Next 22 days"}
HORIZON_SHORT = {1: "1-day", 22: "22-day"}
RECENT, FULL, CUSTOM = "Last 2 years", "Full history", "Custom range"
RECENT_YEARS = 2

# One fixed colour per entity, so filtering never repaints a line. Models take the validated slots in
# order, except LightGBM: its plum replaces the documented fourth slot (amber), which can't be told
# apart from HAR-RV's orange where two lines touch. These four pass every palette check with all pairs
# compared (CVD and normal-vision separation, lightness, >= 3:1 contrast) on both Streamlit backgrounds
# (#ffffff and #0e1117), because Streamlit applies its theme in the browser. Kalman SV and VIX keep
# provisional colours, to be validated against these four when they arrive (phase 3).
SERIES_ORDER = ["ewma", "har", "garch", "gbm", "kalman_sv", "vix"]
SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#a063a5", "#c98500", "#008300"]
# Realized volatility is what every forecast is judged against, so it is drawn last (on top) in neutral
# ink: the palette's secondary ink, dark grey on the light theme (7.9:1 on #ffffff) and light grey on the
# dark one (10.8:1 on #0e1117). If the theme is unknown it falls back to a mid grey that clears 4:1 on both.
REALIZED_INK = {"light": "#52514e", "dark": "#c3c2b7"}
REALIZED_FALLBACK = "#7d7c76"
SPIKE_COLOR = "#898781"
# Neutral wash behind named stress periods and behind the best model's rows in the accuracy table:
# about 1.2:1 against either background, so the theme's ink stays above 10:1 on top of it.
NEUTRAL_WASH = "rgba(137, 135, 129, 0.16)"
# Short labels for the stress periods in config.CRISIS_WINDOWS, for views too narrow for the full names.
SHORT_NAMES = {
    "2008 crisis": "GFC",
    "2010 flash crash": "Flash crash",
    "2011 downgrade": "Downgrade",
    "2015 China devaluation": "China",
    "Feb 2018 vol spike": "Vol spike",
    "Q4 2018 selloff": "Q4 2018",
    "COVID 2020": "COVID",
    "2022 rate hikes": "Rate hikes",
    "April 2025 tariff shock": "Tariffs",
}

# Chart geometry. The figure has a fixed height, so vertical label spacing can be computed exactly;
# the width is only known in the browser, so horizontal spacing assumes a phone-sized plot.
CHART_HEIGHT = 420
PLOT_HEIGHT_PX = 340  # CHART_HEIGHT minus the legend band and the x-axis labels (measured: 355)
NARROW_PLOT_PX = 270  # plot width on a 375px phone (measured: 283 on a 390px one)
LABEL_ROW_PX, MAX_LABEL_ROWS, LABEL_GAP_PX = 13, 3, 6
LINE_LABEL_GAP_PX = 14
# Advance widths of the chart font (Source Sans) in thousandths of an em, measured in the browser, so
# label widths can be computed to within a pixel. Characters not listed count as wide.
GLYPH_EM = {
    **dict.fromkeys("0123456789+", 497), **dict.fromkeys(".,:;'", 249), " ": 200, "-": 311, "–": 480,
    "(": 303, ")": 303, "/": 350, "&": 609, "%": 824,
    "a": 504, "b": 553, "c": 456, "d": 555, "e": 496, "f": 292, "g": 504, "h": 544, "i": 247, "j": 247,
    "k": 495, "l": 255, "m": 829, "n": 547, "o": 542, "p": 555, "q": 555, "r": 347, "s": 419, "t": 338,
    "u": 544, "v": 467, "w": 719, "x": 446, "y": 467, "z": 425,
    "A": 543, "B": 588, "C": 571, "D": 615, "E": 527, "F": 494, "G": 617, "H": 652, "I": 263, "J": 479,
    "K": 579, "L": 486, "M": 727, "N": 647, "O": 664, "P": 566, "Q": 664, "R": 569, "S": 534, "T": 536,
    "U": 645, "V": 515, "W": 786, "X": 513, "Y": 476, "Z": 539,
}

# The app's only custom CSS. It hooks onto data-testid attributes and the app's own vc-* classes, not
# Streamlit's generated class names, so a Streamlit upgrade can at worst undo it, not break the page.
CSS = """
<style>
.stApp { font-variant-numeric: tabular-nums; }
[data-testid="stHeader"] { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 0.75rem; }
.vc-strip { display: flex; flex-wrap: wrap; gap: 0.1rem 1.5rem; padding: 0.15rem 0 0.4rem;
  border-bottom: 1px solid rgba(128, 128, 128, 0.3); font-size: 0.75rem; line-height: 1.5;
  font-family: "Source Code Pro", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.vc-strip span { white-space: nowrap; }
.vc-strip span.vc-wrap { white-space: normal; }
.vc-strip b { font-weight: 400; opacity: 0.6; letter-spacing: 0.04em; margin-right: 0.45em; text-transform: uppercase; }
</style>
"""

st.set_page_config(page_title="Volcast · S&P 500 volatility forecasts", page_icon=":material/insights:", layout="wide")


# ----------------------------------------------------------------------------- data


def _stamp(directory: Path) -> float:
    files = list(directory.glob("*")) if directory.exists() else []
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner="Loading results…")
def load_results(
    directory: str, stamp: float
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    d = Path(directory)
    meta = json.loads((d / "meta.json").read_text())
    forecasts = pd.read_parquet(d / "forecasts.parquet")
    panel = pd.read_parquet(d / "panel.parquet").set_index("date")
    metrics = pd.read_parquet(d / "metrics.parquet")
    extra = {name: pd.read_parquet(d / f"{name}.parquet") for name in OPTIONAL if (d / f"{name}.parquet").exists()}
    return meta, forecasts, panel, metrics, extra


def fmt_vol(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.1f}%"


def fmt_date(ts: pd.Timestamp, fmt: str = "%Y-%m-%d") -> str:
    """ISO date with non-breaking hyphens, so a date never splits across lines."""
    return f"{ts:{fmt}}".replace("-", "\u2011")


def fmt_p(p: float) -> str:
    """A p-value for a table cell."""
    return "—" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def p_text(p: float) -> str:
    """A p-value in a sentence."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def horizon_label(h: int) -> str:
    return HORIZON_LABELS.get(h, f"Next {h} days")


def series_color(name: str) -> str:
    return SERIES_COLORS[SERIES_ORDER.index(name)] if name in SERIES_ORDER else SERIES_COLORS[-1]


def vs_baseline(rel: float) -> str:
    """A loss ratio to the baseline in words: 0.819 -> '18.1% lower'."""
    if rel is None or not np.isfinite(rel):
        return "—"
    pct = (1.0 - rel) * 100
    return "same" if abs(pct) < 0.05 else f"{abs(pct):.1f}% {'lower' if pct > 0 else 'higher'}"


def best_per_horizon(scored: pd.DataFrame) -> dict[int, pd.Series]:
    """Lowest mean QLIKE per horizon."""
    return {int(h): g.loc[g["qlike"].idxmin()] for h, g in scored.groupby("horizon")}


def bias(forecasts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Mean forecast over mean realized variance per model and horizon, on summarize()'s common sample."""
    fc = forecasts[(forecasts["date"] >= start) & (forecasts["date"] <= end)]
    rows = []
    for h in sorted(fc["horizon"].unique()):
        wide = common_sample(fc, int(h))
        y = wide.pop("realized")
        rows += [{"model": m, "horizon": int(h), "bias": wide[m].mean() / y.mean()} for m in wide.columns]
    return pd.DataFrame(rows, columns=["model", "horizon", "bias"])


def text_px(text: str, size: float) -> float:
    """Rendered width of `text` in the chart font at `size` px."""
    return size * sum(GLYPH_EM.get(c, 830) for c in text) / 1000


def status_strip(items: list[tuple[str, str]], wrap: tuple[str, ...] = ()) -> str:
    """One terminal-style line of key-value pairs; only the items named in `wrap` may break across lines."""
    cells = []
    for k, v in items:
        cls = ' class="vc-wrap"' if k in wrap else ""
        cells.append(f"<span{cls}><b>{html.escape(k)}</b>{html.escape(v)}</span>")
    return f'<div class="vc-strip">{"".join(cells)}</div>'


# ----------------------------------------------------------------------------- chart


def _band_labels(
    bands: list[tuple[str, pd.Timestamp, pd.Timestamp]], start: pd.Timestamp, span: float
) -> dict[str, tuple[int, float, str]]:
    """Row, centre (0-1) and text for each stress-period label that fits on a phone-width plot.

    Full names when every one fits, short names otherwise. Labels go left to right into the first of up
    to three rows where they don't touch a neighbour; a label that fits in none is left to the tooltip.
    """

    def place(text_of) -> dict[str, tuple[int, float, str]]:
        ends, placed = [-1.0] * MAX_LABEL_ROWS, {}
        for name, a, b in sorted(bands, key=lambda w: w[1]):
            text = text_of(name)
            half = text_px(text, 11) / NARROW_PLOT_PX / 2
            mid = ((a - start).total_seconds() + (b - start).total_seconds()) / 2 / span
            mid = min(max(mid, half), 1 - half)  # keep the label inside the plot
            for row in range(MAX_LABEL_ROWS):
                if mid - half >= ends[row] + LABEL_GAP_PX / NARROW_PLOT_PX:
                    ends[row], placed[name] = mid + half, (row, mid, text)
                    break
        return placed

    full = place(lambda name: name)
    return full if len(full) == len(bands) else place(lambda name: SHORT_NAMES.get(name, name))


def race_figure(
    lines: list[tuple[str, pd.Series, str]],
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    label_order: list[str],
    styles: dict[str, dict],
    log: bool = False,
) -> go.Figure:
    """Forecast vs realized lines, shaded stress periods, and direct labels where they fit.

    `lines` are (name, annualized vol, colour) in drawing order, so the last one is on top;
    `label_order` ranks the names for the direct labels, so the most important line keeps its label
    when two would collide. `styles` sets each line's width in px and, optionally, its opacity.
    """
    values = pd.concat([s for _, s, _ in lines]).dropna()
    values = values[values > 0]
    peak, low = (float(values.max()), float(values.min())) if len(values) else (5.0, 1.0)
    span = (end - start).total_seconds()
    dates = pd.DatetimeIndex(sorted(set().union(*(s.index for _, s, _ in lines))))

    def frac(t: pd.Timestamp) -> float:  # position along the x-axis, 0 to 1
        return (t - start).total_seconds() / span

    # Stress periods in view. One that fills most of the view is the view itself, so it isn't shaded.
    bands = []
    for name, (a, b) in windows.items():
        a, b = max(a, start), min(b, end)
        if a < b and (b - a).total_seconds() < 0.8 * span:
            bands.append((name, a, b))
    labels = _band_labels(bands, start, span)

    # Leave room above the data for the label rows, so labels never sit on a line.
    rows = 1 + max((row for row, _, _ in labels.values()), default=-1)
    room = (rows * LABEL_ROW_PX + 8) / PLOT_HEIGHT_PX if rows else 0.05
    if log:
        y0, y_peak = np.log10(low * 0.9), np.log10(peak)
        y_range = [y0, y_peak + (y_peak - y0) * room / (1 - room)]
    else:
        y_range = [0.0, max(peak / (1 - room), 5.0)]

    def to_px(v: float) -> float:  # height above the x-axis, in px
        y = np.log10(v) if log else v
        return (y - y_range[0]) / (y_range[1] - y_range[0]) * PLOT_HEIGHT_PX

    fig = go.Figure()
    if bands:
        # An invisible trace that names the stress period in the tooltip, so every band can be
        # identified on hover, including any whose label didn't fit. It has a point on every date
        # (hover info off outside the bands): with gaps, the tooltip would borrow the nearest band's
        # name for dates just outside it.
        period = np.full(len(dates), "", dtype=object)
        for name, a, b in bands:
            period[(dates >= a) & (dates <= b)] = f"<i>{name}</i>"
        fig.add_trace(
            go.Scatter(
                x=dates.to_numpy(), y=np.full(len(dates), low), text=period, name="Stress period", mode="lines",
                line={"width": 0}, showlegend=False, hoverinfo=np.where(period == "", "none", "text"),
            )
        )
    for rank, (name, s, color) in enumerate(lines):
        fig.add_trace(
            go.Scatter(
                x=s.index.to_numpy(), y=s.to_numpy(), name=name, mode="lines",
                legendrank=1 if rank == len(lines) - 1 else 2 + rank,  # realized first, then the fixed model order
                line={"color": color, "width": styles[name]["width"]}, opacity=styles[name].get("opacity", 1.0),
                hovertemplate="<b>%{y:.1f}%</b>",
            )
        )
    # Shapes and annotations are collected and set once: adding them one by one copies the layout each time.
    shapes = [
        {"type": "rect", "xref": "x", "yref": "paper", "x0": a, "x1": b, "y0": 0, "y1": 1,
         "fillcolor": NEUTRAL_WASH, "line": {"width": 0}, "layer": "below"}
        for _, a, b in bands
    ]
    notes = [
        {"x": start + pd.Timedelta(seconds=mid * span), "y": 1, "xref": "x", "yref": "paper", "yanchor": "top",
         "yshift": -3 - row * LABEL_ROW_PX, "text": text, "showarrow": False, "font": {"size": 11}}
        for row, mid, text in labels.values()
    ]

    # Direct labels at the right edge, where the lines end. Labels that would collide aren't nudged
    # apart (that detaches them from their line); the legend and tooltip carry those instead.
    placed: dict[str, float] = {}
    by_name = {name: s for name, s, _ in lines}
    for name in label_order:
        s = by_name.get(name, pd.Series(dtype=float)).dropna()
        if s.empty or frac(s.index[-1]) < 0.96:  # realized stops h days early; skip if it ends far left
            continue
        y_px = to_px(s.iloc[-1])
        if all(abs(y_px - p) >= LINE_LABEL_GAP_PX for p in placed.values()):
            placed[name] = y_px
            notes.append(
                {"x": 1, "xref": "paper", "xanchor": "left", "xshift": 6, "y": np.log10(s.iloc[-1]) if log else s.iloc[-1],
                 "yref": "y", "text": name, "showarrow": False, "font": {"size": 12}}
            )
    right = 10 + max(text_px(name, 12) for name in placed) if placed else 8

    yaxis = {"type": "log" if log else "linear", "range": y_range, "fixedrange": True, "automargin": True,
             "ticksuffix": "%", "zeroline": False}
    if log:  # round ticks: 5%, 10%, 20%, 50%, ...
        top = 10 ** y_range[1]
        ticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500) if low * 0.9 <= v <= top]
        yaxis |= {"tickvals": ticks, "ticktext": [f"{v}%" for v in ticks]}
    fig.update_layout(
        shapes=shapes,
        annotations=notes,
        height=CHART_HEIGHT,
        margin={"l": 0, "r": right, "t": 36, "b": 0},
        hovermode="x unified",
        hoverlabel={"font_size": 13},
        dragmode=False,
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "title": None,
            "itemclick": False, "itemdoubleclick": False,
        },
        xaxis={
            "range": [start, end], "fixedrange": True, "automargin": True, "showgrid": False,
            "ticks": "outside", "hoverformat": "%a %Y-%m-%d",
            "showspikes": True, "spikemode": "across", "spikesnap": "cursor",
            "spikethickness": 1, "spikecolor": SPIKE_COLOR, "spikedash": "solid",
        },
        yaxis=yaxis,
    )
    return fig


# ----------------------------------------------------------------------------- page

st.html(CSS)

if not all((RESULTS_DIR / f).exists() for f in REQUIRED):
    st.title("Volcast")
    st.error(
        f"No results found in `{RESULTS_DIR}`. The app only reads pipeline output; it never downloads "
        "data or fits models. Run the pipeline first:",
    )
    st.code("python -m volcast.pipeline --source yahoo    # or --source sample, offline", language="bash")
    st.stop()

meta, forecasts, panel, metrics, extra = load_results(str(RESULTS_DIR), _stamp(RESULTS_DIR))
labels: dict[str, str] = meta["models"]
horizons: list[int] = sorted(meta["horizons"])
by_h = sorted(horizons, reverse=True)  # the longer horizon leads: smoother, and what VIX covers
model_names = [m for m in SERIES_ORDER if m in labels] + [m for m in labels if m not in SERIES_ORDER]
base_label = labels.get(BASELINE, BASELINE.upper())
oos_start, data_end = pd.Timestamp(meta["oos_start"]), pd.Timestamp(meta["data_end"])
origins = np.sort(forecasts["date"].unique())
last_date = pd.Timestamp(origins[-1])
prev_date = pd.Timestamp(origins[-2]) if len(origins) > 1 else None
best = best_per_horizon(metrics)
n_scored = metrics.groupby("horizon")["n"].max()  # every model is scored on the same days

# ----------------------------------------------------------------------------- header

st.html(
    status_strip(
        [
            ("Source", meta.get("source_label", meta["source"])),
            ("Data through", f"{data_end:%Y-%m-%d}"),
            ("Last refit", f"{forecasts['refit_date'].max():%Y-%m-%d}"),
            ("Out of sample", f"{oos_start:%Y-%m-%d} → {data_end:%Y-%m-%d}"),
            ("Scored", " · ".join(f"{HORIZON_SHORT.get(h, f'{h}-day')} {int(n):,} days" for h, n in n_scored.items())),
        ],
        wrap=("Source",),
    )
)
st.title("Volcast")
st.markdown(
    "Volatility forecasts for the S&P 500, scored out of sample. Each model forecasts annualized volatility "
    "for the next trading day and the next 22, using only data available at each close, and is compared "
    "with the volatility that followed."
)
if meta["source"] == "sample":
    st.info(
        "Showing the offline sample (S&P 500 index 1999–2018 bundled with the `arch` package), "
        "used for development. Live SPY results come from `python -m volcast.pipeline --source yahoo`.",
    )
proxy = meta.get("proxy", {})
if not proxy.get("ok", True):
    st.warning(
        f"Variance proxy check: squared returns average {proxy.get('ret2_over_v', float('nan')):.2f}x the "
        "variance proxy in this data, outside the expected 0.8–1.25 range, so absolute volatility levels "
        "run low. Model comparisons are still like for like. Details under *How this works*.",
    )

# ----------------------------------------------------------------------------- headline + latest

dm_table, mcs_table = extra.get("dm"), extra.get("mcs")


def dm_p(model: str, bench: str, h: int) -> float:
    """Two-sided Diebold–Mariano p-value of `model` against `bench` over the full sample (NaN if absent)."""
    if dm_table is None:
        return float("nan")
    row = dm_table[(dm_table["model"] == model) & (dm_table["benchmark"] == bench) & (dm_table["horizon"] == h)]
    return float(row["p_value"].iloc[0]) if len(row) else float("nan")


def verdict(model: str, h: int) -> str:
    """Whether the best model's edge over EWMA, and over HAR-RV, is statistically significant."""
    if model == BASELINE:
        return f"No model has a lower QLIKE than {base_label}."
    parts = []
    for bench in (BASELINE, "har"):
        p = dm_p(model, bench, h)
        if bench != model and bench in labels and np.isfinite(p):
            word = f"significant at {ALPHA:.0%}" if p < ALPHA else f"not significant at {ALPHA:.0%}"
            parts.append(f"over {labels[bench]}: {word} ({p_text(p)})")
    return f"Edge {'; '.join(parts)}." if parts else ""


headline, latest_card = st.columns([5, 7], gap="medium")

with headline.container(border=True, height="stretch"):
    first, last = metrics["start"].min(), metrics["end"].max()
    kind = "Development sample, not a finding" if meta["source"] == "sample" else "Out-of-sample accuracy"
    st.caption(f"{kind} · QLIKE loss vs. the {base_label} baseline · {fmt_date(first, '%Y-%m')} to {fmt_date(last, '%Y-%m')}")
    for tile, h in zip(st.columns(len(horizons), gap="small"), by_h):
        win = best.get(h)
        if win is not None and win["model"] != BASELINE and np.isfinite(win["qlike_rel"]):
            label, value = f"{horizon_label(h)} · {labels[win['model']]}", f"{(1 - win['qlike_rel']) * 100:.1f}%"
            delta = f"lower QLIKE than {base_label}"
        else:
            label, value, delta = horizon_label(h), "—", f"no model beats {base_label}"
        tile.metric(
            label, value, delta, delta_color="off", delta_arrow="off",
            help=f"How much lower the most accurate model's mean QLIKE loss is than the {base_label} "
            f"baseline's, over {int(n_scored.get(h, 0)):,} out-of-sample trading days.",
        )
        if win is not None:
            tile.caption(verdict(win["model"], h))
    if mcs_table is None:
        st.caption(":material/warning: These results predate the significance tests: rerun the pipeline.")
    else:
        sets = "; ".join(
            f"{HORIZON_SHORT.get(h, f'{h}-day')} "
            + ", ".join(labels[m] for m in mcs_table.query("horizon == @h and in_mcs").sort_values("qlike")["model"])
            for h in by_h
        )
        st.caption(f"90% Model Confidence Set, the models the data can't separate from the best: {sets}.")
    st.caption(
        "QLIKE is the standard loss for variance forecasts: 0 is perfect and lower is better. Every model "
        f"is scored on the same days. Diebold–Mariano tests: two-sided at {ALPHA:.0%}, on QLIKE."
    )

def forecast_vol(model: str, h: int, date: pd.Timestamp | None) -> float:
    f = forecasts[(forecasts["date"] == date) & (forecasts["horizon"] == h) & (forecasts["model"] == model)]
    return to_annual_vol(f["forecast"].iloc[0], h) if len(f) else float("nan")


def gap_to_realized(now: float, recent: float, h: int) -> str:
    period = "on the last day" if h == 1 else f"over the last {h} days"
    if not (np.isfinite(now) and np.isfinite(recent)):
        return ""
    if abs(now - recent) < 0.05:
        return f"In line with realized vol {period} ({recent:.1f}%)"
    return f"{abs(now - recent):.1f} pts {'above' if now > recent else 'below'} realized vol {period} ({recent:.1f}%)"


with latest_card.container(border=True, height="stretch"):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(
            f"Latest forecast · made at the close of {fmt_date(last_date)} · annualized volatility", width="stretch"
        )
        with st.popover("All models", width="content"):
            st.table(
                pd.DataFrame(
                    {f"{horizon_label(h)} (%)": [forecast_vol(m, h, last_date) for m in model_names] for h in by_h},
                    index=pd.Index([labels[m] for m in model_names], name="Model"),
                ).style.format("{:.1f}", na_rep="—")
            )
    tiles = st.columns(len(horizons) + 1, gap="small")
    for tile, h in zip(tiles, by_h):
        m = best[h]["model"] if h in best else model_names[0]
        now, before = forecast_vol(m, h, last_date), forecast_vol(m, h, prev_date)
        with tile:
            st.metric(
                f"{horizon_label(h)} · {labels[m]}", fmt_vol(now),
                f"{now - before:+.1f} pts" if np.isfinite(now - before) else None,
                delta_color="off", delta_description="vs. prior day",
                help=f"Annualized volatility forecast for the {horizon_label(h).lower()} from {labels[m]}, the "
                "most accurate model at this horizon out of sample. Annualized volatility is the standard "
                "deviation of returns scaled to one year, in percent.",
            )
            st.caption(gap_to_realized(now, to_annual_vol(panel["v"].iloc[-h:].sum(), h), h))
    vix = panel["vix"].dropna()
    with tiles[-1]:
        st.metric(
            "VIX", fmt_vol(vix.iloc[-1]) if len(vix) else "n/a",
            f"{vix.iloc[-1] - vix.iloc[-2]:+.1f} pts" if len(vix) > 1 else None,
            delta_color="off", delta_description="vs. prior day",
            help="Cboe's VIX index: the market's 30-day volatility forecast, implied by S&P 500 option prices.",
        )
        st.caption("Market-implied, next 30 calendar days")

# ----------------------------------------------------------------------------- model race

st.subheader("Forecast vs. realized")
st.caption(
    "Forecast: a model's annualized volatility for the period after each close, dated at that close. "
    "Realized: the volatility that followed (grey, drawn on top). Shaded: named stress periods."
)

windows = {
    name: (pd.Timestamp(a), pd.Timestamp(b))
    for name, (a, b) in meta.get("crisis_windows", {}).items()
    if pd.Timestamp(b) >= oos_start and pd.Timestamp(a) <= data_end
}
recent_start = max(oos_start, data_end - pd.DateOffset(years=RECENT_YEARS))
st.session_state.setdefault("scale", "Linear")


def default_scale() -> None:
    """Log on the full history, so calm years aren't squashed by 2008 and 2020; linear elsewhere."""
    st.session_state["scale"] = "Log" if st.session_state["period"] == FULL else "Linear"


f1, f2, f3, f4 = st.columns([1.1, 1.4, 1.7, 0.9], gap="small")
with f1:
    horizon = st.segmented_control(
        "Horizon", horizons, default=by_h[0], format_func=horizon_label, required=True, key="horizon",
        help="Total volatility over the next trading day, or over the next 22 trading days (about a month).",
    )
with f2:
    picked_models = st.pills(
        "Models", model_names, selection_mode="multi", default=model_names, format_func=lambda m: labels[m],
        key="models",
        help="EWMA: exponentially weighted average of past daily variance (RiskMetrics, λ = 0.94), the "
        "baseline. HAR-RV: regression of future variance on the last day, week and month of variance.",
    )
    chosen = [m for m in model_names if m in picked_models]  # fixed order: colours follow the model
with f3:
    period = st.selectbox("Period", [RECENT, FULL, *windows, CUSTOM], key="period", on_change=default_scale)
    problem = None
    if period == RECENT:
        start, end = recent_start, data_end
    elif period == FULL:
        start, end = oos_start, data_end
    elif period == CUSTOM:
        picked = st.date_input(
            "Dates", value=(recent_start.date(), data_end.date()),
            min_value=oos_start.date(), max_value=data_end.date(), key="dates",
        )
        if isinstance(picked, tuple) and len(picked) == 2 and picked[1] > picked[0]:
            start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
        else:
            start = end = None
            problem = "Pick an end date after the start date to finish the range."
    else:
        start, end = windows[period]
with f4:
    scale = st.segmented_control(
        "Scale", ["Linear", "Log"], required=True, key="scale",
        help="A log scale keeps calm periods readable next to the 2008 and 2020 spikes. It defaults to log "
        "on the full history and linear elsewhere.",
    )

if problem is None and not chosen:
    problem = "Select at least one model to draw the chart and score it."
fc = pd.DataFrame()
if problem is None:
    fc = forecasts[(forecasts["horizon"] == horizon) & forecasts["date"].between(start, end)]
    if fc.empty:
        problem = "There are no forecasts in this range. Try a longer one."

if problem is not None:
    st.info(problem)
else:
    wide = fc.pivot(index="date", columns="model", values="forecast")
    realized = fc.groupby("date")["realized"].first()
    vol = wide.apply(lambda s: to_annual_vol(s, horizon))
    vol_realized = to_annual_vol(realized, horizon)
    in_window = forecasts[forecasts["model"].isin(chosen)]
    scored = summarize(in_window, baseline=BASELINE, start=start, end=end)

    # Models in the fixed order, realized last so it's drawn on top; label the most accurate model first.
    lines = [(labels[m], vol[m], series_color(m)) for m in chosen if m in vol.columns]
    lines += [("Realized", vol_realized, REALIZED_INK.get(st.context.theme.type, REALIZED_FALLBACK))]
    rank = scored[scored["horizon"] == horizon].sort_values("qlike")["model"].map(labels).tolist()
    label_order = [*rank, *(n for n, _, _ in lines[:-1] if n not in rank), "Realized"]
    # Realized is drawn last, so on top, and a little thicker than the forecasts. Daily realized volatility
    # is very spiky: lines are thinner on the 1-day horizon, and over more than about four years of it a
    # translucent realized line keeps the forecasts underneath visible.
    styles = {name: {"width": 1.25 if horizon == 1 else 1.75} for name, _, _ in lines}
    styles["Realized"] = {"width": 1.5 if horizon == 1 else 2.25, "opacity": 0.45 if horizon == 1 and len(vol) > 1000 else 1.0}
    st.plotly_chart(
        race_figure(lines, windows, start, end, label_order, styles, log=scale == "Log"),
        width="stretch", theme="streamlit", config={"displayModeBar": False, "responsive": True},
    )
    st.caption(
        f"Annualized volatility, {'log' if scale == 'Log' else 'linear'} scale. Realized covers the "
        f"{'following day' if horizon == 1 else f'following {horizon} trading days'}, so it ends "
        f"{'a day' if horizon == 1 else f'{horizon} trading days'} before the forecasts do."
    )

    # ------------------------------------------------------------------------- accuracy

    st.subheader("Accuracy")
    if scored.empty:
        st.info("No scored forecasts in this window yet: realized volatility isn't known until the horizon has passed.")
    else:
        scored = scored.merge(bias(in_window, start, end), on=["model", "horizon"], how="left")
        scored = scored.sort_values(["horizon", "qlike"]).reset_index(drop=True)
        n_models = scored.groupby("horizon")["model"].transform("size")
        is_best = (scored["qlike"] == scored.groupby("horizon")["qlike"].transform("min")) & (n_models > 1)
        is_base = scored["model"] == BASELINE
        table = pd.DataFrame(
            {
                "Horizon": scored["horizon"].map(lambda h: HORIZON_SHORT.get(h, f"{h}-day")),
                "Model": scored["model"].map(labels),
                "QLIKE": scored["qlike"],
                f"QLIKE vs. {base_label}": np.where(is_base, "baseline", scored["qlike_rel"].map(vs_baseline)),
                "MSE (%²)": scored["mse_pct2"],
                f"MSE vs. {base_label}": np.where(is_base, "baseline", scored["mse_rel"].map(vs_baseline)),
                "Bias (×)": scored["bias"],
                "Days": scored["n"],
            }
        )
        best_rows = set(np.flatnonzero(is_best))
        styled = table.style.apply(
            lambda row: [f"background-color: {NEUTRAL_WASH}" if row.name in best_rows else ""] * len(row), axis=1
        )
        st.dataframe(
            styled,
            hide_index=True,
            width="stretch",
            column_config={
                "QLIKE": st.column_config.NumberColumn(format="%.3f", help="Mean QLIKE loss. 0 is perfect; lower is better."),
                f"QLIKE vs. {base_label}": st.column_config.TextColumn(
                    help=f"Mean QLIKE loss relative to the {base_label} baseline. Lower is better."
                ),
                "MSE (%²)": st.column_config.NumberColumn(
                    format="%.2f", help="Mean squared error on variance, in percent squared. Lower is better."
                ),
                f"MSE vs. {base_label}": st.column_config.TextColumn(
                    help=f"Mean squared error relative to the {base_label} baseline. Lower is better."
                ),
                "Bias (×)": st.column_config.NumberColumn(
                    format="%.2f",
                    help="Mean forecast variance divided by mean realized variance on the scored days. 1.00 is "
                    "unbiased; above 1 means the forecasts ran high on average, below 1 that they ran low.",
                ),
                "Days": st.column_config.NumberColumn(format="localized", help="Trading days scored."),
            },
        )
        missing = "" if BASELINE in chosen else f" Add {base_label} to compare each model with the baseline."
        st.caption(
            f"Window {fmt_date(scored['start'].min())} to {fmt_date(scored['end'].max())}, scoring only days whose "
            "outcome is known, the same days for every model. Shaded row: lowest QLIKE at that horizon." + missing
        )
        st.caption(
            "QLIKE penalizes under-forecasting more than over-forecasting, so read accuracy and bias together. "
            "Bias describes the forecasts; it doesn't explain the accuracy gap."
        )

    with st.expander("Data behind the chart"):
        data = vol[[m for m in model_names if m in vol.columns]].rename(columns=labels)
        data = data.assign(Realized=vol_realized).round(2)
        data = data.rename(columns=lambda c: f"{c} (%)")
        st.dataframe(
            data,
            width="stretch",
            placeholder="—",
            column_config={
                "_index": st.column_config.DateColumn("Forecast date", format="YYYY-MM-DD"),
                **{c: st.column_config.NumberColumn(format="%.2f") for c in data.columns},
            },
        )
        st.caption("Annualized volatility in percent. Realized stays empty until the horizon has passed.")

# ----------------------------------------------------------------------------- model comparison


def tint_rows(rows: set[int]):
    return lambda row: [f"background-color: {NEUTRAL_WASH}" if row.name in rows else ""] * len(row)


if dm_table is not None and mcs_table is not None and "mz" in extra and "regimes" in extra:
    st.subheader("Model comparison")
    first, last = metrics["start"].min(), metrics["end"].max()
    st.caption(
        f"Full out-of-sample period, {fmt_date(first)} to {fmt_date(last)}, every model scored on the same "
        "days. These tables don't follow the controls above."
    )
    order = {m: i for i, m in enumerate(model_names)}
    sig_tab, mz_tab, regime_tab = st.tabs(["Significance", "Calibration", "By regime"])

    with sig_tab:
        benches = [b for b in (BASELINE, "har") if b in labels]
        rows = []
        for _, r in mcs_table.sort_values(["horizon", "qlike"]).iterrows():
            h, m = int(r["horizon"]), r["model"]
            row = {"Horizon": HORIZON_SHORT.get(h, f"{h}-day"), "Model": labels[m], "QLIKE": r["qlike"]}
            for b in benches:
                d = dm_table[(dm_table["horizon"] == h) & (dm_table["model"] == m) & (dm_table["benchmark"] == b)]
                row[f"DM vs. {labels[b]}"] = float(d["stat"].iloc[0]) if len(d) else np.nan
                row[f"p vs. {labels[b]}"] = fmt_p(float(d["p_value"].iloc[0])) if len(d) else "—"
            row["MCS p"] = r["mcs_pvalue"]
            row["In 90% MCS"] = "yes" if r["in_mcs"] else "no"
            rows.append(row)
        sig = pd.DataFrame(rows)
        st.dataframe(
            sig.style.apply(tint_rows(set(np.flatnonzero(sig["In 90% MCS"] == "yes"))), axis=1),
            hide_index=True,
            width="stretch",
            placeholder="—",
            column_config={
                "QLIKE": st.column_config.NumberColumn(format="%.3f", help="Mean QLIKE loss. Lower is better."),
                **{
                    f"DM vs. {labels[b]}": st.column_config.NumberColumn(
                        format="%.2f",
                        help=f"Diebold–Mariano statistic for the QLIKE gap to {labels[b]}. Negative: lower loss than "
                        f"{labels[b]}.",
                    )
                    for b in benches
                },
                **{
                    f"p vs. {labels[b]}": st.column_config.TextColumn(
                        alignment="right", help=f"Two-sided p-value: below {ALPHA} is significant at {ALPHA:.0%}."
                    )
                    for b in benches
                },
                "MCS p": st.column_config.NumberColumn(
                    format="%.3f", help="Model Confidence Set p-value: above 0.10 keeps the model in the 90% set."
                ),
                "In 90% MCS": st.column_config.TextColumn(alignment="center"),
            },
        )
        tests = meta.get("tests", {})
        mcs_cfg = tests.get("mcs", {})
        st.caption(
            "Diebold–Mariano: is the mean QLIKE gap between two models zero? Newey–West variance with h − 1 "
            "lags (22-day forecasts overlap) and the Harvey–Leybourne–Newbold small-sample correction. The "
            "p-values aren't adjusted for testing several pairs; the Model Confidence Set is. Shaded rows: "
            f"the 90% Model Confidence Set (Hansen, Lunde and Nason), {mcs_cfg.get('reps', 1000):,} stationary-"
            f"bootstrap draws with {mcs_cfg.get('block', 22)}-day blocks, seed {mcs_cfg.get('seed', '—')}."
        )

    with mz_tab:
        mz = extra["mz"].assign(_o=lambda x: x["model"].map(order)).sort_values(["horizon", "_o"])
        st.dataframe(
            pd.DataFrame(
                {
                    "Horizon": mz["horizon"].map(lambda h: HORIZON_SHORT.get(h, f"{h}-day")),
                    "Model": mz["model"].map(labels),
                    "α (%²)": mz["alpha"],
                    "β": mz["beta"],
                    "R²": mz["r2"],
                    "p (α = 0, β = 1)": mz["p_value"].map(fmt_p),
                    "Days": mz["n"],
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "α (%²)": st.column_config.NumberColumn(format="%.3f", help="Intercept, in variance units (%²)."),
                "β": st.column_config.NumberColumn(format="%.3f", help="Slope on the forecast. 1 for a well-scaled forecast."),
                "R²": st.column_config.NumberColumn(format="%.3f"),
                "p (α = 0, β = 1)": st.column_config.TextColumn(
                    alignment="right", help="Wald test of α = 0 and β = 1 together, with HAC standard errors."
                ),
                "Days": st.column_config.NumberColumn(format="localized"),
            },
        )
        st.caption(
            "Mincer–Zarnowitz regression of realized on forecast variance: realized = α + β × forecast. A forecast "
            "that is right on average at every level has α = 0 and β = 1; the p-value tests both at once with "
            "Newey–West standard errors. β below 1 means the forecasts swing more than the outcomes. In levels "
            "the fit is dominated by crisis days, so read it with the Bias column and QLIKE."
        )

    with regime_tab:
        reg = extra["regimes"]

        def regime_table(kinds: list[str]) -> None:
            part = reg[reg["kind"].isin(kinds)]
            keys = part[["horizon", "regime"]].drop_duplicates()
            rows = []
            for _, k in keys.iterrows():
                cell = part[(part["horizon"] == k["horizon"]) & (part["regime"] == k["regime"])].set_index("model")
                row = {"Horizon": HORIZON_SHORT.get(int(k["horizon"]), f"{k['horizon']}-day"), "Regime": k["regime"],
                       "Days": int(cell["n"].iloc[0])}
                row |= {labels[m]: cell.loc[m, "qlike"] for m in model_names if m in cell.index}
                rows.append(row)
            table = pd.DataFrame(rows)
            cols = [labels[m] for m in model_names if labels[m] in table.columns]

            def shade_best(row):
                best_col = row[cols].astype(float).idxmin()
                return [f"background-color: {NEUTRAL_WASH}" if c == best_col else "" for c in row.index]

            st.dataframe(
                table.style.apply(shade_best, axis=1),
                hide_index=True,
                width="stretch",
                placeholder="—",
                column_config={"Days": st.column_config.NumberColumn(format="localized"),
                               **{c: st.column_config.NumberColumn(format="%.3f") for c in cols}},
            )

        regime_table(["stress", "vix"])
        st.caption(
            "Mean QLIKE in each regime, lowest shaded. Stress windows are the named periods in the chart. VIX "
            "terciles split the scored days into thirds by VIX at the forecast date; the cut-offs come from these "
            "days, and they only group days: no forecast uses them."
        )
        with st.expander("Each stress window"):
            regime_table(["window"])

# ----------------------------------------------------------------------------- method

issues = meta.get("refit_issues", {})
garch_fallbacks = sum(n for key, n in issues.items() if key.startswith("garch/"))
garch_note = (
    f" (it happened {garch_fallbacks} times in this run; see refits.parquet)" if garch_fallbacks
    else " (it didn't happen in this run)" if "refit_issues" in meta else ""
)

with st.expander("How this works"):
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
week and month of variance. GJR-GARCH(1,1,1) with Student-t errors is fit on close-to-close returns and
forecasts the sum of the next h daily variances analytically; a fit that fails or can't forecast falls
back to the previous refit's parameters{garch_note}. LightGBM learns from features known at each close
(variance averages over 1, 5, 22 and 66 days, recent negative returns, VIX, its 5-day change, implied
over realized variance) with a loss that is QLIKE itself, tuned only inside each training window.

**Scores.** QLIKE and MSE are the two losses that still rank forecasts correctly when realized variance
is itself estimated with noise (Patton 2011). Bias is mean forecast over mean realized variance on the
same days.

**Tests.** Diebold–Mariano compares two models' QLIKE (Newey–West variance, Harvey–Leybourne–Newbold
correction). The Model Confidence Set keeps every model the data can't separate from the best.
Mincer–Zarnowitz regressions check whether forecasts are right on average at every level.

**Proxy check.** Squared returns average {proxy.get('ret2_over_v', float('nan')):.2f}x the variance proxy
in this data{' (outside the expected 0.8–1.25 range: the index sample records stale opens, so absolute levels run low; model comparisons are still like for like)' if not proxy.get('ok', True) else ', within the expected 0.8–1.25 range'}.

**Not yet.** The Kalman-filter stochastic volatility model and the VIX benchmark (phase 3), and the
risk layer (phase 4). The window-by-window accuracy table above isn't significance-tested: only the
full-period tests in *Model comparison* are.
"""
    )
