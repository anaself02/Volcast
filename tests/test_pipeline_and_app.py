"""End to end on the offline sample: pipeline -> results/ -> app."""

import json
from datetime import date

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from volcast.config import ROOT
from volcast.pipeline import main, run

APP = str(ROOT / "app" / "Home.py")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    run("sample", out_dir=out, raw_dir=tmp_path_factory.mktemp("raw"))
    return out


def test_pipeline_writes_the_results_contract(results):
    fc = pd.read_parquet(results / "forecasts.parquet")
    assert list(fc.columns) == ["date", "model", "horizon", "forecast", "realized", "refit_date"]
    metrics = pd.read_parquet(results / "metrics.parquet")
    assert len(metrics) == 4 and {"qlike", "mse_pct2", "qlike_rel", "mse_rel"} <= set(metrics.columns)
    panel = pd.read_parquet(results / "panel.parquet")
    assert list(panel.columns) == ["date", "close", "ret", "v", "vix"]
    meta = json.loads((results / "meta.json").read_text())
    assert meta["source"] == "sample" and meta["oos_start"] == "2005-01-03"
    assert meta["models"] == {"ewma": "EWMA", "har": "HAR-RV"} and meta["horizons"] == [1, 22]
    # The index sample's stale opens make the proxy run low (DESIGN.md §2): flagged, not hidden.
    assert meta["proxy"]["ok"] is False and meta["proxy"]["ret2_over_v"] > 1.25


def test_cli(tmp_path, capsys):
    assert main(["--source", "sample", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "meta.json").exists()
    assert "qlike" in capsys.readouterr().out


def _app(monkeypatch, results_dir):
    monkeypatch.setenv("VOLCAST_RESULTS_DIR", str(results_dir))
    return AppTest.from_file(APP, default_timeout=60).run()


def _accuracy(at):
    return next(d for d in at.dataframe if "Days" in d.value.columns)


def _yaxis(at):
    return json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]["yaxis"]


def _hand_bias(results, model, horizon, start, end):
    """Mean forecast over mean realized variance, on the days where both models and the target are known."""
    fc = pd.read_parquet(results / "forecasts.parquet")
    fc = fc[(fc["horizon"] == horizon) & (fc["date"] >= start) & (fc["date"] <= end)]
    days = fc.pivot(index="date", columns="model", values="forecast")
    days["realized"] = fc.groupby("date")["realized"].first()
    days = days.dropna()
    return days[model].mean() / days["realized"].mean()


def test_app_renders_and_every_control_works(monkeypatch, results):
    at = _app(monkeypatch, results)
    assert not at.exception
    meta = json.loads((results / "meta.json").read_text())

    # Header numbers come straight from results/: the status strip from meta.json, the headline from
    # metrics.parquet, the latest forecast from forecasts.parquet.
    strip = next(e.proto.body for e in at.get("html") if 'class="vc-strip"' in e.proto.body)
    assert f"{meta['data_end']}" in strip and f"{meta['oos_start']}" in strip and "Last refit" in strip
    metrics = pd.read_parquet(results / "metrics.parquet").set_index(["horizon", "model"])
    head_22, head_1, latest_22, latest_1, vix = at.metric[:5]
    for tile, h in [(head_22, 22), (head_1, 1)]:
        assert tile.value == f"{(1 - metrics.loc[(h, 'har'), 'qlike_rel']) * 100:.1f}%"
        assert tile.label.endswith("HAR-RV") and tile.delta == "lower QLIKE than EWMA"
    assert latest_22.label == "Next 22 days · HAR-RV" and latest_1.label == "Next day · HAR-RV"
    assert latest_22.delta.endswith(" pts") and vix.label == "VIX"

    # Caveats stay visible: significance, the sample, the proxy check, the method notes.
    assert any(c.value.endswith("Not yet tested for statistical significance (Diebold–Mariano and Model "
                                "Confidence Set arrive with the next models).") for c in at.caption)
    assert any("offline sample" in i.value for i in at.info)
    assert any("proxy" in w.value for w in at.warning)
    assert any(c.value.startswith("Development sample, not a finding") for c in at.caption)
    assert "How this works" in [e.label for e in at.expander]

    # Default window: the last two years, on a linear scale.
    end = pd.Timestamp(meta["data_end"])
    start = end - pd.DateOffset(years=2)
    assert at.selectbox(key="period").value == "Last 2 years"
    assert at.segmented_control(key="scale").value == "Linear" and _yaxis(at)["type"] == "linear"

    # Accuracy table: both horizons, best model first and tinted, no star column, raw losses kept.
    table = _accuracy(at)
    acc = table.value
    assert list(acc.columns) == [
        "Horizon", "Model", "QLIKE", "QLIKE vs. EWMA", "MSE (%²)", "MSE vs. EWMA", "Bias (×)", "Days"
    ]
    assert acc["Horizon"].tolist() == ["1-day", "1-day", "22-day", "22-day"]
    assert acc["Model"].tolist() == ["HAR-RV", "EWMA"] * 2
    assert acc["QLIKE vs. EWMA"].tolist()[1::2] == ["baseline"] * 2 and acc["QLIKE vs. EWMA"][0].endswith("% lower")
    styles = table.proto.arrow_data.styler.styles
    assert "row0_col0" in styles and "row2_col0" in styles and "row1_col0" not in styles  # tint marks the best
    days_22 = pd.read_parquet(results / "forecasts.parquet").query("horizon == 22 and model == 'har'")
    assert acc["Days"].iloc[2] == days_22["date"].between(start, end)[days_22["realized"].notna()].sum()

    # Bias = mean forecast / mean realized on the same scored days, checked by hand.
    for i, (model, h) in enumerate([("har", 1), ("ewma", 1), ("har", 22), ("ewma", 22)]):
        assert acc["Bias (×)"].iloc[i] == pytest.approx(_hand_bias(results, model, h, start, end), rel=1e-12)

    # The data table shows a dash, not "None", where realized volatility isn't known yet.
    data = next(d for d in at.dataframe if "Realized (%)" in d.value.columns)
    assert data.proto.placeholder == "—" and data.value["Realized (%)"].isna().iloc[-1]

    # Log scale: the default on the full history, linear elsewhere, and the toggle overrides it.
    at.selectbox(key="period").select("Full history").run()
    assert at.segmented_control(key="scale").value == "Log" and _yaxis(at)["type"] == "log"
    assert _accuracy(at).value["Days"].tolist() == [metrics.loc[(h, "har"), "n"] for h in (1, 1, 22, 22)]
    at.segmented_control(key="scale").set_value("Linear").run()
    assert not at.exception and _yaxis(at)["type"] == "linear"
    at.segmented_control(key="scale").set_value("Log").run()
    assert not at.exception and _yaxis(at)["type"] == "log"

    at.segmented_control(key="horizon").set_value(1).run()
    assert not at.exception and any("following day" in c.value for c in at.caption)
    at.selectbox(key="period").select("2008 crisis").run()
    assert at.segmented_control(key="scale").value == "Linear" and _yaxis(at)["type"] == "linear"
    assert not at.exception and (_accuracy(at).value["Days"] < 200).all()
    at.pills(key="models").unselect("har").run()
    assert not at.exception and _accuracy(at).value["Model"].tolist() == ["EWMA", "EWMA"]
    assert "background-color" not in _accuracy(at).proto.arrow_data.styler.styles  # nothing to beat
    at.pills(key="models").unselect("ewma").run()
    assert not at.exception and "Select at least one model" in at.info[-1].value
    at.pills(key="models").select("ewma").select("har").run()
    at.selectbox(key="period").select("Custom range").run()
    assert not at.exception
    at.date_input(key="dates").set_value((date(2008, 10, 1), date(2008, 12, 31))).run()
    assert not at.exception and _accuracy(at).value["Days"].max() < 70
    at.date_input(key="dates").set_value((date(2008, 10, 1), date(2008, 10, 1))).run()
    assert not at.exception and "end date after the start date" in at.info[-1].value


def test_app_explains_missing_results(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path)
    assert not at.exception
    assert "No results found" in at.error[0].value
