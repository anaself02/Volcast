"""End to end on the offline sample: pipeline -> results/ -> app."""

import json
from datetime import date

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from volcast.config import ROOT
from volcast.models import REGISTRY, labels
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
    assert len(metrics) == 2 * len(REGISTRY) and {"qlike", "mse_pct2", "qlike_rel", "mse_rel"} <= set(metrics.columns)
    panel = pd.read_parquet(results / "panel.parquet")
    assert list(panel.columns) == ["date", "close", "ret", "v", "vix"]
    meta = json.loads((results / "meta.json").read_text())
    assert meta["source"] == "sample" and meta["oos_start"] == "2005-01-03"
    assert meta["models"] == labels() and meta["horizons"] == [1, 22]
    # The index sample's stale opens make the proxy run low (DESIGN.md §2): flagged, not hidden.
    assert meta["proxy"]["ok"] is False and meta["proxy"]["ret2_over_v"] > 1.25

    # Phase 2 tables: every model tested against EWMA and HAR, the MCS, Mincer–Zarnowitz, regimes, refits.
    dm = pd.read_parquet(results / "dm.parquet")
    assert list(dm.columns) == ["horizon", "model", "benchmark", "stat", "p_value", "mean_diff", "n", "lags"]
    assert len(dm) == 2 * 2 * (len(REGISTRY) - 1) and dm["p_value"].between(0, 1).all()
    mcs = pd.read_parquet(results / "mcs.parquet")
    assert list(mcs.columns) == ["horizon", "model", "qlike", "mcs_pvalue", "in_mcs"] and len(mcs) == 2 * len(REGISTRY)
    assert mcs.groupby("horizon")["in_mcs"].any().all()  # the set is never empty
    mz = pd.read_parquet(results / "mz.parquet")
    assert {"alpha", "beta", "se_alpha", "se_beta", "r2", "wald", "p_value", "lags"} <= set(mz.columns)
    assert len(mz) == 2 * len(REGISTRY)
    regimes = pd.read_parquet(results / "regimes.parquet")
    assert set(regimes["kind"]) == {"stress", "window", "vix"}
    split = regimes[(regimes["kind"] == "stress") & (regimes["model"] == "ewma")].groupby("horizon")["n"].sum()
    assert split.to_dict() == metrics.query("model == 'ewma'").set_index("horizon")["n"].to_dict()
    refits = pd.read_parquet(results / "refits.parquet")
    assert set(refits["model"]) == set(REGISTRY) and refits["status"].isin(["ok", "fallback"]).all()
    assert meta["tests"]["mcs"]["seed"] == 20_111
    assert set(meta["refit_issues"]) <= {f"{m}/{h}" for m in REGISTRY for h in (1, 22)}


def test_cli(tmp_path, capsys):
    # A late out-of-sample start keeps this end-to-end CLI run short; the fixture above runs the full one.
    assert main(["--source", "sample", "--out", str(tmp_path), "--oos-start", "2017-01-03"]) == 0
    assert (tmp_path / "meta.json").exists()
    assert "qlike" in capsys.readouterr().out


def _app(monkeypatch, results_dir):
    monkeypatch.setenv("VOLCAST_RESULTS_DIR", str(results_dir))
    return AppTest.from_file(APP, default_timeout=60).run()


def _accuracy(at):
    return next(d for d in at.dataframe if "Bias (×)" in d.value.columns)


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


def _hand_days(results, horizon, start, end):
    fc = pd.read_parquet(results / "forecasts.parquet")
    fc = fc[(fc["horizon"] == horizon) & (fc["date"] >= start) & (fc["date"] <= end)]
    days = fc.pivot(index="date", columns="model", values="forecast")
    days["realized"] = fc.groupby("date")["realized"].first()
    return len(days.dropna())


def test_app_renders_and_every_control_works(monkeypatch, results):
    at = _app(monkeypatch, results)
    assert not at.exception
    meta = json.loads((results / "meta.json").read_text())

    # Header numbers come straight from results/: the status strip from meta.json, the headline from
    # metrics.parquet, the latest forecast from forecasts.parquet.
    strip = next(e.proto.body for e in at.get("html") if 'class="vc-strip"' in e.proto.body)
    assert f"{meta['data_end']}" in strip and f"{meta['oos_start']}" in strip and "Last refit" in strip
    names = meta["models"]  # key -> label, in registry order
    key_of = {label: key for key, label in names.items()}
    k = len(names)
    metrics = pd.read_parquet(results / "metrics.parquet").set_index(["horizon", "model"])
    best = {h: metrics.xs(h, level="horizon")["qlike"].idxmin() for h in (1, 22)}
    head_22, head_1, latest_22, latest_1, vix = at.metric[:5]
    for tile, h in [(head_22, 22), (head_1, 1)]:
        assert tile.value == f"{(1 - metrics.loc[(h, best[h]), 'qlike_rel']) * 100:.1f}%"
        assert tile.label.endswith(names[best[h]]) and tile.delta == "lower QLIKE than EWMA"
    assert latest_22.label == f"Next 22 days · {names[best[22]]}" and latest_1.label == f"Next day · {names[best[1]]}"
    assert latest_22.delta.endswith(" pts") and vix.label == "VIX"

    # The headline says whether each best model's edge is significant, from dm.parquet and mcs.parquet.
    dm = pd.read_parquet(results / "dm.parquet").set_index(["horizon", "model", "benchmark"])
    mcs = pd.read_parquet(results / "mcs.parquet")
    edges = [c.value for c in at.caption if c.value.startswith("Edge ")]
    for line, h in zip(edges, (22, 1)):
        if best[h] != "ewma":
            p = dm.loc[(h, best[h], "ewma"), "p_value"]
            assert ("not significant at 5%" if p >= 0.05 else "significant at 5%") in line.split(";")[0]
    mcs_line = next(c.value for c in at.caption if c.value.startswith("90% Model Confidence Set"))
    for h in (1, 22):
        for m in mcs.query("horizon == @h and in_mcs")["model"]:
            assert names[m] in mcs_line
    assert not any("Not yet tested for statistical significance" in c.value for c in at.caption)

    # Model comparison tables: significance (MCS rows shaded), Mincer–Zarnowitz, regimes.
    assert [t.label for t in at.tabs] == ["Significance", "Calibration", "By regime"]
    sig = next(d for d in at.dataframe if "In 90% MCS" in d.value.columns)
    rows = sig.value
    assert len(rows) == 2 * k and (rows["In 90% MCS"] == "yes").sum() == mcs["in_mcs"].sum()
    shaded = {i for i in range(len(rows)) if f"row{i}_col0" in sig.proto.arrow_data.styler.styles}
    assert shaded == set(rows.index[rows["In 90% MCS"] == "yes"])
    for _, r in rows.iterrows():
        h = 1 if r["Horizon"] == "1-day" else 22
        if r["Model"] != "EWMA":
            p = dm.loc[(h, key_of[r["Model"]], "ewma"), "p_value"]
            assert r["p vs. EWMA"] == ("<0.001" if p < 0.001 else f"{p:.3f}")
    mz_rows = next(d for d in at.dataframe if "β" in d.value.columns).value
    assert len(mz_rows) == 2 * k and mz_rows["Days"].gt(0).all()
    regime_rows = [d.value for d in at.dataframe if "Regime" in d.value.columns]
    assert len(regime_rows) == 2 and set(regime_rows[0]["Regime"]) >= {"Inside stress windows", "Outside stress windows"}

    # Caveats stay visible: the sample, the proxy check, the method notes.
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
    assert acc["Horizon"].tolist() == ["1-day"] * k + ["22-day"] * k
    assert sorted(acc["Model"][:k]) == sorted(names.values()) == sorted(acc["Model"][k:])
    assert acc["QLIKE"][:k].is_monotonic_increasing and acc["QLIKE"][k:].is_monotonic_increasing  # best first
    assert (acc.loc[acc["Model"] == "EWMA", "QLIKE vs. EWMA"] == "baseline").all()
    styles = table.proto.arrow_data.styler.styles
    tinted = {i for i in range(2 * k) if f"row{i}_col0" in styles}
    assert tinted == {0, k}  # the tint marks the best model at each horizon, nothing else

    # Days and Bias (mean forecast / mean realized) on the common sample, checked by hand.
    for i, row in acc.iterrows():
        h = 1 if row["Horizon"] == "1-day" else 22
        assert row["Bias (×)"] == pytest.approx(_hand_bias(results, key_of[row["Model"]], h, start, end), rel=1e-12)
        assert row["Days"] == _hand_days(results, h, start, end)

    # The data table shows a dash, not "None", where realized volatility isn't known yet.
    data = next(d for d in at.dataframe if "Realized (%)" in d.value.columns)
    assert data.proto.placeholder == "—" and data.value["Realized (%)"].isna().iloc[-1]

    # Log scale: the default on the full history, linear elsewhere, and the toggle overrides it.
    at.selectbox(key="period").select("Full history").run()
    assert at.segmented_control(key="scale").value == "Log" and _yaxis(at)["type"] == "log"
    assert _accuracy(at).value["Days"].tolist() == [metrics.loc[(h, "ewma"), "n"] for h in [1] * k + [22] * k]
    at.segmented_control(key="scale").set_value("Linear").run()
    assert not at.exception and _yaxis(at)["type"] == "linear"
    at.segmented_control(key="scale").set_value("Log").run()
    assert not at.exception and _yaxis(at)["type"] == "log"

    at.segmented_control(key="horizon").set_value(1).run()
    assert not at.exception and any("following day" in c.value for c in at.caption)
    at.selectbox(key="period").select("2008 crisis").run()
    assert at.segmented_control(key="scale").value == "Linear" and _yaxis(at)["type"] == "linear"
    assert not at.exception and (_accuracy(at).value["Days"] < 200).all()
    for key in names:
        if key != "ewma":
            at.pills(key="models").unselect(key).run()
    assert not at.exception and _accuracy(at).value["Model"].tolist() == ["EWMA", "EWMA"]
    assert "background-color" not in _accuracy(at).proto.arrow_data.styler.styles  # nothing to beat
    at.pills(key="models").unselect("ewma").run()
    assert not at.exception and "Select at least one model" in at.info[-1].value
    for key in names:
        at.pills(key="models").select(key)
    at.run()
    at.selectbox(key="period").select("Custom range").run()
    assert not at.exception
    at.date_input(key="dates").set_value((date(2008, 10, 1), date(2008, 12, 31))).run()
    assert not at.exception and _accuracy(at).value["Days"].max() < 70
    at.date_input(key="dates").set_value((date(2008, 10, 1), date(2008, 10, 1))).run()
    assert not at.exception and "end date after the start date" in at.info[-1].value


def test_app_reads_results_written_before_phase_2(monkeypatch, results, tmp_path):
    for name in ("meta.json", "forecasts.parquet", "panel.parquet", "metrics.parquet"):
        (tmp_path / name).write_bytes((results / name).read_bytes())
    at = _app(monkeypatch, tmp_path)
    assert not at.exception
    assert any("predate the significance tests" in c.value for c in at.caption)
    assert not at.tabs and _accuracy(at).value["Bias (×)"].notna().all()


def test_app_explains_missing_results(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path)
    assert not at.exception
    assert "No results found" in at.error[0].value
