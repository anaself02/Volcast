"""End to end on the offline sample: pipeline -> results/ -> app."""

import json

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


def test_app_renders_and_every_control_works(monkeypatch, results):
    at = _app(monkeypatch, results)
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["EWMA", "HAR-RV"]
    assert at.dataframe[0].value["Model"].tolist() == ["HAR-RV", "EWMA"]

    at.selectbox(key="period").select("2008 crisis").run()
    assert not at.exception and (at.dataframe[0].value["Days scored"] < 200).all()
    at.multiselect(key="models").unselect("har").run()
    assert not at.exception and at.dataframe[0].value["Model"].tolist() == ["EWMA"]
    at.selectbox(key="period").select("Custom range").run()
    assert not at.exception


def test_app_explains_missing_results(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path)
    assert not at.exception
    assert "No results found" in at.error[0].value
