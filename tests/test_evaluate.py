import numpy as np
import pandas as pd
import pytest

from volcast.evaluate import common_sample, qlike, squared_error, summarize


def test_qlike_is_zero_when_perfect_and_positive_otherwise():
    y = np.array([1e-4, 2e-4, 5e-4])
    assert np.allclose(qlike(y, y), 0)
    assert (qlike(y, 1.5 * y) > 0).all() and (qlike(y, 0.5 * y) > 0).all()


def test_qlike_penalizes_under_forecasting_more():
    assert qlike([2.0], [1.0])[0] > qlike([1.0], [2.0])[0]


def test_qlike_rejects_non_positive_forecasts():
    with pytest.raises(ValueError):
        qlike([1.0], [0.0])


def test_squared_error_is_in_percent_squared():
    assert squared_error([2e-4], [1e-4])[0] == pytest.approx(1.0)


def _frame():
    dates = pd.bdate_range("2020-01-01", periods=5)
    rows = []
    for i, d in enumerate(dates):
        realized = 1e-4 * (i + 1)
        rows.append({"date": d, "model": "ewma", "horizon": 1, "forecast": 1e-4, "realized": realized})
        if i != 2:  # "har" misses one day: that day must drop out for everyone
            rows.append({"date": d, "model": "har", "horizon": 1, "forecast": realized, "realized": realized})
    return pd.DataFrame(rows)


def test_models_are_scored_on_a_common_sample():
    wide = common_sample(_frame(), 1)
    assert len(wide) == 4 and set(wide.columns) == {"ewma", "har", "realized"}
    m = summarize(_frame()).set_index("model")
    assert (m["n"] == 4).all()
    assert m.loc["har", "qlike"] == pytest.approx(0) and m.loc["ewma", "qlike_rel"] == 1
    assert m.loc["har", "qlike_rel"] == pytest.approx(0)


def test_summarize_window_and_empty():
    fc = _frame()
    m = summarize(fc, start="2020-01-06")
    assert (m["n"] == 2).all()
    assert summarize(fc, start="2030-01-01").empty
