import numpy as np
import pandas as pd
import pytest

from volcast import data as data_mod
from volcast.data import DataError, align_vix, load_raw, load_sample, validate_ohlc


def _bars(n=60):
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close}, index=idx
    )


def test_clean_bars_pass_unchanged():
    bars = _bars()
    out, report = validate_ohlc(bars)
    pd.testing.assert_frame_equal(out, bars)
    assert report.rows == 60 and report.repaired_bars == 0 and report.max_gap_days == 3


def test_tiny_inconsistency_is_repaired_and_counted():
    bars = _bars()
    bars.iloc[5, bars.columns.get_loc("high")] = bars["close"].iloc[5] * (1 - 1e-5)
    out, report = validate_ohlc(bars)
    assert report.repaired_bars == 1
    assert out["high"].iloc[5] == out[["open", "close"]].iloc[5].max()


def test_large_inconsistency_fails():
    bars = _bars()
    bars.iloc[5, bars.columns.get_loc("low")] = bars["close"].iloc[5] * 1.05
    with pytest.raises(DataError, match="bars break"):
        validate_ohlc(bars)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda b: pd.concat([b, b.iloc[[3]]]).sort_index(), "duplicate"),
        (lambda b: b.iloc[::-1], "not sorted"),
        (lambda b: b.assign(close=-b["close"]), "bars break|non-positive"),
        (lambda b: b.drop(columns="low"), "missing columns"),
        (lambda b: b.drop(index=b.index[10:25]), "gap"),
        (lambda b: b.assign(open=np.where(np.arange(len(b)) < 5, np.nan, b["open"])), "missing prices"),
    ],
)
def test_bad_data_fails_loudly(mutate, message):
    with pytest.raises(DataError, match=message):
        validate_ohlc(mutate(_bars()))


def test_a_few_nan_rows_are_dropped():
    bars = _bars(200)
    bars.iloc[50, 0] = np.nan
    out, report = validate_ohlc(bars)
    assert report.dropped_nan_rows == 1 and len(out) == 199


def test_align_vix_fills_at_most_three_days_backward_looking():
    idx = pd.bdate_range("2020-01-01", periods=8)
    vix = pd.Series([20.0, 21.0], index=idx[[0, 1]])
    out = align_vix(vix, idx)
    assert out.iloc[:5].tolist() == [20, 21, 21, 21, 21]
    assert out.iloc[5:].isna().all()
    assert align_vix(None, idx).isna().all()


class _FakeTicker:
    def __init__(self, frame):
        self.frame = frame

    def history(self, **kwargs):
        if isinstance(self.frame, Exception):
            raise self.frame
        return self.frame


def _patch_yf(monkeypatch, frame):
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", lambda ticker: _FakeTicker(frame))


def test_fetch_yahoo_normalizes_columns_and_timezone(monkeypatch):
    idx = pd.date_range("2024-01-02 00:00", periods=3, freq="B", tz="America/New_York")
    frame = pd.DataFrame(
        {"Open": [1.0, 2, 3], "High": [2.0, 3, 4], "Low": [0.5, 1, 2], "Close": [1.5, 2.5, 3.5], "Volume": [10, 20, 30]},
        index=idx,
    )
    _patch_yf(monkeypatch, frame)
    out = data_mod.fetch_yahoo("SPY")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is None and out.index.name == "date"
    assert out.index[0] == pd.Timestamp("2024-01-02")


@pytest.mark.parametrize("frame", [pd.DataFrame(), RuntimeError("blocked")])
def test_fetch_yahoo_failures_raise_data_error(monkeypatch, frame):
    _patch_yf(monkeypatch, frame)
    with pytest.raises(DataError):
        data_mod.fetch_yahoo("SPY")


def test_yahoo_source_caches_and_reads_back(monkeypatch, tmp_path, raw):
    bars = raw[["open", "high", "low", "close", "volume"]]
    monkeypatch.setattr(data_mod, "fetch_yahoo", lambda ticker, start="1993-01-01": bars)
    monkeypatch.setattr(data_mod, "fetch_vix", lambda start="1990-01-01": raw["vix"])
    fresh, meta = load_raw("yahoo", raw_dir=tmp_path)
    cached, _ = load_raw("yahoo", refresh=False, raw_dir=tmp_path)
    pd.testing.assert_frame_equal(fresh, cached, check_freq=False)
    assert meta["vix_available"] and "days_since_last_bar" in meta


def test_missing_cache_is_a_clear_error(tmp_path):
    with pytest.raises(DataError, match="no cached data"):
        load_raw("yahoo", refresh=False, raw_dir=tmp_path)


def test_unknown_source():
    with pytest.raises(ValueError):
        load_raw("bloomberg")


def test_offline_sample_loads():
    bars, vix = load_sample()
    assert bars.index[0] == pd.Timestamp("1999-01-04") and bars.index[-1] == pd.Timestamp("2018-12-31")
    assert {"open", "high", "low", "close"} <= set(bars.columns)
    raw, meta = load_raw("sample")
    assert meta["validation"]["rows"] == len(raw) and "vix" in raw.columns
