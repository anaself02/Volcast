"""Data loading: Yahoo Finance (live), CBOE (VIX backup) and the bundled sample (offline).

Every source goes through `validate_ohlc`, which fails loudly on bad data instead of letting
it flow silently into the models (DESIGN.md §2).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR

log = logging.getLogger(__name__)

OHLC = ["open", "high", "low", "close"]
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

SOURCES = {
    "yahoo": "SPY daily (Yahoo Finance, dividend-adjusted) and VIX",
    "sample": "Offline sample: S&P 500 index 1999–2018 bundled with the arch package",
}


class DataError(RuntimeError):
    """Data couldn't be fetched or failed validation."""


@dataclass
class ValidationReport:
    rows: int
    start: str
    end: str
    dropped_nan_rows: int
    repaired_bars: int
    zero_range_days: int
    max_gap_days: int

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- loaders


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and a sorted, tz-naive, midnight-normalized index named 'date'."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    return out.sort_index()


def fetch_yahoo(ticker: str, start: str = "1993-01-01") -> pd.DataFrame:
    """Daily bars from Yahoo Finance, adjusted for splits and dividends."""
    import yfinance as yf  # lazy: only the pipeline needs it

    try:
        raw = yf.Ticker(ticker).history(start=start, auto_adjust=True, actions=False)
    except Exception as exc:  # yfinance raises many different types
        raise DataError(f"Yahoo download failed for {ticker}: {exc}") from exc
    if raw is None or raw.empty:
        raise DataError(f"Yahoo returned no data for {ticker}")
    df = _normalize(raw)
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise DataError(f"Yahoo data for {ticker} is missing columns {missing}")
    cols = OHLC + (["volume"] if "volume" in df.columns else [])
    return df[cols].astype(float)


def fetch_vix(start: str = "1990-01-01") -> pd.Series:
    """VIX daily close: Yahoo first, CBOE's history file as the backup."""
    try:
        return fetch_yahoo("^VIX", start)["close"].rename("vix")
    except DataError as exc:
        log.warning("Yahoo VIX failed (%s); trying CBOE", exc)
    try:
        raw = pd.read_csv(CBOE_VIX_URL)
    except Exception as exc:
        raise DataError(f"VIX download failed from both Yahoo and CBOE: {exc}") from exc
    raw.columns = [c.strip().lower() for c in raw.columns]
    dates = pd.to_datetime(raw["date"], format="%m/%d/%Y", errors="coerce")
    vix = pd.Series(raw["close"].astype(float).to_numpy(), index=pd.DatetimeIndex(dates), name="vix")
    vix = vix[vix.index.notna()].sort_index()
    vix.index.name = "date"
    return vix.loc[pd.Timestamp(start):]


def load_sample() -> tuple[pd.DataFrame, pd.Series]:
    """S&P 500 index bars (1999–2018) and VIX (2014–2019) bundled with `arch`. Works offline."""
    from arch.data import sp500, vix

    bars = _normalize(sp500.load())[OHLC + ["volume"]].astype(float)
    vix_close = _normalize(vix.load())["vix"].astype(float)
    return bars, vix_close


# ----------------------------------------------------------------------------- validation


def validate_ohlc(
    df: pd.DataFrame,
    *,
    tol: float = 1e-3,
    max_repair_frac: float = 0.02,
    max_nan_frac: float = 0.01,
    max_gap_days: int = 10,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Check daily bars and repair only tiny rounding inconsistencies.

    Raises DataError for unsorted or duplicate dates, missing columns, too many NaN rows,
    non-positive prices, OHLC inconsistencies larger than `tol` (relative) or on more than
    `max_repair_frac` of rows, and calendar gaps longer than `max_gap_days`.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError("index must be a DatetimeIndex")
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()[:3]
        raise DataError(f"duplicate dates, e.g. {[d.date().isoformat() for d in dupes]}")
    if not df.index.is_monotonic_increasing:
        raise DataError("dates are not sorted")
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise DataError(f"missing columns {missing}")

    out = df.copy()
    nan_rows = out[OHLC].isna().any(axis=1)
    if nan_rows.mean() > max_nan_frac:
        raise DataError(f"{nan_rows.mean():.1%} of rows have missing prices")
    out = out.loc[~nan_rows]
    if len(out) < 30:
        raise DataError(f"only {len(out)} usable rows")
    if (out[OHLC] <= 0).any().any():
        raise DataError("non-positive prices")

    top = out[["open", "close"]].max(axis=1)
    bottom = out[["open", "close"]].min(axis=1)
    high_bad = out["high"] < top
    low_bad = out["low"] > bottom
    size = np.maximum(
        (top / out["high"] - 1).where(high_bad, 0.0),
        (out["low"] / bottom - 1).where(low_bad, 0.0),
    )
    bad = high_bad | low_bad
    if (size > tol).any():
        worst = size[size > tol].sort_values(ascending=False).index[:3]
        raise DataError(
            f"{int((size > tol).sum())} bars break high >= max(open, close) >= min(open, close) >= low "
            f"by more than {tol:.1%}, e.g. {[d.date().isoformat() for d in worst]}"
        )
    if bad.mean() > max_repair_frac:
        raise DataError(f"{bad.mean():.1%} of bars are inconsistent (limit {max_repair_frac:.0%})")
    out.loc[high_bad, "high"] = top[high_bad]
    out.loc[low_bad, "low"] = bottom[low_bad]

    gaps = out.index.to_series().diff().dt.days
    max_gap = int(gaps.max()) if gaps.notna().any() else 0
    if max_gap > max_gap_days:
        gap_end = gaps.idxmax().date().isoformat()
        raise DataError(f"{max_gap}-day gap in the data ending {gap_end}")

    report = ValidationReport(
        rows=len(out),
        start=out.index[0].date().isoformat(),
        end=out.index[-1].date().isoformat(),
        dropped_nan_rows=int(nan_rows.sum()),
        repaired_bars=int(bad.sum()),
        zero_range_days=int((out["high"] == out["low"]).sum()),
        max_gap_days=max_gap,
    )
    return out, report


def align_vix(vix: pd.Series | None, index: pd.DatetimeIndex, max_fill: int = 3) -> pd.Series:
    """VIX on the bar dates, forward-filled over at most `max_fill` missing days (backward-looking only)."""
    if vix is None or vix.empty:
        return pd.Series(np.nan, index=index, name="vix")
    union = vix.reindex(index.union(vix.index)).sort_index()
    return union.ffill(limit=max_fill).reindex(index).rename("vix")


# ----------------------------------------------------------------------------- entry point


def _cache_file(raw_dir: Path, name: str) -> Path:
    return Path(raw_dir) / f"{name}.parquet"


def load_raw(
    source: str = "yahoo",
    *,
    refresh: bool = True,
    raw_dir: Path = RAW_DIR,
    start: str = "1993-01-01",
) -> tuple[pd.DataFrame, dict]:
    """Validated daily bars plus a `vix` column, and metadata about where they came from.

    `source="yahoo"` downloads SPY and VIX (cached to `raw_dir`); with `refresh=False` it reads
    the cache instead. `source="sample"` uses the offline dataset bundled with `arch`.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(SOURCES)}")

    vix: pd.Series | None
    if source == "sample":
        bars, vix = load_sample()
    elif refresh:
        bars = fetch_yahoo("SPY", start)
        try:
            vix = fetch_vix()
        except DataError as exc:
            log.warning("continuing without VIX: %s", exc)
            vix = None
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        bars.to_parquet(_cache_file(raw_dir, "spy"))
        if vix is not None:
            vix.to_frame().to_parquet(_cache_file(raw_dir, "vix"))
    else:
        spy_file = _cache_file(raw_dir, "spy")
        if not spy_file.exists():
            raise DataError(f"no cached data at {spy_file}; run once with refresh enabled")
        bars = pd.read_parquet(spy_file)
        vix_file = _cache_file(raw_dir, "vix")
        vix = pd.read_parquet(vix_file)["vix"] if vix_file.exists() else None

    bars, report = validate_ohlc(bars)
    raw = bars.copy()
    raw["vix"] = align_vix(vix, raw.index)

    meta = {
        "source": source,
        "source_label": SOURCES[source],
        "validation": report.to_dict(),
        "vix_available": bool(raw["vix"].notna().any()),
    }
    if source == "yahoo":
        age = (pd.Timestamp.today().normalize() - raw.index[-1]).days
        meta["days_since_last_bar"] = int(age)
        if age > 7:
            log.warning("latest bar is %d days old (%s)", age, raw.index[-1].date())
    return raw, meta
