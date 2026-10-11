"""Pipeline CLI: data -> targets -> walk-forward -> evaluation -> results/ (the app's only input).

    python -m volcast.pipeline --source yahoo      # live SPY + VIX
    python -m volcast.pipeline --source sample     # offline sample bundled with arch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .backtest import run_backtests
from .config import CRISIS_WINDOWS, RAW_DIR, RESULTS_DIR, BacktestConfig
from .data import load_raw
from .evaluate import summarize
from .models import labels
from .targets import build_panel

log = logging.getLogger("volcast")

PROXY_RATIO_OK = (0.8, 1.25)  # acceptable E[r^2] / E[v]; outside it the proxy's level is suspect


def proxy_diagnostics(panel: pd.DataFrame) -> dict:
    """How the variance proxy's level compares with squared close-to-close returns."""
    ret2 = panel["ret"] ** 2
    by_year = ret2.groupby(panel.index.year).mean() / panel["v"].groupby(panel.index.year).mean()
    ratio = float(ret2.mean() / panel["v"].mean())
    return {
        "ret2_over_v": round(ratio, 3),
        "ret2_over_v_by_year": {int(y): round(float(r), 3) for y, r in by_year.items()},
        "overnight_share_of_v": round(float((panel["r_on"] ** 2).mean() / panel["v"].mean()), 3),
        "ok": bool(PROXY_RATIO_OK[0] <= ratio <= PROXY_RATIO_OK[1]),
    }


def write_results(
    out_dir: Path,
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_dir / "forecasts.parquet", index=False)
    metrics.to_parquet(out_dir / "metrics.parquet", index=False)
    cols = ["close", "ret", "v", "vix"]
    panel[cols].reset_index().to_parquet(out_dir / "panel.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def run(
    source: str = "yahoo",
    *,
    refresh: bool = True,
    out_dir: Path = RESULTS_DIR,
    raw_dir: Path = RAW_DIR,
    cfg: BacktestConfig = BacktestConfig(),
) -> dict:
    """Run everything and write results. Returns the metadata written to meta.json."""
    t0 = time.perf_counter()
    raw, meta = load_raw(source, refresh=refresh, raw_dir=raw_dir)
    panel = build_panel(raw)
    diag = proxy_diagnostics(panel)
    if not diag["ok"]:
        log.warning(
            "variance proxy level looks off: E[r^2]/E[v] = %.2f (expected %.2f-%.2f). "
            "On index data this usually means stale opens; see DESIGN.md §2.",
            diag["ret2_over_v"], *PROXY_RATIO_OK,
        )

    forecasts = run_backtests(panel, cfg=cfg)
    metrics = summarize(forecasts)

    first_origin = forecasts["date"].min()
    meta.update(
        {
            "data_start": panel.index[0].date().isoformat(),
            "data_end": panel.index[-1].date().isoformat(),
            "oos_start": first_origin.date().isoformat(),
            "oos_origins": int(forecasts["date"].nunique()),
            "refit_every": cfg.refit_every,
            "horizons": list(cfg.horizons),
            "models": labels(),
            "crisis_windows": CRISIS_WINDOWS,
            "proxy": diag,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_sec": round(time.perf_counter() - t0, 1),
            "versions": {"volcast": __version__, "pandas": pd.__version__, "numpy": np.__version__},
        }
    )
    write_results(Path(out_dir), panel, forecasts, metrics, meta)
    log.info("wrote %s (%d forecasts, %.1fs)", out_dir, len(forecasts), meta["runtime_sec"])
    return meta


def _print_metrics(out_dir: Path) -> None:
    m = pd.read_parquet(Path(out_dir) / "metrics.parquet")
    if m.empty:
        return
    view = m[["horizon", "model", "n", "qlike", "qlike_rel", "mse_pct2", "mse_rel"]].sort_values(["horizon", "qlike"])
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(view.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="volcast.pipeline", description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["yahoo", "sample"], default="yahoo")
    parser.add_argument("--no-refresh", action="store_true", help="use cached downloads instead of fetching")
    parser.add_argument("--oos-start", default=BacktestConfig.oos_start, help="first out-of-sample origin")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="results directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = replace(BacktestConfig(), oos_start=args.oos_start)
    run(args.source, refresh=not args.no_refresh, out_dir=args.out, cfg=cfg)
    _print_metrics(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
