"""Look-ahead check (DESIGN.md §9).

Shuffle every raw row after a cutoff date, rebuild the panel, rerun the walk-forward and require
every forecast made on or before the cutoff to be unchanged. Any dependence on the future,
through a non-causal feature or a leaky refit, shows up as a changed forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import ModelFactory, refit_positions, walk_forward
from .config import BacktestConfig
from .targets import build_panel


def scramble_after(raw: pd.DataFrame, cutoff, seed: int = 0) -> pd.DataFrame:
    """Copy of `raw` where the rows strictly after `cutoff` are randomly permuted (values only)."""
    after = raw.index > pd.Timestamp(cutoff)
    n = int(after.sum())
    if n < 2:
        raise ValueError("need at least two rows after the cutoff to scramble")
    perm = np.random.default_rng(seed).permutation(n)
    if np.array_equal(perm, np.arange(n)):
        perm = perm[::-1]
    out = raw.copy()
    out.loc[after] = raw.loc[after].to_numpy()[perm]
    return out


def default_cutoffs(index: pd.DatetimeIndex, cfg: BacktestConfig) -> list[pd.Timestamp]:
    """A refit date (catches refit/embargo leaks) and a mid-block date (catches non-causal features)."""
    starts = refit_positions(index, cfg.oos_start, cfg.refit_every)
    if len(starts) < 5:
        raise ValueError("backtest too short for the look-ahead check; need at least 5 refit blocks")
    return [index[starts[2]], index[starts[3] + cfg.refit_every // 2]]


def assert_no_lookahead(
    raw: pd.DataFrame,
    model_factory: ModelFactory,
    horizon: int,
    cfg: BacktestConfig = BacktestConfig(),
    cutoffs: list | None = None,
    rtol: float = 1e-10,
    seed: int = 0,
) -> None:
    """Raise AssertionError if any forecast dated on/before a cutoff depends on data after it."""
    base = walk_forward(build_panel(raw), model_factory, horizon, cfg).set_index("date")["forecast"]
    name = model_factory().name
    for cutoff in cutoffs or default_cutoffs(build_panel(raw).index, cfg):
        cutoff = pd.Timestamp(cutoff)
        scrambled = scramble_after(raw, cutoff, seed)
        test = walk_forward(build_panel(scrambled), model_factory, horizon, cfg).set_index("date")["forecast"]
        a, b = base.loc[:cutoff], test.loc[:cutoff]
        if a.empty:
            raise ValueError(f"cutoff {cutoff.date()} is before the first forecast origin")
        if not a.index.equals(b.index):
            raise AssertionError(f"{name} h={horizon}: forecast dates changed when data after {cutoff.date()} was scrambled")
        same = np.isclose(a.to_numpy(), b.to_numpy(), rtol=rtol, atol=0.0)
        if not same.all():
            first = a.index[~same][0]
            raise AssertionError(
                f"{name} h={horizon}: the forecast made on {first.date()} changed when data after "
                f"{cutoff.date()} was scrambled, so it uses information from the future"
            )
