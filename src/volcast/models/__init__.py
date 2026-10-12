"""Model registry. Every registered model is covered by tests/test_lookahead.py automatically."""

from __future__ import annotations

from .base import VolModel
from .ewma import EWMA
from .garch import GJRGARCH
from .gbm import LightGBM
from .har import HAR

REGISTRY: dict[str, type[VolModel]] = {cls.name: cls for cls in (EWMA, HAR, GJRGARCH, LightGBM)}


def labels() -> dict[str, str]:
    """Display name for each registered model."""
    return {name: cls.label for name, cls in REGISTRY.items()}


__all__ = ["REGISTRY", "VolModel", "EWMA", "GJRGARCH", "HAR", "LightGBM", "labels"]
