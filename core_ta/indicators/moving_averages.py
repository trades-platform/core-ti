"""Moving average indicators: SMA, EMA, WMA."""
from __future__ import annotations

import pandas as pd

from ..registry import register
from ..schema import Column


@register.column(
    "sma",
    requires=[Column("close", dtype="float64")],
    outputs=["sma_{period}"],
    source="builtin",
)
def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(period).mean()


@register.column(
    "ema",
    requires=[Column("close", dtype="float64")],
    outputs=["ema_{period}"],
    source="builtin",
)
def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


@register.column(
    "wma",
    requires=[Column("close", dtype="float64")],
    outputs=["wma_{period}"],
    source="builtin",
)
def wma(close: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average (linearly weighted)."""
    import numpy as np
    weights = np.arange(1, period + 1, dtype=float)
    return close.rolling(period).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
    )
