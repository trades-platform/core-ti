"""Oscillator indicators: RSI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import register
from ..schema import Column


@register.column(
    "rsi",
    requires=[Column("close", dtype="float64")],
    outputs=["rsi_{period}"],
    source="builtin",
)
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing via EWM)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
