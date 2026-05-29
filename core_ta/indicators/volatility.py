"""Volatility indicators: Bollinger Bands, ATR."""
from __future__ import annotations

import pandas as pd

from ..registry import register
from ..schema import Column


@register.column(
    "bb",
    requires=[Column("close", dtype="float64")],
    outputs=["bb_{period}_upper", "bb_{period}_mid", "bb_{period}_lower"],
    source="builtin",
)
def bb(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (upper, mid, lower)."""
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return pd.DataFrame({
        f"bb_{period}_upper": mid + std * s,
        f"bb_{period}_mid": mid,
        f"bb_{period}_lower": mid - std * s,
    })


@register.column(
    "atr",
    requires=[
        Column("high", dtype="float64"),
        Column("low", dtype="float64"),
        Column("close", dtype="float64"),
    ],
    outputs=["atr_{period}"],
    source="builtin",
)
def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()
