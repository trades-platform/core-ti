"""Scalar indicators: Sharpe Ratio, Max Drawdown."""
from __future__ import annotations

import pandas as pd

from ..registry import register
from ..schema import Column


@register.scalar(
    "sharpe_ratio",
    requires=[Column("close", dtype="float64")],
    source="builtin",
)
def sharpe_ratio(close: pd.Series, period: int = 252, risk_free: float = 0.0) -> float:
    """Annualised Sharpe Ratio."""
    returns = close.pct_change().dropna()
    excess = returns - risk_free / period
    std = excess.std()
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * (period ** 0.5))


@register.scalar(
    "max_drawdown",
    requires=[Column("close", dtype="float64")],
    source="builtin",
)
def max_drawdown(close: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.35 = -35%)."""
    roll_max = close.cummax()
    drawdown = (close - roll_max) / roll_max
    return float(drawdown.min())
