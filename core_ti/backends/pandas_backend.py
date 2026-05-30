"""
backends/pandas_backend.py — Default pandas backend.

Each indicator is registered in _IMPL as a callable:
    fn(inputs: dict[str, pd.Series], params: dict[str, Any])
       -> pd.Series | pd.DataFrame | float
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BackendProtocol


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def _sma(inputs: dict, params: dict) -> pd.Series:
    close = inputs["close"]
    return close.rolling(params["period"]).mean()


def _ema(inputs: dict, params: dict) -> pd.Series:
    close = inputs["close"]
    return close.ewm(span=params["period"], adjust=False).mean()


def _wma(inputs: dict, params: dict) -> pd.Series:
    close = inputs["close"]
    period = params["period"]
    weights = np.arange(1, period + 1, dtype=float)

    def _wma_apply(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return close.rolling(period).apply(_wma_apply, raw=True)


def _rsi(inputs: dict, params: dict) -> pd.Series:
    close = inputs["close"]
    period = params["period"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bb(inputs: dict, params: dict) -> pd.DataFrame:
    close = inputs["close"]
    period = params["period"]
    std_mult = params["std"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    p = params["period"]
    return pd.DataFrame({
        f"bb_{p}_upper": mid + std_mult * std,
        f"bb_{p}_mid": mid,
        f"bb_{p}_lower": mid - std_mult * std,
    })


def _atr(inputs: dict, params: dict) -> pd.Series:
    high = inputs["high"]
    low = inputs["low"]
    close = inputs["close"]
    period = params["period"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _macd(inputs: dict, params: dict) -> pd.DataFrame:
    ema_fast = inputs[f"ema_{params['fast']}"]
    ema_slow = inputs[f"ema_{params['slow']}"]
    fast = params["fast"]
    slow = params["slow"]
    signal_period = params["signal"]
    line = ema_fast - ema_slow
    signal = line.ewm(span=signal_period, adjust=False).mean()
    hist = line - signal
    return pd.DataFrame({
        f"macd_{fast}_{slow}_line": line,
        f"macd_{fast}_{slow}_signal": signal,
        f"macd_{fast}_{slow}_hist": hist,
    })


def _sharpe_ratio(inputs: dict, params: dict) -> float:
    close = inputs["close"]
    period = params.get("period", 252)
    risk_free = params.get("risk_free", 0.0)
    returns = close.pct_change().dropna()
    excess = returns - risk_free / period
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * (period ** 0.5))


def _max_drawdown(inputs: dict, params: dict) -> float:
    close = inputs["close"]
    roll_max = close.cummax()
    drawdown = (close - roll_max) / roll_max
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

_IMPL: dict[str, Any] = {
    "sma": _sma,
    "ema": _ema,
    "wma": _wma,
    "rsi": _rsi,
    "bb": _bb,
    "atr": _atr,
    "macd": _macd,
    "sharpe_ratio": _sharpe_ratio,
    "max_drawdown": _max_drawdown,
}


class PandasBackend:
    """Default pandas-based backend."""

    @property
    def name(self) -> str:
        return "pandas"

    def supports(self, indicator_name: str) -> bool:
        return indicator_name in _IMPL

    def compute(
        self,
        indicator_name: str,
        inputs: dict[str, pd.Series],
        params: dict[str, Any],
    ) -> pd.Series | pd.DataFrame | float:
        fn = _IMPL[indicator_name]
        return fn(inputs, params)
