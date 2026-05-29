"""
backends/talib_backend.py — TA-Lib backend stub.

Only active when the `TA-Lib` package is installed.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


class TALibBackend:
    """TA-Lib backend. Requires: pip install TA-Lib"""

    def __init__(self) -> None:
        try:
            import talib as _talib  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

    @property
    def name(self) -> str:
        return "talib"

    def supports(self, indicator_name: str) -> bool:
        if not self._available:
            return False
        return indicator_name in {"sma", "ema", "wma", "rsi", "atr", "bb", "macd"}

    def compute(
        self,
        indicator_name: str,
        inputs: dict[str, pd.Series],
        params: dict[str, Any],
    ) -> pd.Series | pd.DataFrame | float:
        import talib

        if indicator_name == "sma":
            return pd.Series(
                talib.SMA(inputs["close"].values, timeperiod=params["period"]),
                index=inputs["close"].index,
            )
        if indicator_name == "ema":
            return pd.Series(
                talib.EMA(inputs["close"].values, timeperiod=params["period"]),
                index=inputs["close"].index,
            )
        if indicator_name == "wma":
            return pd.Series(
                talib.WMA(inputs["close"].values, timeperiod=params["period"]),
                index=inputs["close"].index,
            )
        if indicator_name == "rsi":
            return pd.Series(
                talib.RSI(inputs["close"].values, timeperiod=params["period"]),
                index=inputs["close"].index,
            )
        if indicator_name == "atr":
            return pd.Series(
                talib.ATR(
                    inputs["high"].values,
                    inputs["low"].values,
                    inputs["close"].values,
                    timeperiod=params["period"],
                ),
                index=inputs["close"].index,
            )
        if indicator_name == "bb":
            upper, mid, lower = talib.BBANDS(
                inputs["close"].values,
                timeperiod=params["period"],
                nbdevup=params["std"],
                nbdevdn=params["std"],
                matype=0,
            )
            p = params["period"]
            return pd.DataFrame({
                f"bb_{p}_upper": upper,
                f"bb_{p}_mid": mid,
                f"bb_{p}_lower": lower,
            }, index=inputs["close"].index)
        if indicator_name == "macd":
            line, signal, hist = talib.MACD(
                inputs["close"].values,
                fastperiod=params["fast"],
                slowperiod=params["slow"],
                signalperiod=params["signal"],
            )
            fast, slow = params["fast"], params["slow"]
            return pd.DataFrame({
                f"macd_{fast}_{slow}_line": line,
                f"macd_{fast}_{slow}_signal": signal,
                f"macd_{fast}_{slow}_hist": hist,
            }, index=inputs["close"].index)
        raise NotImplementedError(f"talib backend: '{indicator_name}' not implemented")
