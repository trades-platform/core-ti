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
        return indicator_name in {"sma", "ema", "wma", "rsi", "atr"}

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
        raise NotImplementedError(f"talib backend: '{indicator_name}' not implemented")
