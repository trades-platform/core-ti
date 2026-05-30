"""backends/tulipy_backend.py — Tulip Indicators backend."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class TulipyBackend:
    """Tulipy backend (Tulip Indicators C library). Requires: pip install tulipy"""

    def __init__(self) -> None:
        try:
            import tulipy as _ti  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

    @property
    def name(self) -> str:
        return "tulipy"

    def supports(self, indicator_name: str) -> bool:
        if not self._available:
            return False
        return indicator_name in {"sma", "ema", "wma", "rsi", "bb", "atr", "macd"}

    def _trim_index(self, idx: pd.Index, arr: np.ndarray) -> pd.Index:
        return idx[-len(arr):]

    def compute(
        self,
        indicator_name: str,
        inputs: dict[str, pd.Series],
        params: dict[str, Any],
    ) -> pd.Series | pd.DataFrame | float:
        import tulipy as ti

        idx = inputs["close"].index

        if indicator_name == "sma":
            result = ti.sma(inputs["close"].values, params["period"])
            return pd.Series(result, index=self._trim_index(idx, result))
        if indicator_name == "ema":
            result = ti.ema(inputs["close"].values, params["period"])
            return pd.Series(result, index=self._trim_index(idx, result))
        if indicator_name == "wma":
            result = ti.wma(inputs["close"].values, params["period"])
            return pd.Series(result, index=self._trim_index(idx, result))
        if indicator_name == "rsi":
            result = ti.rsi(inputs["close"].values, params["period"])
            return pd.Series(result, index=self._trim_index(idx, result))
        if indicator_name == "atr":
            result = ti.atr(
                inputs["high"].values, inputs["low"].values,
                inputs["close"].values, params["period"],
            )
            return pd.Series(result, index=self._trim_index(idx, result))
        if indicator_name == "bb":
            lower, mid, upper = ti.bbands(
                inputs["close"].values, params["period"], params["std"],
            )
            tidx = self._trim_index(idx, upper)
            p = params["period"]
            return pd.DataFrame({
                f"bb_{p}_upper": upper,
                f"bb_{p}_mid": mid,
                f"bb_{p}_lower": lower,
            }, index=tidx)
        if indicator_name == "macd":
            line, signal, hist = ti.macd(
                inputs["close"].values, params["fast"], params["slow"], params["signal"],
            )
            tidx = self._trim_index(idx, line)
            fast, slow = params["fast"], params["slow"]
            return pd.DataFrame({
                f"macd_{fast}_{slow}_line": line,
                f"macd_{fast}_{slow}_signal": signal,
                f"macd_{fast}_{slow}_hist": hist,
            }, index=tidx)
        raise NotImplementedError(f"tulipy backend: '{indicator_name}' not implemented")
