"""
backends/pandas_ta_backend.py — pandas-ta backend stub.

Only active when the `pandas-ta` package is installed.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


class PandasTABackend:
    """pandas-ta backend. Requires: pip install pandas-ta"""

    def __init__(self) -> None:
        try:
            import pandas_ta as _pta  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

    @property
    def name(self) -> str:
        return "pandas_ta"

    def supports(self, indicator_name: str) -> bool:
        if not self._available:
            return False
        return indicator_name in {"sma", "ema", "wma", "rsi", "bb", "atr", "macd"}

    def compute(
        self,
        indicator_name: str,
        inputs: dict[str, pd.Series],
        params: dict[str, Any],
    ) -> pd.Series | pd.DataFrame | float:
        import pandas_ta as pta

        if indicator_name == "sma":
            result = pta.sma(inputs["close"], length=params["period"])
            return result

        if indicator_name == "ema":
            result = pta.ema(inputs["close"], length=params["period"])
            return result

        if indicator_name == "rsi":
            result = pta.rsi(inputs["close"], length=params["period"])
            return result

        raise NotImplementedError(
            f"pandas_ta backend: '{indicator_name}' not implemented"
        )
