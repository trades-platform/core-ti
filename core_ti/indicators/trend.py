"""Trend indicators: MACD (with dynamic DAG dependencies)."""
from __future__ import annotations

import pandas as pd

from ..registry import register
from ..schema import Column


@register.column(
    "macd",
    outputs=["macd_{fast}_{slow}_line", "macd_{fast}_{slow}_signal", "macd_{fast}_{slow}_hist"],
    source="builtin",
)
class MACD:
    """MACD — Moving Average Convergence Divergence.

    Uses EMA dependencies resolved dynamically via DAG.
    Outputs: macd_{fast}_{slow}_line, macd_{fast}_{slow}_signal, macd_{fast}_{slow}_hist
    """

    def requires(self, fast: int = 12, slow: int = 26, signal: int = 9) -> list[Column]:
        return [
            Column(f"ema_{fast}", dtype="float64", indicator="ema", params={"period": fast}),
            Column(f"ema_{slow}", dtype="float64", indicator="ema", params={"period": slow}),
        ]

    def compute(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        **inputs: pd.Series,
    ) -> pd.DataFrame:
        ema_fast = inputs[f"ema_{fast}"]
        ema_slow = inputs[f"ema_{slow}"]
        line = ema_fast - ema_slow
        sig = line.ewm(span=signal, adjust=False).mean()
        hist = line - sig
        return pd.DataFrame({
            f"macd_{fast}_{slow}_line": line,
            f"macd_{fast}_{slow}_signal": sig,
            f"macd_{fast}_{slow}_hist": hist,
        })
