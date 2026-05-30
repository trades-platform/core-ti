"""MA Angle indicator — ATR-normalized moving average slope in degrees."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import register
from ..schema import Column


@register.column(
    "ma_angle",
    outputs=["ma_angle_{ma_period}"],
    source="builtin",
)
class MaAngle:
    """ATR-normalized MA angle indicator.

    Computes the angle of the SMA one-bar delta, normalized by ATR,
    mapped to degrees via arctan with a configurable scale factor.

    Dependencies: SMA(ma_period), ATR(atr_period) — resolved by DAG.
    """

    def requires(self, ma_period: int = 20, atr_period: int = 14, scale: float = 2.5) -> list[Column]:
        return [
            Column(f"sma_{ma_period}", dtype="float64", indicator="sma", params={"period": ma_period}),
            Column(f"atr_{atr_period}", dtype="float64", indicator="atr", params={"period": atr_period}),
        ]

    def compute(
        self,
        ma_period: int = 20,
        atr_period: int = 14,
        scale: float = 2.5,
        **inputs: pd.Series,
    ) -> pd.Series:
        sma = inputs[f"sma_{ma_period}"]
        atr = inputs[f"atr_{atr_period}"]

        sma_delta = sma.diff()
        with np.errstate(divide="ignore", invalid="ignore"):
            atr_norm = sma_delta / atr * scale
        angle = np.degrees(np.arctan(atr_norm.values))
        return pd.Series(angle, index=sma.index, name=f"ma_angle_{ma_period}")
