"""MA Cross indicator — SMA crossover state (1=golden, -1=death)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import register
from ..schema import Column


@register.column(
    "ma_cross",
    outputs=["ma_cross_{fast}_{slow}"],
    source="builtin",
)
class MACross:
    """SMA crossover state indicator.

    Returns 1 when SMA(fast) >= SMA(slow), -1 otherwise.
    NaN when either SMA has not warmed up yet.

    Dependencies: SMA(fast), SMA(slow) — resolved by DAG.
    """

    def requires(self, fast: int = 20, slow: int = 60) -> list[Column]:
        return [
            Column(f"sma_{fast}", dtype="float64", indicator="sma", params={"period": fast}),
            Column(f"sma_{slow}", dtype="float64", indicator="sma", params={"period": slow}),
        ]

    def compute(
        self,
        fast: int = 20,
        slow: int = 60,
        **inputs: pd.Series,
    ) -> pd.Series:
        sma_fast = inputs[f"sma_{fast}"]
        sma_slow = inputs[f"sma_{slow}"]

        result = np.where(sma_fast >= sma_slow, 1, -1).astype(float)
        # Preserve NaN where either SMA is NaN (warmup period)
        nan_mask = sma_fast.isna() | sma_slow.isna()
        result[nan_mask] = np.nan

        return pd.Series(result, index=sma_fast.index, name=f"ma_cross_{fast}_{slow}")
