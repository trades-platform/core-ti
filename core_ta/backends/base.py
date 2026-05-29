"""
backends/base.py — BackendProtocol definition.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class BackendProtocol(Protocol):
    """Protocol that all backend implementations must satisfy."""

    @property
    def name(self) -> str:
        """Unique backend identifier (e.g. 'pandas', 'talib')."""
        ...

    def supports(self, indicator_name: str) -> bool:
        """Return True if this backend has an implementation for indicator_name."""
        ...

    def compute(
        self,
        indicator_name: str,
        inputs: dict[str, pd.Series],
        params: dict[str, Any],
    ) -> pd.Series | pd.DataFrame | float:
        """Execute a single indicator computation.

        Parameters
        ----------
        indicator_name:
            Registered indicator name.
        inputs:
            Mapping of column name → Series (already extracted from df).
        params:
            Resolved parameter values (no _output key).

        Returns
        -------
        pd.Series | pd.DataFrame | float
        """
        ...


def nan_safe_compute(
    backend: BackendProtocol,
    indicator_name: str,
    inputs: dict[str, pd.Series],
    params: dict[str, Any],
    expected_outputs: list[str],
    df_length: int,
) -> pd.Series | pd.DataFrame | float:
    """Wrapper that catches data-insufficiency exceptions and fills NaN.

    Backend implementations do NOT need to handle insufficient data internally.
    This wrapper is responsible for that concern.

    Computation errors that are NOT data-insufficiency (TypeError, KeyError, …)
    are re-raised without modification.
    """
    try:
        return backend.compute(indicator_name, inputs, params)
    except (ValueError, ArithmeticError) as exc:
        # Treat these as potential data-insufficiency errors from backends
        # that don't natively return NaN (e.g. TA-Lib raises ValueError)
        index = next(iter(inputs.values())).index if inputs else range(df_length)
        if len(expected_outputs) == 1 or not expected_outputs:
            return pd.Series(np.nan, index=index, dtype="float64")
        return pd.DataFrame(
            {col: np.nan for col in expected_outputs},
            index=index,
            dtype="float64",
        )
    # TypeError, KeyError, AttributeError etc. propagate — they are bugs, not data issues
