"""Rolling-window helpers backed by bottleneck, preserving pandas
``Series.rolling()`` parity.

Bottleneck's ``move_*`` functions are much faster than pandas rolling
aggregation, but diverge on a few edge cases. These helpers guard them so the
output matches ``rolling()`` point-wise:

- **Non-positive / oversized windows**: ``rolling`` returns all-NaN, whereas
  bottleneck raises. Guarded to all-NaN.
- **Nullable float dtypes** (e.g. ``Float64`` with ``pd.NA``): plain
  ``to_numpy(dtype=float)`` raises; ``na_value=np.nan`` converts cleanly.
- **Non-finite (±inf) inputs**: bottleneck's running-sum state is corrupted once
  an inf leaves the window (``inf - inf = NaN``), poisoning later windows. We
  detect inf and fall back to ``rolling`` for exact parity. NaN inputs are safe
  under bottleneck and do not trigger the fallback.
"""
from __future__ import annotations

import bottleneck as bn
import numpy as np
import pandas as pd


def _prepare(close: pd.Series) -> np.ndarray:
    return close.to_numpy(dtype=float, na_value=np.nan)


def roll_mean(close: pd.Series, period: int) -> pd.Series:
    """Rolling mean matching ``close.rolling(period).mean()``."""
    x = _prepare(close)
    n = x.shape[0]
    if period < 1 or n < period:
        out = np.full(n, np.nan)
    elif np.isinf(x).any():
        return pd.Series(x, index=close.index, name=close.name).rolling(period).mean()
    else:
        out = bn.move_mean(x, period, min_count=period)
    return pd.Series(out, index=close.index, name=close.name)


def roll_std(close: pd.Series, period: int) -> pd.Series:
    """Rolling sample std (ddof=1) matching ``close.rolling(period).std()``."""
    x = _prepare(close)
    n = x.shape[0]
    if period < 1 or n < period:
        out = np.full(n, np.nan)
    elif np.isinf(x).any():
        return pd.Series(x, index=close.index, name=close.name).rolling(period).std()
    else:
        out = bn.move_std(x, period, min_count=period, ddof=1)
    return pd.Series(out, index=close.index, name=close.name)
