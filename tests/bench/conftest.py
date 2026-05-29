"""Benchmark fixtures for cross-backend indicator comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


SIZES = [1_000, 10_000, 100_000]
SIZE_IDS = ["1K", "10K", "100K"]

BACKENDS = ["pandas", "talib", "pandas_ta", "tulipy"]


def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = rng.standard_normal(n).cumsum() + 100
    return pd.DataFrame({
        "open": close + rng.standard_normal(n) * 0.5,
        "high": close + abs(rng.standard_normal(n)),
        "low": close - abs(rng.standard_normal(n)),
        "close": close,
        "volume": rng.uniform(1e5, 1e7, n),
    })


@pytest.fixture(params=SIZES, ids=SIZE_IDS)
def ohlcv(request) -> pd.DataFrame:
    return _make_ohlcv(request.param)


def get_backend(name: str):
    if name == "pandas":
        from core_ta.backends.pandas_backend import PandasBackend
        return PandasBackend()
    if name == "talib":
        try:
            import talib  # noqa: F401
        except ImportError:
            pytest.skip("TA-Lib not installed")
        from core_ta.backends.talib_backend import TALibBackend
        return TALibBackend()
    if name == "pandas_ta":
        try:
            import pandas_ta  # noqa: F401
        except ImportError:
            pytest.skip("pandas-ta not installed")
        from core_ta.backends.pandas_ta_backend import PandasTABackend
        return PandasTABackend()
    if name == "tulipy":
        try:
            import tulipy  # noqa: F401
        except ImportError:
            pytest.skip("tulipy not installed")
        from core_ta.backends.tulipy_backend import TulipyBackend
        return TulipyBackend()
    raise ValueError(f"Unknown backend: {name}")


def _prepare_inputs(backend_name: str, indicator: str, inputs: dict, params: dict):
    if indicator == "macd" and backend_name == "pandas":
        from core_ta.backends.pandas_backend import PandasBackend
        pd_be = PandasBackend()
        inputs[f"ema_{params['fast']}"] = pd_be.compute("ema", inputs, {"period": params["fast"]})
        inputs[f"ema_{params['slow']}"] = pd_be.compute("ema", inputs, {"period": params["slow"]})


def bench_indicator(benchmark, backend_name: str, indicator: str, df: pd.DataFrame, params: dict):
    backend = get_backend(backend_name)
    inputs = {col: df[col] for col in df.columns}

    _prepare_inputs(backend_name, indicator, inputs, params)

    try:
        backend.compute(indicator, inputs, params)
    except NotImplementedError:
        pytest.skip(f"{backend_name} does not implement {indicator}")

    result = benchmark(backend.compute, indicator, inputs, params)
    return result
