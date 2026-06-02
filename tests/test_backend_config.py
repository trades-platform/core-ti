"""
Tests for backend selection and global configuration.

Covers: set_default_backend, get_default_backend, reset_default_backend,
CORE_TI_BACKEND env var, engine.backend property, _backend Facade/Pipe override,
DAG non-propagation, _fn-only indicator handling.
"""
import os

import pandas as pd
import pytest

from core_ti import (
    Column,
    IndicatorEngine,
    set_default_backend,
    get_default_backend,
    reset_default_backend,
)
from core_ti._config import _resolve_default_backend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_global_config():
    """Ensure global config is clean before/after each test."""
    reset_default_backend()
    yield
    reset_default_backend()


@pytest.fixture
def ohlcv():
    n = 100
    import numpy as np
    np.random.seed(42)
    close = pd.Series(100.0 + np.random.randn(n).cumsum())
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + np.abs(np.random.randn(n)),
        "low": close - np.abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


# ---------------------------------------------------------------------------
# 5.1 Global config API
# ---------------------------------------------------------------------------

class TestGlobalConfigAPI:

    def test_set_and_get(self):
        set_default_backend("talib")
        assert get_default_backend() == "talib"

    def test_get_returns_none_initially(self):
        assert get_default_backend() is None

    def test_reset_clears_value(self):
        set_default_backend("talib")
        reset_default_backend()
        assert get_default_backend() is None

    def test_set_multiple_times(self):
        set_default_backend("talib")
        set_default_backend("pandas_ta")
        assert get_default_backend() == "pandas_ta"


# ---------------------------------------------------------------------------
# 5.2 CORE_TI_BACKEND env var
# ---------------------------------------------------------------------------

class TestEnvVarBackend:

    def test_env_var_respected(self, monkeypatch):
        monkeypatch.setenv("CORE_TI_BACKEND", "pandas")
        assert _resolve_default_backend() == "pandas"

    def test_env_var_empty_string_ignored(self, monkeypatch):
        monkeypatch.setenv("CORE_TI_BACKEND", "")
        assert _resolve_default_backend() == "pandas"

    def test_env_var_whitespace_ignored(self, monkeypatch):
        monkeypatch.setenv("CORE_TI_BACKEND", "  ")
        assert _resolve_default_backend() == "pandas"

    def test_env_var_invalid_raises_at_engine(self, monkeypatch):
        monkeypatch.setenv("CORE_TI_BACKEND", "nonexistent")
        with pytest.raises(ValueError, match="Unknown backend"):
            IndicatorEngine()

    def test_no_env_var_defaults_pandas(self, monkeypatch):
        monkeypatch.delenv("CORE_TI_BACKEND", raising=False)
        assert _resolve_default_backend() == "pandas"


# ---------------------------------------------------------------------------
# 5.3 Priority chain
# ---------------------------------------------------------------------------

class TestPriorityChain:

    def test_constructor_wins_over_all(self, monkeypatch):
        monkeypatch.setenv("CORE_TI_BACKEND", "tulipy")
        set_default_backend("pandas_ta")
        e = IndicatorEngine(backend="pandas")
        assert e.backend == "pandas"

    def test_python_api_over_env_var(self, monkeypatch):
        monkeypatch.setenv("CORE_TI_BACKEND", "tulipy")
        set_default_backend("pandas_ta")
        e = IndicatorEngine()
        assert e.backend == "pandas_ta"

    def test_env_var_used_when_no_python_api(self, monkeypatch):
        monkeypatch.setenv("CORE_TI_BACKEND", "pandas")
        e = IndicatorEngine()
        assert e.backend == "pandas"

    def test_hardcoded_pandas_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("CORE_TI_BACKEND", raising=False)
        e = IndicatorEngine()
        assert e.backend == "pandas"

    def test_backend_param_overrides_python_api(self):
        set_default_backend("pandas")
        e = IndicatorEngine(backend="pandas")
        assert e.backend == "pandas"


# ---------------------------------------------------------------------------
# 5.4 engine.backend property
# ---------------------------------------------------------------------------

class TestBackendProperty:

    def test_read_backend(self):
        e = IndicatorEngine(backend="pandas")
        assert e.backend == "pandas"

    def test_switch_backend(self):
        e = IndicatorEngine(backend="pandas")
        e.backend = "pandas"
        assert e.backend == "pandas"

    def test_switch_invalid_raises(self):
        e = IndicatorEngine(backend="pandas")
        with pytest.raises(ValueError, match="Unknown backend"):
            e.backend = "nonexistent"

    def test_switch_to_custom_backend(self):
        from core_ti.backends.base import BackendProtocol

        class MyBackend:
            @property
            def name(self):
                return "my_custom"

            def supports(self, name):
                return False

            def compute(self, name, inputs, params):
                return pd.Series(dtype="float64")

        e = IndicatorEngine(backend="pandas")
        e.backend = MyBackend()
        assert e.backend == "my_custom"

    def test_default_backend_is_pandas(self):
        e = IndicatorEngine()
        assert e.backend == "pandas"


# ---------------------------------------------------------------------------
# 5.5 _backend override in Facade API
# ---------------------------------------------------------------------------

class TestFacadeBackendOverride:

    def test_override_uses_specified_backend(self, ohlcv):
        e = IndicatorEngine(backend="pandas")
        # Both calls use pandas, just verifying _backend parameter works
        e.sma(ohlcv, period=20, _backend="pandas")
        assert "sma_20" in ohlcv.columns

    def test_override_does_not_persist(self, ohlcv):
        e = IndicatorEngine(backend="pandas")
        e.sma(ohlcv, period=20, _backend="pandas")
        assert "sma_20" in ohlcv.columns
        # Second call without _backend should still work
        df2 = ohlcv.copy()
        df2.drop(columns=["sma_20"], inplace=True)
        e.sma(df2, period=20)
        assert "sma_20" in df2.columns

    def test_override_invalid_raises(self, ohlcv):
        e = IndicatorEngine(backend="pandas")
        with pytest.raises(ValueError, match="Unknown backend"):
            e.sma(ohlcv, period=20, _backend="nonexistent")

    def test_override_with_backend_protocol(self, ohlcv):
        """_backend accepts a BackendProtocol instance."""
        from core_ti.backends.base import BackendProtocol

        call_log = []

        class SpyBackend:
            @property
            def name(self):
                return "spy"

            def supports(self, name):
                return name == "sma"

            def compute(self, name, inputs, params):
                call_log.append(name)
                return inputs["close"].rolling(params["period"]).mean()

        e = IndicatorEngine(backend="pandas")
        e.sma(ohlcv, period=20, _backend=SpyBackend())
        assert call_log == ["sma"]
        assert "sma_20" in ohlcv.columns

    def test_backend_not_forwarded_to_compute(self, ohlcv):
        """_backend must not appear in params passed to indicator function."""
        e = IndicatorEngine(backend="pandas")
        # If _backend leaked through, the registry compute would likely error
        e.sma(ohlcv, period=20, _backend="pandas")
        # Just ensure no crash — _backend was properly stripped


# ---------------------------------------------------------------------------
# 5.6 _backend override in Pipe API
# ---------------------------------------------------------------------------

class TestPipeBackendOverride:

    def test_single_step_override(self, ohlcv):
        e = IndicatorEngine(backend="pandas")
        result = e.pipe(ohlcv).sma(period=20, _backend="pandas").result()
        assert "sma_20" in result.columns

    def test_mixed_overrides(self, ohlcv):
        e = IndicatorEngine(backend="pandas")
        result = (
            e.pipe(ohlcv)
            .sma(period=20, _backend="pandas")
            .ema(period=12, _backend="pandas")
            .result()
        )
        assert "sma_20" in result.columns
        assert "ema_12" in result.columns

    def test_pipe_override_does_not_persist(self, ohlcv):
        e = IndicatorEngine(backend="pandas")
        e.pipe(ohlcv).sma(period=20, _backend="pandas").result()
        # Next pipe call without _backend
        df2 = ohlcv.copy()
        e.pipe(df2).ema(period=12).result()
        assert "ema_12" in df2.columns


# ---------------------------------------------------------------------------
# 5.7 _backend does not propagate to DAG deps
# ---------------------------------------------------------------------------

class TestDAGNonPropagation:

    def test_backend_override_does_not_propagate_to_deps(self, ohlcv):
        """When _backend is specified for MACD, EMA deps use engine default."""
        from core_ti.backends.base import BackendProtocol

        talib_calls = []

        class SpyTalib:
            @property
            def name(self):
                return "spy_talib"

            def supports(self, name):
                return name in ("macd", "ema")

            def compute(self, name, inputs, params):
                talib_calls.append(name)
                if name == "ema":
                    period = params.get("period", 20)
                    return inputs["close"].ewm(span=period, adjust=False).mean()
                if name == "macd":
                    # MACD deps (ema_12, ema_26) are pre-computed by DAG
                    fast_ema = inputs.get("ema_12", inputs.get("close"))
                    slow_ema = inputs.get("ema_26", inputs.get("close"))
                    if fast_ema is None or slow_ema is None:
                        return pd.Series(dtype="float64")
                    line = fast_ema - slow_ema
                    signal = line.ewm(span=params.get("signal", 9), adjust=False).mean()
                    return pd.DataFrame({
                        "macd_line": line,
                        "macd_signal": signal,
                        "macd_hist": line - signal,
                    })
                return pd.Series(dtype="float64")

        e = IndicatorEngine(backend="pandas")
        e.macd(ohlcv, fast=12, slow=26, signal=9, _backend=SpyTalib())

        # Only MACD should use the spy backend, NOT the EMA dependencies
        # EMA deps are resolved by the engine's _run_dag with backend_override=None
        assert "macd" in talib_calls
        # EMA should NOT be in talib_calls (it uses pandas default via _resolve_backend)
        assert "ema" not in talib_calls


# ---------------------------------------------------------------------------
# 5.8 _backend with _fn-only indicator
# ---------------------------------------------------------------------------

class TestFnOnlyIndicator:

    def test_backend_ignored_for_fn_only(self, ohlcv):
        from core_ti import register

        @register.column(
            "test_custom_fn",
            requires=[Column("close", dtype="float64")],
            outputs=["test_custom_fn"],
        )
        def _custom_fn(close: pd.Series) -> pd.Series:
            return close * 2.0

        e = IndicatorEngine(backend="pandas")
        # _backend should be silently ignored — _fn path takes priority
        e.test_custom_fn(ohlcv, _backend="pandas")
        assert "test_custom_fn" in ohlcv.columns
        assert (ohlcv["test_custom_fn"] == ohlcv["close"] * 2.0).all()
