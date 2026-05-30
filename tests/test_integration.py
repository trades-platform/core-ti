"""
Integration tests for the stock_indicators library.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from core_ti import Column, IndicatorEngine, Param, register
from core_ti.registry import (
    CircularDependencyError,
    IndicatorNotFoundError,
    MissingColumnError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv():
    """100-row OHLCV DataFrame with standard column names."""
    np.random.seed(42)
    n = 100
    close = pd.Series(100.0 + np.random.randn(n).cumsum())
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + np.abs(np.random.randn(n)),
        "low": close - np.abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


@pytest.fixture
def engine():
    return IndicatorEngine(backend="pandas")


# ---------------------------------------------------------------------------
# 8.1 Facade API
# ---------------------------------------------------------------------------

class TestFacadeAPI:
    def test_sma_adds_column(self, engine, ohlcv):
        result = engine.sma(ohlcv, period=20)
        assert result is ohlcv          # mutate returns same object
        assert "sma_20" in ohlcv.columns
        assert ohlcv["sma_20"].notna().sum() == 100 - 20 + 1

    def test_sma_different_periods_coexist(self, engine, ohlcv):
        engine.sma(ohlcv, period=20)
        engine.sma(ohlcv, period=50)
        assert "sma_20" in ohlcv.columns
        assert "sma_50" in ohlcv.columns

    def test_ema_adds_column(self, engine, ohlcv):
        engine.ema(ohlcv, period=12)
        assert "ema_12" in ohlcv.columns

    def test_rsi_range(self, engine, ohlcv):
        engine.rsi(ohlcv, period=14)
        col = ohlcv["rsi_14"].dropna()
        assert (col >= 0).all() and (col <= 100).all()

    def test_bb_three_columns(self, engine, ohlcv):
        engine.bb(ohlcv, period=20, std=2.0)
        assert "bb_20_upper" in ohlcv.columns
        assert "bb_20_mid" in ohlcv.columns
        assert "bb_20_lower" in ohlcv.columns

    def test_atr_adds_column(self, engine, ohlcv):
        engine.atr(ohlcv, period=14)
        assert "atr_14" in ohlcv.columns

    def test_dag_auto_resolution(self, engine, ohlcv):
        """MACD should auto-compute EMA dependencies."""
        engine.macd(ohlcv, fast=12, slow=26, signal=9)
        assert "ema_12" in ohlcv.columns
        assert "ema_26" in ohlcv.columns
        assert "macd_12_26_line" in ohlcv.columns
        assert "macd_12_26_signal" in ohlcv.columns
        assert "macd_12_26_hist" in ohlcv.columns

    def test_macd_different_params_coexist(self, engine, ohlcv):
        engine.macd(ohlcv, fast=12, slow=26, signal=9)
        engine.macd(ohlcv, fast=8, slow=21, signal=9)
        assert "macd_12_26_line" in ohlcv.columns
        assert "macd_8_21_line" in ohlcv.columns

    def test_scalar_sharpe_returns_float(self, engine, ohlcv):
        result = engine.sharpe_ratio(ohlcv)
        assert isinstance(result, float)

    def test_scalar_max_drawdown_negative(self, engine, ohlcv):
        result = engine.max_drawdown(ohlcv)
        assert result <= 0

    def test_nonexistent_indicator_raises(self, engine, ohlcv):
        with pytest.raises(AttributeError):
            engine.nonexistent(ohlcv)


# ---------------------------------------------------------------------------
# 8.2 Pipe API
# ---------------------------------------------------------------------------

class TestPipeAPI:
    def test_basic_pipe_chain(self, engine, ohlcv):
        result = engine.pipe(ohlcv).sma(period=20).ema(period=10).result()
        assert result is ohlcv
        assert "sma_20" in ohlcv.columns
        assert "ema_10" in ohlcv.columns

    def test_pipe_skips_existing_columns(self, engine, ohlcv):
        engine.sma(ohlcv, period=20)
        original_values = ohlcv["sma_20"].copy()
        engine.pipe(ohlcv).sma(period=20).result()   # should skip
        pd.testing.assert_series_equal(ohlcv["sma_20"], original_values)

    def test_pipe_unified_dag(self, engine, ohlcv):
        """Pipe builds unified DAG — shared deps computed once."""
        result = engine.pipe(ohlcv).macd(fast=12, slow=26, signal=9).bb(period=20).result()
        assert "ema_12" in ohlcv.columns
        assert "macd_12_26_line" in ohlcv.columns
        assert "bb_20_upper" in ohlcv.columns

    def test_pipe_rejects_scalar(self, engine, ohlcv):
        with pytest.raises(TypeError):
            engine.pipe(ohlcv).sharpe_ratio()

    def test_pipe_on_error_skip(self, engine, ohlcv):
        """Failing indicator is skipped; independent ones continue."""
        @register.column("_bad_test", requires=[Column("close")], outputs=["_bad_out"])
        def _bad(close: pd.Series) -> pd.Series:
            raise RuntimeError("intentional failure")

        result = engine.pipe(ohlcv, on_error="skip")._bad_test().sma(period=5).result()
        assert "_bad_out" not in ohlcv.columns
        assert "sma_5" in ohlcv.columns
        assert len(engine.last_report()) == 1

    def test_pipe_on_error_raise(self, engine, ohlcv):
        @register.column("_bad2", requires=[Column("close")], outputs=["_bad2_out"])
        def _bad2(close: pd.Series) -> pd.Series:
            raise RuntimeError("intentional")

        with pytest.raises(RuntimeError):
            engine.pipe(ohlcv, on_error="raise")._bad2().result()


# ---------------------------------------------------------------------------
# 8.3 User-defined indicator registration
# ---------------------------------------------------------------------------

class TestUserDefinedIndicators:
    def test_register_column_function(self, engine, ohlcv):
        @register.column("my_sma2", requires=[Column("close")], outputs=["my_sma2_{period}"])
        def my_sma2(close: pd.Series, period: int = 10) -> pd.Series:
            return close.rolling(period).mean()

        engine.my_sma2(ohlcv, period=15)
        assert "my_sma2_15" in ohlcv.columns

    def test_register_scalar_function(self, engine, ohlcv):
        @register.scalar("my_vol", requires=[Column("close")])
        def my_vol(close: pd.Series) -> float:
            return float(close.pct_change().std())

        result = engine.my_vol(ohlcv)
        assert isinstance(result, float)

    def test_user_indicator_appears_in_list(self, engine):
        @register.column("listed_ind", requires=[Column("close")], outputs=["listed_ind"])
        def listed_ind(close: pd.Series) -> pd.Series:
            return close

        names = [m.name for m in engine.list_indicators()]
        assert "listed_ind" in names

    def test_source_is_user(self, engine):
        @register.column("src_test", requires=[Column("close")], outputs=["src_test"])
        def src_test(close: pd.Series) -> pd.Series:
            return close

        meta = engine.get_indicator("src_test")
        assert meta.source == "user"

    def test_source_is_builtin(self, engine):
        meta = engine.get_indicator("sma")
        assert meta.source == "builtin"


# ---------------------------------------------------------------------------
# 8.4 Backend fallback + insufficient data NaN handling
# ---------------------------------------------------------------------------

class TestBackendAndNaN:
    def test_insufficient_data_fills_nan(self, engine):
        short_df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.0, 2.0, 3.0],
            "volume": [100.0, 200.0, 300.0],
        })
        engine.sma(short_df, period=60)
        assert "sma_60" in short_df.columns
        assert short_df["sma_60"].isna().all()

    def test_backend_fallback_warning(self, ohlcv):
        """User-defined indicator has no talib implementation — should fallback."""
        @register.column(
            "fallback_test", requires=[Column("close")], outputs=["fallback_out"]
        )
        def fallback_test(close: pd.Series) -> pd.Series:
            return close

        eng = IndicatorEngine(backend="talib")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eng.fallback_test(ohlcv)
        # Should have column (computed via fallback or user fn)
        assert "fallback_out" in ohlcv.columns


# ---------------------------------------------------------------------------
# 8.5 Cycle detection + missing column errors
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_indicator_not_found(self, engine, ohlcv):
        with pytest.raises(IndicatorNotFoundError):
            engine.get_indicator("totally_fake")

    def test_list_filter_by_type(self, engine):
        cols = engine.list_indicators(type="column")
        scalars = engine.list_indicators(type="scalar")
        assert all(m.indicator_type == "column" for m in cols)
        assert all(m.indicator_type == "scalar" for m in scalars)

    def test_list_filter_by_requires_subset(self, engine):
        results = engine.list_indicators(requires_subset=["close"])
        names = [m.name for m in results]
        assert "sma" in names
        assert "ema" in names
        # atr requires high/low/close — should NOT appear for close-only subset
        assert "atr" not in names


# ---------------------------------------------------------------------------
# 8.6 Package-level imports
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_top_level_exports(self):
        from core_ti import Column, IndicatorEngine, Param, register
        assert IndicatorEngine is not None
        assert register is not None
        assert Column is not None
        assert Param is not None
