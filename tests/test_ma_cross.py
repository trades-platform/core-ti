"""Unit tests for ma_cross indicator."""
import numpy as np
import pandas as pd
import pytest

from core_ta.indicators.ma_cross import MACross


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


class TestMACrossCompute:
    def setup_method(self):
        self.indicator = MACross()

    def test_golden_cross_when_fast_above_slow(self):
        sma_fast = pd.Series([1.33, 1.34, 1.35])
        sma_slow = pd.Series([1.32, 1.32, 1.32])
        result = self.indicator.compute(fast=20, slow=60, sma_20=sma_fast, sma_60=sma_slow)
        assert (result == 1).all()

    def test_death_cross_when_fast_below_slow(self):
        sma_fast = pd.Series([1.30, 1.29, 1.28])
        sma_slow = pd.Series([1.32, 1.32, 1.32])
        result = self.indicator.compute(fast=20, slow=60, sma_20=sma_fast, sma_60=sma_slow)
        assert (result == -1).all()

    def test_equal_values_are_golden_cross(self):
        sma_fast = pd.Series([1.32, 1.32])
        sma_slow = pd.Series([1.32, 1.32])
        result = self.indicator.compute(fast=20, slow=60, sma_20=sma_fast, sma_60=sma_slow)
        assert (result == 1).all()

    def test_nan_warmup(self):
        sma_fast = pd.Series([np.nan, np.nan, 1.35])
        sma_slow = pd.Series([np.nan, np.nan, 1.32])
        result = self.indicator.compute(fast=20, slow=60, sma_20=sma_fast, sma_60=sma_slow)
        assert result.iloc[:2].isna().all()
        assert result.iloc[2] == 1

    def test_nan_when_only_one_sma_is_nan(self):
        sma_fast = pd.Series([1.35, np.nan, 1.35])
        sma_slow = pd.Series([np.nan, 1.32, 1.32])
        result = self.indicator.compute(fast=20, slow=60, sma_20=sma_fast, sma_60=sma_slow)
        assert result.iloc[0] != result.iloc[0]  # NaN
        assert result.iloc[1] != result.iloc[1]  # NaN
        assert result.iloc[2] == 1

    def test_output_name(self):
        sma_fast = pd.Series([1.35])
        sma_slow = pd.Series([1.32])
        result = self.indicator.compute(fast=20, slow=60, sma_20=sma_fast, sma_60=sma_slow)
        assert result.name == "ma_cross_20_60"

    def test_custom_periods_output_name(self):
        sma_fast = pd.Series([1.35])
        sma_slow = pd.Series([1.32])
        result = self.indicator.compute(fast=10, slow=30, sma_10=sma_fast, sma_30=sma_slow)
        assert result.name == "ma_cross_10_30"

    def test_crossing_transition(self):
        # fast goes from above to below to above
        sma_fast = pd.Series([1.35, 1.33, 1.30, 1.28, 1.33, 1.35])
        sma_slow = pd.Series([1.32, 1.32, 1.32, 1.32, 1.32, 1.32])
        result = self.indicator.compute(fast=20, slow=60, sma_20=sma_fast, sma_60=sma_slow)
        expected = [1, 1, -1, -1, 1, 1]
        assert result.tolist() == expected


class TestMACrossEngine:
    """End-to-end test: DAG resolves SMA dependencies automatically."""

    def test_engine_ma_cross_dag_resolution(self):
        from core_ta import IndicatorEngine

        rng = np.random.default_rng(42)
        n = 200
        close = rng.standard_normal(n).cumsum() + 100
        df = pd.DataFrame({
            "open": close + rng.standard_normal(n) * 0.5,
            "high": close + abs(rng.standard_normal(n)),
            "low": close - abs(rng.standard_normal(n)),
            "close": close,
        })

        engine = IndicatorEngine(backend="pandas")
        result = engine.ma_cross(df, fast=20, slow=60)

        assert "ma_cross_20_60" in result.columns
        assert "sma_20" in result.columns
        assert "sma_60" in result.columns
        valid = result["ma_cross_20_60"].dropna()
        assert len(valid) > 0
        assert set(valid.unique()) <= {1.0, -1.0}

    def test_engine_custom_periods(self):
        from core_ta import IndicatorEngine

        rng = np.random.default_rng(42)
        n = 200
        close = rng.standard_normal(n).cumsum() + 100
        df = pd.DataFrame({
            "open": close + rng.standard_normal(n) * 0.5,
            "high": close + abs(rng.standard_normal(n)),
            "low": close - abs(rng.standard_normal(n)),
            "close": close,
        })

        engine = IndicatorEngine(backend="pandas")
        result = engine.ma_cross(df, fast=10, slow=30)

        assert "ma_cross_10_30" in result.columns
        assert "sma_10" in result.columns
        assert "sma_30" in result.columns
