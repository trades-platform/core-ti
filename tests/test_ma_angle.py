"""Unit tests for ma_angle indicator."""
import math

import numpy as np
import pandas as pd
import pytest

from core_ti.indicators.ma_angle import MaAngle


def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = rng.standard_normal(n).cumsum() + 100
    return pd.DataFrame({
        "open": close + rng.standard_normal(n) * 0.5,
        "high": close + abs(rng.standard_normal(n)),
        "low": close - abs(rng.standard_normal(n)),
        "close": close,
    })


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class TestMaAngleCompute:
    def setup_method(self):
        self.df = _make_ohlcv(200)
        self.indicator = MaAngle()

    def test_default_params_output_name(self):
        ma_period, atr_period, scale = 20, 14, 2.5
        sma = _sma(self.df["close"], ma_period)
        atr = _atr(self.df["high"], self.df["low"], self.df["close"], atr_period)
        result = self.indicator.compute(
            ma_period=ma_period, atr_period=atr_period, scale=scale,
            **{f"sma_{ma_period}": sma, f"atr_{atr_period}": atr},
        )
        assert result.name == "ma_angle_20"

    def test_custom_params_output_name(self):
        result = self.indicator.compute(
            ma_period=10, atr_period=7, scale=3.0,
            **{"sma_10": _sma(self.df["close"], 10),
               "atr_7": _atr(self.df["high"], self.df["low"], self.df["close"], 7)},
        )
        assert result.name == "ma_angle_10"

    def test_angle_sign_declining(self):
        close = pd.Series(np.linspace(100, 80, 100))
        high = close + 1
        low = close - 1
        sma = _sma(close, 20)
        atr = _atr(high, low, close, 14)
        result = self.indicator.compute(
            ma_period=20, atr_period=14, scale=2.5,
            **{"sma_20": sma, "atr_14": atr},
        )
        valid = result.dropna()
        assert (valid < 0).all(), "Declining data should produce negative angles"

    def test_angle_sign_rising(self):
        close = pd.Series(np.linspace(80, 100, 100))
        high = close + 1
        low = close - 1
        sma = _sma(close, 20)
        atr = _atr(high, low, close, 14)
        result = self.indicator.compute(
            ma_period=20, atr_period=14, scale=2.5,
            **{"sma_20": sma, "atr_14": atr},
        )
        valid = result.dropna()
        assert (valid > 0).all(), "Rising data should produce positive angles"

    def test_zero_atr_returns_nan(self):
        sma = pd.Series([100.0] * 50)
        atr = pd.Series([0.0] * 50)
        result = self.indicator.compute(
            ma_period=20, atr_period=14, scale=2.5,
            **{"sma_20": sma, "atr_14": atr},
        )
        assert result.iloc[-1] != result.iloc[-1]  # NaN check

    def test_warmup_bars_are_nan(self):
        sma = _sma(self.df["close"], 20)
        atr = _atr(self.df["high"], self.df["low"], self.df["close"], 14)
        result = self.indicator.compute(
            ma_period=20, atr_period=14, scale=2.5,
            **{"sma_20": sma, "atr_14": atr},
        )
        assert result.iloc[:20].isna().all()
        assert not result.iloc[20] != result.iloc[20]  # bar 20 should be valid

    def test_scale_effect(self):
        close = pd.Series(np.linspace(100, 95, 50))
        high = close + 1
        low = close - 1
        sma = _sma(close, 20)
        atr = _atr(high, low, close, 14)
        inputs = {"sma_20": sma, "atr_14": atr}
        r2 = self.indicator.compute(ma_period=20, atr_period=14, scale=2, **inputs)
        r5 = self.indicator.compute(ma_period=20, atr_period=14, scale=5, **inputs)
        valid_idx = r2.dropna().index.intersection(r5.dropna().index)
        assert (r5.loc[valid_idx].abs() > r2.loc[valid_idx].abs()).all()


class TestMaAngleEngine:
    """End-to-end test: DAG resolves SMA and ATR dependencies automatically."""

    def test_engine_ma_angle_dag_resolution(self):
        from core_ti import IndicatorEngine

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
        result = engine.ma_angle(df, ma_period=20, atr_period=14, scale=2.5)

        assert "ma_angle_20" in result.columns
        assert "sma_20" in result.columns
        assert "atr_14" in result.columns
        valid = result["ma_angle_20"].dropna()
        assert len(valid) > 0, "Should have non-NaN values after warmup"
