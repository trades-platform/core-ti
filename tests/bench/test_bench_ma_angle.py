"""Benchmark: MA Angle via IndicatorEngine (end-to-end DAG resolution)."""
import pytest

from core_ti import IndicatorEngine

pytestmark = pytest.mark.benchmark(group="ma_angle")


@pytest.mark.parametrize("backend", ["pandas", "talib", "tulipy"], ids=["pandas", "talib", "tulipy"])
def test_bench_ma_angle(benchmark, backend, ohlcv):
    engine = IndicatorEngine(backend=backend)
    benchmark(engine.ma_angle, ohlcv, ma_period=20, atr_period=14, scale=2.5)
