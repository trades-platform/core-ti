"""Benchmark: ATR across backends."""
import pytest
from conftest import BACKENDS, bench_indicator

pytestmark = pytest.mark.benchmark(group="atr")


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_bench_atr(benchmark, backend, ohlcv):
    bench_indicator(benchmark, backend, "atr", ohlcv, {"period": 14})
