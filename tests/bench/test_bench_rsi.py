"""Benchmark: RSI across backends."""
import pytest
from conftest import BACKENDS, bench_indicator

pytestmark = pytest.mark.benchmark(group="rsi")


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_bench_rsi(benchmark, backend, ohlcv):
    bench_indicator(benchmark, backend, "rsi", ohlcv, {"period": 14})
