"""Benchmark: SMA across backends."""
import pytest
from conftest import BACKENDS, bench_indicator

pytestmark = pytest.mark.benchmark(group="sma")


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_bench_sma(benchmark, backend, ohlcv):
    bench_indicator(benchmark, backend, "sma", ohlcv, {"period": 20})
