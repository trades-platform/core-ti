"""Benchmark: WMA across backends."""
import pytest
from conftest import BACKENDS, bench_indicator

pytestmark = pytest.mark.benchmark(group="wma")


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_bench_wma(benchmark, backend, ohlcv):
    bench_indicator(benchmark, backend, "wma", ohlcv, {"period": 20})
