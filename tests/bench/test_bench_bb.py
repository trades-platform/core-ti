"""Benchmark: Bollinger Bands across backends."""
import pytest
from conftest import BACKENDS, bench_indicator

pytestmark = pytest.mark.benchmark(group="bb")


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_bench_bb(benchmark, backend, ohlcv):
    bench_indicator(benchmark, backend, "bb", ohlcv, {"period": 20, "std": 2.0})
