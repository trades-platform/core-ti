"""Benchmark: EMA across backends."""
import pytest
from conftest import BACKENDS, bench_indicator

pytestmark = pytest.mark.benchmark(group="ema")


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_bench_ema(benchmark, backend, ohlcv):
    bench_indicator(benchmark, backend, "ema", ohlcv, {"period": 20})
