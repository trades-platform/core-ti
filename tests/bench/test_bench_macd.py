"""Benchmark: MACD across backends."""
import pytest
from conftest import BACKENDS, bench_indicator

pytestmark = pytest.mark.benchmark(group="macd")


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_bench_macd(benchmark, backend, ohlcv):
    bench_indicator(benchmark, backend, "macd", ohlcv, {"fast": 12, "slow": 26, "signal": 9})
