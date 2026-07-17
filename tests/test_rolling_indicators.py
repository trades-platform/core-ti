"""Parity tests: bottleneck-accelerated SMA/BB match the rolling reference.

These lock the correctness contract from the `accelerate-rolling-indicators`
change: the accelerated pandas implementations must equal `Series.rolling()`
semantics point-wise, including NaN placement, ddof=1, and short-series cases.
"""
import numpy as np
import pandas as pd
import pytest

from core_ti.backends.pandas_backend import _sma, _bb
from core_ti.indicators.moving_averages import sma as ind_sma
from core_ti.indicators.volatility import bb as ind_bb


def _make_close(n: int, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n).cumsum() + 100)


def _ref_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def _ref_bb(close: pd.Series, period: int, std_mult: float) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return pd.DataFrame({
        f"bb_{period}_upper": mid + std_mult * s,
        f"bb_{period}_mid": mid,
        f"bb_{period}_lower": mid - std_mult * s,
    })


@pytest.mark.parametrize("period", [1, 2, 14, 20, 50])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 19, 20, 21, 100, 1000])
def test_sma_matches_rolling(period, n):
    close = _make_close(n)
    ref = _ref_sma(close, period).to_numpy()
    for got in (_sma({"close": close}, {"period": period}), ind_sma(close, period)):
        assert got.shape == ref.shape
        assert np.allclose(got.to_numpy(), ref, equal_nan=True)


@pytest.mark.parametrize("period", [1, 2, 14, 20, 50])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 19, 20, 21, 100, 1000])
def test_bb_matches_rolling(period, n):
    close = _make_close(n)
    ref = _ref_bb(close, period, 2.0)
    for got in (_bb({"close": close}, {"period": period, "std": 2.0}), ind_bb(close, period, 2.0)):
        assert list(got.columns) == list(ref.columns)
        assert np.allclose(got.to_numpy(), ref.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("period", [3, 14])
def test_sma_bb_match_rolling_with_nan_inputs(period):
    # copy=True: pandas_ta enables Copy-on-Write process-wide at import, under
    # which to_numpy() returns a read-only view that can't be mutated in place.
    x = _make_close(80).to_numpy(copy=True)
    x[[5, 6, 40, 41, 42]] = np.nan
    close = pd.Series(x)
    assert np.allclose(
        _sma({"close": close}, {"period": period}).to_numpy(),
        _ref_sma(close, period).to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(
        _bb({"close": close}, {"period": period, "std": 2.0}).to_numpy(),
        _ref_bb(close, period, 2.0).to_numpy(),
        equal_nan=True,
    )


def test_bb_uses_sample_std_ddof1():
    close = _make_close(60)
    got = _bb({"close": close}, {"period": 20, "std": 2.0})["bb_20_mid"]
    # mid equals rolling mean; std leg equals ddof=1 rolling std
    upper = _bb({"close": close}, {"period": 20, "std": 2.0})["bb_20_upper"]
    ref_std = close.rolling(20).std(ddof=1)
    assert np.allclose((upper - got).to_numpy(), (2.0 * ref_std).to_numpy(), equal_nan=True)


def test_short_series_all_nan():
    close = _make_close(5)
    out = _sma({"close": close}, {"period": 20})
    assert len(out) == 5 and out.isna().all()


def test_index_and_name_preserved():
    close = pd.Series(
        _make_close(50).to_numpy(), index=pd.RangeIndex(100, 150), name="close"
    )
    out = _sma({"close": close}, {"period": 10})
    assert out.index.equals(close.index)
    assert out.name == "close"


@pytest.mark.parametrize("period", [2, 3, 5])
def test_inf_inputs_match_rolling(period):
    # +/-inf must not poison later windows (bottleneck running-sum corruption):
    # parity with rolling is required.
    # copy=True for a writable array (see test_sma_bb_match_rolling_with_nan_inputs).
    base = _make_close(40).to_numpy(copy=True)
    base[10] = np.inf
    base[25] = -np.inf
    close = pd.Series(base)
    assert np.allclose(
        _sma({"close": close}, {"period": period}).to_numpy(),
        _ref_sma(close, period).to_numpy(),
        equal_nan=True,
    )
    got_bb = _bb({"close": close}, {"period": period, "std": 2.0})
    ref_bb = _ref_bb(close, period, 2.0)
    assert np.allclose(got_bb.to_numpy(), ref_bb.to_numpy(), equal_nan=True)


def test_inf_recovers_after_leaving_window():
    close = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0])
    got = _sma({"close": close}, {"period": 2}).to_numpy()
    ref = close.rolling(2).mean().to_numpy()
    assert np.allclose(got, ref, equal_nan=True)
    # values recover to plain means once inf leaves the window
    assert got[-1] == 6.5 and got[-2] == 5.5


@pytest.mark.parametrize("period", [2, 3])
def test_nullable_float_dtype_inputs(period):
    close = pd.Series([1.0, 2.0, pd.NA, 4.0, 5.0, 6.0], dtype="Float64")
    ref = close.rolling(period).mean().to_numpy(dtype=float, na_value=np.nan)
    got = _sma({"close": close}, {"period": period}).to_numpy()
    assert np.allclose(got, ref, equal_nan=True)
    # BB must not raise on nullable-float input either
    _bb({"close": close}, {"period": period, "std": 2.0})


@pytest.mark.parametrize("period", [0, -1])
def test_non_positive_period_all_nan(period):
    close = _make_close(30)
    out = _sma({"close": close}, {"period": period})
    assert len(out) == len(close) and out.isna().all()
    bb = _bb({"close": close}, {"period": period, "std": 2.0})
    assert bb.isna().all().all()
