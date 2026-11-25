"""Installed registration parity, native ownership and deferred routing."""

import inspect
import sys
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.fft import fftn
from scipy.ndimage import fourier_shift
from skimage import registration as stock

import skimagexx as xx
from skimagexx import _phase_cross_correlation, registration
from skimagexx._compat import FallbackWarning


@pytest.fixture
def public():
    function = registration.phase_cross_correlation
    assert function is _phase_cross_correlation.phase_cross_correlation
    assert inspect.signature(function) == inspect.signature(stock.phase_cross_correlation)
    before = xx.set_fallback_enabled(False)
    workers = xx.get_num_threads()
    xx.set_num_threads(1)
    try:
        yield function
    finally:
        xx.set_num_threads(workers)
        xx.set_fallback_enabled(before)


def assert_results(result, expected):
    assert type(result) is tuple and len(result) == 3
    for result_part, expected_part in zip(result, expected, strict=True):
        assert type(result_part) is type(expected_part)
    assert result[0].dtype == expected[0].dtype
    assert result[0].flags.c_contiguous and result[0].flags.writeable
    assert not np.shares_memory(result[0], expected[0])
    assert_array_equal(result[0], expected[0])
    tolerance = 8e-4 if expected[0].dtype == np.float32 else 8e-8
    assert_allclose(result[1:], expected[1:], atol=tolerance, rtol=tolerance)


def native_result(public, monkeypatch, left, right, options):
    entry = xx._core.phase_cross_correlation
    assert inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core"

    def forbidden(*args, **kwargs):
        pytest.fail("native registration called stock phase_cross_correlation")

    monkeypatch.setattr(stock, "phase_cross_correlation", forbidden)
    calls = []

    def profiler(frame, event, value):
        if event == "c_return" and value is entry:
            calls.append(value)

    previous = sys.getprofile()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sys.setprofile(profiler)
        try:
            result = public(left, right, **options)
        finally:
            sys.setprofile(previous)
    assert calls == [entry]
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("factor", [1, 8, 32])
@pytest.mark.parametrize("normalization", ["phase", None])
@pytest.mark.parametrize("fourier", [False, True])
def test_phase_cross_correlation_hot_swap(dtype, factor, normalization, fourier, public, monkeypatch):
    rng = np.random.default_rng(37281)
    left = rng.normal(size=(29, 37)).astype(dtype)
    if fourier:
        left = fftn(left)
        offset = (4.375, -8.25) if factor > 1 else (4, -8)
        right = (fourier_shift(left, offset) * np.exp(0.3j)).astype(left.dtype)
    else:
        offset = (4, -8)
        right = np.roll(left, offset, axis=(0, 1)) + rng.normal(scale=0.08, size=left.shape).astype(dtype)
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    snapshots = (left.tobytes(), right.tobytes())
    options = dict(space="FoUrIeR" if fourier else "rEal", upsample_factor=factor, normalization=normalization)
    expected = stock.phase_cross_correlation(left, right, **options)
    result = native_result(public, monkeypatch, left, right, options)
    assert_results(result, expected)
    assert_allclose(result[0], -np.array(offset), atol=1.0 / factor, rtol=0)
    assert snapshots == (left.tobytes(), right.tobytes())


@pytest.mark.parametrize("shape, displacement", [((1, 31), (0, 8)), ((8, 1), (4, 0)), ((1, 1), (0, 0))])
@pytest.mark.parametrize("factor", [1, 8])
def test_phase_cross_correlation_length_one_and_midpoint(shape, displacement, factor, public, monkeypatch):
    left = (np.arange(np.prod(shape), dtype=np.float64) + 10).reshape(shape)
    right = np.roll(left, displacement, axis=(0, 1))
    options = dict(upsample_factor=factor)
    expected = stock.phase_cross_correlation(left, right, **options)
    assert_results(native_result(public, monkeypatch, left, right, options), expected)


@pytest.mark.parametrize("domain", [
    "masks", "disambiguate", "noncontiguous", "subclass", "extended", "tagged",
    "swapped", "mixed", "rank", "factor", "zero", "complex-real", "read-only",
])
def test_phase_cross_correlation_deferred(domain, public):
    rng = np.random.default_rng(291)
    left = rng.normal(size=(18, 20))
    right = np.roll(left, (2, -5), axis=(0, 1))
    options = {}
    if domain == "masks":
        options["reference_mask"] = np.ones_like(left, dtype=bool)
    elif domain == "disambiguate":
        options["disambiguate"] = True
    elif domain == "noncontiguous":
        left, right = left.T, right.T
    elif domain == "subclass":
        left, right = left.view(np.ma.MaskedArray), right.view(np.ma.MaskedArray)
    elif domain == "extended":
        left, right = left.astype(np.longdouble), right.astype(np.longdouble)
    elif domain == "tagged":
        left = left.view(np.dtype("f8", metadata={"test": "phase"}))
    elif domain == "swapped":
        left = left.astype(left.dtype.newbyteorder("S"))
    elif domain == "mixed":
        right = right.astype(np.float32)
    elif domain == "rank":
        left, right = left[0], right[0]
    elif domain == "factor":
        options["upsample_factor"] = 33
    elif domain == "zero":
        left = np.zeros_like(left)
    elif domain == "complex-real":
        left, right = left.astype(np.complex128), right.astype(np.complex128)
    elif domain == "read-only":
        left.setflags(write=False)
    with pytest.raises(NotImplementedError, match="outside native coverage"):
        public(left, right, **options)
    xx.set_fallback_enabled(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = stock.phase_cross_correlation(left, right, **options)
        with pytest.warns(FallbackWarning, match="skimage.registration.phase_cross_correlation"):
            actual = public(left, right, **options)
    assert_results(actual, expected)


def test_phase_cross_correlation_rejects_unbounded_storage(public):
    allocation = np.zeros(1, dtype=np.float32)
    invalid = np.lib.stride_tricks.as_strided(allocation, shape=(3, 3), strides=(12, 4))
    assert invalid.flags.c_contiguous
    with pytest.raises(ValueError, match="requires ndarray-backed storage"):
        public(invalid, np.ones((3, 3), dtype=np.float32))


def test_phase_cross_correlation_dtype_alias(public, monkeypatch):
    rng = np.random.default_rng(298)
    dtype = np.dtype("=d", copy=True)
    left = rng.normal(size=(24, 38)).view(dtype)
    right = np.roll(left, (5, -3), axis=(0, 1)).view(dtype)
    options = {"upsample_factor": 4}
    expected = stock.phase_cross_correlation(left, right, **options)
    assert_results(native_result(public, monkeypatch, left, right, options), expected)


def test_phase_cross_correlation_worker_parity(public):
    rng = np.random.default_rng(299)
    left = rng.normal(size=(527, 535)).astype(np.float32)
    right = np.roll(left, (7, -12), axis=(0, 1))
    expected = stock.phase_cross_correlation(left, right)
    serial = public(left, right)
    assert_results(serial, expected)
    try:
        xx.set_num_threads(4)
    except ValueError as error:
        assert "no OpenMP support" in str(error)
        assert xx.get_num_threads() == 1
    else:
        assert_results(public(left, right), serial)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_phase_cross_correlation_fractional_real(dtype, public, monkeypatch):
    rng = np.random.default_rng(300)
    left = rng.normal(size=(35, 39)).astype(dtype)
    shifted = fourier_shift(fftn(left), (3.25, -5.375))
    right = np.ascontiguousarray(np.fft.ifftn(shifted).real, dtype=dtype)
    options = {"upsample_factor": 16}
    expected = stock.phase_cross_correlation(left, right, **options)
    assert_results(native_result(public, monkeypatch, left, right, options), expected)
    assert_array_equal(expected[0], [-3.25, 5.375])
