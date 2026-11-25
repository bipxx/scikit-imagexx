import inspect

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift
from skimage import registration as stock

from skimagexx import _core, _registration_flows, registration as native, set_fallback_enabled
from skimagexx._compat import FallbackWarning


APIS = ("optical_flow_ilk", "optical_flow_tvl1")


def _images(shape, dtype):
    rng = np.random.default_rng(1371)
    reference = gaussian_filter(rng.random(shape).astype(dtype), 0.8)
    return reference, shift(reference, (0.8, -1.25), order=1, mode="nearest")


@pytest.mark.parametrize("api", APIS)
def test_signature(api):
    function = getattr(native, api)
    assert function is getattr(_registration_flows, api)
    assert inspect.signature(function) == inspect.signature(getattr(stock, api))


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("custom", [False, True])
def test_native_parity(api, dtype, custom):
    images = _images((65, 77) if custom else (41, 54), dtype)
    options = {}
    if custom:
        options = {"num_warp": 3, "prefilter": True}
        if api == "optical_flow_ilk":
            options.update(radius=4, gaussian=True)
        else:
            options.update(attachment=12, tightness=0.24, num_iter=7, tol=2e-5)
    expected = getattr(stock, api)(*images, dtype=dtype, **options)
    previous = set_fallback_enabled(False)
    try:
        actual = getattr(native, api)(*images, dtype=dtype, **options)
    finally:
        set_fallback_enabled(previous)
    assert actual.shape == (2, *images[0].shape)
    assert actual.dtype == expected.dtype == dtype
    tolerance = 5e-5 if dtype is np.float32 else 1e-9
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("api", APIS)
def test_dtype_alias_and_bounded_view(api):
    images = _images((14, 19), np.float64)
    dtype = np.dtype("=d", copy=True)
    pair = []
    for image in images:
        backing = np.empty((15, 19), dtype=dtype)
        backing[1:] = image
        pair.append(backing[1:])
    previous = set_fallback_enabled(False)
    try:
        actual = getattr(native, api)(*pair, dtype=dtype, num_warp=1)
        mixed = getattr(native, api)(pair[0].astype(np.float32), pair[1], num_warp=1)
    finally:
        set_fallback_enabled(previous)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, getattr(stock, api)(*pair, dtype=dtype, num_warp=1),
                               atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        mixed, getattr(stock, api)(pair[0].astype(np.float32), pair[1], num_warp=1),
        atol=5e-5, rtol=5e-5,
    )


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("kind", ["metadata", "readonly", "strided", "integer", "subclass",
                                  "byteorder", "longdouble", "volume", "output-metadata"])
def test_unsupported_storage_falls_back(api, kind):
    images = _images((14, 19), np.float32)
    dtype = np.float32
    if kind == "metadata":
        images = tuple(a.view(np.dtype("f4", metadata={"signal": "test"})) for a in images)
    elif kind == "readonly":
        for a in images:
            a.flags.writeable = False
    elif kind == "strided":
        images = tuple(a[:, ::2] for a in images)
    elif kind == "integer":
        images = tuple((a * 255).astype(np.uint8) for a in images)
    elif kind == "subclass":
        class Image(np.ndarray):
            pass

        images = tuple(a.view(Image) for a in images)
    elif kind == "byteorder":
        images = tuple(a.astype(a.dtype.newbyteorder("S")) for a in images)
    elif kind == "longdouble":
        images = tuple(a.astype(np.longdouble) for a in images)
    elif kind == "volume":
        images = tuple(np.repeat(a[..., None], 3, axis=-1) for a in images)
    else:
        dtype = np.dtype("f4", metadata={"signal": "output"})
    if kind != "output-metadata":
        args = (True, 7, 1, False, False) if api.endswith("ilk") else (True, 15, 0.3, 1, 10, 1e-4, False)
        assert getattr(_core, api)(*images, *args) is None
    if kind == "longdouble":
        with pytest.raises(ValueError, match="Cannot convert"):
            getattr(stock, api)(*images, dtype=dtype, num_warp=1)
        with pytest.warns(FallbackWarning, match=api), pytest.raises(ValueError, match="Cannot convert"):
            getattr(native, api)(*images, dtype=dtype, num_warp=1)
        return
    with pytest.warns(FallbackWarning, match=api):
        actual = getattr(native, api)(*images, dtype=dtype, num_warp=1)
    expected = getattr(stock, api)(*images, dtype=dtype, num_warp=1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype.metadata == expected.dtype.metadata


@pytest.mark.parametrize("api", APIS)
def test_worker_parity(api):
    images = _images((514, 521), np.float32)
    call = getattr(native, api)
    options = {"num_warp": 1}
    if api == "optical_flow_tvl1":
        options["num_iter"] = 2
    previous = _core.get_num_threads()
    allow_fallback = set_fallback_enabled(False)
    try:
        _core.set_num_threads(1)
        serial = call(*images, **options)
        try:
            _core.set_num_threads(4)
        except ValueError as error:
            assert "no OpenMP support" in str(error)
            assert _core.get_num_threads() == 1
        else:
            np.testing.assert_array_equal(call(*images, **options), serial)
    finally:
        _core.set_num_threads(previous)
        set_fallback_enabled(allow_fallback)


def test_ill_conditioned_ilk_falls_back():
    rng = np.random.default_rng(1371)
    shape = (35, 53)
    reference = 0.001 * gaussian_filter(rng.random(shape), 0.8)
    reference += 0.01 * np.indices(shape).sum(axis=0)
    moving = shift(reference, (0.8, -1.25), order=1, mode="nearest")
    expected = stock.optical_flow_ilk(reference, moving, dtype=np.float64)
    with pytest.warns(FallbackWarning, match="optical_flow_ilk"):
        actual = native.optical_flow_ilk(reference, moving, dtype=np.float64)
    np.testing.assert_array_equal(actual, expected)
