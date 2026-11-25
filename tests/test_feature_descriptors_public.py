import inspect

import numpy as np
import pytest
from skimage import feature as original

from skimagexx import FallbackWarning, _core, feature, set_fallback_enabled


_APIS = (
    "hog", "daisy", "local_binary_pattern", "multiblock_lbp", "match_template",
    "multiscale_basic_features",
)


@pytest.fixture
def strict():
    previous = set_fallback_enabled(False)
    try:
        yield
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("name", _APIS)
def test_signatures(name):
    assert inspect.signature(getattr(feature, name)) == inspect.signature(getattr(original, name))


@pytest.mark.parametrize("method", ("default", "ror", "uniform", "nri_uniform", "var"))
@pytest.mark.parametrize("points,radius", ((8, 1), (16, 2.25), (4, 0)))
def test_local_binary_pattern(strict, method, points, radius):
    image = np.random.default_rng(201).integers(0, 256, (31, 37), np.uint8)
    image[6:13, 8:16] = 42
    expected = original.local_binary_pattern(image, points, radius, method.upper())
    actual = feature.local_binary_pattern(image, points, radius, method.upper())
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ("f", "d", "Q", "L"))
@pytest.mark.parametrize("row,col", ((0, 0), (0, 2), (2, 0), (3, 4)))
def test_multiblock_lbp(strict, dtype, row, col):
    rng = np.random.default_rng(210)
    ii = rng.integers(0, 255, (19, 25)).cumsum(0).cumsum(1).astype(dtype)
    args = (ii, row, col, 5, 4)
    # Native widths of NumPy unsigned integer aliases depend on the platform.
    if ii.dtype.itemsize == 4 and ii.dtype.kind == "u":
        args = (ii.astype(np.uint64), *args[1:])
    assert feature.multiblock_lbp(*args) == original.multiblock_lbp(*args)


def test_multiblock_out_of_bounds():
    with pytest.raises(ValueError, match="grid must fit"):
        feature.multiblock_lbp(np.zeros((10, 10), np.float32), 6, 6, 2, 2)


def test_multiblock_keeps_whole_table_cast_warning():
    previous = set_fallback_enabled(True)
    try:
        image = np.ones((12, 12), dtype=np.float64)
        image[-1, -1] = 1e100
        with pytest.warns(FallbackWarning, match="multiblock_lbp"), pytest.warns(RuntimeWarning, match="overflow"):
            assert feature.multiblock_lbp(image, 0, 0, 2, 2) == 255
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("norm", ("L1", "L1-sqrt", "L2", "L2-Hys"))
@pytest.mark.parametrize("dtype,channels", ((np.uint8, False), (np.float32, True), (np.float64, False)))
def test_hog(strict, norm, dtype, channels):
    shape = (64, 53, 3) if channels else (64, 53)
    image = np.random.default_rng(217).integers(0, 256, shape, np.uint8).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        image /= 255
    kwargs = dict(
        orientations=8, pixels_per_cell=(5, 7), cells_per_block=(3, 2),
        block_norm=norm, visualize=True, transform_sqrt=channels, feature_vector=False,
        channel_axis=-1 if channels else None,
    )
    expected = original.hog(image, **kwargs)
    actual = feature.hog(image, **kwargs)
    for a, e in zip(actual, expected, strict=True):
        assert a.dtype == e.dtype
        np.testing.assert_allclose(a, e, atol=2e-7, rtol=3e-7)


def test_hog_default_and_worker_counts(strict):
    image = np.random.default_rng(221).integers(0, 255, (512, 524), np.uint8)
    expected = original.hog(image)
    previous = _core.get_num_threads()
    try:
        _core.set_num_threads(1)
        one = feature.hog(image)
        _core.set_num_threads(4)
        four = feature.hog(image)
    finally:
        _core.set_num_threads(previous)
    np.testing.assert_allclose(one, expected, atol=1e-14)
    np.testing.assert_array_equal(four, one)


def test_lbp_worker_counts(strict):
    image = np.random.default_rng(220).integers(0, 255, (520, 521), np.uint8)
    previous = _core.get_num_threads()
    try:
        _core.set_num_threads(1)
        one = feature.local_binary_pattern(image, 8, 1, "uniform")
        _core.set_num_threads(4)
        four = feature.local_binary_pattern(image, 8, 1, "uniform")
    finally:
        _core.set_num_threads(previous)
    np.testing.assert_array_equal(one, four)


@pytest.mark.parametrize("norm", ("l1", "l2", "daisy", "off"))
@pytest.mark.parametrize("dtype", (np.uint8, np.float32, np.float64))
def test_daisy(strict, norm, dtype):
    image = np.random.default_rng(223).integers(0, 256, (48, 55), np.uint8).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        image /= 255
    kwargs = dict(step=3, histograms=5, orientations=6, normalization=norm,
                  sigmas=[1.0, 2.0, 3.5], ring_radii=[3.5, 7])
    expected = original.daisy(image, **kwargs)
    actual = feature.daisy(image, **kwargs)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


def test_daisy_default(strict):
    image = np.random.default_rng(225).random((57, 63))
    np.testing.assert_allclose(feature.daisy(image), original.daisy(image), atol=1e-14)


@pytest.mark.parametrize("dtype", (np.dtype("=f4"), np.dtype("=f8")))
@pytest.mark.parametrize("size", ((4, 6), (12, 13), (1, 1)))
@pytest.mark.parametrize("pad", (False, True))
def test_match_template(strict, dtype, size, pad):
    rng = np.random.default_rng(227)
    image = rng.normal(size=(55, 59)).astype(dtype)
    patch = image[9:9 + size[0], 11:11 + size[1]].copy()
    args = (image, patch)
    kwargs = dict(pad_input=pad, constant_values=0.25)
    expected = original.match_template(*args, **kwargs)
    actual = feature.match_template(*args, **kwargs)
    assert actual.dtype == expected.dtype
    tol = 3e-5 if dtype.itemsize == 4 else 3e-12
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
    if size != (1, 1):
        peak = (9 + size[0] // 2, 11 + size[1] // 2) if pad else (9, 11)
        assert np.unravel_index(actual.argmax(), actual.shape) == peak


def test_match_template_bytes(strict):
    image = np.random.default_rng(228).integers(0, 256, (51, 53), np.uint8)
    patch = image[8:21, 9:24].copy()
    expected = original.match_template(image, patch, pad_input=True)
    actual = feature.match_template(image, patch, pad_input=True)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_match_template_positive_float32(strict):
    rng = np.random.default_rng(228)
    image = rng.random((130, 143), dtype=np.float32)
    patch = image[8:12, 9:14].copy()
    expected = original.match_template(image, patch)
    actual = feature.match_template(image, patch)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


def test_match_template_high_offset_delegates():
    previous = set_fallback_enabled(True)
    try:
        rng = np.random.default_rng(228)
        image = rng.random((37, 42), dtype=np.float32) + 100
        patch = image[8:12, 9:14].copy()
        with pytest.warns(FallbackWarning, match="match_template"):
            actual = feature.match_template(image, patch)
        np.testing.assert_array_equal(actual, original.match_template(image, patch))
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize("name", ("daisy", "match_template", "multiscale_basic_features"))
def test_shared_kernels_with_four_workers(strict, name):
    image = np.random.default_rng(229).random((514, 515), dtype=np.float32)
    if name == "daisy":
        args = (image,)
        kwargs = dict(step=16, radius=4, histograms=3, orientations=3, rings=1)
    elif name == "match_template":
        args = (image, image[8:21, 9:24].copy())
        kwargs = {}
    else:
        args = (image,)
        kwargs = dict(num_sigma=2, sigma_max=3, workers=1)
    func = getattr(feature, name)
    previous = _core.get_num_threads()
    try:
        _core.set_num_threads(1)
        one = func(*args, **kwargs)
        _core.set_num_threads(4)
        four = func(*args, **kwargs)
    finally:
        _core.set_num_threads(previous)
    np.testing.assert_array_equal(one, four)


@pytest.mark.parametrize("dtype,channels", ((np.uint8, False), (np.float32, True), (np.float64, False)))
@pytest.mark.parametrize("selected", ((True, True, True), (False, True, False), (False, False, True)))
def test_multiscale(strict, dtype, channels, selected):
    shape = (47, 53, 3) if channels else (47, 53)
    image = np.random.default_rng(229).integers(0, 256, shape, np.uint8).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        image /= 255
    kwargs = dict(sigma_min=0.7, sigma_max=3.4, num_sigma=3,
                  channel_axis=-1 if channels else None, workers=1)
    expected = original.multiscale_basic_features(image, *selected, **kwargs)
    actual = feature.multiscale_basic_features(image, *selected, **kwargs)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


def _routing_case(name, image):
    if name == "local_binary_pattern":
        return (image, 8, 1), {}
    if name == "multiblock_lbp":
        return (image, 0, 0, 2, 2), {}
    if name == "match_template":
        patch = np.arange(25, dtype=image.dtype).reshape(5, 5).copy()
        return (image, patch), {}
    return (image,), {}


@pytest.mark.parametrize("name", _APIS)
@pytest.mark.parametrize("kind", ("tagged", "subclass", "readonly", "strided"))
def test_fallback_routing(name, kind):
    dtype = np.uint8 if name == "local_binary_pattern" else np.float64
    image = np.random.default_rng(230).integers(0, 100, (38, 76), np.uint8).astype(dtype)
    if dtype is np.float64:
        image /= 255
    if kind == "tagged":
        image = image.view(np.dtype(dtype, metadata={"tag": True}))
    elif kind == "subclass":
        class ArraySubclass(np.ndarray):
            pass
        image = image.view(ArraySubclass)
    elif kind == "readonly":
        image.flags.writeable = False
    else:
        image = image[:, ::2]
    args, kwargs = _routing_case(name, image)
    previous = set_fallback_enabled(True)
    try:
        with pytest.warns(FallbackWarning, match=name):
            actual = getattr(feature, name)(*args, **kwargs)
    finally:
        set_fallback_enabled(previous)
    expected = getattr(original, name)(*args, **kwargs)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name,symbol", (
    ("hog", "hog_native"),
    ("daisy", "daisy_native"),
    ("local_binary_pattern", "local_binary_pattern_u8"),
    ("multiblock_lbp", "multiblock_lbp_native"),
    ("match_template", "match_template_native"),
    ("multiscale_basic_features", "multiscale_basic_features_native"),
))
def test_native_entry_and_dtype_alias(strict, monkeypatch, name, symbol):
    dtype = "=u1" if name == "local_binary_pattern" else "=f8"
    image = np.ones((38, 44), dtype=dtype)
    native = getattr(_core, symbol)
    calls = []

    def spy(*args, **kwargs):
        value = native(*args, **kwargs)
        assert value is not None
        calls.append(name)
        return value

    monkeypatch.setattr(_core, symbol, spy)
    args, kwargs = _routing_case(name, image)
    getattr(feature, name)(*args, **kwargs)
    assert calls == [name]


def test_long_double_uses_fallback():
    previous = set_fallback_enabled(True)
    try:
        image = np.random.default_rng(231).random((38, 44)).astype(np.longdouble)
        with pytest.warns(FallbackWarning, match="hog"):
            value = feature.hog(image)
        np.testing.assert_array_equal(value, original.hog(image))
    finally:
        set_fallback_enabled(previous)
