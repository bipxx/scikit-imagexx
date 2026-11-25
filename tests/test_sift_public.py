"""Native SIFT class parity against the pinned scikit-image reference."""

import inspect
import sys
import warnings

import numpy as np
import pytest

from skimage import feature as original_feature
from skimage.data import camera, coins
import skimagexx as xx
from skimagexx._compat import FallbackWarning
from skimagexx._sift import SIFT


def assert_state(reference, native):
    for attr in ("deltas", "scalespace_sigmas", "positions", "sigmas", "orientations",
                 "octaves", "scales", "keypoints", "descriptors"):
        expected, actual = getattr(reference, attr), getattr(native, attr)
        if expected is None:
            assert actual is None, attr
            continue
        assert actual.dtype == expected.dtype, attr
        if attr in ("positions", "sigmas", "orientations"):
            tol = 1e-6 if expected.dtype == np.float32 else 1e-9
            np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol, err_msg=attr)
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=attr)
    assert native.n_octaves == reference.n_octaves
    assert native.float_dtype == reference.float_dtype


@pytest.fixture
def strict():
    prior = xx.set_fallback_enabled(False)
    workers = xx.get_num_threads()
    xx.set_num_threads(1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FallbackWarning)
            yield
    finally:
        xx.set_num_threads(workers)
        xx.set_fallback_enabled(prior)


@pytest.mark.parametrize(("image_factory", "options"), [
    (camera, {}),
    (lambda: camera().astype(np.float32) / 255, {}),
    (coins, {}),
    (lambda: camera()[::3, ::2], dict(upsampling=1, n_scales=4, n_bins=18, n_hist=3, n_ori=6)),
    (lambda: camera()[::4, ::4], dict(upsampling=4, sigma_in=0.3)),
    (lambda: camera()[::3, ::2].astype(np.float16) / 255, dict(upsampling=1, c_max=1)),
    (lambda: (camera()[::3, ::2].astype(np.uint16) * 257).astype(">u2"), dict(upsampling=1)),
])
def test_sift_public_state(image_factory, options, strict, monkeypatch):
    assert xx.feature.SIFT is SIFT
    original = original_feature.SIFT
    assert inspect.signature(SIFT) == inspect.signature(original)
    image = image_factory()
    expected, actual = original(**options), xx.feature.SIFT(**options)
    assert_state(expected, actual)
    assert expected.detect_and_extract(image) is None
    entry = xx._core.sift
    assert inspect.isbuiltin(entry)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("native SIFT called the original public method")

    for method in ("detect", "extract", "detect_and_extract"):
        monkeypatch.setattr(original, method, forbidden)

    def profile(frame, event, arg):
        if event == "c_return" and arg is entry:
            calls.append(arg)

    before = sys.getprofile()
    try:
        sys.setprofile(profile)
        assert actual.detect_and_extract(image) is None
        assert_state(expected, actual)
        split = xx.feature.SIFT(**options)
        assert split.detect(image) is None
        assert split.descriptors is None
        assert split.extract(image) is None
        assert_state(expected, split)
    finally:
        sys.setprofile(before)
    assert calls == [entry] * 3


def test_sift_state_after_reuse_and_failure(strict):
    image = camera()[::2, ::2]
    target = camera()[::4, ::4].astype(np.float32) / 255
    reference, native = original_feature.SIFT(), xx.feature.SIFT()
    for model in (reference, native):
        with pytest.raises(TypeError, match="has no len"):
            model.extract(image)
    assert_state(reference, native)
    for model in (reference, native):
        model.detect_and_extract(image)
        original_descriptors = model.descriptors
        model.detect(target)
        assert model.descriptors is original_descriptors
        model.orientations += 0.1
        model.extract(target)
        with pytest.raises(RuntimeError, match="SIFT found no features"):
            model.detect_and_extract(np.zeros((32, 32), dtype=np.uint8))
    assert_state(reference, native)


def test_sift_workers(strict):
    image = camera()[::2, ::2]
    serial, parallel = xx.feature.SIFT(), xx.feature.SIFT()
    serial.detect_and_extract(image)
    xx.set_num_threads(2)
    parallel.detect_and_extract(image)
    assert_state(serial, parallel)


def test_sift_validation_and_policy(strict):
    with pytest.raises(ValueError, match="upsampling must be 1, 2 or 4"):
        xx.feature.SIFT(upsampling=3)
    with pytest.raises(ZeroDivisionError):
        xx.feature.SIFT(n_scales=0)
    image = (camera()[::4, ::4].astype(np.float64) / 255).tolist()
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        xx.feature.SIFT().detect(image)
    xx.set_fallback_enabled(True)
    expected, actual = original_feature.SIFT(), xx.feature.SIFT()
    expected.detect(image)
    with pytest.warns(FallbackWarning, match=r"skimage.feature.sift.SIFT"):
        assert actual.detect(image) is None
    assert_state(expected, actual)


def test_sift_zero_norm_fallback_preserves_numpy_diagnostics(strict):
    image = camera()[::4, ::4]
    flat = np.zeros_like(image, dtype=np.float32)
    expected, actual = original_feature.SIFT(), xx.feature.SIFT()
    expected.detect(image)
    actual.detect(image)
    sigmas_before = actual.scalespace_sigmas
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        actual.extract(flat)
    assert actual.descriptors is None
    assert actual.scalespace_sigmas is sigmas_before
    assert_state(expected, actual)
    xx.set_fallback_enabled(True)

    def observed_warnings(model):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            model.extract(flat)
        return seen

    upstream = observed_warnings(expected)
    assert upstream
    received = observed_warnings(actual)
    assert any(w.category is FallbackWarning for w in received)
    assert [(w.category, str(w.message)) for w in received if w.category is not FallbackWarning] == [
        (w.category, str(w.message)) for w in upstream
    ]
    assert_state(expected, actual)
    with np.errstate(invalid="raise"):
        with pytest.raises(FloatingPointError) as stock_error:
            expected.extract(flat)
        with pytest.warns(FallbackWarning), pytest.raises(FloatingPointError) as native_error:
            actual.extract(flat)
    assert str(native_error.value) == str(stock_error.value)


def test_sift_statically_discoverable_members():
    original = original_feature.SIFT
    for name in dir(original):
        if name.startswith("__") and name != "__init__":
            continue
        expected = inspect.getattr_static(original, name)
        actual = inspect.getattr_static(xx.feature.SIFT, name)
        assert type(actual) is type(expected), name
        if isinstance(expected, property):
            expected, actual = expected.fget, actual.fget
        if callable(expected):
            assert inspect.signature(actual) == inspect.signature(expected), name


def test_sift_private_helpers(strict):
    reference, native = original_feature.SIFT(), xx.feature.SIFT()
    row, col = np.arange(4), np.arange(4) / 2
    np.testing.assert_array_equal(native._rotate(row, col, 1.2), reference._rotate(row, col, 1.2))
    xy = np.column_stack((row, col))
    np.testing.assert_array_equal(native._inrange(xy, (4, 4)), reference._inrange(xy, (4, 4)))
    assert native._fit(np.array([3.0, 8.0, 4.0])) == reference._fit(np.array([3.0, 8.0, 4.0]))
    image = camera()[::4, ::4]
    native._set_number_of_octaves(image.shape)
    reference._set_number_of_octaves(image.shape)
    assert native.n_octaves == reference.n_octaves
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        native._preprocess(image)
    xx.set_fallback_enabled(True)

    def candidate(method, *args):
        with pytest.warns(FallbackWarning, match=r"skimage.feature.sift.SIFT"):
            return getattr(native, method)(*args)

    ref_image, native_image = reference._preprocess(image), candidate("_preprocess", image)
    np.testing.assert_array_equal(native_image, ref_image)
    ref_space, native_space = reference._create_scalespace(ref_image), candidate("_create_scalespace", native_image)
    assert len(ref_space) == len(native_space)
    for upstream, actual in zip(ref_space, native_space):
        np.testing.assert_array_equal(actual, upstream)
    dogspace = [np.diff(layer, axis=2) for layer in ref_space]
    groups = reference._find_localize_evaluate(dogspace, image.shape)
    for expected, actual in zip(groups, candidate("_find_localize_evaluate", dogspace, image.shape)):
        np.testing.assert_array_equal(actual, expected)
    reference._compute_orientation(*groups, ref_space)
    gradient = candidate("_compute_orientation", *groups, native_space)
    reference._compute_descriptor(gradient)
    candidate("_compute_descriptor", gradient)
    assert_state(reference, native)
