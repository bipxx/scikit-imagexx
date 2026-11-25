"""Installed random walker dispatch, stock parity and fallback boundaries."""

import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import inspect
from pathlib import Path
import runpy
import warnings

import numpy as np
import pytest
import skimage as upstream
from skimage.segmentation import random_walker as original

import skimagexx as xx
from skimagexx import _core, _random_walker, segmentation


BENCHMARK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/random_walker.py"))


@pytest.fixture
def strict_native():
    fallback = xx.set_fallback_enabled(False)
    try:
        yield segmentation.random_walker
    finally:
        xx.set_fallback_enabled(fallback)


def _small_case(dtype=np.float64, labels_dtype=np.int32):
    rng = np.random.default_rng(902)
    image = rng.random((9, 13)).astype(dtype) if dtype != np.uint8 else rng.integers(
        0, 100, size=(9, 13), dtype=dtype,
    )
    labels = np.zeros(image.shape, dtype=labels_dtype)
    labels[:, 0] = 59
    labels[:, -1] = 7
    return image, labels


def _observe(function, args, options):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = function(*args, **options)
    messages = [(type(item.message), str(item.message)) for item in observed]
    return result, messages


def _assert_result(result, expected, rtol=2e-11, atol=2e-12):
    assert type(result) is type(expected) is np.ndarray
    assert (result.shape, result.strides, result.dtype, result.dtype.metadata) == (
        expected.shape, expected.strides, expected.dtype, expected.dtype.metadata,
    )
    assert result.flags.owndata == expected.flags.owndata
    assert result.flags.writeable == expected.flags.writeable
    if result.dtype.kind == "f":
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(result, expected)


def test_installed_hot_swap_and_aliases(strict_native, monkeypatch):
    from skimagexx.coverage import native_functions

    public = strict_native
    nested = importlib.import_module("skimagexx.segmentation.random_walker_segmentation")
    assert nested.random_walker is public is _random_walker.random_walker
    assert segmentation.__all__.count("random_walker") == 1
    assert tuple(native_functions()).count("segmentation.random_walker") == 1
    assert _random_walker._core is _core
    assert _core.__file__.endswith(tuple(EXTENSION_SUFFIXES))
    assert inspect.signature(public) == inspect.signature(original)

    args = _small_case()
    options = {"beta": 45, "tol": 1e-5, "spacing": (1.0, 1.5)}

    def pipeline(backend):
        return backend.segmentation.random_walker(*args, **options)

    expected = pipeline(upstream)
    native = _core.random_walker
    assert inspect.isbuiltin(native) and native.__module__ == _core.__name__
    calls = []

    def tracked(*received):
        assert received[0] is args[0] and received[1] is args[1]
        calls.append(xx.get_num_threads())
        return native(*received)

    def forbidden(*args, **kwargs):
        pytest.fail("native random walker delegated to the original API")

    monkeypatch.setattr(upstream.segmentation, "random_walker", forbidden)
    monkeypatch.setattr(_core, "random_walker", tracked)
    workers = xx.get_num_threads()
    try:
        for count in (1, 2):
            xx.set_num_threads(count)
            _assert_result(pipeline(xx), expected)
    finally:
        xx.set_num_threads(workers)
    assert calls == [1, 2]


@pytest.mark.parametrize("spec", BENCHMARK["case_specs"](), ids=lambda spec: spec["id"])
def test_public_original_parity(spec, strict_native):
    args, options = BENCHMARK["fixture"](spec, np)
    before = [value.tobytes() for value in args]
    expected, notices = _observe(original, args, options)
    result, actual_notices = _observe(strict_native, args, options)
    _assert_result(result, expected)
    assert actual_notices == notices
    assert [value.tobytes() for value in args] == before
    assert all(not np.shares_memory(result, value) for value in args)


@pytest.mark.parametrize("label_dtype", (np.int8, np.uint16, np.longlong, np.ulonglong))
@pytest.mark.parametrize("probabilities", (False, True))
def test_equivalent_dtypes_and_full_probabilities(label_dtype, probabilities, strict_native):
    image, labels = _small_case(labels_dtype=label_dtype)
    args = (image.view(image.dtype.newbyteorder("=")), labels.view(labels.dtype.newbyteorder("=")))
    options = {"beta": 75.0, "tol": 1e-8, "return_full_prob": probabilities}
    expected, notices = _observe(original, args, options)
    result, actual_notices = _observe(strict_native, args, options)
    _assert_result(result, expected)
    assert actual_notices == notices
    if probabilities:
        assert np.all(result[0, labels == 7] == 1.0)
        assert np.all(result[1, labels == 59] == 1.0)


@pytest.mark.parametrize("dtype", (np.float32, np.uint8))
def test_image_precision(dtype, strict_native):
    args = _small_case(dtype=dtype)
    options = {"beta": 20, "tol": 1e-7, "return_full_prob": True}
    if dtype is np.float32 and np.lib.NumpyVersion(np.__version__) < "2.0.0":
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            strict_native(*args, **options)
        return
    expected, notices = _observe(original, args, options)
    result, actual_notices = _observe(strict_native, args, options)
    _assert_result(result, expected, rtol=6e-6 if dtype is np.float32 else 2e-11)
    assert actual_notices == notices


@pytest.mark.parametrize("boundary", (
    "tagged-image", "tagged-labels", "readonly", "strided", "subclass", "longdouble",
    "negative-label", "constant", "nonfinite", "copy-false", "other-solver", "channel-axis",
))
def test_compatibility_boundaries(boundary, strict_native, monkeypatch):
    image, labels = _small_case()
    options = {}
    if boundary == "tagged-image":
        image = image.view(np.dtype(image.dtype, metadata={"tag": 1}))
    elif boundary == "tagged-labels":
        labels = labels.view(np.dtype(labels.dtype, metadata={"tag": 1}))
    elif boundary == "readonly":
        image.setflags(write=False)
    elif boundary == "strided":
        image, labels = image[:, ::-1], labels[:, ::-1]
    elif boundary == "subclass":
        image = image.view(type("ArraySubclass", (np.ndarray,), {}))
    elif boundary == "longdouble":
        image = image.astype(np.longdouble)
    elif boundary == "negative-label":
        labels[1, 1] = -1
    elif boundary == "constant":
        image.fill(0.5)
    elif boundary == "nonfinite":
        image[1, 1] = np.nan
    elif boundary == "copy-false":
        options["copy"] = False
    elif boundary == "other-solver":
        options["mode"] = "bf"
    else:
        image = image[..., None]
        options["channel_axis"] = -1

    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        strict_native(image, labels, **options)

    sentinel = object()

    def fallback(data, markers, **forwarded):
        assert data is image and markers is labels
        assert {name: forwarded[name] for name in options} == options
        return sentinel

    monkeypatch.setattr(upstream.segmentation, "random_walker", fallback)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning, match="skimage.segmentation.random_walker"):
        assert strict_native(image, labels, **options) is sentinel


def test_core_rejects_unbounded_owner_before_read():
    image, labels = _small_case()
    unbounded = np.lib.stride_tricks.as_strided(
        image[:1].copy(), shape=image.shape, strides=image.strides,
    )
    with pytest.raises(ValueError, match="ndarray-backed storage"):
        _core.random_walker(unbounded, labels, 130, 1e-3, 1e-3, (1, 1, 1), False)
