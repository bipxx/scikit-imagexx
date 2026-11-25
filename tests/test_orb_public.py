"""Public ORB state, native ownership, and pinned-stock parity."""

import inspect
import sys

import numpy as np
import pytest
from skimage import feature as stock
from skimage.feature import ORB as StockORB
from skimagexx import _core, feature, get_num_threads, set_fallback_enabled, set_num_threads
from skimagexx._compat import FallbackWarning


_FIELDS = ("keypoints", "scales", "orientations", "responses", "descriptors")


@pytest.fixture
def orb():
    previous_fallback = set_fallback_enabled(False)
    previous_workers = get_num_threads()
    set_num_threads(1)
    try:
        yield feature.ORB
    finally:
        set_num_threads(previous_workers)
        set_fallback_enabled(previous_fallback)


def _compare(original, candidate):
    for field in (*_FIELDS, "mask_"):
        assert hasattr(original, field) == hasattr(candidate, field)
        a, b = getattr(original, field, None), getattr(candidate, field, None)
        if a is None:
            assert b is None
            continue
        assert type(a) is type(b) is np.ndarray
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.strides == b.strides and a.flags.owndata == b.flags.owndata
        if field in ("orientations", "responses"):
            rtol = 2e-6 if a.dtype == np.float32 else 2e-13
            atol = 2e-7 if a.dtype == np.float32 else 2e-14
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(a, b)


def _hot_swap(monkeypatch, call):
    def forbidden(*args, **kwargs):
        pytest.fail("ORB called the original class or provider on its native route")

    monkeypatch.setattr(stock, "ORB", forbidden)
    import skimage.feature.orb as stock_orb
    for name in ("corner_fast", "corner_orientations", "corner_peaks", "corner_harris",
                 "pyramid_gaussian", "_orb_loop"):
        monkeypatch.setattr(stock_orb, name, forbidden)

    names = ("orb_image", "orb_reduce", "orb_detect_octave", "orb_extract_octave")
    entries = {getattr(_core, name) for name in names}
    assert all(inspect.isbuiltin(entry) and entry.__module__ == "skimagexx._core" for entry in entries)
    calls = set()

    def profile(frame, event, value):
        if event == "c_return" and value in entries:
            calls.add(value.__name__)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        assert call() is None
    finally:
        sys.setprofile(previous)
    return calls


def test_orb_statically_discoverable_signatures(orb):
    assert inspect.signature(orb) == inspect.signature(StockORB)
    signatures = {}
    for name, member in inspect.getmembers_static(StockORB):
        if callable(member):
            try:
                signatures[name] = inspect.signature(member)
            except (TypeError, ValueError):
                continue
    assert {"_build_pyramid", "_detect_octave", "_extract_octave"} <= signatures.keys()
    for name, expected in signatures.items():
        assert inspect.signature(inspect.getattr_static(orb, name)) == expected


def test_orb_private_methods_delegate_with_stock_metadata(orb):
    original, candidate = StockORB(n_scales=3), orb(n_scales=3)
    image = np.random.default_rng(37201).random((92, 99), dtype=np.float32)
    points = np.array([[31, 43], [1, 2]], dtype=np.float32)
    orientations = np.array([0.5, 1], dtype=np.float32)
    methods = {
        "_build_pyramid": {"image": image},
        "_detect_octave": {"octave_image": image},
        "_extract_octave": {
            "octave_image": image, "keypoints": points, "orientations": orientations,
        },
    }
    for name, args in methods.items():
        expected = getattr(original, name)(**args)
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            getattr(candidate, name)(**args)
        set_fallback_enabled(True)
        try:
            with pytest.warns(FallbackWarning, match=r"skimage.feature.ORB"):
                actual = getattr(candidate, name)(**args)
        finally:
            set_fallback_enabled(False)
        assert type(actual) is type(expected) and len(actual) == len(expected)
        if name == "_build_pyramid":
            assert np.shares_memory(actual[0], image) == np.shares_memory(expected[0], image)
        for a, b in zip(actual, expected, strict=True):
            assert a.dtype == b.dtype and a.strides == b.strides
            assert a.flags.owndata == b.flags.owndata and type(a.base) is type(b.base)
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("method", ["detect", "detect_and_extract"])
def test_orb_detect_stock(dtype, method, orb, monkeypatch):
    values = np.random.default_rng(28861).integers(0, 256, size=(133, 155), dtype=np.uint8)
    image = values[::-1, 1::2] if dtype is np.uint8 else values.astype(dtype) / dtype(255)
    if dtype is np.int16:
        image = (values.astype(dtype) - 128) * 128
    original, candidate = StockORB(n_keypoints=87, fast_n=10), orb(n_keypoints=87, fast_n=10)
    _compare(original, candidate)
    assert getattr(original, method)(image) is None
    before = image.tobytes()
    calls = _hot_swap(monkeypatch, lambda: getattr(candidate, method)(image))
    expected = {"orb_image", "orb_reduce", "orb_detect_octave"}
    if method == "detect_and_extract":
        expected.add("orb_extract_octave")
    assert calls == expected
    _compare(original, candidate)
    assert image.tobytes() == before


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_orb_extract_stock_preserves_octave_grouping_and_mask(dtype, orb, monkeypatch):
    image = np.random.default_rng(92834).random((83, 96), dtype=dtype)
    points = np.array([[43.5, 45], [19.8, 19.2], [33, 37], [5, 6], [27, 29], [80, 90]],
                      dtype=dtype)
    scales = np.array([2, 1, 1, 1, 2048, 2], dtype=dtype)
    angles = np.array([0.44, 0.5, 1.8, -1.0, 3.1, np.pi], dtype=dtype)
    before = [value.copy() for value in (points, scales, angles)]
    original, candidate = StockORB(downscale=2), orb(downscale=2)
    assert original.extract(image, points, scales, angles) is None
    assert _hot_swap(monkeypatch, lambda: candidate.extract(image, points, scales, angles)) == {
        "orb_image", "orb_reduce", "orb_extract_octave",
    }
    _compare(original, candidate)
    assert candidate.mask_.tolist() == [True, True, False, True, False]
    for value, initial in zip((points, scales, angles), before):
        np.testing.assert_array_equal(value, initial)


def test_orb_tied_corners_empty_results_and_state(orb):
    image = np.zeros((86, 89))
    image[25:57, 26:62] = 1
    original, candidate = StockORB(n_keypoints=14), orb(n_keypoints=14)
    for detector in (original, candidate):
        detector.detect_and_extract(image)
    _compare(original, candidate)
    for detector in (original, candidate):
        descriptors = detector.descriptors
        detector.detect(np.zeros_like(image))
        assert detector.descriptors is descriptors
        with pytest.raises(RuntimeError, match=r"^ORB found no features"):
            detector.detect_and_extract(np.zeros_like(image))
        assert detector.descriptors is descriptors
        with pytest.raises(ValueError, match=r"need at least one array"):
            detector.extract(image, np.empty((0, 2)), np.empty(0), np.empty(0))
        assert detector.descriptors is descriptors
    _compare(original, candidate)


@pytest.mark.parametrize("image", [np.ones((1, 5)), np.ones((2, 3, 4)), np.zeros((0, 8))])
def test_orb_bad_image_shape(orb, image):
    for method in ("detect", "detect_and_extract"):
        with pytest.raises(ValueError) as expected:
            getattr(StockORB(), method)(image)
        with pytest.raises(ValueError) as actual:
            getattr(orb(), method)(image)
        assert str(actual.value) == str(expected.value)


def test_orb_deferred_options_share_mutable_state(orb):
    image = np.random.default_rng(59376).random((92, 95))
    detector = orb()
    detector.detect_and_extract(image)
    descriptors = detector.descriptors
    detector.fast_n = np.int64(12)
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        detector.detect(image)
    set_fallback_enabled(True)
    with pytest.warns(FallbackWarning, match=r"skimage.feature.ORB"):
        assert detector.detect(image) is None
    original = StockORB(fast_n=detector.fast_n)
    original.descriptors = descriptors
    original.detect(image)
    _compare(original, detector)
    assert detector.descriptors is descriptors


def test_orb_optional_workers_match_serial(orb):
    image = np.random.default_rng(16273).random((563, 521), dtype=np.float32)
    serial, parallel = orb(n_keypoints=91), orb(n_keypoints=91)
    serial.detect_and_extract(image)
    set_num_threads(2)
    parallel.detect_and_extract(image)
    _compare(serial, parallel)
