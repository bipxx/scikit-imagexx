import inspect

import numpy as np
import pytest
from skimage.feature import CENSURE as StockCensure
from skimage.feature.util import FeatureDetector

import skimagexx as xx
from skimagexx import _core
from skimagexx._censure import CENSURE
from skimagexx._compat import FallbackWarning


def _image():
    image = np.random.default_rng(123).integers(0, 256, (97, 105), dtype=np.uint8)
    rows, columns = np.ogrid[:97, :105]
    for row, column, radius in [(27, 31, 4), (42, 66, 7), (67, 49, 10)]:
        image[(rows - row) ** 2 + (columns - column) ** 2 <= radius**2] //= 10
    return image


@pytest.mark.parametrize('mode', ['dOb', 'Octagon', 'STAR'])
@pytest.mark.parametrize('layout', ['uint8', 'f32-reversed', 'f64-fortran', 'int16', 'bool'])
def test_censure_public(mode, layout, monkeypatch):
    assert xx.feature.CENSURE is CENSURE
    assert issubclass(CENSURE, FeatureDetector)
    assert inspect.signature(CENSURE) == inspect.signature(StockCensure)
    assert inspect.signature(CENSURE.detect) == inspect.signature(StockCensure.detect)
    image = _image()
    if layout == 'f32-reversed':
        image = (image.astype(np.float32) / 255)[:, ::-1]
    elif layout == 'f64-fortran':
        image = np.asfortranarray(image.astype(np.float64) / 255)
    elif layout == 'int16':
        image = image.astype(np.int16) * 180 - 20000
    elif layout == 'bool':
        image = image > 100
    before = image.copy()
    reference = StockCensure(mode=mode, max_scale=5)
    reference.detect(image)
    native = _core.censure_detect
    assert inspect.isbuiltin(native)
    calls = []

    def count(*args):
        calls.append(args[0].dtype)
        return native(*args)

    monkeypatch.setattr(_core, 'censure_detect', count)
    previous = xx.set_fallback_enabled(False)
    try:
        detector = xx.feature.CENSURE(mode=mode, max_scale=5)
        assert detector.mode == mode.lower()
        assert detector.keypoints is None and detector.scales is None
        assert detector.detect(image) is None
        first_points = detector.keypoints
        np.testing.assert_array_equal(first_points, reference.keypoints, strict=True)
        np.testing.assert_array_equal(detector.scales, reference.scales, strict=True)
        assert not np.shares_memory(first_points, image)
        assert detector.detect(np.zeros_like(image)) is None
        assert detector.keypoints.shape == (0, 2) and detector.scales.shape == (0,)
        assert detector.keypoints is not first_points
    finally:
        xx.set_fallback_enabled(previous)
    assert calls == [np.dtype('float32' if layout == 'f32-reversed' else 'float64')] * 2
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('mode', ['DoB', 'Octagon', 'STAR'])
def test_censure_mutable_options_and_zero_threshold(mode):
    image = np.zeros((43, 49), dtype=np.float64)
    detector = CENSURE(mode=mode)
    reference = StockCensure(mode=mode)
    detector.max_scale = reference.max_scale = 4
    detector.non_max_threshold = reference.non_max_threshold = 0
    detector.line_threshold = reference.line_threshold = 1
    reference.detect(image)
    previous = xx.set_fallback_enabled(False)
    try:
        detector.detect(image)
        np.testing.assert_array_equal(detector.keypoints, reference.keypoints, strict=True)
        np.testing.assert_array_equal(detector.scales, reference.scales, strict=True)
        first = detector.keypoints
        for invalid in [np.empty((0, 49)), np.zeros((1, 49)), np.zeros((43, 49, 3))]:
            with pytest.raises(ValueError) as expected:
                reference.detect(invalid)
            with pytest.raises(ValueError) as actual:
                detector.detect(invalid)
            assert str(actual.value) == str(expected.value)
            assert detector.keypoints is first
    finally:
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize('options', [dict(mode='circle'), dict(min_scale=0), dict(min_scale=2, max_scale=3)])
def test_censure_invalid_construction(options):
    with pytest.raises(ValueError) as expected:
        StockCensure(**options)
    with pytest.raises(ValueError) as actual:
        CENSURE(**options)
    assert str(actual.value) == str(expected.value)


def test_censure_undersized_dob_is_rejected_safely():
    detector = CENSURE()
    with pytest.raises(ValueError, match='too small for max_scale'):
        detector.detect(np.zeros((27, 79)))
    assert detector.keypoints is None and detector.scales is None


def test_censure_unsupported_input_honors_fallback_policy():
    image = _image()
    options = dict(non_max_threshold=np.float32(0.1))
    detector = CENSURE(**options)
    reference = StockCensure(**options)
    reference.detect(image)
    previous = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            detector.detect(image)
        assert detector.keypoints is None and detector.scales is None
        xx.set_fallback_enabled(True)
        with pytest.warns(FallbackWarning, match='CENSURE'):
            detector.detect(image)
    finally:
        xx.set_fallback_enabled(previous)
    np.testing.assert_array_equal(detector.keypoints, reference.keypoints, strict=True)
    np.testing.assert_array_equal(detector.scales, reference.scales, strict=True)


def test_censure_native_binding_checks_borrowed_storage():
    image = np.lib.stride_tricks.as_strided(np.zeros(2), (30, 30), (240, 8))
    with pytest.raises(ValueError, match='ndarray-backed storage'):
        _core.censure_detect(image, 1, 7, 0, 0.15, 10)
