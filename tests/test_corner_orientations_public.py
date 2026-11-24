"""Public parity and fallback policy tests for feature.corner_orientations."""

import inspect
import warnings

import numpy as np
import pytest

import skimagexx as xx
from skimagexx import _corner_orientations
from skimagexx._compat import FallbackWarning
from fixtures.corner_orientations_v1 import (
    CONTRACT,
    FIXED,
    assert_memoryview_response,
    expected_sha256,
    make_case,
    sha,
)


@pytest.fixture(params=(1, 4))
def candidate(request):
    previous_threads = xx.get_num_threads()
    previous_fallback = xx.set_fallback_enabled(False)
    try:
        xx.set_num_threads(request.param)
        assert xx.get_num_threads() == request.param
        target = xx.feature.corner_orientations
        assert target is _corner_orientations.corner_orientations
        assert inspect.isbuiltin(xx._core.corner_orientations_apply)
        assert str(inspect.signature(target)) == CONTRACT['signature']
        yield target
    finally:
        xx.set_num_threads(previous_threads)
        xx.set_fallback_enabled(previous_fallback)


@pytest.mark.parametrize('case_id', FIXED)
def test_frozen_fixed_corner_orientations(candidate, case_id):
    image, corners, mask, spec = make_case(case_id)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        actual = candidate(image, corners, mask)
    assert not notices
    assert_memoryview_response(actual, len(corners))
    assert not any(np.shares_memory(actual, arg) for arg in (image, corners, mask))
    assert sha(actual) == expected_sha256(spec)


def test_corner_orientations_controls_and_types(candidate):
    from skimage import feature as original

    image, corners, mask, _ = make_case('tiny-interior')
    controls = CONTRACT['controls']

    for name, c_arg, m_arg in (
        ('zero-mask', corners, np.zeros_like(mask)),
        ('one-pixel-mask', corners, np.ones((1, 1), np.uint8)),
        ('reordered-duplicate', corners[[7, 1, 7, 0]].copy(), mask),
        ('empty-coordinates', np.empty((0, 2), dtype=np.intp), mask),
    ):
        actual = candidate(image, c_arg, m_arg)
        assert_memoryview_response(actual, len(c_arg))
        assert sha(actual) == controls[name]['sha256']

    # Bool mask and float32 / float64 images stay native without fallback.
    bool_mask = mask.astype(bool)
    np.testing.assert_array_equal(candidate(image, corners, bool_mask), candidate(image, corners, mask))

    f32_img = (image / 255.0).astype(np.float32)
    f64_img = (image / 255.0).astype(np.float64)
    out_f32 = candidate(f32_img, corners, mask)
    out_f64 = candidate(f64_img, corners, mask)
    assert_memoryview_response(out_f32, len(corners), dtype_str='<f4')
    assert_memoryview_response(out_f64, len(corners), dtype_str='<f8')
    np.testing.assert_array_equal(out_f32, original.corner_orientations(f32_img, corners, mask))
    np.testing.assert_array_equal(out_f64, original.corner_orientations(f64_img, corners, mask))


def test_corner_orientations_repeat_and_readonly(candidate):
    image, corners, mask, spec = make_case('tiny-interior')
    ro_img = image.view()
    ro_img.flags.writeable = False
    first = candidate(ro_img, corners, mask)
    second = candidate(ro_img, corners, mask)
    o1 = assert_memoryview_response(first, len(corners))
    o2 = assert_memoryview_response(second, len(corners))
    assert o1 is not o2 and not np.shares_memory(first, second)
    assert sha(first) == sha(second) == expected_sha256(spec)


def test_corner_orientations_deferred_and_metadata_fallback(candidate, monkeypatch):
    from skimage import feature

    image, corners, mask, _ = make_case('tiny-interior')
    ro_corners = corners.copy()
    ro_corners.flags.writeable = False
    ro_f32 = (image / 255.0).astype(np.float32)
    ro_f32.flags.writeable = False
    oob_corners = np.array([[0, 0]], dtype=np.intp)
    even_mask = np.ones((2, 2), dtype=np.uint8)

    for deferred_args in (
        (image.view(np.dtype('uint8', metadata={'x': 1})), corners, mask),
        (image, corners.view(np.dtype(np.intp, metadata={'x': 1})), mask),
        (image, corners, mask.view(np.dtype('uint8', metadata={'x': 1}))),
        (image.ravel().copy(), corners, mask),
        (ro_f32, corners, mask),
        (image, ro_corners, mask),
        (image, oob_corners, mask),
        (image, corners, even_mask),
        (image[:, ::-1], corners, mask),
    ):
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            candidate(*deferred_args)

    seen = []
    marker = object()

    def provider(*args, **kwargs):
        seen.append((args, kwargs))
        return marker

    monkeypatch.setattr(feature, 'corner_orientations', provider)
    xx.set_fallback_enabled(True)
    tagged_corners = corners.view(np.dtype(np.intp, metadata={'units': 'px'}))
    with pytest.warns(FallbackWarning, match='skimage.feature.corner_orientations'):
        assert candidate(image, tagged_corners, mask) is marker
    (args, kwargs), = seen
    assert not kwargs
    assert len(args) == 3 and args[0] is image and args[1] is tagged_corners and args[2] is mask
