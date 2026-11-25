"""Scoped installed warp_polar parity against scikit-image 0.26.0."""

import inspect
import warnings

import numpy as np
import pytest
import skimage
from scipy import ndimage
from skimage import transform as original
from skimage.transform import _warps

import skimagexx as xx
from skimagexx import _warp_polar


def _image(dtype, channels=False):
    rng = np.random.default_rng(604)
    shape = (17, 23, 3) if channels else (17, 23)
    pixels = rng.integers(20, 206, size=shape, dtype=np.uint8)
    if dtype == np.bool_:
        return pixels > 112
    if dtype == np.uint8:
        return pixels
    return pixels.astype(dtype) / 255


@pytest.mark.parametrize(('dtype', 'options'), [
    (np.float32, {}),
    (np.float64, {'center': (8.19, 12.71), 'radius': 9.25, 'output_shape': (31, 23)}),
    (np.float64, {'scaling': 'log', 'center': (8.13, 12.27), 'radius': 12.5,
                  'output_shape': (32, 17), 'mode': 'reflect', 'order': 2}),
    (np.float32, {'scaling': 'log', 'radius': 8.5, 'output_shape': (31, 23),
                  'mode': 'wrap', 'order': 3}),
    (np.uint8, {'center': (8.13, 12.27), 'output_shape': (35, 19), 'mode': 'symmetric'}),
    (np.uint8, {'center': (7.12, 10.19), 'output_shape': (33, 21), 'mode': 'edge',
                'preserve_range': True, 'clip': False}),
    (np.bool_, {'scaling': 'log', 'center': (8.13, 12.27), 'radius': 10.5,
                'output_shape': (32, 17), 'cval': 1}),
])
def test_warp_polar_strict_hot_swap(dtype, options, monkeypatch):
    assert skimage.__version__ == '0.26.0'
    native, stock = xx.transform.warp_polar, original.warp_polar
    assert native is _warp_polar.warp_polar
    assert inspect.signature(native) == inspect.signature(stock)
    assert inspect.isbuiltin(xx._core.warp_sample)
    assert xx._core.warp_sample.__module__ == 'skimagexx._core'
    image = _image(dtype)
    before = image.tobytes()
    with warnings.catch_warnings(record=True) as reference_warnings:
        warnings.simplefilter('always')
        expected = stock(image, **options)

    def reject(*args, **kwargs):
        pytest.fail('Strict polar warp invoked upstream sampling')

    monkeypatch.setattr(original, 'warp_polar', reject)
    monkeypatch.setattr(_warps, 'warp', reject)
    monkeypatch.setattr(ndimage, 'map_coordinates', reject)
    previous = xx.set_fallback_enabled(False)
    try:
        with warnings.catch_warnings(record=True) as native_warnings:
            warnings.simplefilter('always')
            actual = native(image, **options)
    finally:
        xx.set_fallback_enabled(previous)
    assert [(w.category, str(w.message)) for w in native_warnings] == [
        (w.category, str(w.message)) for w in reference_warnings
    ]
    assert actual.dtype == expected.dtype
    assert actual.flags.owndata and actual.flags.c_contiguous and actual.flags.writeable
    assert not np.shares_memory(actual, image) and image.tobytes() == before
    if dtype == np.bool_:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('axis', [-1, 2, (0,), -2])
def test_warp_polar_color_axis_restored(axis):
    index = axis[0] if isinstance(axis, tuple) else axis
    image = np.moveaxis(_image(np.float32, channels=True), -1, index)
    options = dict(channel_axis=axis, scaling='log', mode='edge', order=1,
                   center=(8.12, 12.39), output_shape=(401, 307), radius=15)
    expected = original.warp_polar(image, **options)
    previous = xx.set_fallback_enabled(False)
    previous_workers = xx.get_num_threads()
    try:
        for workers in (1, 4) if xx._core.has_openmp else (1,):
            xx.set_num_threads(workers)
            actual = xx.transform.warp_polar(image, **options)
            assert actual.dtype == expected.dtype and actual.strides == expected.strides
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
            assert not np.shares_memory(actual, image)
    finally:
        xx.set_num_threads(previous_workers)
        xx.set_fallback_enabled(previous)


@pytest.mark.parametrize(('scaling', 'radius', 'radii'), [
    ('linear', 4, [0, 1, 2, 3]),
    ('log', 16, [1, 2, 4, 8]),
])
def test_warp_polar_angle_and_radius(scaling, radius, radii):
    image = np.arange(21 * 21, dtype=np.float64).reshape(21, 21)
    previous = xx.set_fallback_enabled(False)
    try:
        actual = xx.transform.warp_polar(image, (10, 10), radius=radius,
                                         output_shape=(4, 4), scaling=scaling, order=0)
    finally:
        xx.set_fallback_enabled(previous)
    # The angular endpoint is excluded. The first column represents radius
    # zero for linear sampling and radius one for log sampling.
    dr = np.array([0, 1, 0, -1])[:, None] * radii
    dc = np.array([1, 0, -1, 0])[:, None] * radii
    np.testing.assert_array_equal(actual, image[10 + dr, 10 + dc])


def test_warp_polar_forwards_unsupported_keyword_image(monkeypatch):
    image = _image(np.float32, channels=True)
    received = []

    def stock(*args, **kwargs):
        received.append((args, kwargs))
        return image

    monkeypatch.setattr(original, 'warp_polar', stock)
    previous = xx.set_fallback_enabled(False)
    try:
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            xx.transform.warp_polar(image=image, channel_axis=0)
        assert not received
        xx.set_fallback_enabled(True)
        with pytest.warns(xx.FallbackWarning, match='skimage.transform.warp_polar'):
            assert xx.transform.warp_polar(image=image, channel_axis=0) is image
        assert received == [((), {'image': image, 'channel_axis': 0})]
    finally:
        xx.set_fallback_enabled(previous)
