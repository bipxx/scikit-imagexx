import inspect
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from skimage import segmentation as original

import skimagexx as xx
from skimagexx import _core
from skimagexx._compat import FallbackWarning
from skimagexx._slic import slic


@pytest.fixture
def strict():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)


def frame(shape, dtype):
    random = np.random.default_rng(982017)
    pixels = random.integers(0, 256, size=shape, dtype=np.uint8)
    if np.dtype(dtype).kind == 'f':
        return pixels.astype(dtype) / 19 - 3
    return pixels.astype(dtype)


@pytest.mark.parametrize('shape,dtype,options', [
    ((23, 35), np.uint8, dict(channel_axis=None)),
    ((25, 17), np.float32, dict(channel_axis=None, n_segments=17, compactness=0.25,
                               max_num_iter=5, start_label=0, slic_zero=True, spacing=(2, 0.75))),
    ((21, 41), np.float64, dict(channel_axis=None, n_segments=11, compactness=6,
                               enforce_connectivity=False, max_num_iter=4)),
    ((12, 17, 3), np.uint8, {}),
    ((13, 17, 3), np.float32, dict(n_segments=14, slic_zero=True, sigma=1.0,
                                  spacing=[1.5, 0.5], start_label=0)),
    ((20, 15, 2), np.float64, dict(n_segments=9, convert2lab=False, sigma=(0.5, 1.25))),
    ((5, 11, 13), np.float32, dict(channel_axis=None, n_segments=25, spacing=(2, 1, 0.8),
                                  min_size_factor=0.4, max_size_factor=1.2)),
    ((3, 9, 10, 3), np.float64, dict(n_segments=14, compactness=7, sigma=(0, 1, 1))),
    ((2, 3), np.float64, dict(channel_axis=None, max_num_iter=0, start_label=0)),
    ((3, 1), np.uint8, dict(channel_axis=None, n_segments=1, max_size_factor=50)),
])
def test_slic_matches_original_and_enters_native(shape, dtype, options, strict, monkeypatch):
    image = frame(shape, dtype)
    before = image.copy()
    entry = _core.slic_grid
    calls = []

    def observe(*args, **kwargs):
        calls.append((args[0], kwargs.get('prepared', False)))
        return entry(*args, **kwargs)

    monkeypatch.setattr(_core, 'slic_grid', observe)
    expected = original.slic(image, **options)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = slic(image, **options)
    assert len(calls) == 1
    assert (calls[0][0] is image) == (not calls[0][1])
    assert type(actual) is np.ndarray
    assert actual.dtype == expected.dtype == np.dtype(np.intp)
    assert actual.strides == expected.strides
    assert actual.flags.c_contiguous and actual.flags.writeable
    assert actual.flags.owndata == expected.flags.owndata
    assert not np.shares_memory(actual, image)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('zero', [False, True])
def test_slic_ties_and_workers(zero, strict):
    before = xx.get_num_threads()
    try:
        for shape in ((4, 4), (1, 13), (3, 7, 7), (1, 8, 8)):
            image = np.full(shape, 7., dtype=np.float32)
            for count in (1, 3, 16, 101):
                options = dict(n_segments=count, channel_axis=None, slic_zero=zero, max_num_iter=3)
                expected = original.slic(image, **options)
                for workers in (1, 2):
                    xx.set_num_threads(workers)
                    np.testing.assert_array_equal(slic(image, **options), expected)
    finally:
        xx.set_num_threads(before)


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize('kind', ['metadata', 'longdouble', 'swapped', 'strided', 'readonly',
                                  'subclass', 'mask', 'axis'])
def test_slic_boundary_uses_public_policy(kind, strict):
    image = frame((10, 12), np.float64)
    options = dict(channel_axis=None, n_segments=6)
    if kind == 'metadata':
        image = image.view(np.dtype(np.float64, metadata={'unit': 'pixel'}))
    elif kind == 'longdouble':
        image = image.astype(np.longdouble)
    elif kind == 'swapped':
        image = image.astype(image.dtype.newbyteorder('S'))
    elif kind == 'strided':
        image = image[:, ::-1]
    elif kind == 'readonly':
        image.flags.writeable = False
    elif kind == 'subclass':
        image = image.view(ArraySubclass)
    elif kind == 'mask':
        options['mask'] = image > 1.
    else:
        image = frame((3, 10, 12), np.uint8)
        options['channel_axis'] = 0
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        slic(image, **options)
    xx.set_fallback_enabled(True)
    with pytest.warns(FallbackWarning, match='skimage.segmentation.slic'):
        actual = slic(image, **options)
    np.testing.assert_array_equal(actual, original.slic(image, **options))


def test_slic_bound_entry_rejects_tagged_and_non_native_storage():
    pixels = frame((7, 9), np.float32)
    tagged = pixels.view(np.dtype(np.float32, metadata={'unit': 'pixel'}))
    for refused in (tagged, pixels[:, ::-1], pixels.view(ArraySubclass)):
        assert not _core.slic_accept_image(refused, 2)
        assert _core.slic_grid(refused, 2, 5, 10., 3, (1., 1., 1.), True, .5, 3., False, 1) is None
    assert _core.slic_accept_image(pixels.view(np.dtype('=f4')), 2)


def test_slic_export_signature_and_backend_swap(strict):
    exports = {}
    exec('from skimagexx.segmentation import *', exports)
    assert exports['slic'] is xx.segmentation.slic is slic
    assert inspect.signature(slic) == inspect.signature(original.slic)
    assert inspect.isbuiltin(_core.slic_grid)
    assert _core.slic_grid.__module__ == 'skimagexx._core'
    pixels = frame((13, 17, 3), np.uint8)

    def pipeline(backend):
        return backend.segmentation.slic(pixels, n_segments=9)

    marker = object()
    substitute = SimpleNamespace(segmentation=SimpleNamespace(slic=lambda *_a, **_kw: marker))
    assert pipeline(substitute) is marker
    np.testing.assert_array_equal(pipeline(xx), pipeline(SimpleNamespace(segmentation=original)))
