import inspect

import numpy as np
import pytest
from skimage.segmentation import quickshift as original
from skimagexx import FallbackWarning, _core, segmentation, set_fallback_enabled


def test_signature():
    assert inspect.signature(segmentation.quickshift) == inspect.signature(original)


@pytest.mark.parametrize(
    'shape,dtype,options',
    [
        ((27, 33, 3), 'uint8', {}),
        ((20, 23, 3), 'float32', {}),
        ((27, 19), 'float64', {'convert2lab': False, 'kernel_size': 2.5, 'rng': 7}),
        ((3, 22, 25), 'float32', {
            'convert2lab': False, 'channel_axis': 0, 'ratio': 0.45, 'kernel_size': 3.1,
            'max_dist': 3.25, 'sigma': 0.85,
        }),
        ((21, 22, 2), 'float64', {'convert2lab': False, 'kernel_size': 1, 'max_dist': 0}),
    ],
)
def test_native(shape, dtype, options):
    random = np.random.default_rng(924)
    image = (random.integers(0, 255, shape, dtype=np.uint8) if dtype == 'uint8'
             else random.random(shape).astype(np.dtype(dtype).newbyteorder('=')))
    previous = set_fallback_enabled(False)
    try:
        result = segmentation.quickshift(image, **options)
    finally:
        set_fallback_enabled(previous)
    expected = original(image, **options)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_density_ties():
    image = np.ones((44, 45), dtype=np.float32)
    options = {'convert2lab': False, 'rng': 19, 'kernel_size': 2.5, 'max_dist': 7}
    previous = set_fallback_enabled(False)
    try:
        result = segmentation.quickshift(image, **options)
    finally:
        set_fallback_enabled(previous)
    np.testing.assert_array_equal(result, original(image, **options))


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize('variant', ['metadata', 'strided', 'readonly', 'subclass', 'longdouble', 'tree'])
def test_fallback_routes(variant):
    image = np.random.default_rng(55).random((10, 12, 3))
    if variant == 'metadata':
        image = image.view(np.dtype('f8', metadata={'units': 'test'}))
    elif variant == 'strided':
        image = image[:, ::2]
    elif variant == 'readonly':
        image.flags.writeable = False
    elif variant == 'subclass':
        image = image.view(ArraySubclass)
    elif variant == 'longdouble':
        image = image.astype(np.longdouble)
    options = {'return_tree': variant == 'tree', 'kernel_size': 1, 'convert2lab': False}
    with pytest.warns(FallbackWarning, match='skimage.segmentation.quickshift'):
        result = segmentation.quickshift(image, **options)
    expected = original(image, **options)
    if variant == 'tree':
        for value, reference in zip(result, expected, strict=True):
            assert type(value) is type(reference)
            np.testing.assert_array_equal(value, reference)
    else:
        np.testing.assert_array_equal(result, expected)


def test_native_descriptor_and_storage_guard():
    image = np.zeros((8, 9, 3))
    noise = np.random.default_rng(42).normal(scale=0.00001, size=image.shape[:2])
    alias = image.view(np.dtype('=f8').newbyteorder('='))
    np.testing.assert_array_equal(_core.quickshift_prepared(alias, noise, 5, 10), original(image))
    for invalid in (
        image.view(np.dtype('f8', metadata={'test': True})),
        image.view(ArraySubclass),
        image.astype(np.longdouble),
    ):
        with pytest.raises(TypeError):
            _core.quickshift_prepared(invalid, noise, 5, 10)
    unsafe_root = np.lib.stride_tricks.as_strided(image, shape=image.shape)
    with pytest.raises(ValueError, match='storage'):
        _core.quickshift_prepared(unsafe_root, noise, 5, 10)
