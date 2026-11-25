import inspect

import numpy as np
import pytest
from skimage.segmentation import felzenszwalb as original
from skimagexx import FallbackWarning, _core, segmentation, set_fallback_enabled


def test_signature():
    assert inspect.signature(segmentation.felzenszwalb) == inspect.signature(original)


@pytest.mark.parametrize(
    'shape,dtype,options',
    [
        ((37, 31, 3), 'uint8', {}),
        ((32, 29), 'float32', {'sigma': 0, 'scale': 14.5, 'min_size': 5, 'channel_axis': None}),
        ((3, 18, 21), 'float64', {'sigma': 1.5, 'scale': 40, 'channel_axis': 0}),
        ((1, 15), 'float64', {'min_size': 1}),
        ((12, 1), 'float64', {'min_size': 2}),
        ((1, 1), 'float64', {}),
        ((20, 23, 3), 'float64', {'sigma': 0, 'scale': 31.25, 'min_size': 7}),
    ],
)
def test_native(shape, dtype, options):
    random = np.random.default_rng(918)
    image = random.integers(0, 5, shape).astype(np.dtype(dtype).newbyteorder('='))
    previous = set_fallback_enabled(False)
    try:
        result = segmentation.felzenszwalb(image, **options)
    finally:
        set_fallback_enabled(previous)
    expected = original(image, **options)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize('variant', ['metadata', 'strided', 'readonly', 'subclass', 'longdouble'])
def test_fallback_routes(variant):
    image = np.arange(180, dtype=np.float64).reshape(12, 15) / 180
    if variant == 'metadata':
        image = image.view(np.dtype('f8', metadata={'units': 'test'}))
    elif variant == 'strided':
        image = image[:, ::2]
    elif variant == 'readonly':
        image.flags.writeable = False
    elif variant == 'subclass':
        image = image.view(ArraySubclass)
    else:
        image = image.astype(np.longdouble)
    previous = set_fallback_enabled(True)
    try:
        try:
            expected = original(image)
        except ValueError as error:
            assert variant == 'longdouble'
            with pytest.warns(FallbackWarning, match='skimage.segmentation.felzenszwalb'):
                with pytest.raises(ValueError) as caught:
                    segmentation.felzenszwalb(image)
            assert str(caught.value) == str(error)
        else:
            with pytest.warns(FallbackWarning, match='skimage.segmentation.felzenszwalb'):
                result = segmentation.felzenszwalb(image)
            np.testing.assert_array_equal(result, expected)
    finally:
        set_fallback_enabled(previous)


def test_native_descriptor_and_storage_guard():
    image = np.arange(60, dtype=np.float64).reshape(4, 5, 3)
    alias = image.view(np.dtype('=f8').newbyteorder('='))
    np.testing.assert_array_equal(_core.felzenszwalb_prepared(alias, 1, 2),
                                  original(image, sigma=0, min_size=2))
    for invalid in (
        image.view(np.dtype('f8', metadata={'test': True})),
        image.view(ArraySubclass),
        image.astype(np.longdouble),
    ):
        with pytest.raises(TypeError):
            _core.felzenszwalb_prepared(invalid, 1, 2)
    unsafe_root = np.lib.stride_tricks.as_strided(image, shape=image.shape)
    with pytest.raises(ValueError, match='storage'):
        _core.felzenszwalb_prepared(unsafe_root, 1, 2)
