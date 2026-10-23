"""Private native zoom stage; public resize/rescale remains test-first pending.

The oracle is SciPy's grid-mode zoom, the stage called by scikit-image resize.
These tests do not count the private stage as a completed scikit-image API.
"""
import numpy as np
import pytest
import scipy
from scipy import ndimage as ndi

import skimagexx as xx


MODES = [('reflect', 0), ('grid-constant', 1), ('nearest', 2), ('mirror', 3), ('grid-wrap', 4)]
LEGACY = np.lib.NumpyVersion(scipy.__version__) < '1.18.0'
IDENTITY = np.lib.NumpyVersion(scipy.__version__) >= '1.16.0'


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


def check(image, target, order, mode, cval=.3):
    mode, border = mode
    factors = [1/f for f in np.divide(image.shape, target)]
    expected = ndi.zoom(image, factors, order=order, mode=mode, cval=cval, grid_mode=True)
    snapshot = image.copy()
    identity = IDENTITY and all(z == 1 for z in factors)
    outputs = [xx._core.resize_zoom(image, expected.shape, order, border, cval, LEGACY, identity, simd)
               for simd in [True, False]]
    for result in outputs:
        assert result.dtype == expected.dtype and result.dtype.metadata == expected.dtype.metadata
        assert result.shape == expected.shape and result.strides == expected.strides
        assert not np.shares_memory(result, image)
        np.testing.assert_allclose(result, expected, rtol=3e-13, atol=3e-13, equal_nan=True)
    np.testing.assert_array_equal(outputs[0].view(np.uint64), outputs[1].view(np.uint64))
    np.testing.assert_array_equal(image, snapshot)
    return outputs[0], expected


@pytest.mark.parametrize('shape,target', [((7,), (11,)), ((5, 7), (3, 11)),
    ((3, 4, 5), (4, 3, 7)), ((2, 3, 4, 2), (3, 2, 3, 4))])
@pytest.mark.parametrize('order', range(6))
@pytest.mark.parametrize('mode', MODES)
def test_randomized_dimensions_orders_borders(shape, target, order, mode):
    image = np.random.default_rng(1811).normal(size=shape)
    check(image, target, order, mode)


@pytest.mark.parametrize('layout', ['fortran', 'reversed', 'stepped', 'broadcast',
                                  'readonly', 'unaligned', 'endian', 'tagged'])
@pytest.mark.parametrize('order', [0, 1, 3, 5])
def test_source_layouts(layout, order):
    image = np.random.default_rng(1813).normal(size=(5, 7))
    if layout == 'fortran': image = np.asfortranarray(image)
    elif layout == 'reversed': image = image[::-1, ::-1]
    elif layout == 'stepped': image = image[::2, ::2]
    elif layout == 'broadcast': image = np.broadcast_to(image[:1], image.shape)
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'endian': image = image.astype('>f8')
    elif layout == 'tagged': image = image.view(np.dtype('float64', metadata={'tag': 3}))
    else:
        source = image
        image = np.ndarray(source.shape, source.dtype, buffer=bytearray(source.nbytes+1), offset=1)
        image[...] = source
    check(image, (9, 3), order, ('grid-constant', 1))


@pytest.mark.parametrize('shape,target', [((1,), (7,)), ((1, 3), (5, 1)),
    ((3, 1), (1, 5)), ((1, 1, 2), (3, 2, 1))])
@pytest.mark.parametrize('order', range(6))
@pytest.mark.parametrize('mode', MODES)
def test_short_axes(shape, target, order, mode):
    image = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) - .7
    check(image, target, order, mode)


@pytest.mark.parametrize('order', range(6))
@pytest.mark.parametrize('mode', MODES)
def test_identity_and_signed_zero(order, mode):
    image = np.array([[-0., 0., -.3], [.7, -1., 3.]])
    actual, expected = check(image, image.shape, order, mode)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize('value', [0., -0., np.nan, np.inf, -np.inf, 1e100, 1e-100])
@pytest.mark.parametrize('order', [0, 1, 3])
def test_constant_and_exceptional_values(value, order):
    check(np.full((3, 5), value), (7, 2), order, ('mirror', 3))


def test_native_pixel_stage_has_no_python_image_calls(monkeypatch):
    image = np.random.default_rng(1817).normal(size=(3, 5, 7))
    expected = ndi.zoom(image, (2., .8, 1.3), order=5, mode='grid-constant', cval=.3, grid_mode=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('native pixel stage called a Python image algorithm')
    monkeypatch.setattr(ndi, 'zoom', forbidden)
    monkeypatch.setattr(ndi, 'spline_filter', forbidden)
    monkeypatch.setattr(np, 'pad', forbidden)
    actual = xx._core.resize_zoom(image, expected.shape, 5, 1, .3, LEGACY, False, True)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize('shape,target', [((9,), (5,)), ((5, 7), (3, 11)),
    ((3, 4, 5), (4, 3, 7)), ((2, 3, 4, 2), (3, 2, 3, 4))])
@pytest.mark.parametrize('order', range(6))
@pytest.mark.parametrize('mode,gaussian_mode,border', [('constant', 'constant', 1),
    ('edge', 'nearest', 2), ('reflect', 'mirror', 3), ('symmetric', 'reflect', 0),
    ('wrap', 'wrap', 4)])
def test_float64_resize_gaussian_and_zoom_stages(shape, target, order, mode, gaussian_mode, border):
    # This verifies stage composition with the original resize's default AA;
    # public dtype conversion and clipping remain separate pending work.
    from skimage.transform import resize
    image = np.random.default_rng(1823).normal(size=shape)
    sigma = np.maximum(0, (np.divide(shape, target)-1)/2)
    filtered = xx.filters.gaussian(image, sigma, mode=gaussian_mode, cval=.3, preserve_range=True)
    actual = xx._core.resize_zoom(filtered, target, order, border, .3, LEGACY, False, True)
    expected = resize(image, target, order=order, mode=mode, cval=.3, preserve_range=True, clip=False)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
