"""Typed native sampling stage; public resize/rescale is still pending."""
import numpy as np
import pytest
import scipy
from scipy import ndimage as ndi
import skimagexx as xx

TYPES = (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
         np.int32, np.uint64, np.int64, np.float32, np.float64)
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


def sample(dtype, shape):
    rng = np.random.default_rng(1831)
    if np.dtype(dtype).kind == 'f':
        image = rng.normal(size=shape).astype(dtype)
        image.flat[0] = -0.
        return image
    raw = rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
    image = (raw & 1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)
    if dtype is not np.bool_:
        image.flat[0] = np.iinfo(dtype).min
        image.flat[-1] = np.iinfo(dtype).max
    return image


def check(image, target, order, mode, cval=.3):
    mode, border = mode
    factors = [1/f for f in np.divide(image.shape, target)]
    expected = ndi.zoom(image, factors, order=order, mode=mode, cval=cval, grid_mode=True)
    snapshot = image.tobytes()
    identity = IDENTITY and all(z == 1 for z in factors)
    results = [xx._core.resize_zoom_typed(image, expected.shape, order, border, cval, LEGACY, identity, simd)
               for simd in [True, False]]
    for result in results:
        assert result.dtype == expected.dtype and result.dtype.metadata == expected.dtype.metadata
        assert result.shape == expected.shape and result.strides == expected.strides
        assert not np.shares_memory(result, image)
        if image.dtype.kind in 'biu':
            assert result.tobytes() == expected.tobytes()
        else:
            tolerance = 3e-6 if result.dtype == np.float32 else 3e-13
            np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
            zeros = (result == 0) & (expected == 0)
            np.testing.assert_array_equal(np.signbit(result[zeros]), np.signbit(expected[zeros]))
    assert results[0].tobytes() == results[1].tobytes()
    assert image.tobytes() == snapshot
    return results[0], expected


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('shape,target', [((7,), (11,)), ((5, 7), (3, 11)), ((3, 4, 5), (4, 3, 7))])
def test_nearest_all_storage(dtype, mode, shape, target):
    check(sample(dtype, shape), target, 0, mode)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('shape,target', [((7,), (11,)), ((5, 7), (3, 11)), ((3, 4, 5), (4, 3, 7))])
def test_floating_interpolation(dtype, order, mode, shape, target):
    check(sample(dtype, shape), target, order, mode)


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('layout', ['fortran', 'reversed', 'stepped', 'broadcast',
                                  'readonly', 'unaligned', 'endian', 'tagged'])
def test_storage_normalization(dtype, layout):
    image = sample(dtype, (5, 7))
    if layout == 'fortran': image = np.asfortranarray(image)
    elif layout == 'reversed': image = image[::-1, ::-1]
    elif layout == 'stepped': image = image[::2, ::2]
    elif layout == 'broadcast': image = np.broadcast_to(image[:1], image.shape)
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'endian': image = image.astype(image.dtype.newbyteorder('>'))
    elif layout == 'tagged': image = image.view(np.dtype(image.dtype, metadata={'tag': 3}))
    else:
        source = image
        image = np.ndarray(source.shape, source.dtype, buffer=bytearray(source.nbytes+1), offset=1)
        image[...] = source
    check(image, (9, 3), 3 if image.dtype.kind == 'f' else 0, ('grid-constant', 1))


@pytest.mark.parametrize('dtype', TYPES)
def test_identity(dtype):
    image = sample(dtype, (5, 7))
    check(image, image.shape, 0, ('mirror', 3))


@pytest.mark.parametrize('dtype', TYPES)
def test_empty_output(dtype):
    image = sample(dtype, (5, 7))
    expected = ndi.zoom(image, (0., 1.), order=0, grid_mode=True, mode='mirror')
    actual = xx._core.resize_zoom_typed(image, (0, 7), 0, 3, .3, LEGACY, False, True)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape and actual.strides == expected.strides


@pytest.mark.parametrize('dtype', [np.uint64, np.int64])
def test_full_width_edges(dtype):
    if dtype is np.uint64:
        values = [0, 2**53-1, 2**53+1, 2**63-1, 2**63+4095, 2**64-1]
    else:
        values = [-2**63, -2**53-1, -2**53+1, -1, 0, 2**53-1, 2**53+1, 2**63-1]
    image = np.array(values, dtype=dtype)
    for length in [1, 3, 4, 5, 17, 1027]:
        check(image, (length,), 0, ('mirror', 3))


@pytest.mark.parametrize('layout', ['contiguous', 'reverse', 'broadcast'])
def test_noncanonical_boolean_storage(layout):
    image = np.array([0, 2, 128, 255, 1], dtype=np.uint8).view(np.bool_)
    if layout == 'reverse': image = image[::-1]
    elif layout == 'broadcast': image = np.broadcast_to(image[1:2], (5,))
    check(image, (17,), 0, ('mirror', 3))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('order', [0, 1, 3])
@pytest.mark.parametrize('value', [0., -0., np.nan, np.inf, -np.inf, 1e-30, 1e30])
def test_floating_constants(dtype, order, value):
    check(np.full((3, 5), value, dtype=dtype), (7, 2), order, ('mirror', 3))


@pytest.mark.parametrize('dtype', [np.float32, np.uint64])
@pytest.mark.parametrize('layout', ['reversed', 'fortran', 'unaligned', 'endian-reversed'])
def test_large_storage_normalization(dtype, layout):
    image = sample(dtype, (513, 1025))
    if layout == 'reversed': image = image[::-1, ::-1]
    elif layout == 'fortran': image = np.asfortranarray(image)
    elif layout == 'endian-reversed': image = image.astype(image.dtype.newbyteorder('>'))[::-1, ::-1]
    else:
        source = image
        image = np.ndarray(source.shape, source.dtype, buffer=bytearray(source.nbytes+1), offset=1)
        image[...] = source
    check(image, (257, 513), 0, ('mirror', 3))


@pytest.mark.parametrize('dtype', [np.uint64, np.float32])
def test_native_normalization_has_no_python_image_calls(dtype, monkeypatch):
    image = sample(dtype, (5, 7)).astype(np.dtype(dtype).newbyteorder('>'))[::-1, ::-1]
    order = 3 if dtype is np.float32 else 0
    expected = ndi.zoom(image, (1.8, 3/7), order=order, mode='grid-constant', cval=.3, grid_mode=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('native normalization called a Python image algorithm')
    monkeypatch.setattr(np, 'ascontiguousarray', forbidden)
    monkeypatch.setattr(np, 'pad', forbidden)
    monkeypatch.setattr(ndi, 'zoom', forbidden)
    monkeypatch.setattr(ndi, 'spline_filter', forbidden)
    actual = xx._core.resize_zoom_typed(image, expected.shape, order, 1, .3, LEGACY, False, True)
    if dtype is np.uint64: assert actual.tobytes() == expected.tobytes()
    else: np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
