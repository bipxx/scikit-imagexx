"""Test-first resize/rescale contract against pinned scikit-image.

Compatibility cases exercise the explicit fallback today. Strict native cases
are expected failures only while the API is absent from the native registry;
they must pass without fallback when the port is registered. Pending cases do
not contribute to native coverage or performance claims.
"""
import inspect
import warnings

import numpy as np
import pytest
from skimage import transform as reference

import skimagexx as xx
from skimagexx.coverage import native_functions


TYPES = (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
         np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64)
MODES = ('constant', 'edge', 'reflect', 'symmetric', 'wrap')


def sample(dtype, shape=(3, 5)):
    rng = np.random.default_rng(3217)
    if np.dtype(dtype).kind == 'f':
        return rng.uniform(-.7, 1.3, shape).astype(dtype)
    raw = rng.bit_generator.random_raw(np.prod(shape)).reshape(shape)
    return (raw & 1).astype(dtype) if dtype is np.bool_ else raw.astype(dtype)


def observe(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            result, error = function(*args, **kwargs), None
        except Exception as caught:
            result, error = None, caught
    return result, error, [(type(w.message), str(w.message)) for w in notices]


@pytest.fixture(params=[1, 4], autouse=True)
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


@pytest.fixture(params=['compatibility', 'native'])
def backend(request):
    xx.set_fallback_enabled(request.param == 'compatibility')
    return request.param


def check(backend, name, image, target, **options):
    pending = 'transform.' + name not in native_functions()
    snapshot = image.copy()
    expected, expected_error, expected_notices = observe(
        getattr(reference, name), image, target, **options)
    actual, actual_error, actual_notices = observe(
        getattr(xx.transform, name), image, target, **options)
    if backend == 'native' and pending:
        # Execute pending cases: only the explicit strict-dispatch rejection is
        # expected. Any other error or unexpected success remains a failure.
        if isinstance(actual_error, NotImplementedError):
            assert 'fallback is disabled' in str(actual_error)
            pytest.xfail('native transform.' + name + ' is not implemented')
        pytest.fail('unregistered API unexpectedly executed in strict mode')
    fallback_notices = [w for w in actual_notices if issubclass(w[0], xx.FallbackWarning)]
    assert len(fallback_notices) == (1 if backend == 'compatibility' and pending else 0)
    actual_notices = [w for w in actual_notices if not issubclass(w[0], xx.FallbackWarning)]
    assert actual_notices == expected_notices
    np.testing.assert_array_equal(image, snapshot)
    if expected_error is not None:
        assert type(actual_error) is type(expected_error)
        assert str(actual_error) == str(expected_error)
        return
    if actual_error is not None:
        raise actual_error
    assert type(actual) is type(expected)
    assert actual.shape == expected.shape and actual.strides == expected.strides
    assert actual.dtype == expected.dtype and actual.dtype.metadata == expected.dtype.metadata
    assert not np.shares_memory(image, actual)
    if actual.dtype.kind in 'biu' or options.get('order') == 0:
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = 3e-6 if actual.dtype == np.float32 else 3e-13
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)


@pytest.mark.parametrize('name', ['resize', 'rescale'])
def test_signature(name):
    assert inspect.signature(getattr(xx.transform, name)) == inspect.signature(getattr(reference, name))


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('order', [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('target', [(2, 3), (5, 7)])
def test_resize_types_orders_borders(backend, dtype, order, mode, target):
    check(backend, 'resize', sample(dtype), target, order=order, mode=mode, cval=.3)


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('layout', ['fortran', 'reversed', 'stepped', 'readonly',
                                  'unaligned', 'endian', 'broadcast', 'tagged'])
def test_resize_layouts(backend, dtype, layout):
    image = sample(dtype, (5, 7, 3))
    if layout == 'fortran': image = np.asfortranarray(image)
    elif layout == 'reversed': image = image[::-1, :, ::-1]
    elif layout == 'stepped': image = image[::2, ::2]
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'endian': image = image.astype(image.dtype.newbyteorder('>'))
    elif layout == 'broadcast': image = np.broadcast_to(image[:1], image.shape)
    elif layout == 'tagged': image = image.view(np.dtype(image.dtype, metadata={'tag': 3}))
    else:
        data = image
        image = np.ndarray(data.shape, data.dtype, buffer=bytearray(data.nbytes + 1), offset=1)
        image[...] = data
    check(backend, 'resize', image, (3, 9))


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('order', [None, 0, 1, 3])
@pytest.mark.parametrize('options', [dict(anti_aliasing=False), dict(anti_aliasing=True),
    dict(anti_aliasing=True, anti_aliasing_sigma=(.3, .7)), dict(preserve_range=True),
    dict(clip=False, mode='constant', cval=3)])
def test_resize_filter_and_range_options(backend, dtype, order, options):
    check(backend, 'resize', sample(dtype), (2, 3), order=order, **options)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize('order', [0, 1, 3, 5])
@pytest.mark.parametrize('shape,target', [((7,), (11,)), ((3, 5), (3, 5)),
    ((2, 3, 5), (3, 2, 7)), ((2, 3, 4, 5), (3, 2, 5, 4)),
    ((3, 5), (2.5, 3.5)), ((3, 5), (3, 5, 2)), ((3, 5, 2), (2, 4))])
def test_resize_geometry_and_identity(backend, dtype, order, shape, target):
    check(backend, 'resize', sample(dtype, shape), target, order=order, anti_aliasing=False)


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('scale', [.7, 1., 1.5, (.5, 1.3)])
@pytest.mark.parametrize('axis', [None, 0, -1, 1])
def test_rescale_channels(backend, dtype, scale, axis):
    shape = (5, 7) if axis is None else (3, 5, 7)
    check(backend, 'rescale', sample(dtype, shape), scale, channel_axis=axis)


@pytest.mark.parametrize('name,target,options', [
    ('resize', (5, 7), dict(order=-1)), ('resize', (5, 7), dict(order=6)),
    ('resize', (5, 7), dict(order=1.5)), ('resize', (5, 7), dict(mode='invalid')),
    ('resize', (), {}), ('resize', (-1, 4), {}), ('resize', (0, 4), {}),
    ('resize', (0, 4), dict(anti_aliasing=False)),
    ('resize', (5, 7), dict(anti_aliasing=True, anti_aliasing_sigma=-1)),
    ('resize', (5, 7), dict(anti_aliasing=False, anti_aliasing_sigma=-1)),
    ('resize', (5, 7), dict(anti_aliasing=True, anti_aliasing_sigma=1)),
    ('rescale', (.5, 1., 2.), {}), ('rescale', .7, dict(channel_axis=4)),
])
def test_validation_and_warning_order(backend, name, target, options):
    check(backend, name, sample(np.float64), target, **options)


@pytest.mark.parametrize('value', [0., 1., np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('order', [0, 1, 3])
def test_nonfinite_and_constant(backend, value, order):
    check(backend, 'resize', np.full((3, 5), value), (5, 7), order=order)


@pytest.mark.parametrize('order', [0, 1])
@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('clip', [False, True])
def test_wide_integer_conversion(backend, order, preserve, clip):
    image = np.array([2**53 + 1, 2**63 + 4095, 2**64 - 1], dtype=np.uint64)
    check(backend, 'resize', image, (9,), order=order, preserve_range=preserve,
          clip=clip, anti_aliasing=False)
