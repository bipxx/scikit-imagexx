"""Native in-place clipping: typed bounds, SIMD tails and runtime policy."""
import warnings
import numpy as np
import pytest
import skimagexx as xx

TYPES = (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32,
         np.int32, np.uint64, np.int64, np.float32, np.float64)
# Small metadata probe: avoid assuming a version boundary for runtime semantics.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    _replace_ties = not np.signbit(np.clip(np.array([-0.]), 0., 0.))[0]
    _ignore_nan = not np.isnan(np.clip(np.array([0.]), np.nan, np.nan)[0])
POLICY = 2 if not _replace_ties else 0 if _ignore_nan else 1


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


def data(dtype, size):
    rng = np.random.default_rng(2887)
    if dtype is np.bool_:
        return np.resize(np.array([0, 1, 2, 128, 255], np.uint8), size).view(np.bool_)
    if np.dtype(dtype).kind in 'iu':
        result = rng.bit_generator.random_raw(size).astype(dtype)
        if size: result[0] = np.iinfo(dtype).min
        if size > 1: result[-1] = np.iinfo(dtype).max
        return result
    result = rng.normal(size=size).astype(dtype)
    result[:min(size, 6)] = np.array([-np.inf, np.inf, np.nan, -0., 0., 1.], dtype)[:min(size, 6)]
    return result


def check(image, low, high, policy=POLICY):
    low, high = np.asarray(low, dtype=image.dtype), np.asarray(high, dtype=image.dtype)
    expected = image.copy()
    np.clip(expected, low, high, out=expected)
    address = image.ctypes.data
    actual = xx._core.resize_clip(image, low, high, policy)
    assert actual is image and actual.ctypes.data == address
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('size', [0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 33, 1031, 257*1025])
@pytest.mark.parametrize('bounds', ['full', 'middle', 'reversed', 'point'])
def test_typed_bounds_and_vector_tails(dtype, size, bounds):
    if dtype is np.bool_:
        low, high = {'full': (False, True), 'middle': (False, False),
                     'reversed': (True, False), 'point': (True, True)}[bounds]
    elif np.dtype(dtype).kind in 'iu':
        info = np.iinfo(dtype)
        low, high = {'full': (int(info.min)+3, int(info.max)-7),
                     'middle': (int(info.max)//2-3, int(info.max)//2+7),
                     'reversed': (int(info.max), int(info.min)),
                     'point': (int(info.max)-3, int(info.max)-3)}[bounds]
    else:
        low, high = {'full': (-np.inf, np.inf), 'middle': (-.3, .7),
                     'reversed': (1., -1.), 'point': (0., -0.)}[bounds]
    check(data(dtype, size), low, high)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size', [1, 3, 4, 7, 8, 9, 37, 1031, 257*1025])
@pytest.mark.parametrize('low,high', [(0., 0.), (-0., -0.), (-0., 0.), (0., -0.),
    (np.nan, 1.), (-1., np.nan), (np.nan, np.nan)])
def test_float_zero_ties_and_nan_bounds(dtype, size, low, high):
    check(np.resize(np.array([-0., 0., -1., 1., np.nan], dtype), size), low, high)


@pytest.mark.parametrize('dtype,bits,patterns', [
    (np.float32, np.uint32, [0x7fc00001, 0xffc00002]),
    (np.float64, np.uint64, [0x7ff8000000000001, 0xfff8000000000002])])
def test_nan_payloads(dtype, bits, patterns):
    nans = np.array(patterns, dtype=bits).view(dtype)
    for low, high in [(nans[1], 1.), (-1., nans[1]), (nans[1], nans[0]), (-1., 1.)]:
        check(np.resize(np.array([nans[0], -0., 0., 3.], dtype), 1031), low, high)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('policy', [0, 1, 2])
def test_explicit_policy(dtype, policy):
    image = np.resize(np.array([-0., 0.], dtype), 1031)
    bound = np.asarray(-0., dtype=dtype)
    xx._core.resize_clip(image, bound, bound, policy)
    assert np.signbit(image).all() if policy != 2 else np.array_equal(np.signbit(image), np.arange(1031)%2 == 0)
    bound_nan = np.asarray(np.nan, dtype=dtype)
    image = np.arange(1031, dtype=dtype)
    snapshot = image.tobytes()
    xx._core.resize_clip(image, bound_nan, bound_nan, policy)
    assert image.tobytes() == snapshot if policy == 0 else np.isnan(image).all()


def test_native_stage_has_no_python_image_calls(monkeypatch):
    image = data(np.uint64, 1031)
    low, high = np.asarray(2**63+1, np.uint64), np.asarray(2**64-17, np.uint64)
    expected = np.clip(image, low, high)
    def forbidden(*args, **kwargs):
        raise AssertionError('native clipping called a Python image operation')
    monkeypatch.setattr(np, 'clip', forbidden)
    monkeypatch.setattr(np, 'minimum', forbidden)
    monkeypatch.setattr(np, 'maximum', forbidden)
    actual = xx._core.resize_clip(image, low, high, POLICY)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('layout', ['reverse', 'fortran', 'readonly', 'endian', 'unaligned'])
def test_private_storage_guard(layout):
    image = np.arange(35, dtype=np.float64).reshape(5, 7)
    if layout == 'reverse': image = image[::-1]
    elif layout == 'fortran': image = np.asfortranarray(image)
    elif layout == 'readonly': image.flags.writeable = False
    elif layout == 'endian': image = image.astype('>f8')
    else: image = np.ndarray((5, 7), np.float64, buffer=bytearray(281), offset=1)
    with pytest.raises(ValueError):
        xx._core.resize_clip(image, np.asarray(0.), np.asarray(1.), POLICY)


def test_private_bounds_guard():
    image = np.arange(35, dtype=np.uint64)
    for low in [np.asarray(0.), np.array([], dtype=np.uint64),
                np.array([0, 1], dtype=np.uint64), np.array(1, dtype='>u8')]:
        with pytest.raises(TypeError):
            xx._core.resize_clip(image, low, np.asarray(2**64-1, np.uint64), POLICY)


def test_noncanonical_boolean_bounds():
    for low in [0, 1, 2, 128, 255]:
        for high in [0, 1, 2, 128, 255]:
            a = data(np.bool_, 1031)
            low = np.array([low], np.uint8).view(np.bool_).reshape(())
            high = np.array([high], np.uint8).view(np.bool_).reshape(())
            check(a, low, high)
