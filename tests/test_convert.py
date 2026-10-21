"""Dtype conversion parity: values, range rules, warnings, and copy identity."""
import warnings
import numpy as np
import pytest
from skimage import util as original
import skimagexx as xx

FUNCTIONS = ['img_as_float', 'img_as_float32', 'img_as_float64', 'img_as_ubyte',
             'img_as_uint', 'img_as_int', 'img_as_bool']
DTYPES = [np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
          np.uint64, np.int64, np.float32, np.float64]


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_num_threads(1)
    xx.set_fallback_enabled(previous)


def compare(name, image, force_copy=False):
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        try:
            expected = getattr(original, name)(image, force_copy)
        except (ValueError, TypeError) as error:
            with pytest.raises(type(error), match='zero-size|between -1 and 1|Cannot convert|ArrayType'):
                getattr(xx.util, name)(image, force_copy)
            return
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always')
        actual = getattr(xx.util, name)(image, force_copy)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [(w.category, str(w.message)) for w in expected_warnings]
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert type(actual) == type(expected)
    np.testing.assert_array_equal(actual, expected)
    if isinstance(image, np.ndarray):
        assert (actual is image) == (expected is image)
        assert np.shares_memory(actual, image) == np.shares_memory(expected, image)


def values(dtype):
    if dtype == np.bool_:
        return np.array([False, True, True, False], dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        bounds = np.iinfo(dtype)
        selected = [bounds.min, bounds.max, 0, 1, bounds.max//2, bounds.max//2+1]
        if bounds.min < 0:
            selected += [-1, -2, bounds.min+1]
        return np.array(selected, dtype=dtype)
    return np.array([-1, -0.999, -0.5, -0.0, 0, 0.25, 0.5, 0.999, 1], dtype=dtype)


@pytest.mark.parametrize('name', FUNCTIONS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('force_copy', [False, True])
def test_all_boundary_conversions(name, dtype, force_copy):
    # Repeat boundaries through SIMD blocks and scalar tails, not just tiny arrays.
    compare(name, np.tile(values(dtype), 65), force_copy)


@pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.uint16, np.int16])
@pytest.mark.parametrize('name', FUNCTIONS)
def test_exhaustive_small_integer_inputs(name, dtype):
    bounds = np.iinfo(dtype)
    compare(name, np.arange(int(bounds.min), int(bounds.max)+1, dtype=np.int64).astype(dtype))


@pytest.mark.parametrize('dtype', [np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64])
@pytest.mark.parametrize('name', ['img_as_ubyte', 'img_as_uint', 'img_as_int'])
def test_data_dependent_downcasting_and_negative_wrap(name, dtype):
    low = np.iinfo(dtype).min
    a = np.array([low, 0, 1, 42, 127], dtype=dtype)
    compare(name, np.tile(a, 65))
    if low < 0:
        compare(name, np.tile(np.array([low, low+1, -65535, -257, -1], dtype=np.int64).astype(dtype), 65))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('name', FUNCTIONS)
def test_float_special_values_and_range_errors(name, dtype):
    for a in [np.array([0, 1, -1, np.nan, np.inf, -np.inf], dtype=dtype),
              np.array([0, 2], dtype=dtype), np.array([-2, 0], dtype=dtype)]:
        compare(name, a)


@pytest.mark.parametrize('name', ['img_as_ubyte', 'img_as_uint', 'img_as_int'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_float_quantization_ties(name, dtype):
    scale = {'img_as_ubyte': 255, 'img_as_uint': 65535, 'img_as_int': 32767.5}[name]
    offset = 0.5 if name == 'img_as_int' else 0
    ties = (np.arange(-128, 256)+0.5+offset)/scale
    a = np.concatenate([ties, np.nextafter(ties, -np.inf), np.nextafter(ties, np.inf)])
    a = a[(a >= -1) & (a <= 1)].astype(dtype)
    compare(name, a)


@pytest.mark.parametrize('name', FUNCTIONS)
@pytest.mark.parametrize('shape', [(), (0,), (3, 0, 5), (2, 3, 5, 7)])
def test_dimensions_empty_and_lists(name, shape):
    a = np.zeros(shape, dtype=np.float32)
    compare(name, a)
    compare(name, [0, 1, 2])


@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float32, np.float64])
@pytest.mark.parametrize('name', FUNCTIONS)
def test_strided_unaligned_and_parallel(name, dtype):
    rng = np.random.default_rng(43)
    a = rng.integers(0, 128, (519, 521), dtype=np.int16).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        a /= dtype(127)
    compare(name, a)
    compare(name, a.T[::-1, ::2])
    storage = np.empty(a.nbytes+1, dtype=np.uint8)
    unaligned = np.ndarray(a.shape, dtype=dtype, buffer=storage, offset=1)
    unaligned[:] = a
    compare(name, unaligned)


@pytest.mark.parametrize('dtype', DTYPES+[np.float16])
@pytest.mark.parametrize('clip_negative', [False, True])
def test_dtype_limits(dtype, clip_negative):
    a = np.empty((0,), dtype=dtype)
    assert xx.util.dtype_limits(a, clip_negative) == original.dtype_limits(a, clip_negative)


def test_errstate_for_invalid_cast_and_float_overflow():
    for name, a, category in [('img_as_ubyte', np.array([np.nan]), 'invalid'),
                              ('img_as_float32', np.array([1e300]), 'over'),
                              ('img_as_float32', np.array([1e-300]), 'under')]:
        with np.errstate(**{category: 'raise'}):
            with pytest.raises(FloatingPointError):
                getattr(xx.util, name)(a)
            with pytest.raises(FloatingPointError):
                getattr(original, name)(a)


def test_identity_float16_and_byte_order():
    for a, name in [(np.arange(7, dtype=np.float16), 'img_as_float'),
                    (np.arange(7, dtype='>f4'), 'img_as_float32'),
                    (np.arange(7, dtype='>u2'), 'img_as_uint')]:
        compare(name, a)
        compare(name, a, True)


def test_noncanonical_boolean_storage():
    a = np.tile(np.array([0, 1, 2, 128, 255], dtype=np.uint8), 19).view(np.bool_)
    for name in FUNCTIONS:
        compare(name, a)


def test_float32_cast_rounding_boundaries_and_warnings():
    maximum = float(np.finfo(np.float32).max)
    threshold = maximum+2.0**103
    a = np.array([maximum, np.nextafter(maximum, np.inf), np.nextafter(threshold, 0), threshold,
                  1e300, 1e-300, float(np.nextafter(np.float32(0), np.float32(1))), -1e-40])
    compare('img_as_float32', a)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('name', ['img_as_ubyte', 'img_as_uint', 'img_as_int'])
def test_nan_suppresses_range_validation_but_keeps_cast_warnings(dtype, name):
    a = np.array([np.nan, np.finfo(dtype).max, -np.finfo(dtype).max, 0], dtype=dtype)
    compare(name, a)


def test_root_conversion_aliases_for_hot_swap():
    for name in FUNCTIONS:
        assert getattr(xx, name) is getattr(xx.util, name)
    def pipeline(backend, a):
        return backend.img_as_ubyte(backend.img_as_float32(a))
    import skimage
    a = np.arange(65536, dtype=np.uint16).reshape(256,256)
    np.testing.assert_array_equal(pipeline(xx,a), pipeline(skimage,a))


@pytest.mark.parametrize('dtype,target,raw', [(np.float32,'img_as_float64',0x7f800001),
                                            (np.float64,'img_as_float32',0x7ff0000000000001)])
def test_signaling_nan_float_cast_warnings(dtype, target, raw):
    integer = np.uint32 if dtype == np.float32 else np.uint64
    a = np.tile(np.array([raw], dtype=integer), 19).view(dtype)
    compare(target, a)


@pytest.mark.parametrize('dtype', DTYPES[:9])
@pytest.mark.parametrize('name', FUNCTIONS)
def test_full_width_random_integer_patterns(dtype, name):
    dtype = np.dtype(dtype)
    a = np.random.default_rng(88).integers(0, 256, 129*257*dtype.itemsize, dtype=np.uint8).view(dtype).reshape(129,257)
    if dtype.kind == 'b':
        a = a.view(np.uint8) > 127
    compare(name, a)
