"""Raw integer AA stage: ndimage stores/truncates after each axis."""
import numpy as np
import pytest
import scipy
from scipy import ndimage as ndi
from skimage.transform import resize
import skimagexx as xx
from skimagexx.filters import _gaussian_coefficients

TYPES = (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
         np.float32, np.float64)
MODES = {'reflect': 0, 'constant': 1, 'nearest': 2, 'mirror': 3, 'wrap': 4}
LEGACY = np.lib.NumpyVersion(scipy.__version__) < '1.18.0'


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


def image(dtype, shape=(17, 19)):
    rng = np.random.default_rng(1847)
    if np.dtype(dtype).kind == 'f': return rng.normal(size=shape).astype(dtype)
    result = rng.bit_generator.random_raw(np.prod(shape)).reshape(shape).astype(dtype)
    if result.size:
        result.flat[0] = np.iinfo(dtype).min
        result.flat[-1] = np.iinfo(dtype).max
    return result


def native(a, sigma, mode='reflect', cval=0):
    sigma = list(sigma) if np.iterable(sigma) else [sigma]*a.ndim
    modes = list(mode) if not isinstance(mode, str) else [mode]*a.ndim
    kernels, axes, codes = [], [], []
    for axis, sd in enumerate(sigma):
        if sd > 1e-15:
            axes.append(axis); codes.append(MODES[modes[axis]])
            kernels.append(_gaussian_coefficients(sd, int(4*float(sd)+.5)))
    return xx._core.resize_gaussian_raw(a, kernels, axes, codes, cval)


def check(a, sigma, mode='reflect', cval=0):
    snapshot = a.tobytes()
    expected = ndi.gaussian_filter(a, sigma, mode=mode, cval=cval)
    actual = native(a, sigma, mode, cval)
    assert actual.dtype == expected.dtype and actual.dtype.metadata == expected.dtype.metadata
    assert actual.shape == expected.shape and actual.strides == expected.strides
    assert not np.shares_memory(a, actual)
    if a.dtype.kind in 'iu': assert actual.tobytes() == expected.tobytes()
    else:
        tolerance = 3e-6 if a.dtype.type is np.float32 else 3e-13
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
    assert a.tobytes() == snapshot
    return actual, expected


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('shape,sigma', [((37,), .7), ((17, 19), (.7, 1.3)),
    ((3, 5, 7), (.3, .7, 1.1)), ((2, 3, 4, 5), (0, .7, 0, 1.3))])
def test_types_axes_borders(dtype, mode, shape, sigma):
    check(image(dtype, shape), sigma, mode, .3)


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('sigma', [0, 1e-16, (.7, 0), (0, .7), (.01, .7)])
def test_skipped_and_zero_radius_axes(dtype, sigma):
    check(image(dtype), sigma, 'constant', -.3)


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('layout', ['fortran', 'reverse', 'step', 'broadcast', 'readonly', 'endian', 'unaligned', 'tagged'])
def test_layouts(dtype, layout):
    a = image(dtype)
    if layout == 'fortran': a = np.asfortranarray(a)
    elif layout == 'reverse': a = a[::-1, ::-1]
    elif layout == 'step': a = a[::2, ::2]
    elif layout == 'broadcast': a = np.broadcast_to(a[:1], a.shape)
    elif layout == 'readonly': a.flags.writeable = False
    elif layout == 'endian': a = a.astype(a.dtype.newbyteorder('>'))
    elif layout == 'tagged': a = a.view(np.dtype(a.dtype, metadata={'tag': 3}))
    else:
        data = a
        a = np.ndarray(data.shape, data.dtype, buffer=bytearray(data.nbytes+1), offset=1)
        a[...] = data
    check(a, (.7, 1.3), 'constant', .3)


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('cval', [0., -.3, 255., -300.9, 1e30, np.nan, np.inf, -np.inf])
def test_constant_values(dtype, cval):
    check(image(dtype, (5, 17)), (.7, 1.3), 'constant', cval)


@pytest.mark.parametrize('dtype', TYPES)
def test_per_axis_modes(dtype):
    check(image(dtype, (3, 5, 7)), (.7, 1.3, .3), ('mirror', 'constant', 'nearest'), -.3)


@pytest.mark.parametrize('dtype', TYPES)
def test_zero_sized_axis(dtype):
    check(image(dtype, (0, 7)), (.7, 1.3), 'wrap')


@pytest.mark.parametrize('dtype', TYPES)
def test_parallel_sized_reference(dtype):
    # Cross the worker threshold and SIMD tails; check against SciPy, rather
    # than only comparing two executions of the native implementation.
    check(image(dtype, (257, 1025)), (.7, 1.3), 'constant', -.3)


@pytest.mark.parametrize('dtype', TYPES[:8])
@pytest.mark.parametrize('mode', MODES)
def test_integer_antialiasing_and_nearest_pipeline(dtype, mode):
    a = image(dtype, (17, 19)); target = (7, 11)
    sigma = np.maximum(0, (np.divide(a.shape, target)-1)/2)
    filtered = native(a, sigma, mode, .3)
    code = MODES[mode]
    actual = xx._core.resize_zoom_typed(filtered, target, 0, code, .3, LEGACY, False, True)
    pad_mode = {'reflect': 'symmetric', 'mirror': 'reflect', 'nearest': 'edge', 'wrap': 'wrap', 'constant': 'constant'}[mode]
    expected = resize(a, target, order=0, mode=pad_mode, cval=.3, anti_aliasing=True, clip=False)
    assert actual.tobytes() == expected.tobytes()


def test_native_stage_has_no_python_image_calls(monkeypatch):
    a = image(np.uint64, (17, 19))
    expected = ndi.gaussian_filter(a, (.7, 1.3), mode='mirror')
    def forbidden(*args, **kwargs):
        raise AssertionError('native raw Gaussian called a Python image algorithm')
    monkeypatch.setattr(ndi, 'gaussian_filter', forbidden)
    monkeypatch.setattr(ndi, 'gaussian_filter1d', forbidden)
    monkeypatch.setattr(ndi, 'correlate1d', forbidden)
    actual = native(a, (.7, 1.3), 'mirror')
    assert actual.tobytes() == expected.tobytes()
