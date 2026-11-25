import inspect

import numpy as np
import pytest
import skimage
from skimage import restoration as reference

from skimagexx import _core, restoration
from skimagexx._compat import FallbackWarning, set_fallback_enabled


APIS = (
    'denoise_nl_means',
    'denoise_wavelet',
    'estimate_sigma',
    'inpaint_biharmonic',
    'unwrap_phase',
    'richardson_lucy',
)


@pytest.fixture
def native_only():
    previous = set_fallback_enabled(False)
    try:
        yield
    finally:
        set_fallback_enabled(previous)


@pytest.mark.parametrize('name', APIS)
def test_signature(name):
    assert skimage.__version__ == '0.26.0'
    assert inspect.signature(getattr(restoration, name)) == inspect.signature(getattr(reference, name))


def check(name, *args, rtol=1e-10, atol=1e-12, **options):
    expected = getattr(reference, name)(*args, **options)
    result = getattr(restoration, name)(*args, **options)
    if isinstance(expected, np.ndarray):
        assert result.dtype == expected.dtype
    else:
        assert type(result) is type(expected)
        if isinstance(expected, list):
            assert [type(value) for value in result] == [type(value) for value in expected]
    np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol, strict=True)
    return result


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape, options', [
    ((16, 19), {}),
    ((12, 13, 3), dict(patch_size=4, patch_distance=3, h=0.22, sigma=0.07, channel_axis=-1)),
    ((6, 8, 9), dict(patch_size=3, patch_distance=2, h=0.24, sigma=0.08)),
    ((5, 6, 7, 2), dict(patch_size=5, patch_distance=2, h=0.3, channel_axis=-1)),
    ((1, 8), dict(patch_size=1, patch_distance=2)),
])
def test_nonlocal_means(native_only, dtype, shape, options):
    image = np.random.default_rng(108).random(shape).astype(dtype)
    check('denoise_nl_means', image, **options, rtol=2e-6)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape, options', [
    ((31, 27), {}),
    ((33,), dict(wavelet='haar', wavelet_levels=2, sigma=0.12, method='VisuShrink')),
    ((25, 35), dict(wavelet='db2', wavelet_levels=2, sigma=0.1, mode='hard')),
    ((29, 22, 3), dict(wavelet='db2', sigma=0.2, method='VisuShrink', channel_axis=-1)),
    ((48, 50), dict(wavelet='haar', sigma=0.2, wavelet_levels=0)),
])
def test_wavelet(native_only, dtype, shape, options):
    image = np.random.default_rng(109).random(shape).astype(dtype)
    check('denoise_wavelet', image, **options, rtol=8e-6, atol=8e-7)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape, options', [
    ((23, 27), {}),
    ((34,), {}),
    ((32, 37, 3), dict(channel_axis=-1)),
    ((32, 37, 3), dict(channel_axis=2, average_sigmas=True)),
])
def test_estimate_sigma(native_only, dtype, shape, options):
    image = np.random.default_rng(110).random(shape).astype(dtype)
    check('estimate_sigma', image, **options, rtol=2e-6)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('channels', [False, True])
def test_inpaint(native_only, dtype, channels):
    rng = np.random.default_rng(111)
    mask = rng.random((31, 29)) < 0.08
    mask[:3, :4] = True
    mask[10:15, 16:20] = True
    image = rng.random((*mask.shape, 3) if channels else mask.shape).astype(dtype)
    original = image.copy()
    # The native factorization is binary64 while the float32 reference solves
    # with a binary32 sparse factorization. Compare with a dtype-specific bound.
    tol = 3e-6 if dtype is np.float32 else 1e-10
    result = check('inpaint_biharmonic', image, mask, channel_axis=-1 if channels else None,
                   rtol=tol, atol=tol)
    np.testing.assert_array_equal(result[~mask], original[~mask])
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(71,), (29, 32), (8, 10, 11)])
def test_unwrap(native_only, dtype, shape):
    coords = np.indices(shape, dtype=np.float64)
    phase = sum((axis + 1) * 0.27 * coord for axis, coord in enumerate(coords))
    image = ((phase + np.pi) % (2 * np.pi) - np.pi).astype(dtype)
    check('unwrap_phase', image, rtol=0, atol=1e-13)
    random_phase = np.random.default_rng(112).uniform(-np.pi, np.pi, shape).astype(dtype)
    check('unwrap_phase', random_phase, rtol=0, atol=1e-13)


def test_unwrap_integrable_field(native_only):
    # Check both repeated and distinct reliability values on a field large
    # enough to use the partial ordering when its gauge can be certified.
    y, x = np.indices((64, 67), dtype=np.float64)
    base = x * 0.16 + y * 0.22
    noise = np.random.default_rng(116).normal(0, 0.05, base.shape)
    for phase in (base, base + noise):
        image = (phase + np.pi) % (2 * np.pi) - np.pi
        check('unwrap_phase', image, rtol=0, atol=1e-13)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('psf_shape, options', [
    ((5, 5), {}),
    ((4, 7), dict(num_iter=4, clip=False, filter_epsilon=0.02)),
    ((3, 4), dict(num_iter=0)),
])
def test_richardson_lucy(native_only, dtype, psf_shape, options):
    rng = np.random.default_rng(113)
    image = (0.25 + 0.6 * rng.random((29, 27))).astype(dtype)
    psf = rng.random(psf_shape).astype(dtype)
    psf /= psf.sum()
    check('richardson_lucy', image, psf, **options, rtol=2e-5, atol=3e-7)


def test_worker_budget(native_only):
    rng = np.random.default_rng(114)
    image = (0.3 + 0.4 * rng.random((510, 511))).astype(np.float32)
    psf = np.full((7, 7), 1 / 49, dtype=np.float32)
    saved = _core.get_num_threads()
    try:
        _core.set_num_threads(1)
        a = check('richardson_lucy', image, psf, num_iter=2, rtol=6e-6, atol=2e-7)
        _core.set_num_threads(4)
        b = restoration.richardson_lucy(image, psf, num_iter=2)
        np.testing.assert_array_equal(a, b)
    finally:
        _core.set_num_threads(saved)


@pytest.mark.parametrize('name', APIS)
def test_layout_policy(name):
    class TaggedArray(np.ndarray):
        pass

    image = np.random.default_rng(115).random((12, 15))
    readonly = image.copy()
    readonly.setflags(write=False)
    alias = image.view(np.dtype('=f8'))
    extra = {
        'denoise_nl_means': (3, 1),
        'denoise_wavelet': (),
        'estimate_sigma': (),
        'inpaint_biharmonic': (np.zeros(image.shape, dtype=bool),),
        'unwrap_phase': (),
        'richardson_lucy': (np.full((3, 3), 1 / 9), 2),
    }[name]
    args = (alias, *extra)
    previous = set_fallback_enabled(False)
    try:
        check(name, *args)
    finally:
        set_fallback_enabled(previous)
    tagged = image.view(np.dtype('f8', metadata={'units': 'test'}))
    for other in (image.view(TaggedArray), readonly, image[:, ::-1], tagged,
                  image.astype('>f8'), image.astype(np.longdouble)):
        previous = set_fallback_enabled(False)
        try:
            with pytest.raises(NotImplementedError, match='fallback is disabled'):
                getattr(restoration, name)(other, *extra)
        finally:
            set_fallback_enabled(previous)
    with pytest.warns(FallbackWarning, match=name):
        if name == 'unwrap_phase':
            with pytest.raises(ValueError, match='read-only'):
                getattr(restoration, name)(readonly, *extra)
        else:
            getattr(restoration, name)(readonly, *extra)


def test_native_binding_descriptor_rejection():
    array = np.ones((5, 6))

    class ArraySubclass(np.ndarray):
        pass

    for unsupported in (array.astype(np.longdouble), array.view(ArraySubclass),
                        array.view(np.dtype('f8', metadata={'tag': True})), array[:, ::-1]):
        assert _core.restoration_unwrap(unsupported, True) is None
    np.testing.assert_array_equal(
        _core.restoration_unwrap(array.view(np.dtype('=f8')), True), array
    )


def test_secondary_input_descriptor_rejection(native_only):
    image = np.ones((7, 8))
    mask = np.zeros(image.shape, dtype=bool)
    psf = np.full((3, 3), 1 / 9)
    for name, other in (
        ('inpaint_biharmonic', mask.view(np.dtype('?', metadata={'tag': True}))),
        ('richardson_lucy', psf.astype(np.longdouble)),
        ('richardson_lucy', psf.view(np.dtype('f8', metadata={'tag': True}))),
    ):
        with pytest.raises(NotImplementedError, match='fallback is disabled'):
            getattr(restoration, name)(image, other)


@pytest.mark.parametrize('name, args, options', [
    ('denoise_nl_means', (np.ones((10, 10)),), {'fast_mode': False, 'patch_distance': 1}),
    ('denoise_wavelet', (np.ones((12, 12)),), {'wavelet': 'sym2', 'sigma': 0.1}),
    ('estimate_sigma', (np.ones((5, 6, 7)),), {'channel_axis': 0}),
    ('inpaint_biharmonic', (np.ones((10, 10)), np.zeros((10, 10), bool)), {'split_into_regions': True}),
    ('unwrap_phase', (np.ones((5, 6)),), {'wrap_around': True}),
    ('richardson_lucy', (np.ones(12), np.ones(3)), {'num_iter': 1}),
])
def test_out_of_domain(name, args, options, native_only):
    with pytest.raises(NotImplementedError, match='fallback is disabled'):
        getattr(restoration, name)(*args, **options)
