import importlib.util
import inspect

import numpy as np
import pytest
import skimage
from skimage import restoration as reference

from skimagexx import _core, restoration
from skimagexx._compat import FallbackWarning, set_fallback_enabled


_HAS_PYWT = importlib.util.find_spec('pywt') is not None
_ROOT_HALF = np.sqrt(0.5)
_DB2_LOW = np.array([
    -0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416,
], dtype=np.float64)
_DB2_HIGH = np.array([
    -0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037,
], dtype=np.float64)


def _sym_index(pos, extent):
    if extent == 1:
        return 0
    period = 2 * extent
    wrapped = pos % period
    return wrapped if wrapped < extent else period - wrapped - 1


def _dwt_down_1d(signal, wavelet):
    n = len(signal)
    length = 4 if wavelet == 'db2' else 2
    out_len = (n + length - 1) // 2
    if length == 2:
        idx = 2 * np.arange(out_len)
        a = signal[idx]
        b = signal[np.minimum(idx + 1, n - 1)]
        return _ROOT_HALF * (a + b), _ROOT_HALF * (a - b)
    approx = np.empty(out_len, dtype=np.float64)
    detail = np.empty(out_len, dtype=np.float64)
    for i in range(out_len):
        lo_acc = 0.0
        hi_acc = 0.0
        for k in range(4):
            val = signal[_sym_index(2 * i + 1 - k, n)]
            lo_acc += _DB2_LOW[k] * val
            hi_acc += _DB2_HIGH[k] * val
        approx[i] = lo_acc
        detail[i] = hi_acc
    return approx, detail


def _idwt_up_1d(approx, detail, wavelet, out_len):
    length = 4 if wavelet == 'db2' else 2
    count = len(approx)
    out = np.empty(out_len, dtype=np.float64)
    if length == 2:
        low = _ROOT_HALF * approx
        high = _ROOT_HALF * detail
        for i in range(count):
            if 2 * i < out_len:
                out[2 * i] = low[i] + high[i]
            if 2 * i + 1 < out_len:
                out[2 * i + 1] = low[i] - high[i]
        return out
    for i in range(out_len):
        lo_acc = 0.0
        hi_acc = 0.0
        for k in range((1 - i) & 1, 4, 2):
            j = (i + k - 1) // 2
            if 0 <= j < count:
                lo_acc += _DB2_LOW[k] * approx[j]
                hi_acc += _DB2_HIGH[k] * detail[j]
        out[i] = lo_acc + hi_acc
    return out


def _oracle_estimate_sigma(image, average_sigmas=False, channel_axis=None):
    arr = np.asarray(image, dtype=np.float64)
    if channel_axis is not None:
        arr = np.moveaxis(arr, channel_axis, -1)
        vals = [_oracle_estimate_sigma(arr[..., c]) for c in range(arr.shape[-1])]
        return np.mean(vals) if average_sigmas else [np.float64(v) for v in vals]
    if arr.ndim == 1:
        _, detail = _dwt_down_1d(arr, 'db2')
    else:
        rows_hi = np.array([_dwt_down_1d(row, 'db2')[1] for row in arr])
        detail = np.array([_dwt_down_1d(col, 'db2')[1] for col in rows_hi.T]).T
    nonzero = np.abs(detail.ravel())
    nonzero = nonzero[nonzero != 0.0]
    return np.float64(np.median(nonzero) / 0.6744897501960817)


def _threshold(band, thresh, mode):
    if mode == 'soft':
        return np.sign(band) * np.maximum(np.abs(band) - thresh, 0.0)
    return np.where(np.abs(band) >= thresh, band, 0.0)


def _oracle_denoise_wavelet(
    image, sigma=None, wavelet='db1', mode='soft', wavelet_levels=None,
    method='BayesShrink', channel_axis=None,
):
    arr = np.asarray(image, dtype=np.float64)
    if channel_axis is not None:
        moved = np.moveaxis(arr, channel_axis, -1)
        out = np.empty_like(moved, dtype=image.dtype)
        for c in range(moved.shape[-1]):
            out[..., c] = _oracle_denoise_wavelet(
                moved[..., c].astype(image.dtype), sigma=sigma, wavelet=wavelet,
                mode=mode, wavelet_levels=wavelet_levels, method=method,
            )
        return np.moveaxis(out, -1, channel_axis)
    length = 4 if wavelet == 'db2' else 2
    max_level = (min(arr.shape) // (length - 1)).bit_length() - 1
    levels = max(max_level - 3, 1) if wavelet_levels is None else wavelet_levels
    if levels == 0:
        return arr.astype(image.dtype)
    coeffs = []
    cur = arr.copy()
    if arr.ndim == 1:
        shapes = []
        for _ in range(levels):
            shapes.append(len(cur))
            cur, d = _dwt_down_1d(cur, wavelet)
            coeffs.append(d)
        if sigma is None:
            nonzero = np.abs(coeffs[0].ravel())
            nonzero = nonzero[nonzero != 0.0]
            sigma = float(np.median(nonzero) / 0.6744897501960817)
        else:
            sigma = float(sigma)
        denom = len(arr)
        for i, d in enumerate(coeffs):
            if method == 'VisuShrink':
                t = sigma * np.sqrt(2.0 * np.log(denom))
            else:
                var_x = max(np.mean(d ** 2) - sigma ** 2, float(np.finfo(image.dtype).eps))
                t = (sigma ** 2) / np.sqrt(var_x)
            coeffs[i] = _threshold(d, t, mode)
        for d, out_len in zip(reversed(coeffs), reversed(shapes)):
            cur = _idwt_up_1d(cur, d, wavelet, out_len)
        return cur.astype(image.dtype)
    shapes = []
    for _ in range(levels):
        shapes.append(cur.shape)
        rows = [_dwt_down_1d(r, wavelet) for r in cur]
        la = np.array([p[0] for p in rows])
        ld = np.array([p[1] for p in rows])
        ca = [_dwt_down_1d(c, wavelet) for c in la.T]
        cd = [_dwt_down_1d(c, wavelet) for c in ld.T]
        ll = np.array([p[0] for p in ca]).T
        lh = np.array([p[1] for p in ca]).T
        hl = np.array([p[0] for p in cd]).T
        hh = np.array([p[1] for p in cd]).T
        coeffs.append([lh, hl, hh])
        cur = ll
    if sigma is None:
        nonzero = np.abs(coeffs[0][2].ravel())
        nonzero = nonzero[nonzero != 0.0]
        sigma = float(np.median(nonzero) / 0.6744897501960817)
    else:
        sigma = float(sigma)
    denom = arr.size
    for level_idx, bands in enumerate(coeffs):
        for b_idx, d in enumerate(bands):
            if method == 'VisuShrink':
                t = sigma * np.sqrt(2.0 * np.log(denom))
            else:
                var_x = max(np.mean(d ** 2) - sigma ** 2, float(np.finfo(image.dtype).eps))
                t = (sigma ** 2) / np.sqrt(var_x)
            bands[b_idx] = _threshold(d, t, mode)
    for (lh, hl, hh), (rlen, clen) in zip(reversed(coeffs), reversed(shapes)):
        la = np.array([_idwt_up_1d(cur[:, c], lh[:, c], wavelet, rlen) for c in range(cur.shape[1])]).T
        ld = np.array([_idwt_up_1d(hl[:, c], hh[:, c], wavelet, rlen) for c in range(hl.shape[1])]).T
        cur = np.array([_idwt_up_1d(la[r], ld[r], wavelet, clen) for r in range(rlen)])
    return cur.astype(image.dtype)


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
    if not _HAS_PYWT and name in ('denoise_wavelet', 'estimate_sigma'):
        with pytest.raises(ImportError, match='PyWavelets'):
            getattr(reference, name)(*args, **options)
        if name == 'estimate_sigma':
            expected = _oracle_estimate_sigma(*args, **options)
        else:
            expected = _oracle_denoise_wavelet(*args, **options)
    else:
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
        elif not _HAS_PYWT and name in ('denoise_wavelet', 'estimate_sigma'):
            with pytest.raises(ImportError, match='PyWavelets'):
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
