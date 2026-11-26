"""Source-ordered LPI filters with owned C++ FFT stages.

NumPy provides coordinates, dtype conversion, padding, products, reciprocal/
gain/Wiener arithmetic, permutation, cropping and absolute value. Impulse
callbacks and user-supplied predefined filters remain Python providers; this
is native-partial FFT execution, not a fully C++ LPI implementation.

The assessed source scope is the frozen 80 public specifications, including
standard real image arrays, real/complex callback kernels, shape-only cache
reuse and source-visible cache mutation. Extended callback FFT dtypes and
non-ndarray-backed storage retain the FFT binding's explicit refusal boundary.
Arbitrary object/protocol inputs and hostile callback mutation are not claimed
as covered. Each transform snapshots the existing global worker budget; there
is no additional pool or per-instance worker state.
"""
import numpy as np

from . import _core

_LEGACY_COMPLEX_SIGN = np.sign(1j) == 1


def _check_2d(array, arg_name):
    # Source check_nD validates a temporary asanyarray without replacing the
    # caller's object used by subsequent operations.
    array = np.asanyarray(array)
    if array.size == 0:
        raise ValueError(f"The parameter `{arg_name}` cannot be an empty array")
    if array.ndim != 2:
        raise ValueError(f"The parameter `{arg_name}` must be a 2-dimensional array")


def _supported_float_type(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        raise ValueError("complex valued input is not supported")
    return {"e": np.float32, "f": np.float32, "d": np.float64}.get(dtype.char, np.float64)


def _fft(array, forward):
    return _core.fft_c2c(array, tuple(range(array.ndim)), forward, _core.get_num_threads())


def _inverse_fft(spectrum, shape):
    if shape is None:
        return _fft(spectrum, False)
    return _core.fft_c2r(spectrum, shape, (0, 1), _core.get_num_threads())


def _shifted_magnitude(spectrum, shape, output_shape):
    if shape is None:
        # Magnitude commutes with the shift. Permute the smaller real buffer.
        magnitude = np.abs(_inverse_fft(spectrum, shape))
        return _center(_ifftshift(magnitude), output_shape)
    out = _center(_ifftshift(_inverse_fft(spectrum, shape)), output_shape)
    np.abs(out, out=out)
    return out


def _full_spectrum(half, shape):
    width = half.shape[1]
    full = np.empty(shape, dtype=half.dtype)
    full[:, :width] = half
    np.conjugate(half[:1, :0:-1], out=full[:1, width:])
    np.conjugate(half[:0:-1, :0:-1], out=full[1:, width:])
    return full


def _limit_spectra(F, G, shape):
    # NumPy 1's sign of a purely imaginary value is real. Limiting or
    # clipping conjugate coefficients can then break Hermitian symmetry.
    if shape is not None and _LEGACY_COMPLEX_SIGN and np.any((F.real == 0) & (F.imag != 0)):
        return _full_spectrum(F, shape), _full_spectrum(G, shape), None
    return F, G, shape


def _min_limit(x, val=np.finfo(float).eps):
    _core.fft_check(x)
    magnitude = np.abs(x)
    mask = magnitude < val
    x[mask] = np.sign(x[mask]) * val
    return magnitude


def _center(x, oshape):
    if x.ndim == 2 and len(oshape) == 2:
        rows, cols = oshape
        row = (x.shape[0] - rows) // 2
        col = (x.shape[1] - cols) // 2
        return x[row:row + rows, col:col + cols]
    start = (np.array(x.shape) - np.array(oshape)) // 2
    return x[tuple(slice(s, s + n) for s, n in zip(start, oshape))]


def _pad(data, shape):
    dtype = data.dtype
    # Cast directly into the padded buffer for dtypes that PocketFFT would
    # otherwise convert across the entire padded extent.
    if type(data) is np.ndarray:
        if dtype.kind in "biu":
            dtype = np.float64
        elif dtype.char == "e":
            dtype = np.float32
    out = np.zeros(shape, dtype=dtype)
    # Allocation and the preceding impulse callback can invalidate an earlier
    # view. Validate the current allocation immediately before this copy.
    if isinstance(data, np.ndarray):
        _core.fft_check(data)
    out[tuple(slice(0, n) for n in data.shape)] = data
    return out


def _ifftshift(array):
    # NumPy's permutation only; no NumPy/SciPy transform entrypoint is called.
    axes = tuple(range(array.ndim))
    return np.roll(array, tuple(-(n // 2) for n in array.shape), axes)


def _coordinates(dshape, even_offset, dtype):
    rows, cols = map(int, dshape)
    row = np.arange(rows) + (even_offset[0] - (rows - 1) / 2.0)
    col = np.arange(cols) + (even_offset[1] - (cols - 1) / 2.0)
    if dtype is np.float32:
        # Match the callback's coordinate views and their owning array.
        size = rows * cols
        coords = np.empty((size, 2), dtype=dtype, order="F" if size > 1 else "C")
        coords[:, 0].reshape(rows, cols)[:] = row[:, None]
        coords[:, 1].reshape(rows, cols)[:] = col
        return coords
    grid = np.empty((2, rows, cols), dtype=dtype)
    grid[0] = row[:, None]
    grid[1] = col
    return grid.reshape(2, -1).T


class LPIFilter2D:
    """Linear Position-Invariant Filter (2-dimensional)."""

    def __init__(self, impulse_response, **filter_params):
        if not callable(impulse_response):
            raise ValueError("Impulse response must be a callable.")
        self.impulse_response = impulse_response
        self.filter_params = filter_params
        self._cache = None

    def _prepare(self, data):
        F, G, _ = self._prepare_spectra(data)
        return F, G

    def _prepare_spectra(self, data, half=False):
        dshape = np.array(data.shape)
        even_offset = (dshape % 2 == 0).astype(int)
        dshape += even_offset
        oshape = np.array(data.shape) * 2 - 1

        float_dtype = _supported_float_type(data.dtype)
        if isinstance(data, np.ndarray):
            _core.fft_check(data)
        data = data.astype(float_dtype, copy=False)

        if half or self._cache is None or np.any(self._cache.shape != oshape):
            coords = _coordinates(dshape, even_offset, float_dtype)
            f = self.impulse_response(
                coords[:, 0], coords[:, 1], **self.filter_params
            ).reshape(dshape)
            f = _pad(f, oshape)
            if half and f.dtype.kind in "biuf":
                axes = (0, 1)
                F = _core.fft_r2c(f, axes, _core.get_num_threads())
                data = _pad(data, oshape)
                G = _core.fft_r2c(data, axes, _core.get_num_threads())
                return F, G, tuple(oshape)
            F = _fft(f, True)
            self._cache = F
        else:
            F = self._cache

        data = _pad(data, oshape)
        G = _fft(data, True)
        # G's preparation/FFT allocate after the cache was retrieved. The
        # cache is publicly mutable, so check it before its next array read.
        _core.fft_check(F)
        return F, G, None

    def __call__(self, data):
        _check_2d(data, "data")
        F, G = self._prepare(data)
        out = _fft(F * G, False)
        return np.abs(_center(out, data.shape))


def _new_filter(impulse_response, filter_params):
    try:
        return LPIFilter2D(impulse_response, **filter_params)
    except TypeError as error:
        # Python includes the qualified constructor in malformed **mapping
        # errors. Preserve the public source message without spoofing the
        # owned class's module or translating errors raised by user mappings.
        prefix = __name__ + ".LPIFilter2D() argument after **"
        message = str(error)
        if message.startswith(prefix):
            raise TypeError("skimage.filters.lpi_filter.LPIFilter2D() argument after **"
                            + message[len(prefix):]) from None
        raise


def filter_forward(data, impulse_response=None, filter_params=None, predefined_filter=None):
    if filter_params is None:
        filter_params = {}
    _check_2d(data, "data")
    if predefined_filter is None:
        filt = _new_filter(impulse_response, filter_params)
        F, G, shape = filt._prepare_spectra(data, half=True)
        if shape is not None and F.dtype == G.dtype:
            np.multiply(F, G, out=G)
        else:
            G = F * G
        return np.abs(_center(_inverse_fft(G, shape), data.shape))
    return predefined_filter(data)


def filter_inverse(data, impulse_response=None, filter_params=None, max_gain=2,
                   predefined_filter=None):
    if filter_params is None:
        filter_params = {}
    _check_2d(data, "data")
    if predefined_filter is None:
        filt = _new_filter(impulse_response, filter_params)
        F, G, shape = filt._prepare_spectra(
            data, half=isinstance(max_gain, (int, float, np.integer, np.floating))
        )
    else:
        F, G = predefined_filter._prepare(data)
        shape = None

    F, G, shape = _limit_spectra(F, G, shape)
    magnitude = _min_limit(F, val=np.finfo(F.real.dtype).eps)
    if shape is None:
        # F can still be the predefined filter's public cache.
        del magnitude
        F = 1 / F
        magnitude = np.abs(F)
    else:
        np.divide(1, F, out=F)
        np.abs(F, out=magnitude)
    mask = magnitude > max_gain
    F[mask] = np.sign(F[mask]) * max_gain
    if shape is not None and F.dtype == G.dtype:
        np.multiply(G, F, out=G)
    else:
        G = G * F
    return _shifted_magnitude(G, shape, data.shape)


def wiener(data, impulse_response=None, filter_params=None, K=0.25, predefined_filter=None):
    if filter_params is None:
        filter_params = {}
    _check_2d(data, "data")
    if not isinstance(K, float):
        _check_2d(K, "K")
    if predefined_filter is None:
        filt = _new_filter(impulse_response, filter_params)
        F, G, shape = filt._prepare_spectra(data, half=type(K) is float)
    else:
        F, G = predefined_filter._prepare(data)
        shape = None

    F, G, shape = _limit_spectra(F, G, shape)
    limits = np.finfo(F.real.dtype)
    H_mag_sqr = _min_limit(F, val=limits.eps)
    if isinstance(K, np.ndarray):
        _core.fft_check(K)
    if (np.result_type(F, G, K) == F.dtype and np.geterr()["under"] == "ignore"
            and H_mag_sqr.max(initial=0) <= float(limits.max) ** 0.5 * 0.5):
        np.square(F.real, out=H_mag_sqr)
        H_mag_sqr += np.square(F.imag)
    else:
        # Preserve magnitude rounding for widened results and NumPy's
        # original floating-point warnings for exceptional inputs.
        H_mag_sqr = np.abs(F)
        np.square(H_mag_sqr, out=H_mag_sqr)
    if shape is None:
        F = 1 / F
    else:
        np.divide(1, F, out=F)
    F *= H_mag_sqr
    denominator = H_mag_sqr + K
    if shape is not None and denominator.dtype == F.real.dtype:
        np.divide(F, denominator, out=F)
    else:
        F = F / denominator
    del denominator
    if shape is not None and F.dtype == G.dtype:
        np.multiply(G, F, out=G)
    else:
        G = G * F
    return _shifted_magnitude(G, shape, data.shape)
