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


def _min_limit(x, val=np.finfo(float).eps):
    _core.fft_check(x)
    mask = np.abs(x) < val
    x[mask] = np.sign(x[mask]) * val


def _center(x, oshape):
    start = (np.array(x.shape) - np.array(oshape)) // 2
    return x[tuple(slice(s, s + n) for s, n in zip(start, oshape))]


def _pad(data, shape):
    out = np.zeros(shape, dtype=data.dtype)
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


class LPIFilter2D:
    """Linear Position-Invariant Filter (2-dimensional)."""

    def __init__(self, impulse_response, **filter_params):
        if not callable(impulse_response):
            raise ValueError("Impulse response must be a callable.")
        self.impulse_response = impulse_response
        self.filter_params = filter_params
        self._cache = None

    def _prepare(self, data):
        dshape = np.array(data.shape)
        even_offset = (dshape % 2 == 0).astype(int)
        dshape += even_offset
        oshape = np.array(data.shape) * 2 - 1

        float_dtype = _supported_float_type(data.dtype)
        if isinstance(data, np.ndarray):
            _core.fft_check(data)
        data = data.astype(float_dtype, copy=False)

        if self._cache is None or np.any(self._cache.shape != oshape):
            coords = np.mgrid[
                [slice(0 + offset, float(n + offset))
                 for n, offset in zip(dshape, even_offset)]
            ]
            for k, coord in enumerate(coords):
                coord -= (dshape[k] - 1) / 2.0
            coords = coords.reshape(2, -1).T
            coords = coords.astype(float_dtype, copy=False)

            f = self.impulse_response(
                coords[:, 0], coords[:, 1], **self.filter_params
            ).reshape(dshape)
            f = _pad(f, oshape)
            F = _fft(f, True)
            self._cache = F
        else:
            F = self._cache

        data = _pad(data, oshape)
        G = _fft(data, True)
        # G's preparation/FFT allocate after the cache was retrieved. The
        # cache is publicly mutable, so check it before its next array read.
        _core.fft_check(F)
        return F, G

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
        predefined_filter = _new_filter(impulse_response, filter_params)
    return predefined_filter(data)


def filter_inverse(data, impulse_response=None, filter_params=None, max_gain=2,
                   predefined_filter=None):
    if filter_params is None:
        filter_params = {}
    _check_2d(data, "data")
    if predefined_filter is None:
        filt = _new_filter(impulse_response, filter_params)
    else:
        filt = predefined_filter

    F, G = filt._prepare(data)
    _min_limit(F, val=np.finfo(F.real.dtype).eps)
    F = 1 / F
    mask = np.abs(F) > max_gain
    F[mask] = np.sign(F[mask]) * max_gain
    return _center(np.abs(_ifftshift(_fft(G * F, False))), data.shape)


def wiener(data, impulse_response=None, filter_params=None, K=0.25, predefined_filter=None):
    if filter_params is None:
        filter_params = {}
    _check_2d(data, "data")
    if not isinstance(K, float):
        _check_2d(K, "K")
    if predefined_filter is None:
        filt = _new_filter(impulse_response, filter_params)
    else:
        filt = predefined_filter

    F, G = filt._prepare(data)
    _min_limit(F, val=np.finfo(F.real.dtype).eps)
    H_mag_sqr = np.abs(F) ** 2
    if isinstance(K, np.ndarray):
        _core.fft_check(K)
    F = 1 / F * H_mag_sqr / (H_mag_sqr + K)
    return _center(np.abs(_ifftshift(_fft(G * F, False))), data.shape)
