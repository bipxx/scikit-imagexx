"""Chan–Vese evolution for ordinary two-dimensional grayscale arrays."""

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


_MAX_PIXELS = 1024 * 1024
_MAX_ITERATIONS = 10_000
_DTYPES = frozenset(
    np.dtype(dtype).num for dtype in (
        np.bool_, np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.uintc,
        np.int_, np.uint, np.longlong, np.ulonglong, np.float16, np.float32, np.float64,
    )
)
_SINGLE = frozenset(np.dtype(dtype).num for dtype in (np.float16, np.float32))


def _ordinary_image(array):
    return (type(array) is np.ndarray and array.ndim == 2 and 0 < array.size <= _MAX_PIXELS
            and array.dtype.num in _DTYPES and array.dtype.isnative and array.dtype.metadata is None
            and array.flags.c_contiguous and array.flags.aligned and array.flags.writeable
            and _ordinary_storage(array))


def _weight(value, limit):
    return type(value) in (int, float) and 0 <= value <= limit


def _initial_level_set(value, shape, dtype):
    if type(value) is str:
        if value == 'checkerboard':
            rows = np.arange(shape[0], dtype=dtype)
            columns = np.arange(shape[1], dtype=dtype)
            rows *= np.pi / 5
            columns *= np.pi / 5
            return np.sin(rows)[:, None] * np.sin(columns)
        if value in ('disk', 'small disk'):
            row, col = ((extent - 1) // 2 for extent in shape)
            radius = float(min(row, col))
            if radius == 0:
                return None
            rows = np.arange(shape[0], dtype=np.float64) - row
            columns = np.arange(shape[1], dtype=np.float64) - col
            distance = np.sqrt(np.square(rows)[:, None] + np.square(columns))
            if value == 'small disk':
                radius /= 2
                return ((radius - distance) / (radius * 3)).astype(dtype, copy=False)
            return ((radius - distance) / radius).astype(dtype, copy=False)
    elif _ordinary_image(value) and value.shape == shape:
        if value.dtype != dtype:
            _core.fft_check(value)
            # The native finite-domain check will route overflowing casts to
            # the original implementation, which owns the warning or error.
            with np.errstate(over='ignore', invalid='ignore'):
                return value.astype(dtype, copy=False)
        return value
    return None


def chan_vese(
    image,
    mu=0.25,
    lambda1=1.0,
    lambda2=1.0,
    tol=1e-3,
    max_num_iter=500,
    dt=0.5,
    init_level_set='checkerboard',
    extended_output=False,
):
    """Segment a grayscale image by evolving a Chan–Vese level set."""
    if (_ordinary_image(image)
            and _weight(mu, 16) and _weight(lambda1, 16) and _weight(lambda2, 16)
            and _weight(dt, 16) and _weight(tol, 2)
            and type(max_num_iter) is int and 0 <= max_num_iter <= _MAX_ITERATIONS
            and type(extended_output) is bool and np.geterr()['under'] == 'ignore'):
        dtype = np.dtype(np.float32 if image.dtype.num in _SINGLE else np.float64)
        phi = _initial_level_set(init_level_set, image.shape, dtype)
        if phi is not None:
            if image.dtype != dtype:
                _core.fft_check(image)
            converted = image.astype(dtype, copy=False)
            reduction_buffer = np.getbufsize() if np.__version__.startswith('1.') else 0
            result = _core.chan_vese_evolve(
                converted, phi, mu, lambda1, lambda2, tol, max_num_iter, dt, extended_output, reduction_buffer,
            )
            if result is not None:
                return result
    return fallback('segmentation', 'chan_vese', image, mu=mu, lambda1=lambda1, lambda2=lambda2,
                    tol=tol, max_num_iter=max_num_iter, dt=dt, init_level_set=init_level_set,
                    extended_output=extended_output)
