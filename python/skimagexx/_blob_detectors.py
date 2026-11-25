"""Native scale-space blob detection for ordinary bounded 2D images."""

import math
from functools import lru_cache

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


def _real_in(value, lower, upper):
    return type(value) in (int, float) and lower <= value <= upper


def _sigma_pair(value):
    if _real_in(value, 0.5, 128):
        return value, value
    if type(value) in (tuple, list) and len(value) == 2 and all(_real_in(x, 0.5, 128) for x in value):
        return tuple(value)
    return None


@lru_cache(maxsize=64)
def _scales(method, lo, hi, steps, log_scale, dtype):
    if method == 2:
        if log_scale:
            values = np.logspace(math.log(lo[0], 10), math.log(hi[0], 10), steps)
        else:
            values = np.linspace(lo[0], hi[0], steps)
        result = tuple((float(s), float(s)) for s in values)
    else:
        low, high = np.asarray(lo, dtype=dtype), np.asarray(hi, dtype=dtype)
        if method == 0:
            count = int(np.mean(np.log(high / low) / np.log(steps) + 1))
            if not 1 <= count <= 64:
                return None
            values = [low * (steps**i) for i in range(count + 1)]
        elif log_scale:
            values = np.logspace(np.log10(low), np.log10(high), steps)
        else:
            values = np.linspace(low, high, steps)
        result = tuple((float(y), float(x)) for y, x in values)
    if any(not _real_in(s, 0.5, 128) for pair in result for s in pair):
        return None
    return result


def _detect(method, image, min_sigma, max_sigma, steps, threshold, overlap, log_scale,
            threshold_rel, exclude_border):
    if (type(image) is not np.ndarray or image.ndim != 2 or not image.size
            or max(image.shape) > 4096 or image.dtype.char not in 'Bfd'
            or image.dtype.metadata is not None or not image.dtype.isnative
            or not image.flags.c_contiguous or not image.flags.aligned or not image.flags.writeable
            or not _ordinary_storage(image)
            or (threshold is not None and not _real_in(threshold, -1e12, 1e12))
            or (threshold_rel is not None and not _real_in(threshold_rel, 0, 1))
            or not _real_in(overlap, 0, 1)):
        return None

    scalar_sigma = type(min_sigma) in (int, float) and type(max_sigma) in (int, float)
    lower, upper = _sigma_pair(min_sigma), _sigma_pair(max_sigma)
    if lower is None or upper is None or any(x > y for x, y in zip(lower, upper)):
        return None
    if method == 2 and not scalar_sigma:
        return None
    if type(exclude_border) in (bool, int):
        exclude_border = (int(exclude_border),) * 2
    if (type(exclude_border) is not tuple or len(exclude_border) != 2
            or any(type(value) not in (int, bool) or not 0 <= value <= 4096 for value in exclude_border)):
        return None
    if method == 0:
        if not _real_in(steps, 1 + 1 / 1024, 16):
            return None
        normalization = 1 / (steps - 1)
    else:
        if type(steps) is not int or not 1 <= steps <= 64 or type(log_scale) is not bool:
            return None
        normalization = 1
    dtype = np.float32 if image.dtype.char == 'f' else np.float64
    scales = _scales(method, lower, upper, steps, log_scale, dtype)
    if scales is None:
        return None
    return _core.blob_detect(image, method, scales, scalar_sigma, normalization, threshold,
                             threshold_rel, overlap, exclude_border)


def blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=0.5, overlap=0.5,
             *, threshold_rel=None, exclude_border=False):
    """Find blobs with a scale-normalized difference of Gaussians."""
    result = _detect(0, image, min_sigma, max_sigma, sigma_ratio, threshold, overlap, False,
                     threshold_rel, exclude_border)
    if result is not None:
        return result
    return fallback('feature', 'blob_dog', image, min_sigma, max_sigma, sigma_ratio, threshold,
                    overlap, threshold_rel=threshold_rel, exclude_border=exclude_border)


def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5,
             log_scale=False, *, threshold_rel=None, exclude_border=False):
    """Find blobs with a scale-normalized Laplacian of Gaussians."""
    result = _detect(1, image, min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale,
                     threshold_rel, exclude_border)
    if result is not None:
        return result
    return fallback('feature', 'blob_log', image, min_sigma, max_sigma, num_sigma, threshold,
                    overlap, log_scale, threshold_rel=threshold_rel, exclude_border=exclude_border)


def blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01, overlap=0.5,
             log_scale=False, *, threshold_rel=None):
    """Find blobs with an approximate determinant of the Hessian."""
    result = _detect(2, image, min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale,
                     threshold_rel, False)
    if result is not None:
        return result
    return fallback('feature', 'blob_doh', image, min_sigma, max_sigma, num_sigma, threshold,
                    overlap, log_scale, threshold_rel=threshold_rel)
