"""Bounded native peak selection and subpixel refinement."""

from functools import lru_cache
import math

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


_WIDE_RELATIVE_PRODUCT = type(np.float32(1) * 1.0) is np.float64


def _ordinary(array):
    return (type(array) is np.ndarray and array.ndim == 2 and array.dtype.isnative
            and array.dtype.metadata is None and array.flags.c_contiguous and array.flags.aligned
            and array.flags.writeable and _ordinary_storage(array))


def _ordinary_image(image):
    return (_ordinary(image) and image.dtype.char in 'Bfd' and 0 < image.size <= 16777216
            and max(image.shape) <= 65536)


def _ordinary_integer(array):
    return _ordinary(array) and array.dtype.kind == 'i' and array.dtype.itemsize in (4, 8)


def _border_width(exclude, distance, shape):
    if type(exclude) is bool:
        return (distance, distance) if exclude else (0, 0)
    if type(exclude) is int and exclude >= 0:
        return min(exclude, shape[0]), min(exclude, shape[1])
    if (type(exclude) is tuple and len(exclude) == 2
            and all(type(value) in (int, bool) and value >= 0 for value in exclude)):
        return tuple(min(value, dimension) for value, dimension in zip(exclude, shape))
    return None


def _cap(value):
    if type(value) is int and value >= 0:
        return min(value, 2**31)
    if type(value) is float and value == np.inf:
        return 2**31
    return None


def _threshold(value):
    return (value is None or type(value) is int and abs(value) <= 2**53
            or type(value) is float and math.isfinite(value) and abs(value) <= 1e30)


def _find_peaks(image, min_distance, threshold_abs, threshold_rel, exclude_border, num_peaks,
                footprint, labels, num_peaks_per_label, p_norm, corner, indices):
    if (not _ordinary_image(image) or type(min_distance) is not int or not 1 <= min_distance <= 8
            or not _threshold(threshold_abs) or not _threshold(threshold_rel) or footprint is not None
            or image.dtype.char == 'B' and type(threshold_rel) is int
            or type(p_norm) not in (int, float) or not (1 <= p_norm <= 32 or p_norm == np.inf)
            or type(indices) is not bool):
        return None
    if labels is not None and (not _ordinary_integer(labels) or labels.shape != image.shape):
        return None
    border = _border_width(exclude_border, min_distance, image.shape)
    cap, cap_label = _cap(num_peaks), _cap(num_peaks_per_label)
    if border is None or cap is None or cap_label is None:
        return None
    norm = {np.inf: 0, 1: 1, 2: 2}.get(p_norm, 3)
    return _core.peak_local_max_2d(image, labels, min_distance, threshold_abs, threshold_rel,
                                  border, cap, cap_label, norm, corner, indices, _WIDE_RELATIVE_PRODUCT, p_norm)


def peak_local_max(image, min_distance=1, threshold_abs=None, threshold_rel=None,
                   exclude_border=True, num_peaks=np.inf, footprint=None, labels=None,
                   num_peaks_per_label=np.inf, p_norm=np.inf):
    """Find peaks in an image and return their coordinates."""
    result = _find_peaks(image, min_distance, threshold_abs, threshold_rel, exclude_border,
                         num_peaks, footprint, labels, num_peaks_per_label, p_norm, False, True)
    if result is not None:
        return result
    return fallback('feature', 'peak_local_max', image, min_distance=min_distance,
                    threshold_abs=threshold_abs, threshold_rel=threshold_rel, exclude_border=exclude_border,
                    num_peaks=num_peaks, footprint=footprint, labels=labels,
                    num_peaks_per_label=num_peaks_per_label, p_norm=p_norm)


def corner_peaks(image, min_distance=1, threshold_abs=None, threshold_rel=None,
                 exclude_border=True, indices=True, num_peaks=np.inf, footprint=None, labels=None,
                 *, num_peaks_per_label=np.inf, p_norm=np.inf):
    """Find peaks in a corner response, suppressing connected equal peaks."""
    result = _find_peaks(image, min_distance, threshold_abs, threshold_rel, exclude_border,
                         num_peaks, footprint, labels, num_peaks_per_label, p_norm, True, indices)
    if result is not None:
        return result
    return fallback('feature', 'corner_peaks', image, min_distance=min_distance,
                    threshold_abs=threshold_abs, threshold_rel=threshold_rel, exclude_border=exclude_border,
                    indices=indices, num_peaks=num_peaks, footprint=footprint, labels=labels,
                    num_peaks_per_label=num_peaks_per_label, p_norm=p_norm)


@lru_cache(maxsize=64)
def _f_critical_values(window_size, alpha):
    from scipy.stats import f

    redundancy = window_size**2 - 2
    return (float(f.isf(1 - alpha, redundancy, redundancy)),
            float(f.isf(alpha, redundancy, redundancy)))


def corner_subpix(image, corners, window_size=11, alpha=0.99):
    """Refine integer corner positions to subpixel precision."""
    if (_ordinary_image(image) and _ordinary_integer(corners) and corners.shape[1] == 2
            and corners.shape[0] <= 262144 and type(window_size) is int and 3 <= window_size <= 31
            and type(alpha) is float and 0.5 < alpha < 1):
        dot, edge = _f_critical_values(window_size, alpha)
        if math.isfinite(dot) and math.isfinite(edge):
            result = _core.corner_subpix_2d(image, corners, window_size, dot, edge)
            if result is not None:
                return result
    return fallback('feature', 'corner_subpix', image, corners, window_size=window_size, alpha=alpha)
