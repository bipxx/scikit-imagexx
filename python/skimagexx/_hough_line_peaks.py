"""Bounded Hough peak detection with the original NumPy cap ordering."""

import math

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


def _ordinary(array, rank, dtype):
    return (type(array) is np.ndarray and array.ndim == rank and array.dtype == dtype
            and array.dtype.metadata is None and array.flags.c_contiguous and array.flags.aligned
            and _ordinary_storage(array))


def hough_line_peaks(hspace, angles, dists, min_distance=9, min_angle=10, threshold=None, num_peaks=np.inf):
    """Return peak votes and their coordinates in the supplied Hough space."""
    if (_ordinary(hspace, 2, np.uint64) and 0 < hspace.size < 262144
            and hspace.shape[0] <= 1024 and hspace.shape[1] <= 512
            and _ordinary(angles, 1, np.float64) and angles.size == hspace.shape[1]
            and _ordinary(dists, 1, np.float64) and dists.size == hspace.shape[0]
            and type(min_distance) is int and 0 <= min_distance <= 32
            and type(min_angle) is int and 0 <= min(min_angle, angles.size) <= 32
            and (threshold is None or type(threshold) is int and 0 <= threshold <= 2**53
                 or type(threshold) is float and threshold >= 0 and math.isfinite(threshold))
            and (type(num_peaks) is int and num_peaks >= 0 or type(num_peaks) is float and num_peaks == np.inf)):
        result = _core.hough_line_peak_indices(hspace, angles, dists, min_distance,
                                               min(min_angle, angles.size), threshold)
        if result is not None:
            votes, rows, columns = result
            if not votes.size:
                return np.array([]), np.array([]), np.array([])
            if num_peaks < votes.size:
                order = np.argsort(votes)[::-1][:num_peaks]
                votes, rows, columns = votes[order], rows[order], columns[order]
            if not columns.size:
                return votes, np.array([]), np.array([])
            return votes, angles[columns], dists[rows]

    return fallback('transform', 'hough_line_peaks', hspace, angles, dists,
                    min_distance=min_distance, min_angle=min_angle, threshold=threshold, num_peaks=num_peaks)
