"""Bounded circular local peaks with NumPy ordering and SciPy spatial pruning."""

import math

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


def _ordinary(array, ndim, dtype):
    return (type(array) is np.ndarray and array.ndim == ndim and array.dtype == dtype
            and array.dtype.metadata is None and array.flags.c_contiguous and array.flags.aligned
            and _ordinary_storage(array))


def _cap(value):
    return type(value) is int and value >= 0 or type(value) is float and value == np.inf


def _keep_distant(x, y, min_xdistance, min_ydistance, limit):
    from scipy.spatial import cKDTree

    coordinates = np.stack((x, y), axis=1)
    tree = cKDTree(coordinates)
    radius = np.hypot(min_xdistance, min_ydistance)
    rejected = np.zeros(len(x), dtype=bool)
    count = 0
    for i, point in enumerate(coordinates):
        if count >= limit:
            rejected[i:] = True
            break
        if not rejected[i]:
            for j in tree.query_ball_point(point, radius):
                if (j > i and abs(x[j] - x[i]) <= min_xdistance
                        and abs(y[j] - y[i]) <= min_ydistance):
                    rejected[j] = True
            count += 1
    return ~rejected


def _finish(planes, radii, min_xdistance, min_ydistance, num_peaks, total_num_peaks, normalize):
    votes, x, y, r = [], [], [], []
    for radius, (plane_votes, plane_x, plane_y) in zip(radii, planes):
        if num_peaks < plane_votes.size:
            order = np.argsort(plane_votes)[::-1][:num_peaks]
            plane_votes, plane_x, plane_y = plane_votes[order], plane_x[order], plane_y[order]
        votes.extend(plane_votes)
        x.extend(plane_x)
        y.extend(plane_y)
        r.extend((radius,) * plane_votes.size)

    votes, x, y, r = np.array(votes), np.array(x), np.array(y), np.array(r)
    order = np.argsort(votes / r if normalize else votes)
    values = (votes[order][::-1], x[order][::-1], y[order][::-1], r[order][::-1])
    limit = votes.size if total_num_peaks == np.inf else total_num_peaks
    if (min_xdistance, min_ydistance) == (1, 1) or not votes.size:
        return tuple(value[:limit] for value in values)
    keep = _keep_distant(values[1], values[2], min_xdistance, min_ydistance, limit)
    return tuple(value[keep] for value in values)


def hough_circle_peaks(hspaces, radii, min_xdistance=1, min_ydistance=1, threshold=None,
                       num_peaks=np.inf, total_num_peaks=np.inf, normalize=False):
    """Return the raw votes, center coordinates and radii of circular peaks."""
    if (_ordinary(hspaces, 3, np.float64) and _ordinary(radii, 1, np.intp)
            and 1 <= hspaces.shape[0] <= 16 and hspaces.shape[0] == radii.size
            and 1 <= min(hspaces.shape[1:]) <= max(hspaces.shape[1:]) <= 256
            and type(min_xdistance) is int and 0 <= min_xdistance <= min(32, hspaces.shape[2])
            and type(min_ydistance) is int and 0 <= min_ydistance <= 32
            and (threshold is None or type(threshold) is int and 0 <= threshold <= 2**53
                 or type(threshold) is float and threshold >= 0 and math.isfinite(threshold))
            and _cap(num_peaks) and _cap(total_num_peaks) and type(normalize) is bool):
        _core.hough_circle_peak_check_radii(radii)
        if (radii > 0).all():
            planes = _core.hough_circle_peak_indices(hspaces, min_xdistance, min_ydistance, threshold)
            if planes is not None:
                return _finish(planes, radii, min_xdistance, min_ydistance, num_peaks, total_num_peaks, normalize)

    return fallback('transform', 'hough_circle_peaks', hspaces, radii, min_xdistance=min_xdistance,
                    min_ydistance=min_ydistance, threshold=threshold, num_peaks=num_peaks,
                    total_num_peaks=total_num_peaks, normalize=normalize)
