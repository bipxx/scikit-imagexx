"""Bounded owned local extrema for two-dimensional integer plateaus."""

import numpy as np

from . import _core
from ._compat import fallback


def _has_numpy_span(array):
    """Prove the two-dimensional read extent from a NumPy-owned allocation."""
    owner = array
    while type(owner.base) is np.ndarray:
        owner = owner.base
    if not owner.flags.owndata or not (owner.flags.c_contiguous or owner.flags.f_contiguous):
        return False
    if owner is array:
        return True

    start = array.__array_interface__['data'][0] - owner.__array_interface__['data'][0]
    row, column = ((length - 1) * stride for length, stride in zip(array.shape, array.strides))
    return (start + min(0, row) + min(0, column) >= 0
            and start + max(0, row) + max(0, column) + array.itemsize <= owner.nbytes)


def _connectivity(footprint, connectivity):
    if footprint is None:
        if connectivity is None:
            return 2
        if type(connectivity) is int and connectivity in (1, 2):
            return connectivity
        return None
    if (type(footprint) is not np.ndarray or footprint.dtype != np.dtype(bool)
            or footprint.shape != (3, 3) or not _has_numpy_span(footprint)):
        return None
    # Original skips the first sorted offset, which presumes a present center.
    if not (footprint[1, 1] and footprint[0, 1] and footprint[1, 0]
            and footprint[1, 2] and footprint[2, 1]):
        return None
    corners = (footprint[0, 0], footprint[0, 2], footprint[2, 0], footprint[2, 2])
    if all(corners):
        return 2
    if not any(corners):
        return 1
    return None


def _local_extrema(name, image, footprint, connectivity, indices, allow_borders):
    if (type(image) is np.ndarray and image.ndim == 2 and min(image.shape) >= 3
            and image.dtype.kind in 'iu' and image.dtype.isnative
            and image.dtype.itemsize in (1, 2, 4, 8) and image.dtype.metadata is None
            and type(indices) is bool and type(allow_borders) is bool
            and _has_numpy_span(image)):
        neighbors = _connectivity(footprint, connectivity)
        if neighbors is not None:
            entry = getattr(_core, 'local_extrema_integer_mask', None)
            if entry is not None:
                image = np.require(image, requirements=('C', 'A'))
                flags = entry(image, name == 'local_minima', neighbors, allow_borders)
                if allow_borders:
                    flags = flags[1:-1, 1:-1]
                return np.nonzero(flags) if indices else flags.view(np.bool_)
    return fallback('morphology', name, image, footprint=footprint,
                    connectivity=connectivity, indices=indices, allow_borders=allow_borders)


def local_maxima(image, footprint=None, connectivity=None, indices=False, allow_borders=True):
    """Return the mask or indices of strictly maximal connected plateaus."""
    return _local_extrema('local_maxima', image, footprint, connectivity, indices, allow_borders)


def local_minima(image, footprint=None, connectivity=None, indices=False, allow_borders=True):
    """Return the mask or indices of strictly minimal connected plateaus."""
    return _local_extrema('local_minima', image, footprint, connectivity, indices, allow_borders)
