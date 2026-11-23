"""Bounded circle voting with stock Bresenham geometry and vote order."""

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


_IMAGE_TYPES = (np.dtype(bool), np.dtype(np.uint8))


def _radii(radius):
    if type(radius) is int:
        return (radius,) if 0 <= radius <= 64 else None
    if (type(radius) is np.ndarray and radius.ndim == 1 and 0 < radius.size <= 16
            and radius.dtype.kind in 'iu' and radius.dtype.isnative and radius.dtype.metadata is None
            and radius.flags.aligned and radius.flags.c_contiguous and _ordinary_storage(radius)):
        radius = radius.tolist()
    if type(radius) in (tuple, list) and 0 < len(radius) <= 16:
        prepared = tuple(radius)
        if all(type(value) is int and 0 <= value <= 64 for value in prepared):
            return prepared
    return None


def hough_circle(image, radius, normalize=True, full_output=False):
    """Return circular Hough accumulators in the supplied radius order."""
    if (type(image) is np.ndarray and image.ndim == 2 and image.dtype in _IMAGE_TYPES
            and image.dtype.metadata is None and image.flags.c_contiguous and _ordinary_storage(image)
            and image.size and max(image.shape) <= 512
            and type(normalize) is bool and type(full_output) is bool):
        radii = _radii(radius)
        if radii is not None:
            result = _core.hough_circle_accumulate(image, radii, normalize, full_output)
            if result is not None:
                return result
    return fallback('transform', 'hough_circle', image, radius, normalize=normalize, full_output=full_output)
