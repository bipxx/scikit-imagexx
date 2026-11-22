"""Bounded serial Hough ellipse search."""

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


_MASK_DTYPES = (np.dtype(bool), np.dtype(np.uint8))


def hough_ellipse(image, threshold=4, accuracy=1, min_size=4, max_size=None):
    """Return detected ellipses in the original endpoint-pair order."""
    if (type(image) is np.ndarray and image.ndim == 2 and max(image.shape) <= 64
            and image.dtype in _MASK_DTYPES and image.dtype.metadata is None
            and image.flags.c_contiguous and _ordinary_storage(image)
            and type(threshold) is int and 0 <= threshold <= 128
            and type(accuracy) in (float, int) and accuracy in (0.75, 1, 1.5, 2)
            and type(min_size) is int and 1 <= min_size <= 64
            and (max_size is None or type(max_size) is int and 1 <= max_size <= 32)):
        result = _core.hough_ellipse_bounded(image, threshold, accuracy, min_size, max_size)
        if result is not None:
            return result
    return fallback('transform', 'hough_ellipse', image, threshold=threshold, accuracy=accuracy,
                    min_size=min_size, max_size=max_size)
