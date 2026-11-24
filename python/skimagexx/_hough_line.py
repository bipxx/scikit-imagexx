"""Bounded Hough line voting with the original NumPy angle provider."""

import numpy as np

from . import _core
from ._compat import fallback

_MASK_DTYPES = (np.dtype(bool), np.dtype(np.uint8))


def hough_line(image, theta=None):
    """Return the Hough accumulator, tested angles and signed distances."""
    image_is_native = (type(image) is np.ndarray and image.ndim == 2
                       and image.dtype in _MASK_DTYPES
                       and image.dtype.metadata is None and image.flags.c_contiguous
                       and 0 < image.size <= 1048576 and max(image.shape) <= 4096)
    valid_theta = isinstance(theta, np.ndarray) and theta.ndim == 1 and theta.dtype == np.float64
    if image_is_native:
        angles = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False) if theta is None else theta
        if (type(angles) is np.ndarray and (theta is None or valid_theta)
                and angles.dtype.metadata is None and angles.flags.c_contiguous and angles.flags.aligned
                and 0 < angles.size <= 1024 and image.size * angles.size <= 134217728):
            accumulator = _core.hough_line_accumulate(image, angles)
            rows = accumulator.shape[0]
            offset = rows // 2
            return accumulator, angles, np.linspace(-offset, offset, rows)

    # The stock compiled loop makes unchecked writes for nonfinite projections.
    if isinstance(theta, np.ndarray) and getattr(image, "ndim", None) == 2:
        if type(theta) is not np.ndarray:
            raise ValueError("hough_line requires ordinary theta storage")
        if theta.ndim == 1 and theta.dtype.kind == 'f' and theta.dtype.itemsize == 8:
            _core.hough_line_check_theta(theta)
    return fallback("transform", "hough_line", image, theta=theta)
