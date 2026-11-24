"""Progressive line search with NumPy-owned theta and random state."""

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


_MASK_DTYPES = (np.dtype(bool), np.dtype(np.uint8))


def probabilistic_hough_line(image, threshold=10, line_length=50, line_gap=10, theta=None, rng=None):
    """Return detected pairs of (x, y) endpoints in detection order."""
    if isinstance(theta, np.ndarray) and theta.ndim == 1 and theta.dtype.kind == 'f' and theta.dtype.itemsize == 8:
        # Stock voting assumes every projected bin is representable.
        if type(theta) is not np.ndarray:
            raise ValueError("probabilistic_hough_line requires ordinary theta storage")
        _core.probabilistic_hough_check_theta(theta)
        if not np.isfinite(theta).all():
            raise ValueError("probabilistic_hough_line requires finite theta")

    if (type(image) is np.ndarray and image.ndim == 2 and 0 < min(image.shape) <= max(image.shape) <= 256
            and image.dtype in _MASK_DTYPES and image.dtype.metadata is None
            and image.flags.c_contiguous and image.flags.aligned and _ordinary_storage(image)
            and type(threshold) is int and 1 <= threshold <= 4096
            and type(line_length) is int and 0 <= line_length <= 256
            and type(line_gap) is int and 0 <= line_gap <= 256
            and (theta is None or type(theta) is np.ndarray and theta.ndim == 1 and theta.dtype == np.float64
                 and theta.dtype.metadata is None and theta.flags.c_contiguous and theta.flags.aligned
                 and theta.size <= 256)
            and (type(rng) is int and 0 <= rng < 2**64
                 or type(rng) is np.random.Generator and type(rng.bit_generator) is np.random.PCG64)):
        count = _core.probabilistic_hough_foreground(image)
        if count is not None:
            angles = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False) if theta is None else theta
            cosines, sines = np.cos(angles), np.sin(angles)
            generator = np.random.default_rng(rng)
            permutation = np.arange(count, dtype=np.intp)
            generator.shuffle(permutation)
            return _core.probabilistic_hough_line_ordered(
                image, cosines, sines, permutation, threshold, line_length, line_gap)

    return fallback("transform", "probabilistic_hough_line", image, threshold=threshold, line_length=line_length,
                    line_gap=line_gap, theta=theta, rng=rng)
